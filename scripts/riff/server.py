"""Riff explainer: record yourself playing and talking in the browser, then watch the video with its tab.

The page records the camera and microphone (MediaRecorder) and, in parallel, the microphone's raw samples (an
AudioWorklet) as a lossless WAV, since the string classifiers never saw Opus-coded audio. The server lines the WAV up
with the video's own audio track, runs gtp.pipeline on it and keeps the result as a take folder (gtp.takes) plus the
video, so the tab editor can open riffs too:

  data/riffs/<take_id>/
    audio.wav, prediction.json, meta.json   as in gtp.takes; audio.wav starts at the video's time 0
    video.webm (or .mp4)                    the recording as the browser made it, remuxed so it can be seeked
    riff.json                               video file name, audio offset, processing time
  data/riffs/.incoming/<take_id>/           uploads waiting for (or failed in) processing; retried at start

Usage:
  uv run scripts/riff/server.py              # then open http://127.0.0.1:8766 in Chrome
  uv run scripts/riff/server.py port=8800 riffs_dir=/path/to/riffs device=cpu
  uv run scripts/takes/editor.py takes_dir=data/riffs   # correct a riff's tab in the editor

API (JSON):
  GET  /api/riffs              riff summaries, newest first, with uploads still processing
  GET  /api/riffs/<id>         {status: done, prediction, meta, riff} or {status: queued|processing|failed, ...}
  GET  /api/riffs/<id>/video   the video (byte ranges for seeking)
  POST /api/riffs              multipart form: video, audio (WAV), mic (label) -> {id}
"""

import email.parser
import email.policy
import io
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import hydra
import numpy as np
import scipy.signal
import soundfile as sf

from gtp import REPO_ROOT, takes
from gtp.pipeline import STD_TUNING, Pipeline

STATIC = Path(__file__).parent / 'static'
CONTENT_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.webm': 'video/webm',
    '.mp4': 'video/mp4',
}
VIDEO_TYPES = {'video/webm': '.webm', 'video/mp4': '.mp4'}
RIFF_ROUTE = re.compile(r'^/api/riffs/(?P<id>[^/]+)(?P<rest>/[a-z]+)?$')
SYNC_SR = 16000
MAX_SYNC_LAG = 2.0


def video_file(d):
    return next((p for p in sorted(d.glob('video.*')) if p.suffix in CONTENT_TYPES), None)


def summary(root, take_id):
    d = root / take_id
    prediction = takes.read_json(d / 'prediction.json')
    riff = takes.read_json(d / 'riff.json') if (d / 'riff.json').exists() else {}
    return {
        'id': take_id,
        'status': 'done',
        'duration': riff.get('duration'),
        'notes': len(prediction['notes']),
    }


class Riffs:
    """Riff folders under `root` and the one worker that turns uploads into them, a recording at a time."""

    def __init__(self, root, pipeline, cfg):
        self.root, self.pipeline, self.cfg = Path(root), pipeline, cfg
        self.incoming = self.root / '.incoming'
        self.jobs = {}
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        for d in sorted(self.incoming.iterdir()) if self.incoming.exists() else []:
            if (d / 'job.json').exists():
                self.enqueue(d.name, takes.read_json(d / 'job.json'))
        threading.Thread(target=self.work, daemon=True).start()

    def enqueue(self, take_id, job):
        self.jobs[take_id] = {'id': take_id, 'status': 'queued', 'duration': job.get('duration'), 'error': None}
        self.queue.put(take_id)

    def add(self, video, video_type, wav, mic):
        """Store an upload and queue it; returns the new riff's id."""
        if video_type not in VIDEO_TYPES:
            raise ValueError(f'unsupported video type {video_type!r}')
        info = sf.info(io.BytesIO(wav))
        with self.lock:
            take_id = takes.new_take_id('riff')
            while (self.root / take_id).exists() or (self.incoming / take_id).exists():
                time.sleep(1)
                take_id = takes.new_take_id('riff')
            d = self.incoming / take_id
            d.mkdir(parents=True)
        (d / f'upload{VIDEO_TYPES[video_type]}').write_bytes(video)
        (d / 'audio.wav').write_bytes(wav)
        job = {'mic': mic, 'duration': round(info.duration, 2), 'received': takes.now()}
        takes.write_json(d / 'job.json', job)
        self.enqueue(take_id, job)
        return take_id

    def work(self):
        while True:
            take_id = self.queue.get()
            job = self.jobs[take_id]
            job |= {'status': 'processing', 'started': time.time()}
            try:
                self.process(take_id)
                del self.jobs[take_id]
            except Exception as e:
                traceback.print_exc()
                job |= {'status': 'failed', 'error': f'{type(e).__name__}: {e}'}

    def ffmpeg(self, *args):
        """ffmpeg's stdout; raises with its error message."""
        out = subprocess.run(
            [self.cfg.ffmpeg, '-hide_banner', '-v', 'error', '-y', *map(str, args)], capture_output=True
        )
        if out.returncode:
            raise RuntimeError(f'ffmpeg failed: {out.stderr.decode(errors="replace").strip()}')
        return out.stdout

    def audio_start(self, video):
        """Start time of the video's audio stream on the video's timeline."""
        ffprobe = Path(self.cfg.ffmpeg).with_name('ffprobe')
        query = ['-select_streams', 'a:0', '-show_entries', 'stream=start_time', '-of', 'json']
        out = subprocess.run([ffprobe, '-v', 'error', *query, video], check=True, capture_output=True).stdout
        streams = json.loads(out).get('streams') or [{}]
        return float(streams[0].get('start_time', 0) or 0)

    def offset(self, video, pcm, sr):
        """(seconds, confidence): where the WAV's first sample falls on the video's timeline, from the
        cross-correlation of the WAV with the video's own (lossy) audio track."""
        raw = self.ffmpeg('-i', video, '-vn', '-ac', '1', '-ar', SYNC_SR, '-f', 'f32le', '-')
        coded = np.frombuffer(raw, dtype=np.float32)
        lossless = scipy.signal.resample_poly(pcm, SYNC_SR, sr).astype(np.float32)
        if not len(coded) or not np.any(coded) or not np.any(lossless):
            return self.audio_start(video), 0.0
        corr = scipy.signal.correlate(coded, lossless, mode='full', method='fft')
        lags = scipy.signal.correlation_lags(len(coded), len(lossless))
        near = np.abs(lags) <= MAX_SYNC_LAG * SYNC_SR
        best = np.argmax(np.where(near, corr, -np.inf))
        confidence = corr[best] / (np.linalg.norm(coded) * np.linalg.norm(lossless))
        return self.audio_start(video) + lags[best] / SYNC_SR, float(confidence)

    def process(self, take_id):
        src = self.incoming / take_id
        job = takes.read_json(src / 'job.json')
        upload = next(src.glob('upload.*'))
        # MediaRecorder files carry no duration or cues, so the browser could not seek in them; a remux adds both.
        video = src / f'video{upload.suffix}'
        self.ffmpeg(
            '-i', upload, '-c', 'copy', *(['-movflags', '+faststart'] if upload.suffix == '.mp4' else []), video
        )
        pcm, sr = sf.read(str(src / 'audio.wav'), dtype='float32', always_2d=True)
        pcm = pcm.mean(axis=1)
        t0 = time.time()
        shift, confidence = self.offset(video, pcm, sr)
        n = round(shift * sr)
        pcm = np.concatenate([np.zeros(n, dtype=np.float32), pcm]) if n > 0 else pcm[-n:]
        print(f'{take_id}: {len(pcm) / sr:.1f}s at {sr} Hz, audio offset {shift:+.3f}s (match {confidence:.2f})')
        result = self.pipeline.run(pcm, sr, lam=self.cfg.lam, beams=self.cfg.beams)
        prediction = takes.prediction_doc(
            take_id, result['placed'], result['tuning'], 0, None, result['pipeline'], result['unplaced']
        )
        meta = {'input_device': job.get('mic', ''), 'capo': '0', 'tuning_name': 'standard', 'source': 'riff page'}
        tmp = self.incoming / f'{take_id}.done'
        shutil.rmtree(tmp, ignore_errors=True)
        d = takes.create(tmp, take_id, pcm, sr, prediction, meta)
        shutil.move(video, d / video.name)
        riff = {
            'video': video.name,
            'audio_offset': round(shift, 4),
            'sync_confidence': round(confidence, 3),
            'sample_rate': sr,
            'duration': round(len(pcm) / sr, 2),
            'mic': job.get('mic', ''),
            'received': job['received'],
            'processing_s': round(time.time() - t0, 1),
        }
        takes.write_json(d / 'riff.json', riff)
        d.rename(self.root / take_id)
        tmp.rmdir()
        shutil.rmtree(src)
        print(f'{take_id}: {len(result["placed"])} notes in {riff["processing_s"]}s', flush=True)

    def list(self):
        done = [summary(self.root, t) for t in takes.take_ids(self.root)]
        pending = [
            {k: j[k] for k in ('id', 'status', 'duration', 'error')} | {'notes': None} for j in self.jobs.values()
        ]
        return sorted(pending, key=lambda j: j['id'], reverse=True) + done

    def get(self, take_id):
        if take_id in self.jobs:
            job = dict(self.jobs[take_id])
            if 'started' in job:
                job['elapsed'] = round(time.time() - job.pop('started'), 1)
            waiting = list(self.queue.queue)
            busy = sum(j['status'] == 'processing' for j in self.jobs.values())
            job['ahead'] = busy + waiting.index(take_id) if take_id in waiting else 0
            return job
        d = takes.take_dir(self.root, take_id)
        if not (d / 'prediction.json').exists():
            raise FileNotFoundError(take_id)
        take = takes.load(d)
        riff = takes.read_json(d / 'riff.json') if (d / 'riff.json').exists() else {}
        return {'id': take_id, 'status': 'done', 'prediction': take['prediction'], 'meta': take['meta'], 'riff': riff}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, riffs, **kwargs):
        self.riffs = riffs
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        if not self.path.startswith('/api/riffs/') or self.command != 'GET':
            super().log_message(fmt, *args)

    def send(self, body, content_type, status=HTTPStatus.OK, headers=()):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=HTTPStatus.OK):
        self.send(json.dumps(obj).encode(), 'application/json', status)

    def error(self, status, message):
        self.send_json({'error': message}, status)

    def handle_errors(self, route):
        try:
            route()
        except FileNotFoundError as e:
            self.error(HTTPStatus.NOT_FOUND, f'not found: {e}')
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            self.error(HTTPStatus.BAD_REQUEST, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self.handle_errors(self.get)

    def do_POST(self):
        self.handle_errors(self.post)

    def send_video(self, path):
        """The file, or the byte range the browser asks for: a video element seeks with range requests."""
        size = path.stat().st_size
        content_type = CONTENT_TYPES[path.suffix]
        m = re.match(r'bytes=(\d*)-(\d*)$', self.headers.get('Range', ''))
        if not m or (not m[1] and not m[2]):
            return self.send(path.read_bytes(), content_type, headers=[('Accept-Ranges', 'bytes')])
        start, end = (int(m[1]), int(m[2] or size - 1)) if m[1] else (size - int(m[2]), size - 1)
        end = min(end, size - 1)
        if start > end:
            headers = [('Content-Range', f'bytes */{size}')]
            return self.send(b'', content_type, HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, headers)
        with path.open('rb') as f:
            f.seek(start)
            body = f.read(end - start + 1)
        headers = [('Accept-Ranges', 'bytes'), ('Content-Range', f'bytes {start}-{end}/{size}')]
        self.send(body, content_type, HTTPStatus.PARTIAL_CONTENT, headers)

    def get(self):
        path = urlparse(self.path).path
        if path == '/api/riffs':
            return self.send_json(self.riffs.list())
        if m := RIFF_ROUTE.match(path):
            if m['rest'] is None:
                return self.send_json(self.riffs.get(m['id']))
            video = video_file(takes.take_dir(self.riffs.root, m['id']))
            if m['rest'] == '/video' and video:
                return self.send_video(video)
            raise FileNotFoundError(path)
        name = 'index.html' if path == '/' else path.lstrip('/')
        static = STATIC / name
        if '/' in name or not static.is_file():
            raise FileNotFoundError(path)
        self.send(static.read_bytes(), CONTENT_TYPES.get(static.suffix, 'application/octet-stream'))

    def post(self):
        if urlparse(self.path).path != '/api/riffs':
            raise FileNotFoundError(self.path)
        body = self.rfile.read(int(self.headers['Content-Length']))
        head = f'Content-Type: {self.headers["Content-Type"]}\r\n\r\n'.encode()
        form = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(head + body)
        if not form.is_multipart():
            raise ValueError('expected multipart/form-data')
        parts = {p.get_param('name', header='content-disposition'): p for p in form.iter_parts()}
        video, wav = parts['video'], parts['audio']
        mic = parts['mic'].get_content().strip() if 'mic' in parts else ''
        video_type = video.get_content_type()
        take_id = self.riffs.add(video.get_payload(decode=True), video_type, wav.get_payload(decode=True), mic)
        self.send_json({'id': take_id}, HTTPStatus.ACCEPTED)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'takes'), config_name='riff')
def main(cfg):
    root = REPO_ROOT / cfg.riffs_dir
    print('Loading the models...', flush=True)
    pipeline = Pipeline(device=cfg.device)
    print(f'{len(pipeline.classifiers)} string classifiers, device {pipeline.device}; tuning {STD_TUNING}, no capo')
    riffs = Riffs(root, pipeline, cfg)
    server = ThreadingHTTPServer((cfg.host, cfg.port), partial(Handler, riffs=riffs))
    print(f'{len(takes.take_ids(root))} riffs in {root}')
    print(f'Riffs: http://{cfg.host}:{cfg.port}/  (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
