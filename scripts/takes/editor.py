"""Offline tab editor for home-recording takes (see gtp.takes), served on localhost.

Opens the take list; each take shows the predicted tab over its waveform for correction, and
saves the corrected tab to the take's edited.json. No network access beyond localhost.

Usage:
  uv run scripts/takes/editor.py             # then open http://127.0.0.1:8765
  uv run scripts/takes/editor.py port=8800 takes_dir=/path/to/takes

API (JSON):
  GET  /api/takes                  take summaries
  GET  /api/takes/<id>             {prediction, edited (null until saved), meta, meta_fields, max_fret}
  GET  /api/takes/<id>/audio.wav
  PUT  /api/takes/<id>/edited      {notes, status, base_updated} -> saved edited.json
  PUT  /api/takes/<id>/meta        {field: value} -> saved meta.json
"""

import json
import re
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import hydra

from gtp import REPO_ROOT, takes
from gtp.stage2.tokenizer import MAX_FRET

STATIC = Path(__file__).parent / 'static'
CONTENT_TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css'}
TAKE_ROUTE = re.compile(r'^/api/takes/(?P<id>[^/]+)(?P<rest>/[a-z.]+)?$')


def summary(root, take_id):
    take = takes.load(root / take_id)
    prediction, edited, meta = take['prediction'], take['edited'], take['meta']
    counts = takes.counts(takes.compare(prediction, edited)) if edited else None
    return {
        'id': take_id,
        'created': prediction['created'],
        'guitar': meta.get('guitar', ''),
        'input': meta.get('mic') or meta.get('input_device', ''),
        'status': edited['status'] if edited else None,
        'updated': edited['updated'] if edited else None,
        'pred_notes': len(prediction['notes']),
        'edited_notes': len(edited['notes']) if edited else None,
        'changed': counts['changed'] if counts else 0,
    }


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def send(self, body, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=HTTPStatus.OK):
        self.send(json.dumps(obj).encode(), 'application/json', status)

    def error(self, status, message):
        self.send_json({'error': message}, status)

    def take(self, take_id):
        d = takes.take_dir(self.root, take_id)
        if not (d / 'prediction.json').exists():
            raise FileNotFoundError(take_id)
        return d

    def handle_errors(self, route):
        try:
            route()
        except FileNotFoundError as e:
            self.error(HTTPStatus.NOT_FOUND, f'not found: {e}')
        except takes.Conflict as e:
            self.error(HTTPStatus.CONFLICT, str(e))
        except (ValueError, KeyError, TypeError) as e:
            self.error(HTTPStatus.BAD_REQUEST, str(e))

    def do_GET(self):
        self.handle_errors(self.get)

    def do_PUT(self):
        self.handle_errors(self.put)

    def get(self):
        path = urlparse(self.path).path
        if path == '/api/takes':
            return self.send_json([summary(self.root, t) for t in takes.take_ids(self.root)])
        if m := TAKE_ROUTE.match(path):
            d = self.take(m['id'])
            if m['rest'] is None:
                return self.send_json({**takes.load(d), 'meta_fields': takes.META_FIELDS, 'max_fret': MAX_FRET})
            if m['rest'] == '/audio.wav':
                return self.send((d / 'audio.wav').read_bytes(), 'audio/wav')
            raise FileNotFoundError(path)
        name = 'index.html' if path == '/' else path.lstrip('/')
        static = STATIC / name
        if '/' in name or not static.is_file():
            raise FileNotFoundError(path)
        self.send(static.read_bytes(), CONTENT_TYPES.get(static.suffix, 'application/octet-stream'))

    def put(self):
        m = TAKE_ROUTE.match(urlparse(self.path).path)
        if not m:
            raise FileNotFoundError(self.path)
        d = self.take(m['id'])
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if m['rest'] == '/edited':
            return self.send_json(takes.save_edited(d, body['notes'], body['status'], body.get('base_updated')))
        if m['rest'] == '/meta':
            return self.send_json(takes.save_meta(d, body))
        raise FileNotFoundError(self.path)


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'takes'), config_name='editor')
def main(cfg):
    root = REPO_ROOT / cfg.takes_dir
    server = ThreadingHTTPServer((cfg.host, cfg.port), partial(Handler, root=root))
    print(f'{len(takes.take_ids(root))} takes in {root}')
    print(f'Editor: http://{cfg.host}:{cfg.port}/  (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
