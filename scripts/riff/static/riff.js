'use strict';

const STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E'];
const LAYOUT = { left: 30, ruler: 22, tabTop: 48, gap: 26 };
// Container/codec choices in order of preference; the audio track only serves playback and lining up the WAV.
const MIME_TYPES = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4;codecs=avc1,opus', 'video/mp4'];
// A sounding note stays highlighted at least this long, so short notes still flash.
const MIN_LIT = 0.15;
const POLL_MS = 1000;

const $ = (sel) => document.querySelector(sel);
const canvas = $('#canvas');
const ctx2d = canvas.getContext('2d');
const hscroll = $('#hscroll');
const video = $('#video');

function message(text) {
  const el = $('#message');
  el.textContent = text;
  clearTimeout(message.timer);
  if (text) message.timer = setTimeout(() => (el.textContent = ''), 8000);
}

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

function clock(seconds) {
  const m = Math.floor(seconds / 60);
  return `${m}:${(seconds - 60 * m).toFixed(1).padStart(4, '0')}`;
}

function recordedAt(id) {
  const m = id.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : '';
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

let routeToken = 0;

async function route() {
  const token = ++routeToken;
  const m = location.hash.match(/^#\/riff\/(.+)$/);
  try {
    if (m) {
      closeDevices();
      $('#record-view').hidden = true;
      $('#riff-view').hidden = false;
      await openRiff(decodeURIComponent(m[1]), token);
    } else {
      closeRiff();
      $('#riff-view').hidden = true;
      $('#record-view').hidden = false;
      document.title = 'GTP riffs';
      await Promise.all([showList(), openDevices()]);
    }
  } catch (err) {
    message(err.message);
    if (m) showProcessing(`Cannot open this riff: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Recording: MediaRecorder for the video, an AudioWorklet for the lossless samples
// ---------------------------------------------------------------------------

const rec = {
  stream: null,
  ctx: null,
  recorder: null,
  mime: '',
  chunks: [],
  pcm: [],
  recording: false,
  uploading: false,
  t0: 0,
  level: 0,
  peak: 0,
};

async function listDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  for (const [kind, sel] of [['audioinput', '#mic'], ['videoinput', '#cam']]) {
    const select = $(sel);
    const current = rec.stream && rec.stream.getTracks().find((t) => `${t.kind}input` === kind);
    const id = current ? current.getSettings().deviceId : select.value;
    select.innerHTML = '';
    for (const d of devices.filter((x) => x.kind === kind)) {
      select.add(new Option(d.label || `${kind} ${select.length + 1}`, d.deviceId, false, d.deviceId === id));
    }
  }
}

function closeDevices() {
  if (rec.recording) stopRecording();
  if (rec.stream) rec.stream.getTracks().forEach((t) => t.stop());
  if (rec.ctx) rec.ctx.close();
  rec.stream = null;
  rec.ctx = null;
  $('#preview').srcObject = null;
}

async function openDevices() {
  if (rec.recording || rec.uploading) return;
  closeDevices();
  // Echo cancellation, noise suppression and automatic gain are made for voice calls and damage guitar audio.
  const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 };
  const videoWanted = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if ($('#mic').value) audio.deviceId = { exact: $('#mic').value };
  if ($('#cam').value) videoWanted.deviceId = { exact: $('#cam').value };
  const stream = await navigator.mediaDevices.getUserMedia({ audio, video: videoWanted });
  if (location.hash.startsWith('#/riff/')) {
    stream.getTracks().forEach((t) => t.stop());
    return;
  }
  rec.stream = stream;
  $('#preview').srcObject = stream;
  await listDevices();

  const settings = stream.getAudioTracks()[0].getSettings();
  // Capturing at the track's own rate keeps the browser from resampling the samples.
  rec.ctx = new AudioContext(settings.sampleRate ? { sampleRate: settings.sampleRate } : {});
  await rec.ctx.audioWorklet.addModule('/pcm-worklet.js');
  const tap = new AudioWorkletNode(rec.ctx, 'pcm-tap', { numberOfOutputs: 0 });
  tap.port.onmessage = (e) => onSamples(e.data);
  rec.ctx.createMediaStreamSource(stream).connect(tap);
  showAudioSettings(settings);
}

function showAudioSettings(s) {
  const flag = (name, on) => `${name} ${on ? 'ON' : 'off'}`;
  const processing = [
    flag('echo cancellation', s.echoCancellation),
    flag('noise suppression', s.noiseSuppression),
    flag('auto gain', s.autoGainControl),
  ];
  const rate = rec.ctx ? rec.ctx.sampleRate : s.sampleRate;
  $('#audio-settings').textContent = `${rate} Hz, ${s.channelCount || '?'} channel(s); ${processing.join(', ')}`;
  if (s.echoCancellation || s.noiseSuppression || s.autoGainControl) {
    message('The browser kept some voice processing on: the guitar audio may suffer.');
  }
}

function onSamples(block) {
  let peak = 0;
  for (let i = 0; i < block.length; i++) peak = Math.max(peak, Math.abs(block[i]));
  rec.level = Math.max(peak, rec.level);
  if (rec.recording) {
    rec.pcm.push(block);
    rec.peak = Math.max(rec.peak, peak);
  }
}

async function startRecording() {
  if (rec.recording || rec.uploading) return;
  if (!rec.stream) await openDevices();
  await rec.ctx.resume();
  rec.mime = MIME_TYPES.find((t) => MediaRecorder.isTypeSupported(t)) || '';
  rec.recorder = new MediaRecorder(rec.stream, {
    mimeType: rec.mime,
    audioBitsPerSecond: 256000,
    videoBitsPerSecond: 5000000,
  });
  rec.chunks = [];
  rec.pcm = [];
  rec.peak = 0;
  rec.recorder.ondataavailable = (e) => e.data.size && rec.chunks.push(e.data);
  rec.recorder.onstop = upload;
  rec.recorder.start(1000);
  rec.recording = true;
  rec.t0 = performance.now();
  $('#record').classList.add('on');
  $('#record').innerHTML = '&#9632; Stop';
  $('#rec-badge').hidden = false;
  $('#rec-status').textContent = 'Recording: play and talk; Space stops.';
  ['#mic', '#cam'].forEach((s) => ($(s).disabled = true));
}

function stopRecording() {
  if (!rec.recording) return;
  rec.recording = false;
  rec.uploading = true;
  rec.recorder.stop();
  $('#record').classList.remove('on');
  $('#record').innerHTML = '&#9679; Record';
  $('#record').disabled = true;
  $('#rec-badge').hidden = true;
  $('#rec-status').textContent = 'Uploading…';
}

async function upload() {
  try {
    const seconds = rec.pcm.reduce((sum, b) => sum + b.length, 0) / rec.ctx.sampleRate;
    if (!rec.pcm.length) throw new Error('No audio samples arrived from the microphone');
    const type = (rec.mime || (rec.chunks[0] && rec.chunks[0].type) || 'video/webm').split(';')[0];
    const form = new FormData();
    form.append('video', new Blob(rec.chunks, { type }), `video.${type === 'video/mp4' ? 'mp4' : 'webm'}`);
    form.append('audio', new Blob([encodeWav(rec.pcm, rec.ctx.sampleRate)], { type: 'audio/wav' }), 'audio.wav');
    form.append('mic', $('#mic').selectedOptions[0] ? $('#mic').selectedOptions[0].text : '');
    $('#rec-status').textContent = `Uploading ${seconds.toFixed(1)} s…`;
    const { id } = await api('/api/riffs', { method: 'POST', body: form });
    rec.pcm = [];
    rec.chunks = [];
    location.hash = `#/riff/${encodeURIComponent(id)}`;
  } catch (err) {
    message(`Upload failed: ${err.message}`);
    $('#rec-status').textContent = 'Press Space or the button to start; again to stop.';
  } finally {
    rec.uploading = false;
    $('#record').disabled = false;
    ['#mic', '#cam'].forEach((s) => ($(s).disabled = false));
  }
}

function toggleRecording() {
  if (rec.recording) stopRecording();
  else startRecording().catch((err) => message(`Cannot record: ${err.message}`));
}

async function showList() {
  const rows = await api('/api/riffs');
  const tbody = $('#riff-table tbody');
  tbody.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    const cells = [r.id, recordedAt(r.id), r.duration == null ? '' : clock(r.duration), r.notes ?? '', r.status];
    cells.forEach((text, i) => {
      const td = document.createElement('td');
      td.textContent = text;
      if (i === 4) td.className = `status-${r.status}`;
      tr.appendChild(td);
    });
    tr.addEventListener('click', () => (location.hash = `#/riff/${encodeURIComponent(r.id)}`));
    tbody.appendChild(tr);
  }
  if (!rows.length) tbody.innerHTML = '<tr><td colspan="5" class="hint">No riffs yet.</td></tr>';
}

// ---------------------------------------------------------------------------
// Playback: the video with its tab underneath
// ---------------------------------------------------------------------------

let R = null;

function closeRiff() {
  video.pause();
  video.removeAttribute('src');
  video.load();
  R = null;
}

function showProcessing(html) {
  $('#processing').innerHTML = html;
  $('#processing').hidden = false;
  video.hidden = canvas.hidden = hscroll.hidden = true;
}

async function openRiff(id, token) {
  closeRiff();
  $('#riff-title').textContent = id;
  $('#riff-info').textContent = '';
  document.title = `${id} · GTP riffs`;
  let riff = await api(`/api/riffs/${encodeURIComponent(id)}`);
  while (riff.status !== 'done') {
    if (riff.status === 'failed') {
      showProcessing(`Processing failed: <span class="mono">${riff.error}</span>`);
      return;
    }
    const length = riff.duration ? ` of ${riff.duration.toFixed(1)} s` : '';
    const state =
      riff.status === 'queued'
        ? `Waiting for ${riff.ahead} earlier recording(s) to finish`
        : `Transcribing the recording${length}: ${(riff.elapsed || 0).toFixed(0)} s so far`;
    const expect = riff.duration ? ` It takes about ${Math.round(riff.duration * 1.8 + 5)} s.` : '';
    showProcessing(`<span class="spinner"></span>${state}.${expect}<br><span class="hint">This page updates by itself.</span>`);
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    if (token !== routeToken) return;
    riff = await api(`/api/riffs/${encodeURIComponent(id)}`);
  }
  if (token !== routeToken) return;
  const notes = riff.prediction.notes;
  R = {
    id,
    notes,
    duration: Math.max(riff.riff.duration || 0, ...notes.map((n) => n.end)),
    pps: 120,
    needsDraw: true,
  };
  $('#processing').hidden = true;
  video.hidden = canvas.hidden = hscroll.hidden = false;
  video.src = `/api/riffs/${encodeURIComponent(id)}/video`;
  const info = [`${notes.length} notes`];
  if (riff.prediction.unplaced.length) info.push(`${riff.prediction.unplaced.length} without a tab`);
  if (riff.riff.processing_s != null) info.push(`transcribed in ${riff.riff.processing_s} s`);
  $('#riff-info').textContent = info.join(' · ');
  resizeCanvas();
}

function togglePlay() {
  if (!R) return;
  if (video.paused) video.play();
  else video.pause();
}

function seek(t) {
  if (!R) return;
  video.currentTime = Math.max(0, Math.min(R.duration, t));
  R.needsDraw = true;
}

function stringY(s) {
  return LAYOUT.tabTop + (s - 1) * LAYOUT.gap;
}

function tx(t) {
  return LAYOUT.left + t * R.pps - hscroll.scrollLeft;
}

function tAt(x) {
  return (x - LAYOUT.left + hscroll.scrollLeft) / R.pps;
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (R) {
    $('#hspacer').style.width = `${LAYOUT.left + R.duration * R.pps + 60}px`;
    R.needsDraw = true;
  }
}

function zoom(factor, anchorX = canvas.clientWidth / 2) {
  const t = tAt(anchorX);
  R.pps = Math.max(20, Math.min(1500, R.pps * factor));
  resizeCanvas();
  hscroll.scrollLeft = LAYOUT.left + t * R.pps - anchorX;
}

function rulerStep() {
  return [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60].find((s) => s * R.pps >= 60) || 120;
}

function draw() {
  const c = ctx2d;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  c.fillStyle = '#ffffff';
  c.fillRect(0, 0, w, h);
  const t0 = tAt(LAYOUT.left);
  const t1 = tAt(w);
  const now = video.currentTime || 0;

  const step = rulerStep();
  c.font = '11px ui-monospace, Menlo, monospace';
  c.textBaseline = 'middle';
  c.textAlign = 'left';
  for (let t = Math.max(0, Math.floor(t0 / step) * step); t <= t1; t += step) {
    const x = Math.round(tx(t)) + 0.5;
    c.strokeStyle = '#eeeef2';
    c.beginPath();
    c.moveTo(x, LAYOUT.ruler);
    c.lineTo(x, h);
    c.stroke();
    c.fillStyle = '#6b6b70';
    c.fillText(step < 1 ? t.toFixed(2) : clock(t), x + 3, LAYOUT.ruler / 2);
  }
  c.strokeStyle = '#d9d9de';
  c.beginPath();
  c.moveTo(0, LAYOUT.ruler + 0.5);
  c.lineTo(w, LAYOUT.ruler + 0.5);
  c.stroke();

  c.strokeStyle = '#b9b9c0';
  for (let s = 1; s <= 6; s++) {
    const y = stringY(s) + 0.5;
    c.beginPath();
    c.moveTo(LAYOUT.left, y);
    c.lineTo(w, y);
    c.stroke();
  }

  // Notes that would hide each other on one string are laid side by side; their bars keep the true onset.
  const rightEdge = new Map();
  c.font = '13px ui-monospace, Menlo, monospace';
  c.textAlign = 'center';
  for (const n of R.notes) {
    if (n.end < t0 - 1 || n.start > t1) continue;
    const lit = n.start <= now && now < Math.max(n.end, n.start + MIN_LIT);
    const past = !lit && n.start < now;
    const y = stringY(n.string);
    c.fillStyle = lit ? 'rgba(47,111,222,0.35)' : 'rgba(80,80,90,0.15)';
    c.fillRect(tx(n.start), y - 2, Math.max(2, (n.end - n.start) * R.pps), 4);
    const label = String(n.fret);
    const bw = Math.max(17, c.measureText(label).width + 8);
    const x = Math.max(tx(n.start), (rightEdge.get(n.string) ?? -Infinity) + 2);
    rightEdge.set(n.string, x - 1 + bw);
    c.beginPath();
    c.roundRect(x - 1, y - 9, bw, 18, 3);
    c.fillStyle = lit ? '#2f6fde' : '#ffffff';
    c.fill();
    c.strokeStyle = lit ? '#2f6fde' : past ? '#a9a9b0' : '#55555a';
    c.stroke();
    c.fillStyle = lit ? '#ffffff' : past ? '#8a8a92' : '#1d1d1f';
    c.fillText(label, x - 1 + bw / 2, y + 0.5);
  }

  const px = Math.round(tx(now)) + 0.5;
  if (px >= LAYOUT.left) {
    c.strokeStyle = '#c8312b';
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(px, 0);
    c.lineTo(px, h);
    c.stroke();
    c.lineWidth = 1;
  }

  c.fillStyle = '#f4f4f6';
  c.fillRect(0, LAYOUT.ruler + 1, LAYOUT.left - 4, h - LAYOUT.ruler - 1);
  c.font = '12px ui-monospace, Menlo, monospace';
  c.fillStyle = '#55555a';
  for (let s = 1; s <= 6; s++) c.fillText(STRING_NAMES[s - 1], (LAYOUT.left - 4) / 2, stringY(s) + 0.5);
}

// ---------------------------------------------------------------------------
// Input and the frame loop
// ---------------------------------------------------------------------------

canvas.addEventListener('click', (e) => {
  if (!R) return;
  const x = e.clientX - canvas.getBoundingClientRect().left;
  seek(tAt(Math.max(LAYOUT.left, x)));
});

canvas.addEventListener(
  'wheel',
  (e) => {
    if (!R) return;
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) zoom(Math.exp(-e.deltaY * 0.01), e.clientX - canvas.getBoundingClientRect().left);
    else hscroll.scrollLeft += e.deltaX || e.deltaY;
  },
  { passive: false },
);

hscroll.addEventListener('scroll', () => R && (R.needsDraw = true));
window.addEventListener('resize', resizeCanvas);
['play', 'pause', 'seeked', 'loadedmetadata'].forEach((ev) => video.addEventListener(ev, () => R && (R.needsDraw = true)));

document.addEventListener('keydown', (e) => {
  if (e.key !== ' ' || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  e.preventDefault();
  if (!$('#record-view').hidden) toggleRecording();
  else togglePlay();
});

// Chrome starts an AudioContext suspended until the page gets a click or a key press.
['pointerdown', 'keydown'].forEach((ev) => document.addEventListener(ev, () => rec.ctx && rec.ctx.resume()));
document.addEventListener('click', (e) => {
  const button = e.target.closest('button');
  if (button) button.blur();
});
['#mic', '#cam'].forEach((sel) =>
  $(sel).addEventListener('change', (e) => {
    e.target.blur();
    openDevices().catch((err) => message(`Cannot open the device: ${err.message}`));
  }),
);
$('#record').addEventListener('click', toggleRecording);
$('#play').addEventListener('click', togglePlay);
$('#zoom-in').addEventListener('click', () => zoom(1.5));
$('#zoom-out').addEventListener('click', () => zoom(1 / 1.5));
window.addEventListener('hashchange', route);
window.addEventListener('beforeunload', (e) => {
  if (rec.recording || rec.uploading) e.preventDefault();
});

function tick() {
  if (rec.recording) $('#elapsed').textContent = clock((performance.now() - rec.t0) / 1000);
  if (!$('#record-view').hidden) {
    const level = rec.level;
    rec.level *= 0.97;
    $('#meter-bar').style.width = `${Math.min(100, Math.sqrt(level) * 100)}%`;
    $('#meter-bar').classList.toggle('hot', level >= 0.99);
    $('#peak').textContent = rec.recording ? `peak ${rec.peak.toFixed(2)}${rec.peak >= 0.99 ? ' CLIPPING' : ''}` : '';
  }
  if (R) {
    const t = video.currentTime || 0;
    if (!video.paused) {
      if ($('#follow').checked) {
        const w = canvas.clientWidth;
        const x = tx(t);
        if (x > w - 0.15 * (w - LAYOUT.left) || x < LAYOUT.left) {
          hscroll.scrollLeft = Math.max(0, t * R.pps - 0.1 * (w - LAYOUT.left));
        }
      }
      R.needsDraw = true;
    }
    if (R.needsDraw) {
      R.needsDraw = false;
      $('#time').textContent = `${clock(t)} / ${clock(R.duration)}`;
      $('#play').innerHTML = video.paused ? '&#9654;' : '&#10073;&#10073;';
      draw();
    }
  }
  requestAnimationFrame(tick);
}

route();
requestAnimationFrame(tick);
