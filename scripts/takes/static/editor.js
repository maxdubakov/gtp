'use strict';

const STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E'];
const PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
// Same thresholds as gtp.takes: smaller onset edits are float noise; closer onsets on one string are one pluck.
const ONSET_EPS = 0.001;
const SAME_PLUCK = 0.03;
const LAYOUT = { left: 30, ruler: 22, tabTop: 46, gap: 26, wavePad: 26 };

const $ = (sel) => document.querySelector(sel);
const canvas = $('#canvas');
const ctx2d = canvas.getContext('2d');
const hscroll = $('#hscroll');
const audio = new Audio();
audio.preservesPitch = true;

let S = null;

function pitchName(p) {
  return `${PITCH_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
}

function message(text, ok = false) {
  const el = $('#message');
  el.textContent = text;
  el.style.color = ok ? 'var(--added)' : 'var(--deleted)';
  clearTimeout(message.timer);
  message.timer = setTimeout(() => (el.textContent = ''), 4000);
}

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

// ---------------------------------------------------------------------------
// Routing and the take list
// ---------------------------------------------------------------------------

function recordedAt(id) {
  const m = id.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : '';
}

async function showList() {
  closeTake();
  $('#take-view').hidden = true;
  $('#list-view').hidden = false;
  document.title = 'GTP takes';
  const rows = await api('/api/takes');
  const tbody = $('#take-table tbody');
  tbody.innerHTML = '';
  for (const t of rows) {
    const tr = document.createElement('tr');
    const status = t.status || 'not started';
    const notes = t.edited_notes == null ? `${t.pred_notes}` : `${t.pred_notes} → ${t.edited_notes}`;
    const cells = [t.id, recordedAt(t.id), t.guitar, t.input, status, notes, t.status ? t.changed : ''];
    cells.forEach((text, i) => {
      const td = document.createElement('td');
      td.textContent = text;
      if (i === 4) td.className = `status-${t.status}`;
      tr.appendChild(td);
    });
    tr.addEventListener('click', () => (location.hash = `#/take/${encodeURIComponent(t.id)}`));
    tbody.appendChild(tr);
  }
  if (!rows.length) tbody.innerHTML = '<tr><td colspan="7" class="hint">No takes yet.</td></tr>';
}

async function route() {
  const m = location.hash.match(/^#\/take\/(.+)$/);
  try {
    if (m) await openTake(decodeURIComponent(m[1]));
    else await showList();
  } catch (err) {
    message(err.message);
  }
}

// ---------------------------------------------------------------------------
// Take state
// ---------------------------------------------------------------------------

function draftKey(id) {
  return `gtp-take-draft:${id}`;
}

function readDraft(id) {
  try {
    return JSON.parse(localStorage.getItem(draftKey(id)));
  } catch {
    return null;
  }
}

function writeDraft() {
  if (!S) return;
  try {
    if (isDirty()) {
      const draft = { notes: S.notes, status: S.status, base: S.baseUpdated, at: new Date().toISOString() };
      localStorage.setItem(draftKey(S.id), JSON.stringify(draft));
    } else {
      localStorage.removeItem(draftKey(S.id));
    }
  } catch {
    // Storage can be unavailable (private window); the draft is only a convenience.
  }
}

function editedFromPrediction(prediction) {
  return prediction.notes.map((n) => ({
    id: n.id,
    source_id: n.id,
    pitch: n.pitch,
    start: n.start,
    end: n.end,
    velocity: n.velocity,
    string: n.string,
    fret: n.fret,
  }));
}

function closeTake() {
  clearTimeout(changed.timer);
  writeDraft();
  audio.pause();
  synth.reset(0);
  if (audio.src) URL.revokeObjectURL(audio.src);
  audio.removeAttribute('src');
  S = null;
}

async function openTake(id) {
  closeTake();
  const take = await api(`/api/takes/${encodeURIComponent(id)}`);
  const prediction = take.prediction;
  const edited = take.edited;
  S = {
    id,
    prediction,
    predById: new Map(prediction.notes.map((n) => [n.id, n])),
    tuning: prediction.tuning,
    maxFret: take.max_fret,
    meta: take.meta,
    metaFields: take.meta_fields,
    notes: edited ? edited.notes : editedFromPrediction(prediction),
    status: edited ? edited.status : 'draft',
    baseUpdated: edited ? edited.updated : null,
    savedJson: '',
    selected: null,
    cursorString: 1,
    history: [],
    future: [],
    pps: 150,
    duration: 0,
    peaks: null,
    region: null,
    loop: false,
    hits: [],
    fretBuffer: '',
    fretTimer: null,
    drag: null,
  };
  S.savedJson = stateJson();
  const draft = readDraft(id);
  if (draft && draft.base === S.baseUpdated && JSON.stringify({ notes: draft.notes, status: draft.status }) !== S.savedJson) {
    if (confirm(`Restore the unsaved edits of this take from ${draft.at}?`)) {
      S.notes = draft.notes;
      S.status = draft.status;
    }
  }
  sortNotes();
  $('#list-view').hidden = true;
  $('#take-view').hidden = false;
  $('#take-title').textContent = id;
  document.title = `${id} · GTP takes`;
  $('#status').value = S.status;
  buildMetaForm();
  resizeCanvas();
  refresh();

  const bytes = await (await fetch(`/api/takes/${encodeURIComponent(id)}/audio.wav`)).arrayBuffer();
  if (!S || S.id !== id) return;
  audio.src = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  audio.playbackRate = Number($('#rate').value);
  audio.volume = Number($('#vol-audio').value);
  const decoded = await new OfflineAudioContext(1, 1, 44100).decodeAudioData(bytes.slice(0));
  if (!S || S.id !== id) return;
  S.duration = decoded.duration;
  S.peaks = computePeaks(decoded);
  const lastNote = Math.max(0, ...S.notes.map((n) => n.end));
  S.duration = Math.max(S.duration, lastNote);
  layoutWidth();
  canvas.focus();
  refresh();
}

function stateJson() {
  return JSON.stringify({ notes: S.notes, status: S.status });
}

function isDirty() {
  return S && stateJson() !== S.savedJson;
}

function sortNotes() {
  S.notes.sort((a, b) => a.start - b.start || a.string - b.string);
}

function noteKind(n) {
  if (n.source_id == null) return 'added';
  const p = S.predById.get(n.source_id);
  const same = p.string === n.string && p.fret === n.fret && Math.abs(p.start - n.start) <= ONSET_EPS;
  return same ? 'same' : 'changed';
}

function ghosts() {
  const bySource = new Map(S.notes.filter((n) => n.source_id != null).map((n) => [n.source_id, n]));
  return S.prediction.notes
    .map((p) => ({ p, e: bySource.get(p.id) }))
    .filter(({ e }) => !e || noteKind(e) === 'changed');
}

function conflicts() {
  const ids = new Set();
  for (let s = 1; s <= 6; s++) {
    const on = S.notes.filter((n) => n.string === s);
    for (let i = 1; i < on.length; i++) {
      if (on[i].start - on[i - 1].start < SAME_PLUCK) {
        ids.add(on[i].id);
        ids.add(on[i - 1].id);
      }
    }
  }
  return ids;
}

function nextId() {
  return Math.max(-1, ...S.prediction.notes.map((n) => n.id), ...S.notes.map((n) => n.id)) + 1;
}

function selectedNote() {
  return S && S.selected && S.selected.kind === 'note' ? S.notes.find((n) => n.id === S.selected.id) : null;
}

function selectedGhost() {
  return S && S.selected && S.selected.kind === 'ghost' ? S.predById.get(S.selected.id) : null;
}

function snapshot() {
  return { notes: JSON.stringify(S.notes), status: S.status, selected: S.selected };
}

function restore(snap) {
  S.notes = JSON.parse(snap.notes);
  S.status = snap.status;
  S.selected = snap.selected;
  $('#status').value = S.status;
}

// Every change goes through edit(): `fn` mutates S and returns false to refuse the change.
function edit(fn) {
  const before = snapshot();
  if (fn() === false) return false;
  S.history.push(before);
  if (S.history.length > 500) S.history.shift();
  S.future = [];
  sortNotes();
  changed();
  return true;
}

function undo() {
  if (!S.history.length) return;
  S.future.push(snapshot());
  restore(S.history.pop());
  changed();
}

function redo() {
  if (!S.future.length) return;
  S.history.push(snapshot());
  restore(S.future.pop());
  changed();
}

function changed() {
  clearTimeout(changed.timer);
  changed.timer = setTimeout(writeDraft, 400);
  synth.reset(audio.currentTime);
  refresh();
}

// ---------------------------------------------------------------------------
// Editing operations
// ---------------------------------------------------------------------------

function audition(n) {
  if (audio.paused) synth.audition(n.pitch);
}

function setFret(n, fret) {
  if (fret > S.maxFret) return message(`Fret ${fret} is above ${S.maxFret}`);
  edit(() => {
    n.fret = fret;
    n.pitch = S.tuning[n.string - 1] + fret;
  });
  audition(n);
}

function moveString(n, step) {
  const s = n.string + step;
  if (s < 1 || s > 6) return;
  const fret = n.pitch - S.tuning[s - 1];
  if (fret < 0 || fret > S.maxFret) {
    return message(`${pitchName(n.pitch)} cannot be played on the ${STRING_NAMES[s - 1]} string`);
  }
  edit(() => {
    n.string = s;
    n.fret = fret;
  });
  S.cursorString = s;
}

function shiftPitch(n, step) {
  const fret = n.fret + step;
  if (fret < 0 || fret > S.maxFret) return message(`Fret ${fret} is out of range on the ${STRING_NAMES[n.string - 1]} string`);
  setFret(n, fret);
}

function nudge(n, seconds) {
  const d = Math.max(-n.start, seconds);
  edit(() => {
    n.start = round4(n.start + d);
    n.end = round4(n.end + d);
  });
}

function round4(x) {
  return Math.round(x * 1e4) / 1e4;
}

function deleteNote(n) {
  const i = S.notes.indexOf(n);
  edit(() => {
    S.notes.splice(i, 1);
    const next = S.notes[Math.min(i, S.notes.length - 1)];
    S.selected = next ? { kind: 'note', id: next.id } : null;
  });
}

function addNote(start, string) {
  const n = {
    id: nextId(),
    source_id: null,
    pitch: S.tuning[string - 1],
    start: round4(Math.max(0, start)),
    end: round4(Math.max(0, start) + 0.3),
    velocity: null,
    string,
    fret: 0,
  };
  edit(() => {
    S.notes.push(n);
    S.selected = { kind: 'note', id: n.id };
  });
  S.cursorString = string;
  message('Added on fret 0: type the fret', true);
  audition(n);
}

function revert() {
  const n = selectedNote();
  const ghost = selectedGhost();
  if (n && n.source_id != null) {
    const p = S.predById.get(n.source_id);
    edit(() => Object.assign(n, { pitch: p.pitch, start: p.start, end: p.end, string: p.string, fret: p.fret }));
  } else if (ghost) {
    const e = S.notes.find((x) => x.source_id === ghost.id);
    if (e) {
      S.selected = { kind: 'note', id: e.id };
      return revert();
    }
    const id = S.notes.some((x) => x.id === ghost.id) ? nextId() : ghost.id;
    const { pitch, start, end, velocity, string, fret } = ghost;
    edit(() => {
      S.notes.push({ id, source_id: ghost.id, pitch, start, end, velocity, string, fret });
      S.selected = { kind: 'note', id };
    });
  }
}

function select(i) {
  if (!S.notes.length) return;
  const n = S.notes[Math.max(0, Math.min(S.notes.length - 1, i))];
  S.selected = { kind: 'note', id: n.id };
  S.cursorString = n.string;
  scrollTo(n.start);
  refresh();
}

function stepSelection(step) {
  const n = selectedNote();
  if (n) return select(S.notes.indexOf(n) + step);
  const t = audio.currentTime;
  const after = S.notes.findIndex((x) => x.start >= t);
  select(step > 0 ? (after < 0 ? S.notes.length - 1 : after) : (after < 0 ? S.notes.length : after) - 1);
}

function typeFret(digit) {
  const n = selectedNote();
  if (!n) return message('Select a note first');
  S.fretBuffer += digit;
  clearTimeout(S.fretTimer);
  const value = Number(S.fretBuffer);
  if (S.fretBuffer.length >= 2 || S.fretBuffer === '0' || value * 10 > S.maxFret) flushFret();
  else S.fretTimer = setTimeout(flushFret, 700);
  refresh();
}

function flushFret() {
  clearTimeout(S.fretTimer);
  if (!S.fretBuffer) return;
  const value = Number(S.fretBuffer);
  S.fretBuffer = '';
  const n = selectedNote();
  if (n) setFret(n, value);
  refresh();
}

async function save() {
  flushFret();
  try {
    const body = JSON.stringify({ notes: S.notes, status: S.status, base_updated: S.baseUpdated });
    const id = S.id;
    const saved = await api(`/api/takes/${encodeURIComponent(id)}/edited`, { method: 'PUT', body });
    if (!S || S.id !== id) return;
    S.baseUpdated = saved.updated;
    S.notes = saved.notes;
    S.savedJson = stateJson();
    writeDraft();
    message(`Saved ${saved.updated}`, true);
    refresh();
  } catch (err) {
    message(`Save failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Playback: the recording through <audio> (keeps the pitch when slowed), the tab through a synth
// ---------------------------------------------------------------------------

const synth = {
  ctx: null,
  out: null,
  buffers: new Map(),
  voices: new Set(),
  cursor: 0,

  ensure() {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.out = this.ctx.createGain();
      this.out.connect(this.ctx.destination);
      this.setVolume(Number($('#vol-tab').value));
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
  },

  setVolume(v) {
    if (this.out) this.out.gain.value = v;
  },

  // Karplus-Strong pluck, cached per pitch.
  buffer(pitch) {
    if (!this.buffers.has(pitch)) {
      const sr = this.ctx.sampleRate;
      const period = Math.max(2, Math.round(sr / (440 * 2 ** ((pitch - 69) / 12))));
      const length = Math.floor(sr * 2);
      const buf = this.ctx.createBuffer(1, length, sr);
      const data = buf.getChannelData(0);
      const ring = new Float32Array(period);
      let prev = 0;
      for (let i = 0; i < period; i++) {
        prev = 0.5 * prev + 0.5 * (Math.random() * 2 - 1);
        ring[i] = prev;
      }
      for (let i = 0, j = 0; i < length; i++) {
        const k = (j + 1) % period;
        data[i] = ring[j];
        ring[j] = 0.996 * 0.5 * (ring[j] + ring[k]);
        j = k;
      }
      this.buffers.set(pitch, buf);
    }
    return this.buffers.get(pitch);
  },

  play(pitch, when, duration, dest) {
    const src = this.ctx.createBufferSource();
    src.buffer = this.buffer(pitch);
    const env = this.ctx.createGain();
    env.gain.setValueAtTime(0.45, when);
    env.gain.setTargetAtTime(0, when + duration, 0.06);
    src.connect(env).connect(dest);
    src.start(when);
    src.stop(when + duration + 0.5);
    this.voices.add(src);
    src.onended = () => this.voices.delete(src);
  },

  audition(pitch) {
    this.ensure();
    this.play(pitch, this.ctx.currentTime, 0.7, this.ctx.destination);
  },

  reset(t) {
    for (const v of this.voices) {
      try {
        v.stop();
      } catch {
        // already stopped
      }
    }
    this.voices.clear();
    this.cursor = t;
  },

  // Schedule the tab notes that start in the next 200 ms of recording time.
  schedule(t) {
    if (!this.ctx || !this.out || this.out.gain.value === 0 || !S) {
      this.cursor = t;
      return;
    }
    const rate = audio.playbackRate;
    const horizon = t + 0.2 * rate;
    for (const n of S.notes) {
      if (n.start >= this.cursor && n.start < horizon) {
        const when = this.ctx.currentTime + Math.max(0, (n.start - t) / rate);
        this.play(n.pitch, when, Math.max(0.08, (n.end - n.start) / rate), this.out);
      }
    }
    this.cursor = Math.max(this.cursor, horizon);
  },
};

function togglePlay() {
  if (!audio.src) return;
  if (audio.paused) {
    synth.ensure();
    if (S.loop && S.region && (audio.currentTime < S.region.start || audio.currentTime >= S.region.end)) {
      audio.currentTime = S.region.start;
    }
    synth.reset(audio.currentTime);
    audio.play();
  } else {
    audio.pause();
    synth.reset(audio.currentTime);
  }
  refresh();
}

function seek(t) {
  audio.currentTime = Math.max(0, Math.min(S.duration, t));
  synth.reset(audio.currentTime);
  refresh();
}

function playFrom(n) {
  seek(n.start - 0.05);
  if (audio.paused) togglePlay();
}

const MIXES = [
  ['1', '0', 'audio'],
  ['0', '0.7', 'tab'],
  ['0.8', '0.7', 'audio + tab'],
];

function cycleMix() {
  const now = MIXES.findIndex(([a, t]) => $('#vol-audio').value === a && $('#vol-tab').value === t);
  const [a, t, name] = MIXES[(now + 1) % MIXES.length];
  $('#vol-audio').value = a;
  $('#vol-tab').value = t;
  audio.volume = Number(a);
  synth.ensure();
  synth.setVolume(Number(t));
  message(`Listening to ${name}`, true);
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function computePeaks(buffer) {
  const data = buffer.getChannelData(0);
  const block = 32;
  const n = Math.ceil(data.length / block);
  const min = new Float32Array(n);
  const max = new Float32Array(n);
  let peak = 1e-6;
  for (let b = 0; b < n; b++) {
    let lo = 0;
    let hi = 0;
    const end = Math.min(data.length, (b + 1) * block);
    for (let i = b * block; i < end; i++) {
      const v = data[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    min[b] = lo;
    max[b] = hi;
    peak = Math.max(peak, hi, -lo);
  }
  return { min, max, n, perSec: buffer.sampleRate / block, scale: 1 / peak };
}

function stringY(s) {
  return LAYOUT.tabTop + (s - 1) * LAYOUT.gap;
}

function waveTop() {
  return stringY(6) + LAYOUT.wavePad;
}

function tx(t) {
  return LAYOUT.left + t * S.pps - hscroll.scrollLeft;
}

function tAt(x) {
  return (x - LAYOUT.left + hscroll.scrollLeft) / S.pps;
}

function viewWidth() {
  return canvas.clientWidth;
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (S) layoutWidth();
}

function layoutWidth() {
  $('#hspacer').style.width = `${LAYOUT.left + S.duration * S.pps + 60}px`;
}

function scrollTo(t) {
  const x = tx(t);
  const w = viewWidth();
  if (x < LAYOUT.left + 20 || x > w - 40) hscroll.scrollLeft = Math.max(0, t * S.pps - 0.3 * (w - LAYOUT.left));
}

function zoom(factor, anchorX = viewWidth() / 2) {
  const t = tAt(anchorX);
  S.pps = Math.max(10, Math.min(3000, S.pps * factor));
  layoutWidth();
  hscroll.scrollLeft = LAYOUT.left + t * S.pps - anchorX;
  refresh();
}

function rulerStep() {
  return [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60].find((s) => s * S.pps >= 60) || 120;
}

const COLORS = {
  same: { fill: '#ffffff', stroke: '#55555a', text: '#1d1d1f', bar: 'rgba(80,80,90,0.18)' },
  changed: { fill: '#fff1dc', stroke: '#e07b00', text: '#8a4b00', bar: 'rgba(224,123,0,0.25)' },
  added: { fill: '#e3f5e8', stroke: '#1e9e4a', text: '#11612d', bar: 'rgba(30,158,74,0.25)' },
};

function roundRect(x, y, w, h, r) {
  ctx2d.beginPath();
  ctx2d.roundRect(x, y, w, h, r);
}

function draw() {
  const c = ctx2d;
  const w = viewWidth();
  const h = canvas.clientHeight;
  c.clearRect(0, 0, w, h);
  c.fillStyle = '#ffffff';
  c.fillRect(0, 0, w, h);
  if (!S) return;
  const t0 = tAt(LAYOUT.left);
  const t1 = tAt(w);
  S.hits = [];

  if (S.region) {
    c.fillStyle = S.loop ? 'rgba(47,111,222,0.13)' : 'rgba(47,111,222,0.05)';
    c.fillRect(tx(S.region.start), 0, (S.region.end - S.region.start) * S.pps, h);
  }

  const step = rulerStep();
  c.font = '11px ui-monospace, Menlo, monospace';
  c.textBaseline = 'middle';
  c.textAlign = 'left';
  for (let t = Math.max(0, Math.floor(t0 / step) * step); t <= t1; t += step) {
    const x = Math.round(tx(t)) + 0.5;
    c.strokeStyle = '#e4e4e8';
    c.beginPath();
    c.moveTo(x, LAYOUT.ruler);
    c.lineTo(x, h);
    c.stroke();
    c.fillStyle = '#6b6b70';
    c.fillText(step < 1 ? t.toFixed(2) : t.toFixed(0), x + 3, LAYOUT.ruler / 2);
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

  if ($('#ghost').checked) {
    c.font = '9px ui-monospace, Menlo, monospace';
    c.textAlign = 'center';
    for (const { p, e } of ghosts()) {
      if (p.end < t0 || p.start > t1) continue;
      const x = tx(p.start);
      const y = stringY(p.string) - 13;
      const label = String(p.fret);
      const bw = Math.max(13, c.measureText(label).width + 6);
      const selected = S.selected && S.selected.kind === 'ghost' && S.selected.id === p.id;
      c.setLineDash([3, 2]);
      roundRect(x - 1, y - 5, bw, 10, 3);
      c.fillStyle = 'rgba(255,255,255,0.85)';
      c.fill();
      c.strokeStyle = selected ? '#2f6fde' : e ? '#9a9aa2' : '#c8312b';
      c.stroke();
      c.setLineDash([]);
      c.fillStyle = e ? '#8a8a92' : '#c8312b';
      c.fillText(label, x - 1 + bw / 2, y + 0.5);
      S.hits.push({ x0: x - 1, y0: y - 5, x1: x - 1 + bw, y1: y + 5, sel: { kind: 'ghost', id: p.id } });
    }
  }

  const clash = conflicts();
  const sel = selectedNote();
  // Notes that would hide each other on one string are laid side by side; their bars keep the true onset.
  const rightEdge = new Map();
  c.font = '12px ui-monospace, Menlo, monospace';
  c.textAlign = 'center';
  for (const n of S.notes) {
    if (n.end < t0 || n.start > t1) continue;
    const kind = noteKind(n);
    const col = COLORS[kind];
    const y = stringY(n.string);
    c.fillStyle = col.bar;
    c.fillRect(tx(n.start), y - 2, Math.max(2, (n.end - n.start) * S.pps), 4);
    const isSel = sel === n;
    const label = isSel && S.fretBuffer ? `${S.fretBuffer}_` : String(n.fret);
    const bw = Math.max(15, c.measureText(label).width + 7);
    const x = Math.max(tx(n.start), (rightEdge.get(n.string) ?? -Infinity) + 2);
    rightEdge.set(n.string, x - 1 + bw);
    roundRect(x - 1, y - 8, bw, 16, 3);
    c.fillStyle = isSel ? '#2f6fde' : col.fill;
    c.fill();
    c.lineWidth = clash.has(n.id) ? 2 : 1;
    c.strokeStyle = clash.has(n.id) ? '#c8312b' : isSel ? '#2f6fde' : col.stroke;
    c.stroke();
    c.lineWidth = 1;
    c.fillStyle = isSel ? '#ffffff' : col.text;
    c.fillText(label, x - 1 + bw / 2, y + 0.5);
    S.hits.push({ x0: x - 1, y0: y - 8, x1: x - 1 + bw, y1: y + 8, sel: { kind: 'note', id: n.id } });
  }

  const top = waveTop();
  const waveH = h - top - 8;
  const mid = top + waveH / 2;
  c.strokeStyle = '#e4e4e8';
  c.beginPath();
  c.moveTo(LAYOUT.left, mid + 0.5);
  c.lineTo(w, mid + 0.5);
  c.stroke();
  if (S.peaks) {
    const pk = S.peaks;
    const amp = (waveH / 2) * pk.scale;
    c.fillStyle = '#7d93bd';
    for (let x = LAYOUT.left; x < w; x++) {
      const b0 = Math.floor(tAt(x) * pk.perSec);
      const b1 = Math.max(b0 + 1, Math.floor(tAt(x + 1) * pk.perSec));
      if (b0 >= pk.n) break;
      if (b1 <= 0) continue;
      let lo = 0;
      let hi = 0;
      for (let b = Math.max(0, b0); b < Math.min(pk.n, b1); b++) {
        if (pk.min[b] < lo) lo = pk.min[b];
        if (pk.max[b] > hi) hi = pk.max[b];
      }
      c.fillRect(x, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
    }
  } else {
    c.fillStyle = '#6b6b70';
    c.textAlign = 'left';
    c.fillText('loading audio…', LAYOUT.left + 8, mid);
  }

  const px = Math.round(tx(audio.currentTime || 0)) + 0.5;
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
  c.fillRect(0, LAYOUT.ruler + 1, LAYOUT.left - 4, top - LAYOUT.ruler - 12);
  c.font = '12px ui-monospace, Menlo, monospace';
  c.textAlign = 'center';
  for (let s = 1; s <= 6; s++) {
    c.fillStyle = s === S.cursorString ? '#2f6fde' : '#55555a';
    c.fillText(STRING_NAMES[s - 1], (LAYOUT.left - 4) / 2, stringY(s) + 0.5);
  }
}

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------

function refresh() {
  if (!S) return;
  S.needsDraw = true;
  const dirty = isDirty();
  $('#dirty').textContent = dirty ? '● unsaved' : S.baseUpdated ? 'saved' : 'not saved yet';
  $('#undo').disabled = !S.history.length;
  $('#redo').disabled = !S.future.length;
  $('#play').innerHTML = audio.paused ? '&#9654;' : '&#10073;&#10073;';
  $('#loop').classList.toggle('on', S.loop);
  const kinds = S.notes.map(noteKind);
  const used = new Set(S.notes.map((n) => n.source_id));
  const deleted = S.prediction.notes.filter((p) => !used.has(p.id)).length;
  const nChanged = kinds.filter((k) => k === 'changed').length;
  const nAdded = kinds.filter((k) => k === 'added').length;
  $('#counts').textContent = `${S.notes.length} notes · ${nChanged} changed · ${nAdded} added · ${deleted} deleted`;
  renderNotePanel();
}

function renderNotePanel() {
  const info = $('#note-info');
  const table = $('#evidence');
  table.innerHTML = '';
  const n = selectedNote();
  const ghost = selectedGhost();
  if (!n && !ghost) {
    info.className = 'hint';
    info.textContent = 'Click a note or press →.';
    return;
  }
  info.className = '';
  const note = n || ghost;
  const pred = n ? (n.source_id != null ? S.predById.get(n.source_id) : null) : ghost;
  const lines = [];
  const where = `string ${note.string} (${STRING_NAMES[note.string - 1]}) fret ${note.fret}`;
  const timing = `onset ${note.start.toFixed(3)} s, ${(note.end - note.start).toFixed(2)} s long`;
  lines.push(`<b>${pitchName(note.pitch)}</b> (${note.pitch}) · ${where} · ${timing}`);
  if (n) {
    const kind = noteKind(n);
    const label = { same: 'as predicted', changed: 'changed', added: 'added (Stage 1 missed it)' }[kind];
    lines.push(`Note #${n.id}: <span style="color:${COLORS[kind].stroke}">${label}</span>`);
  } else {
    const e = S.notes.find((x) => x.source_id === ghost.id);
    lines.push(e ? `Prediction #${ghost.id}, now note #${e.id}` : `Prediction #${ghost.id}: <span style="color:var(--deleted)">deleted</span> (R restores it)`);
  }
  if (pred && n && noteKind(n) !== 'same') {
    lines.push(`Predicted: ${pitchName(pred.pitch)} on string ${pred.string} fret ${pred.fret} at ${pred.start.toFixed(3)} s`);
  }
  if (pred && pred.velocity != null) lines.push(`Velocity ${pred.velocity}`);
  info.innerHTML = lines.join('<br>');

  const ev = pred && pred.evidence;
  const head = document.createElement('tr');
  head.innerHTML = `<td></td><td>fret</td><td colspan="2">${ev ? `classifier (for ${pitchName(pred.pitch)})` : 'no classifier evidence'}</td><td></td>`;
  table.appendChild(head);
  for (let s = 1; s <= 6; s++) {
    const fret = note.pitch - S.tuning[s - 1];
    const playable = fret >= 0 && fret <= S.maxFret;
    const p = ev ? ev[s - 1] : null;
    const marks = [];
    if (pred && pred.string === s) marks.push('model');
    if (n && n.string === s) marks.push('tab');
    const tr = document.createElement('tr');
    if (note.string === s) tr.className = 'current';
    tr.innerHTML =
      `<td>${s} ${STRING_NAMES[s - 1]}</td><td>${playable ? fret : '–'}</td>` +
      `<td class="bar">${p != null ? `<div style="width:${Math.round(p * 100)}%"></div>` : ''}</td>` +
      `<td>${p != null ? `${(p * 100).toFixed(1)}%` : ''}</td><td class="hint">${marks.join(', ')}</td>`;
    table.appendChild(tr);
  }
}

function buildMetaForm() {
  const form = $('#meta-form');
  form.innerHTML = '';
  for (const field of S.metaFields) {
    const label = document.createElement('label');
    label.textContent = field.replace('_', ' ');
    label.htmlFor = `meta-${field}`;
    const input = document.createElement(field === 'notes' ? 'textarea' : 'input');
    input.id = `meta-${field}`;
    input.name = field;
    input.value = S.meta[field] || '';
    if (input.tagName === 'INPUT') input.type = 'text';
    if (field === 'source') input.readOnly = true;
    form.append(label, input);
  }
  $('#meta-status').textContent = '';
}

async function saveMeta() {
  const meta = Object.fromEntries(new FormData($('#meta-form')).entries());
  try {
    S.meta = await api(`/api/takes/${encodeURIComponent(S.id)}/meta`, { method: 'PUT', body: JSON.stringify(meta) });
    $('#meta-status').textContent = 'saved';
  } catch (err) {
    $('#meta-status').textContent = `save failed: ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

function canvasPos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function zoneAt(y) {
  if (y < LAYOUT.ruler) return 'ruler';
  return y < waveTop() - 8 ? 'tab' : 'wave';
}

function hitAt(x, y) {
  for (let i = S.hits.length - 1; i >= 0; i--) {
    const h = S.hits[i];
    if (x >= h.x0 && x <= h.x1 && y >= h.y0 && y <= h.y1) return h.sel;
  }
  return null;
}

function nearestString(y) {
  return Math.max(1, Math.min(6, Math.round((y - LAYOUT.tabTop) / LAYOUT.gap) + 1));
}

canvas.addEventListener('mousedown', (e) => {
  if (!S) return;
  canvas.focus();
  const { x, y } = canvasPos(e);
  S.drag = { x0: x, zone: zoneAt(y), moved: false };
  if (S.drag.zone === 'tab') {
    flushFret();
    const hit = hitAt(x, y);
    if (hit) {
      S.selected = hit;
      const note = hit.kind === 'note' ? selectedNote() : selectedGhost();
      S.cursorString = note.string;
    } else {
      S.selected = null;
      S.cursorString = nearestString(y);
    }
    refresh();
  }
});

window.addEventListener('mousemove', (e) => {
  if (!S || !S.drag || S.drag.zone === 'tab') return;
  const { x } = canvasPos(e);
  if (Math.abs(x - S.drag.x0) < 4 && !S.drag.moved) return;
  S.drag.moved = true;
  const [a, b] = [tAt(S.drag.x0), tAt(x)].map((t) => Math.max(0, Math.min(S.duration, t))).sort((p, q) => p - q);
  S.region = { start: a, end: b };
  S.loop = true;
  refresh();
});

window.addEventListener('mouseup', (e) => {
  if (!S || !S.drag) return;
  const drag = S.drag;
  S.drag = null;
  if (drag.zone !== 'tab' && !drag.moved) seek(tAt(canvasPos(e).x));
});

canvas.addEventListener('dblclick', (e) => {
  if (!S) return;
  const { x, y } = canvasPos(e);
  if (zoneAt(y) !== 'tab' || hitAt(x, y)) return;
  addNote(tAt(x), nearestString(y));
});

canvas.addEventListener(
  'wheel',
  (e) => {
    if (!S) return;
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) zoom(Math.exp(-e.deltaY * 0.01), canvasPos(e).x);
    else hscroll.scrollLeft += e.deltaX || e.deltaY;
  },
  { passive: false },
);

hscroll.addEventListener('scroll', () => S && (S.needsDraw = true));
window.addEventListener('resize', () => {
  resizeCanvas();
  if (S) S.needsDraw = true;
});

document.addEventListener('keydown', (e) => {
  if (!S || $('#take-view').hidden) return;
  const mod = e.metaKey || e.ctrlKey;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (mod && key === 's') {
    e.preventDefault();
    save();
    return;
  }
  if (e.target instanceof Element && e.target.matches('input[type=text], textarea, select')) return;
  if (mod && key === 'z') {
    e.preventDefault();
    flushFret();
    if (e.shiftKey) redo();
    else undo();
    return;
  }
  if (mod && key === 'y') {
    e.preventDefault();
    redo();
    return;
  }
  if (mod) return;
  if (/^[0-9]$/.test(key)) {
    e.preventDefault();
    typeFret(key);
    return;
  }
  flushFret();
  const n = selectedNote();
  const handled = {
    ' ': () => togglePlay(),
    ArrowLeft: () => (e.altKey ? n && nudge(n, e.shiftKey ? -0.05 : -0.01) : stepSelection(-1)),
    ArrowRight: () => (e.altKey ? n && nudge(n, e.shiftKey ? 0.05 : 0.01) : stepSelection(1)),
    ArrowUp: () => n && (e.shiftKey ? shiftPitch(n, 1) : moveString(n, -1)),
    ArrowDown: () => n && (e.shiftKey ? shiftPitch(n, -1) : moveString(n, 1)),
    '+': () => n && shiftPitch(n, 1),
    '=': () => n && shiftPitch(n, 1),
    '-': () => n && shiftPitch(n, -1),
    Delete: () => n && deleteNote(n),
    Backspace: () => n && deleteNote(n),
    Enter: () => {},
    Escape: () => {
      S.selected = null;
      refresh();
    },
    Home: () => seek(0),
    a: () => addNote(audio.currentTime, n ? n.string : S.cursorString),
    r: () => revert(),
    p: () => (n || selectedGhost()) && synth.audition((n || selectedGhost()).pitch),
    o: () => (n || selectedGhost()) && playFrom(n || selectedGhost()),
    l: () => {
      if (!S.region) return message('Drag on the waveform to select a loop region');
      S.loop = !S.loop;
      refresh();
    },
    t: () => cycleMix(),
    g: () => {
      $('#ghost').checked = !$('#ghost').checked;
      refresh();
    },
  }[key];
  if (handled) {
    e.preventDefault();
    handled();
  }
});

$('#toolbar').addEventListener('click', (e) => {
  const button = e.target.closest('button');
  if (button) button.blur();
});
$('#toolbar').addEventListener('change', (e) => {
  if (e.target.matches('input[type=checkbox], select')) e.target.blur();
});
$('#play').addEventListener('click', togglePlay);
$('#rate').addEventListener('change', (e) => {
  audio.playbackRate = Number(e.target.value);
  synth.reset(audio.currentTime);
});
$('#loop').addEventListener('click', () => {
  if (!S.region) return message('Drag on the waveform to select a loop region');
  S.loop = !S.loop;
  refresh();
});
$('#vol-audio').addEventListener('input', (e) => (audio.volume = Number(e.target.value)));
$('#vol-tab').addEventListener('input', (e) => {
  synth.ensure();
  synth.setVolume(Number(e.target.value));
});
$('#zoom-in').addEventListener('click', () => zoom(1.5));
$('#zoom-out').addEventListener('click', () => zoom(1 / 1.5));
$('#ghost').addEventListener('change', refresh);
$('#undo').addEventListener('click', undo);
$('#redo').addEventListener('click', redo);
$('#save').addEventListener('click', save);
$('#status').addEventListener('change', (e) => {
  const status = e.target.value;
  edit(() => {
    S.status = status;
  });
});
$('#meta-save').addEventListener('click', saveMeta);
audio.addEventListener('ended', refresh);
audio.addEventListener('pause', refresh);
audio.addEventListener('play', refresh);

window.addEventListener('beforeunload', (e) => {
  if (isDirty()) e.preventDefault();
});
window.addEventListener('hashchange', route);

function tick() {
  if (S) {
    if (!audio.paused) {
      let t = audio.currentTime;
      if (S.loop && S.region && t >= S.region.end) {
        audio.currentTime = S.region.start;
        synth.reset(S.region.start);
        t = S.region.start;
      }
      synth.schedule(t);
      if ($('#follow').checked) {
        const x = tx(t);
        const w = viewWidth();
        if (x > w - 0.15 * (w - LAYOUT.left) || x < LAYOUT.left) {
          hscroll.scrollLeft = Math.max(0, t * S.pps - 0.1 * (w - LAYOUT.left));
        }
      }
      S.needsDraw = true;
    }
    if (S.needsDraw) {
      S.needsDraw = false;
      $('#time').textContent = `${(audio.currentTime || 0).toFixed(2)} / ${S.duration.toFixed(2)} s`;
      draw();
    }
  }
  requestAnimationFrame(tick);
}

route();
requestAnimationFrame(tick);
