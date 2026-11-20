'use strict';

// Mono 32-bit float WAV of Float32Array `chunks`: keeps the captured samples exactly.
function encodeWav(chunks, sampleRate) {
  const n = chunks.reduce((sum, c) => sum + c.length, 0);
  const buffer = new ArrayBuffer(44 + 4 * n);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  text(0, 'RIFF');
  view.setUint32(4, 36 + 4 * n, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, 4 * sampleRate, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 32, true);
  text(36, 'data');
  view.setUint32(40, 4 * n, true);
  const samples = new Float32Array(buffer, 44, n);
  let offset = 0;
  for (const c of chunks) {
    samples.set(c, offset);
    offset += c.length;
  }
  return buffer;
}
