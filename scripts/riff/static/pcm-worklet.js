'use strict';

const BLOCK = 4096;

// Sends the input, mixed to mono, to the page in blocks of BLOCK frames.
class PcmTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(BLOCK);
    this.n = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let x = 0;
      for (const c of channels) x += c[i];
      this.block[this.n++] = x / channels.length;
      if (this.n === BLOCK) {
        this.port.postMessage(this.block, [this.block.buffer]);
        this.block = new Float32Array(BLOCK);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-tap', PcmTap);
