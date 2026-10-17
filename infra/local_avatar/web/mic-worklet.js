// Converts the microphone to 16-bit PCM and posts ~20 ms chunks to the page.
// The AudioContext runs at 16 kHz, so the browser does the resampling.
class MicPcm extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Int16Array(320);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      this.buffer[this.filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.filled === this.buffer.length) {
        this.port.postMessage(this.buffer.buffer.slice(0));
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-pcm', MicPcm);
