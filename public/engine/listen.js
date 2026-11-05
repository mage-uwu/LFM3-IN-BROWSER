// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Microphone input for push to talk: the samples as they arrive (MicCapture), and a resampler to the speech
// encoder's 16 kHz that can run while they arrive and gives the same samples as resampling the whole clip at once
// (Resampler: a windowed-sinc filter; an output sample is written once all the input it reads is in).

export class MicCapture {
  // ctx: an AudioContext (created in a tap); stream: a MediaStream with an audio track.
  // onSamples(Float32Array at ctx.sampleRate), onLevel(0..1) for the button's meter
  constructor(ctx, stream, { onSamples, onLevel } = {}) {
    Object.assign(this, { ctx, stream, onSamples, onLevel });
  }
  start() {
    const c = this.ctx;
    this.src = c.createMediaStreamSource(this.stream);
    this.proc = c.createScriptProcessor(2048, 1, 1);
    this.sink = c.createGain();
    this.sink.gain.value = 0;   // a processor only runs when connected to the output: through a silent gain
    this.proc.onaudioprocess = (e) => {
      const x = e.inputBuffer.getChannelData(0);
      let s = 0;
      for (let i = 0; i < x.length; i++) s += x[i] * x[i];
      this.onLevel?.(Math.min(1, Math.sqrt(s / x.length) * 8));
      this.onSamples?.(new Float32Array(x));
    };
    this.src.connect(this.proc);
    this.proc.connect(this.sink);
    this.sink.connect(c.destination);
  }
  stop() { // also releases the microphone, so the phone plays the answer through its speaker again
    if (this.proc) this.proc.onaudioprocess = null;
    try { this.src?.disconnect(); this.proc?.disconnect(); this.sink?.disconnect(); } catch { /* gone */ }
    this.stream.getTracks().forEach((t) => t.stop());
  }
}

export class Resampler {
  constructor(from, to = 16000, zeros = 16) {
    this.from = from; this.to = to;
    this.step = from / to;                                  // input samples per output sample
    this.fc = (0.5 * Math.min(1, to / from)) * 0.94;        // cutoff (cycles per input sample), below the new Nyquist
    this.half = Math.ceil(zeros / (2 * this.fc) / 2);       // kernel half-width in input samples
    this.x = new Float32Array(Math.ceil(from * 31));
    this.n = 0;
    this.out = 0;                                           // output samples written
    // the kernel's taps for 256 fractional positions between input samples (48 kHz -> 16 kHz only uses the first)
    const P = (this.phases = 256), taps = 2 * this.half;
    this.table = new Float32Array((P + 1) * taps);
    for (let p = 0; p <= P; p++) {
      let ws = 0;
      for (let t = 0; t < taps; t++) {
        const d = p / P + this.half - 1 - t;               // distance from tap j = c - half + 1 + t to pos = c + p / P
        const w = 0.5 + 0.5 * Math.cos((Math.PI * d) / (this.half + 1));   // Hann window over the kernel
        const h = (d === 0 ? 2 * this.fc : Math.sin(2 * Math.PI * this.fc * d) / (Math.PI * d)) * w;
        this.table[p * taps + t] = h;
        ws += h;
      }
      for (let t = 0; t < taps; t++) this.table[p * taps + t] /= ws;   // unit gain at DC
    }
  }
  push(x) {
    const k = Math.min(x.length, this.x.length - this.n);
    this.x.set(x.subarray(0, k), this.n);
    this.n += k;
    return this.emit(false);
  }
  finish() { return this.emit(true); }
  emit(final) {
    if (this.from === this.to) { const y = this.x.slice(this.out, this.n); this.out = this.n; return y; }
    const last = final ? Math.floor((this.n * this.to) / this.from) : Math.floor((this.n - 1 - this.half) / this.step) + 1;
    const count = Math.max(0, last - this.out), y = new Float32Array(count), { half, step, phases, table, x, n } = this;
    const taps = 2 * half;
    for (let o = 0; o < count; o++) {
      const pos = (this.out + o) * step, c = Math.floor(pos), base = Math.round((pos - c) * phases) * taps, j0 = c - half + 1;
      let s = 0;
      for (let t = 0; t < taps; t++) { const j = j0 + t; if (j >= 0 && j < n) s += table[base + t] * x[j]; }
      y[o] = s;
    }
    this.out += count;
    return y;
  }
}

// mono samples at any rate -> 16 kHz in one go (a recorded clip), with the same filter
export function resample16k(x, rate) {
  if (rate === 16000) return x;
  const r = new Resampler(rate, 16000);
  const a = r.push(x), b = r.finish(), y = new Float32Array(a.length + b.length);
  y.set(a); y.set(b, a.length);
  return y;
}
