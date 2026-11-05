// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// LFM3-Omni's speech encoder on WebGPU: 16 kHz audio -> 128-bin log-mel (JS, as the feature extractor) -> three
// stride-2 convolutions over the (mel, time) plane (8 frames become one 768-wide vector) -> an LFM2 stack of 17
// layers (gated short convolutions of width 3, and causal attention over a 32-token window) -> a projector to
// 2048-wide rows that the language model reads in place of its <|audio|> placeholders (12.5 per second of speech).
import * as K from "./kernels.js";
import { buf, f, Kernel, ubuf, writeStream } from "./lfm3.js";
import { BIAS_ACT, GRID, LN_ROWS, MMV, readBack, RMS_ROWS, SILU_ROWS } from "./vision.js";

export const ACFG = {
  D: 768, I: 1792, H: 8, KVH: 4, HD: 96, C: 256, MEL: 128, OUT: 2048, eps: 1e-5, lnEps: 1e-5, theta: 1e6, window: 32,
  rate: 16000, nfft: 400, hop: 160, audioToken: 125031,
  types: ["conv", "conv", "attn", "conv", "conv", "attn", "conv", "conv", "attn", "conv", "conv", "attn", "conv", "conv",
    "attn", "attn", "attn"],
};
const U = GPUBufferUsage;

// ---- the log-mel front end (Lfm3OmniFeatureExtractor) ---------------------------------------------------------
const hzToMel = (f) => (f < 1000 ? (3 * f) / 200 : 15 + (Math.log(f / 1000) * 27) / Math.log(6.4));
const melToHz = (m) => (m < 15 ? (200 * m) / 3 : 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)));
let FILTERS = null;
function melFilters() { // [128, 201]: transformers' mel_filter_bank(norm="slaney", mel_scale="slaney"), transposed
  if (FILTERS) return FILTERS;
  const nb = ACFG.nfft / 2 + 1, nm = ACFG.MEL, lo = hzToMel(0), hi = hzToMel(ACFG.rate / 2);
  const ff = Array.from({ length: nm + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (nm + 1)));
  const fft = Array.from({ length: nb }, (_, i) => ((ACFG.rate / 2) * i) / (nb - 1));
  FILTERS = new Float32Array(nm * nb);
  for (let m = 0; m < nm; m++) {
    const enorm = 2 / (ff[m + 2] - ff[m]);
    for (let k = 0; k < nb; k++) {
      const down = (fft[k] - ff[m]) / (ff[m + 1] - ff[m]);     // -slopes[:, m] / diff[m]
      const up = (ff[m + 2] - fft[k]) / (ff[m + 2] - ff[m + 1]); // slopes[:, m + 2] / diff[m + 1]
      FILTERS[m * nb + k] = Math.max(0, Math.min(down, up)) * enorm;
    }
  }
  return FILTERS;
}
// one log-mel frame: samples [t * hop, t * hop + nfft) of x (zeros past its end) -> out[m * stride + col]
let TABLES = null;
function melFrame(x, t, out, stride, col = t) {
  const { nfft, hop, MEL } = ACFG, nb = nfft / 2 + 1;
  if (!TABLES) {
    const win = new Float64Array(nfft), cs = new Float64Array(nfft), sn = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) {
      win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nfft);   // periodic Hann
      cs[i] = Math.cos((2 * Math.PI * i) / nfft); sn[i] = Math.sin((2 * Math.PI * i) / nfft);
    }
    TABLES = { win, cs, sn, fr: new Float64Array(nfft), pw: new Float64Array(nb) };
  }
  const { win, cs, sn, fr, pw } = TABLES, F = melFilters();
  for (let i = 0; i < nfft; i++) { const j = t * hop + i; fr[i] = (j < x.length ? x[j] : 0) * win[i]; }
  for (let k = 0; k < nb; k++) { // the 400-point DFT, bin by bin
    let re = 0, im = 0, idx = 0;
    for (let i = 0; i < nfft; i++) { re += fr[i] * cs[idx]; im -= fr[i] * sn[idx]; idx += k; if (idx >= nfft) idx -= nfft; }
    pw[k] = re * re + im * im;
  }
  for (let m = 0; m < MEL; m++) {
    let s = 0;
    for (let k = 0; k < nb; k++) s += F[m * nb + k] * pw[k];
    out[m * stride + col] = (Math.max(Math.log10(s), -12) + 4) / 4;
  }
}
// frames for n samples: the extractor pads the end to whole hops (at least one frame)
const frameCount = (n) => {
  const { nfft, hop } = ACFG;
  const padded = n < nfft ? nfft : n + (((-(n - nfft)) % hop) + hop) % hop;
  return 1 + (padded - nfft) / hop;
};
// mono 16 kHz samples -> { mel: Float32Array [128, frames] (mel-major), frames, tokens: ceil(frames / 8) }
export function logMel(samples) {
  const frames = frameCount(samples.length), mel = new Float32Array(ACFG.MEL * frames);
  for (let t = 0; t < frames; t++) melFrame(samples, t, mel, frames);
  return { mel, frames, tokens: Math.ceil(frames / 8) };
}

// The same log-mel computed while the audio arrives (push to talk): frame t is final once samples up to
// t * 160 + 400 are in, and finish() pads the end as logMel does, so the frames equal logMel of the whole clip.
// The encoder is causal and token t reads frames up to 8t + 7, so the first floor(frames / 8) tokens of a partial
// encoding are already the final ones (complete()).
export class MelStream {
  constructor(maxSeconds = 30) {
    this.x = new Float32Array(ACFG.rate * maxSeconds + ACFG.nfft);
    this.n = 0;
    this.cols = new Float32Array(ACFG.MEL * (frameCount(this.x.length) + 1));   // frame-major while streaming
    this.frames = 0;
    this.done = false;
  }
  get seconds() { return this.n / ACFG.rate; }
  get full() { return this.n >= this.x.length - ACFG.nfft; }
  push(samples) {
    const k = Math.min(samples.length, this.x.length - ACFG.nfft - this.n);
    if (k > 0) { this.x.set(samples.subarray(0, k), this.n); this.n += k; }
    const { hop, nfft, MEL } = ACFG, col = new Float32Array(MEL);
    const view = this.x.subarray(0, this.n);
    while (this.frames * hop + nfft <= this.n) { this.frame(view, col); }
  }
  frame(view, col) {
    melFrame(view, this.frames, col, 1, 0);
    this.cols.set(col, this.frames * ACFG.MEL);
    this.frames++;
  }
  finish() { // the last frames, zero-padded like logMel
    const total = frameCount(this.n), col = new Float32Array(ACFG.MEL), view = this.x.subarray(0, this.n);
    while (this.frames < total) this.frame(view, col);
    this.done = true;
  }
  complete() { return this.done ? Math.ceil(this.frames / 8) : Math.floor(this.frames / 8); }   // tokens that are final
  // the [128, width] mel-major layout the encoder reads: the frames so far, zeros after them
  layout(width = this.frames) {
    const { MEL } = ACFG, mel = new Float32Array(MEL * width);
    for (let t = 0; t < this.frames; t++) for (let m = 0; m < MEL; m++) mel[m * width + t] = this.cols[t * MEL + m];
    return mel;
  }
}

// ---- kernels ---------------------------------------------------------------------------------------------------
// the downsampler: conv_in (1 -> 256 channels, 3x3, stride 2) + SiLU over the [128 mel, T time] plane
const CONV_IN = GRID + /* wgsl */ `
struct P { H: u32, W: u32, Ho: u32, Wo: u32 };   // input H x W, output Ho x Wo, 256 channels
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= 256u * p.Ho * p.Wo) { return; }
  let c = i / (p.Ho * p.Wo); let hw = i % (p.Ho * p.Wo); let h = hw / p.Wo; let ww = hw % p.Wo;
  var acc = b[c];
  for (var ky = 0u; ky < 3u; ky++) { for (var kx = 0u; kx < 3u; kx++) {
    let r = i32(2u * h + ky) - 1; let q = i32(2u * ww + kx) - 1;
    if (r >= 0 && q >= 0 && r < i32(p.H) && q < i32(p.W)) { acc += w[c * 9u + ky * 3u + kx] * x[u32(r) * p.W + u32(q)]; }
  } }
  y[i] = acc / (1.0 + exp(-acc));
}`;
// depthwise 3x3 stride 2 + bias, on [256, H, W]
const DW_S2 = GRID + /* wgsl */ `
struct P { H: u32, W: u32, Ho: u32, Wo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= 256u * p.Ho * p.Wo) { return; }
  let c = i / (p.Ho * p.Wo); let hw = i % (p.Ho * p.Wo); let h = hw / p.Wo; let ww = hw % p.Wo;
  var acc = b[c];
  for (var ky = 0u; ky < 3u; ky++) { for (var kx = 0u; kx < 3u; kx++) {
    let r = i32(2u * h + ky) - 1; let q = i32(2u * ww + kx) - 1;
    if (r >= 0 && q >= 0 && r < i32(p.H) && q < i32(p.W)) { acc += w[c * 9u + ky * 3u + kx] * x[(c * p.H + u32(r)) * p.W + u32(q)]; }
  } }
  y[i] = acc;
}`;
// pointwise 256 -> 256 + bias + SiLU; flat = 1 writes [W, 256 * H] (the encoder's input rows) instead of [256, H, W]
const PW = GRID + /* wgsl */ `
struct P { H: u32, W: u32, flat: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  let HW = p.H * p.W;
  if (i >= 256u * HW) { return; }
  let c = i / HW; let hw = i % HW; let h = hw / p.W; let ww = hw % p.W;
  var acc = b[c];
  for (var k = 0u; k < 256u; k++) { acc += w[c * 256u + k] * x[k * HW + hw]; }
  let v = acc / (1.0 + exp(-acc));
  if (p.flat == 1u) { y[ww * 256u * p.H + c * p.H + h] = v; } else { y[i] = v; }
}`;
// LFM2 short convolution: in_proj gives [B | C | x]; y = C * causal depthwise conv (width 3) of B*x
const SHORT_CONV = GRID + /* wgsl */ `
struct P { D: u32, n: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> t3: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.n) { return; }
  let t = i / p.D; let c = i % p.D; let S = 3u * p.D;
  var acc = 0.0;
  for (var k = 0u; k < 3u; k++) {
    let s = i32(t) + i32(k) - 2;
    if (s >= 0) { let r = u32(s) * S; acc += w[c * 3u + k] * t3[r + c] * t3[r + 2u * p.D + c]; }
  }
  y[i] = t3[t * S + p.D + c] * acc;
}`;
// per-head RMSNorm of q (8 heads) and k (4 heads) in [T, 1536] rows, then RoPE over the 96-dim head (position = t)
const QK_ROPE = /* wgsl */ `
struct P { eps: f32, theta: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qn: array<f32>;
@group(0) @binding(2) var<storage, read> kn: array<f32>;
@group(0) @binding(3) var<storage, read_write> qkv: array<f32>;
var<workgroup> red: array<f32, 128>;
var<workgroup> xs: array<f32, 96>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x; let hd = wg.y;   // 0-7 q heads, 8-11 k heads
  let isK = hd >= 8u;
  var off = hd * 96u;
  if (isK) { off = 768u + (hd - 8u) * 96u; }
  let base = t * 1536u + off;
  var v = 0.0;
  if (li < 96u) { v = qkv[base + li]; }
  red[li] = v * v;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 96.0 + p.eps);
  var xn = 0.0;
  if (li < 96u) {
    var wv = qn[li];
    if (isK) { wv = kn[li]; }
    xn = v * inv * wv;
    xs[li] = xn;
  }
  workgroupBarrier();
  if (li < 96u) {
    let j = li % 48u;
    let ang = f32(t) * pow(p.theta, -f32(2u * j) / 96.0);
    var rot = 0.0;
    if (li < 48u) { rot = -xs[li + 48u]; } else { rot = xs[li - 48u]; }
    qkv[base + li] = xn * cos(ang) + rot * sin(ang);
  }
}`;
// causal attention over the last 32 positions (8 query heads share 4 key/value heads)
const ATTN_WIN = /* wgsl */ `
struct P { T: u32, scale: f32, win: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
var<workgroup> qs: array<f32, 96>;
var<workgroup> ss: array<f32, 32>;
@compute @workgroup_size(96)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x; let h = wg.y; let kvh = h / 2u;
  qs[li] = qkv[t * 1536u + h * 96u + li] * p.scale;
  workgroupBarrier();
  let lo = select(0u, t + 1u - p.win, t + 1u > p.win);
  let n = t + 1u - lo;   // keys lo..t (at most 32)
  if (li < n) {
    let kb = (lo + li) * 1536u + 768u + kvh * 96u;
    var d = 0.0;
    for (var i = 0u; i < 96u; i++) { d += qs[i] * qkv[kb + i]; }
    ss[li] = d;
  }
  workgroupBarrier();
  var m = -1.0e30;
  for (var i = 0u; i < n; i++) { m = max(m, ss[i]); }
  var l = 0.0; var acc = 0.0;
  for (var i = 0u; i < n; i++) {
    let pr = exp(ss[i] - m);
    l += pr;
    acc += pr * qkv[(lo + i) * 1536u + 1152u + kvh * 96u + li];
  }
  o[t * 768u + h * 96u + li] = acc / l;
}`;

// ---- the model -------------------------------------------------------------------------------------------------
export class Audio {
  constructor(device, { quant = "q8", maxTokens = 375, prequant = null } = {}) {  // 375 tokens = 30 s of speech
    this.device = device;
    this.quant = quant;
    this.maxT = maxTokens;
    this.prequant = prequant;
    this.plans = new Map();
  }
  qm(N, Kd, label) {
    const G = (N * Kd) / (this.quant === "q4g64" ? 64 : 32);
    return { N, K: Kd, quant: this.quant, q: buf(this.device, G * 32, undefined, label + ".q"), s: buf(this.device, G * 4, undefined, label + ".s") };
  }
  allocWeights() {
    const { D, I, C, OUT } = ACFG, d = this.device, v = (n) => buf(d, n * 4);
    this.w = {
      convInW: v(C * 9), convInB: v(C), convMidW: v(C * 9), convMidB: v(C), linMidW: v(C * C), linMidB: v(C),
      convEndW: v(C * 9), convEndB: v(C), linEndW: v(C * C), linEndB: v(C),
      linOut: this.qm(D, C * 16, "a.linOut"), linOutB: v(D), layers: [], finalNorm: v(D),
    };
    ACFG.types.forEach((type, i) => {
      const L = { type, opNorm: v(D), ffnNorm: v(D), w13: this.qm(2 * I, D, `a${i}.w13`), w2: this.qm(D, I, `a${i}.w2`) };
      if (type === "conv") Object.assign(L, { convW: v(D * 3), inProj: this.qm(3 * D, D, `a${i}.in`), outProj: this.qm(D, D, `a${i}.out`) });
      else Object.assign(L, { qNorm: v(96), kNorm: v(96), qkv: this.qm(2 * D, D, `a${i}.qkv`), oProj: this.qm(D, D, `a${i}.o`) });
      this.w.layers.push(L);
    });
    this.w.proj = { lnW: v(D), lnB: v(D), l1: this.qm(OUT, D, "a.l1"), l1b: v(OUT), l2: this.qm(OUT, OUT, "a.l2"), l2b: v(OUT) };
  }
  weightList() { // the export order (tools/audio_export.py writes the same keys)
    const W = this.w, L = [];
    for (const k of ["convInW", "convInB", "convMidW", "convMidB", "linMidW", "linMidB", "convEndW", "convEndB", "linEndW", "linEndB"]) L.push([k, W[k]]);
    L.push(["linOut.q", W.linOut.q], ["linOut.s", W.linOut.s], ["linOutB", W.linOutB]);
    W.layers.forEach((l, i) => {
      for (const k of ["opNorm", "ffnNorm", "convW", "qNorm", "kNorm"]) if (l[k]) L.push([`${i}.${k}`, l[k]]);
      for (const k of ["inProj", "outProj", "qkv", "oProj", "w13", "w2"]) if (l[k]) L.push([`${i}.${k}.q`, l[k].q], [`${i}.${k}.s`, l[k].s]);
    });
    const p = W.proj;
    L.push(["finalNorm", W.finalNorm], ["ln.w", p.lnW], ["ln.b", p.lnB], ["l1.q", p.l1.q], ["l1.s", p.l1.s], ["l1.b", p.l1b],
      ["l2.q", p.l2.q], ["l2.s", p.l2.s], ["l2.b", p.l2b]);
    return L;
  }
  async load(onProgress = () => {}) {
    this.allocWeights();
    await writeStream(this.device, this.weightList(), this.prequant.manifest, this.prequant.chunks(), onProgress);
    this.build();
  }
  build() {
    const d = this.device, { D, I, C, MEL, OUT } = ACFG, T = this.maxT, F = T * 8;
    this.k = {
      gemm: new Kernel(d, ["u", "r", "r", "r", "r", "w"], { main: K.gemvSource(this.quant, MMV, { batched: true }) }, "a.gemm"),
      convIn: new Kernel(d, ["u", "r", "r", "r", "w"], { main: CONV_IN }, "a.convIn"),
      dw: new Kernel(d, ["u", "r", "r", "r", "w"], { main: DW_S2 }, "a.dw"),
      pw: new Kernel(d, ["u", "r", "r", "r", "w"], { main: PW }, "a.pw"),
      rms: new Kernel(d, ["u", "r", "r", "w"], { main: RMS_ROWS }, "a.rms"),
      ln: new Kernel(d, ["u", "r", "r", "r", "w"], { main: LN_ROWS }, "a.ln"),
      sconv: new Kernel(d, ["u", "r", "r", "w"], { main: SHORT_CONV }, "a.sconv"),
      qk: new Kernel(d, ["u", "r", "r", "w"], { main: QK_ROPE }, "a.qk"),
      attn: new Kernel(d, ["u", "r", "w"], { main: ATTN_WIN }, "a.attn"),
      silu: new Kernel(d, ["u", "r", "w"], { main: SILU_ROWS }, "a.silu"),
      bias: new Kernel(d, ["u", "r", "w"], { main: BIAS_ACT }, "a.bias"),
    };
    const b = (n) => buf(d, n * 4);
    this.st = buf(d, 32, U.STORAGE | U.COPY_DST);
    Object.assign(this, { mel: b(MEL * F), s1: b(C * 64 * (F / 2)), s2: b(C * 32 * (F / 4)), s3: b(C * 32 * (F / 4)),
      s4: b(C * 16 * (F / 8)), xin: b(T * C * 16), h: b(T * D), xn: b(T * D), t3: b(T * 3 * D), c: b(T * D),
      u: b(T * 2 * I), a: b(T * I), ln: b(T * D), p1: b(T * OUT), emb: b(T * OUT) });
  }
  plan(frames) {
    if (this.plans.has(frames)) return this.plans.get(frames);
    const d = this.device, k = this.k, { D, I, C, MEL, OUT } = ACFG, steps = [];
    const W1 = Math.ceil(frames / 2), W2 = Math.ceil(W1 / 2), T = Math.ceil(W2 / 2);
    const grid = (n) => { const w = Math.ceil(n / 256); return [Math.min(w, 65535), Math.ceil(w / 65535)]; };
    const gemm = (m, x, y, rows, acc = 0) => steps.push({ p: k.gemm, wg: [Math.ceil(m.N / 8), 1, Math.ceil(rows / MMV)],
      bg: k.gemm.bind([ubuf(d, [m.N, m.K, m.N, acc, rows, 0, 0, 0]), this.st, m.q, m.s, x, y]) });
    const ew = (kern, n, bufs, u) => steps.push({ p: kern, wg: [...grid(n), 1], bg: kern.bind([ubuf(d, u), ...bufs]) });
    const W = this.w;
    ew(k.convIn, C * 64 * W1, [this.mel, W.convInW, W.convInB, this.s1], [MEL, frames, 64, W1]);
    ew(k.dw, C * 32 * W2, [this.s1, W.convMidW, W.convMidB, this.s2], [64, W1, 32, W2]);
    ew(k.pw, C * 32 * W2, [this.s2, W.linMidW, W.linMidB, this.s3], [32, W2, 0, 0]);
    ew(k.dw, C * 16 * T, [this.s3, W.convEndW, W.convEndB, this.s4], [32, W2, 16, T]);
    ew(k.pw, C * 16 * T, [this.s4, W.linEndW, W.linEndB, this.xin], [16, T, 1, 0]);
    gemm(W.linOut, this.xin, this.h, T);
    ew(k.bias, T * D, [W.linOutB, this.h], [D, T * D, 0, 0]);
    const rmsU = ubuf(d, [D, f(ACFG.eps), 0, 0]);
    const rms = (w, x, y) => steps.push({ p: k.rms, wg: [T, 1, 1], bg: k.rms.bind([rmsU, w, x, y]) });
    for (const L of W.layers) {
      rms(L.opNorm, this.h, this.xn);
      if (L.type === "conv") {
        gemm(L.inProj, this.xn, this.t3, T);
        ew(k.sconv, T * D, [this.t3, L.convW, this.c], [D, T * D, 0, 0]);
        gemm(L.outProj, this.c, this.h, T, 1);
      } else {
        gemm(L.qkv, this.xn, this.t3, T);
        steps.push({ p: k.qk, wg: [T, 12, 1], bg: k.qk.bind([ubuf(d, [f(ACFG.eps), f(ACFG.theta), 0, 0]), L.qNorm, L.kNorm, this.t3]) });
        steps.push({ p: k.attn, wg: [T, 8, 1], bg: k.attn.bind([ubuf(d, [T, f(1 / Math.sqrt(96)), ACFG.window, 0]), this.t3, this.c]) });
        gemm(L.oProj, this.c, this.h, T, 1);
      }
      rms(L.ffnNorm, this.h, this.xn);
      gemm(L.w13, this.xn, this.u, T);
      ew(k.silu, T * I, [this.u, this.a], [I, T * I, 0, 0]);
      gemm(L.w2, this.a, this.h, T, 1);
      steps.push({ layerEnd: true });
    }
    rms(W.finalNorm, this.h, this.xn);
    const P = W.proj;
    steps.push({ p: k.ln, wg: [T, 1, 1], bg: k.ln.bind([ubuf(d, [D, f(ACFG.lnEps), 0, 0]), P.lnW, P.lnB, this.xn, this.ln]) });
    gemm(P.l1, this.ln, this.p1, T);
    ew(k.bias, T * OUT, [P.l1b, this.p1], [OUT, T * OUT, 1, 0]);
    gemm(P.l2, this.p1, this.emb, T);
    ew(k.bias, T * OUT, [P.l2b, this.emb], [OUT, T * OUT, 0, 0]);
    const plan = { steps, T };
    if (this.plans.size >= 24) this.plans.delete(this.plans.keys().next().value);   // the oldest width
    this.plans.set(frames, plan);
    return plan;
  }
  // mel: Float32Array [128, frames] (logMel). Returns the GPU buffer of [tokens, 2048] rows for the language model.
  async encode(mel, frames, { debug = false } = {}) {
    const d = this.device, { D } = ACFG;
    const { steps, T } = this.plan(frames);
    if (T > this.maxT) throw new Error(`audio too long: ${T} tokens (at most ${this.maxT})`);
    d.queue.writeBuffer(this.mel, 0, mel);
    let dbg = null, li = 0;
    if (debug) dbg = buf(d, ACFG.types.length * T * D * 4, U.COPY_DST | U.COPY_SRC);
    const enc = d.createCommandEncoder();
    let pass = enc.beginComputePass();
    for (const s of steps) {
      if (s.layerEnd) {
        if (dbg) { pass.end(); enc.copyBufferToBuffer(this.h, 0, dbg, li * T * D * 4, T * D * 4); pass = enc.beginComputePass(); }
        li++;
        continue;
      }
      pass.setPipeline(s.p.pipelines.main);
      pass.setBindGroup(0, s.bg);
      pass.dispatchWorkgroups(...s.wg);
    }
    pass.end();
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
    const out = { emb: this.emb, rows: T };
    if (dbg) {
      out.layers = new Float32Array(await readBack(d, dbg, ACFG.types.length * T * D * 4));
      out.emb32 = new Float32Array(await readBack(d, this.emb, T * ACFG.OUT * 4));
      dbg.destroy();
    }
    return out;
  }
}
