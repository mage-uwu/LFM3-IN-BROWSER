// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// LFM3-Omni's speech output on WebGPU.
//  * DepthDecoder: from the language model's hidden state, the 12 codebook codes of one 80 ms speech frame, revealed
//    MaskGIT-style over 3 passes of 4 codebooks (a bidirectional 3-layer LFM3 over 12 slots), sampled on the CPU
//    with the model's defaults (temperature 0.4, repetition penalty 1.2 over the last 64 frames); and the sum of the
//    12 code embeddings that the language model reads as its next input (LFM3.setExternalEmbeddings).
//  * Codec: the Liquid codec's decoder, streaming: codes -> FSQ values -> 512-wide embeddings, each repeated over 6
//    STFT steps -> a causal 8-layer LFM3 (short convolutions of width 7, attention over a 30-step window, sandwich
//    norms) carrying its state from chunk to chunk -> log-magnitude and phase -> inverse STFT (1280-point, hop 320)
//    with overlap-add: 1920 samples of 24 kHz audio per frame.
import * as K from "./kernels.js";
import { buf, f, Kernel, ubuf, writeStream } from "./lfm3.js";
import { BIAS_ACT, GRID, MMV, readBack, RMS_ROWS, SILU_ROWS } from "./vision.js";

export const SCFG = {
  NQ: 12, CB: 2017, EOS: 2016, D: 1024, I: 2816, H: 32, KVH: 8, HD: 32, layers: 3, eps: 1e-5, theta: 1e6, LMD: 2048,
  schedule: [4, 4, 4], temperature: 0.4, penalty: 1.2, history: 64, nText: 6, nAudio: 12,
  mixedStart: 125015, textEnd: 125014, imEnd: 124900,
};
export const CCFG = {
  D: 512, I: 2304, H: 8, KVH: 2, HD: 64, types: ["conv", "conv", "attn", "conv", "attn", "conv", "attn", "conv"],
  window: 30, ring: 128, stack: 6, nfft: 1280, hop: 320, rate: 24000, levels: [8, 7, 6, 6], QD: 256, eps: 1e-5,
  theta: 1e6, maxFrames: 8,
};
const U = GPUBufferUsage;

// one weight of a quantized [N, Kd] matrix (q8: groups of 32, f32 scale; q4g64: groups of 64, f16 scale and min)
function deq(quant) {
  return quant === "q4g64" ? /* wgsl */ `
fn wAt(row: u32, k: u32, Kd: u32) -> f32 {
  let gi = row * (Kd / 64u) + k / 64u; let j = k % 64u;
  let sm = unpack2x16float(ws[gi]);
  let u = wq[gi * 2u + j / 32u][(j / 8u) % 4u];
  return f32((u >> (4u * (j % 8u))) & 15u) * sm.x + sm.y;
}` : /* wgsl */ `
fn wAt(row: u32, k: u32, Kd: u32) -> f32 {
  let gi = row * (Kd / 32u) + k / 32u; let j = k % 32u;
  let u = wq[gi * 2u + j / 16u][(j / 4u) % 4u];
  return f32(extractBits(i32(u), 8u * (j % 4u), 8u)) * bitcast<f32>(ws[gi]);
}`;
}
// depth decoder input: slot i = depth_linear row i + (revealed ? embedding_i[code_i] : mask embedding)
const DEPTH_IN = (quant) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> codes: array<u32>;   // 0xffffffff: masked
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read> din: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> x: array<f32>;
${deq(quant)}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = wg.y; let k = wg.x * 256u + li;
  let c = codes[i];
  var e = mask[k];
  if (c != 0xffffffffu) { e = wAt(i * 2017u + c, k, 1024u); }
  x[i * 1024u + k] = din[i * 1024u + k] + e;
}`;
// the language model's input for a frame: the sum of the 12 codebooks' rows of the codebook embedding
const EMBED_SUM = (quant) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
${deq(quant)}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let k = wg.x * 256u + li;
  var s = 0.0;
  for (var i = 0u; i < 12u; i++) { s += wAt(i * 2017u + codes[i], k, 2048u); }
  y[k] = s;
}`;
// logits of codebook i: its embedding table times its normalized slot (8 rows per workgroup, 32 lanes per row)
const LOGITS = (quant) => /* wgsl */ `
@group(0) @binding(0) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> ws: array<u32>;
@group(0) @binding(2) var<storage, read> xs: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
${deq(quant)}
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = wg.y; let r = wg.x * 8u + li / 32u; let t = li % 32u;
  var s = 0.0;
  if (r < 2017u) { for (var k = t; k < 1024u; k += 32u) { s += wAt(i * 2017u + r, k, 1024u) * xs[i * 1024u + k]; } }
  red[li] = s;
  workgroupBarrier();
  for (var st = 16u; st > 0u; st >>= 1u) { if (t < st) { red[li] += red[li + st]; } workgroupBarrier(); }
  if (t == 0u && r < 2017u) { out[i * 2017u + r] = red[li]; }
}`;
// RMSNorm with a weight per row (the 12 codebooks' own output norms)
const RMS_ROWW = /* wgsl */ `
struct P { D: u32, eps: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let base = wg.x * p.D;
  var ss = 0.0;
  for (var i = li; i < p.D; i += 256u) { let v = x[base + i]; ss += v * v; }
  red[li] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { y[base + i] = w[base + i] * (x[base + i] * inv); }
}`;
// LFM3 attention, part 1: per-head RMSNorm of q and k and RoPE (positions p.pos0 + row) in rows [q | k | v | gate];
// part 2 stores k and v in a ring of past positions (codec) or leaves them in place (depth decoder)
const QK_ROPE = /* wgsl */ `
struct P { H: u32, KVH: u32, HD: u32, stride: u32, pos0: u32, eps: f32, theta: f32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qn: array<f32>;
@group(0) @binding(2) var<storage, read> kn: array<f32>;
@group(0) @binding(3) var<storage, read_write> qkv: array<f32>;
var<workgroup> red: array<f32, 64>;
var<workgroup> xs: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x; let hd = wg.y;
  let isK = hd >= p.H;
  var off = hd * p.HD;
  if (isK) { off = p.H * p.HD + (hd - p.H) * p.HD; }
  let base = t * p.stride + off;
  var v = 0.0;
  if (li < p.HD) { v = qkv[base + li]; }
  red[li] = v * v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.HD) + p.eps);
  var xn = 0.0;
  if (li < p.HD) {
    var wv = qn[li];
    if (isK) { wv = kn[li]; }
    xn = v * inv * wv;
    xs[li] = xn;
  }
  workgroupBarrier();
  if (li < p.HD) {
    let half = p.HD / 2u;
    let j = li % half;
    let ang = f32(p.pos0 + t) * pow(p.theta, -f32(2u * j) / f32(p.HD));
    var rot = 0.0;
    if (li < half) { rot = -xs[li + half]; } else { rot = xs[li - half]; }
    qkv[base + li] = xn * cos(ang) + rot * sin(ang);
  }
}`;
// depth decoder attention: bidirectional over the 12 slots, 32 heads of 32 sharing 8 key/value heads, sigmoid gate
const ATTN12 = /* wgsl */ `
@group(0) @binding(0) var<storage, read> qkv: array<f32>;   // rows of 1568: q 1024 | k 256 | v 256 | gate 32
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
var<workgroup> sc: array<f32, 12>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x; let h = wg.y; let kvh = h / 4u;
  if (li < 12u) {
    var d = 0.0;
    for (var i = 0u; i < 32u; i++) { d += qkv[t * 1568u + h * 32u + i] * qkv[li * 1568u + 1024u + kvh * 32u + i]; }
    sc[li] = d * 0.17677669529663687;
  }
  workgroupBarrier();
  var m = -1.0e30;
  for (var j = 0u; j < 12u; j++) { m = max(m, sc[j]); }
  var l = 0.0; var acc = 0.0;
  for (var j = 0u; j < 12u; j++) { let e = exp(sc[j] - m); l += e; acc += e * qkv[j * 1568u + 1280u + kvh * 32u + li]; }
  let gate = 1.0 / (1.0 + exp(-qkv[t * 1568u + 1536u + h]));
  o[t * 1024u + h * 32u + li] = acc / l * gate;
}`;

// ---- the depth decoder -----------------------------------------------------------------------------------------
export class DepthDecoder {
  constructor(device, { quant = "q8", prequant = null } = {}) {
    this.device = device; this.quant = quant; this.prequant = prequant;
    this.rand = Math.random;
  }
  qm(N, Kd, label) {
    const G = (N * Kd) / (this.quant === "q4g64" ? 64 : 32);
    return { N, K: Kd, quant: this.quant, q: buf(this.device, G * 32, undefined, label + ".q"), s: buf(this.device, G * 4, undefined, label + ".s") };
  }
  allocWeights() {
    const { D, I, NQ, CB, LMD } = SCFG, v = (n) => buf(this.device, n * 4);
    this.w = { dl: this.qm(NQ * D, LMD, "s.dl"), dlB: v(NQ * D), mask: v(D), tab: this.qm(NQ * CB, D, "s.tab"), tabNorm: v(NQ * D),
      layers: [], finalNorm: v(D), cbe: this.qm(NQ * CB, LMD, "s.cbe") };
    for (let i = 0; i < SCFG.layers; i++) {
      this.w.layers.push({ opNorm: v(D), ffnNorm: v(D), qNorm: v(32), kNorm: v(32), qkvg: this.qm(1568, D, `s${i}.qkvg`),
        oProj: this.qm(D, D, `s${i}.o`), w13: this.qm(2 * I, D, `s${i}.w13`), w2: this.qm(D, I, `s${i}.w2`) });
    }
  }
  weightList() { // the export order (tools/speech_export.py)
    const W = this.w, L = [["dl.q", W.dl.q], ["dl.s", W.dl.s], ["dl.b", W.dlB], ["mask", W.mask], ["tab.q", W.tab.q], ["tab.s", W.tab.s], ["tabNorm", W.tabNorm]];
    W.layers.forEach((l, i) => {
      L.push([`${i}.opNorm`, l.opNorm], [`${i}.ffnNorm`, l.ffnNorm], [`${i}.qNorm`, l.qNorm], [`${i}.kNorm`, l.kNorm]);
      for (const k of ["qkvg", "oProj", "w13", "w2"]) L.push([`${i}.${k}.q`, l[k].q], [`${i}.${k}.s`, l[k].s]);
    });
    L.push(["finalNorm", W.finalNorm], ["cbe.q", W.cbe.q], ["cbe.s", W.cbe.s]);
    return L;
  }
  async load(onProgress = () => {}) {
    this.allocWeights();
    await writeStream(this.device, this.weightList(), this.prequant.manifest, this.prequant.chunks(), onProgress);
    this.build();
  }
  build() {
    const d = this.device, { D, I, NQ, CB, LMD } = SCFG, q = this.quant, b = (n) => buf(d, n * 4);
    const kern = (kinds, src, label) => new Kernel(d, kinds, { main: src }, label);
    this.k = {
      gemm: kern(["u", "r", "r", "r", "r", "w"], K.gemvSource(q, MMV, { batched: true }), "s.gemm"),
      bias: kern(["u", "r", "w"], BIAS_ACT, "s.bias"),
      din: kern(["r", "r", "r", "r", "r", "w"], DEPTH_IN(q), "s.din"),
      rms: kern(["u", "r", "r", "w"], RMS_ROWS, "s.rms"),
      rmsw: kern(["u", "r", "r", "w"], RMS_ROWW, "s.rmsw"),
      qk: kern(["u", "r", "r", "w"], QK_ROPE, "s.qk"),
      attn: kern(["r", "w"], ATTN12, "s.attn"),
      silu: kern(["u", "r", "w"], SILU_ROWS, "s.silu"),
      logits: kern(["r", "r", "r", "w"], LOGITS(q), "s.logits"),
      esum: kern(["r", "r", "r", "w"], EMBED_SUM(q), "s.esum"),
    };
    this.st = buf(d, 32, U.STORAGE | U.COPY_DST);
    Object.assign(this, { codes: buf(d, 64, U.STORAGE | U.COPY_DST), din: b(NQ * D), x: b(NQ * D), h: b(NQ * D), xn: b(NQ * D),
      qkv: b(NQ * 1568), o: b(NQ * D), u: b(NQ * 2 * I), a: b(NQ * I), lg: b(NQ * CB), frame: b(LMD),
      lgRead: buf(d, NQ * CB * 4, U.MAP_READ | U.COPY_DST) });
    const W = this.w, k = this.k, ub = (w) => ubuf(d, w);
    const gemm = (m, x, y, rows, acc = 0) => ({ p: k.gemm, wg: [Math.ceil(m.N / 8), 1, Math.ceil(rows / MMV)], bg: k.gemm.bind([ub([m.N, m.K, m.N, acc, rows, 0, 0, 0]), this.st, m.q, m.s, x, y]) });
    const grid = (n) => [Math.ceil(n / 256), 1, 1];
    this.hidden = null;
    this.headSteps = (hlast) => [gemm(W.dl, hlast, this.din, 1),
      { p: k.bias, wg: grid(NQ * D), bg: k.bias.bind([ub([NQ * D, NQ * D, 0, 0]), W.dlB, this.din]) }];
    const rmsU = ub([D, f(SCFG.eps), 0, 0]);
    const steps = [{ p: k.din, wg: [D / 256, NQ, 1], bg: k.din.bind([this.codes, W.tab.q, W.tab.s, this.din, W.mask, this.h]) }];
    for (const L of W.layers) {
      steps.push({ p: k.rms, wg: [NQ, 1, 1], bg: k.rms.bind([rmsU, L.opNorm, this.h, this.xn]) });
      steps.push(gemm(L.qkvg, this.xn, this.qkv, NQ));
      steps.push({ p: k.qk, wg: [NQ, 40, 1], bg: k.qk.bind([ub([32, 8, 32, 1568, 0, f(SCFG.eps), f(SCFG.theta), 0]), L.qNorm, L.kNorm, this.qkv]) });
      steps.push({ p: k.attn, wg: [NQ, 32, 1], bg: k.attn.bind([this.qkv, this.o]) });
      steps.push(gemm(L.oProj, this.o, this.h, NQ, 1));
      steps.push({ p: k.rms, wg: [NQ, 1, 1], bg: k.rms.bind([rmsU, L.ffnNorm, this.h, this.xn]) });
      steps.push(gemm(L.w13, this.xn, this.u, NQ));
      steps.push({ p: k.silu, wg: grid(NQ * I), bg: k.silu.bind([ub([I, NQ * I, 0, 0]), this.u, this.a]) });
      steps.push(gemm(L.w2, this.a, this.h, NQ, 1));
    }
    steps.push({ p: k.rms, wg: [NQ, 1, 1], bg: k.rms.bind([rmsU, W.finalNorm, this.h, this.xn]) });
    steps.push({ p: k.rmsw, wg: [NQ, 1, 1], bg: k.rmsw.bind([rmsU, W.tabNorm, this.xn, this.o]) });
    steps.push({ p: k.logits, wg: [Math.ceil(CB / 8), NQ, 1], bg: k.logits.bind([W.tab.q, W.tab.s, this.o, this.lg]) });
    this.passSteps = steps;
    this.sumStep = { p: k.esum, wg: [LMD / 256, 1, 1], bg: k.esum.bind([this.codes, W.cbe.q, W.cbe.s, this.frame]) };
  }
  run(steps, after) {
    const d = this.device, enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    for (const s of steps) { pass.setPipeline(s.p.pipelines.main); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.wg); }
    pass.end();
    after?.(enc);
    d.queue.submit([enc.finish()]);
  }
  async logits(codes) { // one pass: scores of all 12 codebooks given the revealed codes (-1: masked)
    const d = this.device;
    d.queue.writeBuffer(this.codes, 0, Uint32Array.from(codes, (c) => (c < 0 ? 0xffffffff : c)));
    this.run(this.passSteps, (enc) => enc.copyBufferToBuffer(this.lg, 0, this.lgRead, 0, SCFG.NQ * SCFG.CB * 4));
    await this.lgRead.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.lgRead.getMappedRange().slice(0));
    this.lgRead.unmap();
    return out;
  }
  // one frame from the language model's hidden state (a GPU buffer of 2048 floats). history: earlier frames
  // (arrays of 12 codes) for the repetition penalty; greedy: argmax and the fixed order 0..11 (tests).
  async sampleFrame(hlast, history = [], { greedy = false, firstLogits = null } = {}) {
    const { NQ, CB, schedule, temperature, penalty } = SCFG;
    if (this.hidden !== hlast) { this.headBG = this.headSteps(hlast); this.hidden = hlast; }
    this.run(this.headBG);
    const codes = new Array(NQ).fill(-1);
    const order = Array.from({ length: NQ }, (_, i) => i);
    if (!greedy) for (let i = NQ - 1; i > 0; i--) { const j = Math.floor(this.rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    let at = 0;
    for (const n of schedule) {
      const lg = await this.logits(codes);
      if (firstLogits && at === 0) firstLogits.push(lg);
      for (const i of order.slice(at, at + n)) {
        const s = lg.subarray(i * CB, (i + 1) * CB);
        codes[i] = greedy ? argmax(s) : sample(s, history.map((fr) => fr[i]), penalty, temperature, this.rand);
      }
      at += n;
    }
    return codes;
  }
  // the language model's input for a frame (the sum of its code embeddings), written into the frame buffer
  embedFrame(codes) {
    this.device.queue.writeBuffer(this.codes, 0, Uint32Array.from(codes));
    this.run([this.sumStep]);
    return this.frame;
  }
}
function argmax(s) { let b = 0; for (let i = 1; i < s.length; i++) if (s[i] > s[b]) b = i; return b; }
function sample(logits, hist, penalty, temperature, rand) {
  const s = Float64Array.from(logits);
  for (const c of new Set(hist)) s[c] = s[c] < 0 ? s[c] * penalty : s[c] / penalty;
  let m = -Infinity;
  for (let i = 0; i < s.length; i++) { s[i] /= temperature; if (s[i] > m) m = s[i]; }
  let z = 0;
  for (let i = 0; i < s.length; i++) { s[i] = Math.exp(s[i] - m); z += s[i]; }
  let r = rand() * z;
  for (let i = 0; i < s.length; i++) { r -= s[i]; if (r <= 0) return i; }
  return s.length - 1;
}

// ---- the codec decoder -----------------------------------------------------------------------------------------
// tiny f32 linear layer: y[t, n] = b[n] + sum_k W[n, k] x[t, k]
const LIN32 = GRID + /* wgsl */ `
struct P { N: u32, K: u32, n: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.n) { return; }
  let t = i / p.N; let r = i % p.N;
  var s = b[r];
  for (var k = 0u; k < p.K; k++) { s += w[r * p.K + k] * x[t * p.K + k]; }
  y[i] = s;
}`;
// each frame's 512-wide embedding repeated over its 6 STFT steps
const REPEAT6 = GRID + /* wgsl */ `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> e: array<f32>;
@group(0) @binding(2) var<storage, read_write> h: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.n) { return; }
  let t = i / 512u; let c = i % 512u;
  h[i] = e[(t / 6u) * 512u + c];
}`;
// short convolution of width 7 with the 6 previous steps of B*x carried in `prev` (oldest first); also writes the
// last 6 steps of B*x for the next chunk
const CONV7 = GRID + /* wgsl */ `
struct P { D: u32, M: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> t3: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> prev: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> next: array<f32>;
fn bx(s: i32, c: u32) -> f32 {
  if (s < 0) { return prev[u32(6 + s) * p.D + c]; }
  let r = u32(s) * 3u * p.D;
  return t3[r + c] * t3[r + 2u * p.D + c];
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.M * p.D) { return; }
  let t = i / p.D; let c = i % p.D;
  var acc = 0.0;
  for (var k = 0u; k < 7u; k++) { acc += w[c * 7u + k] * bx(i32(t) + i32(k) - 6, c); }
  y[i] = t3[t * 3u * p.D + p.D + c] * acc;
  if (t < 6u) { next[t * p.D + c] = bx(i32(p.M) - 6 + i32(t), c); }
}`;
// store this chunk's keys and values in the ring of past positions
const KV_STORE = GRID + /* wgsl */ `
struct P { M: u32, pos0: u32, ring: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;   // rows of 776: q 512 | k 128 | v 128 | gate 8
@group(0) @binding(2) var<storage, read_write> kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vc: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.M * 128u) { return; }
  let t = i / 128u; let c = i % 128u;
  let slot = (p.pos0 + t) % p.ring;
  kc[slot * 128u + c] = qkv[t * 776u + 512u + c];
  vc[slot * 128u + c] = qkv[t * 776u + 640u + c];
}`;
// causal attention over the last 30 positions (8 heads of 64 sharing 2 key/value heads), sigmoid gate per head
const ATTN_RING = /* wgsl */ `
struct P { pos0: u32, win: u32, ring: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> qs: array<f32, 64>;
var<workgroup> sc: array<f32, 32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x; let h = wg.y; let kvh = h / 4u;
  let pos = p.pos0 + t;
  qs[li] = qkv[t * 776u + h * 64u + li] * 0.125;
  workgroupBarrier();
  let lo = select(0u, pos + 1u - p.win, pos + 1u > p.win);
  let n = pos + 1u - lo;
  if (li < n) {
    let slot = (lo + li) % p.ring;
    var d = 0.0;
    for (var i = 0u; i < 64u; i++) { d += qs[i] * kc[slot * 128u + kvh * 64u + i]; }
    sc[li] = d;
  }
  workgroupBarrier();
  var m = -1.0e30;
  for (var j = 0u; j < n; j++) { m = max(m, sc[j]); }
  var l = 0.0; var acc = 0.0;
  for (var j = 0u; j < n; j++) { let e = exp(sc[j] - m); l += e; acc += e * vc[((lo + j) % p.ring) * 128u + kvh * 64u + li]; }
  let gate = 1.0 / (1.0 + exp(-qkv[t * 776u + 768u + h]));
  o[t * 512u + h * 64u + li] = acc / l * gate;
}`;
// sandwich norm + residual: h += w * x / rms(x), row by row
const ADD_NORMED = /* wgsl */ `
struct P { D: u32, eps: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> h: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let base = wg.x * p.D;
  var ss = 0.0;
  for (var i = li; i < p.D; i += 256u) { let v = x[base + i]; ss += v * v; }
  red[li] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { h[base + i] += w[i] * (x[base + i] * inv); }
}`;
// inverse STFT frames: irfft of exp(log magnitude) * e^(i phase), times the synthesis window
const ISTFT = /* wgsl */ `
struct P { M: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> spec: array<f32>;   // rows of 1282: log magnitude 641 | phase 641
@group(0) @binding(2) var<storage, read> cosT: array<f32>;   // cos(2 pi i / 1280)
@group(0) @binding(3) var<storage, read> win: array<f32>;
@group(0) @binding(4) var<storage, read_write> fr: array<f32>;
var<workgroup> re: array<f32, 641>;
var<workgroup> im: array<f32, 641>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wg.y;
  for (var k = li; k < 641u; k += 256u) {
    let a = exp(spec[m * 1282u + k]); let ph = spec[m * 1282u + 641u + k];
    re[k] = a * cos(ph); im[k] = a * sin(ph);
  }
  workgroupBarrier();
  let n = wg.x * 256u + li;
  if (n >= 1280u) { return; }
  var s = re[0] + select(re[640], -re[640], (n & 1u) == 1u);
  for (var k = 1u; k < 640u; k++) {
    let idx = (k * n) % 1280u;
    s += 2.0 * (re[k] * cosT[idx] - im[k] * cosT[(idx + 960u) % 1280u]);   // sin x = cos(x - pi/2)
  }
  fr[m * 1280u + n] = s / 1280.0 * win[n];
}`;

export class Codec {
  constructor(device, { quant = "q8", prequant = null } = {}) {
    this.device = device; this.quant = quant; this.prequant = prequant;
  }
  qm(N, Kd, label) {
    const G = (N * Kd) / (this.quant === "q4g64" ? 64 : 32);
    return { N, K: Kd, quant: this.quant, q: buf(this.device, G * 32, undefined, label + ".q"), s: buf(this.device, G * 4, undefined, label + ".s") };
  }
  allocWeights() {
    const { D, I, QD } = CCFG, v = (n) => buf(this.device, n * 4);
    this.w = { poW: v(QD * 48), poB: v(QD), fqW: v(D * QD), fqB: v(D), layers: [], finalNorm: v(D), lin: this.qm(1282, D, "c.lin"), linB: v(1282) };
    CCFG.types.forEach((type, i) => {
      const L = { type, opNorm: v(D), ffnNorm: v(D), postOp: v(D), postFfn: v(D), w13: this.qm(2 * I, D, `c${i}.w13`), w2: this.qm(D, I, `c${i}.w2`) };
      if (type === "conv") Object.assign(L, { convW: v(D * 7), inProj: this.qm(3 * D, D, `c${i}.in`), outProj: this.qm(D, D, `c${i}.out`) });
      else Object.assign(L, { qNorm: v(64), kNorm: v(64), qkvg: this.qm(776, D, `c${i}.qkvg`), oProj: this.qm(D, D, `c${i}.o`) });
      this.w.layers.push(L);
    });
  }
  weightList() { // the export order (tools/speech_export.py)
    const W = this.w, L = [["po.w", W.poW], ["po.b", W.poB], ["fq.w", W.fqW], ["fq.b", W.fqB]];
    W.layers.forEach((l, i) => {
      for (const k of ["opNorm", "ffnNorm", "postOp", "postFfn", "convW", "qNorm", "kNorm"]) if (l[k]) L.push([`${i}.${k}`, l[k]]);
      for (const k of ["inProj", "outProj", "qkvg", "oProj", "w13", "w2"]) if (l[k]) L.push([`${i}.${k}.q`, l[k].q], [`${i}.${k}.s`, l[k].s]);
    });
    L.push(["finalNorm", W.finalNorm], ["lin.q", W.lin.q], ["lin.s", W.lin.s], ["lin.b", W.linB]);
    return L;
  }
  async load(onProgress = () => {}) {
    this.allocWeights();
    await writeStream(this.device, this.weightList(), this.prequant.manifest, this.prequant.chunks(), onProgress);
    this.build();
  }
  build() {
    const d = this.device, { D, I, QD, ring, maxFrames } = CCFG, q = this.quant, b = (n) => buf(d, n * 4), M = maxFrames * 6;
    const kern = (kinds, src, label) => new Kernel(d, kinds, { main: src }, label);
    this.k = {
      gemm: kern(["u", "r", "r", "r", "r", "w"], K.gemvSource(q, MMV, { batched: true }), "c.gemm"),
      lin32: kern(["u", "r", "r", "r", "w"], LIN32, "c.lin32"), rep: kern(["u", "r", "w"], REPEAT6, "c.rep"),
      rms: kern(["u", "r", "r", "w"], RMS_ROWS, "c.rms"), addn: kern(["u", "r", "r", "w"], ADD_NORMED, "c.addn"),
      conv: kern(["u", "r", "r", "r", "w", "w"], CONV7, "c.conv"), qk: kern(["u", "r", "r", "w"], QK_ROPE, "c.qk"),
      kv: kern(["u", "r", "w", "w"], KV_STORE, "c.kv"), attn: kern(["u", "r", "r", "r", "w"], ATTN_RING, "c.attn"),
      silu: kern(["u", "r", "w"], SILU_ROWS, "c.silu"), bias: kern(["u", "r", "w"], BIAS_ACT, "c.bias"),
      istft: kern(["u", "r", "r", "r", "w"], ISTFT, "c.istft"),
    };
    this.st = buf(d, 32, U.STORAGE | U.COPY_DST);
    Object.assign(this, { lat: b(maxFrames * 48), q1: b(maxFrames * QD), emb: b(maxFrames * D), h: b(M * D), xn: b(M * D),
      t3: b(M * 3 * D), c: b(M * D), u: b(M * 2 * I), a: b(M * I), spec: b(M * 1282), fr: b(M * 1280),
      frRead: buf(d, M * 1280 * 4, U.MAP_READ | U.COPY_DST) });
    const cosT = new Float32Array(1280), win = new Float32Array(1280);
    for (let i = 0; i < 1280; i++) { cosT[i] = Math.cos((2 * Math.PI * i) / 1280); win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / 1280); }
    this.cosT = b(1280); this.win = b(1280);
    d.queue.writeBuffer(this.cosT, 0, cosT); d.queue.writeBuffer(this.win, 0, win);
    this.winF = win;
    for (const L of this.w.layers) {
      if (L.type === "conv") { L.prev = b(6 * D); L.next = b(6 * D); } else { L.kc = b(ring * 128); L.vc = b(ring * 128); }
    }
    this.plans = new Map();
    this.reset();
  }
  reset() { // a new utterance
    const d = this.device, enc = d.createCommandEncoder();
    for (const L of this.w.layers) for (const k of ["prev", "kc", "vc"]) if (L[k]) enc.clearBuffer(L[k]);
    d.queue.submit([enc.finish()]);
    this.pos = 0; this.steps = 0;
    this.acc = new Float32Array(0); this.env = new Float32Array(0); this.emitted = 480;
  }
  plan(F, pos0) { // F frames starting at STFT step pos0 (uniforms depend on pos0: rebuilt per call, bind groups cached by F)
    const d = this.device, k = this.k, { D, I, QD, ring, window } = CCFG, M = F * 6, W = this.w, steps = [];
    const ub = (w) => ubuf(d, w), grid = (n) => [Math.ceil(n / 256), 1, 1];
    const gemm = (m, x, y, rows, acc = 0) => steps.push({ p: k.gemm, wg: [Math.ceil(m.N / 8), 1, Math.ceil(rows / MMV)], bg: k.gemm.bind([ub([m.N, m.K, m.N, acc, rows, 0, 0, 0]), this.st, m.q, m.s, x, y]) });
    steps.push({ p: k.lin32, wg: grid(F * QD), bg: k.lin32.bind([ub([QD, 48, F * QD, 0]), W.poW, W.poB, this.lat, this.q1]) });
    steps.push({ p: k.lin32, wg: grid(F * D), bg: k.lin32.bind([ub([D, QD, F * D, 0]), W.fqW, W.fqB, this.q1, this.emb]) });
    steps.push({ p: k.rep, wg: grid(M * D), bg: k.rep.bind([ub([M * D, 0, 0, 0]), this.emb, this.h]) });
    const rmsU = ub([D, f(CCFG.eps), 0, 0]);
    const rms = (w, x, y) => steps.push({ p: k.rms, wg: [M, 1, 1], bg: k.rms.bind([rmsU, w, x, y]) });
    const addn = (w, x, h) => steps.push({ p: k.addn, wg: [M, 1, 1], bg: k.addn.bind([rmsU, w, x, h]) });
    for (const L of W.layers) {
      rms(L.opNorm, this.h, this.xn);
      if (L.type === "conv") {
        gemm(L.inProj, this.xn, this.t3, M);
        steps.push({ p: k.conv, wg: grid(M * D), bg: k.conv.bind([ub([D, M, 0, 0]), this.t3, L.convW, L.prev, this.c, L.next]) });
        steps.push({ copy: [L.next, L.prev, 6 * D * 4] });
        gemm(L.outProj, this.c, this.xn, M);
      } else {
        gemm(L.qkvg, this.xn, this.t3, M);
        steps.push({ p: k.qk, wg: [M, 10, 1], bg: k.qk.bind([ub([8, 2, 64, 776, pos0, f(CCFG.eps), f(CCFG.theta), 0]), L.qNorm, L.kNorm, this.t3]) });
        steps.push({ p: k.kv, wg: grid(M * 128), bg: k.kv.bind([ub([M, pos0, ring, 0]), this.t3, L.kc, L.vc]) });
        steps.push({ p: k.attn, wg: [M, 8, 1], bg: k.attn.bind([ub([pos0, window, ring, 0]), this.t3, L.kc, L.vc, this.c]) });
        gemm(L.oProj, this.c, this.xn, M);
      }
      addn(L.postOp, this.xn, this.h);
      rms(L.ffnNorm, this.h, this.xn);
      gemm(L.w13, this.xn, this.u, M);
      steps.push({ p: k.silu, wg: grid(M * I), bg: k.silu.bind([ub([I, M * I, 0, 0]), this.u, this.a]) });
      gemm(L.w2, this.a, this.xn, M);
      addn(L.postFfn, this.xn, this.h);
    }
    rms(W.finalNorm, this.h, this.xn);
    gemm(W.lin, this.xn, this.spec, M);
    steps.push({ p: k.bias, wg: grid(M * 1282), bg: k.bias.bind([ub([1282, M * 1282, 0, 0]), W.linB, this.spec]) });
    steps.push({ p: k.istft, wg: [5, M, 1], bg: k.istft.bind([ub([M, 0, 0, 0]), this.spec, this.cosT, this.win, this.fr]) });
    return steps;
  }
  fsq(frames) { // codes -> the 48 FSQ values of each frame (12 codebooks x 4 digits in [-1, 1])
    const { levels } = CCFG, basis = [1, 8, 56, 336], half = levels.map((l) => Math.floor(l / 2));
    const out = new Float32Array(frames.length * 48);
    frames.forEach((codes, f) => codes.forEach((c, q) => { for (let j = 0; j < 4; j++) out[f * 48 + q * 4 + j] = ((Math.floor(c / basis[j]) % levels[j]) - half[j]) / half[j]; }));
    return out;
  }
  // decode frames (arrays of 12 codes) of one utterance; returns the samples that are complete (24 kHz). The last
  // call passes final = true to flush the tail. Frames go through in chunks of at most maxFrames.
  async decode(frames, { final = false } = {}) {
    const outs = [];
    for (let s = 0; s < frames.length; s += CCFG.maxFrames) {
      const part = frames.slice(s, s + CCFG.maxFrames);
      outs.push(await this.decodeChunk(part, final && s + CCFG.maxFrames >= frames.length));
    }
    if (!frames.length && final) outs.push(this.flush(true));
    const n = outs.reduce((a, o) => a + o.length, 0), out = new Float32Array(n);
    let o = 0;
    for (const x of outs) { out.set(x, o); o += x.length; }
    return out;
  }
  async decodeChunk(frames, final) {
    const d = this.device, F = frames.length, M = F * 6, { hop, nfft } = CCFG;
    d.queue.writeBuffer(this.lat, 0, this.fsq(frames));
    const steps = this.plan(F, this.steps);
    const enc = d.createCommandEncoder();
    let pass = enc.beginComputePass();
    for (const s of steps) {
      if (s.copy) { pass.end(); enc.copyBufferToBuffer(s.copy[0], 0, s.copy[1], 0, s.copy[2]); pass = enc.beginComputePass(); continue; }
      pass.setPipeline(s.p.pipelines.main); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.wg);
    }
    pass.end();
    enc.copyBufferToBuffer(this.fr, 0, this.frRead, 0, M * nfft * 4);
    d.queue.submit([enc.finish()]);
    await this.frRead.mapAsync(GPUMapMode.READ);
    const fr = new Float32Array(this.frRead.getMappedRange().slice(0, M * nfft * 4));
    this.frRead.unmap();
    // overlap-add at the absolute step positions, with the window's squared envelope
    const start = this.steps * hop, end = (this.steps + M - 1) * hop + nfft;
    if (this.acc.length < end) { const a = new Float32Array(end), e = new Float32Array(end); a.set(this.acc); e.set(this.env); this.acc = a; this.env = e; }
    for (let m = 0; m < M; m++) {
      const o = start + m * hop;
      for (let n = 0; n < nfft; n++) { this.acc[o + n] += fr[m * nfft + n]; this.env[o + n] += this.winF[n] * this.winF[n]; }
    }
    this.steps += M;
    return this.flush(final);
  }
  flush(final) { // samples no later frame can reach (the first 480 and, at the end, the last 480 are trimmed)
    const trim = (CCFG.nfft - CCFG.hop) / 2, total = (this.steps - 1) * CCFG.hop + CCFG.nfft;
    const upto = final ? total - trim : this.steps * CCFG.hop;
    const out = new Float32Array(Math.max(0, upto - this.emitted));
    for (let i = 0; i < out.length; i++) { const j = this.emitted + i; out[i] = this.env[j] > 1e-11 ? this.acc[j] / this.env[j] : 0; }
    this.emitted = Math.max(this.emitted, upto);
    return out;
  }
}
