// LFM3-Omni's vision tower (LFM3-VL) and its projector on WebGPU. An image is resized to multiples of 32 pixels,
// cut into 16x16 patches (each repeated over 2 time steps: 1536 values) in 2x2 merge-block order, and encoded by 14
// layers over the patch grid: gated depthwise 3x3 convolution, or bidirectional grouped-query attention with an
// axial 2D RoPE, each followed by a SwiGLU feed-forward. The projector merges every 2x2 block into one 2048-wide row:
// the language model reads those rows in place of its <|image|> placeholders (LFM3.setExternalEmbeddings).
import * as K from "./kernels.js";
import { buf, f, Kernel, ubuf, writeStream } from "./lfm3.js";

export const VCFG = {
  D: 1024, I: 2560, H: 16, KVH: 8, HD: 64, PD: 1536, P: 16, MERGE: 2, eps: 1e-5, theta: 1e6,
  PROJ: 4096, OUT: 2048, lnEps: 1e-6, imageToken: 125030,
  types: ["conv", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv"],
  minPixels: 65536, maxPixels: 200704,  // 256x256 .. 448x448 worth of pixels (the model allows up to 2M)
};
export const MMV = 32;  // tokens per block of the matrix kernel
const U = GPUBufferUsage;

// ---- preprocessing (the image processor's smart_resize, bicubic resampling, rescale/normalize, patchify) --------
const pyRound = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };
export function smartResize(height, width, { factor = 32, minPixels = VCFG.minPixels, maxPixels = VCFG.maxPixels } = {}) {
  let h = pyRound(height / factor) * factor, w = pyRound(width / factor) * factor;
  if (h * w > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    h = Math.max(factor, Math.floor(height / beta / factor) * factor);
    w = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    h = Math.ceil((height * beta) / factor) * factor;
    w = Math.ceil((width * beta) / factor) * factor;
  }
  return [h, w];
}

function cubic(x) { // Keys bicubic, a = -0.5 (as PIL)
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}
function coeffs(inSize, outSize) { // PIL's antialiased resampling taps
  const scale = inSize / outSize, fs = Math.max(scale, 1), support = 2 * fs, out = [];
  for (let x = 0; x < outSize; x++) {
    const center = (x + 0.5) * scale;
    const lo = Math.max(Math.trunc(center - support + 0.5), 0), hi = Math.min(Math.trunc(center + support + 0.5), inSize);
    const w = [];
    let sum = 0;
    for (let i = lo; i < hi; i++) { const v = cubic((i - center + 0.5) / fs); w.push(v); sum += v; }
    out.push({ lo, w: w.map((v) => (sum ? v / sum : 0)) });
  }
  return out;
}
// RGBA bytes (width x height) -> resized RGB floats in 0..255, rounded like an 8-bit image
export function resizeBicubic(rgba, width, height, outW, outH) {
  const cx = coeffs(width, outW), cy = coeffs(height, outH);
  const tmp = new Float32Array(height * outW * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < outW; x++) {
      const { lo, w } = cx[x];
      let r = 0, g = 0, b = 0;
      for (let i = 0; i < w.length; i++) { const o = (y * width + lo + i) * 4; r += w[i] * rgba[o]; g += w[i] * rgba[o + 1]; b += w[i] * rgba[o + 2]; }
      const t = (y * outW + x) * 3;
      tmp[t] = r; tmp[t + 1] = g; tmp[t + 2] = b;
    }
  }
  const out = new Float32Array(outH * outW * 3);
  for (let y = 0; y < outH; y++) {
    const { lo, w } = cy[y];
    for (let x = 0; x < outW; x++) {
      let r = 0, g = 0, b = 0;
      for (let i = 0; i < w.length; i++) { const t = ((lo + i) * outW + x) * 3; r += w[i] * tmp[t]; g += w[i] * tmp[t + 1]; b += w[i] * tmp[t + 2]; }
      const o = (y * outW + x) * 3;
      out[o] = Math.min(255, Math.max(0, Math.round(r)));
      out[o + 1] = Math.min(255, Math.max(0, Math.round(g)));
      out[o + 2] = Math.min(255, Math.max(0, Math.round(b)));
    }
  }
  return out;
}
// -> { patches: Float32Array [T, 1536], gh, gw, T, rows: T / 4 }  (T = number of 16x16 patches)
export function preprocess(rgba, width, height, opts = {}) {
  const [h, w] = smartResize(height, width, opts);
  const rgb = resizeBicubic(rgba, width, height, w, h);
  const { P, MERGE } = VCFG, gh = h / P, gw = w / P, T = gh * gw;
  const patches = new Float32Array(T * VCFG.PD);
  let t = 0;
  for (let bh = 0; bh < gh / MERGE; bh++) for (let bw = 0; bw < gw / MERGE; bw++) {
    for (let mh = 0; mh < MERGE; mh++) for (let mw = 0; mw < MERGE; mw++, t++) {
      const py = (bh * MERGE + mh) * P, px = (bw * MERGE + mw) * P, base = t * VCFG.PD;
      for (let c = 0; c < 3; c++) for (let ty = 0; ty < 2; ty++) for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) {
        const v = rgb[((py + y) * w + px + x) * 3 + c] / 255;   // rescale, then normalize with mean = std = 0.5
        patches[base + c * 512 + ty * 256 + y * P + x] = (v - 0.5) / 0.5;
      }
    }
  }
  return { patches, gh, gw, T, rows: T / (MERGE * MERGE), height: h, width: w };
}

// ---- kernels ---------------------------------------------------------------------------------------------------
// token t (2x2 merge-block order) <-> patch (row, col)
export const GRID = /* wgsl */ `
fn tokenRC(t: u32, gw: u32) -> vec2<u32> {
  let wb = gw / 2u; let blk = t / 4u; let r = t % 4u;
  return vec2<u32>((blk / wb) * 2u + r / 2u, (blk % wb) * 2u + r % 2u);
}
fn tokenAt(row: u32, col: u32, gw: u32) -> u32 { return ((row / 2u) * (gw / 2u) + col / 2u) * 4u + (row % 2u) * 2u + col % 2u; }
fn flat(wg: vec3<u32>, li: u32) -> u32 { return (wg.y * 65535u + wg.x) * 256u + li; }
`;
export const RMS_ROWS = /* wgsl */ `
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
  for (var i = li; i < p.D; i += 256u) { y[base + i] = w[i] * (x[base + i] * inv); }
}`;
export const LN_ROWS = /* wgsl */ `
struct P { D: u32, eps: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> mean: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let base = wg.x * p.D;
  var s1 = 0.0;
  for (var i = li; i < p.D; i += 256u) { s1 += x[base + i]; }
  red[li] = s1;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  if (li == 0u) { mean = red[0] / f32(p.D); }
  workgroupBarrier();
  let mu = mean;
  var s2 = 0.0;
  for (var i = li; i < p.D; i += 256u) { let d = x[base + i] - mu; s2 += d * d; }
  red[li] = s2;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { y[base + i] = (x[base + i] - mu) * inv * w[i] + b[i]; }
}`;
// conv operator: in_proj gives [B | C | x]; the depthwise 3x3 convolution runs over B*x on the patch grid
const GATE = GRID + /* wgsl */ `
struct P { D: u32, n: u32, gh: u32, gw: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> t3: array<f32>;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.n) { return; }
  let t = i / p.D; let c = i % p.D;
  g[i] = t3[t * 3u * p.D + c] * t3[t * 3u * p.D + 2u * p.D + c];
}`;
const DWCONV = GRID + /* wgsl */ `
struct P { D: u32, n: u32, gh: u32, gw: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> t3: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.n) { return; }
  let t = i / p.D; let c = i % p.D;
  let rc = tokenRC(t, p.gw);
  var acc = 0.0;
  for (var dy = 0u; dy < 3u; dy++) {
    for (var dx = 0u; dx < 3u; dx++) {
      let r = i32(rc.x) + i32(dy) - 1; let q = i32(rc.y) + i32(dx) - 1;
      if (r >= 0 && q >= 0 && r < i32(p.gh) && q < i32(p.gw)) {
        acc += w[c * 9u + dy * 3u + dx] * g[tokenAt(u32(r), u32(q), p.gw) * p.D + c];
      }
    }
  }
  y[i] = t3[t * 3u * p.D + p.D + c] * acc;
}`;
// attention: per-head RMSNorm of q and k, then the axial RoPE (rows on dims 0-15 and 32-47, columns on 16-31, 48-63)
const QK_NORM_ROPE = GRID + /* wgsl */ `
struct P { gw: u32, eps: f32, theta: f32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qn: array<f32>;
@group(0) @binding(2) var<storage, read> kn: array<f32>;
@group(0) @binding(3) var<storage, read_write> qkv: array<f32>;
var<workgroup> red: array<f32, 64>;
var<workgroup> xs: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x; let hd = wg.y;   // heads 0-15: q, 16-23: k
  let isK = hd >= 16u;
  var off = hd * 64u;
  if (isK) { off = 1024u + (hd - 16u) * 64u; }
  let base = t * 2048u + off;
  let v = qkv[base + li];
  red[li] = v * v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  var wv = qn[li];
  if (isK) { wv = kn[li]; }
  let xn = v * inverseSqrt(red[0] / 64.0 + p.eps) * wv;
  xs[li] = xn;
  workgroupBarrier();
  let rc = tokenRC(t, p.gw);
  let j = li % 32u;
  var pos = f32(rc.x);
  if (j >= 16u) { pos = f32(rc.y); }
  let ang = pos * pow(p.theta, -f32(2u * (j % 16u)) / 32.0);
  var rot = 0.0;
  if (li < 32u) { rot = -xs[li + 32u]; } else { rot = xs[li - 32u]; }
  qkv[base + li] = xn * cos(ang) + rot * sin(ang);
}`;
// bidirectional attention over all T patches; 16 query heads share 8 key/value heads; online softmax in tiles of 64
const ATTN = /* wgsl */ `
struct P { T: u32, scale: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
var<workgroup> qs: array<f32, 64>;
var<workgroup> ss: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x; let h = wg.y; let kvh = h / 2u;
  qs[li] = qkv[t * 2048u + h * 64u + li] * p.scale;
  workgroupBarrier();
  var m = -1.0e30; var l = 0.0; var acc = 0.0;
  for (var k0 = 0u; k0 < p.T; k0 += 64u) {
    let kk = k0 + li;
    var sc = -1.0e30;
    if (kk < p.T) {
      let kb = kk * 2048u + 1024u + kvh * 64u;
      var d = 0.0;
      for (var i = 0u; i < 64u; i++) { d += qs[i] * qkv[kb + i]; }
      sc = d;
    }
    ss[li] = sc;
    workgroupBarrier();
    var tm = -1.0e30;
    for (var i = 0u; i < 64u; i++) { tm = max(tm, ss[i]); }
    let mn = max(m, tm);
    let corr = exp(m - mn);
    l *= corr; acc *= corr;
    for (var i = 0u; i < 64u; i++) {
      let kv = k0 + i;
      if (kv < p.T) {
        let pr = exp(ss[i] - mn);
        l += pr;
        acc += pr * qkv[kv * 2048u + 1536u + kvh * 64u + li];
      }
    }
    m = mn;
    workgroupBarrier();
  }
  o[t * 1024u + h * 64u + li] = acc / l;
}`;
export const SILU_ROWS = GRID + /* wgsl */ `
struct P { I: u32, n: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.n) { return; }
  let t = i / p.I; let c = i % p.I;
  let x = u[t * 2u * p.I + c];
  a[i] = x / (1.0 + exp(-x)) * u[t * 2u * p.I + p.I + c];
}`;
// y += bias, then (act = 1) exact GELU: x * (1 + erf(x / sqrt 2)) / 2 (erf to 1.5e-7, Abramowitz & Stegun 7.1.26)
export const BIAS_ACT = GRID + /* wgsl */ `
struct P { N: u32, n: u32, act: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn erf(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let r = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(r, -r, x < 0.0);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = flat(wg, li);
  if (i >= p.n) { return; }
  var v = y[i] + b[i % p.N];
  if (p.act == 1u) { v = 0.5 * v * (1.0 + erf(v * 0.70710678118654752)); }
  y[i] = v;
}`;

// ---- the model -------------------------------------------------------------------------------------------------
export class Vision {
  constructor(device, { quant = "q8", maxTokens = VCFG.maxPixels / 256, prequant = null } = {}) {
    this.device = device;
    this.quant = quant;
    this.maxT = Math.ceil(maxTokens / 4) * 4;
    this.prequant = prequant;   // {manifest: [{key, bytes}], chunks()} as for the language model
    this.plans = new Map();
  }
  qbytes(N, Kd) {
    if (this.quant === "q4g64") { const G = (N * Kd) / 64; return [G * 32, G * 4]; }
    const G = (N * Kd) / 32;
    return [G * 32, G * 4];   // q8
  }
  qm(N, Kd, label) {
    const [a, b] = this.qbytes(N, Kd);
    return { N, K: Kd, quant: this.quant, q: buf(this.device, a, undefined, label + ".q"), s: buf(this.device, b, undefined, label + ".s") };
  }
  allocWeights() {
    const { D, I, PD, PROJ, OUT } = VCFG, d = this.device, v = (n) => buf(d, n * 4);
    this.w = { patch: this.qm(D, PD, "v.patch"), embNorm: v(D), layers: [] };
    VCFG.types.forEach((type, i) => {
      const L = { type, opNorm: v(D), ffnNorm: v(D), w13: this.qm(2 * I, D, `v${i}.w13`), w2: this.qm(D, I, `v${i}.w2`) };
      if (type === "conv") Object.assign(L, { convW: v(D * 9), inProj: this.qm(3 * D, D, `v${i}.in`), outProj: this.qm(D, D, `v${i}.out`) });
      else Object.assign(L, { qNorm: v(64), kNorm: v(64), qkv: this.qm(2 * D, D, `v${i}.qkv`), oProj: this.qm(D, D, `v${i}.o`) });
      this.w.layers.push(L);
    });
    this.w.proj = { lnW: v(D), lnB: v(D), l1: this.qm(PROJ, 4 * D, "v.l1"), l1b: v(PROJ), l2: this.qm(OUT, PROJ, "v.l2"), l2b: v(OUT) };
  }
  weightList() { // the export order (tools/vision_export.py writes the same keys)
    const W = this.w, L = [["patch.q", W.patch.q], ["patch.s", W.patch.s], ["embNorm", W.embNorm]];
    W.layers.forEach((l, i) => {
      for (const k of ["opNorm", "ffnNorm", "convW", "qNorm", "kNorm"]) if (l[k]) L.push([`${i}.${k}`, l[k]]);
      for (const k of ["inProj", "outProj", "qkv", "oProj", "w13", "w2"]) if (l[k]) L.push([`${i}.${k}.q`, l[k].q], [`${i}.${k}.s`, l[k].s]);
    });
    const p = W.proj;
    L.push(["ln.w", p.lnW], ["ln.b", p.lnB], ["l1.q", p.l1.q], ["l1.s", p.l1.s], ["l1.b", p.l1b], ["l2.q", p.l2.q], ["l2.s", p.l2.s], ["l2.b", p.l2b]);
    return L;
  }
  async load(onProgress = () => {}) {
    this.allocWeights();
    await writeStream(this.device, this.weightList(), this.prequant.manifest, this.prequant.chunks(), onProgress);
    this.build();
  }

  build() {
    const d = this.device, { D, I, OUT, PROJ } = VCFG, T = this.maxT;
    this.k = {
      gemm: new Kernel(d, ["u", "r", "r", "r", "r", "w"], { main: K.gemvSource(this.quant, MMV, { batched: true }) }, "v.gemm"),
      rms: new Kernel(d, ["u", "r", "r", "w"], { main: RMS_ROWS }, "v.rms"),
      ln: new Kernel(d, ["u", "r", "r", "r", "w"], { main: LN_ROWS }, "v.ln"),
      gate: new Kernel(d, ["u", "r", "w"], { main: GATE }, "v.gate"),
      dw: new Kernel(d, ["u", "r", "r", "r", "w"], { main: DWCONV }, "v.dwconv"),
      qk: new Kernel(d, ["u", "r", "r", "w"], { main: QK_NORM_ROPE }, "v.qk"),
      attn: new Kernel(d, ["u", "r", "w"], { main: ATTN }, "v.attn"),
      silu: new Kernel(d, ["u", "r", "w"], { main: SILU_ROWS }, "v.silu"),
      bias: new Kernel(d, ["u", "r", "w"], { main: BIAS_ACT }, "v.bias"),
    };
    const b = (n) => buf(d, n * 4);
    this.st = buf(d, 32, U.STORAGE | U.COPY_DST);  // the matrix kernel's (unused) decode-state binding
    Object.assign(this, { xin: b(T * VCFG.PD), h: b(T * D), xn: b(T * D), t3: b(T * 3 * D), g: b(T * D), c: b(T * D),
      u: b(T * 2 * I), a: b(T * I), ln: b(T * D), p1: b((T / 4) * PROJ), emb: b((T / 4) * OUT) });
  }

  // the dispatch plan for a T-patch image on a gh x gw grid (cached per grid)
  plan(T, gh, gw) {
    const key = `${gh}x${gw}`;
    if (this.plans.has(key)) return this.plans.get(key);
    const d = this.device, k = this.k, { D, I, PROJ, OUT } = VCFG, steps = [];
    const grid = (n) => { const w = Math.ceil(n / 256); return [Math.min(w, 65535), Math.ceil(w / 65535)]; };
    const gemm = (m, x, y, rows, acc = 0) => steps.push({ p: k.gemm, wg: [Math.ceil(m.N / 8), 1, Math.ceil(rows / MMV)],
      bg: k.gemm.bind([ubuf(d, [m.N, m.K, m.N, acc, rows, 0, 0, 0]), this.st, m.q, m.s, x, y]) });
    const rmsU = ubuf(d, [D, f(VCFG.eps), 0, 0]);
    const rms = (w, x, y) => steps.push({ p: k.rms, wg: [T, 1, 1], bg: k.rms.bind([rmsU, w, x, y]) });
    const ew = (kern, n, bufs, u) => steps.push({ p: kern, wg: [...grid(n), 1], bg: kern.bind([ubuf(d, u), ...bufs]) });
    gemm(this.w.patch, this.xin, this.xn, T);
    rms(this.w.embNorm, this.xn, this.h);
    for (const L of this.w.layers) {
      rms(L.opNorm, this.h, this.xn);
      if (L.type === "conv") {
        gemm(L.inProj, this.xn, this.t3, T);
        ew(k.gate, T * D, [this.t3, this.g], [D, T * D, gh, gw]);
        ew(k.dw, T * D, [this.t3, this.g, L.convW, this.c], [D, T * D, gh, gw]);
        gemm(L.outProj, this.c, this.h, T, 1);
      } else {
        gemm(L.qkv, this.xn, this.t3, T);
        steps.push({ p: k.qk, wg: [T, 24, 1], bg: k.qk.bind([ubuf(d, [gw, f(VCFG.eps), f(VCFG.theta), 0]), L.qNorm, L.kNorm, this.t3]) });
        steps.push({ p: k.attn, wg: [T, 16, 1], bg: k.attn.bind([ubuf(d, [T, f(1 / 8), 0, 0]), this.t3, this.c]) });
        gemm(L.oProj, this.c, this.h, T, 1);
      }
      rms(L.ffnNorm, this.h, this.xn);
      gemm(L.w13, this.xn, this.u, T);
      ew(k.silu, T * I, [this.u, this.a], [I, T * I, 0, 0]);
      gemm(L.w2, this.a, this.h, T, 1);
      steps.push({ layerEnd: true });
    }
    const P = this.w.proj, R = T / 4;
    steps.push({ p: k.ln, wg: [T, 1, 1], bg: k.ln.bind([ubuf(d, [D, f(VCFG.lnEps), 0, 0]), P.lnW, P.lnB, this.h, this.ln]) });
    gemm(P.l1, this.ln, this.p1, R);
    ew(k.bias, R * PROJ, [P.l1b, this.p1], [PROJ, R * PROJ, 1, 0]);
    gemm(P.l2, this.p1, this.emb, R);
    ew(k.bias, R * OUT, [P.l2b, this.emb], [OUT, R * OUT, 0, 0]);
    this.plans.set(key, steps);
    return steps;
  }

  // patches: Float32Array [T, 1536] (preprocess). Returns the GPU buffer of [T/4, 2048] rows for the language model.
  // debug: also read back the patch features after every layer ([layers, T, 1024]) and the final ones.
  async encode(patches, gh, gw, { debug = false } = {}) {
    const d = this.device, T = gh * gw, { D } = VCFG;
    if (T > this.maxT) throw new Error(`image too large: ${T} patches (at most ${this.maxT})`);
    d.queue.writeBuffer(this.xin, 0, patches);
    const steps = this.plan(T, gh, gw);
    let dbg = null, li = 0;
    if (debug) dbg = buf(d, VCFG.types.length * T * D * 4, U.COPY_DST | U.COPY_SRC);
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
    const out = { emb: this.emb, rows: T / 4 };
    if (dbg) {
      out.layers = new Float32Array(await readBack(d, dbg, VCFG.types.length * T * D * 4));
      out.emb32 = new Float32Array(await readBack(d, this.emb, (T / 4) * VCFG.OUT * 4));
      dbg.destroy();
    }
    return out;
  }
}

export async function readBack(d, src, size) {
  const rb = buf(d, size, U.MAP_READ | U.COPY_DST);
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, size);
  d.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const copy = rb.getMappedRange().slice(0);
  rb.destroy();
  return copy;
}
