// WGSL compute kernels for the LFM3 text backbone. No WebGPU extensions are required (f16 values are packed into
// u32 with pack2x16float / unpack2x16float), so the same shaders run on iOS Safari, Android Chrome and desktops.
//
// Shared layout conventions
//   activations: f32, row-major [M, width] (M = tokens in this call: 1 when decoding, up to MMAX when prefilling)
//   q4 weights (default): a matrix W [N, K] is split into groups of 32 along K. Per group: 4 u32 holding 32 4-bit
//     codes (element 8*j+i is nibble i of word j) in `wq` (as vec4<u32>), and one u32 in `ws` holding
//     pack2x16float(scale, min). w = code * scale + min.
//   q4g64 weights: the same with groups of 64 (8 u32 of codes, 2 x vec4<u32>, per group); prepared offline (GPTQ)
//     and loaded as is: the format that fits the model in an artifact's 1 GiB asset store.
//   q8 weights: per group 8 u32 of signed bytes (element 4*j+i is byte i of word j) in `wq` (as 2 x vec4<u32>), and
//     one f32 scale in `ws` (bitcast). w = byte * scale.
//   KV cache: per layer [KVH, slots, 128] as packed f16 pairs (u32 [KVH, slots, 64]). Sliding layers use a ring of
//     512 slots (position % 512); full-attention layers one slot per position.
//   state (storage u32): [0] pos of the first token of this call, [1] M, [2] step, [3] token (decode input),
//     [4] decode steps written to `outIds`, [5] done: a stop token was sampled, so later steps (already queued on the
//     GPU) leave the caches, the conv states and the decode state untouched.

export const MMAX = 16; // tokens per prefill chunk

const STATE = /* wgsl */ `
struct State { pos: u32, M: u32, step: u32, token: u32, nout: u32, done: u32, pad1: u32, pad2: u32 };
`;

// y[m, n] (+)= sum_k W[n, k] x[m, k]   -- one workgroup computes 8 rows; 32 threads share each row's K range.
// batched: M rows come from the uniform (T tokens in all; workgroup z picks a block of MM of them) instead of the
// decode state, for encoders that run over a whole image or clip at once (engine/vision.js).
export function gemvSource(quant, mm, { batched = false } = {}) {
  const q4 = quant === "q4" || quant === "q4g64";
  const gs = quant === "q4g64" ? 64 : 32;   // weights per group (one f16 scale + min each)
  const groupDot = q4 ? /* wgsl */ `
      let sm = unpack2x16float(ws[gi]);
      for (var h = 0u; h < GS / 32u; h++) {
        let qw = wq[gi * (GS / 32u) + h];
        for (var j = 0u; j < 4u; j++) {
          let u = qw[j];
          let lo = vec4<f32>(f32(u & 15u), f32((u >> 4u) & 15u), f32((u >> 8u) & 15u), f32((u >> 12u) & 15u)) * sm.x + sm.y;
          let hi = vec4<f32>(f32((u >> 16u) & 15u), f32((u >> 20u) & 15u), f32((u >> 24u) & 15u), f32(u >> 28u)) * sm.x + sm.y;
          for (var m = 0u; m < MM; m++) {
            if (m < M) {
              let xb = xo + m * K4 + g * (GS / 4u) + h * 8u + j * 2u;
              acc[m] += dot(lo, x[xb]) + dot(hi, x[xb + 1u]);
            }
          }
        }
      }` : /* wgsl */ `
      let s = bitcast<f32>(ws[gi]);
      for (var h = 0u; h < 2u; h++) {
        let qw = wq[gi * 2u + h];
        for (var j = 0u; j < 4u; j++) {
          let u = qw[j];
          let v = vec4<f32>(f32(extractBits(i32(u), 0u, 8u)), f32(extractBits(i32(u), 8u, 8u)),
                            f32(extractBits(i32(u), 16u, 8u)), f32(extractBits(i32(u), 24u, 8u))) * s;
          for (var m = 0u; m < MM; m++) {
            if (m < M) {
              acc[m] += dot(v, x[xo + m * K4 + g * 8u + h * 4u + j]);
            }
          }
        }
      }`;
  const mDecl = batched
    ? `let mo = wg.z * MM;
  let M = min(p.T - mo, MM);
  let xo = mo * K4;`
    : `if (li == 0u) { wgM = min(st.M, MM); }
  let M = workgroupUniformLoad(&wgM);   // barriers below depend on it: it must be uniform
  let mo = 0u;
  let xo = 0u;`;
  return STATE + /* wgsl */ `
struct P { N: u32, K: u32, yStride: u32, accumulate: u32, T: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: State;
@group(0) @binding(2) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> ws: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
const MM = ${mm}u;
const GS = ${gs}u;
const ROWS = 8u;
const TPR = 32u;
var<workgroup> red: array<f32, 256>;
var<workgroup> wgM: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = li / TPR;
  let t = li % TPR;
  let row = (wg.y * 65535u + wg.x) * ROWS + r;
  let G = p.K / GS;
  let K4 = p.K / 4u;
  ${mDecl}
  var acc: array<f32, ${mm}>;
  if (row < p.N) {
    for (var g = t; g < G; g += TPR) {
      let gi = row * G + g;
      ${groupDot}
    }
  }
  for (var m = 0u; m < MM; m++) {
    if (m >= M) { break; }
    red[li] = acc[m];
    workgroupBarrier();
    for (var s = 16u; s > 0u; s >>= 1u) {
      if (t < s) { red[li] += red[li + s]; }
      workgroupBarrier();
    }
    if (t == 0u && row < p.N) {
      let o = (mo + m) * p.yStride + row;
      if (p.accumulate == 1u) { y[o] += red[li]; } else { y[o] = red[li]; }
    }
    workgroupBarrier();
  }
}`;
}

// h = embedding row of the token (dequantized), for each of the M tokens. ids: from `ids` (prefill) or st.token.
export function embedSource(quant) {
  const q4 = quant === "q4" || quant === "q4g64";
  const gs = quant === "q4g64" ? 64 : 32;
  const deq = q4 ? /* wgsl */ `
    let gi = row * G + g;
    let sm = unpack2x16float(ws[gi]);
    let u = wq[gi * (GS / 32u) + j / 32u][(j / 8u) % 4u];
    let code = (u >> (4u * (j % 8u))) & 15u;
    return f32(code) * sm.x + sm.y;` : /* wgsl */ `
    let gi = row * G + g;
    let s = bitcast<f32>(ws[gi]);
    let u = wq[gi * 2u + j / 16u][(j / 4u) % 4u];
    return f32(extractBits(i32(u), 8u * (j % 4u), 8u)) * s;`;
  return STATE + /* wgsl */ `
@group(0) @binding(0) var<storage, read> st: State;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> ws: array<u32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> h: array<f32>;
struct E { fromState: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(5) var<uniform> e: E;   // 1 when decoding: the token is st.token
@group(0) @binding(6) var<storage, read> ext: array<f32>;   // image / audio features: id 0x80000000 | row
const D = 2048u;
const GS = ${gs}u;
fn wAt(row: u32, k: u32) -> f32 {
  let G = D / GS;
  let g = k / GS;
  let j = k % GS;
  ${deq}
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wg.y;
  if (m >= st.M) { return; }
  var tok = ids[m];
  if (e.fromState == 1u) { tok = st.token; }
  let k = wg.x * 256u + li;
  if (tok >= 0x80000000u) { h[m * D + k] = ext[(tok & 0x7fffffffu) * D + k]; return; }
  h[m * D + k] = wAt(tok, k);
}`;
}

// y = w * x / rms(x) for each of M rows of width D (one workgroup per row).
export const RMSNORM = STATE + /* wgsl */ `
struct P { D: u32, eps: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: State;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> wgM: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wg.x;
  if (li == 0u) { wgM = st.M; }
  if (m >= workgroupUniformLoad(&wgM)) { return; }
  let base = m * p.D;
  var ss = 0.0;
  for (var i = li; i < p.D; i += 256u) { let v = x[base + i]; ss += v * v; }
  red[li] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { y[base + i] = w[i] * (x[base + i] * inv); }
}`;

// Sandwich norm + residual, then the next norm:  h += wPost * norm(o);  xn = wNext * norm(h)
export const POST_ADD_NORM = STATE + /* wgsl */ `
struct P { D: u32, eps: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: State;
@group(0) @binding(2) var<storage, read> wPost: array<f32>;
@group(0) @binding(3) var<storage, read> wNext: array<f32>;
@group(0) @binding(4) var<storage, read> o: array<f32>;
@group(0) @binding(5) var<storage, read_write> h: array<f32>;
@group(0) @binding(6) var<storage, read_write> xn: array<f32>;
var<workgroup> red: array<f32, 256>;
fn total(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let r = red[0];
  workgroupBarrier();
  return r;
}
var<workgroup> wgM: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wg.x;
  if (li == 0u) { wgM = st.M; }
  if (m >= workgroupUniformLoad(&wgM)) { return; }
  let base = m * p.D;
  var ss = 0.0;
  for (var i = li; i < p.D; i += 256u) { let v = o[base + i]; ss += v * v; }
  let inv = inverseSqrt(total(ss, li) / f32(p.D) + p.eps);
  var ss2 = 0.0;
  for (var i = li; i < p.D; i += 256u) {
    let v = h[base + i] + wPost[i] * (o[base + i] * inv);
    h[base + i] = v;
    ss2 += v * v;
  }
  let inv2 = inverseSqrt(total(ss2, li) / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { xn[base + i] = wNext[i] * (h[base + i] * inv2); }
}`;

// a = x[:, :I], b = x[:, I:]  ->  y = silu(a) * b     (w1 and w3 are one matrix, so one GEMV gives both)
export const SILU_MUL = STATE + /* wgsl */ `
struct P { I: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: State;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let m = gid.y;
  if (i >= p.I || m >= st.M) { return; }
  let a = x[m * 2u * p.I + i];
  let b = x[m * 2u * p.I + p.I + i];
  y[m * p.I + i] = a / (1.0 + exp(-a)) * b;
}`;

// Gated short convolution: bcx [M, 3D] = (B, C, x);  u = B*x;  out_t = sum_j w[j] u_{t-6+j};  y = C * out.
// state [D, 6]: the last 6 values of u per channel (oldest first).
export const CONV_MIX = STATE + /* wgsl */ `
const D = 2048u;
const W = 7u;
@group(0) @binding(0) var<storage, read> st: State;
@group(0) @binding(1) var<storage, read> w: array<f32>;          // [D, 7]
@group(0) @binding(2) var<storage, read> bcx: array<f32>;
@group(0) @binding(3) var<storage, read_write> cs: array<f32>;   // [D, 6]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= D || st.done == 1u) { return; }
  var s: array<f32, 6>;
  for (var j = 0u; j < 6u; j++) { s[j] = cs[c * 6u + j]; }
  var wk: array<f32, 7>;
  for (var j = 0u; j < W; j++) { wk[j] = w[c * W + j]; }
  for (var m = 0u; m < st.M; m++) {
    let base = m * 3u * D;
    let u = bcx[base + c] * bcx[base + 2u * D + c];
    var acc = wk[6] * u;
    for (var j = 0u; j < 6u; j++) { acc += wk[j] * s[j]; }
    for (var j = 0u; j < 5u; j++) { s[j] = s[j + 1u]; }
    s[5] = u;
    y[m * D + c] = bcx[base + D + c] * acc;
  }
  for (var j = 0u; j < 6u; j++) { cs[c * 6u + j] = s[j]; }
}`;

// After the fused q|k|v|gate GEMV (width 3088): per-head RMSNorm of q and k, RoPE (zero-tail partial RoPE on
// full-attention layers), write q to `qo` [M, 16, 128] and k, v into the cache slots of their positions.
// One workgroup (128 threads, one per head dim) per (token, unit): units 0-15 are q heads, 16-19 k, 20-23 v.
export const QK_ROPE_STORE = STATE + /* wgsl */ `
struct P { theta: f32, ropePairs: u32, slots: u32, ring: u32, eps: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: State;
@group(0) @binding(2) var<storage, read> qkvg: array<f32>;      // [M, 3088]
@group(0) @binding(3) var<storage, read> qn: array<f32>;        // [128]
@group(0) @binding(4) var<storage, read> kn: array<f32>;        // [128]
@group(0) @binding(5) var<storage, read_write> qo: array<f32>;  // [M, 16, 128]
@group(0) @binding(6) var<storage, read_write> kc: array<u32>;  // [4, slots, 64]
@group(0) @binding(7) var<storage, read_write> vc: array<u32>;
const HD = 128u;
const QKVG = 3088u;
var<workgroup> v: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
var<workgroup> wgM: u32;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  let unit = wg.x;
  let m = wg.y;
  if (d == 0u) { wgM = select(st.M, 0u, st.done == 1u); }
  if (m >= workgroupUniformLoad(&wgM)) { return; }
  let pos = st.pos + m;
  let slot = select(pos, pos % p.slots, p.ring == 1u);
  let base = m * QKVG;
  var val = 0.0;
  if (unit < 16u) { val = qkvg[base + unit * HD + d]; }
  else if (unit < 20u) { val = qkvg[base + 2048u + (unit - 16u) * HD + d]; }
  else { val = qkvg[base + 2560u + (unit - 20u) * HD + d]; }
  if (unit >= 20u) {   // v: stored as is
    v[d] = val;
    workgroupBarrier();
    if (d < 64u) {
      let kvh = unit - 20u;
      vc[(kvh * p.slots + slot) * 64u + d] = pack2x16float(vec2<f32>(v[2u * d], v[2u * d + 1u]));
    }
    return;
  }
  red[d] = val * val;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (d < s) { red[d] += red[d + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 128.0 + p.eps);
  var wn = 0.0;
  if (unit < 16u) { wn = qn[d]; } else { wn = kn[d]; }
  v[d] = wn * (val * inv);
  workgroupBarrier();
  // rotate_half RoPE: dims (i, i+64) rotate by pos * inv_freq[i]; inv_freq[i] = theta^(-2i/128) for i < ropePairs
  let i = d % 64u;
  var outv = v[d];
  if (i < p.ropePairs) {
    let ang = f32(pos) * pow(p.theta, -f32(2u * i) / 128.0);
    let c = cos(ang);
    let sn = sin(ang);
    if (d < 64u) { outv = v[d] * c - v[d + 64u] * sn; } else { outv = v[d] * c + v[d - 64u] * sn; }
  }
  workgroupBarrier();
  v[d] = outv;
  workgroupBarrier();
  if (unit < 16u) {
    qo[(m * 16u + unit) * HD + d] = v[d];
  } else if (d < 64u) {
    let kvh = unit - 16u;
    kc[(kvh * p.slots + slot) * 64u + d] = pack2x16float(vec2<f32>(v[2u * d], v[2u * d + 1u]));
  }
}`;

// Attention, split over chunks of 256 keys (flash-decoding): one workgroup per (q head, token, chunk) writes the
// chunk's max, sum and unnormalised output; ATTN_COMBINE merges the chunks and applies the head's sigmoid gate.
export const ATTN_PARTIAL = STATE + /* wgsl */ `
struct P { slots: u32, ring: u32, window: u32, nchunks: u32 };   // nchunks: stride (max chunks)
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: State;
@group(0) @binding(2) var<storage, read> q: array<f32>;         // [M, 16, 128]
@group(0) @binding(3) var<storage, read> kc: array<u32>;
@group(0) @binding(4) var<storage, read> vc: array<u32>;
@group(0) @binding(5) var<storage, read_write> pml: array<vec2<f32>>;   // [M, 16, nchunks]
@group(0) @binding(6) var<storage, read_write> po: array<f32>;          // [M, 16, nchunks, 128]
const HD = 128u;
var<workgroup> qs: array<f32, 128>;
var<workgroup> pr: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
var<workgroup> half_: array<f32, 128>;
var<workgroup> wgM: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x;
  let m = wg.y;
  let c = wg.z;
  if (li == 0u) { wgM = st.M; }
  if (m >= workgroupUniformLoad(&wgM)) { return; }
  let pos = st.pos + m;
  var lo = 0u;
  if (p.window > 0u && pos + 1u > p.window) { lo = pos + 1u - p.window; }
  let kp = lo + c * 256u + li;
  let kvh = h / 4u;
  let pi = (m * 16u + h) * p.nchunks + c;
  if (li < HD) { qs[li] = q[(m * 16u + h) * HD + li]; }
  workgroupBarrier();
  var sc = -3.0e38;
  let valid = kp <= pos;
  if (valid) {
    let slot = select(kp, kp % p.slots, p.ring == 1u);
    let kb = (kvh * p.slots + slot) * 64u;
    var acc = 0.0;
    for (var j = 0u; j < 64u; j++) {
      let kk = unpack2x16float(kc[kb + j]);
      acc += qs[2u * j] * kk.x + qs[2u * j + 1u] * kk.y;
    }
    sc = acc * 0.08838834764831845;   // 1/sqrt(128)
  }
  red[li] = sc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] = max(red[li], red[li + s]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  var e = 0.0;
  if (valid) { e = exp(sc - mx); }
  pr[li] = e;
  red[li] = e;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let l = red[0];
  // weighted values: threads 0-127 take keys 0-127, threads 128-255 keys 128-255, for head dim li % 128
  let d = li % HD;
  let k0 = (li / HD) * 128u;
  var o = 0.0;
  for (var i = 0u; i < 128u; i++) {
    let kk = k0 + i;
    let kpos = lo + c * 256u + kk;
    if (kpos <= pos && pr[kk] > 0.0) {
      let slot = select(kpos, kpos % p.slots, p.ring == 1u);
      let vv = unpack2x16float(vc[(kvh * p.slots + slot) * 64u + d / 2u]);
      o += pr[kk] * select(vv.x, vv.y, (d & 1u) == 1u);
    }
  }
  if (li >= HD) { half_[d] = o; }
  workgroupBarrier();
  if (li < HD) {
    po[pi * HD + d] = o + half_[d];
    if (li == 0u) { pml[pi] = vec2<f32>(mx, l); }
  }
}`;

export const ATTN_COMBINE = STATE + /* wgsl */ `
struct P { nchunks: u32, window: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: State;
@group(0) @binding(2) var<storage, read> pml: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> po: array<f32>;
@group(0) @binding(4) var<storage, read> qkvg: array<f32>;     // the gate logits sit at [m, 3072 + h]
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;  // [M, 2048]
const HD = 128u;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  let h = wg.x;
  let m = wg.y;
  if (m >= st.M) { return; }
  let base = (m * 16u + h) * p.nchunks;
  let pos = st.pos + m;
  var lo = 0u;
  if (p.window > 0u && pos + 1u > p.window) { lo = pos + 1u - p.window; }
  let nch = min((pos - lo) / 256u + 1u, p.nchunks);   // chunks this token's keys span (later ones are stale)
  var mx = -3.0e38;
  for (var c = 0u; c < nch; c++) { let ml = pml[base + c]; if (ml.y > 0.0) { mx = max(mx, ml.x); } }
  var l = 0.0;
  var o = 0.0;
  for (var c = 0u; c < nch; c++) {
    let ml = pml[base + c];
    if (ml.y > 0.0) {
      let f = exp(ml.x - mx);
      l += ml.y * f;
      o += po[(base + c) * HD + d] * f;
    }
  }
  let g = qkvg[m * 3088u + 3072u + h];
  outp[m * 2048u + h * HD + d] = (o / l) / (1.0 + exp(-g));
}`;

// Greedy pick over the vocabulary, two passes: (1) per workgroup arg-max of 1024 logits, (2) one workgroup over
// those. Pass 2 also advances the decode state: token -> st.token, outIds[st.nout] = token, pos += 1.
export const ARGMAX1 = /* wgsl */ `
struct P { V: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<vec2<f32>>;   // (value, bitcast index)
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  var best = -3.0e38;
  var idx = 0u;
  for (var k = 0u; k < 4u; k++) {
    let i = wg.x * 1024u + k * 256u + li;
    // text replies never pick the media placeholders or the speech-mode openers (as the model's text mode):
    // <|audio_start|> 125012, <|mixed_start|> 125015, <|video|> <|image|> <|audio|> <|audio_out|> 125029-125032
    let banned = i == 125012u || i == 125015u || (i >= 125029u && i <= 125032u);
    if (i < p.V && !banned) { let v = logits[i]; if (v > best) { best = v; idx = i; } }
  }
  bv[li] = best; bi[li] = idx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s && (bv[li + s] > bv[li] || (bv[li + s] == bv[li] && bi[li + s] < bi[li]))) { bv[li] = bv[li + s]; bi[li] = bi[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { part[wg.x] = vec2<f32>(bv[0], bitcast<f32>(bi[0])); }
}`;

export const ARGMAX2 = STATE + /* wgsl */ `
struct P { nparts: u32, advance: u32, stop0: u32, stop1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> st: State;
@group(0) @binding(2) var<storage, read> part: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> outIds: array<u32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var best = -3.0e38;
  var idx = 0u;
  for (var i = li; i < p.nparts; i += 256u) {
    let v = part[i];
    let ix = bitcast<u32>(v.y);
    if (v.x > best || (v.x == best && ix < idx)) { best = v.x; idx = ix; }
  }
  bv[li] = best; bi[li] = idx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s && (bv[li + s] > bv[li] || (bv[li + s] == bv[li] && bi[li + s] < bi[li]))) { bv[li] = bv[li + s]; bi[li] = bi[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u && st.done == 0u) {
    let tok = bi[0];
    if (p.advance == 1u) {
      st.pos = st.pos + st.M;   // the tokens just processed are now in the caches
      st.M = 1u;
      st.token = tok;
      outIds[st.nout] = tok;
      st.nout = st.nout + 1u;
      if (tok == p.stop0 || tok == p.stop1) { st.done = 1u; }
    } else {
      st.token = tok;
    }
  }
}`;

// Weight conversion on the GPU: bf16 rows (2 per u32) -> q4 / q8 groups at a destination row offset.
export function quantizeSource(quant) {
  if (quant === "q4") {
    return /* wgsl */ `
struct P { rows: u32, K: u32, rowOffset: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;          // bf16 pairs
@group(0) @binding(2) var<storage, read_write> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> ws: array<u32>;
fn bf(i: u32) -> f32 { let u = src[i / 2u]; return bitcast<f32>(select(u << 16u, u & 0xffff0000u, (i & 1u) == 1u)); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let G = p.K / 32u;
  let gidx = gid.y * 65535u * 64u + gid.x;
  if (gidx >= p.rows * G) { return; }
  let r = gidx / G;
  let g = gidx % G;
  let b = r * p.K + g * 32u;
  var vals: array<f32, 32>;
  var mn0 = 3.0e38; var mx0 = -3.0e38;
  for (var i = 0u; i < 32u; i++) { let v = bf(b + i); vals[i] = v; mn0 = min(mn0, v); mx0 = max(mx0, v); }
  // clip search: shrink the range around its midpoint and keep the lowest squared error (against the stored f16
  // scale and min, so what is measured is what is used)
  let c = 0.5 * (mn0 + mx0);
  var packed = 0u;
  var bestErr = 3.0e38;
  for (var t = 0u; t < 6u; t++) {
    let r = 1.0 - 0.04 * f32(t);
    let mn = c + (mn0 - c) * r;
    let mx = c + (mx0 - c) * r;
    var sc = (mx - mn) / 15.0;
    if (sc == 0.0) { sc = 1.0; }
    let pk = pack2x16float(vec2<f32>(sc, mn));
    let smt = unpack2x16float(pk);
    var err = 0.0;
    for (var i = 0u; i < 32u; i++) {
      let qv = clamp(round((vals[i] - smt.y) / smt.x), 0.0, 15.0);
      let e = qv * smt.x + smt.y - vals[i];
      err += e * e;
    }
    if (err < bestErr) { bestErr = err; packed = pk; }
  }
  let sm = unpack2x16float(packed);
  var words = vec4<u32>(0u);
  for (var i = 0u; i < 32u; i++) {
    let qv = u32(clamp(round((vals[i] - sm.y) / sm.x), 0.0, 15.0));
    words[i / 8u] = words[i / 8u] | (qv << (4u * (i % 8u)));
  }
  let o = (p.rowOffset + r) * G + g;
  wq[o] = words;
  ws[o] = packed;
}`;
  }
  return /* wgsl */ `
struct P { rows: u32, K: u32, rowOffset: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> ws: array<u32>;
fn bf(i: u32) -> f32 { let u = src[i / 2u]; return bitcast<f32>(select(u << 16u, u & 0xffff0000u, (i & 1u) == 1u)); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let G = p.K / 32u;
  let gidx = gid.y * 65535u * 64u + gid.x;
  if (gidx >= p.rows * G) { return; }
  let r = gidx / G;
  let g = gidx % G;
  let b = r * p.K + g * 32u;
  var am = 0.0;
  for (var i = 0u; i < 32u; i++) { am = max(am, abs(bf(b + i))); }
  var sc = am / 127.0;
  if (sc == 0.0) { sc = 1.0; }
  var words: array<u32, 8>;
  for (var i = 0u; i < 8u; i++) { words[i] = 0u; }
  for (var i = 0u; i < 32u; i++) {
    let qv = i32(clamp(round(bf(b + i) / sc), -127.0, 127.0));
    words[i / 4u] = words[i / 4u] | ((u32(qv) & 255u) << (8u * (i % 4u)));
  }
  let o = (p.rowOffset + r) * G + g;
  wq[o * 2u] = vec4<u32>(words[0], words[1], words[2], words[3]);
  wq[o * 2u + 1u] = vec4<u32>(words[4], words[5], words[6], words[7]);
  ws[o] = bitcast<u32>(sc);
}`;
}

// bf16 -> f32 (norm and conv weights)
export const BF16_TO_F32 = /* wgsl */ `
struct P { n: u32, offset: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let u = src[i / 2u];
  dst[p.offset + i] = bitcast<f32>(select(u << 16u, u & 0xffff0000u, (i & 1u) == 1u));
}`;
