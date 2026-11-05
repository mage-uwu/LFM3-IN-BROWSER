// LFM3-Omni-2B text backbone on WebGPU: loads the vanilla weights straight from Hugging Face, quantizes them on
// the GPU, and runs prefill (chunks of up to MMAX tokens) and GPU-driven decoding.
import * as K from "./kernels.js";

export const CFG = {
  D: 2048, I: 7680, H: 16, KVH: 4, HD: 128, V: 128000, layers: 24, eps: 1e-5, window: 256, ring: 512,
  types: ["conv", "conv", "full", "conv", "sliding", "conv", "full", "conv", "sliding", "conv", "full", "conv",
    "sliding", "conv", "full", "conv", "sliding", "conv", "full", "conv", "sliding", "conv", "full", "conv"],
  stopIds: [124900, 125014], // <|im_end|>, <|text_end|>
  repo: "LiquidAI/LFM3-Omni-2B",
};
const QKVG = 3088;
const U = GPUBufferUsage;

// ---------------------------------------------------------------------------------------------------------------
// GPU helpers

export function buf(device, size, usage = U.STORAGE | U.COPY_DST | U.COPY_SRC, label) {
  return device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage, label });
}
export function ubuf(device, words, label) { // uniform buffer from u32/f32 values (array of [type, value])
  const b = device.createBuffer({ size: 16 * Math.ceil(words.length / 4), usage: U.UNIFORM | U.COPY_DST, label });
  const dv = new DataView(new ArrayBuffer(16 * Math.ceil(words.length / 4)));
  words.forEach((w, i) => (typeof w === "object" ? dv.setFloat32(4 * i, w.f, true) : dv.setUint32(4 * i, w, true)));
  device.queue.writeBuffer(b, 0, dv.buffer);
  return b;
}
export const f = (v) => ({ f: v });

export class Kernel {
  // kinds: "u" uniform, "r" read-only storage, "w" storage. One pipeline per source variant, one layout.
  constructor(device, kinds, sources, label) {
    this.device = device;
    this.layout = device.createBindGroupLayout({
      label,
      entries: kinds.map((k, i) => ({
        binding: i, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: k === "u" ? "uniform" : k === "r" ? "read-only-storage" : "storage" },
      })),
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.pipelines = {};
    for (const [name, code] of Object.entries(sources)) {
      const module = device.createShaderModule({ code, label: `${label}:${name}` });
      this.pipelines[name] = device.createComputePipeline({ layout: pl, compute: { module, entryPoint: "main" }, label: `${label}:${name}` });
    }
  }
  bind(buffers, label) {
    return this.device.createBindGroup({ layout: this.layout, label, entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  }
}

// Pre-quantized weights: a byte stream (async iterable of Uint8Array) holding every buffer of `list` ([key, GPUBuffer])
// back to back, in that order, checked against the export's manifest ([{key, bytes}]). Written as the bytes arrive.
export async function writeStream(d, list, man, chunks, onProgress = () => {}) {
  if (man.length !== list.length || man.some((m, i) => m.key !== list[i][0] || m.bytes > list[i][1].size)) {
    throw new Error("the stored weights do not match this engine's layout");
  }
  const total = man.reduce((a, m) => a + m.bytes, 0);
  let idx = 0, off = 0, got = 0, sinceSync = 0, carry = null;
  for await (let data of chunks) {
    if (carry) { const c = new Uint8Array(carry.length + data.length); c.set(carry); c.set(data, carry.length); data = c; carry = null; }
    let p = 0;
    while (p < data.length && idx < man.length) {
      const need = man[idx].bytes - off, take = Math.min(need, data.length - p);
      if (take < need) { // part of this buffer: write whole words, keep the rest for the next chunk
        const t4 = take & ~3;
        if (t4) d.queue.writeBuffer(list[idx][1], off, data, p, t4);
        off += t4; p += t4;
        carry = data.slice(p);
        p = data.length;
        break;
      }
      d.queue.writeBuffer(list[idx][1], off, data, p, take);
      p += take; idx++; off = 0;
    }
    got += data.length - (carry ? carry.length : 0);
    sinceSync += data.length;
    onProgress({ phase: "download", got, total });
    if (sinceSync > (64 << 20)) { await d.queue.onSubmittedWorkDone(); sinceSync = 0; }
  }
  if (idx !== man.length || (carry && carry.length)) throw new Error(`the stored weights ended early (${idx} of ${man.length} tensors)`);
  await d.queue.onSubmittedWorkDone();
}

// ---------------------------------------------------------------------------------------------------------------
// safetensors over HTTP

// a source is a URL (HTTP Range requests) or a File/Blob (a model.safetensors the user picked)
async function readBytes(src, start, end) {
  if (typeof src !== "string") return src.slice(start, end).arrayBuffer();
  const r = await fetch(src, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.arrayBuffer();
}

export async function readHeader(src) {
  const n = Number(new DataView(await readBytes(src, 0, 8)).getBigUint64(0, true));
  if (!(n > 0 && n < 1e8)) throw new Error("this is not a safetensors file");
  const header = JSON.parse(new TextDecoder().decode(await readBytes(src, 8, 8 + n)));
  delete header.__metadata__;
  for (const v of Object.values(header)) { v.start = 8 + n + v.data_offsets[0]; v.end = 8 + n + v.data_offsets[1]; }
  return header;
}

// Streams bytes [lo, hi) with a Range request, resuming after network errors; calls onChunk(Uint8Array, offset).
async function streamRange(url, lo, hi, onChunk, onProgress) {
  if (typeof url !== "string") { // a local file: read it in 8 MB slices
    for (let pos = lo; pos < hi;) {
      const end = Math.min(hi, pos + (8 << 20));
      await onChunk(new Uint8Array(await url.slice(pos, end).arrayBuffer()), pos);
      onProgress?.(end - lo, hi - lo);
      pos = end;
    }
    return;
  }
  let pos = lo;
  for (let attempt = 0; pos < hi; attempt++) {
    try {
      const r = await fetch(url, { headers: { Range: `bytes=${pos}-${hi - 1}` } });
      if (r.status !== 206 && r.status !== 200) throw new Error(`HTTP ${r.status}`);
      const reader = r.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        await onChunk(value, pos);
        pos += value.length;
        onProgress?.(pos - lo, hi - lo);
        attempt = 0;
      }
    } catch (e) {
      if (attempt >= 5) throw e;
      await new Promise((res) => setTimeout(res, 1000 * 2 ** attempt));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The model

export class LFM3 {
  constructor(device, opts = {}) {
    this.device = device;
    this.quant = opts.quant || "q4";          // "q4" (about 1.26 GB) or "q8" (about 1.9 GB; near-exact)
    // the tied embedding / output head turns the final state into logits: kept 8-bit (+130 MB) unless asked
    this.headQuant = opts.headQuant || "q8";
    // the short-conv operator multiplies three projections of its input (B*x, then C*conv), so 4-bit noise
    // compounds: measured, every conv layer added 3-9% residual error at 4-bit and attention layers ~0
    this.convQuant = opts.convQuant || "q8";
    this.mlpQuant = opts.mlpQuant || this.quant;    // feed-forward w1|w3 and w2
    this.attnQuant = opts.attnQuant || this.quant;  // q|k|v|gate and o
    // pre-quantized weights (GPTQ, 4-bit in groups of 64, every matrix): {manifest: [{key, bytes}], chunks()}
    this.prequant = opts.prequant || null;
    if (this.prequant) this.quant = this.headQuant = this.convQuant = this.mlpQuant = this.attnQuant = "q4g64";
    this.embedQuant = this.quant === "q8" ? "q8" : this.headQuant;
    this.ctx = opts.ctx || 8192;              // full-attention cache length
    this.maxChunks = Math.ceil(this.ctx / 256);
    // where the weights come from: a File (picked model.safetensors) or a URL (default: the official repository)
    this.url = opts.file || opts.url || `https://huggingface.co/${CFG.repo}/resolve/main/model.safetensors`;
    this.pos = 0;
    this.nout = 0;
  }

  // ---- weights ------------------------------------------------------------------------------------------------
  qbytes(N, Kd, quant) { // q-weight and scale bytes for an [N, K] matrix
    if (quant === "q4g64") { const G = (N * Kd) / 64; return [G * 32, G * 4]; }
    const G = (N * Kd) / 32;
    return quant === "q4" ? [G * 16, G * 4] : [G * 32, G * 4];
  }
  qmatrix(N, Kd, label, quant = this.quant) {
    const [a, b] = this.qbytes(N, Kd, quant);
    return { N, K: Kd, quant, q: buf(this.device, a, undefined, label + ".q"), s: buf(this.device, b, undefined, label + ".s") };
  }

  allocWeights() {
    const { D, I, layers, types } = CFG;
    const d = this.device;
    this.w = {
      embed: this.qmatrix(CFG.V, D, "embed", this.embedQuant),
      finalNorm: buf(d, D * 4),
      layers: [],
    };
    for (let i = 0; i < layers; i++) {
      const mq = this.quant === "q8" ? "q8" : this.mlpQuant;
      const L = { type: types[i], opNorm: buf(d, D * 4), ffnNorm: buf(d, D * 4), postOp: buf(d, D * 4), postFfn: buf(d, D * 4),
        w13: this.qmatrix(2 * I, D, `l${i}.w13`, mq), w2: this.qmatrix(D, I, `l${i}.w2`, mq) };
      if (L.type === "conv") {
        L.convW = buf(d, D * 7 * 4);
        const cq = this.quant === "q8" ? "q8" : this.convQuant;
        L.inProj = this.qmatrix(3 * D, D, `l${i}.in`, cq);
        L.outProj = this.qmatrix(D, D, `l${i}.out`, cq);
      } else {
        const aq = this.quant === "q8" ? "q8" : this.attnQuant;
        L.qkvg = this.qmatrix(QKVG, D, `l${i}.qkvg`, aq);
        L.oProj = this.qmatrix(D, D, `l${i}.o`, aq);
        L.qNorm = buf(d, 128 * 4);
        L.kNorm = buf(d, 128 * 4);
      }
      this.w.layers.push(L);
    }
  }

  // where each checkpoint tensor goes: {mat, rowOffset} for matrices, {vec, offset} for f32 vectors
  route(name) {
    const m = name.match(/^model\.language_model\.(.+)$/);
    if (!m) return null;
    const n = m[1];
    if (n === "embed_tokens.weight") return { mat: this.w.embed, row: 0 };
    if (n === "embedding_norm.weight") return { vec: this.w.finalNorm };
    const lm = n.match(/^layers\.(\d+)\.(.+)\.weight$/);
    if (!lm) return null;
    const L = this.w.layers[+lm[1]];
    const t = lm[2];
    const map = {
      "operator_norm": { vec: L.opNorm }, "ffn_norm": { vec: L.ffnNorm }, "post_operator_norm": { vec: L.postOp },
      "post_ffn_norm": { vec: L.postFfn }, "feed_forward.w1": { mat: L.w13, row: 0 },
      "feed_forward.w3": { mat: L.w13, row: CFG.I }, "feed_forward.w2": { mat: L.w2, row: 0 },
      "conv.conv": { vec: L.convW }, "conv.in_proj": { mat: L.inProj, row: 0 }, "conv.out_proj": { mat: L.outProj, row: 0 },
      "self_attn.q_proj": { mat: L.qkvg, row: 0 }, "self_attn.k_proj": { mat: L.qkvg, row: 2048 },
      "self_attn.v_proj": { mat: L.qkvg, row: 2560 }, "self_attn.gate_proj": { mat: L.qkvg, row: 3072 },
      "self_attn.o_proj": { mat: L.oProj, row: 0 }, "self_attn.q_norm": { vec: L.qNorm }, "self_attn.k_norm": { vec: L.kNorm },
    };
    return map[t] || null;
  }

  weightList() { // every weight buffer with a stable key, for the cache
    const L = [["embed.q", this.w.embed.q], ["embed.s", this.w.embed.s], ["finalNorm", this.w.finalNorm]];
    this.w.layers.forEach((l, i) => {
      for (const k of ["opNorm", "ffnNorm", "postOp", "postFfn", "convW", "qNorm", "kNorm"]) if (l[k]) L.push([`${i}.${k}`, l[k]]);
      for (const k of ["w13", "w2", "inProj", "outProj", "qkvg", "oProj"]) if (l[k]) L.push([`${i}.${k}.q`, l[k].q], [`${i}.${k}.s`, l[k].s]);
    });
    return L;
  }
  cacheName() {
    const pre = this.prequant ? `-${this.prequant.id || "gptq"}` : "";
    return `lfm3-webgpu-${this.quant}-${this.embedQuant}head-${this.convQuant}conv-${this.mlpQuant}mlp-${this.attnQuant}attn-v3${pre}`;
  }
  cacheKey(k) { return `https://lfm3-webgpu.cache/${this.quant}/${k}`; }

  static async cached(opts = "q8") { // is a saved copy in this browser? (a quant name, or the constructor's options)
    try {
      const m = new LFM3(null, typeof opts === "string" ? { quant: opts } : opts);
      const c = await caches.open(m.cacheName());
      return !!(await c.match(m.cacheKey("complete")));
    } catch { return false; }
  }

  async loadFromCache(onProgress) {
    if (!("caches" in globalThis)) return false;
    let cache;
    try {  // storage can be unavailable (a sandboxed frame, private browsing): then there is no saved copy
      cache = await caches.open(this.cacheName());
      if (!(await cache.match(this.cacheKey("complete")))) return false;
    } catch { return false; }
    const list = this.weightList();
    let n = 0;
    for (const [k, b] of list) {
      const r = await cache.match(this.cacheKey(k));
      if (!r) return false;
      const ab = await r.arrayBuffer();
      if (ab.byteLength !== b.size) return false;
      this.device.queue.writeBuffer(b, 0, ab);
      onProgress({ phase: "cache", got: ++n, total: list.length });
      if (n % 8 === 0) await this.device.queue.onSubmittedWorkDone();
    }
    await this.device.queue.onSubmittedWorkDone();
    return true;
  }

  async saveToCache(onProgress) {
    if (!("caches" in globalThis)) return false;
    try {
      await navigator.storage?.persist?.();
      const est = await navigator.storage?.estimate?.();
      const need = this.weightList().reduce((a, [, b]) => a + b.size, 0);
      if (est && est.quota - est.usage < need * 1.1) return false;
      const cache = await caches.open(this.cacheName());
      const list = this.weightList();
      let n = 0;
      for (const [k, b] of list) {
        const rb = buf(this.device, b.size, U.MAP_READ | U.COPY_DST);
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(b, 0, rb, 0, b.size);
        this.device.queue.submit([enc.finish()]);
        await rb.mapAsync(GPUMapMode.READ);
        const copy = rb.getMappedRange().slice(0);
        rb.destroy();
        await cache.put(this.cacheKey(k), new Response(copy));
        onProgress({ phase: "save", got: ++n, total: list.length });
      }
      await cache.put(this.cacheKey("complete"), new Response("1"));
      return true;
    } catch (e) {
      console.warn("weight cache not saved:", e);
      return false;
    }
  }

  weightQuants() { // the quant types of the matrices (one GEMV pipeline pair each)
    const L = this.w.layers;
    return [this.w.embed.quant, ...L.flatMap((l) => ["w13", "w2", "inProj", "outProj", "qkvg", "oProj"].filter((k) => l[k]).map((k) => l[k].quant))];
  }

  // pre-quantized weights, written in manifest order straight into the GPU buffers as the bytes arrive
  async loadPrequantized(onProgress) {
    await writeStream(this.device, this.weightList(), this.prequant.manifest, this.prequant.chunks(), onProgress);
  }

  // load: from the browser's cache if a previous visit saved the weights, else the pre-quantized weights if given,
  // else stream the checkpoint (Hugging Face, a URL or a picked file) and quantize it on the GPU
  async load(onProgress = () => {}, { cache = false, cacheOnly = false } = {}) {
    this.allocWeights();
    if (cache && (await this.loadFromCache(onProgress))) { this.build(); this.fromCache = true; return; }
    if (cacheOnly) throw new Error("No saved copy in this browser: choose the model file");
    if (this.prequant) {
      await this.loadPrequantized(onProgress);
      this.build();
      if (cache) this.cached = await this.saveToCache(onProgress);
      return;
    }
    await this.convertFromHub(onProgress);
    if (cache) this.cached = await this.saveToCache(onProgress);
  }

  async convertFromHub(onProgress) {
    const d = this.device;
    this.qk = new Kernel(d, ["u", "r", "w", "w"], { q4: K.quantizeSource("q4"), q8: K.quantizeSource("q8") }, "quantize");
    this.bk = new Kernel(d, ["u", "r", "w"], { main: K.BF16_TO_F32 }, "bf16");
    const header = await readHeader(this.url);
    const tensors = Object.entries(header)
      .map(([name, v]) => ({ name, ...v, dst: this.route(name) }))
      .filter((t) => t.dst)
      .sort((a, b) => a.start - b.start);
    const missing = tensors.length !== 1 + 1 + 24 * 4 + 24 * 3 + 13 * 3 + 11 * 7;
    if (missing) throw new Error(`this file does not look like LFM3-Omni-2B's model.safetensors (${tensors.length} text tensors)`);
    const lo = tensors[0].start, hi = tensors[tensors.length - 1].end;
    const STAGE = 8 << 20;
    const stage = buf(d, STAGE, U.STORAGE | U.COPY_DST);
    let ti = 0;
    let pend = new Uint8Array(STAGE + (1 << 16));  // bytes of the current tensor not yet converted
    let pendLen = 0, done = 0, sinceSync = 0;
    const flushRows = async (t, final) => {
      // convert whole rows (matrices) or the whole tensor (vectors) from `pend`
      if (t.dst.vec) {
        if (!final) return;
        d.queue.writeBuffer(stage, 0, pend.buffer, 0, Math.ceil(pendLen / 4) * 4);
        const n = pendLen / 2;
        const u = ubuf(d, [n, 0]);
        const enc = d.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.bk.pipelines.main);
        pass.setBindGroup(0, this.bk.bind([u, stage, t.dst.vec]));
        pass.dispatchWorkgroups(Math.ceil(n / 256));
        pass.end();
        d.queue.submit([enc.finish()]);
        pendLen = 0;
        return;
      }
      const Kd = t.shape[1], rowBytes = Kd * 2;
      const rows = Math.min(Math.floor(pendLen / rowBytes), Math.floor(STAGE / rowBytes));  // fits the stage
      if (rows === 0) return;
      const bytes = rows * rowBytes;
      d.queue.writeBuffer(stage, 0, pend.buffer, 0, bytes);
      const groups = rows * (Kd / 32);
      const u = ubuf(d, [rows, Kd, t.dst.row + t.rowsDone, 0]);
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.qk.pipelines[t.dst.mat.quant]);
      pass.setBindGroup(0, this.qk.bind([u, stage, t.dst.mat.q, t.dst.mat.s]));
      const wgs = Math.ceil(groups / 64);
      pass.dispatchWorkgroups(Math.min(wgs, 65535), Math.ceil(wgs / 65535));
      pass.end();
      d.queue.submit([enc.finish()]);
      t.rowsDone += rows;
      pend.copyWithin(0, bytes, pendLen);
      pendLen -= bytes;
      sinceSync += bytes;
      if (sinceSync > (64 << 20)) { await d.queue.onSubmittedWorkDone(); sinceSync = 0; }
    };
    for (const t of tensors) t.rowsDone = 0;
    await streamRange(this.url, lo, hi, async (chunk, off) => {
      let p = 0;
      while (p < chunk.length && ti < tensors.length) {
        const t = tensors[ti];
        const abs = off + p;
        if (abs < t.start) { p += Math.min(chunk.length - p, t.start - abs); continue; }  // gap (none expected)
        const take = Math.min(chunk.length - p, t.end - abs, pend.length - pendLen);
        pend.set(chunk.subarray(p, p + take), pendLen);
        pendLen += take;
        p += take;
        const complete = abs + take === t.end;
        if (complete || pendLen >= STAGE) {
          await flushRows(t, complete);
          while (t.dst.mat && pendLen >= (complete ? t.shape[1] * 2 : STAGE)) await flushRows(t, complete);
        }
        if (complete) {
          if (t.dst.mat && t.rowsDone !== t.shape[0]) throw new Error(`${t.name}: ${t.rowsDone} rows`);
          pendLen = 0;
          ti++;
          done++;
        }
      }
    }, (got, total) => onProgress({ phase: "download", got, total, tensors: done }));
    await d.queue.onSubmittedWorkDone();
    stage.destroy();
    this.build();
  }

  // Rows the prompt can take instead of token embeddings (image or audio features, [rows, D] f32): a prompt id
  // with the top bit set, 0x80000000 | r, reads row r of this buffer.
  setExternalEmbeddings(buffer) {
    if (!buffer) this.extDummy ||= buf(this.device, CFG.D * 4);
    this.embedBG = [0, 1].map((fromState) => this.k.embed.bind([this.st, this.w.embed.q, this.w.embed.s, this.ids, this.h,
      this.embedU[fromState], buffer || this.extDummy]));
  }

  // ---- runtime buffers and bind groups ------------------------------------------------------------------------
  build() {
    const d = this.device;
    const { D, I, layers } = CFG;
    const M = K.MMAX;
    this.k = {
      gemv: new Kernel(d, ["u", "r", "r", "r", "r", "w"], Object.fromEntries([...new Set(this.weightQuants())]
        .flatMap((q) => [[`${q}m1`, K.gemvSource(q, 1)], [`${q}m16`, K.gemvSource(q, M)]])), "gemv"),
      embed: new Kernel(d, ["r", "r", "r", "r", "w", "u", "r"], { main: K.embedSource(this.w.embed.quant) }, "embed"),
      rms: new Kernel(d, ["u", "r", "r", "r", "w"], { main: K.RMSNORM }, "rmsnorm"),
      pan: new Kernel(d, ["u", "r", "r", "r", "r", "w", "w"], { main: K.POST_ADD_NORM }, "post_add_norm"),
      silu: new Kernel(d, ["u", "r", "r", "w"], { main: K.SILU_MUL }, "silu_mul"),
      conv: new Kernel(d, ["r", "r", "r", "w", "w"], { main: K.CONV_MIX }, "conv_mix"),
      rope: new Kernel(d, ["u", "r", "r", "r", "r", "w", "w", "w"], { main: K.QK_ROPE_STORE }, "qk_rope_store"),
      attn: new Kernel(d, ["u", "r", "r", "r", "r", "w", "w"], { main: K.ATTN_PARTIAL }, "attn_partial"),
      comb: new Kernel(d, ["u", "r", "r", "r", "r", "w"], { main: K.ATTN_COMBINE }, "attn_combine"),
      am1: new Kernel(d, ["u", "r", "w"], { main: K.ARGMAX1 }, "argmax1"),
      am2: new Kernel(d, ["u", "w", "r", "w"], { main: K.ARGMAX2 }, "argmax2"),
    };
    const b = (n) => buf(d, n);
    this.st = buf(d, 32, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    this.st1 = buf(d, 32, U.STORAGE | U.COPY_DST);
    d.queue.writeBuffer(this.st1, 0, new Uint32Array([0, 1, 0, 0, 0, 0, 0, 0]));
    this.ids = b(M * 4);
    this.h = b(M * D * 4); this.xn = b(M * D * 4); this.o = b(M * D * 4);
    this.tA = b(M * 2 * I * 4); this.tB = b(M * I * 4);
    this.qbuf = b(M * 16 * 128 * 4);
    this.pml = b(M * 16 * this.maxChunks * 8);
    this.po = b(M * 16 * this.maxChunks * 128 * 4);
    this.hlast = b(D * 4);
    this.logits = b(CFG.V * 4);
    this.parts = b(Math.ceil(CFG.V / 1024) * 8);
    this.outIds = b(65536 * 4);
    this.readSt = buf(d, 32, U.MAP_READ | U.COPY_DST);
    const gemv = (mat, x, y, accumulate = 0, st = this.st) => {
      const u = ubuf(d, [mat.N, mat.K, mat.N, accumulate, 0, 0, 0, 0]);
      return { kind: "gemv", N: mat.N, quant: mat.quant, bg: this.k.gemv.bind([u, st, mat.q, mat.s, x, y]) };
    };
    const normU = ubuf(d, [D, f(CFG.eps), 0, 0]);
    this.embedU = [0, 1].map((fromState) => ubuf(d, [fromState, 0, 0, 0]));
    this.setExternalEmbeddings(null);
    this.firstNorm = this.k.rms.bind([normU, this.st, this.w.layers[0].opNorm, this.h, this.xn]);
    this.plan = [];
    for (let i = 0; i < layers; i++) {
      const L = this.w.layers[i];
      const next = i + 1 < layers ? this.w.layers[i + 1].opNorm : this.w.finalNorm;
      const steps = [];
      if (L.type === "conv") {
        L.cs = b(D * 6 * 4);
        steps.push(gemv(L.inProj, this.xn, this.tA));
        steps.push({ kind: "conv", bg: this.k.conv.bind([this.st, L.convW, this.tA, L.cs, this.tB]) });
        steps.push(gemv(L.outProj, this.tB, this.o));
      } else {
        const full = L.type === "full";
        L.slots = full ? this.ctx : CFG.ring;
        L.kc = b(4 * L.slots * 64 * 4); L.vc = b(4 * L.slots * 64 * 4);
        const ru = ubuf(d, [f(full ? 1e6 : 1e4), full ? 32 : 64, L.slots, full ? 0 : 1, f(CFG.eps), 0, 0, 0]);
        const win = full ? 0 : CFG.window;
        const au = ubuf(d, [L.slots, full ? 0 : 1, win, this.maxChunks]);
        const cu = ubuf(d, [this.maxChunks, win, 0, 0]);
        steps.push(gemv(L.qkvg, this.xn, this.tA));
        steps.push({ kind: "rope", bg: this.k.rope.bind([ru, this.st, this.tA, L.qNorm, L.kNorm, this.qbuf, L.kc, L.vc]) });
        steps.push({ kind: "attn", full, bg: this.k.attn.bind([au, this.st, this.qbuf, L.kc, L.vc, this.pml, this.po]) });
        steps.push({ kind: "comb", bg: this.k.comb.bind([cu, this.st, this.pml, this.po, this.tA, this.tB]) });
        steps.push(gemv(L.oProj, this.tB, this.o));
      }
      steps.push({ kind: "pan", bg: this.k.pan.bind([normU, this.st, L.postOp, L.ffnNorm, this.o, this.h, this.xn]) });
      steps.push(gemv(L.w13, this.xn, this.tA));
      steps.push({ kind: "silu", bg: this.k.silu.bind([ubuf(d, [I, 0, 0, 0]), this.st, this.tA, this.tB]) });
      steps.push(gemv(L.w2, this.tB, this.o));
      steps.push({ kind: "pan", bg: this.k.pan.bind([normU, this.st, L.postFfn, next, this.o, this.h, this.xn]) });
      this.plan.push(steps);
    }
    this.head = gemv(this.w.embed, this.hlast, this.logits, 0, this.st1);
    this.am1BG = this.k.am1.bind([ubuf(d, [CFG.V, 0, 0, 0]), this.logits, this.parts]);
    const np = Math.ceil(CFG.V / 1024), none = 0xffffffff;
    this.am2BG = [[0, ...CFG.stopIds.slice(0, 2)], [1, ...CFG.stopIds.slice(0, 2)], [1, none, none]]  // [2]: never stops (benchmarks)
      .map(([adv, s0, s1]) => this.k.am2.bind([ubuf(d, [np, adv, s0, s1]), this.st, this.parts, this.outIds]));
  }

  // encode one forward over M tokens starting at position pos (pos/M are also in this.st on the GPU).
  // debugBuf: copies the last token's residual stream after every layer (24 x D f32) for layer-by-layer checks.
  encodeForward(enc, M, pos, { fromState = false, debugBuf } = {}) {
    const k = this.k;
    const pipe = M === 1 ? "m1" : "m16";  // + the matrix's quant type
    let pass = enc.beginComputePass();
    pass.setPipeline(k.embed.pipelines.main);
    pass.setBindGroup(0, this.embedBG[fromState ? 1 : 0]);
    pass.dispatchWorkgroups(CFG.D / 256, M);
    pass.setPipeline(k.rms.pipelines.main);
    pass.setBindGroup(0, this.firstNorm);
    pass.dispatchWorkgroups(M);
    for (let i = 0; i < CFG.layers; i++) {
      for (const s of this.plan[i]) {
        switch (s.kind) {
          case "gemv": pass.setPipeline(k.gemv.pipelines[s.quant + pipe]); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(Math.ceil(s.N / 8)); break;
          case "conv": pass.setPipeline(k.conv.pipelines.main); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(CFG.D / 256); break;
          case "rope": pass.setPipeline(k.rope.pipelines.main); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(24, M); break;
          case "attn": {
            const last = pos + M - 1;
            pass.setPipeline(k.attn.pipelines.main); pass.setBindGroup(0, s.bg);
            // a sliding window (256 keys from each query's own start) is one chunk; full layers span the prefix
            pass.dispatchWorkgroups(16, M, s.full ? Math.min(this.maxChunks, Math.ceil((last + 1) / 256)) : 1);
            break;
          }
          case "comb": pass.setPipeline(k.comb.pipelines.main); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(16, M); break;
          case "pan": pass.setPipeline(k.pan.pipelines.main); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(M); break;
          case "silu": pass.setPipeline(k.silu.pipelines.main); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(Math.ceil(CFG.I / 256), M); break;
        }
      }
      if (debugBuf) {
        pass.end();
        enc.copyBufferToBuffer(this.h, (M - 1) * CFG.D * 4, debugBuf, i * CFG.D * 4, CFG.D * 4);
        pass = enc.beginComputePass();
      }
    }
    pass.end();
  }

  encodeHead(enc, M, advance, noStop = false) {
    // xn holds embedding_norm(h) for all M rows; the head runs on the last one
    enc.copyBufferToBuffer(this.xn, (M - 1) * CFG.D * 4, this.hlast, 0, CFG.D * 4);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.k.gemv.pipelines[this.head.quant + "m1"]); pass.setBindGroup(0, this.head.bg); pass.dispatchWorkgroups(CFG.V / 8);
    pass.setPipeline(this.k.am1.pipelines.main); pass.setBindGroup(0, this.am1BG); pass.dispatchWorkgroups(Math.ceil(CFG.V / 1024));
    pass.setPipeline(this.k.am2.pipelines.main); pass.setBindGroup(0, this.am2BG[noStop ? 2 : advance ? 1 : 0]); pass.dispatchWorkgroups(1);
    pass.end();
  }

  setState(pos, M, token = 0, nout = this.nout) {
    this.device.queue.writeBuffer(this.st, 0, new Uint32Array([pos, M, 0, token, nout, 0, 0, 0]));
  }

  // ---- public API ---------------------------------------------------------------------------------------------
  reset() {
    this.pos = 0;
    this.nout = 0;
    const d = this.device, enc = d.createCommandEncoder();
    for (const L of this.w.layers) {
      if (L.cs) enc.clearBuffer(L.cs);
    }
    d.queue.submit([enc.finish()]);
  }

  // Feed ids (any length) at the current position. With `sample`, the next token is picked on the GPU (greedy) and
  // becomes the decode input. `logitsOut` (optional Float32Array[V]) receives the last position's logits.
  // hidden: keep the last position's final hidden state in `hlast` without the head (the speech decoder reads it)
  async prefill(ids, { sample = true, logitsOut, debugHidden, hidden = false } = {}) {
    const d = this.device;
    for (let s = 0; s < ids.length; s += K.MMAX) {
      const chunk = ids.slice(s, s + K.MMAX);
      const M = chunk.length, last = s + M >= ids.length;
      d.queue.writeBuffer(this.ids, 0, Uint32Array.from(chunk));
      this.setState(this.pos, M);
      const enc = d.createCommandEncoder();
      const dbg = [];
      let dbuf;
      if (debugHidden && last) dbuf = buf(d, CFG.layers * CFG.D * 4);
      this.encodeForward(enc, M, this.pos, { debugBuf: dbuf });
      if (dbuf) {
        const rb = buf(d, CFG.layers * CFG.D * 4, U.MAP_READ | U.COPY_DST);
        enc.copyBufferToBuffer(dbuf, 0, rb, 0, CFG.layers * CFG.D * 4);
        dbg.push(rb);
      }
      if (last && (sample || logitsOut)) this.encodeHead(enc, M, sample);
      else if (last && hidden) enc.copyBufferToBuffer(this.xn, (M - 1) * CFG.D * 4, this.hlast, 0, CFG.D * 4);
      let lrb;
      if (last && logitsOut) { lrb = buf(d, CFG.V * 4, U.MAP_READ | U.COPY_DST); enc.copyBufferToBuffer(this.logits, 0, lrb, 0, CFG.V * 4); }
      d.queue.submit([enc.finish()]);
      if (lrb) { await lrb.mapAsync(GPUMapMode.READ); logitsOut.set(new Float32Array(lrb.getMappedRange())); lrb.destroy(); }
      for (const rb of dbg) { await rb.mapAsync(GPUMapMode.READ); debugHidden.push(new Float32Array(rb.getMappedRange().slice(0))); rb.destroy(); }
      this.pos += M;
    }
    if (sample) this.nout += 1;  // argmax2 recorded the sampled token in outIds
    await d.queue.onSubmittedWorkDone();
  }

  // GPU-driven greedy decoding: `steps` tokens per submit, no CPU round trip between them. Stops at a stop id
  // (the GPU stops updating state after it) or after maxNew tokens. onToken(id) streams; returns the ids.
  async generate(maxNew, { onToken, batch = 8, shouldStop, noStop = false } = {}) {
    const d = this.device;
    const out = [];
    // the token sampled by prefill is outIds[nout-1]
    const first = await this.readIds(this.nout - 1, 1);
    out.push(first[0]);
    onToken?.(first[0]);
    if (!noStop && CFG.stopIds.includes(first[0])) return out;
    while (out.length < maxNew) {
      const n = Math.min(batch, maxNew - out.length);
      const enc = d.createCommandEncoder();
      for (let s = 0; s < n; s++) {
        this.encodeForward(enc, 1, this.pos + s, { fromState: true });
        this.encodeHead(enc, 1, true, noStop);
      }
      enc.copyBufferToBuffer(this.st, 0, this.readSt, 0, 32);
      d.queue.submit([enc.finish()]);
      await this.readSt.mapAsync(GPUMapMode.READ);
      const st = new Uint32Array(this.readSt.getMappedRange().slice(0));
      this.readSt.unmap();
      const ids = await this.readIds(this.nout, st[4] - this.nout);
      this.nout = st[4];
      this.pos = st[0];
      let stop = false;
      for (const id of ids) {
        out.push(id);
        onToken?.(id);
        if (!noStop && CFG.stopIds.includes(id)) { stop = true; break; }
      }
      if (stop || st[5] === 1 || shouldStop?.()) break;
    }
    return out;
  }

  async readIds(start, n) {
    if (n <= 0) return [];
    const d = this.device;
    const rb = buf(d, n * 4, U.MAP_READ | U.COPY_DST);
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.outIds, start * 4, rb, 0, Math.ceil(n * 4 / 4) * 4);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const ids = Array.from(new Uint32Array(rb.getMappedRange()).slice(0, n));
    rb.destroy();
    return ids;
  }

  // Copies of the state (System 3: a curated experience prefilled once, every session starts from a copy)
  async snapshot() {
    const d = this.device, enc = d.createCommandEncoder();
    const snap = { pos: this.pos, layers: [] };
    for (const L of this.w.layers) {
      const s = {};
      if (L.cs) { s.cs = buf(d, L.cs.size); enc.copyBufferToBuffer(L.cs, 0, s.cs, 0, L.cs.size); }
      if (L.kc) {
        for (const key of ["kc", "vc"]) { s[key] = buf(d, L[key].size); enc.copyBufferToBuffer(L[key], 0, s[key], 0, L[key].size); }
      }
      snap.layers.push(s);
    }
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
    return snap;
  }
  restore(snap) {
    const d = this.device, enc = d.createCommandEncoder();
    snap.layers.forEach((s, i) => {
      const L = this.w.layers[i];
      for (const key of ["cs", "kc", "vc"]) if (s[key]) enc.copyBufferToBuffer(s[key], 0, L[key], 0, s[key].size);
    });
    d.queue.submit([enc.finish()]);
    this.pos = snap.pos;
  }
}

export async function initDevice() {
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser (iOS: Safari 26 or later)");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const want = (n) => Math.min(adapter.limits[n], 1 << 30);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: want("maxStorageBufferBindingSize"),
      maxBufferSize: want("maxBufferSize"),
      maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10),
    },
  });
  const info = adapter.info || {};
  return { adapter, device, info: { vendor: info.vendor, architecture: info.architecture, description: info.description,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize, maxBufferSize: device.limits.maxBufferSize } };
}
