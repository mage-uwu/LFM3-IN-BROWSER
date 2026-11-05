// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Byte containers: the model's weights travel as lossless 8-bit grayscale PNGs (tools/png_chunks.py), which any
// static host or CDN serves (some, like a claude.ai artifact's asset store, take images but not binary files).
// Decoding is the image data inflated with the browser's own DecompressionStream, minus the one filter byte (always
// 0) at the start of every row.
const SIG = [137, 80, 78, 71, 13, 10, 26, 10];

export async function pngPayload(input, nbytes) {
  const b = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (!SIG.every((v, i) => b[i] === v)) throw new Error("weights: not a PNG container");
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let p = 8, width = 0, height = 0;
  const idat = [];
  while (p + 8 <= b.length) {
    const len = dv.getUint32(p);
    const type = String.fromCharCode(b[p + 4], b[p + 5], b[p + 6], b[p + 7]);
    if (type === "IHDR") {
      width = dv.getUint32(p + 8);
      height = dv.getUint32(p + 12);
      if (b[p + 16] !== 8 || b[p + 17] !== 0 || b[p + 20] !== 0) throw new Error("weights: unexpected PNG format");
    } else if (type === "IDAT") idat.push(b.subarray(p + 8, p + 8 + len));
    else if (type === "IEND") break;
    p += 12 + len;
  }
  const stream = new Blob(idat).stream().pipeThrough(new DecompressionStream("deflate"));
  const raw = new Uint8Array(await new Response(stream).arrayBuffer());
  const stride = width + 1;
  if (raw.length !== height * stride) throw new Error("weights: PNG data has the wrong size");
  const n = nbytes ?? width * height;
  if (n > width * height) throw new Error("weights: PNG holds fewer bytes than expected");
  const out = new Uint8Array(n);
  for (let r = 0, o = 0; o < n; r++, o += width) {
    if (raw[r * stride] !== 0) throw new Error("weights: unsupported PNG row filter");
    out.set(raw.subarray(r * stride + 1, r * stride + 1 + Math.min(width, n - o)), o);
  }
  return out;
}

// the weights as an async sequence of byte arrays, two parts ahead of the GPU upload; loadOne(chunk, index) gives a
// part's bytes (fetched and decoded, or read back from the device)
export async function* prefetched(chunks, loadOne, { ahead = 2 } = {}) {
  const pending = [];
  let next = 0;
  const start = () => {
    const i = next++;
    const pr = loadOne(chunks[i], i);
    pr.catch(() => {});  // surfaced when awaited below
    pending.push(pr);
  };
  while (next < chunks.length && pending.length < ahead) start();
  while (pending.length) {
    const data = await pending.shift();
    if (next < chunks.length) start();
    yield data;
  }
}

// PNG containers fetched with fetchBytes(chunk) and decoded
export const pngChunks = (chunks, fetchBytes, opts) =>
  prefetched(chunks, async (c) => pngPayload(await fetchBytes(c), c.bytes), opts);
