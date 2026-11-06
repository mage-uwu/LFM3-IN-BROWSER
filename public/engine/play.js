// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Playing a spoken reply as it is made: 24 kHz chunks (Chat.speak's onAudio) scheduled back to back on one
// AudioContext, which must be created or resumed in a tap on iOS. When the device makes speech slower than it
// plays, a chunk arrives after the last one ended; playback then waits for a longer stretch (doubling, up to 4 s),
// so the reply pauses between phrases instead of stuttering.
//
//   const player = makePlayer();        // in a tap
//   player.begin(); ... onAudio: (x) => player.play(x) ...; player.end();
export function makePlayer() {
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ctx = new AC();
  let queue = [], queued = 0, playing = false, next = 0, ended = true, need = 0.3;
  const live = new Set();
  function schedule(x) {
    const b = ctx.createBuffer(1, x.length, 24000);
    b.getChannelData(0).set(x);
    const src = ctx.createBufferSource();
    src.buffer = b;
    src.connect(ctx.destination);
    src.start(next);
    next += b.duration;
    live.add(src);
    src.onended = () => live.delete(src);
  }
  function pump() {
    if (!playing) {
      if (!queue.length || (queued < need && !ended)) return;
      playing = true;
      next = ctx.currentTime + 0.05;
    }
    while (queue.length) {
      if (next < ctx.currentTime) { playing = false; need = Math.min(4, need * 2 + 0.25); pump(); return; }   // ran dry
      const x = queue.shift();
      queued -= x.length / 24000;
      schedule(x);
    }
  }
  const P = {
    ctx,
    begin() { P.stop(); ended = false; need = 0.3; },
    play(x) { if (!x.length) return; queue.push(x); queued += x.length / 24000; pump(); },
    end() { ended = true; pump(); },
    stop() {
      for (const s of live) { try { s.stop(); } catch { /* ended */ } }
      live.clear();
      queue = []; queued = 0; playing = false; next = 0; ended = true;
    },
    left() { return Math.max(0, next - ctx.currentTime) + queued; },   // seconds not yet heard
  };
  return P;
}
