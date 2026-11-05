// A small IndexedDB key-value store: the page keeps the model's weights (as the decoded parts it streamed) and the
// tokenizer on the device, so later visits load without downloading, and an interrupted first load resumes where it
// stopped. Every call can fail (private browsing, storage blocked or full): callers catch and carry on without it.
const DB = "lfm3", OS = "kv";
let dbp = null;

function db() {
  if (!dbp) {
    dbp = new Promise((res, rej) => {
      const r = indexedDB.open(DB, 1);
      r.onupgradeneeded = () => r.result.createObjectStore(OS);
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error || new Error("storage unavailable"));
      r.onblocked = () => rej(new Error("storage blocked"));
    }).catch((e) => { dbp = null; throw e; });
  }
  return dbp;
}

async function tx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction(OS, mode);
    const r = fn(t.objectStore(OS));
    t.oncomplete = () => res(r ? r.result : undefined);
    t.onerror = t.onabort = () => rej(t.error || new Error("storage error"));
  });
}

export const store = {
  get: (k) => tx("readonly", (s) => s.get(k)),
  put: (k, v) => tx("readwrite", (s) => s.put(v, k)),
  del: (k) => tx("readwrite", (s) => s.delete(k)),
  keys: () => tx("readonly", (s) => s.getAllKeys()),
  async delWhere(pred) {
    for (const k of await store.keys()) if (pred(String(k))) await store.del(k);
  },
};

// ask the browser to keep this page's storage (no prompt on Safari or Chrome; it decides from engagement)
export async function persist() {
  try { return (await navigator.storage?.persist?.()) ?? false; } catch { return false; }
}
export async function room() { // free bytes the page may still store, or null if the browser does not say
  try {
    const e = await navigator.storage?.estimate?.();
    return e && e.quota ? e.quota - (e.usage || 0) : null;
  } catch { return null; }
}
