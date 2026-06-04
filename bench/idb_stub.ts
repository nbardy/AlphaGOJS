// Minimal in-memory IndexedDB stub so the real browser GPUTrainer (which persists
// checkpoints via src/idb_storage.ts) can run headless under Bun. Rolled by hand
// (no fake-indexeddb dependency) — it implements only the tiny surface idb_storage
// touches: open → {onupgradeneeded, onsuccess}; db.transaction().objectStore() with
// put / delete / getAll. Results are returned via the async onsuccess callback,
// matching the real event-driven API.

interface FakeReq<T = any> {
  result?: T;
  onsuccess?: (e: { target: { result: T } }) => void;
  onerror?: (e: any) => void;
  onupgradeneeded?: (e: { target: { result: any } }) => void;
}

function fire<T>(req: FakeReq<T>, result?: T) {
  queueMicrotask(() => {
    if (result !== undefined) req.result = result;
    req.onsuccess?.({ target: { result: result as T } });
  });
}

export function installFakeIndexedDB() {
  const stores = new Map<string, Map<number, any>>();

  const db = {
    objectStoreNames: { contains: (n: string) => stores.has(n) },
    createObjectStore: (n: string) => {
      stores.set(n, new Map());
      return {};
    },
    transaction: (_names: string[] | string, _mode?: string) => ({
      objectStore: (n: string) => {
        if (!stores.has(n)) stores.set(n, new Map());
        const m = stores.get(n)!;
        return {
          put: (obj: any) => { const req: FakeReq = {}; m.set(obj.id, obj); fire(req); return req; },
          delete: (id: number) => { const req: FakeReq = {}; m.delete(id); fire(req); return req; },
          getAll: () => { const req: FakeReq = {}; fire(req, [...m.values()]); return req; },
        };
      },
    }),
  };

  (globalThis as any).indexedDB = {
    open: (_name: string, _version?: number) => {
      const req: FakeReq = {};
      queueMicrotask(() => {
        req.onupgradeneeded?.({ target: { result: db } }); // fresh DB → create stores
        req.onsuccess?.({ target: { result: db } });
      });
      return req;
    },
  };
}
