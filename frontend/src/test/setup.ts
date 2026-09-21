/**
 * Test setup: route relative `/api` fetches to a live SentinAI backend
 * (SENTINAI_API_URL, default http://127.0.0.1:8000). Tests skip when it is unreachable.
 */
import React from "react";
import { vi } from "vitest";

export const API_BASE = (import.meta as unknown as { env: Record<string, string | undefined> }).env.SENTINAI_API_URL || "http://127.0.0.1:8000";

const realFetch = globalThis.fetch.bind(globalThis);

// jsdom's FormData / File are not understood by Node's native fetch (the request never
// serialises), so multipart bodies are encoded by hand here. Browsers do this natively.
function readBlob(blob: Blob): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(new Uint8Array(fr.result as ArrayBuffer));
    fr.onerror = () => reject(fr.error);
    fr.readAsArrayBuffer(blob);
  });
}

async function encodeMultipart(form: FormData): Promise<{ body: Uint8Array; contentType: string }> {
  const boundary = `----sentinai${Math.random().toString(16).slice(2)}`;
  const enc = new TextEncoder();
  const chunks: Uint8Array[] = [];
  for (const [name, value] of form.entries()) {
    if (typeof value === "string") {
      chunks.push(enc.encode(`--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}\r\n`));
    } else {
      const file = value as File;
      chunks.push(enc.encode(`--${boundary}\r\nContent-Disposition: form-data; name="${name}"; filename="${file.name}"\r\nContent-Type: ${file.type || "application/octet-stream"}\r\n\r\n`));
      chunks.push(await readBlob(file));
      chunks.push(enc.encode("\r\n"));
    }
  }
  chunks.push(enc.encode(`--${boundary}--\r\n`));
  const body = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let off = 0;
  for (const c of chunks) {
    body.set(c, off);
    off += c.length;
  }
  return { body, contentType: `multipart/form-data; boundary=${boundary}` };
}

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === "string" && input.startsWith("/") ? `${API_BASE}${input}` : input;
  if (init?.body instanceof FormData) {
    const { body, contentType } = await encodeMultipart(init.body);
    init = { ...init, body: body as unknown as BodyInit, headers: { ...(init.headers as Record<string, string>), "Content-Type": contentType } };
  }
  return realFetch(url as RequestInfo, init);
}) as typeof fetch;

// jsdom lacks these browser APIs used by charts.
(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
window.URL.createObjectURL = vi.fn(() => "blob:mock");
window.URL.revokeObjectURL = vi.fn();

// Plotly requires a real canvas/WebGL; replace the chart with a data dump we can assert on.
vi.mock("../components/ui", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../components/ui")>();
  return {
    ...mod,
    Plot: ({ data }: { data: unknown[] }) => React.createElement("div", { "data-testid": "plot", "data-traces": String(data.length) }, JSON.stringify(data).slice(0, 200)),
  };
});

export async function backendAvailable(): Promise<boolean> {
  try {
    const r = await realFetch(`${API_BASE}/api/health`);
    return r.ok;
  } catch {
    return false;
  }
}
