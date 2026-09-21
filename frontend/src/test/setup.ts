/**
 * Test setup: route relative `/api` fetches to a live SentinAI backend
 * (SENTINAI_API_URL, default http://127.0.0.1:8000). Tests skip when it is unreachable.
 */
import React from "react";
import { vi } from "vitest";

export const API_BASE = (import.meta as unknown as { env: Record<string, string | undefined> }).env.SENTINAI_API_URL || "http://127.0.0.1:8000";

const realFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === "string" && input.startsWith("/") ? `${API_BASE}${input}` : input;
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
