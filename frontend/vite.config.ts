import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The browser never talks to the backend directly: /api is proxied by the dev server
// (and by nginx / the FastAPI static mount in production).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  preview: { host: "0.0.0.0", port: 5173, allowedHosts: true },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 4000 },
});
