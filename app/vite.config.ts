import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";
import path from "node:path";
import fs from "node:fs";

/**
 * Voice switch, VITE_AUDIO (read by src/lib/clips.ts too):
 *   pack (default)  public/audio/ as it is. The repo ships NO speech there (only README.md), so a public deploy publishes
 *                   no Apple text-to-speech; recordings + pack.json dropped into public/audio/ switch the voice on.
 *   placeholder     also emits the macOS text-to-speech placeholders from audio-placeholder/ as audio/* (local demos,
 *                   video recording). Do not deploy this build publicly (Apple's licence for system-voice output).
 *   none            no audio files at all, even if public/audio/ holds a pack (the app shows text + icons only).
 */
const AUDIO = process.env.VITE_AUDIO === "placeholder" || process.env.VITE_AUDIO === "none" ? process.env.VITE_AUDIO : "pack";
process.env.VITE_AUDIO = AUDIO;
const PUBLIC = path.resolve(import.meta.dirname, "public");
const PLACEHOLDER = path.resolve(import.meta.dirname, "audio-placeholder");
const AUDIO_FILE = /^audio\/(?!README).+\.(mp3|m4a|json)$/i;

function farzAudio(): Plugin {
  const walk = (d: string, rel = ""): string[] =>
    fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name), `${rel}${e.name}/`) : [`${rel}${e.name}`]);
  return {
    name: "farz-audio",
    config: () => (AUDIO === "none" ? { build: { copyPublicDir: false } } : {}),
    configureServer(server) {
      if (AUDIO !== "placeholder") return;
      server.middlewares.use((req, res, next) => {
        const m = /\/audio\/([\w.-]+\.(mp3|json))$/.exec(req.url ?? "");
        const f = m && path.join(PLACEHOLDER, m[1]);
        if (!f || !fs.existsSync(f)) return next();
        res.setHeader("Content-Type", f.endsWith(".json") ? "application/json" : "audio/mpeg");
        res.end(fs.readFileSync(f));
      });
    },
    generateBundle() {
      if (AUDIO === "none") {
        // publicDir is not copied: emit every public file except the audio pack
        for (const rel of walk(PUBLIC)) if (!AUDIO_FILE.test(rel)) this.emitFile({ type: "asset", fileName: rel, source: fs.readFileSync(path.join(PUBLIC, rel)) });
      } else if (AUDIO === "placeholder") {
        // audio-placeholder/ is not in the public export: without it this build simply has no voice
        if (!fs.existsSync(PLACEHOLDER)) return this.warn("VITE_AUDIO=placeholder but app/audio-placeholder/ is missing: no voice in this build");
        for (const f of fs.readdirSync(PLACEHOLDER)) if (/\.(mp3|json)$/.test(f)) this.emitFile({ type: "asset", fileName: `audio/${f}`, source: fs.readFileSync(path.join(PLACEHOLDER, f)) });
      }
    },
  };
}

export default defineConfig({
  // relative base so the same dist/ works at a domain root (Vercel, Lovable) or a sub-path
  base: "./",
  resolve: {
    alias: [
      // use the build that loads the wasm runtime from ort.env.wasm.wasmPaths (our own /ort/ folder, copied
      // from node_modules by scripts/copy-ort.mjs) instead of letting the bundler emit a second copy
      { find: /^onnxruntime-web\/wasm$/, replacement: path.resolve(import.meta.dirname, "node_modules/onnxruntime-web/dist/ort.wasm.min.mjs") },
    ],
  },
  define: { "import.meta.env.VITE_AUDIO": JSON.stringify(AUDIO) },
  worker: { format: "es" },
  build: { target: "es2022", sourcemap: false, chunkSizeWarningLimit: 900 },
  plugins: [
    farzAudio(),
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      injectRegister: false,
      manifest: {
        name: "فرز Farz",
        short_name: "فرز",
        description: "فحص ذاتي لجودة البن الأخضر، يشتغل بدون نت — Offline green-coffee bean self-check",
        lang: "ar",
        dir: "rtl",
        start_url: "./",
        scope: "./",
        display: "standalone",
        orientation: "portrait",
        background_color: "#2b1b14",
        theme_color: "#2b1b14",
        categories: ["utilities", "productivity"],
        icons: [
          { src: "icons/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icons/icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "icons/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
          { src: "icons/icon.svg", sizes: "any", type: "image/svg+xml" },
        ],
      },
      workbox: {
        // precache EVERYTHING the app needs offline: shell, worker, ORT wasm runtime, model, labels, audio, demo trays
        globPatterns: ["**/*.{html,js,mjs,css,svg,png,jpg,jpeg,webp,woff2,wasm,onnx,json,mp3,m4a,webmanifest}"],
        globIgnores: ["**/README*"],
        maximumFileSizeToCacheInBytes: 32 * 1024 * 1024,
        navigateFallback: "index.html",
        navigateFallbackDenylist: [/^\/(models|ort|audio|demo|calib-demo)\//],
        cleanupOutdatedCaches: true,
        clientsClaim: true,
        skipWaiting: true,
      },
    }),
  ],
  preview: { port: 4180, strictPort: true },
});
