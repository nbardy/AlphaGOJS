import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 6974 },
  build: {
    outDir: 'dist',
    target: 'esnext',
    minify: 'esbuild',
    sourcemap: true,
  },
  // Required for WebGPU (needs secure context)
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
