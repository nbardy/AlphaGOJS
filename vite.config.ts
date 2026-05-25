import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 6974 },
  base: '/AlphaGOJS/',
  build: {
    outDir: 'docs',
    target: 'esnext',
    minify: 'esbuild',
    sourcemap: true,
  },
  worker: {
    format: 'es',
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
