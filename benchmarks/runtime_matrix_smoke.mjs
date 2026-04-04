#!/usr/bin/env node
/**
 * Smoke matrix: a few **distinct** runtime / URL combinations (not full factorial).
 *
 * What exists elsewhere:
 *   - `npm run bench:all` — build + WebGPU spread + loop decomposition + system (pipelines subset).
 *   - `system_interface_benchmark.mjs` — sweep `--pipelines=` (worker vs main-thread CPU).
 *   - `loop_decomposition_benchmark.mjs` — sim_random / sim_forward / full + `--webgpuEnv=1`.
 *   - Native WGSL throughput: `bench:webgpu:spread` (not TF.js).
 *
 * This script runs **system_interface_benchmark** twice (default URL vs `webgpuEnv=1`) with the
 * same light timing so you can compare WGSL sim on/off across **gpu_worker** pipelines.
 * It does **not** measure “unified memory / zero-copy” (not exposed as one knob in browser TF.js).
 *
 * Usage:
 *   node benchmarks/runtime_matrix_smoke.mjs
 *   node benchmarks/runtime_matrix_smoke.mjs --skip-build
 *
 * Prereq: `npm run build` (unless --skip-build), Puppeteer Chrome.
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const systemBench = path.join(root, 'benchmarks/system_interface_benchmark.mjs');

const skipBuild = process.argv.includes('--skip-build');

const commonArgs = [
  '--duration=5',
  '--runs=1',
  '--warmup=1',
  '--inferenceRuns=12',
  '--ticks=16',
  '--pipelines=single_gpu_phased,full_gpu_resident,cpu_actors_gpu_learner',
  '--timeoutMs=300000',
  '--protocolTimeoutMs=600000'
];

const scenarios = [
  { id: 'page_default', pageQuery: '', label: 'index.html (default query)' },
  { id: 'page_webgpuEnv', pageQuery: 'webgpuEnv=1', label: 'index.html?webgpuEnv=1 (WGSL sim when GPU allows)' }
];

function main() {
  if (!skipBuild) {
    const b = spawnSync('npm', ['run', 'build'], { cwd: root, stdio: 'inherit', shell: process.platform === 'win32' });
    if ((b.status ?? 1) !== 0) {
      process.exit(b.status ?? 1);
    }
  }

  const parentDir = path.join(
    root,
    'benchmarks',
    'results',
    'matrix-smoke-' + new Date().toISOString().replace(/[:.]/g, '-')
  );
  fs.mkdirSync(parentDir, { recursive: true });

  const lines = ['# Runtime matrix (smoke)', 'parentDir=' + parentDir, ''];

  let failed = false;
  for (const s of scenarios) {
    const outDir = path.join(parentDir, s.id);
    fs.mkdirSync(outDir, { recursive: true });
    const args = [
      systemBench,
      ...commonArgs,
      '--outDir=' + outDir,
      '--runId=' + s.id
    ];
    if (s.pageQuery) {
      args.push('--pageQuery=' + s.pageQuery);
    }

    console.log('\n======== matrix: ' + s.label + ' ========\n');
    const r = spawnSync(process.execPath, args, { cwd: root, stdio: 'inherit' });
    const ok = (r.status ?? 1) === 0;
    if (!ok) failed = true;
    lines.push('## ' + s.label);
    lines.push('status=' + (ok ? 'ok' : 'failed'));
    lines.push('outDir=' + path.relative(root, outDir));
    lines.push('');
  }

  fs.writeFileSync(path.join(parentDir, 'matrix_summary.md'), lines.join('\n').trimEnd() + '\n');
  console.log('\nWrote ' + path.relative(root, path.join(parentDir, 'matrix_summary.md')));
  process.exit(failed ? 1 : 0);
}

main();
