#!/usr/bin/env node
/**
 * Run multiple benchmarks in sequence. Intended for local smoke or nightly.
 *
 * Usage:
 *   node benchmarks/run_all_benchmarks.mjs
 *   node benchmarks/run_all_benchmarks.mjs --smoke
 *   node benchmarks/run_all_benchmarks.mjs --skip-build
 *   node benchmarks/run_all_benchmarks.mjs --with-native-webgpu   # parity (needs adapter)
 *
 * Default: production build, WGSL spread throughput, loop bench (short), system bench (short).
 * Omits webgpu:parity unless --with-native-webgpu (often fails without GPU/Dawn).
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const out = {
    smoke: false,
    skipBuild: false,
    withNativeWebgpu: false
  };
  for (const arg of argv) {
    if (arg === '--smoke') out.smoke = true;
    else if (arg === '--skip-build') out.skipBuild = true;
    else if (arg === '--with-native-webgpu') out.withNativeWebgpu = true;
  }
  return out;
}

function runNpm(script, extraArgs = []) {
  const args = ['run', script, '--', ...extraArgs];
  return spawnSync('npm', args, { cwd: root, stdio: 'inherit', shell: true });
}

function runNode(relPath, args = []) {
  const script = path.join(root, relPath);
  return spawnSync(process.execPath, [script, ...args], { cwd: root, stdio: 'inherit' });
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  const failures = [];

  console.log('\n======== AlphaPlague benchmark bundle ========\n');
  console.log('root=' + root + ' smoke=' + flags.smoke + ' skipBuild=' + flags.skipBuild);

  if (!flags.skipBuild) {
    console.log('\n--- npm run build ---\n');
    const b = runNpm('build');
    if ((b.status ?? 1) !== 0) failures.push('npm run build');
  }

  console.log('\n--- bench:webgpu:spread ---\n');
  const sp = runNpm('bench:webgpu:spread');
  if ((sp.status ?? 1) !== 0) failures.push('bench:webgpu:spread');

  if (flags.withNativeWebgpu) {
    console.log('\n--- bench:webgpu:parity ---\n');
    const pr = runNpm('bench:webgpu:parity');
    if ((pr.status ?? 1) !== 0) failures.push('bench:webgpu:parity');
  }

  const loopArgs = flags.smoke
    ? ['--duration=3', '--runs=1', '--warmup=1']
    : ['--duration=8', '--runs=2', '--warmup=2'];
  console.log('\n--- bench:loop ' + loopArgs.join(' ') + ' ---\n');
  const lp = runNode('benchmarks/loop_decomposition_benchmark.mjs', loopArgs);
  if ((lp.status ?? 1) !== 0) failures.push('bench:loop');

  const sysArgs = flags.smoke
    ? ['--duration=4', '--runs=1', '--warmup=1', '--pipelines=single_gpu_phased,cpu_actors_gpu_learner']
    : ['--duration=10', '--runs=2', '--warmup=3'];
  console.log('\n--- bench:system (headless) ' + sysArgs.join(' ') + ' ---\n');
  const sy = runNode('benchmarks/system_interface_benchmark.mjs', sysArgs);
  if ((sy.status ?? 1) !== 0) failures.push('bench:system:headless');

  console.log('\n======== Done ========\n');
  if (failures.length) {
    console.error('Failed steps: ' + failures.join(', '));
    process.exitCode = 1;
  } else {
    console.log('All steps completed with exit code 0.');
  }
}

main();
