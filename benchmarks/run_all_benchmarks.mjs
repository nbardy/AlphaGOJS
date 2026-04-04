#!/usr/bin/env node
/**
 * Run multiple benchmarks in sequence with terse console output.
 *
 * Usage:
 *   node benchmarks/run_all_benchmarks.mjs
 *   node benchmarks/run_all_benchmarks.mjs --smoke
 *   node benchmarks/run_all_benchmarks.mjs --skip-build
 *   node benchmarks/run_all_benchmarks.mjs --with-native-webgpu
 *   node benchmarks/run_all_benchmarks.mjs --full-system
 *   node benchmarks/run_all_benchmarks.mjs --outDir=benchmarks/results/my-run
 *
 * Prerequisites: `npm run build` output in `docs/`; Puppeteer Chrome:
 *   npx puppeteer browsers install chrome
 */
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readTextIfExists, relativeOutputPath } from './benchmark_output.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const out = {
    smoke: false,
    skipBuild: false,
    withNativeWebgpu: false,
    fullSystem: false,
    outDir: ''
  };
  for (const arg of argv) {
    if (arg === '--smoke') out.smoke = true;
    else if (arg === '--skip-build') out.skipBuild = true;
    else if (arg === '--with-native-webgpu') out.withNativeWebgpu = true;
    else if (arg === '--full-system') out.fullSystem = true;
    else if (arg.startsWith('--outDir=')) out.outDir = arg.slice('--outDir='.length);
  }
  return out;
}

function makeRunId(flags) {
  const label = flags.smoke ? 'bench-all-smoke' : 'bench-all';
  return new Date().toISOString().replace(/[:.]/g, '-') + '-' + label;
}

function resolveBundleDir(flags) {
  if (flags.outDir) return path.resolve(root, flags.outDir);
  return path.join(root, 'benchmarks', 'results', makeRunId(flags));
}

function tailFile(filePath, maxLines = 20) {
  const text = readTextIfExists(filePath);
  if (!text) return '';
  const lines = text.trimEnd().split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - maxLines)).join('\n');
}

function runLoggedCommand(command, args, options) {
  const stdoutFd = fs.openSync(options.stdoutPath, 'w');
  const stderrFd = fs.openSync(options.stderrPath, 'w');
  try {
    return spawnSync(command, args, {
      cwd: root,
      env: { ...process.env, ...(options.env || {}) },
      shell: !!options.shell,
      stdio: ['ignore', stdoutFd, stderrFd]
    });
  } finally {
    fs.closeSync(stdoutFd);
    fs.closeSync(stderrFd);
  }
}

function summaryPathFor(benchmarkId, outputDir) {
  return path.join(outputDir, benchmarkId + '.summary.md');
}

function jsonPathFor(benchmarkId, outputDir) {
  return path.join(outputDir, benchmarkId + '.json');
}

function reportStep(step, result, bundleLines) {
  const ok = (result.status ?? 1) === 0;
  const summaryPath = step.benchmarkId ? summaryPathFor(step.benchmarkId, step.outputDir) : null;
  const jsonPath = step.benchmarkId ? jsonPathFor(step.benchmarkId, step.outputDir) : null;
  const summaryText = summaryPath ? readTextIfExists(summaryPath) : null;
  const stdoutRel = relativeOutputPath(step.stdoutPath);
  const stderrRel = relativeOutputPath(step.stderrPath);

  console.log('\n--- ' + step.label + ' ---');
  bundleLines.push('');
  bundleLines.push('## ' + step.label);
  bundleLines.push('status=' + (ok ? 'ok' : 'failed'));

  if (summaryText) {
    process.stdout.write(summaryText.endsWith('\n') ? summaryText : summaryText + '\n');
    bundleLines.push(summaryText.trimEnd());
  } else if (step.fallbackSummaryFromStdout) {
    const fallback = step.fallbackSummaryFromStdout(step.stdoutPath);
    if (fallback) {
      console.log(fallback);
      bundleLines.push(fallback);
    }
  }

  if (jsonPath && fs.existsSync(jsonPath)) {
    const jsonRel = relativeOutputPath(jsonPath);
    console.log('saved json=' + jsonRel);
    bundleLines.push('saved json=' + jsonRel);
  }
  if (summaryPath && fs.existsSync(summaryPath)) {
    const summaryRel = relativeOutputPath(summaryPath);
    console.log('saved summary=' + summaryRel);
    bundleLines.push('saved summary=' + summaryRel);
  }

  console.log('stdout log=' + stdoutRel);
  console.log('stderr log=' + stderrRel);
  bundleLines.push('stdout log=' + stdoutRel);
  bundleLines.push('stderr log=' + stderrRel);

  if (!ok) {
    const tail = tailFile(step.stderrPath, 25) || tailFile(step.stdoutPath, 25);
    if (tail) {
      console.log('last log lines:\n' + tail);
      bundleLines.push('last log lines:');
      bundleLines.push('```');
      bundleLines.push(tail);
      bundleLines.push('```');
    }
  }
}

function buildSteps(flags, outputDir, benchEnv) {
  const steps = [];

  if (!flags.skipBuild) {
    steps.push({
      id: 'build',
      label: 'npm run build',
      outputDir,
      stdoutPath: path.join(outputDir, 'build.stdout.log'),
      stderrPath: path.join(outputDir, 'build.stderr.log'),
      run() {
        return runLoggedCommand('npm', ['run', 'build'], {
          stdoutPath: this.stdoutPath,
          stderrPath: this.stderrPath,
          shell: process.platform === 'win32'
        });
      }
    });
  }

  steps.push({
    id: 'bench-webgpu-spread',
    label: 'bench:webgpu:spread',
    benchmarkId: 'webgpu_plague_spread_throughput',
    outputDir,
    stdoutPath: path.join(outputDir, 'bench-webgpu-spread.stdout.log'),
    stderrPath: path.join(outputDir, 'bench-webgpu-spread.stderr.log'),
    run() {
      return runLoggedCommand('npm', ['run', 'bench:webgpu:spread'], {
        env: benchEnv,
        stdoutPath: this.stdoutPath,
        stderrPath: this.stderrPath,
        shell: process.platform === 'win32'
      });
    }
  });

  if (flags.withNativeWebgpu) {
    steps.push({
      id: 'bench-webgpu-parity',
      label: 'bench:webgpu:parity',
      outputDir,
      stdoutPath: path.join(outputDir, 'bench-webgpu-parity.stdout.log'),
      stderrPath: path.join(outputDir, 'bench-webgpu-parity.stderr.log'),
      fallbackSummaryFromStdout(stdoutPath) {
        const text = readTextIfExists(stdoutPath);
        return text ? text.trim() : '';
      },
      run() {
        return runLoggedCommand('npm', ['run', 'bench:webgpu:parity'], {
          stdoutPath: this.stdoutPath,
          stderrPath: this.stderrPath,
          shell: process.platform === 'win32'
        });
      }
    });
  }

  const loopArgs = flags.smoke
    ? ['--duration=3', '--runs=1', '--warmup=1']
    : ['--duration=8', '--runs=2', '--warmup=2'];
  steps.push({
    id: 'bench-loop',
    label: 'bench:loop ' + loopArgs.join(' '),
    benchmarkId: 'loop_decomposition_benchmark',
    outputDir,
    stdoutPath: path.join(outputDir, 'bench-loop.stdout.log'),
    stderrPath: path.join(outputDir, 'bench-loop.stderr.log'),
    run() {
      return runLoggedCommand(process.execPath, [path.join(root, 'benchmarks/loop_decomposition_benchmark.mjs'), ...loopArgs], {
        env: benchEnv,
        stdoutPath: this.stdoutPath,
        stderrPath: this.stderrPath
      });
    }
  });

  const sysArgs = flags.smoke
    ? [
        '--duration=4',
        '--runs=1',
        '--warmup=1',
        '--pipelines=single_gpu_phased,cpu_actors_gpu_learner',
        '--inferenceRuns=12'
      ]
    : [
        '--duration=10',
        '--runs=2',
        '--warmup=3',
        '--inferenceRuns=24'
      ];
  if (!flags.smoke && !flags.fullSystem) {
    sysArgs.push('--pipelines=single_gpu_phased,cpu_actors_gpu_learner');
  }

  steps.push({
    id: 'bench-system-headless',
    label: 'bench:system:headless ' + sysArgs.join(' '),
    benchmarkId: 'system_interface_benchmark',
    outputDir,
    stdoutPath: path.join(outputDir, 'bench-system-headless.stdout.log'),
    stderrPath: path.join(outputDir, 'bench-system-headless.stderr.log'),
    run() {
      return runLoggedCommand(process.execPath, [path.join(root, 'benchmarks/system_interface_benchmark.mjs'), ...sysArgs], {
        env: benchEnv,
        stdoutPath: this.stdoutPath,
        stderrPath: this.stderrPath
      });
    }
  });

  return steps;
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  const outputDir = resolveBundleDir(flags);
  fs.mkdirSync(outputDir, { recursive: true });

  const runId = path.basename(outputDir);
  const benchEnv = {
    BENCH_OUTPUT_DIR: outputDir,
    BENCH_BUNDLE_ID: runId
  };

  const manifest = {
    timestamp: new Date().toISOString(),
    root,
    outputDir,
    flags,
    steps: []
  };
  const bundleLines = [
    '# AlphaPlague benchmark bundle',
    'root=' + root,
    'output_dir=' + relativeOutputPath(outputDir),
    'smoke=' + flags.smoke + ' skipBuild=' + flags.skipBuild
      + ' withNativeWebgpu=' + flags.withNativeWebgpu
      + ' fullSystem=' + flags.fullSystem
  ];

  const steps = buildSteps(flags, outputDir, benchEnv);
  const failures = [];

  console.log('\n======== AlphaPlague benchmark bundle ========\n');
  console.log('output dir=' + relativeOutputPath(outputDir));
  console.log(
    'smoke=' + flags.smoke + ' skipBuild=' + flags.skipBuild
    + ' withNativeWebgpu=' + flags.withNativeWebgpu
    + ' fullSystem=' + flags.fullSystem
  );

  for (const step of steps) {
    const result = step.run();
    const ok = (result.status ?? 1) === 0;
    if (!ok) failures.push(step.label);

    manifest.steps.push({
      id: step.id,
      label: step.label,
      benchmarkId: step.benchmarkId || null,
      status: result.status ?? 1,
      stdoutPath: step.stdoutPath,
      stderrPath: step.stderrPath,
      summaryPath: step.benchmarkId ? summaryPathFor(step.benchmarkId, outputDir) : null,
      jsonPath: step.benchmarkId ? jsonPathFor(step.benchmarkId, outputDir) : null
    });

    reportStep(step, result, bundleLines);
  }

  const summaryPath = path.join(outputDir, 'bundle_summary.md');
  const manifestPath = path.join(outputDir, 'bundle_manifest.json');
  fs.writeFileSync(summaryPath, bundleLines.join('\n').trimEnd() + '\n');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');

  console.log('\n======== Done ========\n');
  console.log('bundle summary=' + relativeOutputPath(summaryPath));
  console.log('bundle manifest=' + relativeOutputPath(manifestPath));

  if (failures.length) {
    console.error('Failed steps: ' + failures.join(', '));
    process.exitCode = 1;
  } else {
    console.log('All steps completed with exit code 0.');
  }
}

main();
