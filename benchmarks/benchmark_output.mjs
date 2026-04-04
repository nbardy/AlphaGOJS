import fs from 'node:fs';
import path from 'node:path';

function sanitizeSegment(value) {
  const cleaned = String(value || '')
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return cleaned || 'run';
}

function defaultRunId() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function getArgValue(argv, key) {
  const prefix = '--' + key + '=';
  for (const arg of argv) {
    if (arg === '--' + key) return 'true';
    if (arg.startsWith(prefix)) return arg.slice(prefix.length);
  }
  return null;
}

function parseBoolArg(rawValue, fallback = false) {
  if (rawValue == null) return fallback;
  if (rawValue === '' || rawValue === 'true' || rawValue === '1') return true;
  if (rawValue === 'false' || rawValue === '0') return false;
  return fallback;
}

function ensureParentDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

export function prepareBenchmarkOutput(benchmarkId, argv = process.argv.slice(2)) {
  const runId = sanitizeSegment(
    process.env.BENCH_BUNDLE_ID || getArgValue(argv, 'runId') || defaultRunId()
  );
  const baseOutputDir = getArgValue(argv, 'outDir') || process.env.BENCH_OUTPUT_DIR || '';
  const outputDir = baseOutputDir
    ? path.resolve(baseOutputDir)
    : path.resolve(process.cwd(), 'benchmarks', 'results', runId + '-' + sanitizeSegment(benchmarkId));
  const jsonPath = path.resolve(
    getArgValue(argv, 'jsonOut') || path.join(outputDir, benchmarkId + '.json')
  );
  const summaryPath = path.resolve(
    getArgValue(argv, 'summaryOut') || path.join(outputDir, benchmarkId + '.summary.md')
  );
  const printJson = parseBoolArg(getArgValue(argv, 'printJson'), false);
  const quiet = parseBoolArg(getArgValue(argv, 'quiet'), false);

  ensureParentDir(jsonPath);
  ensureParentDir(summaryPath);

  return {
    benchmarkId,
    runId,
    outputDir,
    jsonPath,
    summaryPath,
    printJson,
    quiet
  };
}

export function relativeOutputPath(filePath) {
  const rel = path.relative(process.cwd(), filePath);
  return rel && !rel.startsWith('..') ? rel : filePath;
}

export function writeBenchmarkArtifacts(output, payload, summaryLines) {
  const summaryText = (summaryLines || []).join('\n').trimEnd() + '\n';
  fs.writeFileSync(output.jsonPath, JSON.stringify(payload, null, 2) + '\n');
  fs.writeFileSync(output.summaryPath, summaryText);
  return { summaryText };
}

export function emitBenchmarkReport(output, payload, summaryLines) {
  const { summaryText } = writeBenchmarkArtifacts(output, payload, summaryLines);
  if (!output.quiet) {
    process.stdout.write(summaryText);
    console.log('saved json=' + relativeOutputPath(output.jsonPath));
    console.log('saved summary=' + relativeOutputPath(output.summaryPath));
    if (output.printJson) {
      console.log(JSON.stringify(payload, null, 2));
    }
  }
  return { summaryText };
}

export function formatNumber(value, digits = 2, fallback = 'n/a') {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : fallback;
}

export function computePercentDelta(value, baseline) {
  if (!Number.isFinite(value) || !Number.isFinite(baseline) || baseline === 0) return null;
  return ((value - baseline) / Math.abs(baseline)) * 100;
}

export function formatSignedPercent(delta, digits = 1, fallback = 'n/a') {
  if (!Number.isFinite(delta)) return fallback;
  const rounded = Number(delta).toFixed(digits);
  return (delta >= 0 ? '+' : '') + rounded + '%';
}

export function readJsonIfExists(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (err) {
    return null;
  }
}

export function readTextIfExists(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch (err) {
    return null;
  }
}
