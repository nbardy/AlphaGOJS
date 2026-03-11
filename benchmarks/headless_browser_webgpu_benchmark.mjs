#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function clampInt(v, fallback, min, max) {
  const n = Number.parseInt(v, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function parseArgs(argv) {
  const out = {
    warmup: 3,
    runs: 10,
    seqLen: 128,
    hiddenDim: 128,
    timeoutMs: 180000
  };

  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const [key, value = ''] = arg.slice(2).split('=', 2);
    if (key === 'warmup') out.warmup = clampInt(value, out.warmup, 0, 20);
    else if (key === 'runs') out.runs = clampInt(value, out.runs, 1, 100);
    else if (key === 'seq') out.seqLen = clampInt(value, out.seqLen, 16, 512);
    else if (key === 'hidden') out.hiddenDim = clampInt(value, out.hiddenDim, 16, 512);
    else if (key === 'timeoutMs') out.timeoutMs = clampInt(value, out.timeoutMs, 10000, 600000);
  }

  return out;
}

function chromeArgs() {
  const args = [
    '--enable-unsafe-webgpu',
    '--allow-file-access-from-files',
    '--no-sandbox',
    '--disable-dev-shm-usage'
  ];

  if (process.platform === 'linux') {
    // Chrome WebGPU guidance: Vulkan + ANGLE Vulkan on Linux.
    args.push('--enable-features=Vulkan');
    args.push('--use-angle=vulkan');
  }
  if (process.platform === 'darwin') {
    args.push('--use-angle=metal');
  }

  return args;
}

async function loadPuppeteer() {
  try {
    const mod = await import('puppeteer');
    return mod.default || mod;
  } catch (e) {
    throw new Error('`puppeteer` is not installed. Run `npm install` first.');
  }
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2));
  const puppeteer = await loadPuppeteer();

  const benchmarkPath = path.resolve(process.cwd(), 'benchmarks/browser_webgpu_benchmark.html');
  const benchmarkUrl = pathToFileURL(benchmarkPath).toString();

  const browser = await puppeteer.launch({
    headless: true,
    args: chromeArgs()
  });

  try {
    const page = await browser.newPage();
    await page.goto(benchmarkUrl, { waitUntil: 'load' });

    await page.$eval('#warmup', (el, v) => { el.value = String(v); }, cfg.warmup);
    await page.$eval('#runs', (el, v) => { el.value = String(v); }, cfg.runs);
    await page.$eval('#seq', (el, v) => { el.value = String(v); }, cfg.seqLen);
    await page.$eval('#hidden', (el, v) => { el.value = String(v); }, cfg.hiddenDim);

    await page.click('#run');

    await page.waitForFunction(() => {
      const text = document.getElementById('raw')?.textContent || '';
      if (!text || text.includes('pending')) return false;
      try {
        const data = JSON.parse(text);
        return !!data;
      } catch (e) {
        return false;
      }
    }, { timeout: cfg.timeoutMs });

    const raw = await page.$eval('#raw', (el) => el.textContent || '{}');
    const data = JSON.parse(raw);

    if (data.error) {
      throw new Error('Browser benchmark reported error: ' + data.error);
    }

    const env = data.env || {};
    const derived = data.derived || {};
    const transformer = derived.transformer || {};
    const upload = derived.uploadGbps || {};

    console.log('Headless browser WebGPU benchmark complete');
    console.log('config warmup=' + cfg.warmup + ' runs=' + cfg.runs + ' seq=' + cfg.seqLen + ' hidden=' + cfg.hiddenDim);
    console.log('ua=' + (env.userAgent || 'unknown'));
    console.log('upload median GB/s=' + (Number.isFinite(upload.median) ? upload.median.toFixed(3) : 'n/a'));
    console.log('transformer e2e tokens/s=' + (Number.isFinite(transformer.e2eTokensPerSec) ? transformer.e2eTokensPerSec.toFixed(1) : 'n/a'));
    console.log(JSON.stringify(data, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  const msg = err && err.message ? err.message : String(err);
  console.error('Headless browser WebGPU benchmark failed:', msg);
  process.exitCode = 1;
});
