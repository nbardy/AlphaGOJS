#!/usr/bin/env node
/**
 * Launch the built app in Puppeteer, wait for RL progress, capture screenshots.
 *
 * Usage:
 *   npm run build && npm run launch:verify
 *   node benchmarks/launch_page_verify.mjs
 *   node benchmarks/launch_page_verify.mjs --pageQuery=preset=fast
 *   node benchmarks/launch_page_verify.mjs --url=http://localhost:8080/?preset=fast
 *   node benchmarks/launch_page_verify.mjs --outDir=benchmarks/results/my-run
 *
 * Prereq: `npm run build` (unless --url= dev server) and Puppeteer Chrome:
 *   npx puppeteer browsers install chrome
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  getPuppeteerLaunchOptions,
  loadPuppeteer,
  resolveBuiltAppFileUrl,
  waitForAppReady
} from './puppeteer_bench_common.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const out = {
    pageQuery: 'preset=fast',
    url: '',
    outDir: '',
    maxWaitSec: 120,
    minGames: 10,
    requireGeneration: true,
    requireChartData: true,
    skipBuildCheck: false
  };
  for (const arg of argv) {
    if (arg.startsWith('--pageQuery=')) out.pageQuery = arg.slice('--pageQuery='.length);
    else if (arg.startsWith('--url=')) out.url = arg.slice('--url='.length);
    else if (arg.startsWith('--outDir=')) out.outDir = arg.slice('--outDir='.length);
    else if (arg.startsWith('--maxWaitSec=')) {
      out.maxWaitSec = Math.max(30, parseInt(arg.slice('--maxWaitSec='.length), 10) || out.maxWaitSec);
    } else if (arg.startsWith('--minGames=')) {
      out.minGames = Math.max(1, parseInt(arg.slice('--minGames='.length), 10) || out.minGames);
    } else if (arg === '--no-require-gen') out.requireGeneration = false;
    else if (arg === '--no-require-chart') out.requireChartData = false;
    else if (arg === '--skip-build-check') out.skipBuildCheck = true;
  }
  return out;
}

function applyPageQuery(fileUrl, pageQuery) {
  if (!pageQuery || !String(pageQuery).trim()) return fileUrl;
  const u = new URL(fileUrl);
  const q = new URLSearchParams(String(pageQuery).replace(/^\?/, ''));
  for (const [k, v] of q.entries()) {
    u.searchParams.set(k, v);
  }
  return u.toString();
}

function stampDir() {
  return path.join(
    root,
    'benchmarks',
    'results',
    'launch-verify-' + new Date().toISOString().replace(/[:.]/g, '-')
  );
}

async function readStats(page) {
  return page.evaluate(() => {
    const ui = window.__alphaPlague;
    const s = ui && ui.trainer && ui.trainer.getStats ? ui.trainer.getStats() : {};
    return {
      gamesCompleted: s.gamesCompleted || 0,
      generation: s.generation || 0,
      loss: s.loss || 0,
      elo: s.elo || 0,
      workerReady: s.workerReady,
      lastWorkerError: s.lastWorkerError || '',
      metricsLen: ui && ui.metrics ? ui.metrics.length : 0,
      errorBanner: (document.getElementById('error-banner') || {}).textContent || ''
    };
  });
}

/** Loss chart is 3rd canvas in #charts-grid; true when a data point/line is drawn. */
async function lossChartHasData(page) {
  return page.evaluate(() => {
    const grid = document.getElementById('charts-grid');
    if (!grid) return false;
    const canvases = grid.querySelectorAll('canvas');
    const lossCanvas = canvases[2];
    if (!lossCanvas) return false;
    const ctx = lossCanvas.getContext('2d');
    const w = lossCanvas.width;
    const h = lossCanvas.height;
    const img = ctx.getImageData(0, 0, w, h).data;
    // Plot area roughly x>=48, y>=22 — look for loss line yellow (#ffcc00) or dot.
    for (let y = 22; y < h - 22; y++) {
      for (let x = 48; x < w - 12; x++) {
        const i = (y * w + x) * 4;
        const r = img[i];
        const g = img[i + 1];
        const b = img[i + 2];
        const a = img[i + 3];
        if (a < 200) continue;
        // Skip background (#0d0d22), grid (#1a1a3a), text (#6666aa), title (#8888cc)
        if (r < 0x30 && g < 0x30 && b < 0x40) continue;
        if (r === 0x66 && g === 0x66 && b === 0xaa) continue;
        if (r === 0x88 && g === 0x88 && b === 0xcc) continue;
        if (r === 0x1a && g === 0x1a && b === 0x3a) continue;
        if (r === 0x44 && g === 0x44 && b === 0x66) continue;
        // Loss color #ffcc00 or nearby
        if (r > 0xc0 && g > 0x90 && b < 0x80) return true;
        // Any bright accent in plot (green/orange/pink series)
        if (r > 0x90 || g > 0x90) return true;
      }
    }
    return false;
  });
}

async function screenshotEl(page, selector, outPath, pad) {
  pad = pad || 8;
  const el = await page.$(selector);
  if (!el) {
    console.warn('launch:verify skip screenshot — missing', selector);
    return false;
  }
  const box = await el.boundingBox();
  if (!box) return false;
  await page.screenshot({
    path: outPath,
    clip: {
      x: Math.max(0, box.x - pad),
      y: Math.max(0, box.y - pad),
      width: box.width + pad * 2,
      height: box.height + pad * 2
    }
  });
  return true;
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2));
  const outDir = cfg.outDir ? path.resolve(root, cfg.outDir) : stampDir();
  fs.mkdirSync(outDir, { recursive: true });

  let targetUrl = cfg.url;
  if (!targetUrl) {
    if (!cfg.skipBuildCheck) {
      const { indexPath } = resolveBuiltAppFileUrl(root);
      if (!fs.existsSync(indexPath)) {
        console.error('Missing docs/index.html — run `npm run build` first or pass --url=');
        process.exit(1);
      }
    }
    const { fileUrl } = resolveBuiltAppFileUrl(root);
    targetUrl = applyPageQuery(fileUrl, cfg.pageQuery);
  }

  const puppeteer = await loadPuppeteer();
  const browser = await puppeteer.launch(
    getPuppeteerLaunchOptions({ headless: true, protocolTimeout: 600000 })
  );

  const logs = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 960, height: 1400, deviceScaleFactor: 1 });
    page.on('console', (msg) => {
      const t = msg.type();
      const text = msg.text();
      if (t === 'error' || text.includes('[gpu_owner]') || text.includes('[tf]')) {
        logs.push(t + ': ' + text);
      }
    });
    page.on('pageerror', (err) => logs.push('PAGEERROR: ' + err.message));

    console.log('launch:verify opening', targetUrl);
    await page.goto(targetUrl, { waitUntil: 'networkidle0', timeout: 120000 });
    await waitForAppReady(page, 120000);

    const deadline = Date.now() + cfg.maxWaitSec * 1000;
    let stats = await readStats(page);
    let lastLog = 0;

    while (Date.now() < deadline) {
      stats = await readStats(page);
      const okGames = stats.gamesCompleted >= cfg.minGames;
      const okGen = !cfg.requireGeneration || stats.generation >= 1;
      const okMetrics = !cfg.requireChartData || stats.metricsLen >= 1;
      const okChart = !cfg.requireChartData || (okMetrics && (await lossChartHasData(page)));
      if (okGames && okGen && okMetrics && okChart && !stats.lastWorkerError) break;

      if (Date.now() - lastLog > 5000) {
        console.log(
          'launch:verify waiting… games=',
          stats.gamesCompleted,
          'gen=',
          stats.generation,
          'metrics=',
          stats.metricsLen,
          stats.lastWorkerError ? '(worker error)' : ''
        );
        lastLog = Date.now();
      }
      await new Promise((r) => setTimeout(r, 1000));
    }

    // Allow one RAF cycle after metrics snapshot before screenshots.
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));

    stats = await readStats(page);
    const chartOk = !cfg.requireChartData || (stats.metricsLen >= 1 && (await lossChartHasData(page)));
    fs.writeFileSync(path.join(outDir, 'stats.json'), JSON.stringify({ ...stats, lossChartHasData: chartOk }, null, 2) + '\n');
    if (logs.length) {
      fs.writeFileSync(path.join(outDir, 'console.log'), logs.join('\n') + '\n');
    }

    await page.screenshot({ path: path.join(outDir, '01-full-page.png'), fullPage: true });
    await screenshotEl(page, '#game-grid', path.join(outDir, '02-game-grid.png'));
    await screenshotEl(page, '#charts-section', path.join(outDir, '03-charts-section.png'));
    await screenshotEl(page, '#stats', path.join(outDir, '04-stats-bar.png'));

    const summary = [
      '# Launch page verify',
      '',
      'url=' + targetUrl,
      'outDir=' + outDir,
      '',
      '## Stats',
      '- gamesCompleted: ' + stats.gamesCompleted,
      '- generation: ' + stats.generation,
      '- loss: ' + stats.loss,
      '- metrics snapshots: ' + stats.metricsLen,
      '- loss chart drawn: ' + (chartOk ? 'yes' : 'no'),
      '- workerError: ' + (stats.lastWorkerError || '(none)'),
      '',
      '## Screenshots',
      '- 01-full-page.png',
      '- 02-game-grid.png',
      '- 03-charts-section.png',
      '- 04-stats-bar.png',
      ''
    ].join('\n');
    fs.writeFileSync(path.join(outDir, 'summary.md'), summary);

    const passGames = stats.gamesCompleted >= cfg.minGames;
    const passGen = !cfg.requireGeneration || stats.generation >= 1;
    const passChart = !cfg.requireChartData || chartOk;
    const passErr = !stats.lastWorkerError;

    console.log(summary);
    if (!passGames || !passGen || !passChart || !passErr) {
      console.error('launch:verify FAILED');
      if (!passGames) console.error('  need gamesCompleted >=', cfg.minGames, 'got', stats.gamesCompleted);
      if (!passGen) console.error('  need generation >= 1 got', stats.generation);
      if (!passChart) console.error('  need loss chart data (metricsLen=', stats.metricsLen, ')');
      if (!passErr) console.error('  worker error:', stats.lastWorkerError);
      process.exit(1);
    }
    console.log('launch:verify OK');
  } finally {
    await browser.close();
  }
}

main().catch((e) => {
  console.error('launch:verify error:', e.message || e);
  process.exit(1);
});
