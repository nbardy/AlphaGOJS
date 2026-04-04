/**
 * Shared Puppeteer + Chrome flags for benchmarks that open docs/index.html.
 * Keeps loop_decomposition and system_interface benches in sync.
 */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

/** Chrome flags for WebGPU + file:// docs in headless CI. */
export function getChromeLaunchArgs() {
  const args = [
    '--enable-unsafe-webgpu',
    '--allow-file-access-from-files',
    '--no-sandbox',
    '--disable-dev-shm-usage'
  ];
  if (process.platform === 'linux') {
    args.push('--enable-features=Vulkan');
    args.push('--use-angle=vulkan');
  }
  if (process.platform === 'darwin') {
    args.push('--use-angle=metal');
  }
  return args;
}

export async function loadPuppeteer() {
  try {
    const mod = await import('puppeteer');
    return mod.default || mod;
  } catch (e) {
    throw new Error('`puppeteer` is not installed. Run `npm install`.');
  }
}

export async function waitForAppReady(page, timeoutMs) {
  await page.waitForFunction(() => {
    const ui = window.__alphaPlague;
    return !!(ui && ui.trainer && typeof ui.trainer.getStats === 'function');
  }, { timeout: timeoutMs });
}

/**
 * Merge these into `puppeteer.launch(...)` so benchmarks can use **your** Chrome/Chromium.
 *
 * Set either env var to the real binary (not the `.app` folder on macOS):
 *   `PUPPETEER_EXECUTABLE_PATH` — official Puppeteer name
 *   `CHROME_PATH` — short alias for this repo
 *
 * macOS example:
 *   export CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
 *
 * If neither is set, Puppeteer uses its **downloaded** Chrome-for-Testing (what
 * `npx puppeteer browsers install chrome` installs). That avoids version skew with
 * the bundled protocol client but is not required if your system Chrome works.
 *
 * @param {object} [overrides] — e.g. `{ headless: true, protocolTimeout: 600000 }`
 */
export function getPuppeteerLaunchOptions(overrides = {}) {
  const out = {
    args: getChromeLaunchArgs(),
    headless: true,
    ...overrides
  };
  const chromePath = (process.env.PUPPETEER_EXECUTABLE_PATH || process.env.CHROME_PATH || '').trim();
  if (chromePath) {
    out.executablePath = chromePath;
  }
  return out;
}

/**
 * @param {string} cwd - usually process.cwd()
 * @returns {{ indexPath: string, fileUrl: string }}
 */
export function resolveBuiltAppFileUrl(cwd) {
  const indexPath = path.resolve(cwd || process.cwd(), 'docs/index.html');
  if (!fs.existsSync(indexPath)) {
    throw new Error('Missing docs/index.html. Run `npm run build` first.');
  }
  return { indexPath, fileUrl: pathToFileURL(indexPath).toString() };
}
