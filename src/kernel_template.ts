import baseWgsl from './fused_ppo.wgsl?raw';

export function generateKernel(D: number): string {
  let wgsl = baseWgsl.replace(
    /const D: u32 = \d+u;/,
    `const D: u32 = ${D}u;`
  );
  // Profiling ablation (headless only): set globalThis.__ABLATE__ to a comma list of
  // phase names (CONV2BWD,REDUCE,CONV1BWD,EMBEDBWD) to zero each phase's loop bound and
  // measure its wall-clock cost. No-op in the browser (globalThis.__ABLATE__ undefined).
  const ablate: string = (globalThis as any).__ABLATE__ ?? '';
  for (const name of ablate.split(',').map((s) => s.trim()).filter(Boolean)) {
    wgsl = wgsl.replace(`const RUN_${name}: u32 = 1u;`, `const RUN_${name}: u32 = 0u;`);
  }
  return wgsl;
}
