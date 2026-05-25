import baseWgsl from './fused_ppo.wgsl?raw';

export function generateKernel(D: number): string {
  return baseWgsl.replace(
    /const D: u32 = \d+u;/,
    `const D: u32 = ${D}u;`
  );
}
