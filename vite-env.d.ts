/// <reference types="vite/client" />

declare module '*.wgsl?raw' {
  const content: string;
  export default content;
}

// Float16Array — Stage 3 TC39, available in Chrome 127+ / Bun
declare class Float16Array extends TypedArray {
  constructor(length: number);
  constructor(array: ArrayLike<number> | ArrayBufferLike);
  constructor(buffer: ArrayBufferLike, byteOffset?: number, length?: number);
  static from(source: ArrayLike<number>): Float16Array;
  [index: number]: number;
  readonly length: number;
  readonly BYTES_PER_ELEMENT: 2;
  readonly buffer: ArrayBuffer;
  readonly byteLength: number;
  readonly byteOffset: number;
  set(array: ArrayLike<number>, offset?: number): void;
  slice(start?: number, end?: number): Float16Array;
  subarray(begin?: number, end?: number): Float16Array;
}

interface TypedArray {
  readonly length: number;
  readonly buffer: ArrayBufferLike;
  readonly byteLength: number;
  readonly byteOffset: number;
}
