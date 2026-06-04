# WebGPU Profiling Tools for AlphaGOJS: Kernel Timing Analysis

**Target Environment**: Headless bun-webgpu (Bun runtime) + Dawn backend → Metal on Apple M4

**Problem**: ppo_step dominates runStep time (~96%). Need precise profiling to locate bottlenecks within the fused kernel.

---

## 1. WebGPU TIMESTAMP QUERIES: Theory and Caveats

### Overview
WebGPU's optional `'timestamp-query'` feature allows you to record precise GPU timestamps at pass boundaries (compute or render). The mechanism:

- **GPUQuerySet Creation**: `device.createQuerySet({ type: 'timestamp', count: N })`
- **Recording**: Insert via `timestampWrites` descriptor on `beginComputePass()` (indices map to querySet slots)
- **Resolution**: Two 64-bit BigInt timestamps per query set invocation
- **Reading**: Resolve to buffer via `commandEncoder.resolveQuerySet()`, then map and read as BigInt64Array (nanoseconds)

### Quantization and Precision

**Security Caveat**: WebGPU timestamps are **quantized to 100-microsecond (0.1ms) granularity** due to timing attack mitigations. This is a hard constraint in the spec for isolated execution contexts.

- **Raw resolution**: 100 μs (10^5 ns minimum observable difference)
- **Data type**: 64-bit signed integer (BigInt64Array), stored as nanoseconds
- **Practical use**: Works well for millisecond-scale kernels; sub-millisecond timing differences are unreliable

**Native API Contrast**: Metal's direct GPU timestamps (via MTLCommandBuffer or native calibration) offer nanosecond precision by default, but are not exposed through WebGPU.

### Device Feature Exposure

To use timestamp queries:

```javascript
const device = await adapter.requestDevice({
  requiredFeatures: ["timestamp-query"],
});
```

### bun-webgpu Timestamp Support: **UNCERTAIN**

**Status**: bun-webgpu is an FFI wrapper around Dawn. The project uses Dawn's bindings and exposes a large portion of the WebGPU spec (78%+ conformance test pass rate), but **timestamp-query is not explicitly documented in the README or repository**.

**Next Steps to Verify**:
1. Try requesting the feature in headless.ts; if unsupported, device creation will throw
2. Check bun-webgpu's issue tracker or contact maintainers (repo: github.com/kommander/bun-webgpu)
3. Fall back to ablation or DAWN_TRACE if unavailable

### Metal (TBDR) Limitations

**Apple Silicon (M4) Issue**: Metal on Apple Silicon uses Tile-Based Deferred Rendering (TBDR). Timestamp queries require the GPU to report capability at discrete dispatch/draw boundaries. Apple Silicon reports support only at **atStageBoundary** (not atDispatchBoundary), which makes per-compute-pass timestamps **unreliable or unavailable** on M4 via standard WebGPU.

**Implication**: Even if bun-webgpu exposes timestamp-query, results from Metal may be quantized differently or fail silently. Test on M4 hardware to confirm.

---

## 2. EXTERNAL PROFILING TOOLS: Coverage and Applicability

### 2.1 Xcode Metal GPU Debugger (macOS/M4)

**Status**: ✅ **WORKS** for headless Dawn/bun process

**What it captures**:
- Full compute kernel disassembly and register/memory pressure
- Per-pass duration, thread occupancy, ALU/memory utilization
- Memory bandwidth, texture/buffer access patterns

**How to attach**:
1. Run bun under Xcode's debugger:
   ```bash
   cd /Users/nicholasbardy/git/AlphaGOJS
   export DAWN_TRACE_FILE_BASE=/tmp/gpu_trace  # Optional: also generate .gputrace file
   export METAL_CAPTURE_ENABLED=1              # Enable inline Metal frame capture
   xcode ... bun bench/headless.ts
   ```
2. Locate GPU process via Xcode's Debug → Attach to Process
3. Click Metal "M" button; select Command Queue or Device as source (not "Frames" — those don't work for headless)
4. Analyze shader time, register usage, etc. in the Metal timeline

**Gotchas**:
- Each capture requires a process restart (Chrome/Bun hangs after capture)
- Requires native debugging (not browser-based)
- No real-time iteration; each capture is a one-shot snapshot

**Recommended for**: Detailed register-level analysis, memory patterns, final validation

---

### 2.2 Chrome about:gpu (Browser Only)

**Status**: ❌ **Does not apply** — AlphaGOJS runs headless via bun-webgpu, not in a browser

This tool only works in Chrome's GPU process when running WebGPU in-browser. Headless bun-webgpu does not expose this interface.

---

### 2.3 RenderDoc (Cross-Platform GPU Debugger)

**Status**: ⚠️ **PARTIAL** — Limited WebGPU support; Metal backend unstable

**Current Scope (as of RenderDoc v1.44, May 2026)**:
- WebGPU/Dawn capture works **only on Windows with D3D12 backend**, Chrome v144+
- Metal backend capture: **Unsupported or undocumented**
- Headless/native process capture: Not designed for this use case

**Why it doesn't help for this project**:
- Bun runs on macOS/Metal backend (not Windows/D3D12)
- RenderDoc expects browser-initiated captures, not headless CLI processes
- Even if ported, unlikely to beat Xcode's native integration on Apple silicon

**Recommendation**: Skip RenderDoc for this profiling task.

---

### 2.4 PIX (Windows Only)

**Status**: ❌ **Not applicable** — Windows-only, Direct3D 12 only

PIX is Microsoft's GPU debugger for Windows game development. AlphaGOJS targets macOS/Metal, so this tool is out of scope.

---

### 2.5 Dawn's Native Tracing: DAWN_TRACE (Metal Backend)

**Status**: ✅ **WORKS** — Generates native Metal GPU trace

**Mechanism**:
1. Set environment variables:
   ```bash
   export DAWN_TRACE_FILE_BASE=/tmp/alphagojs_trace
   export DAWN_TRACE_DEVICE_FILTER="GPU"  # Optional: filter by device label
   ```
2. Run your headless kernel:
   ```bash
   bun bench/headless.ts
   ```
3. A `.gputrace` file is written to `/tmp/alphagojs_trace`

**How to view**:
- Load the `.gputrace` file directly into **Xcode's Metal Debugger** (File → Open)
- Xcode will parse and display the full GPU timeline: dispatch counts, kernel times, memory transfers, etc.

**Advantages**:
- Zero code modifications needed
- Captures **all GPU work** from device creation to destruction
- Metal-native format compatible with Xcode's analysis tools
- Good for whole-run profiling (not frame-specific)

**Limitations**:
- Trace file size can be large (MB–GB for long runs)
- No selective per-kernel or per-phase capture
- Opens in Xcode, not a standalone viewer

**Recommended for**: Full-run GPU timeline, hardware utilization overview

---

## 3. INTRA-KERNEL TIMING: The Ablation Method

### Problem Statement
WebGPU timestamps are coarse (100 μs quantization, per-pass only). Within your fused kernel (`rollout_step`, `gae_scan`, `ppo_step`), individual phases (e.g., forward pass vs. backward pass in ppo_step) cannot be separated by GPU timestamps alone.

### Ablation Strategy: Selective Early Return / Phase Stubs

The classic solution is to **fork the kernel and disable/short-circuit individual phases**, then measure wall-clock delta:

```
Baseline run (all phases enabled):  t_total ms
Run with Phase A stubbed out:       t_without_A ms
Phase A contribution:               t_total - t_without_A ms
```

### Implementation for AlphaGOJS

**Structure in fused_ppo.wgsl**:

The fused kernel has three entry points:
- `rollout_step`: Simulation phase (forward inference on policy)
- `gae_scan`: Value estimation and advantage calculation
- `ppo_step`: Forward (policy/value eval) + backward (gradient computation + Adam step)

**To ablate ppo_step**:

1. **Identify sub-phases** in the ppo_step entry point:
   - Phase 1: Forward pass (forward_pass_loop)
   - Phase 2: Advantage computation (advantage_loop)
   - Phase 3: Backward pass (backward_pass_loop)
   - Phase 4: Gradient reduction (reduction_loop)
   - Phase 5: Adam update (adam_step_loop)

2. **Create variant kernels** (one per ablation):
   ```
   fused_ppo_ablate_backward.wgsl   // stub out Phase 3
   fused_ppo_ablate_grad_reduce.wgsl // stub out Phase 4
   etc.
   ```

3. **Stub pattern** (example: skip backward):
   ```wgsl
   @compute @workgroup_size(WG)
   fn ppo_step(
     @builtin(global_invocation_id) gid: vec3<u32>,
   ) {
     let bid = gid.x;
     if (bid >= params.batch_size) { return; }

     // Phase 1: Forward
     forward_pass_loop();
     
     // Phase 2: Advantage
     advantage_loop();
     
     // Phase 3: STUB OUT (original backward_pass_loop is replaced with early return)
     // backward_pass_loop();  // <-- Commented out
     workgroupBarrier();
     return;  // <-- Early exit instead of continuing
     
     // Phases 4, 5 (unreachable)
   }
   ```

4. **Measure wall-clock in headless.ts**:
   ```javascript
   const perPhaseMs = {};
   
   // Baseline
   trainer.ppoPipeline = baseline_pipeline;
   const t0 = performance.now();
   for (let i = 0; i < 10; i++) await trainer.runStep();
   perPhaseMs.full = (performance.now() - t0) / 10;
   
   // Ablate backward
   trainer.ppoPipeline = ablate_backward_pipeline;
   const t1 = performance.now();
   for (let i = 0; i < 10; i++) await trainer.runStep();
   perPhaseMs.without_backward = (performance.now() - t1) / 10;
   
   perPhaseMs.backward = perPhaseMs.full - perPhaseMs.without_backward;
   ```

### Why WGSL Doesn't Have clock()

WGSL has **no built-in shader clock or timer intrinsic**. Unlike CUDA (`clock64()`) or Vulkan (subpass_in or native timestamp extension), WebGPU shaders cannot measure time from within the GPU. This is by design—it prevents timing side-channels. Thus, in-shader timing is impossible; wall-clock ablation is the standard approach.

### Caveats
- **Early return still executes other workgroups**: Each invocation that skips Phase 3 still launches, synchronizes barriers, etc. You measure the stub overhead too.
- **Cache/memory state changes**: Disabling a phase may warm or cool L1/L2 cache differently, slightly biasing successor phases.
- **Variance**: Run multiple iterations (10–50) to smooth out scheduler jitter; report mean ± std dev.

---

## 4. DESIGN SPEC: Adding Timestamp Queries to gpu_harness.ts

### Assumptions
- `bun-webgpu` supports `'timestamp-query'` feature (unconfirmed; test first)
- Metal TBDR quantization is acceptable (100 μs granularity)
- Timestamps are recorded at compute-pass boundaries

### Goal
Measure:
1. **rollout_step** duration (per-board simulation)
2. **gae_scan** duration (advantage/value estimation)
3. **ppo_step per epoch** (PPO training; run for ppoEpochs times)
4. Aggregate timing breakdown across a full runStep

### Buffer and Query Design

```typescript
// In gpu_harness.ts, add to GPUTrainer class:

// Timestamps: [rollout_start, rollout_end, gae_start, gae_end, ppo_epoch_0_start, ..., ppo_epoch_N_end]
// For ppoEpochs=3: capacity = 2 + 2 + 2*(3) = 10 timestamps
private readonly timestampCapacity = 2 + 2 + 2 * CONFIG.ppoEpochs;
private timestampQuerySet!: GPUQuerySet;
private timestampBuffer!: GPUBuffer;        // Query resolve buffer
private timestampReadBuffer!: GPUBuffer;    // Mappable CPU-readable buffer

// Timing results (nanoseconds)
private timestampResults: BigInt64Array | null = null;
```

### Initialization (in init() method)

```typescript
// Request timestamp-query feature
const device = await adapter.requestDevice({
  requiredFeatures: ["shader-f16", "timestamp-query"],
});

// Create query set
this.timestampQuerySet = this.device.createQuerySet({
  type: "timestamp",
  count: this.timestampCapacity,
});

// Query resolve buffer (8 bytes per timestamp)
this.timestampBuffer = this.device.createBuffer({
  size: 8 * this.timestampCapacity,
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
});

// CPU-readable copy buffer
this.timestampReadBuffer = this.device.createBuffer({
  size: 8 * this.timestampCapacity,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
```

### Modified runStep() Structure

```typescript
async runStep() {
  const start = performance.now();
  // ... existing setup code (opponent loading, board init) ...

  const encoder = this.device.createCommandEncoder();

  // === ROLLOUT ===
  let qIdx = 0;
  {
    const pass = encoder.beginComputePass({
      timestampWrites: {
        querySet: this.timestampQuerySet,
        beginningOfPassWriteIndex: qIdx,     // index 0
        endOfPassWriteIndex: qIdx + 1,       // index 1
      }
    });
    pass.setPipeline(this.rolloutPipeline);
    pass.setBindGroup(0, this.bindGroup);
    for (let step = 0; step < CONFIG.maxSteps; step++) {
      this.writeParams(step, useOpponent, eloScale);
      pass.dispatchWorkgroups(CONFIG.numBoards);
      if (step % 20 === 0 && await this.allGamesDone()) break;
    }
    pass.end();
  }
  qIdx += 2;

  // === GAE ===
  {
    const pass = encoder.beginComputePass({
      timestampWrites: {
        querySet: this.timestampQuerySet,
        beginningOfPassWriteIndex: qIdx,     // index 2
        endOfPassWriteIndex: qIdx + 1,       // index 3
      }
    });
    this.writeParams(0, useOpponent, eloScale);
    pass.setPipeline(this.gaePipeline);
    pass.setBindGroup(0, this.gaeBindGroup);
    pass.dispatchWorkgroups(Math.ceil(CONFIG.numBoards / 64));
    pass.end();
  }
  qIdx += 2;

  // === PPO EPOCHS ===
  for (let epoch = 0; epoch < CONFIG.ppoEpochs; epoch++) {
    const pass = encoder.beginComputePass({
      timestampWrites: {
        querySet: this.timestampQuerySet,
        beginningOfPassWriteIndex: qIdx,     // index 4, 6, 8, ...
        endOfPassWriteIndex: qIdx + 1,       // index 5, 7, 9, ...
      }
    });
    this.writeParams(0, useOpponent, eloScale);
    pass.setPipeline(this.ppoPipeline);
    pass.setBindGroup(0, this.ppoBindGroup);
    pass.dispatchWorkgroups(CONFIG.numBoards);
    pass.end();
    qIdx += 2;
  }

  // === RESOLVE TIMESTAMPS ===
  encoder.resolveQuerySet(
    this.timestampQuerySet,
    0,
    this.timestampCapacity,
    this.timestampBuffer,
    0
  );

  // === COPY TO CPU-READABLE BUFFER ===
  encoder.copyBufferToBuffer(
    this.timestampBuffer, 0,
    this.timestampReadBuffer, 0,
    8 * this.timestampCapacity
  );

  this.device.queue.submit([encoder.finish()]);

  // === READ TIMESTAMPS (after GPU sync) ===
  await this.timestampReadBuffer.mapAsync(GPUMapMode.READ);
  const mapped = this.timestampReadBuffer.getMappedRange();
  this.timestampResults = new BigInt64Array(mapped);
  this.timestampReadBuffer.unmap();

  // === PARSE AND COMPUTE DELTAS ===
  const timingsMs = this.computeTimingBreakdown();

  // === EXISTING TRAINING STATS ===
  await this.device.queue.onSubmittedWorkDone();
  const { totalSteps, loss, entropy } = await this.readTrainingStats();
  
  // ... rest of runStep (stats reporting, checkpoint, etc.) ...
}

private computeTimingBreakdown(): {
  rolloutMs: number;
  gaeMs: number;
  ppoPerEpochMs: number[];
  ppoPpoTotalMs: number;
} {
  const ts = this.timestampResults!;
  const nsToms = (ns: bigint) => Number(ns) / 1e6;  // nanoseconds to milliseconds

  const rolloutMs = nsToms(ts[1] - ts[0]);
  const gaeMs = nsToms(ts[3] - ts[2]);
  
  const ppoPerEpochMs: number[] = [];
  let ppoPpoTotalMs = 0;
  for (let e = 0; e < CONFIG.ppoEpochs; e++) {
    const idx0 = 4 + e * 2;
    const idx1 = idx0 + 1;
    const epochMs = nsToms(ts[idx1] - ts[idx0]);
    ppoPerEpochMs.push(epochMs);
    ppoPpoTotalMs += epochMs;
  }

  return { rolloutMs, gaeMs, ppoPerEpochMs, ppoPpoTotalMs };
}
```

### Reporting and Logging

Update onStats callback or logging to include:

```typescript
if (this.onStats) {
  const timings = this.computeTimingBreakdown();
  this.onStats({
    // ... existing fields ...
    timingsMs: {
      rollout: timings.rolloutMs,
      gae: timings.gaeMs,
      ppoEpochs: timings.ppoPerEpochMs,
      ppoPpoTotal: timings.ppoPpoTotalMs,
    }
  });
}
```

### Headless Bench Integration

In bench/headless.ts:

```typescript
for (let i = 0; i < STEPS; i++) {
  const a = performance.now();
  await trainer.runStep();
  const wallClockMs = performance.now() - a;
  perStepMs.push(wallClockMs);

  if (last.timingsMs) {
    console.log(
      `GPU breakdown: rollout=${last.timingsMs.rollout.toFixed(1)}ms ` +
      `gae=${last.timingsMs.gae.toFixed(1)}ms ` +
      `ppo=${last.timingsMs.ppoPpoTotal.toFixed(1)}ms ` +
      `(epochs: ${last.timingsMs.ppoEpochs.map(ms => ms.toFixed(1)).join(', ')}ms)`
    );
  }
}
```

### Caveats and Limitations

1. **Feature Availability**: If `'timestamp-query'` is not available, device creation will throw. Wrap in try-catch and fall back to ablation if needed.
2. **Quantization**: Expect ±100 μs uncertainty on M4/Metal. Use rolling averages (10+ steps) to smooth noise.
3. **Per-Dispatch Granularity**: Timestamps are per-compute-pass, not per-dispatch. To time individual rollout steps, you'd need 600+ separate passes (impractical); ablation is better.
4. **Barrier Sync**: `workgroupBarrier()` in the kernel doesn't insert timestamp markers; all threads in a pass contribute equally to the pass duration.

---

## 5. RECOMMENDATION: Which Methods to Use

### For AlphaGOJS, in priority order:

#### **1. PRIMARY: DAWN_TRACE → Xcode Metal Debugger** ✅✅

**Why**:
- Zero code changes needed
- Captures full GPU timeline (hardware utilization, memory bandwidth, per-kernel duration)
- Native Metal integration on M4 is excellent
- Identifies register pressure, occupancy, and memory bottlenecks

**How**:
```bash
export DAWN_TRACE_FILE_BASE=/tmp/alphagojs_trace
bun bench/headless.ts STEPS=5 B=256
# Load /tmp/alphagojs_trace in Xcode Metal Debugger
```

**Use for**: First-pass profiling, high-level bottleneck identification

---

#### **2. SECONDARY: Ablation Method** ✅

**Why**:
- Isolates individual kernel phases (forward vs. backward in ppo_step)
- No feature availability uncertainties
- Works even if timestamps fail
- Direct cause-and-effect measurement

**How**:
1. Create `fused_ppo_ablate_backward.wgsl` (disable backward pass)
2. Recompile and run: `bun bench/headless.ts STEPS=20`
3. Compare wall-clock time: `baseline - ablated = phase contribution`
4. Repeat for each phase of interest

**Use for**: Pinpointing which ppo_step sub-phase dominates the 96%

---

#### **3. TERTIARY: WebGPU Timestamp Queries** ⚠️ (If time permits)

**Why**:
- Per-pass GPU timing in code
- Useful for repeated profiling runs
- Granular breakdown between rollout, GAE, PPO epochs

**Caveat**: Requires unconfirmed bun-webgpu feature support + Metal TBDR limitations

**Use for**: Only if primary + secondary don't pinpoint the issue. Test feature availability first.

---

### Workflow

1. **Run DAWN_TRACE → Xcode** to get hardware-level timeline. Confirm ppo_step is indeed the bottleneck and examine register usage, L1/L2 hit rates, etc.
2. **Run ablation**: Disable backward pass, recompile, re-run. If time drops from ~96% to ~40%, backward is the culprit.
3. **Ablate sub-phases**: Disable gradient reduction, Adam step, etc., to narrow further.
4. **Optional**: Add timestamp queries to headless.ts if automation/CI profiling is needed.

This combination provides both hardware-level visibility and algorithmic-level attribution.

---

## References

- [WebGPU Timestamp Queries - MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUQuerySet)
- [WebGPU Timing Performance - WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html)
- [Profiling WebGPU with Xcode - Toji.dev](https://toji.dev/webgpu-profiling/xcode.html)
- [Profiling WebGPU with RenderDoc - Toji.dev](https://toji.dev/webgpu-profiling/renderdoc.html)
- [Debugging Dawn - Google Dawn Source Docs](https://dawn.googlesource.com/dawn/+/HEAD/docs/dawn/debugging.md)
- [Timestamp Query Implementation Examples - OmarShehata/webgpu-compute-rasterizer](https://github.com/OmarShehata/webgpu-compute-rasterizer/blob/main/how-to-use-timestamp-queries.md)
- [GPU Timestamp Units on Metal - Issue #1325, gpuweb/gpuweb](https://github.com/gpuweb/gpuweb/issues/1325)
- [Timestamp-Query Unimplementable on TBDR - Issue #2046, gpuweb/gpuweb](https://github.com/gpuweb/gpuweb/issues/2046)
- [bun-webgpu Repository](https://github.com/kommander/bun-webgpu)
- [PIX on Windows - Microsoft Docs](https://learn.microsoft.com/en-us/windows/win32/direct3dtools/pix/articles/general/pix-overview)

