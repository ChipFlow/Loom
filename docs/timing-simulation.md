# Timing Simulation in GEM

This document explains GEM's boomerang evaluation architecture and how timing simulation with per-gate delays can be implemented efficiently on GPU.

## Background: The Simulation Challenge

GEM simulates And-Inverter Graphs (AIGs) where every node is either:
- A **primary input** (value comes from VCD stimulus)
- An **AND gate** with two inputs (possibly inverted)

Traditional simulation evaluates gates in topological order, which is inherently serial. GPUs excel at massive parallelism - thousands of threads doing the same operation on different data. GEM bridges this gap with the **boomerang** architecture.

## Boomerang Evaluation

### Core Concept

The boomerang structure is a **hierarchical reduction tree** that maps an AIG onto GPU threads. It's called "boomerang" because data flows down the tree during reduction, then results are written back out at various levels - like a boomerang going out and returning.

### Hierarchy Structure

GEM uses `BOOMERANG_NUM_STAGES = 13`, meaning the tree has 2^13 = 8192 leaf positions:

```
Level 0 (inputs):   8192 positions
Level 1:            4096 positions  (8192 / 2)
Level 2:            2048 positions
Level 3:            1024 positions
Level 4:             512 positions
Level 5:             256 positions
Level 6:             128 positions
Level 7:              64 positions
Level 8:              32 positions
Level 9:              16 positions
Level 10:              8 positions
Level 11:              4 positions
Level 12:              2 positions
Level 13 (output):     1 position
```

Each level halves the number of positions by computing AND gates that combine pairs.

### Thread Organization

A GPU block has **256 threads** (`threadIdx.x` = 0..255). Each thread holds a **32-bit word** where each bit represents an independent Boolean signal:

```
Thread 0:   [bit0, bit1, bit2, ... bit31]  = 32 Boolean signals
Thread 1:   [bit0, bit1, bit2, ... bit31]  = 32 Boolean signals
...
Thread 255: [bit0, bit1, bit2, ... bit31]  = 32 Boolean signals
            ─────────────────────────────
            Total: 256 × 32 = 8192 signals per level
```

**Thread position** refers to `threadIdx.x` - which of the 256 threads we're addressing. Each thread position processes 32 signals in parallel using SIMD operations.

### Memory Layout

```cpp
__shared__ u32 shared_metadata[256];   // Partition configuration
__shared__ u32 shared_writeouts[256];  // Output staging area
__shared__ u32 shared_state[256];      // Working state (8192 bits)
```

The `shared_state` array holds the current level's values during reduction.

## The Reduction Process

### Phase 1: Level 0 → Level 1 (hier[0])

Only threads 128-255 are active. Each computes 32 AND gates in parallel:

```cpp
if(threadIdx.x >= 128) {
    u32 hier_input_a = shared_state[threadIdx.x - 128];  // From threads 0-127
    u32 hier_input_b = hier_input;                        // This thread's data

    // 32 AND gates computed simultaneously (one per bit)
    u32 ret = (hier_input_a ^ hier_flag_xora) &
              ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);

    shared_state[threadIdx.x] = ret;
}
```

The `xora`, `xorb`, and `orb` flags encode:
- `xora/xorb`: Input inversions (for AND-inverter graph)
- `orb`: Passthrough mode (when output equals input A, skip the AND)

Visual representation:
```
Before:  [T0][T1]...[T127] [T128][T129]...[T255]
              │                  │
              └───────┬──────────┘
                      │
                   AND gates (128 threads × 32 bits = 4096 gates)
                      │
                      ▼
After:   [----unused----] [T128][T129]...[T255]
                          (128 × 32 = 4096 results)
```

### Phase 2: Levels 1-3 (Shared Memory)

```cpp
for(int hi = 1; hi <= 3; ++hi) {
    int hier_width = 1 << (7 - hi);  // 64, 32, 16
    if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
        u32 hier_input_a = shared_state[threadIdx.x + hier_width];
        u32 hier_input_b = shared_state[threadIdx.x + hier_width * 2];
        u32 ret = (hier_input_a ^ xora) & ((hier_input_b ^ xorb) | orb);
        shared_state[threadIdx.x] = ret;
    }
    __syncthreads();  // Barrier between levels
}
```

Each level activates fewer threads:
- Level 1: threads 64-127 (64 threads → 2048 gates)
- Level 2: threads 32-63 (32 threads → 1024 gates)
- Level 3: threads 16-31 (16 threads → 512 gates)

### Phase 3: Levels 4-7 (Warp Shuffle)

Within a single warp (32 threads), data exchange uses fast shuffle instructions instead of shared memory:

```cpp
if(threadIdx.x < 32) {
    for(int hi = 4; hi <= 7; ++hi) {
        int hier_width = 1 << (7 - hi);  // 8, 4, 2, 1
        u32 hier_input_a = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width);
        u32 hier_input_b = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width * 2);
        if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
            tmp_cur_hi = (hier_input_a ^ xora) & ((hier_input_b ^ xorb) | orb);
        }
    }
}
```

No synchronization needed - warp shuffle is implicitly synchronized.

### Phase 4: Levels 8-12 (Bit Operations)

The final levels operate on bits within a single u32, computed by thread 0 only:

```cpp
if(threadIdx.x == 0) {
    // Level 8: 32 → 16 (operates on upper/lower halves)
    u32 r8 = ((v1 << 16) ^ xora) & ((v1 ^ xorb) | orb) & 0xffff0000;

    // Level 9: 16 → 8
    u32 r9 = ((r8 >> 8) ^ xora) & (((r8 >> 16) ^ xorb) | orb) & 0xff00;

    // Level 10: 8 → 4
    u32 r10 = ((r9 >> 4) ^ xora) & (((r9 >> 8) ^ xorb) | orb) & 0xf0;

    // Level 11: 4 → 2
    u32 r11 = ((r10 >> 2) ^ xora) & (((r10 >> 4) ^ xorb) | orb) & 0b1100;

    // Level 12: 2 → 1
    u32 r12 = ((r11 >> 1) ^ xora) & (((r11 >> 2) ^ xorb) | orb) & 0b10;

    tmp_cur_hi = r8 | r9 | r10 | r11 | r12;
}
```

### Write-Outs

Results are captured at various levels (not just the final output) and written to global memory:

```cpp
if((writeout_hook_i >> 8) == bs_i) {
    shared_writeouts[threadIdx.x] = shared_state[writeout_hook_i & 255];
}
```

This is the "return" part of the boomerang - results flow back from intermediate levels.

## Timing Simulation Approaches

### Approach Comparison

| Approach | Parallelism | Memory | Accuracy | GPU Fit |
|----------|-------------|--------|----------|---------|
| Event-driven | Poor (serial queue) | Low | Exact | Bad |
| Time-wheel | Medium | High | Configurable | Medium |
| **Levelized** | **Excellent** | **Low** | **Conservative** | **Best** |
| Oblivious | Maximum | Very High | Exact | Wasteful |

### Recommended: Levelized with Delay Accumulation

This approach piggybacks on the existing boomerang structure with minimal changes.

#### Data Structure Addition

```cpp
// Add to shared memory (256 bytes additional)
__shared__ u8 shared_arrival[256];  // One arrival time per thread position
```

Each thread position stores a single 8-bit arrival time representing the **maximum arrival across all 32 bits** in that position.

#### Modified AND Gate Evaluation

```cpp
// Current (value only):
u32 ret = (hier_input_a ^ xora) & ((hier_input_b ^ xorb) | orb);
shared_state[threadIdx.x] = ret;

// With timing (add ~4 instructions):
u32 ret = (hier_input_a ^ xora) & ((hier_input_b ^ xorb) | orb);
shared_state[threadIdx.x] = ret;

u8 arr_a = shared_arrival[threadIdx.x - offset_a];
u8 arr_b = shared_arrival[threadIdx.x - offset_b];
u8 arr_ret = min(max(arr_a, arr_b) + GATE_DELAY, 255);  // Saturating add
shared_arrival[threadIdx.x] = arr_ret;
```

#### Complexity Analysis

- **Same number of kernel launches** as zero-delay simulation
- **O(levels × cycles)** - identical to current
- **~256 bytes additional shared memory** per partition
- **Estimated 10-20% performance overhead**

## The Approximation Trade-off

### What We Track

One arrival time per thread position (256 values) instead of per signal (8192 values).

### Implications

If thread position 50 contains signals A, B, C with different true arrivals:
```
Signal A: 15ps (shortest path)
Signal B: 23ps (longest path)
Signal C: 8ps  (medium path)
```

We store only: `arrival[50] = 23ps` (the maximum).

### Why This Works

1. **Conservative**: We might report false violations, but never miss real ones
2. **Correlated signals**: Signals at the same thread position are often topologically nearby with similar timing
3. **Endpoint focus**: We ultimately only care about arrivals at DFF D inputs

### When Full Accuracy is Needed

For bit-accurate timing, you would need:
```cpp
// 8KB additional shared memory (may exceed limits)
__shared__ u8 shared_arrival[256][32];  // Per-bit arrivals
```

This is feasible but significantly increases memory pressure and computation.

## Implementation Phases

### Phase 1: CPU Timing Analysis (Completed)

- Liberty parser for delay extraction
- Static timing analysis on AIG
- CPU reference simulation with delays
- Timing violation detection

### Phase 2: Hybrid GPU+CPU (Completed)

- GPU performs zero-delay value simulation
- CPU performs timing analysis on results
- Validates infrastructure without kernel changes

### Phase 3: GPU Arrival Tracking (Future)

- Add `shared_arrival[256]` to kernel
- Track arrivals during boomerang reduction
- Report max arrival at cycle boundaries
- Check violations at DFF endpoints

### Phase 4: Full Integration (Future)

- Timing violation events via event buffer
- Per-cycle timing reports
- Integration with output VCD

## Delay Data Encoding

### Script Format

The existing boomerang section has padding that can store delay data:

```
Current format per thread per stage:
  [xora: u32]
  [xorb: u32]
  [orb:  u32]
  [padding: u32]  ← Can store delay here
```

### PackedDelay Structure

```rust
#[repr(C)]
pub struct PackedDelay {
    pub rise_ps: u16,  // Rising edge delay in picoseconds
    pub fall_ps: u16,  // Falling edge delay in picoseconds
}
```

For simplified timing, a single uniform delay constant can be used instead of per-gate delays.

## Timing Violation Detection

### At Each Cycle Boundary

```cpp
// After boomerang completes, before next cycle
if(threadIdx.x < num_dffs) {
    u8 data_arrival = get_arrival_at_dff_d_input(dff_id);
    i8 setup_slack = CLOCK_PERIOD - SETUP_TIME - data_arrival;

    if(setup_slack < 0) {
        // Report via event buffer
        write_event(SETUP_VIOLATION, cycle, dff_id, setup_slack);
    }
}
```

### Event Buffer Integration

```rust
pub enum EventType {
    Stop = 0,
    Finish = 1,
    Display = 2,
    AssertFail = 3,
    SetupViolation = 4,   // Timing events
    HoldViolation = 5,
}
```

## Timing-Aware Bit Packing

### The Problem

Each thread position holds 32 signals packed into a u32. When tracking timing with one arrival value per thread position, we approximate all 32 signals as having the same arrival time (the maximum).

This approximation is accurate when signals in the same thread have similar timing. But the default placement algorithm uses **first-fit** for bit assignment:

```rust
// Default: first available slot
for i in 0..hier[selected_level].len() {
    if hier[selected_level][i] == usize::MAX {
        slot_at_level = i;  // First-fit, not timing-aware
        break;
    }
}
```

This can result in signals with very different timing sharing a thread:

```
Thread 50 (accidental grouping):
  bit 0: level 5,  ~5ps arrival
  bit 1: level 12, ~12ps arrival  ← 7ps difference!
  bit 2: level 6,  ~6ps arrival

Thread 50 (timing-aware grouping):
  bit 0: level 5, ~5ps arrival
  bit 1: level 5, ~5ps arrival    ← similar timing
  bit 2: level 6, ~6ps arrival
```

### Current Timing Correlation

The placement algorithm already computes **logic levels**:

```rust
// Level = max(level of inputs) + 1
level[node] = max(level[input_a], level[input_b]) + 1;
```

Logic level correlates with timing (more levels = more gate delays), but signals at the same level can still have different actual delays due to:
- Different gate types (AND2_00_0 vs AND2_11_1)
- Different wire loads
- Path reconvergence

### Solution: Sort by Timing Before Packing

Before assigning bit positions, sort signals by their estimated arrival time:

```rust
// Collect nodes at this level
let mut nodes_to_place: Vec<_> = candidates
    .filter(|n| level[n] == selected_level)
    .collect();

// Sort by arrival time (level as proxy, or actual timing if available)
nodes_to_place.sort_by_key(|n| arrival_estimate[n]);

// Place in sorted order - similar timing ends up in same thread
for (slot, node) in nodes_to_place.iter().enumerate() {
    place_bit(..., slot, *node);
}
```

### Alternative Approaches

| Approach | Complexity | Effectiveness | When to Use |
|----------|------------|---------------|-------------|
| **Sort by timing** | Low | Good | Default choice |
| Timing-aware partitioning | High | Best | Large designs |
| Post-placement swapping | Medium | Good | Fine-tuning |
| Timing bands | Low | Moderate | Simple heuristic |

### Timing Bands

Group signals into arrival time bands:

```
Band 0: 0-10ps   → Threads 0-63
Band 1: 10-20ps  → Threads 64-127
Band 2: 20-30ps  → Threads 128-191
Band 3: 30+ps    → Threads 192-255
```

### Measuring Packing Quality

Diagnostic to measure timing variance per thread:

```rust
fn analyze_timing_packing(hier: &Hierarchy, arrivals: &[u64]) {
    for thread in 0..256 {
        let times: Vec<_> = get_bits_in_thread(hier, thread)
            .map(|b| arrivals[b])
            .collect();

        let range = times.iter().max() - times.iter().min();
        let variance = compute_variance(&times);

        if range > threshold {
            warn!("Thread {} has {}ps timing spread", thread, range);
        }
    }
}
```

### Impact on Approximation Accuracy

With timing-aware packing:
- **Reduced false positives**: Fewer spurious timing violations from max approximation
- **Tighter bounds**: Per-thread arrival closer to actual signal arrivals
- **Better critical path identification**: Max arrival more accurately reflects true critical path

## Performance Expectations

| Metric | Zero-Delay | With Timing |
|--------|------------|-------------|
| Kernel launches | N | N |
| Shared memory | 3KB | 3.25KB |
| Registers | ~32 | ~36 |
| Instructions/gate | ~5 | ~9 |
| **Estimated overhead** | - | **15-25%** |

The overhead is modest because:
1. Timing operations are simple (max, add)
2. Memory access pattern is identical
3. No additional synchronization needed
4. Same parallelism structure

## References

- `src/pe.rs` - Partition executor and boomerang stage construction
- `csrc/kernel_v1_impl.cuh` - GPU kernel implementation
- `src/flatten.rs` - Script generation with timing data
- `src/event_buffer.rs` - GPU→CPU event communication
- `src/liberty_parser.rs` - Timing library parsing
