# Selective X-Propagation for Loom

## Summary

This document proposes adding **selective unknown-value (X) propagation** to Loom's
gate-level simulator. Rather than uniformly upgrading the entire simulator to
four-state logic (2x storage, ~2-3x ALU cost), we use **static analysis at
compile time** to identify which signals can carry X values, and only apply
X-aware simulation to the affected partitions. The rest of the design continues
to run with the existing fast two-state kernel.

## Motivation

Loom currently simulates in pure two-state Boolean logic (0/1). All DFFs and
SRAMs start at 0. This is fast but has two significant drawbacks:

1. **Undetected initialisation bugs**: If a design reads a register before it
   has been written through a proper reset sequence, the simulator silently
   returns 0 instead of flagging the value as unknown. Real hardware would
   produce an arbitrary bit pattern.

2. **SRAM initialisation masking**: SRAMs read as all-zeros before being
   written. A design that depends on SRAM initialisation may appear to work
   in simulation but fail on silicon.

3. **No RTL/gate-level mismatch detection**: RTL simulators (Icarus, VCS,
   Questa) propagate X values that expose these bugs. When comparing Loom's
   gate-level results against an RTL reference, false mismatches arise because
   Loom resolves unknowns to zero.

Naively upgrading the entire simulator to four-state would halve throughput
(double storage per signal, double ALU per gate). The key insight is that in a
well-designed SoC after reset, typically **<5% of signals** are genuinely
X-capable (uninitialised memories, clock-domain crossings, registers without
reset). We should only pay the overhead where it matters.

## Background: X in And-Inverter Graphs

Loom's core IR is an And-Inverter Graph (AIG). Every gate is an AND with
optional input inversions. The boomerang reduction tree computes:

```
ret = (a ^ xora) & ((b ^ xorb) | orb)
```

Where `xora`/`xorb` encode inversions and `orb` encodes pass-through (when
`orb = 0xFFFFFFFF`, input `b` is forced to 1, making the gate a buffer for `a`).

The AND gate has a favourable property for X propagation:

| a | b | a AND b |
|---|---|---------|
| 0 | 0 | 0       |
| 0 | 1 | 0       |
| 0 | X | **0**   |
| 1 | 0 | 0       |
| 1 | 1 | 1       |
| 1 | X | **X**   |
| X | 0 | **0**   |
| X | 1 | **X**   |
| X | X | **X**   |

A known-zero on either input forces the output to known-zero regardless of X
on the other input. This means X does not spread as aggressively through AND
logic as it would through XOR or MUX logic. AIG-based designs have a natural
tendency to "absorb" X values at AND gates with known-zero inputs.

NOT (inversion) simply preserves the X mask: `NOT(X) = X`, `NOT(0) = 1`,
`NOT(1) = 0`.

## Design

### Phase 1: Static X-Source Analysis (Compile Time)

At AIG construction time, identify all **X sources** -- signals whose initial
value is unknown:

1. **DFF Q outputs**: Every DFF output is an X source at cycle 0. In real
   hardware, flip-flop power-on state is indeterminate.

2. **SRAM read ports**: All 32 data-output pins of each `RAMBlock` are X
   sources. Memory contents are undefined until written.

3. **Undriven primary inputs**: If a primary input is not driven by the
   testbench VCD, it should be marked X. (Currently Loom warns about missing
   PI signals; this would upgrade that warning to X propagation.)

These are identified directly from `AIG.drivers` (variant `DFF(_)` and
`SRAM(_)`) and from `AIG.dffs` / `AIG.srams`.

### Phase 2: Forward Cone Computation (Compile Time)

Compute the **forward cone of influence** from all X sources through the AIG.
Since AIG pins are guaranteed to be in topological order, this is a single
linear-time forward pass:

```
x_capable = BitVec::zeros(num_aigpins + 1)

// Mark X sources
for each DFF:
    x_capable.set(dff.q)
for each SRAM:
    for each read data pin:
        x_capable.set(pin)

// Forward propagate (pins are in topological order)
for aigpin in 1..=num_aigpins:
    if let AndGate(a_iv, b_iv) = drivers[aigpin]:
        a = a_iv >> 1   // strip inversion bit
        b = b_iv >> 1
        if x_capable[a] || x_capable[b]:
            x_capable.set(aigpin)
```

This is O(V + E) on the AIG -- negligible compared to partitioning.

**Sequential propagation**: X-capability also propagates through DFF feedback
loops. If a DFF's D input is X-capable, its Q output remains X-capable (even
after the first clock edge, because it may have captured an X value). The
analysis iterates until a fixpoint:

```
loop:
    changed = false
    for each DFF:
        d_pin = dff.d_iv >> 1
        if x_capable[d_pin] && !x_capable[dff.q]:
            x_capable.set(dff.q)
            changed = true
    if !changed:
        break
    // Re-run forward cone from newly-marked DFF Q outputs
```

In practice this converges in 1-2 iterations because most feedback loops
go through DFFs that are already marked.

### Phase 3: Partition Classification

After partitioning (mt-kahypar), classify each partition:

- **X-capable**: Contains at least one X-capable aigpin, OR reads input state
  from an X-capable partition's output. Run with the X-aware kernel variant.

- **X-free**: All signals are provably not-X. Run with the existing fast
  two-state kernel.

The classification must account for inter-partition communication:

```
loop:
    changed = false
    for each partition P:
        if P is already X-capable:
            continue
        for each global-read in P's script:
            source_word = identify source partition and state word
            if source partition is X-capable:
                mark P as X-capable
                changed = true
                break
    if !changed:
        break
```

### Phase 4: X-Mask Representation

For X-capable partitions, each signal carries a **sideband X mask** alongside
its value:

- `v` (value bit): The Boolean value. When `x = 1`, this is a "best guess"
  (we use 0 by convention, matching Loom's current behaviour for
  backwards-compatible output).
- `x` (X-mask bit): 1 = unknown, 0 = known.

This doubles the per-signal storage within X-capable partitions only.

#### State Buffer Layout

```
State buffer (current):
  [word 0] [word 1] ... [word N-1]   ← N u32 words, 32 signals each

State buffer (with X sideband):
  [word 0] [word 1] ... [word N-1]   ← value words (unchanged)
  [word N] [word N+1] ... [word 2N-1] ← X-mask words (new, same layout)
```

X-free partitions only read/write the value section. X-capable partitions
read/write both sections. The sideband occupies the same state-buffer address
space with a fixed offset of `state_size` words.

### Phase 5: X-Aware Boomerang Kernel

The existing boomerang gate computation:

```
ret_v = (a ^ xora) & ((b ^ xorb) | orb)
```

The X-aware version computes in parallel:

```
// Effective inputs after inversion and OR-bypass
a_eff   = a_v ^ xora
b_eff   = (b_v ^ xorb) | orb
b_eff_x = b_x & ~orb        // OR-bypass forces bits to known-1

// Value: same as before (X bits treated as 0 in value lane)
ret_v = a_eff & b_eff

// X mask: result is X when both inputs are not-known-zero AND at
// least one input is X
//
//   known_zero_a = ~a_x & ~a_eff  (not X, and effective value is 0)
//   known_zero_b = ~b_eff_x & ~b_eff
//   ret_x = (a_x | b_eff_x) & ~known_zero_a & ~known_zero_b
//
// Expanded:
ret_x = (a_x | b_eff_x) & (a_eff | a_x) & (b_eff | b_eff_x)
```

**Verification of the X-mask formula:**

| a | b | orb=0 | a_eff(xora=0) | b_eff(xorb=0) | b_eff_x | ret_v | ret_x | Expected |
|---|---|-------|------|------|---------|-------|-------|----------|
| 0 | 0 |  0    |  0   |  0   |   0     |   0   |   0   |  0       |
| 0 | X |  0    |  0   |  0   |   1     |   0   |   0   |  0 (0&X=0) |
| 1 | X |  0    |  1   |  0   |   1     |   0   |   1   |  X (1&X=X) |
| X | 0 |  0    |  0   |  0   |   0     |   0   |   0   |  0 (X&0=0) |
| X | 1 |  0    |  0   |  1   |   0     |   0   |   1   |  X (X&1=X) |
| X | X |  0    |  0   |  0   |   1     |   0   |   1   |  X (X&X=X) |
| X | * |  1    |  0   |  1   |   0     |   0   |   1   |  X (pass-thru of X a) |
| 0 | * |  1    |  0   |  1   |   0     |   0   |   0   |  0 (pass-thru of 0 a) |
| 1 | * |  1    |  1   |  1   |   0     |   1   |   0   |  1 (pass-thru of 1 a) |

Wait -- row for `a=X, b=*, orb=1` (pass-through): `a_x=1, a_eff=0, b_eff=1, b_eff_x=0`.
`ret_x = (1|0) & (0|1) & (1|0) = 1 & 1 & 1 = 1`. Correct.

The X-mask computation adds **5 extra bitwise operations** per boomerang level
(2 OR, 2 AND, 1 AND-NOT -- plus loading the X mask values). This is roughly
1.5-1.6x the CPU ALU work per gate, but only within X-capable partitions.

#### GPU Resource Impact

Per X-capable partition:

| Resource | Current | With X-mask | Delta |
|----------|---------|-------------|-------|
| Shared state (threadgroup mem) | 256 x u32 = 1 KB | 512 x u32 = 2 KB | +1 KB |
| Shared arrival (threadgroup mem) | 256 x u16 = 512 B | unchanged | 0 |
| Global state buffer | N words | 2N words | +N words |
| ALU per boomerang stage | ~3 ops/thread | ~7 ops/thread | ~2.3x |

Metal threadgroup memory limit is 32 KB; current usage is ~4 KB. The extra
1 KB is well within budget.

### Phase 6: Boundary Protocol

When signals cross from an X-capable partition to another partition:

- **X-capable -> X-capable**: Both value and X-mask words are communicated
  through the state buffer. The reading partition loads both.

- **X-capable -> X-free**: At the boundary, we **assert** that no X values
  cross. If an X-mask bit is set for a signal read by an X-free partition,
  this is a simulation error (the design has an X reaching a region we
  statically proved should be X-free). Report it and optionally halt.

- **X-free -> X-capable**: The reading partition loads the value word normally
  and treats the X-mask as all-zero for those inputs. No overhead.

### Phase 7: Dynamic X Narrowing (Optional Enhancement)

After the design's reset sequence completes, most DFFs will hold known values
and the X-mask will be all-zeros across most of the state. The simulator can
detect this:

```
every K cycles (e.g., K = 1000):
    for each X-capable partition:
        if all X-mask words in this partition's state are zero:
            switch partition to fast two-state kernel
```

This gives the best of both worlds: full X-propagation during initialisation
(when it matters most), then automatic fallback to maximum throughput once
the design is in steady state.

For designs with a clear reset phase, this means the performance overhead
of X-propagation is confined to the first few thousand cycles.

## Expected Performance Impact

### Compile Time

The static analysis adds:
- X-source identification: O(|DFFs| + |SRAMs|) -- negligible
- Forward cone computation: O(|AIG pins|) -- one linear pass
- Partition classification: O(|partitions| x |global reads|) -- negligible
- Fixpoint iteration: 1-2 rounds of the above

Total: well under 1% of the partitioning time.

### Simulation Time

For a typical SoC with reset:

| Phase | X-capable partitions | Overhead |
|-------|---------------------|----------|
| Before reset (cycles 0-100) | ~30-50% of partitions | ~1.5-2x slowdown |
| During reset (cycles 100-1000) | Shrinking as resets propagate | Decreasing |
| After reset (steady state) | <5% of partitions (uninitialised SRAM, CDC) | <10% overhead |
| After dynamic narrowing | ~0% | ~0% overhead |

The overall impact on a full simulation run is estimated at **5-15% throughput
reduction** compared to current two-state, in exchange for catching
initialisation bugs that would otherwise escape to silicon.

## Implementation Status

All stages are implemented. Enable with `--xprop` on `loom sim`.

### Stage 1: Static X-Source Analysis (`src/aig.rs`)

- `compute_x_sources() -> Vec<bool>` -- identifies DFF Q outputs and SRAM read data ports
- `compute_x_capable_pins() -> (Vec<bool>, XPropStats)` -- forward cone + fixpoint iteration

### Stage 2: Partition Classification (`src/flatten.rs`)

- `FlattenedScriptV1` fields: `xprop_enabled`, `partition_x_capable`, `xprop_state_offset`
- `effective_state_size()` returns `2 * reg_io_state_size` when xprop enabled
- Metadata words 8 (is_x_capable) and 9 (xmask_state_offset) encode per-partition xprop info
- X-propagation is configured at simulation startup, no separate mapping step needed

### Stage 3: X-Aware CPU Reference Kernel (`src/sim/cpu_reference.rs`)

- `simulate_block_v1_xprop()` -- dual-lane value + X-mask computation
- `sanity_check_cpu_xprop()` -- CPU vs GPU comparison for both lanes
- Non-X-capable partitions delegate to `simulate_block_v1` (zero overhead)

### Stage 4: CLI and VCD Integration (`src/bin/loom.rs`, `src/sim/vcd_io.rs`)

- `loom sim --xprop` runs static analysis, logs X-capable pin/partition stats, and enables X-aware simulation with doubled state buffer
- `write_output_vcd_xprop()` emits `Value::X` for X-masked primary outputs
- `expand_states_for_xprop()` / `split_xprop_states()` for state buffer management

### Stage 5: GPU Kernels (`csrc/kernel_v1.metal`, `csrc/kernel_v1_impl.cuh`)

- Dual-lane X-mask tracking through all kernel phases (global read, boomerang, SRAM, DFF writeout)
- `is_x_capable` branch is uniform per threadgroup (zero warp/SIMD divergence)
- Shared memory: `shared_state_x[256]` + `shared_writeouts_x[256]` (+2KB, within 32KB limit)
- SRAM X-mask shadow via `sram_xmask` buffer (buffer slot 7 on Metal)

### Stage 6: Diagnostics

- Compile-time: X-source count, X-capable pin %, X-capable partition %
- Runtime: first-cycle-X-free detection, final-cycle X warning
- CPU sanity check uses xprop variant when enabled

### Stage 7: Benchmarks (`benches/xprop.rs`)

- Criterion micro-benchmarks: two_state vs xprop_xfree vs xprop_xcapable
- X-free partitions: zero overhead confirmed
- X-capable partitions: ~1.5-1.6x CPU overhead (well within 2x budget)

### Dynamic Narrowing (Future Enhancement)

- Periodic X-mask scan on CPU between GPU batches
- Partition kernel hot-swapping from X-aware to fast mode
- Statistics reporting: "X cleared after N cycles"

## Prior Art

- **[Mixed 2-4 State Simulation with VCS (Chaudhry et al., 1997)](https://ieeexplore.ieee.org/document/588537)**: Proved that mixed two-state / four-state simulation is viable within a single Verilog simulator, validating the core approach.

- **[A Two-State Methodology for RTL Logic Simulation (1999)](https://ieeexplore.ieee.org/document/782029)**: Eliminated X entirely using random two-state initialisation, arguing it catches more bugs than X-optimistic RTL simulation. Their technique for handling Z-state boundaries is relevant to our boundary protocol.

- **[Essent (Beamer, 2020-2021)](https://scottbeamer.net/pubs/beamer-dac2020.pdf)**: Demonstrated that partitioning a compiled simulator by signal activity yields significant speedups. Their activity-proportional execution is analogous to our dynamic X narrowing -- skip work for partitions where nothing interesting is happening.

- **[Synopsys VCS T-Prop](https://semiengineering.com/simulation-with-taint-propagation-for-security-verification/)**: Taint propagation in VCS uses a parallel sideband bit per signal to track data flow for security verification. Our X-mask is architecturally identical -- a sideband bit that propagates alongside the value using different rules.

- **[Chris Drake, "Improving Verilog Four State Logic" (2024)](https://cjdrake.substack.com/p/improving-verilog-four-state-logic)**: Argues that Verilog conflates "uninitialised" and "don't care" uses of X, and proposes splitting them. Our approach naturally supports this: the X-mask tracks only uninitialised/unknown values, not synthesis don't-cares (which are already resolved by the synthesis tool before Loom sees the netlist).

## What This Does NOT Address

- **Z (high-impedance)**: Loom does not support tri-state buses and this
  proposal does not add support. Z would require a third state bit and bus
  resolution logic.

- **X-optimism in RTL control flow**: Since Loom operates on gate-level
  netlists (post-synthesis), there are no `if`/`case` statements to
  be X-optimistic about. Gate-level X propagation through AND/OR truth tables
  is naturally correct (though potentially X-pessimistic -- see below).

- **X-pessimism reduction**: Gate-level simulation is inherently X-pessimistic
  in some cases (e.g., `a XOR a` should be 0 even if `a` is X, but
  standard gate-level propagation gives X). Since AIGs decompose XOR into
  AND/NOT, this pessimism is present. Addressing this would require symbolic
  analysis or reconvergence detection, which is out of scope.

- **Strength modelling**: No weak/strong drive strength tracking. All known
  values are "strong."

## Design Decisions

1. **SRAM X granularity**: Conservative whole-SRAM approach -- all reads return
   X until any write occurs. This is pessimistic but correct and simple.
   Per-address tracking (requiring `8192 x 32` bits of shadow state per SRAM
   block) is a potential follow-up for reduced pessimism.

2. **Reset-aware analysis**: Skipped for v1 -- all DFF Q outputs start as X
   regardless of whether they have async reset. Identifying reset nets is
   fragile (varies by synthesis tool and coding style), and the fixpoint
   iteration naturally resolves DFFs that become non-X after reset propagates.

3. **VCD X output**: Yes -- `write_output_vcd_xprop()` emits `Value::X` when
   the X-mask bit is set for a primary output. This is the primary user-visible
   benefit. The `vcd-ng` crate already supports `Value::X`. Downstream tools
   that only handle two-state VCD would need updating, but Verilog-standard
   four-state VCD is widely supported.

4. **Partition-level granularity**: X-awareness is applied at partition
   granularity (entire partition runs X-aware kernel or not). This provides
   better steady-state performance than per-signal tracking when most
   partitions are X-free (~95% typical), at the cost of slightly more
   pessimism at partition boundaries.

5. **Runtime CLI flag**: X-propagation is controlled by `--xprop` on `loom sim`,
   rather than a compile-time feature flag. No new Cargo dependencies are needed.

6. **State buffer layout**: When xprop is enabled, the state buffer doubles.
   Value words occupy `[0 .. reg_io_state_size)` and X-mask words occupy
   `[reg_io_state_size .. 2*reg_io_state_size)` per cycle. Per-partition
   metadata word 8 stores the `is_x_capable` flag, and word 9 stores the
   `xmask_state_offset` (equal to `reg_io_state_size` for X-capable
   partitions). X-free partitions ignore both fields and run unchanged.
