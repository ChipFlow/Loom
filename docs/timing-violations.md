# Timing Violation Detection

Guide to enabling, reading, and debugging setup/hold timing violations in GEM.

## Overview

Setup and hold violations occur when data arrives too late (setup) or too early (hold) relative to the clock edge at a flip-flop. GEM checks for these violations during GPU simulation by tracking **arrival times** — the accumulated gate delay from primary inputs or DFF outputs through combinational logic to the next DFF data input.

**Approximation model**: GEM tracks one arrival time per 32-signal group (one GPU thread position). The arrival is the **maximum** across all 32 signals in the group. This is conservative: it may over-report violations but will never miss a real one. See [Reducing False Positives](#the-approximation-caveat) for details.

## Enabling Timing Checks

### Prerequisites

1. **SDF file** with back-annotated delays from your place-and-route tool
2. **Gate-level netlist** synthesized to `aigpdk.lib` cells

### Step-by-step

1. **Generate SDF** from your P&R tool (or use `scripts/generate_sdf.py` for test designs):

   ```bash
   # Example: OpenROAD flow output
   ls my_build/6_final.sdf
   ```

2. **Run the simulator** with `--sdf` and a clock period:

   **Metal (macOS)**:
   ```bash
   cargo run -r --features metal --bin loom -- sim \
       design.gv input.vcd output.vcd 1 \
       --sdf design.sdf \
       --sdf-corner typ
   ```

   **CUDA (NVIDIA)**:
   ```bash
   cargo run -r --features cuda --bin loom -- sim \
       design.gv input.vcd output.vcd 8 \
       --sdf design.sdf \
       --sdf-corner typ \
       --enable-timing \
       --timing-clock-period 1200
   ```

   **cosim (co-simulation)**:
   ```bash
   cargo run -r --features metal --bin loom -- cosim \
       design.gv \
       --config testbench.json \
       --sdf design.sdf \
       --sdf-corner typ
   ```

### CLI Flags Reference

| Flag | Binary | Description |
|------|--------|-------------|
| `--sdf <path>` | all | Path to SDF file with back-annotated delays |
| `--sdf-corner <min\|typ\|max>` | all | Which SDF corner to use (default: `typ`) |
| `--sdf-debug` | all | Print unmatched SDF instances for debugging |
| `--enable-timing` | `loom sim` | Enable timing analysis (arrival + violation checks) |
| `--timing-clock-period <ps>` | `loom sim` | Clock period in picoseconds (default: 1000) |
| `--timing-report-violations` | `loom sim` | Report all violations, not just summary |
| `--liberty <path>` | `loom sim` | Liberty library for timing data (optional, falls back to AIGPDK defaults) |

### Example: inv_chain_pnr Test Case

```bash
# Run with SDF timing
cargo run -r --features metal --bin loom -- sim \
    tests/timing_test/inv_chain_pnr/6_final.v \
    tests/timing_test/inv_chain_pnr/input.vcd \
    tests/timing_test/inv_chain_pnr/output.vcd 1 \
    --sdf tests/timing_test/inv_chain_pnr/6_final.sdf
```

## Reading Violation Reports

### Setup Violation Format

```
[cycle 42] SETUP VIOLATION: word 5 arrival=900ps setup=200ps slack=-100ps
```

| Field | Meaning |
|-------|---------|
| **cycle** | Simulation cycle where the violation occurred |
| **word** | State word index — identifies a group of 32 DFF data inputs |
| **arrival** | Maximum accumulated gate delay to this word's signals (picoseconds) |
| **setup** | DFF setup time constraint from SDF/Liberty (picoseconds) |
| **slack** | `clock_period - arrival - setup`. Negative = violation amount |

### Hold Violation Format

```
[cycle 11] HOLD VIOLATION: word 3 arrival=10ps hold=50ps slack=-40ps
```

| Field | Meaning |
|-------|---------|
| **cycle** | Simulation cycle where the violation occurred |
| **word** | State word index |
| **arrival** | Accumulated gate delay to this word's signals (picoseconds) |
| **hold** | DFF hold time constraint from SDF/Liberty (picoseconds) |
| **slack** | `arrival - hold`. Negative = violation amount |

### Summary Statistics

At the end of simulation, GEM prints totals:

```
Simulation complete: 1000 cycles, 5 setup violations, 0 hold violations
```

## Tracing Violations to Source Signals

When you see a violation on a specific word, follow this workflow to identify the offending signals and their logic cone.

### 1. Get the Word Index

From the log: `word 5` means state word index 5.

### 2. Map Word to DFF Signals

Each word covers 32 bits of state. The DFFs in that word have `data_state_pos / 32 == word_index`. To find which DFFs:

- Look at the `dff_constraints` entries in the `FlattenedScriptV1`:
  ```
  dff_constraints entries where data_state_pos / 32 == 5
  → cell_id values → netlist cell names
  ```

- In `gpu_sim`, violations are logged with word IDs that map directly to the `output_map` positions. Each word covers bit positions `word * 32` through `word * 32 + 31`.

### 3. Trace Backwards with netlist_graph

Use the [netlist_graph tool](../scripts/netlist_graph/) to trace the combinational logic cone feeding the DFF:

```bash
cd scripts/netlist_graph

# Find the DFF data input driver chain
uv run netlist-graph drivers design.v "dff_name.D" -d 10

# Search for DFFs matching a pattern
uv run netlist-graph search design.v "dff_out*"
```

### 4. Detailed CPU Timing Analysis

For per-signal accuracy (no 32-signal approximation), use `timing_sim_cpu`:

```bash
# Generate a watchlist for signals of interest
cd scripts/netlist_graph
uv run netlist-graph watchlist design.v watch.json dff_name signal1 signal2

# Run CPU timing simulation with per-signal tracing
cargo run -r --bin timing_sim_cpu -- design.v input.vcd \
    --liberty sky130.lib --clock-period 1200 \
    --watchlist watch.json --trace-output trace.csv
```

The CSV trace shows per-cycle arrival times for each watched signal, allowing you to pinpoint exactly which path is critical.

## The Approximation Caveat

GEM tracks **one arrival time per 32-signal group** (one GPU thread position). The tracked value is the maximum arrival across all 32 signals in that thread. This means:

- **Conservative**: If any signal in the group has a long path, the arrival for the entire group reflects that worst case. Violations may be reported for signals that individually meet timing.
- **Never misses real violations**: A real violation always results in a reported violation (the max is >= any individual signal's arrival).

### Reducing False Positives

If a violation is reported but you suspect it's a false positive from the approximation:

1. **Use `timing_sim_cpu`** for per-signal accuracy (see [Detailed CPU Timing Analysis](#4-detailed-cpu-timing-analysis) above).
2. **Timing-aware bit packing** groups signals with similar arrival times into the same thread, reducing the approximation error. See `docs/timing-simulation.md` § "Timing-Aware Bit Packing" for details.

## Common Scenarios

**Setup violations on many words, same cycle**: The clock period is likely too tight for the design. The combinational logic depth exceeds what can settle in one clock period. Try increasing the clock period.

**Setup violation on a single word**: A critical path through one specific logic cone. Use `netlist_graph drivers` to trace the path and identify the bottleneck.

**Hold violation**: Rare with SKY130 process (negative hold times clamp to 0 in the SDF). If seen, the design likely has minimum-delay paths that are too short. Check for direct connections between DFF outputs and nearby DFF inputs with minimal combinational logic.

**Violations only on first cycle**: The `arrival > 0` guard in the GPU kernel skips setup checks when arrival is zero (meaning no data has propagated through combinational logic yet). If you see violations on cycle 0, they are hold violations — setup violations on cycle 0 are suppressed by design.
