# VCD Timing Bug Test Case

This directory contains a minimal test case demonstrating the VCD input timing bug in GEM.

## The Bug

When input signals change at the **same VCD timestamp** as a clock edge, GEM captures the **old** value instead of the **new** value.

### Root Cause

In the `loom sim` implementation (`src/bin/loom.rs`):

```rust
for pos in std::mem::take(&mut delayed_bit_changes) {
    state[(pos >> 5) as usize] ^= 1u32 << (pos & 31);
}
// ...
// When processing value changes:
delayed_bit_changes.insert(pos);  // ALL changes are delayed!
```

**Problem:** ALL input value changes (including non-clock signals) are inserted into `delayed_bit_changes` and only applied at the NEXT timestamp. This means when a data signal and clock edge occur at the same VCD timestamp, the simulation runs with the OLD data value.

### Expected Verilog Semantics

In Verilog, when a VCD is generated, values at each timestamp represent the final settled state after all delta cycles. If `d` and `clk` both change at the same timestamp, the flip-flop should capture the NEW value of `d`.

## Test Files

- `dff_test.v` - Simple D flip-flop RTL
- `dff_test_tb.v` - Testbench with simultaneous input/clock changes
- `dff_test.vcd` - Golden VCD from iverilog
- `dff_test_synth.gv` - Synthesized netlist for GEM
- `dff_test.gemparts` - GEM partition file
- `gem_output.vcd` - GEM simulation output (contains bug)

## Expected vs Actual Results

### Critical test at t=70000 (d:0→1 simultaneous with clk posedge)
- **Expected (iverilog):** q stays 1 (captures new d=1)
- **Actual (GEM):** q becomes 0 (captures old d=0) **← BUG**

### Critical test at t=100000 (d:1→0 simultaneous with clk posedge)
- **Expected (iverilog):** q becomes 0 (captures new d=0)
- **Actual (GEM):** q becomes 1 (captures old d=1) **← BUG**

## Proposed Fix

Change the input handling to apply non-clock input changes **immediately** instead of delaying them:

1. Only delay clock signal changes (needed for edge detection)
2. Apply data input changes directly to `state` at current timestamp
3. This ensures simulation sees current input values when clock edge triggers

## Running the Test

```bash
# Generate golden VCD
cd tests/timing_test
iverilog -o dff_test.vvp dff_test.v dff_test_tb.v
vvp dff_test.vvp

# Synthesize for GEM
yosys -s synth.tcl

# Create GEM partitions
cargo run -r --bin loom -- map \
    tests/timing_test/dff_test_synth.gv \
    tests/timing_test/dff_test.gemparts

# Run GEM simulation
cargo run -r --features metal --bin loom -- sim \
    tests/timing_test/dff_test_synth.gv \
    tests/timing_test/dff_test.gemparts \
    tests/timing_test/dff_test.vcd \
    tests/timing_test/gem_output.vcd 1

# Compare outputs
diff <(grep "^[01]!" tests/timing_test/dff_test.vcd) \
     <(grep "^[01]!" tests/timing_test/gem_output.vcd)
```
