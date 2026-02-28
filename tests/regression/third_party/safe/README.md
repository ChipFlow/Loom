# Safe Module Test Case

**Source**: [sva-playground](https://github.com/bluecmd/sva-playground) by Christian Svensson
**License**: MIT License (see ../LICENSE_sva-playground)
**Original Files**: `backdoor/safe.sv` and `backdoor/tb.sv`

## Description

A PIN-based safe cracker FSM with immediate assertions to verify no backdoor codes exist. Tests assertion checking in state machine designs with multiple states and complex transition logic.

## Features

- 9-state FSM (PIN_0-3, SECRET_PIN_1-3, LOCKOUT, UNLOCKED)
- Correct password: `c0de` (transitions through PIN states)
- Backdoor password: `f00f` (transitions through SECRET_PIN states)
- 2 GEM_ASSERT cells generated during synthesis:
  1. Validates correct password usage
  2. Detects backdoor code (intentional assertion failure for testing)
- $display statements for state transitions and unlock events

## Test Sequence

The testbench (`safe_tb.v`) tries three passwords:

1. **c0de** (correct) - Should unlock via PIN states
2. **1234** (wrong) - Should lockout immediately
3. **f00f** (backdoor) - Should unlock but trigger assertion

## Synthesis Results

```
✓ Module: safe
✓ 2 GEM_ASSERT cells generated (cells 143, 144)
✓ 1 partition created
✓ 19 endpoints
✓ 133 state bits
```

## Known Issues

**Functional Mismatch between iverilog and GEM**:
- iverilog simulation: FSM works correctly, unlocked goes high, assertions fire
- GEM simulation: FSM doesn't transition correctly, unlocked stays low
- Root cause: Under investigation - may be related to FSM synthesis optimization or VCD input reading

This is a valuable test case that identified a potential issue with GEM's handling of complex FSM designs with immediate assertions.

## Running the Test

```bash
# Generate VCD input
iverilog -DSYNTHESIS -o safe_iverilog safe.v safe_tb.v
./safe_iverilog

# Synthesize
yosys -s synth_safe.tcl

# Simulate with Metal (partitioning happens automatically)
cargo run -r --features metal --bin loom -- sim safe_synth.gv safe.vcd gem_output_safe.vcd 48 --input-vcd-scope safe_tb
```

## Files

- `safe.v` - FSM module with assertions (adapted from SystemVerilog)
- `safe_tb.v` - Testbench with password testing task
- `synth_safe.tcl` - Yosys synthesis script for AIGPDK
- `safe_synth.gv` - Synthesized gate-level netlist
- `safe_synth.json` - Yosys JSON output
