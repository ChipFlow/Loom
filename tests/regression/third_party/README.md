# Third-Party Test Cases

This directory contains test cases sourced from open-source projects to validate GEM's support for non-synthesizable SystemVerilog constructs ($display, assertions, etc.).

All test cases have been adapted for GEM's synthesis flow while preserving their original functionality and intent.

## Test Cases

### 1. Safe Module (FSM with Assertions)
- **Source**: [sva-playground](https://github.com/bluecmd/sva-playground) by Christian Svensson
- **License**: MIT License (see LICENSE_sva-playground)
- **Original**: `backdoor/safe.sv` and `backdoor/tb.sv`
- **Adapted for GEM**: `safe/`
- **Description**: A PIN-based safe cracker FSM with immediate assertions to verify no backdoor codes exist. Tests assertion checking in state machine designs.
- **Features**:
  - Immediate assertions (`assert(condition)`)
  - FSM with multiple states
  - Assertion fires when unexpected unlock codes are detected

### 2. PicoRV32 Memory Interface (Display Statements)
- **Source**: [PicoRV32](https://github.com/YosysHQ/picorv32) by Clifford Wolf
- **License**: ISC License / Public Domain (see LICENSE_picorv32)
- **Original**: `testbench_ez.v`
- **Adapted for GEM**: `picorv32_mem/`
- **Description**: Simplified memory interface testbench with $display statements for instruction fetch, read, and write operations.
- **Features**:
  - Multiple $display statements with format strings
  - Memory-mapped interface protocol
  - Real-world CPU interface pattern

## Attribution Requirements

When using or modifying these examples:
1. Preserve the original copyright and license notices
2. Include attribution to the original authors
3. Note any modifications made for GEM compatibility

## Adaptations for GEM

All examples have been adapted to work with GEM's synthesis flow:
- Wrapped in `ifdef FORMAL/SYNTHESIS` blocks as needed
- Added synthesis scripts targeting AIGPDK library
- Created testbenches that generate VCD input files
- Modified for GEM's supported SystemVerilog subset

## Running Tests

Each test directory contains:
- `<module>.v` - The design under test
- `<module>_tb.v` - Testbench for VCD generation
- `synth_<module>.tcl` - Yosys synthesis script
- `run_test.sh` - Script to synthesize, map, and simulate

Run all third-party tests:
```bash
cd tests/regression
./run_all.sh third_party/*
```

Run a specific test:
```bash
cd tests/regression/third_party/safe
./run_test.sh
```
