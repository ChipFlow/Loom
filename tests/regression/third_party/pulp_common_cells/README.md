# PULP Platform Common Cells - GEM Regression Tests

This directory contains modules adapted from the [PULP Platform common_cells](https://github.com/pulp-platform/common_cells) repository for GEM regression testing.

## Attribution

- **Original Source:** https://github.com/pulp-platform/common_cells
- **Authors:** ETH Zurich and University of Bologna
- **License:** Solderpad Hardware License, Version 0.51 (SHL-0.51)

## About PULP Platform

The PULP (Parallel Ultra-Low-Power) platform is an open-source multi-core computing platform developed as part of the collaboration between ETH Zurich and the University of Bologna. The common_cells repository contains reusable SystemVerilog components used across PULP projects.

## Modules Included

### delta_counter.v
An up/down counter with variable delta increment/decrement. Features:
- Configurable bit width
- Up/down counting with variable step size
- Load and clear functionality
- Optional sticky overflow detection

Original file: `src/delta_counter.sv`
Authors: Florian Zaruba (ETH Zurich)

### lfsr_8bit.v
An 8-bit Linear Feedback Shift Register. Features:
- Configurable seed value
- Enable signal for pausing LFSR progression
- One-hot output (for cache replacement policies)
- Binary output (lower 3 bits for 8-way selection)
- Polynomial: x^8 + x^4 + x^3 + x^2 + 1

Original file: `src/lfsr_8bit.sv`
Authors: Igor Loi (Univ. Bologna), Florian Zaruba (ETH Zurich)

## Modifications for GEM

The original SystemVerilog files have been adapted to Verilog for broader tool compatibility:
- Converted `logic` types to `reg`/`wire`
- Converted `always_comb`/`always_ff` to `always @(*)`/`always @(posedge ...)`
- Added self-contained testbenches within `ifndef SYNTHESIS` blocks
- Added GEM-specific test cases

## License

See LICENSE_pulp_common_cells for the full Solderpad Hardware License text.
