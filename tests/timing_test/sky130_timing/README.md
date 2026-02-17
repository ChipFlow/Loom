# SKY130 Timing Test Suite

Small test circuits using SKY130 HD standard cells with analytically
verifiable timing properties. These serve as ground truth for validating
Loom's timing simulation (both CPU and GPU paths).

## Test Circuits

### 1. `inv_chain.v` - Inverter Chain
- **Circuit**: DFF -> 16 inverters (sky130_fd_sc_hd__inv_1) -> DFF
- **Logic function**: Identity (even number of inversions)
- **Expected combo delay**: 16 x inv_delay (from Liberty)
- **Expected arrival at capture DFF**: clk_to_q + 16 x inv_delay
- **Purpose**: Validates basic delay accumulation through a linear chain

### 2. `logic_cone.v` - Convergent Logic Cone
- **Circuit**: 4 DFFs -> tree of nand2/nor2/and2 gates -> DFF
- **Logic function**: Specific Boolean function of 4 inputs
- **Expected combo delay**: Critical path through deepest branch
- **Purpose**: Validates max-of-inputs arrival time propagation

### 3. `setup_violation.v` - Setup Time Violation
- **Circuit**: Same as inv_chain but designed to violate setup at tight clock
- **Expected**: TIMING PASSED at 10ns clock, TIMING FAILED at 1ns clock
- **Purpose**: Validates setup/hold checking

## Generating Reference Values

The expected timing values are computed analytically from the SKY130 HD
Liberty file (sky130_fd_sc_hd__tt_025C_1v80.lib). Key values at typical corner:

| Parameter | Value (ps) |
|-----------|-----------|
| inv_1 tpd (rise) | ~28 |
| inv_1 tpd (fall) | ~18 |
| nand2_1 tpd | ~30-40 |
| and2_1 tpd | ~50-60 |
| dfxtp_1 clk->Q | ~310 |
| dfxtp_1 setup | ~80 |
| dfxtp_1 hold | ~-40 |

Note: Actual values depend on load capacitance and input transition time.
The defaults in `TimingLibrary::default_sky130()` are approximate.

## Running

```sh
# Run with default SKY130 timing values
cargo run -r --bin timing_sim_cpu -- tests/timing_test/sky130_timing/inv_chain.v \
    tests/timing_test/sky130_timing/inv_chain.vcd --clock-period 10000

# Run with real Liberty file (if available)
cargo run -r --bin timing_sim_cpu -- tests/timing_test/sky130_timing/inv_chain.v \
    tests/timing_test/sky130_timing/inv_chain.vcd --clock-period 10000 \
    --liberty path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
```
