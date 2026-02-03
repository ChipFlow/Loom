# GEM Simulation Architecture

This document describes GEM's internal simulation architecture based on investigation and testing.

## Overview

GEM (GPU-accelerated Emulator-inspired RTL simulation) compiles gate-level netlists into GPU kernels that simulate designs 5-40X faster than CPU-based simulators. It works like an FPGA-based RTL emulator by converting designs into an and-inverter graph (AIG), partitioning it for GPU blocks, and generating optimized GPU code.

## Pipeline Stages

```
Verilog Netlist → NetlistDB → AIG → StagedAIG → Partitions → FlattenedScript → GPU Kernel
                     ↓            ↓                    ↓            ↓
                  Parse      Synthesis         Hypergraph     Instruction
                  Netlist    to AIGs          Partitioning    Generation
```

### 1. NetlistDB (Input Parsing)

**Input**: Gate-level Verilog (`.gv` files) from synthesis tools (Yosys, Design Compiler)

**Process**:
- Parses structural Verilog using `sverilogparse` crate
- Creates flattened netlist database with cells, pins, nets
- Identifies primary inputs, outputs, clock signals
- Stores connectivity in CSR (Compressed Sparse Row) format

**Key Limitations**:
- Only supports synthesized gate-level netlists (not RTL)
- No behavioral Verilog constructs (always blocks, if/case statements)
- Expects standard cells from supported libraries (AIGPDK)

### 2. AIG (And-Inverter Graph)

**Process**: Converts gate-level netlist to AIG representation

**Data Structure**:
```rust
pub enum DriverType {
    AndGate,           // Basic AND gate
    DFF,               // D flip-flop
    ClockGate,         // Clock gating cell
    RAMBlock,          // Memory block
    GemAssert,         // Assertion checking
    GemDisplay,        // Display output
    // ... more types
}
```

**Statistics** (example from safe.v):
- **157 AIG pins**: Internal circuit nodes
- **133 AND gates**: Logic operations
- **16 DFF cells**: Sequential elements
- **2 GEM_ASSERT cells**: Assertion nodes
- **480 total pins**: Including I/O

**Key Features**:
- Clock inference from DFF connections
- Assertion cell detection (`GEM_ASSERT`, `GEM_DISPLAY`)
- Endpoint grouping for outputs and registers

### 3. StagedAIG (Pipeline Staging)

**Purpose**: Split deep combinational logic into pipeline stages

**Process**:
- Analyzes combinational depth between registers
- Splits logic at `--level-split` thresholds
- Creates pipeline stages to fit GPU resource constraints

**When Needed**:
- Designs with very deep combinational paths (>50 levels)
- When single-stage partitioning fails resource limits
- Use `--level-split 30` or `--level-split 20,40` to force splits

### 4. Partitioning (Hypergraph Cut)

**Tool**: mt-kahypar hypergraph partitioner

**Constraints** (GPU block resources):
- Max 8191 unique inputs per partition
- Max 8191 unique outputs per partition
- Max 4095 intermediate pins alive per stage
- Max 64 SRAM output groups

**Process**:
- Interactive partitioning (`cut_map_interactive`)
- Tries 1 partition first, then increases if needed
- Merges partitions to minimize inter-partition communication

**Output**: `.gemparts` file with partition assignments

### 5. FlattenedScript (GPU Instruction Generation)

**Process**: Generates GPU execution script from partitions

**Script Components**:
- **Boomerang stages**: Hierarchical 8192→1 reduction structure
- **State buffer**: Packed 32-bit words for all register values
- **SRAM interface**: Memory block read/write operations
- **Assertion positions**: Bit positions for assertion conditions
- **Display positions**: Enable bits and argument positions

**Statistics** (example):
```
reg/io state size: 133 bits → 5 words (32-bit)
script size: 30208 instructions
assertion_positions: [(cell_id, bit_pos, msg_id, type)]
display_positions: [(cell_id, enable_pos, format, arg_positions, widths)]
```

**Key Insight**: All state is packed into a flat bit array, indexed by position in 32-bit words.

### 6. GPU Kernel Execution

**Kernel Types**:
- `kernel_v1.cu` / `kernel_v1_impl.cuh`: CUDA implementation
- `kernel_v1.metal`: Metal (Apple Silicon) implementation

**Execution Model**:
- Each GPU block simulates one partition
- Multiple blocks run in parallel
- State synchronized between stages
- CPU checks assertion/display conditions after GPU completes

## VCD Input/Output

### Input VCD Requirements

**Critical Discovery**: GEM expects VCD signals at **absolute top-level** (no module hierarchy).

**Expected Signal Format**:
```vcd
$var reg 1 ! clk $end
$var reg 1 " reset $end
$var reg 4 # din [3:0] $end
$var reg 1 $ din_valid $end
```

**NOT** (with module scope):
```vcd
$scope module testbench $end
  $scope module dut $end
    $var wire 1 ! clk $end
    ...
```

**Signal Matching**:
- GEM looks for signals matching synthesized module port names
- Uses `HierName()` (empty hierarchy) for matching
- If signals are scoped under modules, GEM reports:
  ```
  WARN (GATESIM_VCDI_MISSING_PI) Primary input port (HierName(), "reset", None) not present in the VCD input
  ```

**VCD Scope Option**:
- `--input-vcd-scope <scope>`: Specify module hierarchy to read from
- **Current Issue**: Even with scope specified, signal matching fails
- **Workaround**: Generate VCD with signals at absolute top level

### Output VCD Structure

GEM generates minimal VCD with only primary outputs:
```vcd
$timescale 1 ns $end
$scope module gem_top_module $end
$var wire 1 ! unlocked $end
$upscope $end
```

Internal states and intermediate signals are not dumped.

## Assertion and Display Support

### Assertion Infrastructure

**Synthesis Flow**:
```
Verilog assert() → Yosys $check cell → techmap gem_formal.v → GEM_ASSERT cell
```

**Runtime**:
- GEM stores assertion positions in `FlattenedScript`
- CPU checks assertion bits after GPU simulation
- Configurable actions: Log, Pause, Terminate

**AssertConfig**:
```rust
pub struct AssertConfig {
    pub on_failure: AssertAction,  // Log, Pause, Terminate
    pub max_failures: Option<u32>,
}
```

### Display Infrastructure

**Synthesis Flow**:
```
Verilog $display() → Yosys $print cell → techmap gem_formal.v → GEM_DISPLAY cell
```

**Runtime**:
- Format strings stored in JSON metadata
- CPU checks display enable bits after GPU simulation
- Arguments extracted from state buffer positions

**Limitation**: Format string preservation depends on Yosys synthesis preserving attributes.

## Debug Information

### Enabling Debug Output

```bash
# Metal simulation with debug logging
RUST_LOG=debug cargo run -r --features metal --bin metal_test -- <args>

# CPU verification (slower but validates GPU results)
cargo run -r --features metal --bin metal_test -- <args> --check-with-cpu
```

### Key Debug Messages

**AIG Construction**:
```
Found GEM_ASSERT cell 143 (condition_iv=0, en_iv=0, a_iv=76, clken_iv=2)
Found GEM_DISPLAY cell 24 (enable_iv=2, clken_iv=2, args=32)
```

**Partitioning**:
```
netlist has 480 pins, 157 aig pins, 133 and gates
current: 19 endpoints, try 1 parts
after merging: 1 parts
```

**Flattening**:
```
Built script for 48 blocks, reg/io state size 133, sram size 0, script size 30208
Assertion: cell=144, pos=4195 (word=131, bit=3), msg_id=144, type=None
Display: cell=24, enable_pos=5154 (word=161, bit=2), format='...', args=[...]
```

**VCD Reading**:
```
WARN (GATESIM_VCDI_MISSING_PI) Primary input port (HierName(), "reset", None) not present
```

## Performance Characteristics

### Speedup vs CPU

- Simple designs: 5-10X faster
- Complex designs: 10-40X faster
- Depends on:
  - Number of GPU SMs (streaming multiprocessors)
  - Partition granularity
  - VCD I/O overhead

### Resource Scaling

**GPU Block Count**: Set `NUM_BLOCKS` to 2× number of GPU SMs
- Apple M4 Pro: 48 blocks (24 SMs × 2)
- NVIDIA GPUs: Check SM count with `nvidia-smi`

**Memory Usage**:
- State buffer: `num_blocks × state_size × num_cycles × 4 bytes`
- Script: `script_size × 4 bytes` (shared across blocks)

## Known Issues and Limitations

### 1. VCD Hierarchy Mismatch
**Issue**: GEM expects flat VCD signal hierarchy
**Impact**: Missing input signals cause incorrect simulation results
**Workaround**: Generate VCD with `$dumpvars(1, sig1, sig2, ...)` at top level
**Status**: Under investigation

### 2. Complex FSM Designs
**Issue**: Some FSM designs don't simulate correctly even with proper VCD
**Example**: safe.v (9-state PIN cracker FSM)
**Possible Causes**:
- Synthesis optimization changes FSM encoding
- Initial state handling differences
- Reset timing issues
**Status**: Identified through third-party test suite

### 3. Format String Preservation
**Issue**: Yosys synthesis may not preserve `gem_format` attributes
**Impact**: Display messages show placeholders instead of actual format strings
**Workaround**: Extract format strings from pre-synthesis JSON
**Status**: Tool limitation, not GEM bug

## Investigation Methodology

This documentation was created through systematic investigation:

1. **Structure Analysis**: Examined source code in `src/aig.rs`, `src/flatten.rs`, `src/staging.rs`
2. **Debug Tracing**: Used `RUST_LOG=debug` to capture internal state
3. **Netlist Inspection**: Analyzed synthesized `.gv` files with `grep`
4. **VCD Comparison**: Compared iverilog vs GEM VCD outputs
5. **Test Case Development**: Created minimal reproducible examples
6. **Iterative Debugging**: Progressively simplified designs to isolate issues

## References

- Main codebase: `src/` directory
- EDA infrastructure: `eda-infra-rs/` submodule (netlistdb, vcd-ng, ulib)
- AIGPDK library: `aigpdk/` directory
- Test cases: `tests/` directory
- Third-party examples: `tests/regression/third_party/`

---

**Document Version**: 1.0
**Last Updated**: 2025-01-08
**Authors**: NVIDIA GEM Team + Claude Code Investigation
