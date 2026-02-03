# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GEM (GPU-accelerated Emulator-inspired RTL simulation) is a CUDA-accelerated RTL logic simulator developed by NVIDIA Research. It works like an FPGA-based RTL emulator: it synthesizes designs into an and-inverter graph (AIG), maps them to a virtual manycore Boolean processor, then emulates on CUDA-compatible GPUs for 5-40X speedup over CPU-based simulators.

**Key limitation**: Only supports non-interactive testbenches with static input waveforms (VCD). Synchronous logic only - no latches or async sequential logic.

## Build Commands

Requires CUDA and Rust toolchain (via rustup.rs).

```bash
# Initialize submodules (required first time)
git submodule update --init --recursive

# Build and run mapping tool (compiles design to .gemparts)
cargo run -r --features cuda --bin cut_map_interactive -- --help

# Build and run simulator
cargo run -r --features cuda --bin cuda_test -- --help
```

## Typical Workflow

1. **Memory synthesis** (Yosys): Map memories using `memlib_yosys.txt` → outputs `memory_mapped.v`
2. **Logic synthesis** (DC or Yosys): Synthesize to `aigpdk.lib` cells → outputs `gatelevel.gv`
3. **GEM mapping**: `cut_map_interactive gatelevel.gv result.gemparts`
4. **Simulation**: `cuda_test gatelevel.gv result.gemparts input.vcd output.vcd NUM_BLOCKS`

Set `NUM_BLOCKS` to 2× the number of GPU streaming multiprocessors (SMs).

## Architecture

### Core Pipeline

```
NetlistDB (Verilog) → AIG → StagedAIG → Partitions → FlattenedScript → CUDA Kernel
```

### Key Modules (`src/`)

- **`aigpdk.rs`**: Defines the AIGPDK standard cell library interface (AND gates, DFFs, clock gates, SRAMs)
- **`aig.rs`**: And-inverter graph representation. Converts NetlistDB to AIG with DriverType (AndGate, DFF, RAMBlock, etc.) and EndpointGroup abstractions
- **`staging.rs`**: Splits AIG into pipeline stages based on `--level-split` thresholds for deep circuits
- **`repcut.rs`**: Hypergraph partitioning using mt-kahypar for mapping to GPU blocks
- **`pe.rs`**: Partition executor - builds BoomerangStage structures (hierarchical 8192→1 reduction) that map to GPU block resources
- **`flatten.rs`**: Generates FlattenedScriptV1 - the final GPU execution script with packed instructions

### CUDA Kernels (`csrc/`)

- **`kernel_v1.cu`/`kernel_v1_impl.cuh`**: GPU simulation kernel implementing the Boolean processor

### Binary Tools (`src/bin/`)

- **`cut_map_interactive.rs`**: Main compilation tool - partitions design iteratively until all endpoints map successfully
- **`cuda_test.rs`**: Main simulator - runs GPU simulation with VCD I/O
- Other `*_test.rs` files: Development/debugging utilities

### Dependencies (`eda-infra-rs` submodule)

Open-source Rust gate-level EDA infrastructure (https://github.com/gzz2000/eda-infra-rs):

- **`netlistdb`**: Flattened gate-level circuit netlist database. Stores cells, pins, nets with `Direction` (I/O), hierarchical names (`HierName`), and CSR-based connectivity (`VecCSR`). Created via `NetlistDB::from_sverilog_file()`.
- **`sverilogparse`**: Structural Verilog parser. Parses modules, wire definitions, assigns, and cell instantiations. Use `SVerilog::parse_str()`. Supports wire expressions including bit selects, slices, and concatenations.
- **`vcd-ng`**: VCD (Value Change Dump) reader/writer. `Parser` for reading with `FastFlow` for high-performance streaming. `Writer` for output generation.
- **`ulib`**: Universal computing library for heterogeneous CPU/CUDA memory. Key types: `UVec<T>` (universal vector with automatic host/device sync), `Device` enum (CPU or CUDA(id)), `AsUPtrMut` trait. Enable CUDA with `--features cuda`.
- **`ucc`**: Build system for Rust-C++-CUDA interop. Manages C++ header dependencies between crates (`export_csrc`/`import_csrc`), compiles CUDA sources (`cl_cuda()`), generates FFI bindings (`bindgen()`), and creates `compile_commands.json` for LSP.
- **`clilog`**: Logging wrapper over `log` crate with message type tagging and automatic suppression. Macros: `clilog::info!()`, `clilog::debug!()`, `clilog::warn!()`. Timer support via `clilog::stimer!()`/`clilog::finish!()`.

### AIG PDK Files (`aigpdk/`)

- `aigpdk.lib`/`aigpdk.db`: Liberty library for DC synthesis
- `aigpdk_nomem.lib`: Library without memory cells (for Yosys)
- `aigpdk.v`: Verilog models including `CKLNQD` clock gate
- `memlib_yosys.txt`: Memory mapping rules for Yosys

## Key Constraints

GPU block resource limits (from `pe.rs`):
- Max 8191 unique inputs per partition
- Max 8191 unique outputs per partition
- Max 4095 intermediate pins alive per stage
- Max 64 SRAM output groups

If mapping fails with "single endpoint cannot map", use `--level-split` to force stage splits (e.g., `--level-split 30` or `--level-split 20,40`).

## Testing

```bash
# Run with CPU baseline verification
cargo run -r --features cuda --bin cuda_test -- ... --check-with-cpu

# Limit simulation cycles
cargo run -r --features cuda --bin cuda_test -- ... --max-cycles 1000
```
