# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GEM (GPU-accelerated Emulator-inspired RTL simulation) is a GPU-accelerated RTL logic simulator originally developed by NVIDIA Research. It works like an FPGA-based RTL emulator: it synthesizes designs into an and-inverter graph (AIG), maps them to a virtual manycore Boolean processor, then emulates on GPUs for 5-40X speedup over CPU-based simulators.

Supports two GPU backends: **CUDA** (NVIDIA GPUs) and **Metal** (Apple Silicon Macs).

**Key limitation**: Only supports non-interactive testbenches with static input waveforms (VCD). Synchronous logic only - no latches or async sequential logic.

## Build Commands

Requires Rust toolchain (via rustup.rs) and either CUDA or Metal support.

```bash
# Initialize submodules (required first time)
git submodule update --init --recursive

# --- Metal (macOS) ---

# Build and run mapping tool (compiles design to .gemparts)
cargo run -r --features metal --bin cut_map_interactive -- --help

# Build and run Metal simulator
cargo run -r --features metal --bin metal_test -- --help

# --- CUDA (Linux/NVIDIA) ---

# Build and run mapping tool
cargo run -r --features cuda --bin cut_map_interactive -- --help

# Build and run CUDA simulator
cargo run -r --features cuda --bin cuda_test -- --help
```

## Typical Workflow

1. **Memory synthesis** (Yosys): Map memories using `memlib_yosys.txt` → outputs `memory_mapped.v`
2. **Logic synthesis** (DC or Yosys): Synthesize to `aigpdk.lib` cells → outputs `gatelevel.gv`
3. **GEM mapping**: `cut_map_interactive gatelevel.gv result.gemparts`
4. **Simulation**: `cuda_test` (NVIDIA) or `metal_test` (macOS) with `gatelevel.gv result.gemparts input.vcd output.vcd NUM_BLOCKS`

Set `NUM_BLOCKS` to 2× the number of GPU streaming multiprocessors (SMs) for CUDA, or 1 for Metal.

## Architecture

### Core Pipeline

```
NetlistDB (Verilog) → AIG → StagedAIG → Partitions → FlattenedScript → GPU Kernel (CUDA or Metal)
```

### Key Modules (`src/`)

- **`aigpdk.rs`**: Defines the AIGPDK standard cell library interface (AND gates, DFFs, clock gates, SRAMs)
- **`aig.rs`**: And-inverter graph representation. Converts NetlistDB to AIG with DriverType (AndGate, DFF, RAMBlock, etc.) and EndpointGroup abstractions
- **`staging.rs`**: Splits AIG into pipeline stages based on `--level-split` thresholds for deep circuits
- **`repcut.rs`**: Hypergraph partitioning using mt-kahypar for mapping to GPU blocks
- **`pe.rs`**: Partition executor - builds BoomerangStage structures (hierarchical 8192→1 reduction) that map to GPU block resources
- **`flatten.rs`**: Generates FlattenedScriptV1 - the final GPU execution script with packed instructions

### GPU Kernels (`csrc/`)

- **`kernel_v1.cu`/`kernel_v1_impl.cuh`**: CUDA simulation kernel implementing the Boolean processor
- **`kernel_v1.metal`**: Metal simulation kernel (macOS Apple Silicon)

### Binary Tools (`src/bin/`)

- **`cut_map_interactive.rs`**: Main compilation tool - partitions design iteratively until all endpoints map successfully
- **`cuda_test.rs`**: CUDA simulator - runs GPU simulation with VCD I/O
- **`metal_test.rs`**: Metal simulator - runs GPU simulation on macOS with VCD I/O
- **`gpu_sim.rs`**: GPU co-simulation binary
- Other `*_test.rs` files: Development/debugging utilities

### Dependencies (`eda-infra-rs` submodule)

Open-source Rust gate-level EDA infrastructure (https://github.com/gzz2000/eda-infra-rs):

- **`netlistdb`**: Flattened gate-level circuit netlist database. Stores cells, pins, nets with `Direction` (I/O), hierarchical names (`HierName`), and CSR-based connectivity (`VecCSR`). Created via `NetlistDB::from_sverilog_file()`.
- **`sverilogparse`**: Structural Verilog parser. Parses modules, wire definitions, assigns, and cell instantiations. Use `SVerilog::parse_str()`. Supports wire expressions including bit selects, slices, and concatenations.
- **`vcd-ng`**: VCD (Value Change Dump) reader/writer. `Parser` for reading with `FastFlow` for high-performance streaming. `Writer` for output generation.
- **`ulib`**: Universal computing library for heterogeneous CPU/GPU memory. Key types: `UVec<T>` (universal vector with automatic host/device sync), `Device` enum (CPU, CUDA(id), or Metal), `AsUPtrMut` trait. Enable with `--features cuda` or `--features metal`.
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
# Run with CPU baseline verification (CUDA)
cargo run -r --features cuda --bin cuda_test -- ... --check-with-cpu

# Limit simulation cycles (CUDA)
cargo run -r --features cuda --bin cuda_test -- ... --max-cycles 1000

# Metal equivalent
cargo run -r --features metal --bin metal_test -- ... --max-cycles 1000
```

## Benchmarks

Pre-synthesized benchmark designs are in `benchmarks/dataset/` (git submodule). See `benchmarks/README.md` for full instructions.

```bash
# Generate partition file (NVDLA - smallest, good for testing)
cargo run -r --features metal --bin cut_map_interactive -- \
    benchmarks/dataset/nvdlaAIG.gv \
    benchmarks/nvdla.gemparts

# Run Metal simulation benchmark
cargo run -r --features metal --bin metal_test -- \
    benchmarks/dataset/nvdlaAIG.gv \
    benchmarks/nvdla.gemparts \
    benchmarks/dataset/nvdla.pdp_16x6x16_4x2_split_max_int8_0.vcd \
    benchmarks/nvdla_output.vcd \
    1

# Criterion micro-benchmarks (no GPU required)
cargo bench --bench event_buffer
```

Available designs: NVDLA (254 MB), Rocket (124 MB), Gemmini (165 MB).

## Debugging Tools

### Netlist Graph Analysis (`scripts/netlist_graph/`)

**IMPORTANT**: Use this tool for tracing signal paths in post-synthesis netlists. It's much faster than manual grep-based analysis.

```bash
cd scripts/netlist_graph

# Trace what drives a signal (backwards through logic)
uv run netlist-graph drivers <netlist.v> "<signal>" -d 8

# Trace where a signal goes (forwards through logic)
uv run netlist-graph loads <netlist.v> "<signal>" -d 5

# Find path between two signals
uv run netlist-graph path <netlist.v> "<source>" "<target>"

# Search for nets matching pattern
uv run netlist-graph search <netlist.v> "<pattern>"

# Generate watchlist JSON for timing_sim_cpu
uv run netlist-graph watchlist <netlist.v> output.json signal1 signal2 ...

# Interactive mode for exploration
uv run netlist-graph interactive <netlist.v>
```

Example debugging session:
```bash
# Why isn't flash_ack being asserted?
uv run netlist-graph drivers tests/timing_test/minimal_build/6_final.v "spiflash.ctrl.wb_bus__ack" -d 5

# Trace reset path to CPU
uv run netlist-graph path tests/timing_test/minimal_build/6_final.v "gpio_in[40]" "ibus__cyc"
```

### Timing Violation Detection

See `docs/timing-violations.md` for the full guide on enabling GPU-side setup/hold violation checks, interpreting violation reports, and tracing violations back to source signals using `netlist_graph`.

### Timing Simulation with Signal Tracing

```bash
# Create watchlist and trace signals
cargo run -r --bin timing_sim_cpu -- netlist.v \
  --config testbench.json \
  --watchlist signals.json \
  --trace-output trace.csv \
  --max-cycles 1000
```
