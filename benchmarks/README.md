# GEM Benchmarks

This directory contains benchmark infrastructure for GEM.

## Dataset

The `dataset/` submodule contains pre-synthesized designs from the GEM paper:

| Design | Netlist | VCD | Description |
|--------|---------|-----|-------------|
| NVDLA | 254 MB | 20 MB | NVIDIA Deep Learning Accelerator |
| Rocket | 124 MB | 1.3 GB | RISC-V Rocket Core |
| Gemmini | 165 MB | 219 MB | Systolic Array Accelerator |

## Setup

Clone with submodules to get the dataset:
```bash
git clone --recursive https://github.com/ChipFlow/GEM.git
# Or if already cloned:
git submodule update --init --recursive
```

## Running Benchmarks

### 1. Run Metal simulation

Partitioning happens automatically at startup.

```bash
# NVDLA benchmark (smallest, good for testing)
cargo run -r --features metal --bin loom -- sim \
    benchmarks/dataset/nvdlaAIG.gv \
    benchmarks/dataset/nvdla.pdp_16x6x16_4x2_split_max_int8_0.vcd \
    benchmarks/nvdla_output.vcd \
    1

# Rocket benchmark
cargo run -r --features metal --bin loom -- sim \
    benchmarks/dataset/rocketAIG.gv \
    benchmarks/dataset/rocket.median.vcd \
    benchmarks/rocket_output.vcd \
    1
```

### 2. Criterion micro-benchmarks

```bash
cargo bench --bench event_buffer
```

## CI Benchmarks

The GitHub Actions workflow runs:
- Event buffer micro-benchmarks on every push
- Metal simulation timing on macOS runners (with timing report)

See `.github/workflows/ci.yml` for details.
