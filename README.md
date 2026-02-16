# Loom

![CI](https://github.com/ChipFlow/Loom/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Rust](https://img.shields.io/badge/rust-edition%202021-orange)

Loom is a GPU-accelerated RTL logic simulator. Like a Jacquard loom weaving patterns from punched cards, Loom maps gate-level netlists onto a virtual manycore Boolean processor and executes them on GPUs, delivering 5-40X speedup over CPU-based RTL simulators.

Loom builds on the excellent [GEM](https://github.com/NVlabs/GEM) research by Zizheng Guo, Yanqing Zhang, Runsheng Wang, Yibo Lin, and Haoxing Ren at NVIDIA Research. [ChipFlow](https://chipflow.io) extends their work with:

- **Metal backend** for Apple Silicon Macs (in addition to the original CUDA backend)
- **Liberty timing support** — load real cell delays from Liberty files (e.g. SKY130) for timing-annotated simulation
- **SDF back-annotation** — post-layout timing from Standard Delay Format files
- **Setup/hold violation detection** — both CPU and GPU-side checking
- **Significant performance optimizations** to the partition mapping pipeline
- **CI/CD** with automated testing across both backends

### Roadmap: Timing Simulation

The goal is GPU-accelerated gate-level simulation with real cell timing — a first for open source. Current status:

| Component | Status |
|-----------|--------|
| Liberty file parsing | Done — loads SKY130 HD cell delays |
| Gate delay computation | Done — per-AIG-pin delays from Liberty |
| SDF back-annotation | Done — post-layout delays from SDF files |
| CPU timing simulation | Done — arrival time propagation with setup/hold checking |
| GPU timing simulation | Done — setup/hold violation detection on GPU |
| SKY130 timing test suite | Done — post-P&R test circuits with SDF |

Next steps:
1. Timing-aware bit packing for improved GPU utilization
2. Multi-clock domain support
3. Unified `loom sim` subcommand (sim logic currently in platform-specific binaries)

## Quick Start

Requires the [Rust toolchain](https://rustup.rs/).

```sh
git clone https://github.com/ChipFlow/Loom.git
cd Loom
git submodule update --init --recursive
```

### Build (Metal - macOS)

```sh
cargo build -r --features metal --bin metal_test
```

### Build (CUDA - Linux)

Requires CUDA toolkit installed.

```sh
cargo build -r --features cuda --bin cuda_test
```

## Usage

Loom operates in two phases:

1. **Map** your synthesized gate-level netlist to a `.gemparts` file (one-time cost):

```sh
# The `loom` binary works without GPU features:
cargo run -r --bin loom -- map design.gv design.gemparts

# Or with the legacy name (equivalent):
cargo run -r --features metal --bin cut_map_interactive -- design.gv design.gemparts
```

2. **Simulate** with a VCD input waveform:

```sh
# Metal (macOS) - use NUM_BLOCKS=1
cargo run -r --features metal --bin metal_test -- design.gv design.gemparts input.vcd output.vcd 1

# CUDA (Linux) - set NUM_BLOCKS to 2x your GPU's SM count
cargo run -r --features cuda --bin cuda_test -- design.gv design.gemparts input.vcd output.vcd NUM_BLOCKS
```

**See [docs/usage.md](./docs/usage.md) for full documentation** including synthesis preparation, VCD scope handling, and troubleshooting.

## Documentation

Browse the full documentation [online](https://chipflow.github.io/Loom/) or build it locally with [mdbook](https://rust-lang.github.io/mdBook/):

```sh
mdbook serve   # opens at http://localhost:3000
```

## Limitations

- Only supports non-interactive testbenches (static VCD input waveforms)
- Synchronous logic only (no latches or async sequential logic)
- Clock gates must use the `CKLNQD` module from `aigpdk.v`

## Benchmarks

Pre-synthesized benchmark designs are in `benchmarks/dataset/` (git submodule). See [benchmarks/README.md](benchmarks/README.md) for instructions.

Available designs: NVDLA, Rocket, Gemmini.

## Citation

Loom builds on the GEM research. Please cite the original paper if you find this work useful.

``` bibtex
@inproceedings{gem,
 author = {Guo, Zizheng and Zhang, Yanqing and Wang, Runsheng and Lin, Yibo and Ren, Haoxing},
 booktitle = {Proceedings of the 62nd Annual Design Automation Conference 2025},
 organization = {IEEE},
 title = {{GEM}: {GPU}-Accelerated Emulator-Inspired {RTL} Simulation},
 year = {2025}
}
```

## License

Apache-2.0. See [LICENSE](./LICENSE) for details.
