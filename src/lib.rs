// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loom — GPU-accelerated RTL logic simulator.
//!
//! Loom compiles gate-level netlists into GPU-executable Boolean processor
//! programs and simulates them on CUDA or Metal GPUs, achieving 5–40× speedup
//! over CPU-based RTL simulators.
//!
//! # Pipeline
//!
//! ```text
//! Verilog netlist
//!   → NetlistDB        (netlistdb crate — flat gate-level database)
//!   → AIG              (aig — and-inverter graph with DFFs, SRAMs, clock gates)
//!   → StagedAIG        (staging — pipeline stages for deep circuits)
//!   → Partitions       (repcut + pe — hypergraph partitioning → GPU block mapping)
//!   → FlattenedScript   (flatten — packed GPU instruction stream)
//!   → GPU kernel        (CUDA or Metal — Boolean processor execution)
//! ```
//!
//! # Key modules
//!
//! - [`aigpdk`] — AIGPDK standard cell library interface (AND gates, DFFs, clock gates, SRAMs)
//! - [`sky130`] / [`sky130_pdk`] — SKY130 cell library support with Liberty timing
//! - [`aig`] — And-inverter graph construction from [`netlistdb::NetlistDB`]
//! - [`liberty_parser`] — Liberty (.lib) file parser for cell timing data
//! - [`staging`] — Splits the AIG into pipeline stages via `--level-split` thresholds
//! - [`repcut`] — Hypergraph partitioning using mt-kahypar
//! - [`pe`] — Partition executor: maps partitions to GPU block resources (Boomerang stages)
//! - [`flatten`] — Generates the final packed GPU execution script ([`flatten::FlattenedScriptV1`])
//! - [`event_buffer`] — Timing event buffer for arrival-time propagation
//! - [`sdf_parser`] — SDF (Standard Delay Format) file parser for back-annotation
//! - [`testbench`] — Testbench configuration and VCD-driven simulation setup
//! - [`display`] — Display/assertion support infrastructure

pub mod aigpdk;

pub mod sky130;

pub mod sky130_pdk;

pub mod aig;

pub mod liberty_parser;

pub mod staging;

pub mod repcut;

pub mod pe;

pub mod flatten;

pub mod event_buffer;

pub mod display;

pub mod testbench;

pub mod sdf_parser;
