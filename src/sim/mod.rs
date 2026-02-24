// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared simulation infrastructure for GPU simulation binaries.
//!
//! This module provides common code used by the `loom sim` and `loom cosim`
//! subcommands:
//!
//! - [`cpu_reference`] — CPU-side block simulator for validation
//! - [`vcd_io`] — VCD input parsing and output writing utilities
//! - [`setup`] — Design loading pipeline (netlist → AIG → script)

#[cfg(feature = "metal")]
pub mod cosim_metal;
pub mod cpu_reference;
pub mod setup;
pub mod vcd_io;
