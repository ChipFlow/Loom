// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Design loading pipeline: netlist → AIG → staged → script → SDF.
//!
//! Shared between all simulation binaries.

use std::path::{Path, PathBuf};

use crate::aig::{DriverType, AIG};
use crate::aigpdk::AIGPDKLeafPins;
use crate::display::extract_display_info_from_json;
use crate::flatten::FlattenedScriptV1;
use crate::pe::{process_partitions, Partition};
use crate::repcut::RCHyperGraph;
use crate::sky130::{detect_library_from_file, CellLibrary, SKY130LeafPins};
use crate::staging::{build_staged_aigs, StagedAIG};
use netlistdb::NetlistDB;
use rayon::prelude::*;

/// Parameters for loading a design.
pub struct DesignArgs {
    pub netlist_verilog: PathBuf,
    pub top_module: Option<String>,
    pub level_split: Vec<usize>,
    pub num_blocks: usize,
    pub json_path: Option<PathBuf>,
    pub sdf: Option<PathBuf>,
    pub sdf_corner: String,
    pub sdf_debug: bool,
    /// Clock period in picoseconds for SDF timing. Defaults to 25000 if not set.
    pub clock_period_ps: Option<u64>,
    /// Enable selective X-propagation.
    pub xprop: bool,
}

/// Result of loading a design: everything needed for simulation.
pub struct LoadedDesign {
    pub netlistdb: NetlistDB,
    pub aig: AIG,
    pub script: FlattenedScriptV1,
    /// Path to JSON file with display format strings.
    pub json_path: PathBuf,
}

/// Load a design through the full pipeline: netlist → AIG → staged → script.
///
/// Detects cell library (AIGPDK or SKY130), loads display info from JSON,
/// builds the flattened script, and optionally loads SDF timing data.
pub fn load_design(args: &DesignArgs) -> LoadedDesign {
    // Detect cell library
    let lib = detect_library_from_file(&args.netlist_verilog).expect("Failed to read netlist file");
    clilog::info!("Detected cell library: {}", lib);

    if lib == CellLibrary::Mixed {
        panic!("Mixed AIGPDK and SKY130 cells in netlist not supported");
    }

    let netlistdb = match lib {
        CellLibrary::SKY130 => NetlistDB::from_sverilog_file(
            &args.netlist_verilog,
            args.top_module.as_deref(),
            &SKY130LeafPins,
        )
        .expect("cannot build netlist"),
        CellLibrary::AIGPDK | CellLibrary::Mixed => NetlistDB::from_sverilog_file(
            &args.netlist_verilog,
            args.top_module.as_deref(),
            &AIGPDKLeafPins(),
        )
        .expect("cannot build netlist"),
    };

    let mut aig = AIG::from_netlistdb(&netlistdb);

    // Load display format info from JSON if available
    let json_path = args
        .json_path
        .clone()
        .unwrap_or_else(|| args.netlist_verilog.with_extension("json"));
    if json_path.exists() {
        match extract_display_info_from_json(&json_path) {
            Ok(display_info) => {
                if !display_info.is_empty() {
                    clilog::info!(
                        "Loaded {} display format strings from {:?}",
                        display_info.len(),
                        json_path
                    );
                    aig.populate_display_info(&display_info);
                }
            }
            Err(e) => {
                clilog::warn!("Could not load display info from JSON: {}", e);
            }
        }
    }

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let parts_in_stages: Vec<Vec<Partition>> = generate_partitions(&aig, &stageds, 0);
    clilog::info!(
        "# of effective partitions in each stage: {:?}",
        parts_in_stages
            .iter()
            .map(|ps| ps.len())
            .collect::<Vec<_>>()
    );

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    let staged_refs: Vec<_> = stageds.iter().map(|(_, _, staged)| staged).collect();
    let parts_refs: Vec<_> = parts_in_stages.iter().map(|ps| ps.as_slice()).collect();

    let mut script = if args.xprop {
        let (x_capable, stats) = aig.compute_x_capable_pins();
        clilog::info!(
            "X-propagation: {}/{} pins ({:.1}%) X-capable, {} fixpoint iterations",
            stats.num_x_capable_pins,
            stats.total_pins,
            if stats.total_pins > 0 {
                stats.num_x_capable_pins as f64 / stats.total_pins as f64 * 100.0
            } else {
                0.0
            },
            stats.fixpoint_iterations
        );
        FlattenedScriptV1::from_with_xprop(
            &aig,
            &staged_refs,
            &parts_refs,
            args.num_blocks,
            input_layout,
            &x_capable,
        )
    } else {
        FlattenedScriptV1::from(
            &aig,
            &staged_refs,
            &parts_refs,
            args.num_blocks,
            input_layout,
        )
    };

    if args.xprop {
        let x_parts = script.partition_x_capable.iter().filter(|&&x| x).count();
        let total_parts = script.partition_x_capable.len();
        clilog::info!(
            "X-propagation: {}/{} partitions X-aware",
            x_parts,
            total_parts
        );
    }

    // Load SDF timing data if provided
    if let Some(ref sdf_path) = args.sdf {
        load_sdf(
            &mut script,
            &aig,
            &netlistdb,
            sdf_path,
            &args.sdf_corner,
            args.sdf_debug,
            args.clock_period_ps,
        );
    }

    // Print script hash
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut s = DefaultHasher::new();
    script.blocks_data.hash(&mut s);
    println!("Script hash: {}", s.finish());

    LoadedDesign {
        netlistdb,
        aig,
        script,
        json_path,
    }
}

/// Load SDF timing data into a script.
///
/// `clock_period_ps` overrides the default 25000ps clock period used for timing.
pub fn load_sdf(
    script: &mut FlattenedScriptV1,
    aig: &AIG,
    netlistdb: &NetlistDB,
    sdf_path: &Path,
    sdf_corner: &str,
    sdf_debug: bool,
    clock_period_ps: Option<u64>,
) {
    let clock_ps = clock_period_ps.unwrap_or(25000);
    let corner = match sdf_corner {
        "min" => crate::sdf_parser::SdfCorner::Min,
        "max" => crate::sdf_parser::SdfCorner::Max,
        _ => crate::sdf_parser::SdfCorner::Typ,
    };
    clilog::info!(
        "Loading SDF: {:?} (corner: {}, clock_period={}ps)",
        sdf_path,
        sdf_corner,
        clock_ps
    );
    match crate::sdf_parser::SdfFile::parse_file(sdf_path, corner) {
        Ok(sdf) => {
            clilog::info!("SDF loaded: {}", sdf.summary());
            script.load_timing_from_sdf(aig, netlistdb, &sdf, clock_ps, None, sdf_debug);
            script.inject_timing_to_script();
        }
        Err(e) => clilog::warn!("Failed to load SDF: {}", e),
    }
}

/// Build timing constraint buffer for GPU-side setup/hold checking.
///
/// Returns `Some((clock_ps, constraint_buffer))` if timing is enabled,
/// where `constraint_buffer` = `[clock_ps, constraints[0], constraints[1], ...]`.
pub fn build_timing_constraints(script: &FlattenedScriptV1) -> Option<Vec<u32>> {
    if script.timing_enabled && !script.dff_constraints.is_empty() {
        let (clock_ps, constraints) = script.build_timing_constraint_buffer();
        let non_zero = constraints.iter().filter(|&&v| v != 0).count();
        clilog::info!(
            "Timing constraints: {} words, {} with DFF constraints, clock_period={}ps",
            constraints.len(),
            non_zero,
            clock_ps
        );
        let mut buf = Vec::with_capacity(1 + constraints.len());
        buf.push(clock_ps);
        buf.extend_from_slice(&constraints);
        Some(buf)
    } else {
        None
    }
}

/// Invoke the mt-kahypar partitioner.
fn run_par(hg: &RCHyperGraph, num_parts: usize) -> Vec<Vec<usize>> {
    clilog::debug!("invoking partitioner (#parts {})", num_parts);
    // mt-kahypar requires k >= 2, handle k=1 manually
    if num_parts == 1 {
        return vec![(0..hg.num_vertices()).collect()];
    }

    let parts_ids = hg.partition(num_parts);
    let mut parts = vec![vec![]; num_parts];
    for (i, part_id) in parts_ids.into_iter().enumerate() {
        parts[part_id].push(i);
    }
    parts
}

/// Generate partitions from the AIG and staged AIGs.
///
/// Iteratively partitions endpoints using mt-kahypar hypergraph partitioning.
/// `max_stage_degrad` controls how many degradation layers are allowed
/// during partition merging (0 = no degradation).
fn generate_partitions(
    aig: &AIG,
    stageds: &[(usize, usize, StagedAIG)],
    max_stage_degrad: usize,
) -> Vec<Vec<Partition>> {
    stageds
        .iter()
        .map(|&(l, r, ref staged)| {
            clilog::info!(
                "interactive partitioning stage {}-{}",
                l,
                match r {
                    usize::MAX => "max".to_string(),
                    r => format!("{}", r),
                }
            );

            let mut parts_good: Vec<(Vec<usize>, Partition)> = Vec::new();
            let mut unrealized_endpoints =
                (0..staged.num_endpoint_groups()).collect::<Vec<_>>();
            let mut division = 600;

            while !unrealized_endpoints.is_empty() {
                division = (division / 2).max(1);
                let num_parts = (unrealized_endpoints.len() + division - 1) / division;
                clilog::info!(
                    "current: {} endpoints, try {} parts",
                    unrealized_endpoints.len(),
                    num_parts
                );
                let staged_ur = staged.to_endpoint_subset(&unrealized_endpoints);
                let hg_ur = RCHyperGraph::from_staged_aig(aig, &staged_ur);
                let mut parts_indices = run_par(&hg_ur, num_parts);
                for idcs in &mut parts_indices {
                    for i in idcs {
                        *i = unrealized_endpoints[*i];
                    }
                }
                let parts_try = parts_indices
                    .par_iter()
                    .map(|endpts| Partition::build_one(aig, staged, endpts))
                    .collect::<Vec<_>>();
                let mut new_unrealized_endpoints = Vec::new();
                for (idx, part_opt) in parts_indices.into_iter().zip(parts_try.into_iter()) {
                    match part_opt {
                        Some(part) => {
                            parts_good.push((idx, part));
                        }
                        None => {
                            if idx.len() == 1 {
                                panic!("A single endpoint still cannot map, you need to increase level cut granularity.");
                            }
                            for endpt_i in idx {
                                new_unrealized_endpoints.push(endpt_i);
                            }
                        }
                    }
                }
                new_unrealized_endpoints.sort_unstable();
                unrealized_endpoints = new_unrealized_endpoints;
            }

            clilog::info!(
                "interactive partition completed: {} in total. merging started.",
                parts_good.len()
            );

            let (parts_indices_good, prebuilt): (Vec<_>, Vec<_>) =
                parts_good.into_iter().unzip();
            let effective_parts = process_partitions(
                aig,
                staged,
                parts_indices_good,
                Some(prebuilt),
                max_stage_degrad,
            )
            .unwrap();
            clilog::info!("after merging: {} parts.", effective_parts.len());
            effective_parts
        })
        .collect::<Vec<_>>()
}
