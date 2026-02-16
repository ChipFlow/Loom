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
use crate::pe::Partition;
use crate::sky130::{detect_library_from_file, CellLibrary, SKY130LeafPins};
use crate::staging::build_staged_aigs;
use netlistdb::NetlistDB;

/// Parameters for loading a design.
pub struct DesignArgs {
    pub netlist_verilog: PathBuf,
    pub top_module: Option<String>,
    pub level_split: Vec<usize>,
    pub gemparts: PathBuf,
    pub num_blocks: usize,
    pub json_path: Option<PathBuf>,
    pub sdf: Option<PathBuf>,
    pub sdf_corner: String,
    pub sdf_debug: bool,
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
    let lib =
        detect_library_from_file(&args.netlist_verilog).expect("Failed to read netlist file");
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

    let f = std::fs::File::open(&args.gemparts).unwrap();
    let mut buf = std::io::BufReader::new(f);
    let parts_in_stages: Vec<Vec<Partition>> = serde_bare::from_reader(&mut buf).unwrap();
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

    let mut script = FlattenedScriptV1::from(
        &aig,
        &stageds
            .iter()
            .map(|(_, _, staged)| staged)
            .collect::<Vec<_>>(),
        &parts_in_stages
            .iter()
            .map(|ps| ps.as_slice())
            .collect::<Vec<_>>(),
        args.num_blocks,
        input_layout,
    );

    // Load SDF timing data if provided
    if let Some(ref sdf_path) = args.sdf {
        load_sdf(&mut script, &aig, &netlistdb, sdf_path, &args.sdf_corner, args.sdf_debug);
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
pub fn load_sdf(
    script: &mut FlattenedScriptV1,
    aig: &AIG,
    netlistdb: &NetlistDB,
    sdf_path: &Path,
    sdf_corner: &str,
    sdf_debug: bool,
) {
    let corner = match sdf_corner {
        "min" => crate::sdf_parser::SdfCorner::Min,
        "max" => crate::sdf_parser::SdfCorner::Max,
        _ => crate::sdf_parser::SdfCorner::Typ,
    };
    clilog::info!("Loading SDF: {:?} (corner: {})", sdf_path, sdf_corner);
    match crate::sdf_parser::SdfFile::parse_file(sdf_path, corner) {
        Ok(sdf) => {
            clilog::info!("SDF loaded: {}", sdf.summary());
            script.load_timing_from_sdf(aig, netlistdb, &sdf, 25000, None, sdf_debug);
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
