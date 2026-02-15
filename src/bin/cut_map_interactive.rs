// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! This is an experimental interactive cutting-then-mapping
//! implementation.
//!
//! The key idea is to only repartition the endpoint groups that
//! are unable to be mapped.

use std::path::PathBuf;
use gem::repcut::RCHyperGraph;
use gem::aigpdk::AIGPDKLeafPins;
use gem::sky130::{SKY130LeafPins, CellLibrary, detect_library_from_file};
use gem::aig::AIG;
use gem::staging::build_staged_aigs;
use gem::pe::{process_partitions, Partition};
use netlistdb::NetlistDB;
use rayon::prelude::*;

/// Call builtin partitioner.
fn run_par(hg: &RCHyperGraph, num_parts: usize) -> Vec<Vec<usize>> {
    clilog::debug!("invoking partitioner (#parts {})", num_parts);
	// Handle the special case where num_parts = 1
    // mt-kahypar requires k >= 2, so we handle k=1 manually
    if num_parts == 1 {
        let mut parts = vec![vec![]; 1];
        // Put all vertices in the single partition
        for i in 0..hg.num_vertices() {
            parts[0].push(i);
        }
        return parts;
    }
	
    let parts_ids = hg.partition(num_parts);
    let mut parts = vec![vec![]; num_parts];
    for (i, part_id) in parts_ids.into_iter().enumerate() {
        parts[part_id].push(i);
    }
    parts
}

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Gate-level verilog path synthesized in our provided library.
    ///
    /// If your design is still at RTL level, you should synthesize it
    /// in yosys first.
    netlist_verilog: PathBuf,
    /// Top module type in netlist to analyze.
    ///
    /// If not specified, we will guess it from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,
    /// Level split thresholds.
    #[clap(long, value_delimiter=',')]
    level_split: Vec<usize>,
    /// Output path for the serialized partitions.
    parts_out: PathBuf,
    /// The maximum allowance of layers for merging-induced degradations.
    ///
    /// By default is 0, meaning no degradation is allowed.
    #[clap(long, default_value_t=0)]
    max_stage_degrad: usize,
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    // Detect cell library
    let lib = detect_library_from_file(&args.netlist_verilog)
        .expect("Failed to read netlist file");
    clilog::info!("Detected cell library: {}", lib);

    if lib == CellLibrary::Mixed {
        panic!("Mixed AIGPDK and SKY130 cells in netlist not supported");
    }

    // Use appropriate LeafPinProvider based on detected library
    let netlistdb = match lib {
        CellLibrary::SKY130 => {
            NetlistDB::from_sverilog_file(
                &args.netlist_verilog,
                args.top_module.as_deref(),
                &SKY130LeafPins
            ).expect("cannot build netlist")
        }
        CellLibrary::AIGPDK | CellLibrary::Mixed => {
            NetlistDB::from_sverilog_file(
                &args.netlist_verilog,
                args.top_module.as_deref(),
                &AIGPDKLeafPins()
            ).expect("cannot build netlist")
        }
    };

    let aig = AIG::from_netlistdb(&netlistdb);
    println!("netlist has {} pins, {} aig pins, {} and gates",
             netlistdb.num_pins, aig.num_aigpins, aig.and_gate_cache.len());

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let stages_effective_parts = stageds.iter().map(|&(l, r, ref staged)| {
        clilog::info!("interactive partitioning stage {}-{}", l, match r {
            usize::MAX => "max".to_string(),
            r @ _ => format!("{}", r)
        });

        let mut parts_good: Vec<(Vec<usize>, Partition)> = Vec::new();
        // always made sure that staged output pins are at fronts.
        let mut unrealized_endpoints = (0..staged.num_endpoint_groups()).collect::<Vec<_>>();
        let mut division = 600;

        while !unrealized_endpoints.is_empty() {
            division = (division / 2).max(1);
            let num_parts = (unrealized_endpoints.len() + division - 1) / division;
            clilog::info!("current: {} endpoints, try {} parts", unrealized_endpoints.len(), num_parts);
            let staged_ur = staged.to_endpoint_subset(&unrealized_endpoints);
            let hg_ur = RCHyperGraph::from_staged_aig(&aig, &staged_ur);
            let mut parts_indices = run_par(&hg_ur, num_parts);
            for idcs in &mut parts_indices {
                for i in idcs {
                    *i = unrealized_endpoints[*i];
                }
            }
            let parts_try = parts_indices.par_iter()
                .map(|endpts| Partition::build_one(&aig, staged, endpts))
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

        clilog::info!("interactive partition completed: {} in total. merging started.",
                      parts_good.len());

        let (parts_indices_good, prebuilt): (Vec<_>, Vec<_>) = parts_good.into_iter().unzip();
        let effective_parts = process_partitions(
            &aig, staged, parts_indices_good, Some(prebuilt), args.max_stage_degrad
        ).unwrap();
        clilog::info!("after merging: {} parts.", effective_parts.len());
        effective_parts
    }).collect::<Vec<_>>();

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stages_effective_parts).unwrap();
}