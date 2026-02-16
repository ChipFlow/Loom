// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Unified CLI for the Loom GPU-accelerated RTL simulator.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::AIG;
use gem::pe::{process_partitions, Partition};
use gem::repcut::RCHyperGraph;
use gem::sky130::{detect_library_from_file, CellLibrary, SKY130LeafPins};
use gem::staging::build_staged_aigs;
use netlistdb::NetlistDB;
use rayon::prelude::*;

#[derive(Parser)]
#[command(name = "loom", about = "Loom — GPU-accelerated RTL logic simulator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Map a synthesized gate-level netlist to a .gemparts partition file.
    ///
    /// This is the first step in the Loom workflow. The resulting .gemparts file
    /// is then used by the simulator (metal_test / cuda_test) to run the design
    /// on a GPU.
    Map(MapArgs),

    /// Run a GPU simulation (not yet implemented).
    ///
    /// Simulation logic has not yet been extracted into a reusable library.
    /// Use `metal_test` (macOS) or `cuda_test` (Linux/CUDA) directly for now.
    Sim(SimArgs),
}

#[derive(Parser)]
struct MapArgs {
    /// Gate-level Verilog path synthesized with the AIGPDK or SKY130 library.
    ///
    /// If your design is still at RTL level, you must synthesize it first.
    /// See usage.md for synthesis instructions.
    netlist_verilog: PathBuf,

    /// Output path for the serialized partition file (.gemparts).
    parts_out: PathBuf,

    /// Top module name in the netlist.
    ///
    /// If not specified, Loom guesses the top module from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,

    /// Level split thresholds for deep circuits.
    ///
    /// If mapping fails because a single endpoint cannot be partitioned,
    /// add level-split thresholds (e.g. --level-split 30 or --level-split 20,40).
    /// Remember to pass the same thresholds when simulating.
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,

    /// Maximum stage degradation layers allowed during partition merging.
    ///
    /// Default is 0, meaning no degradation is allowed.
    #[clap(long, default_value_t = 0)]
    max_stage_degrad: usize,
}

#[derive(Parser)]
struct SimArgs {
    /// Remaining arguments (currently unused).
    #[clap(trailing_var_arg = true, allow_hyphen_values = true)]
    _args: Vec<String>,
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

fn cmd_map(args: MapArgs) {
    clilog::info!("Loom map args:\n{:#?}", args.netlist_verilog);

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

    let aig = AIG::from_netlistdb(&netlistdb);
    println!(
        "netlist has {} pins, {} aig pins, {} and gates",
        netlistdb.num_pins,
        aig.num_aigpins,
        aig.and_gate_cache.len()
    );

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let stages_effective_parts = stageds
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
                let hg_ur = RCHyperGraph::from_staged_aig(&aig, &staged_ur);
                let mut parts_indices = run_par(&hg_ur, num_parts);
                for idcs in &mut parts_indices {
                    for i in idcs {
                        *i = unrealized_endpoints[*i];
                    }
                }
                let parts_try = parts_indices
                    .par_iter()
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

            clilog::info!(
                "interactive partition completed: {} in total. merging started.",
                parts_good.len()
            );

            let (parts_indices_good, prebuilt): (Vec<_>, Vec<_>) =
                parts_good.into_iter().unzip();
            let effective_parts = process_partitions(
                &aig,
                staged,
                parts_indices_good,
                Some(prebuilt),
                args.max_stage_degrad,
            )
            .unwrap();
            clilog::info!("after merging: {} parts.", effective_parts.len());
            effective_parts
        })
        .collect::<Vec<_>>();

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stages_effective_parts).unwrap();
}

fn cmd_sim(_args: SimArgs) {
    eprintln!(
        "\
loom sim is not yet implemented.

The simulation logic is currently embedded in the platform-specific binaries.
Please use the appropriate binary directly:

  Metal (macOS):
    cargo run -r --features metal --bin metal_test -- \\
      <netlist.gv> <parts.gemparts> <input.vcd> <output.vcd> <num_blocks>

  CUDA (Linux):
    cargo run -r --features cuda --bin cuda_test -- \\
      <netlist.gv> <parts.gemparts> <input.vcd> <output.vcd> <num_blocks>

A unified `loom sim` command will be available in a future release once the
simulation logic is extracted into reusable library modules."
    );
    std::process::exit(1);
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let cli = Cli::parse();

    match cli.command {
        Commands::Map(args) => cmd_map(args),
        Commands::Sim(args) => cmd_sim(args),
    }
}
