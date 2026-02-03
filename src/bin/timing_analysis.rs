// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Static Timing Analysis (STA) tool for GEM.
//!
//! Performs timing analysis on gate-level netlists synthesized in AIGPDK,
//! computing arrival times and checking setup/hold constraints.
//!
//! Usage:
//!   cargo run -r --bin timing_analysis -- <netlist.gv> [options]

use gem::aig::AIG;
use gem::aigpdk::AIGPDKLeafPins;
use gem::liberty_parser::TimingLibrary;
use netlistdb::NetlistDB;
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command(name = "timing_analysis")]
#[command(about = "Static timing analysis for AIGPDK gate-level netlists")]
struct Args {
    /// Gate-level verilog path synthesized in AIGPDK library.
    netlist_verilog: PathBuf,

    /// Top module type in netlist to analyze.
    /// If not specified, will be inferred from hierarchy.
    #[clap(long)]
    top_module: Option<String>,

    /// Clock period in picoseconds (default: 1000 = 1ns).
    #[clap(long, default_value = "1000")]
    clock_period: u64,

    /// Path to Liberty library file.
    /// If not specified, uses default AIGPDK library.
    #[clap(long)]
    liberty: Option<PathBuf>,

    /// Number of critical paths to report (default: 10).
    #[clap(long, default_value = "10")]
    num_paths: usize,

    /// Show detailed path trace for each critical path.
    #[clap(long)]
    trace_paths: bool,

    /// Report all timing violations (not just summary).
    #[clap(long)]
    report_violations: bool,

    /// Output format: text, json.
    #[clap(long, default_value = "text")]
    format: String,
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);

    let args = <Args as clap::Parser>::parse();
    clilog::info!("Timing analysis args:\n{:#?}", args);

    // Load Liberty library
    let lib = if let Some(lib_path) = &args.liberty {
        TimingLibrary::from_file(lib_path).expect("Failed to load Liberty library")
    } else {
        TimingLibrary::load_aigpdk().expect("Failed to load default AIGPDK library")
    };
    clilog::info!("Loaded Liberty library: {}", lib.name);

    // Load netlist
    clilog::info!("Loading netlist: {:?}", args.netlist_verilog);
    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins(),
    )
    .expect("Failed to build netlist");

    clilog::info!(
        "Netlist loaded: {} pins, {} cells",
        netlistdb.num_pins,
        netlistdb.num_cells
    );

    // Build AIG
    clilog::info!("Building AIG representation...");
    let mut aig = AIG::from_netlistdb(&netlistdb);
    clilog::info!(
        "AIG built: {} pins, {} DFFs, {} SRAMs",
        aig.num_aigpins,
        aig.dffs.len(),
        aig.srams.len()
    );

    // Load timing data
    aig.load_timing_library(&lib);
    aig.clock_period_ps = args.clock_period;

    // Run STA
    clilog::info!("Running static timing analysis...");
    let report = aig.compute_timing();

    // Output results
    match args.format.as_str() {
        "json" => print_json_report(&aig, &report, &args),
        _ => print_text_report(&aig, &report, &args, &netlistdb),
    }

    // Exit with error code if violations
    if report.has_violations() {
        std::process::exit(1);
    }
}

fn print_text_report(aig: &AIG, report: &gem::aig::TimingReport, args: &Args, netlistdb: &NetlistDB) {
    println!();
    println!("{}", report);

    // Print clock period info
    println!("Clock period: {} ps ({:.3} ns)", args.clock_period, args.clock_period as f64 / 1000.0);
    println!();

    // Print critical paths
    println!("=== Critical Paths (Top {}) ===", args.num_paths);
    let critical_paths = aig.get_critical_paths(args.num_paths);

    for (i, (endpoint, arrival)) in critical_paths.iter().enumerate() {
        // Try to find the name from netlistdb
        let name = find_pin_name(netlistdb, aig, *endpoint);
        println!(
            "#{}: endpoint={} (aigpin {}) arrival={} ps, slack={} ps",
            i + 1,
            name,
            endpoint,
            arrival,
            args.clock_period as i64 - *arrival as i64
        );

        if args.trace_paths {
            let path = aig.trace_critical_path(*endpoint);
            println!("    Path trace:");
            for (j, (node, arr)) in path.iter().enumerate() {
                let node_name = find_pin_name(netlistdb, aig, *node);
                let driver = match &aig.drivers[*node] {
                    gem::aig::DriverType::AndGate(_, _) => "AND",
                    gem::aig::DriverType::InputPort(_) => "INPUT",
                    gem::aig::DriverType::DFF(_) => "DFF_Q",
                    gem::aig::DriverType::SRAM(_) => "SRAM",
                    gem::aig::DriverType::InputClockFlag(_, _) => "CLK_FLAG",
                    gem::aig::DriverType::Tie0 => "TIE0",
                };
                println!("      [{:3}] {} ({}) @ {} ps", j, node_name, driver, arr);
            }
        }
    }
    println!();

    // Report violations if requested
    if args.report_violations && report.has_violations() {
        println!("=== Timing Violations ===");

        for (i, ((_cell_id, dff), (&setup_slack, &hold_slack))) in aig
            .dffs
            .iter()
            .zip(aig.setup_slacks.iter().zip(aig.hold_slacks.iter()))
            .enumerate()
        {
            if setup_slack < 0 || hold_slack < 0 {
                let d_idx = dff.d_iv >> 1;
                let name = find_pin_name(netlistdb, aig, d_idx);
                println!("DFF #{}: D={}", i, name);
                if setup_slack < 0 {
                    println!("  SETUP VIOLATION: slack = {} ps", setup_slack);
                }
                if hold_slack < 0 {
                    println!("  HOLD VIOLATION: slack = {} ps", hold_slack);
                }
            }
        }
        println!();
    }

    // Summary
    if report.has_violations() {
        println!("TIMING ANALYSIS: FAILED ({} setup, {} hold violations)",
                 report.setup_violations, report.hold_violations);
    } else {
        println!("TIMING ANALYSIS: PASSED");
    }
}

fn print_json_report(aig: &AIG, report: &gem::aig::TimingReport, args: &Args) {
    use std::collections::HashMap;

    let mut json: HashMap<String, serde_json::Value> = HashMap::new();

    json.insert(
        "clock_period_ps".to_string(),
        serde_json::json!(args.clock_period),
    );
    json.insert(
        "num_endpoints".to_string(),
        serde_json::json!(report.num_endpoints),
    );
    json.insert(
        "critical_path_delay_ps".to_string(),
        serde_json::json!(report.critical_path_delay),
    );
    json.insert(
        "worst_setup_slack_ps".to_string(),
        serde_json::json!(report.worst_setup_slack),
    );
    json.insert(
        "worst_hold_slack_ps".to_string(),
        serde_json::json!(report.worst_hold_slack),
    );
    json.insert(
        "setup_violations".to_string(),
        serde_json::json!(report.setup_violations),
    );
    json.insert(
        "hold_violations".to_string(),
        serde_json::json!(report.hold_violations),
    );
    json.insert(
        "timing_met".to_string(),
        serde_json::json!(!report.has_violations()),
    );

    // Critical paths
    let paths: Vec<_> = aig
        .get_critical_paths(args.num_paths)
        .iter()
        .map(|(ep, arr)| {
            serde_json::json!({
                "endpoint_aigpin": ep,
                "arrival_ps": arr,
                "slack_ps": args.clock_period as i64 - *arr as i64
            })
        })
        .collect();
    json.insert("critical_paths".to_string(), serde_json::json!(paths));

    println!("{}", serde_json::to_string_pretty(&json).unwrap());
}

/// Try to find a human-readable name for an AIG pin by looking through the netlistdb mapping.
fn find_pin_name(netlistdb: &NetlistDB, aig: &AIG, aigpin: usize) -> String {
    use netlistdb::GeneralPinName;

    // Search pin2aigpin_iv for a match
    for (pinid, &aigpin_iv) in aig.pin2aigpin_iv.iter().enumerate() {
        if aigpin_iv >> 1 == aigpin {
            return netlistdb.pinnames[pinid].dbg_fmt_pin();
        }
    }

    // Fallback to aigpin number
    format!("aigpin_{}", aigpin)
}
