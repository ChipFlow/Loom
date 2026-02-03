// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! CPU-based timing simulation reference implementation.
//!
//! This simulates gate-level netlists with per-gate delays, tracking arrival
//! times during simulation. It serves as a reference for validating the GPU
//! timing implementation.
//!
//! Usage:
//!   cargo run -r --bin timing_sim_cpu -- <netlist.gv> <input.vcd> [options]

use gem::aig::{DriverType, AIG};
use gem::aigpdk::AIGPDKLeafPins;
use gem::flatten::PackedDelay;
use gem::liberty_parser::TimingLibrary;
use gem::sky130::{detect_library_from_file, extract_cell_type, CellLibrary, SKY130LeafPins};
use netlistdb::{Direction, GeneralPinName, NetlistDB};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use sverilogparse::SVerilogRange;
use vcd_ng::{FFValueChange, FastFlow, FastFlowToken, Parser, Scope, ScopeItem, Var};

#[derive(clap::Parser, Debug)]
#[command(name = "timing_sim_cpu")]
#[command(about = "CPU timing simulation with per-gate delays")]
struct Args {
    /// Gate-level verilog path synthesized in AIGPDK or SKY130 library.
    netlist_verilog: PathBuf,

    /// VCD input signal path
    input_vcd: PathBuf,

    /// Top module type in netlist.
    #[clap(long)]
    top_module: Option<String>,

    /// The scope path of top module in the input VCD.
    #[clap(long)]
    input_vcd_scope: Option<String>,

    /// Clock period in picoseconds (default: 1000 = 1ns).
    #[clap(long, default_value = "1000")]
    clock_period: u64,

    /// Path to Liberty library file.
    #[clap(long)]
    liberty: Option<PathBuf>,

    /// Maximum cycles to simulate.
    #[clap(long)]
    max_cycles: Option<usize>,

    /// Report timing violations during simulation.
    #[clap(long)]
    report_violations: bool,

    /// Verbose output with per-cycle timing.
    #[clap(long)]
    verbose: bool,
}

/// Timing state for CPU simulation.
struct TimingState {
    /// Current logic value for each AIG pin (0 or 1).
    values: Vec<u8>,
    /// Arrival time for each AIG pin (in picoseconds).
    arrivals: Vec<u64>,
    /// Gate delays for each AIG pin.
    delays: Vec<PackedDelay>,
    /// Setup time for DFFs.
    setup_time_ps: u64,
    /// Hold time for DFFs.
    hold_time_ps: u64,
}

impl TimingState {
    fn new(num_aigpins: usize, lib: &TimingLibrary) -> Self {
        let dff_timing = lib.dff_timing();

        Self {
            values: vec![0; num_aigpins + 1],
            arrivals: vec![0; num_aigpins + 1],
            delays: vec![PackedDelay::default(); num_aigpins + 1],
            setup_time_ps: dff_timing.as_ref().map(|t| t.max_setup()).unwrap_or(0),
            hold_time_ps: dff_timing.as_ref().map(|t| t.max_hold()).unwrap_or(0),
        }
    }

    /// Initialize delays from AIG driver types.
    fn init_delays(&mut self, aig: &AIG, lib: &TimingLibrary) {
        let and_delay = lib.and_gate_delay("AND2_00_0").unwrap_or((1, 1));
        let dff_timing = lib.dff_timing();
        let sram_timing = lib.sram_timing();

        for i in 1..=aig.num_aigpins {
            let delay = match &aig.drivers[i] {
                DriverType::AndGate(_, _) => PackedDelay::from_u64(and_delay.0, and_delay.1),
                DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) | DriverType::Tie0 => {
                    PackedDelay::default()
                }
                DriverType::DFF(_) => dff_timing
                    .as_ref()
                    .map(|t| PackedDelay::from_u64(t.clk_to_q_rise_ps, t.clk_to_q_fall_ps))
                    .unwrap_or_default(),
                DriverType::SRAM(_) => sram_timing
                    .as_ref()
                    .map(|t| {
                        PackedDelay::from_u64(
                            t.read_clk_to_data_rise_ps,
                            t.read_clk_to_data_fall_ps,
                        )
                    })
                    .unwrap_or(PackedDelay::new(1, 1)),
            };
            self.delays[i] = delay;
        }
    }

    /// Evaluate an AND gate with timing.
    fn eval_and(&mut self, idx: usize, a_iv: usize, b_iv: usize) {
        let a_idx = a_iv >> 1;
        let b_idx = b_iv >> 1;
        let a_inv = (a_iv & 1) != 0;
        let b_inv = (b_iv & 1) != 0;

        let a_val = if a_idx == 0 {
            0
        } else {
            self.values[a_idx] ^ (a_inv as u8)
        };
        let b_val = if b_idx == 0 {
            0
        } else {
            self.values[b_idx] ^ (b_inv as u8)
        };

        self.values[idx] = a_val & b_val;

        // Arrival time is max of inputs plus gate delay
        let a_arr = if a_idx == 0 { 0 } else { self.arrivals[a_idx] };
        let b_arr = if b_idx == 0 { 0 } else { self.arrivals[b_idx] };
        let delay = self.delays[idx].max_delay() as u64;
        self.arrivals[idx] = a_arr.max(b_arr) + delay;
    }

    /// Reset arrival times to zero (for new cycle).
    fn reset_arrivals(&mut self) {
        for arr in &mut self.arrivals {
            *arr = 0;
        }
    }
}

/// Statistics from timing simulation.
#[derive(Debug, Default)]
struct TimingStats {
    cycles_simulated: usize,
    max_combinational_delay: u64,
    setup_violations: usize,
    hold_violations: usize,
    worst_setup_slack: i64,
    worst_hold_slack: i64,
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);

    let args = <Args as clap::Parser>::parse();
    clilog::info!("Timing simulation args:\n{:#?}", args);

    // Detect cell library
    let cell_library = detect_library_from_file(&args.netlist_verilog)
        .expect("Failed to read netlist file for library detection");
    clilog::info!("Detected cell library: {}", cell_library);

    if cell_library == CellLibrary::Mixed {
        panic!("Mixed AIGPDK and SKY130 cells in netlist not supported");
    }

    // Load Liberty library (or use defaults for SKY130)
    let lib = if let Some(lib_path) = &args.liberty {
        TimingLibrary::from_file(lib_path).expect("Failed to load Liberty library")
    } else if cell_library == CellLibrary::SKY130 {
        clilog::info!("Using default SKY130 timing values (no liberty file)");
        TimingLibrary::default_sky130()
    } else {
        TimingLibrary::load_aigpdk().expect("Failed to load default AIGPDK library")
    };
    clilog::info!("Loaded timing library: {}", lib.name);

    // Load netlist with appropriate LeafPinProvider
    clilog::info!("Loading netlist: {:?}", args.netlist_verilog);
    let netlistdb = match cell_library {
        CellLibrary::SKY130 => NetlistDB::from_sverilog_file(
            &args.netlist_verilog,
            args.top_module.as_deref(),
            &SKY130LeafPins,
        )
        .expect("Failed to build netlist"),
        CellLibrary::AIGPDK | CellLibrary::Mixed => NetlistDB::from_sverilog_file(
            &args.netlist_verilog,
            args.top_module.as_deref(),
            &AIGPDKLeafPins(),
        )
        .expect("Failed to build netlist"),
    };

    // Build AIG
    let aig = AIG::from_netlistdb(&netlistdb);
    clilog::info!(
        "AIG: {} pins, {} DFFs, {} SRAMs",
        aig.num_aigpins,
        aig.dffs.len(),
        aig.srams.len()
    );

    // Initialize timing state
    let mut state = TimingState::new(aig.num_aigpins, &lib);
    state.init_delays(&aig, &lib);

    // Identify clock ports for posedge detection
    let mut posedge_monitor = std::collections::HashSet::new();
    for cellid in 1..netlistdb.num_cells {
        let celltype = netlistdb.celltypes[cellid].as_str();

        // Check for DFF cells (AIGPDK or SKY130)
        let is_dff = match cell_library {
            CellLibrary::SKY130 => {
                let ct = extract_cell_type(celltype);
                matches!(
                    ct,
                    "dfxtp" | "dfrtp" | "dfrbp" | "dfstp" | "dfbbp" | "edfxtp" | "sdfxtp"
                )
            }
            _ => matches!(celltype, "DFF" | "DFFSR"),
        };

        // Check for SRAM cells
        let is_sram = match cell_library {
            CellLibrary::SKY130 => celltype.starts_with("CF_SRAM_"),
            _ => celltype == "$__RAMGEM_SYNC_",
        };

        if is_dff || is_sram {
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pin_name = netlistdb.pinnames[pinid].1.as_str();

                // Check for clock pins
                let is_clk = matches!(pin_name, "CLK" | "CLKin" | "PORT_R_CLK" | "PORT_W_CLK");

                if is_clk {
                    let netid = netlistdb.pin2net[pinid];
                    if Some(netid) == netlistdb.net_zero || Some(netid) == netlistdb.net_one {
                        continue;
                    }
                    let root = netlistdb.net2pin.items[netlistdb.net2pin.start[netid]];
                    if netlistdb.pin2cell[root] == 0 {
                        posedge_monitor.insert(root);
                    }
                }
            }
        }
    }

    // Parse VCD header
    let input_vcd = File::open(&args.input_vcd).expect("Failed to open VCD");
    let mut bufrd = BufReader::with_capacity(65536, input_vcd);
    let mut vcd_parser = Parser::new(&mut bufrd);
    let header = vcd_parser.parse_header().expect("Failed to parse VCD header");
    drop(vcd_parser);
    let mut vcd_file = bufrd.into_inner();
    vcd_file.seek(SeekFrom::Start(0)).unwrap();
    let mut vcdflow = FastFlow::new(vcd_file, 65536);

    // Find top scope in VCD
    let top_scope = find_top_scope(
        &header.items[..],
        args.input_vcd_scope.as_deref().unwrap_or(""),
    )
    .expect("Top scope not found in VCD");

    // Map VCD signals to netlist pins
    let mut vcd2inp = HashMap::new();
    for scope_item in &top_scope.children[..] {
        if let ScopeItem::Var(var) = scope_item {
            match_vcd_var_to_pins(&netlistdb, var, &mut vcd2inp);
        }
    }

    clilog::info!(
        "Mapped {} VCD signals, {} posedge clocks monitored",
        vcd2inp.len(),
        posedge_monitor.len()
    );

    // Simulation loop
    let mut stats = TimingStats::default();
    let mut vcd_time = u64::MAX;
    let mut last_rising_edge = false;
    let mut circ_state = vec![0u8; netlistdb.num_pins];

    // Initialize constant nets
    if let Some(netid) = netlistdb.net_one {
        for pinid in netlistdb.net2pin.iter_set(netid) {
            circ_state[pinid] = 1;
        }
    }

    clilog::info!("Starting timing simulation...");

    while let Some(tok) = vcdflow.next_token().unwrap() {
        match tok {
            FastFlowToken::Timestamp(t) => {
                if t == vcd_time {
                    continue;
                }

                if last_rising_edge {
                    stats.cycles_simulated += 1;

                    if let Some(max_cycles) = args.max_cycles {
                        if stats.cycles_simulated >= max_cycles {
                            clilog::info!("Reached max cycles: {}", max_cycles);
                            break;
                        }
                    }

                    // Reset arrival times for new cycle
                    state.reset_arrivals();

                    // Latch DFF values
                    for cellid in 1..netlistdb.num_cells {
                        let celltype = netlistdb.celltypes[cellid].as_str();
                        let is_dff = match cell_library {
                            CellLibrary::SKY130 => {
                                let ct = extract_cell_type(celltype);
                                matches!(
                                    ct,
                                    "dfxtp"
                                        | "dfrtp"
                                        | "dfrbp"
                                        | "dfstp"
                                        | "dfbbp"
                                        | "edfxtp"
                                        | "sdfxtp"
                                )
                            }
                            _ => matches!(celltype, "DFF" | "DFFSR"),
                        };

                        if is_dff {
                            let mut pinid_d = usize::MAX;
                            let mut pinid_q = usize::MAX;
                            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                                match netlistdb.pinnames[pinid].1.as_str() {
                                    "D" => pinid_d = pinid,
                                    "Q" => pinid_q = pinid,
                                    _ => {}
                                }
                            }
                            if pinid_d != usize::MAX && pinid_q != usize::MAX {
                                circ_state[pinid_q] = circ_state[pinid_d];
                            }
                        }
                    }

                    // Propagate combinational logic through AIG with timing
                    let mut max_arrival = 0u64;
                    for i in 1..=aig.num_aigpins {
                        match &aig.drivers[i] {
                            DriverType::AndGate(a, b) => {
                                state.eval_and(i, *a, *b);
                                max_arrival = max_arrival.max(state.arrivals[i]);
                            }
                            DriverType::InputPort(pinid) => {
                                // Get value from netlist state
                                state.values[i] = circ_state[*pinid];
                                state.arrivals[i] = 0; // Inputs arrive at t=0
                            }
                            DriverType::DFF(cell_idx) => {
                                // Get Q value from DFF (latched at cycle start)
                                if aig.dffs.contains_key(cell_idx) {
                                    // The DFF output has clk-to-Q delay
                                    state.arrivals[i] = state.delays[i].max_delay() as u64;
                                }
                            }
                            DriverType::InputClockFlag(_, _) | DriverType::Tie0 => {
                                state.arrivals[i] = 0;
                            }
                            DriverType::SRAM(_) => {
                                state.arrivals[i] = state.delays[i].max_delay() as u64;
                            }
                        }
                    }

                    stats.max_combinational_delay =
                        stats.max_combinational_delay.max(max_arrival);

                    // Check setup/hold for all DFFs
                    for (_cell_id, dff) in &aig.dffs {
                        let d_idx = dff.d_iv >> 1;
                        if d_idx > 0 && d_idx <= aig.num_aigpins {
                            let data_arrival = state.arrivals[d_idx];

                            // Setup check: data must arrive before clock_period - setup_time
                            let setup_slack = (args.clock_period as i64)
                                - (data_arrival as i64)
                                - (state.setup_time_ps as i64);

                            // Hold check: data must be stable for hold_time after clock
                            let hold_slack = (data_arrival as i64) - (state.hold_time_ps as i64);

                            if setup_slack < stats.worst_setup_slack || stats.cycles_simulated == 1
                            {
                                stats.worst_setup_slack = setup_slack;
                            }
                            if hold_slack < stats.worst_hold_slack || stats.cycles_simulated == 1 {
                                stats.worst_hold_slack = hold_slack;
                            }

                            if setup_slack < 0 {
                                stats.setup_violations += 1;
                                if args.report_violations {
                                    clilog::warn!(
                                        "Cycle {}: Setup violation, slack={}ps, data_arrival={}ps",
                                        stats.cycles_simulated,
                                        setup_slack,
                                        data_arrival
                                    );
                                }
                            }
                            if hold_slack < 0 {
                                stats.hold_violations += 1;
                                if args.report_violations {
                                    clilog::warn!(
                                        "Cycle {}: Hold violation, slack={}ps",
                                        stats.cycles_simulated,
                                        hold_slack
                                    );
                                }
                            }
                        }
                    }

                    if args.verbose && stats.cycles_simulated % 100 == 0 {
                        clilog::info!(
                            "Cycle {}: max_delay={}ps",
                            stats.cycles_simulated,
                            max_arrival
                        );
                    }

                    // Update netlist state from AIG for next iteration
                    for (pinid, &aigpin_iv) in aig.pin2aigpin_iv.iter().enumerate() {
                        if aigpin_iv != usize::MAX {
                            let idx = aigpin_iv >> 1;
                            let inv = (aigpin_iv & 1) != 0;
                            if idx > 0 && idx <= aig.num_aigpins {
                                circ_state[pinid] = state.values[idx] ^ (inv as u8);
                            }
                        }
                    }
                }

                vcd_time = t;
                last_rising_edge = false;
                for &clk in &posedge_monitor {
                    circ_state[clk] = 0;
                }
            }
            FastFlowToken::Value(FFValueChange { id, bits }) => {
                for (pos, &b) in bits.iter().enumerate() {
                    if let Some(&pin) = vcd2inp.get(&(id.0, pos)) {
                        if b == b'1' && posedge_monitor.contains(&pin) {
                            last_rising_edge = true;
                        }
                        circ_state[pin] = match b {
                            b'1' => 1,
                            _ => 0,
                        };
                    }
                }
            }
        }
    }

    // Print results
    println!();
    println!("=== Timing Simulation Results ===");
    println!("Cycles simulated: {}", stats.cycles_simulated);
    println!("Clock period: {} ps", args.clock_period);
    println!();
    println!("Max combinational delay: {} ps", stats.max_combinational_delay);
    println!(
        "Critical path slack: {} ps",
        args.clock_period as i64 - stats.max_combinational_delay as i64
    );
    println!();
    println!("Worst setup slack: {} ps", stats.worst_setup_slack);
    println!("Worst hold slack: {} ps", stats.worst_hold_slack);
    println!("Setup violations: {}", stats.setup_violations);
    println!("Hold violations: {}", stats.hold_violations);
    println!();

    if stats.setup_violations > 0 || stats.hold_violations > 0 {
        println!("TIMING: FAILED");
        std::process::exit(1);
    } else {
        println!("TIMING: PASSED");
    }
}

fn find_top_scope<'i>(items: &'i [ScopeItem], top_scope: &str) -> Option<&'i Scope> {
    for item in items {
        if let ScopeItem::Scope(scope) = item {
            if let Some(s1) = match_scope_path(top_scope, scope.identifier.as_str()) {
                return match s1 {
                    "" => Some(scope),
                    _ => find_top_scope(&scope.children[..], s1),
                };
            }
        }
    }
    None
}

fn match_scope_path<'i>(mut scope: &'i str, cur: &str) -> Option<&'i str> {
    if scope.is_empty() {
        return Some("");
    }
    if scope.starts_with('/') {
        scope = &scope[1..];
    }
    if scope.is_empty() {
        Some("")
    } else if scope.starts_with(cur) {
        if scope.len() == cur.len() {
            Some("")
        } else if scope.as_bytes()[cur.len()] == b'/' {
            Some(&scope[cur.len() + 1..])
        } else {
            None
        }
    } else {
        None
    }
}

fn match_vcd_var_to_pins(
    netlistdb: &NetlistDB,
    var: &Var,
    vcd2inp: &mut HashMap<(u64, usize), usize>,
) {
    let mut match_one_input = |i: Option<isize>, vcd_pos: usize| {
        use compact_str::CompactString;
        use netlistdb::HierName;

        let key = (
            HierName::single(CompactString::new_inline("")),
            var.reference.as_str(),
            i,
        );
        if let Some(&id) = netlistdb.pinname2id.get(&key as &dyn GeneralPinName) {
            // Direction::O means output from port (input to circuit)
            if netlistdb.pindirect[id] != Direction::O {
                return;
            }
            vcd2inp.insert((var.code.0, vcd_pos), id);
        }
    };

    use vcd_ng::ReferenceIndex::*;
    match var.index {
        None => match var.size {
            1 => match_one_input(None, 0),
            w => {
                for (pos, i) in (0..w).rev().enumerate() {
                    match_one_input(Some(i as isize), pos)
                }
            }
        },
        Some(BitSelect(i)) => match_one_input(Some(i as isize), 0),
        Some(Range(a, b)) => {
            for (pos, i) in SVerilogRange(a as isize, b as isize).enumerate() {
                match_one_input(Some(i), pos);
            }
        }
    }
}
