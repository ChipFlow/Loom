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
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom, Write};
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

    /// Output events JSON file for UART TX decoding.
    #[clap(long)]
    output_events: Option<PathBuf>,

    /// UART baud rate (default: 115200).
    #[clap(long, default_value = "115200")]
    baud_rate: u32,

    /// UART TX GPIO index (default: 6 for Caravel).
    #[clap(long, default_value = "6")]
    uart_tx_gpio: usize,

    /// Firmware binary to load into QSPI flash for functional simulation.
    #[clap(long)]
    firmware: Option<PathBuf>,

    /// Firmware offset in flash (default: 0x100000 for ChipFlow).
    #[clap(long, default_value = "1048576")]
    firmware_offset: usize,

    /// Flash clock GPIO index (default: 0 for Caravel).
    #[clap(long, default_value = "0")]
    flash_clk_gpio: usize,

    /// Flash CSN GPIO index (default: 1 for Caravel).
    #[clap(long, default_value = "1")]
    flash_csn_gpio: usize,

    /// Flash D0 GPIO index (default: 2 for Caravel).
    #[clap(long, default_value = "2")]
    flash_d0_gpio: usize,
}

/// UART TX decoder state machine.
#[derive(Debug, Clone, Copy, PartialEq)]
enum UartState {
    Idle,
    StartBit { start_cycle: usize },
    DataBits { start_cycle: usize, bits_received: u8, value: u8 },
    StopBit { start_cycle: usize, value: u8 },
}

/// Decoded UART event.
#[derive(Debug, Serialize)]
struct UartEvent {
    timestamp: usize,
    peripheral: String,
    event: String,
    payload: u8,
}

/// QSPI Flash simulator for functional simulation.
struct QspiFlash {
    data: Vec<u8>,
    // State
    last_clk: bool,
    last_csn: bool,
    bit_count: u8,
    byte_count: u32,
    curr_byte: u8,
    out_buffer: u8,
    command: u8,
    addr: u32,
    data_width: u8, // 1 for single SPI, 4 for quad
}

impl QspiFlash {
    fn new() -> Self {
        // 16MB flash, initialized to 0xFF (erased state)
        Self {
            data: vec![0xFF; 16 * 1024 * 1024],
            last_clk: false,
            last_csn: true,
            bit_count: 0,
            byte_count: 0,
            curr_byte: 0,
            out_buffer: 0,
            command: 0,
            addr: 0,
            data_width: 1,
        }
    }

    fn load_firmware(&mut self, path: &std::path::Path, offset: usize) -> std::io::Result<usize> {
        use std::io::Read;
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let len = buf.len();
        if offset + len > self.data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Firmware too large for flash",
            ));
        }
        self.data[offset..offset + len].copy_from_slice(&buf);
        Ok(len)
    }

    fn process_byte(&mut self) {
        self.out_buffer = 0;
        if self.byte_count == 0 {
            // Command byte
            self.addr = 0;
            self.data_width = 1;
            self.command = self.curr_byte;
            match self.command {
                0xAB => {} // Power up
                0x03 | 0x9F | 0xFF | 0x35 | 0x31 | 0x50 | 0x05 | 0x01 | 0x06 => {} // Various
                0xEB => self.data_width = 4, // Quad read
                _ => {} // Ignore unknown commands
            }
        } else {
            match self.command {
                0x03 => {
                    // Single read: 3 address bytes, then data
                    if self.byte_count <= 3 {
                        self.addr |= (self.curr_byte as u32) << ((3 - self.byte_count) * 8);
                    }
                    if self.byte_count >= 3 {
                        let idx = (self.addr & 0x00FFFFFF) as usize;
                        self.out_buffer = if idx < self.data.len() {
                            self.data[idx]
                        } else {
                            0xFF
                        };
                        self.addr = self.addr.wrapping_add(1) & 0x00FFFFFF;
                    }
                }
                0xEB => {
                    // Quad read: 3 address bytes + 1 mode + 2 dummy, then data
                    if self.byte_count <= 3 {
                        self.addr |= (self.curr_byte as u32) << ((3 - self.byte_count) * 8);
                    }
                    if self.byte_count >= 6 {
                        let idx = (self.addr & 0x00FFFFFF) as usize;
                        self.out_buffer = if idx < self.data.len() {
                            self.data[idx]
                        } else {
                            0xFF
                        };
                        self.addr = self.addr.wrapping_add(1) & 0x00FFFFFF;
                    }
                }
                0x9F => {
                    // Read ID
                    const FLASH_ID: [u8; 4] = [0xCA, 0x7C, 0xA7, 0xFF];
                    self.out_buffer = FLASH_ID[(self.byte_count as usize) % FLASH_ID.len()];
                }
                _ => {}
            }
        }
    }

    /// Step the flash simulation. Returns the value to drive on d_i (4 bits).
    fn step(&mut self, clk: bool, csn: bool, d_o: u8) -> u8 {
        let mut d_i = 0u8;

        if csn && !self.last_csn {
            // Rising edge of CSN - deselect, reset state
            self.bit_count = 0;
            self.byte_count = 0;
            self.data_width = 1;
        } else if clk && !self.last_clk && !csn {
            // Rising clock edge while selected - sample input
            if self.data_width == 4 {
                self.curr_byte = (self.curr_byte << 4) | (d_o & 0xF);
            } else {
                self.curr_byte = (self.curr_byte << 1) | (d_o & 0x1);
            }
            self.out_buffer = self.out_buffer << self.data_width;
            self.bit_count += self.data_width;
            if self.bit_count >= 8 {
                self.process_byte();
                self.byte_count += 1;
                self.bit_count = 0;
            }
        } else if !clk && self.last_clk && !csn {
            // Falling clock edge while selected - output data
            if self.data_width == 4 {
                d_i = (self.out_buffer >> 4) & 0xF;
            } else {
                d_i = ((self.out_buffer >> 7) & 0x1) << 1; // MISO on d[1]
            }
        }

        self.last_clk = clk;
        self.last_csn = csn;
        d_i
    }
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

/// Trace a clock pin back through buffers/inverters to find the primary input (cell 0).
/// Returns Some(pinid) if a primary input is found, None otherwise.
fn trace_clock_to_primary_input(
    netlistdb: &NetlistDB,
    start_pinid: usize,
    cell_library: CellLibrary,
    verbose: bool,
) -> Option<usize> {
    let mut current_pinid = start_pinid;
    let mut visited = std::collections::HashSet::new();
    let mut depth = 0;

    loop {
        if visited.contains(&current_pinid) {
            if verbose {
                clilog::debug!("  Clock trace: cycle detected at depth {}", depth);
            }
            return None;
        }
        visited.insert(current_pinid);

        if visited.len() > 10000 {
            if verbose {
                clilog::debug!("  Clock trace: safety limit exceeded at depth {}", depth);
            }
            return None;
        }

        // If this is an input pin, follow the net to its driver
        if netlistdb.pindirect[current_pinid] == Direction::I {
            let netid = netlistdb.pin2net[current_pinid];
            if Some(netid) == netlistdb.net_zero || Some(netid) == netlistdb.net_one {
                if verbose {
                    clilog::debug!("  Clock trace: hit constant net at depth {}", depth);
                }
                return None;
            }

            // Find driver pin on the net
            let net_pins_start = netlistdb.net2pin.start[netid];
            let net_pins_end = if netid + 1 < netlistdb.net2pin.start.len() {
                netlistdb.net2pin.start[netid + 1]
            } else {
                netlistdb.net2pin.items.len()
            };

            let mut driver_pin = None;
            for &np in &netlistdb.net2pin.items[net_pins_start..net_pins_end] {
                // Check for output (driver) pin
                if netlistdb.pindirect[np] == Direction::O {
                    driver_pin = Some(np);
                    break;
                }
                // Check for primary input (cell 0)
                if netlistdb.pin2cell[np] == 0 {
                    driver_pin = Some(np);
                    break;
                }
            }

            match driver_pin {
                Some(dp) => {
                    current_pinid = dp;
                    depth += 1;
                }
                None => {
                    if verbose {
                        clilog::debug!("  Clock trace: no driver found for net {} at depth {}", netid, depth);
                    }
                    return None;
                }
            }
            continue;
        }

        // This is an output pin - check if it's from cell 0 (primary input)
        let cellid = netlistdb.pin2cell[current_pinid];
        if cellid == 0 {
            use netlistdb::GeneralPinName;
            if verbose && depth <= 5 {
                clilog::debug!(
                    "  Clock trace: found primary input {} at depth {}",
                    netlistdb.pinnames[current_pinid].dbg_fmt_pin(),
                    depth
                );
            }
            return Some(current_pinid);
        }

        // Check if this cell is a buffer/inverter that we can trace through
        let celltype = netlistdb.celltypes[cellid].as_str();

        let is_buffer_or_inv = match cell_library {
            CellLibrary::SKY130 => {
                let ct = extract_cell_type(celltype);
                ct.starts_with("inv")
                    || ct.starts_with("clkinv")
                    || ct.starts_with("buf")
                    || ct.starts_with("clkbuf")
                    || ct.starts_with("clkdlybuf")
            }
            _ => matches!(celltype, "INV" | "BUF"),
        };

        if !is_buffer_or_inv {
            if verbose && depth <= 5 {
                clilog::debug!(
                    "  Clock trace: hit non-buffer cell {} ({}) at depth {}",
                    cellid,
                    celltype,
                    depth
                );
            }
            return None;
        }

        // Find the input pin "A" of the buffer/inverter
        let mut input_pin = None;
        for ipin in netlistdb.cell2pin.iter_set(cellid) {
            if netlistdb.pindirect[ipin] == Direction::I {
                let pin_name = netlistdb.pinnames[ipin].1.as_str();
                if pin_name == "A" {
                    input_pin = Some(ipin);
                    break;
                }
            }
        }

        match input_pin {
            Some(ip) => {
                current_pinid = ip;
                depth += 1;
            }
            None => return None,
        }
    }
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
                    // Trace clock pin back through buffers/inverters to primary input
                    // Only log verbose for first few DFFs to avoid spam
                    let log_this = args.verbose && posedge_monitor.is_empty();
                    if let Some(primary_clk_pin) =
                        trace_clock_to_primary_input(&netlistdb, pinid, cell_library, log_this)
                    {
                        posedge_monitor.insert(primary_clk_pin);
                    } else if args.verbose && posedge_monitor.is_empty() {
                        clilog::debug!(
                            "Clock pin {} could not be traced to primary input",
                            pinid
                        );
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

    // UART TX monitoring setup
    let uart_tx_pin = if args.output_events.is_some() {
        // Find gpio_out[uart_tx_gpio] pin
        let gpio_out_name = format!("gpio_out[{}]", args.uart_tx_gpio);
        let mut found_pin = None;
        for pinid in 0..netlistdb.num_pins {
            if netlistdb.pin2cell[pinid] == 0 {
                // Primary IO
                let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                if pin_name.contains(&gpio_out_name) || pin_name.ends_with(&format!("gpio_out:{}", args.uart_tx_gpio)) {
                    found_pin = Some(pinid);
                    clilog::info!("Found UART TX on pin {}: {}", pinid, pin_name);
                    break;
                }
            }
        }
        if found_pin.is_none() {
            clilog::warn!("Could not find gpio_out[{}] for UART TX monitoring", args.uart_tx_gpio);
        }
        found_pin
    } else {
        None
    };

    let clock_hz = 1_000_000_000_000u64 / args.clock_period;
    let cycles_per_bit = (clock_hz / args.baud_rate as u64) as usize;
    let mut uart_state = UartState::Idle;
    let mut uart_events: Vec<UartEvent> = Vec::new();
    let mut uart_last_tx = 1u8; // UART idles high

    // Track last flash d_in value to re-apply after VCD overwrites
    let mut flash_last_d_in = 0xFu8; // Flash idles with all lines high

    if let Some(tx_pin) = uart_tx_pin {
        let has_aig_mapping = aig.pin2aigpin_iv.get(tx_pin).map_or(false, |&v| v != usize::MAX);
        clilog::info!(
            "UART monitoring: baud={}, clock={}Hz, cycles_per_bit={}, has_aig_mapping={}",
            args.baud_rate,
            clock_hz,
            cycles_per_bit,
            has_aig_mapping
        );
        if !has_aig_mapping {
            clilog::warn!("UART TX pin {} has no AIG mapping - output won't be tracked!", tx_pin);
        }
    }

    // QSPI Flash setup for functional simulation
    let mut flash = if args.firmware.is_some() {
        Some(QspiFlash::new())
    } else {
        None
    };

    // Helper to find gpio pin by index
    let find_gpio_pin = |gpio_type: &str, idx: usize| -> Option<usize> {
        let gpio_name = format!("{}[{}]", gpio_type, idx);
        for pinid in 0..netlistdb.num_pins {
            if netlistdb.pin2cell[pinid] == 0 {
                let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                if pin_name.contains(&gpio_name) || pin_name.ends_with(&format!("{}:{}", gpio_type, idx)) {
                    return Some(pinid);
                }
            }
        }
        None
    };

    // Find flash GPIO pins
    let flash_clk_out = find_gpio_pin("gpio_out", args.flash_clk_gpio);
    let flash_csn_out = find_gpio_pin("gpio_out", args.flash_csn_gpio);
    let flash_d_out: Vec<Option<usize>> = (0..4)
        .map(|i| find_gpio_pin("gpio_out", args.flash_d0_gpio + i))
        .collect();
    let flash_d_in: Vec<Option<usize>> = (0..4)
        .map(|i| find_gpio_pin("gpio_in", args.flash_d0_gpio + i))
        .collect();

    if let Some(ref mut fl) = flash {
        if let Some(fw_path) = &args.firmware {
            match fl.load_firmware(fw_path, args.firmware_offset) {
                Ok(size) => {
                    clilog::info!(
                        "Loaded {} bytes firmware from {:?} at offset 0x{:X}",
                        size,
                        fw_path,
                        args.firmware_offset
                    );
                }
                Err(e) => {
                    clilog::error!("Failed to load firmware: {}", e);
                    std::process::exit(1);
                }
            }
        }

        // Log flash pin mappings and AIG status
        clilog::info!(
            "Flash pins: clk={:?}, csn={:?}, d_out={:?}, d_in={:?}",
            flash_clk_out,
            flash_csn_out,
            flash_d_out,
            flash_d_in
        );

        // Check AIG mappings for flash pins
        if let Some(clk_pin) = flash_clk_out {
            let has_aig = aig.pin2aigpin_iv.get(clk_pin).map_or(false, |&v| v != usize::MAX);
            clilog::info!("Flash CLK pin {} AIG mapping: {}", clk_pin, has_aig);
        }
        if let Some(csn_pin) = flash_csn_out {
            let has_aig = aig.pin2aigpin_iv.get(csn_pin).map_or(false, |&v| v != usize::MAX);
            clilog::info!("Flash CSN pin {} AIG mapping: {}", csn_pin, has_aig);
        }
    }

    clilog::info!("Starting timing simulation...");

    // Pre-run: process initial VCD values and evaluate AIG to get correct initial state
    // This is needed so that on the first clock edge, DFFs latch the correct D values
    let mut initial_phase = true;
    let mut initial_inputs_set = false;

    while let Some(tok) = vcdflow.next_token().unwrap() {
        match tok {
            FastFlowToken::Timestamp(t) => {
                if t == vcd_time {
                    continue;
                }

                // Initial phase: after time 0 values are set, evaluate AIG to get correct D inputs
                if initial_phase && t > 0 && initial_inputs_set {
                    initial_phase = false;
                    clilog::debug!("Initial phase: evaluating AIG with reset state");

                    // Evaluate combinational logic to set up correct D values
                    for i in 1..=aig.num_aigpins {
                        match &aig.drivers[i] {
                            DriverType::AndGate(a, b) => {
                                state.eval_and(i, *a, *b);
                            }
                            DriverType::InputPort(pinid) => {
                                state.values[i] = circ_state[*pinid];
                            }
                            DriverType::DFF(_) | DriverType::InputClockFlag(_, _)
                            | DriverType::Tie0 | DriverType::SRAM(_) => {
                                // DFFs start at 0, others don't change
                            }
                        }
                    }

                    // Update circ_state from AIG
                    for (pinid, &aigpin_iv) in aig.pin2aigpin_iv.iter().enumerate() {
                        if aigpin_iv != usize::MAX {
                            let idx = aigpin_iv >> 1;
                            let inv = (aigpin_iv & 1) != 0;
                            if idx > 0 && idx <= aig.num_aigpins {
                                circ_state[pinid] = state.values[idx] ^ (inv as u8);
                            }
                        }
                    }

                    // Find reset signal value (gpio_in[40])
                    let reset_pin = find_gpio_pin("gpio_in", 40);
                    clilog::debug!(
                        "Initial: gpio_in[40] (reset) = {:?}, gpio_out[0] (flash clk) = {}, gpio_out[1] (csn) = {}",
                        reset_pin.map(|p| circ_state[p]),
                        flash_clk_out.map(|p| circ_state[p]).unwrap_or(0),
                        flash_csn_out.map(|p| circ_state[p]).unwrap_or(0)
                    );
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
                            let mut pinid_de = usize::MAX;  // Data enable for edfxtp
                            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                                match netlistdb.pinnames[pinid].1.as_str() {
                                    "D" => pinid_d = pinid,
                                    "Q" => pinid_q = pinid,
                                    "DE" => pinid_de = pinid,  // Enable input
                                    _ => {}
                                }
                            }
                            if pinid_d != usize::MAX && pinid_q != usize::MAX {
                                // For enable DFFs (edfxtp), only update Q if DE is high
                                let should_latch = if pinid_de != usize::MAX {
                                    circ_state[pinid_de] != 0
                                } else {
                                    true  // Regular DFFs always latch
                                };
                                if should_latch {
                                    circ_state[pinid_q] = circ_state[pinid_d];
                                }
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
                                // Find the Q pin for this cell and get its value
                                for pinid in netlistdb.cell2pin.iter_set(*cell_idx) {
                                    if netlistdb.pinnames[pinid].1.as_str() == "Q" {
                                        state.values[i] = circ_state[pinid];
                                        break;
                                    }
                                }
                                // The DFF output has clk-to-Q delay
                                state.arrivals[i] = state.delays[i].max_delay() as u64;
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

                    // Step QSPI flash simulation
                    if let Some(ref mut fl) = flash {
                        // Read flash interface outputs from design
                        let clk = flash_clk_out.map(|p| circ_state[p] != 0).unwrap_or(false);
                        let csn = flash_csn_out.map(|p| circ_state[p] != 0).unwrap_or(true);
                        let mut d_out = 0u8;
                        for (i, opt_pin) in flash_d_out.iter().enumerate() {
                            if let Some(pin) = opt_pin {
                                if circ_state[*pin] != 0 {
                                    d_out |= 1 << i;
                                }
                            }
                        }

                        // Debug: log flash activity in early cycles
                        if args.verbose && stats.cycles_simulated <= 25 {
                            let reset_val = find_gpio_pin("gpio_in", 40).map(|p| circ_state[p]).unwrap_or(255);

                            // Count how many non-zero values in circ_state (rough proxy for "active" state)
                            let nonzero_count = circ_state.iter().filter(|&&v| v != 0).count();

                            clilog::debug!(
                                "Cycle {}: reset={}, clk={}, csn={}, d_out=0x{:X}, byte_count={}, nonzero_pins={}",
                                stats.cycles_simulated, reset_val, clk, csn, d_out, fl.byte_count, nonzero_count
                            );
                        }

                        // Step flash and get response
                        let d_in = fl.step(clk, csn, d_out);
                        flash_last_d_in = d_in;

                        // Drive flash data inputs back into design
                        for (i, opt_pin) in flash_d_in.iter().enumerate() {
                            if let Some(pin) = opt_pin {
                                circ_state[*pin] = ((d_in >> i) & 1) as u8;
                            }
                        }
                    }

                    // UART TX decoding
                    if let Some(tx_pin) = uart_tx_pin {
                        let tx = circ_state[tx_pin];
                        let cycle = stats.cycles_simulated;

                        // Debug: print TX value periodically
                        if args.verbose && cycle % 1000 == 0 {
                            clilog::debug!("Cycle {}: UART TX = {}", cycle, tx);
                        }

                        uart_state = match uart_state {
                            UartState::Idle => {
                                if uart_last_tx == 1 && tx == 0 {
                                    // Falling edge - start bit detected
                                    UartState::StartBit { start_cycle: cycle }
                                } else {
                                    UartState::Idle
                                }
                            }
                            UartState::StartBit { start_cycle } => {
                                // Sample at middle of start bit
                                if cycle >= start_cycle + cycles_per_bit / 2 {
                                    if tx == 0 {
                                        // Valid start bit, move to data
                                        UartState::DataBits {
                                            start_cycle: start_cycle + cycles_per_bit,
                                            bits_received: 0,
                                            value: 0,
                                        }
                                    } else {
                                        // False start, go back to idle
                                        UartState::Idle
                                    }
                                } else {
                                    UartState::StartBit { start_cycle }
                                }
                            }
                            UartState::DataBits { start_cycle, bits_received, value } => {
                                // Sample at middle of each bit
                                let bit_center = start_cycle + (bits_received as usize) * cycles_per_bit + cycles_per_bit / 2;
                                if cycle >= bit_center {
                                    let new_value = value | ((tx as u8) << bits_received);
                                    if bits_received >= 7 {
                                        // All 8 bits received, expect stop bit
                                        UartState::StopBit {
                                            start_cycle: start_cycle + 8 * cycles_per_bit,
                                            value: new_value,
                                        }
                                    } else {
                                        UartState::DataBits {
                                            start_cycle,
                                            bits_received: bits_received + 1,
                                            value: new_value,
                                        }
                                    }
                                } else {
                                    UartState::DataBits { start_cycle, bits_received, value }
                                }
                            }
                            UartState::StopBit { start_cycle, value } => {
                                // Sample at middle of stop bit
                                if cycle >= start_cycle + cycles_per_bit / 2 {
                                    if tx == 1 {
                                        // Valid stop bit - record the byte
                                        uart_events.push(UartEvent {
                                            timestamp: cycle,
                                            peripheral: "uart_0".to_string(),
                                            event: "tx".to_string(),
                                            payload: value,
                                        });
                                        if args.verbose {
                                            let ch = if value >= 32 && value < 127 {
                                                value as char
                                            } else {
                                                '.'
                                            };
                                            clilog::info!(
                                                "UART TX @ cycle {}: 0x{:02X} '{}'",
                                                cycle, value, ch
                                            );
                                        }
                                    }
                                    UartState::Idle
                                } else {
                                    UartState::StopBit { start_cycle, value }
                                }
                            }
                        };
                        uart_last_tx = tx;
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
                // Re-apply flash d_in values after VCD might have overwritten them
                if flash.is_some() {
                    for (i, opt_pin) in flash_d_in.iter().enumerate() {
                        if let Some(pin) = opt_pin {
                            circ_state[*pin] = ((flash_last_d_in >> i) & 1) as u8;
                        }
                    }
                }
                // Mark that initial inputs have been set (for cycle 0)
                if initial_phase {
                    initial_inputs_set = true;
                }
            }
        }
    }

    // Write UART events if requested
    if let Some(output_path) = &args.output_events {
        clilog::info!("Captured {} UART TX events", uart_events.len());

        #[derive(Serialize)]
        struct EventsOutput {
            events: Vec<UartEvent>,
        }

        let output = EventsOutput { events: uart_events };
        let json = serde_json::to_string_pretty(&output).expect("Failed to serialize events");
        let mut file = File::create(output_path).expect("Failed to create events file");
        file.write_all(json.as_bytes()).expect("Failed to write events");
        clilog::info!("Wrote events to {:?}", output_path);
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
