// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! And-inverter graph format
//!
//! An AIG is derived from netlistdb synthesized in AIGPDK or SKY130.

use crate::aigpdk::AIGPDK_SRAM_ADDR_WIDTH;
use crate::sky130::{extract_cell_type, is_sky130_cell};
use crate::sky130_pdk::{
    decompose_with_pdk, is_multi_output_cell, is_sequential_cell, is_tie_cell, CellInputs,
    PdkModels,
};
use indexmap::{IndexMap, IndexSet};
use netlistdb::{Direction, GeneralPinName, NetlistDB};
use smallvec::SmallVec;

/// Work item for iterative DFS traversal.
/// Using two-phase approach: Visit pushes dependencies, Process computes outputs.
#[derive(Clone, Copy)]
enum WorkItem {
    /// First phase: check if visited, mark in-stack, push Process then dependencies
    Visit(usize),
    /// Second phase: all dependencies ready, compute output, clear in-stack
    Process(usize),
}

/// A DFF.
#[derive(Debug, Default, Clone)]
pub struct DFF {
    /// The D input pin with invert (last bit)
    pub d_iv: usize,
    /// If the DFF is enabled, i.e., if the clock, S, or R is active.
    pub en_iv: usize,
    /// The Q pin output with invert.
    pub q: usize,
}

/// Simulation control type for $stop and $finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimControlType {
    /// $stop - pause simulation
    Stop,
    /// $finish - terminate simulation
    Finish,
}

/// A simulation control node for $stop/$finish system tasks.
/// These are triggered when their condition input is true.
#[derive(Debug, Default, Clone)]
pub struct SimControlNode {
    /// The control type (Stop or Finish)
    pub control_type: Option<SimControlType>,
    /// The condition input pin with invert (last bit)
    /// When this evaluates to 1, the control action fires.
    pub condition_iv: usize,
    /// Optional message ID for display purposes
    pub message_id: u32,
}

/// A display node for $display/$write system tasks.
/// These output formatted messages when triggered.
#[derive(Debug, Default, Clone)]
pub struct DisplayNode {
    /// The enable condition input pin with invert (last bit)
    /// When this evaluates to 1, the display fires.
    pub enable_iv: usize,
    /// The clock enable signal (posedge trigger)
    pub clken_iv: usize,
    /// The format string for this display
    pub format: String,
    /// The argument signals (each entry is aigpin_iv)
    pub args_iv: Vec<usize>,
    /// The bit widths of each argument
    pub arg_widths: Vec<u32>,
    /// The cell name (for matching with JSON attributes)
    pub cell_name: String,
}

/// A ram block resembling the interface of `$__RAMGEM_SYNC_`.
#[derive(Debug, Default, Clone)]
pub struct RAMBlock {
    pub port_r_addr_iv: [usize; AIGPDK_SRAM_ADDR_WIDTH],

    /// controls whether r_rd_data should update. (from read clock)
    pub port_r_en_iv: usize,
    pub port_r_rd_data: [usize; 32],

    pub port_w_addr_iv: [usize; AIGPDK_SRAM_ADDR_WIDTH],
    /// controls whether memory should be updated.
    ///
    /// this is a combination of write enable and write clock.
    pub port_w_wr_en_iv: [usize; 32],
    pub port_w_wr_data_iv: [usize; 32],
}

/// A type of endpoint group. can be a primary output-related pin,
/// a D flip-flop, a ram block, or a simulation control node.
///
/// A group means a task for the partition to complete.
/// For primary output pins, the task is just to store.
/// For DFFs, the task is to store only when the clock is enable.
/// For RAMBlocks, the task is to simulate a sync SRAM.
/// For SimControl, the task is to check the condition and fire an event.
/// A StagedIOPin indicates a temporary live pin between different
/// major stages but reside in the same simulated cycle.
#[derive(Debug, Copy, Clone)]
pub enum EndpointGroup<'i> {
    PrimaryOutput(usize),
    DFF(&'i DFF),
    RAMBlock(&'i RAMBlock),
    SimControl(&'i SimControlNode),
    Display(&'i DisplayNode),
    StagedIOPin(usize),
}

impl EndpointGroup<'_> {
    /// Enumerate all related aigpin inputs for this endpoint group.
    ///
    /// The enumerated inputs may have duplicates.
    pub fn for_each_input(self, mut f_nz: impl FnMut(usize)) {
        let mut f = |i| {
            if i >= 1 {
                f_nz(i);
            }
        };
        match self {
            Self::PrimaryOutput(idx) => f(idx >> 1),
            Self::DFF(dff) => {
                f(dff.en_iv >> 1);
                f(dff.d_iv >> 1);
            }
            Self::RAMBlock(ram) => {
                f(ram.port_r_en_iv >> 1);
                for i in 0..13 {
                    f(ram.port_r_addr_iv[i] >> 1);
                    f(ram.port_w_addr_iv[i] >> 1);
                }
                for i in 0..32 {
                    f(ram.port_w_wr_en_iv[i] >> 1);
                    f(ram.port_w_wr_data_iv[i] >> 1);
                }
            }
            Self::SimControl(ctrl) => {
                f(ctrl.condition_iv >> 1);
            }
            Self::Display(disp) => {
                f(disp.enable_iv >> 1);
                for &arg_iv in &disp.args_iv {
                    f(arg_iv >> 1);
                }
            }
            Self::StagedIOPin(idx) => f(idx),
        }
    }
}

/// The driver type of an AIG pin.
#[derive(Debug, Clone)]
pub enum DriverType {
    /// Driven by an and gate.
    ///
    /// The inversion bit is stored as the last bits in
    /// two input indices.
    ///
    /// Only this type has combinational fan-in.
    AndGate(usize, usize),
    /// Driven by a primary input port (with its netlistdb id).
    InputPort(usize),
    /// Driven by a clock flag (with clock port netlistdb id, and pos/negedge)
    InputClockFlag(usize, u8),
    /// Driven by a DFF (with its index)
    DFF(usize),
    /// Driven by a 13-bit by 32-bit RAM block (with its index)
    SRAM(usize),
    /// Tie0: tied to zero. Only the 0-th aig pin is allowed to have this.
    Tie0,
}

/// An AIG associated with a netlistdb.
#[derive(Debug)]
pub struct AIG {
    /// The number of AIG pins.
    ///
    /// This number might be smaller than num_pins in netlistdb,
    /// because inverters and buffers are merged when possible.
    /// It might also be larger because we may add mux circuits.
    ///
    /// AIG pins are numbered from 1 to num_aigpins inclusive.
    /// The AIG pin id zero (0) is tied to 0.
    ///
    /// AIG pins are guaranteed to have topological order.
    pub num_aigpins: usize,
    /// The mapping from a netlistdb pin to an AIG pin.
    ///
    /// The inversion bit is stored as the last bit.
    /// E.g., `pin2aigpin_iv[pin_id] = aigpin_id << 1 | invert`.
    pub pin2aigpin_iv: Vec<usize>,
    /// The clock pins map. Every clock pin has a pair of flag pins
    /// showing if they are posedge/negedge.
    ///
    /// The flag pin can be empty which means the circuit is not
    /// active with that edge.
    pub clock_pin2aigpins: IndexMap<usize, (usize, usize)>,
    /// The driver types of AIG pins.
    pub drivers: Vec<DriverType>,
    /// A cache for identical and gates.
    pub and_gate_cache: IndexMap<(usize, usize), usize>,
    /// Unique primary output aigpin indices
    pub primary_outputs: IndexSet<usize>,
    /// The D flip-flops (DFFs), indexed by cell id
    pub dffs: IndexMap<usize, DFF>,
    /// The SRAMs, indexed by cell id
    pub srams: IndexMap<usize, RAMBlock>,
    /// The simulation control nodes ($stop/$finish), indexed by cell id
    pub simcontrols: IndexMap<usize, SimControlNode>,
    /// The display nodes ($display/$write), indexed by cell id
    pub displays: IndexMap<usize, DisplayNode>,
    /// The fanout CSR start array.
    pub fanouts_start: Vec<usize>,
    /// The fanout CSR array.
    pub fanouts: Vec<usize>,

    // === Timing Analysis Fields ===
    /// Arrival times for each AIG pin: (min_ps, max_ps).
    /// Computed during STA traversal. Index 0 is unused (Tie0 has arrival 0).
    /// Empty until `compute_timing()` is called.
    pub arrival_times: Vec<(u64, u64)>,

    /// Gate delays for each AIG pin: (rise_ps, fall_ps).
    /// Loaded from Liberty library. Used during timing propagation.
    pub gate_delays: Vec<(u64, u64)>,

    /// Setup slack for each DFF (indexed by position in `dffs`).
    /// Negative values indicate timing violations.
    /// Computed as: clock_arrival - data_arrival - setup_time
    pub setup_slacks: Vec<i64>,

    /// Hold slack for each DFF (indexed by position in `dffs`).
    /// Negative values indicate timing violations.
    /// Computed as: data_arrival - clock_arrival - hold_time
    pub hold_slacks: Vec<i64>,

    /// DFF timing constraints loaded from Liberty.
    pub dff_timing: Option<crate::liberty_parser::DFFTiming>,

    /// Clock period in picoseconds (for STA calculations).
    /// Default is 1000ps (1ns) if not specified.
    pub clock_period_ps: u64,

    // === SDF Back-Annotation Support ===
    /// Maps AIG pin index → list of (netlistdb cell_id, cell_type, output_pin_name).
    /// Accumulated: inverters sharing an AIG pin (via invert-bit reuse) push
    /// their origins rather than overwriting. During SDF loading, delays from
    /// all origins are summed (they form a serial chain, not parallel paths).
    /// Internal AND gates from multi-gate decompositions get empty Vec (zero delay).
    /// Index 0 is Tie0 (empty). Populated during AIG construction from netlistdb.
    pub aigpin_cell_origins: Vec<Vec<(usize, String, String)>>,
}

impl Default for AIG {
    fn default() -> Self {
        Self {
            num_aigpins: 0,
            pin2aigpin_iv: Vec::new(),
            clock_pin2aigpins: IndexMap::new(),
            drivers: Vec::new(),
            and_gate_cache: IndexMap::new(),
            primary_outputs: IndexSet::new(),
            dffs: IndexMap::new(),
            srams: IndexMap::new(),
            simcontrols: IndexMap::new(),
            displays: IndexMap::new(),
            fanouts_start: Vec::new(),
            fanouts: Vec::new(),
            // Timing fields
            arrival_times: Vec::new(),
            gate_delays: Vec::new(),
            setup_slacks: Vec::new(),
            hold_slacks: Vec::new(),
            dff_timing: None,
            clock_period_ps: 1000, // Default 1ns clock period
            aigpin_cell_origins: Vec::new(),
        }
    }
}

impl AIG {
    fn add_aigpin(&mut self, driver: DriverType) -> usize {
        self.num_aigpins += 1;
        self.drivers.push(driver);
        self.aigpin_cell_origins.push(Vec::new());
        self.num_aigpins
    }

    fn add_and_gate(&mut self, a: usize, b: usize) -> usize {
        assert_ne!(a | 1, usize::MAX);
        assert_ne!(b | 1, usize::MAX);
        if a == 0 || b == 0 {
            return 0;
        }
        if a == 1 {
            return b;
        }
        if b == 1 {
            return a;
        }
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        if let Some(o) = self.and_gate_cache.get(&(a, b)) {
            return o << 1;
        }
        let aigpin = self.add_aigpin(DriverType::AndGate(a, b));
        self.and_gate_cache.insert((a, b), aigpin);
        aigpin << 1
    }

    /// given a clock pin, trace back to clock root and return its
    /// enable signal (with invert bit).
    ///
    /// if result is 0, that means the pin is dangled.
    /// if an error occurs because of a undecipherable multi-input cell,
    /// we will return in error the last output pin index of that cell.
    fn trace_clock_pin(
        &mut self,
        netlistdb: &NetlistDB,
        start_pinid: usize,
        start_is_negedge: bool,
        // should we ignore cklnqd in this tracing.
        // if set to true, we will treat cklnqd as a simple buffer.
        // otherwise, we assert that cklnqd/en is already built in
        // our aig mapping (pin2aigpin_iv).
        ignore_cklnqd: bool,
    ) -> Result<usize, usize> {
        // Iterative implementation using explicit stack
        // Each entry: (pinid, is_negedge, is_processing_cklnqd, cklnqd_cp_result)
        // For CKLNQD, we need to first trace CP, then combine with EN
        let mut current_pinid = start_pinid;
        let mut current_is_negedge = start_is_negedge;

        // Stack for CKLNQD cells that need post-processing
        // (pinid of EN pin, is_negedge at that point)
        let mut cklnqd_stack: Vec<(usize, bool)> = Vec::new();

        let mut iteration = 0;
        let mut visited_pins: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut path: Vec<(usize, String)> = Vec::new();
        loop {
            iteration += 1;
            {
                let pin_name = netlistdb.pinnames[current_pinid].dbg_fmt_pin();
                let dir = if netlistdb.pindirect[current_pinid] == Direction::I {
                    "I"
                } else {
                    "O"
                };
                path.push((current_pinid, format!("{}:{}", pin_name, dir)));
            }
            if visited_pins.contains(&current_pinid) {
                panic!(
                    "trace_clock_pin cycle detected!\nstart={}, iteration={}\npath:\n{}",
                    start_pinid,
                    iteration,
                    path.iter()
                        .enumerate()
                        .map(|(i, (pid, name))| format!("  {:3}: pin {} - {}", i, pid, name))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
            }
            visited_pins.insert(current_pinid);

            if iteration > 100000 {
                panic!(
                    "trace_clock_pin infinite loop detected: pinid={}, start={}, iteration={}",
                    current_pinid, start_pinid, iteration
                );
            }
            let pinid = current_pinid;
            let is_negedge = current_is_negedge;

            if netlistdb.pindirect[pinid] == Direction::I {
                let netid = netlistdb.pin2net[pinid];
                if Some(netid) == netlistdb.net_zero || Some(netid) == netlistdb.net_one {
                    // Reached a constant - process any pending CKLNQD
                    let mut result = 0usize;
                    while let Some((en_pin, _)) = cklnqd_stack.pop() {
                        if ignore_cklnqd {
                            // Just return the traced clock, ignore enable
                        } else {
                            let en_iv = self.pin2aigpin_iv[en_pin];
                            assert_ne!(en_iv, usize::MAX, "clken not built");
                            result = self.add_and_gate(result, en_iv);
                        }
                    }
                    return Ok(result);
                }
                // Follow net to driving pin - find the output pin (driver) on the net
                let net_pins_start = netlistdb.net2pin.start[netid];
                let net_pins_end = if netid + 1 < netlistdb.net2pin.start.len() {
                    netlistdb.net2pin.start[netid + 1]
                } else {
                    netlistdb.net2pin.items.len()
                };
                let mut driver_pin = None;
                for &np in &netlistdb.net2pin.items[net_pins_start..net_pins_end] {
                    // Check if this pin is an output (driver)
                    if netlistdb.pindirect[np] == Direction::O {
                        driver_pin = Some(np);
                        break;
                    }
                    // Also check for primary input (cell 0)
                    if netlistdb.pin2cell[np] == 0 {
                        driver_pin = Some(np);
                        break;
                    }
                }
                if let Some(dp) = driver_pin {
                    current_pinid = dp;
                    continue;
                }
                // Net has no driver - treat this input pin as a primary input (clock source)
                // This handles floating nets / undriven primary inputs
                // Create a clock signal for this undriven net
                let clkentry = self
                    .clock_pin2aigpins
                    .entry(pinid)
                    .or_insert((usize::MAX, usize::MAX));
                let clksignal = match is_negedge {
                    false => clkentry.0,
                    true => clkentry.1,
                };
                let mut result = if clksignal != usize::MAX {
                    clksignal << 1
                } else {
                    let aigpin =
                        self.add_aigpin(DriverType::InputClockFlag(pinid, is_negedge as u8));
                    let clkentry = self.clock_pin2aigpins.get_mut(&pinid).unwrap();
                    let clksignal = match is_negedge {
                        false => &mut clkentry.0,
                        true => &mut clkentry.1,
                    };
                    *clksignal = aigpin;
                    aigpin << 1
                };

                // Process any pending CKLNQD
                while let Some((en_pin, _)) = cklnqd_stack.pop() {
                    if !ignore_cklnqd {
                        let en_iv = self.pin2aigpin_iv[en_pin];
                        assert_ne!(en_iv, usize::MAX, "clken not built");
                        result = self.add_and_gate(result, en_iv);
                    }
                }
                return Ok(result);
            }

            let cellid = netlistdb.pin2cell[pinid];
            if cellid == 0 {
                // Reached primary input - this is the clock source
                let clkentry = self
                    .clock_pin2aigpins
                    .entry(pinid)
                    .or_insert((usize::MAX, usize::MAX));
                let clksignal = match is_negedge {
                    false => clkentry.0,
                    true => clkentry.1,
                };
                let mut result = if clksignal != usize::MAX {
                    clksignal << 1
                } else {
                    let aigpin =
                        self.add_aigpin(DriverType::InputClockFlag(pinid, is_negedge as u8));
                    let clkentry = self.clock_pin2aigpins.get_mut(&pinid).unwrap();
                    let clksignal = match is_negedge {
                        false => &mut clkentry.0,
                        true => &mut clkentry.1,
                    };
                    *clksignal = aigpin;
                    aigpin << 1
                };

                // Process any pending CKLNQD
                while let Some((en_pin, _)) = cklnqd_stack.pop() {
                    if !ignore_cklnqd {
                        let en_iv = self.pin2aigpin_iv[en_pin];
                        assert_ne!(en_iv, usize::MAX, "clken not built");
                        result = self.add_and_gate(result, en_iv);
                    }
                }
                return Ok(result);
            }

            let mut pin_a = usize::MAX;
            let mut pin_cp = usize::MAX;
            let mut pin_en = usize::MAX;
            let celltype = netlistdb.celltypes[cellid].as_str();

            // Determine if this is a clock buffer/inverter (AIGPDK or SKY130)
            let is_inv = celltype == "INV"
                || (is_sky130_cell(celltype) && {
                    let ct = extract_cell_type(celltype);
                    ct.starts_with("inv") || ct.starts_with("clkinv")
                });
            let is_buf = celltype == "BUF"
                || (is_sky130_cell(celltype) && {
                    let ct = extract_cell_type(celltype);
                    ct.starts_with("buf")
                        || ct.starts_with("clkbuf")
                        || ct.starts_with("clkdlybuf")
                        || ct.starts_with("lpflow_isobufsrc")
                        || ct.starts_with("lpflow_inputiso")
                });

            if !is_inv && !is_buf && celltype != "CKLNQD" {
                clilog::error!(
                    "cell type {} not supported on clock path. expecting only INV, BUF, or CKLNQD (or SKY130 equivalents)",
                    celltype
                );
                return Err(pinid);
            }

            for ipin in netlistdb.cell2pin.iter_set(cellid) {
                if netlistdb.pindirect[ipin] == Direction::I {
                    match netlistdb.pinnames[ipin].1.as_str() {
                        "A" => pin_a = ipin,
                        "CP" => pin_cp = ipin,
                        "E" => pin_en = ipin,
                        i @ _ => {
                            clilog::error!(
                                "input pin {} unexpected for ck element {}",
                                i,
                                celltype
                            );
                            return Err(ipin);
                        }
                    }
                }
            }

            if is_inv {
                assert_ne!(pin_a, usize::MAX);
                current_pinid = pin_a;
                current_is_negedge = !is_negedge;
            } else if is_buf {
                assert_ne!(pin_a, usize::MAX);
                current_pinid = pin_a;
                // is_negedge stays the same
            } else if celltype == "CKLNQD" {
                assert_ne!(pin_cp, usize::MAX);
                assert_ne!(pin_en, usize::MAX);
                // Push CKLNQD for post-processing, continue with CP
                cklnqd_stack.push((pin_en, is_negedge));
                current_pinid = pin_cp;
                // is_negedge stays the same
            } else {
                unreachable!()
            }
        }
    }

    /// Iteratively add AIG pins for netlistdb pins using explicit stack.
    ///
    /// for sequential logics like DFF and RAM,
    /// 1. their netlist pin inputs are not patched,
    /// 2. their aig pin inputs (in dffs and srams arrays) will be
    ///    patched to include mux -- but not inside this function.
    /// 3. their netlist/aig outputs are directly built here,
    ///    with possible patches for asynchronous DFFSR polyfill.
    fn dfs_netlistdb_build_aig(
        &mut self,
        netlistdb: &NetlistDB,
        topo_vis: &mut Vec<bool>,
        topo_instack: &mut Vec<bool>,
        start_pinid: usize,
        pdk_models: Option<&PdkModels>,
    ) {
        let mut work_stack: Vec<WorkItem> = vec![WorkItem::Visit(start_pinid)];

        while let Some(item) = work_stack.pop() {
            match item {
                WorkItem::Visit(pinid) => {
                    if topo_vis[pinid] {
                        continue;
                    }
                    if topo_instack[pinid] {
                        panic!(
                            "circuit has a loop around pin {}",
                            netlistdb.pinnames[pinid].dbg_fmt_pin()
                        );
                    }

                    topo_vis[pinid] = true;
                    topo_instack[pinid] = true;

                    let netid = netlistdb.pin2net[pinid];
                    let cellid = netlistdb.pin2cell[pinid];
                    let celltype = netlistdb.celltypes[cellid].as_str();

                    // Handle input pins
                    if netlistdb.pindirect[pinid] == Direction::I {
                        if Some(netid) == netlistdb.net_zero {
                            self.pin2aigpin_iv[pinid] = 0;
                            topo_instack[pinid] = false;
                        } else if Some(netid) == netlistdb.net_one {
                            self.pin2aigpin_iv[pinid] = 1;
                            topo_instack[pinid] = false;
                        } else {
                            // Find the actual driver pin on the net
                            let net_pins_start = netlistdb.net2pin.start[netid];
                            let net_pins_end = if netid + 1 < netlistdb.net2pin.start.len() {
                                netlistdb.net2pin.start[netid + 1]
                            } else {
                                netlistdb.net2pin.items.len()
                            };
                            let mut driver_pin = None;
                            for &np in &netlistdb.net2pin.items[net_pins_start..net_pins_end] {
                                // Check if this pin is an output (driver)
                                if netlistdb.pindirect[np] == Direction::O {
                                    driver_pin = Some(np);
                                    break;
                                }
                                // Also check for primary input (cell 0)
                                if netlistdb.pin2cell[np] == 0 {
                                    driver_pin = Some(np);
                                    break;
                                }
                            }
                            if let Some(root) = driver_pin {
                                // Push process, then dependency
                                work_stack.push(WorkItem::Process(pinid));
                                work_stack.push(WorkItem::Visit(root));
                            } else {
                                // Net has no driver - tie to 0
                                // This handles floating nets / undriven signals in post-P&R netlists
                                self.pin2aigpin_iv[pinid] = 0;
                                topo_instack[pinid] = false;
                            }
                        }
                        continue;
                    }

                    // Handle primary input ports (cellid == 0, output direction)
                    if cellid == 0 {
                        let aigpin = self.add_aigpin(DriverType::InputPort(pinid));
                        self.pin2aigpin_iv[pinid] = aigpin << 1;
                        topo_instack[pinid] = false;
                        continue;
                    }

                    // Handle AIGPDK DFF/DFFSR
                    if matches!(celltype, "DFF" | "DFFSR") {
                        // Pre-create DFF Q output
                        let q = self.add_aigpin(DriverType::DFF(cellid));
                        // Record cell origin for DFF Q output (for SDF CLK→Q delay)
                        self.aigpin_cell_origins[q].push((
                            cellid,
                            celltype.to_string(),
                            "Q".to_string(),
                        ));
                        let dff = self.dffs.entry(cellid).or_default();
                        dff.q = q;

                        // Push process then S/R dependencies
                        work_stack.push(WorkItem::Process(pinid));
                        for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                            if matches!(netlistdb.pinnames[dep_pinid].1.as_str(), "S" | "R") {
                                work_stack.push(WorkItem::Visit(dep_pinid));
                            }
                        }
                        continue;
                    }

                    // Handle LATCH
                    if celltype == "LATCH" {
                        panic!(
                            "latches are intentionally UNSUPPORTED by GEM, \
                                except in identified gated clocks. \n\
                                you can link a FF&MUX-based LATCH module, \
                                but most likely that is NOT the right solution. \n\
                                check all your assignments inside always@(*) block \
                                to make sure they cover all scenarios."
                        );
                    }

                    // Handle SRAM
                    if celltype == "$__RAMGEM_SYNC_" {
                        let o = self.add_aigpin(DriverType::SRAM(cellid));
                        self.pin2aigpin_iv[pinid] = o << 1;
                        assert_eq!(netlistdb.pinnames[pinid].1.as_str(), "PORT_R_RD_DATA");
                        // Record cell origin for SRAM data output
                        self.aigpin_cell_origins[o].push((
                            cellid,
                            celltype.to_string(),
                            "PORT_R_RD_DATA".to_string(),
                        ));
                        let sram = self.srams.entry(cellid).or_default();
                        sram.port_r_rd_data[netlistdb.pinnames[pinid].2.unwrap() as usize] = o;
                        topo_instack[pinid] = false;
                        continue;
                    }

                    // Handle CKLNQD
                    if celltype == "CKLNQD" {
                        let mut prev_cp = usize::MAX;
                        let mut prev_en = usize::MAX;
                        for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                            match netlistdb.pinnames[dep_pinid].1.as_str() {
                                "CP" => prev_cp = dep_pinid,
                                "E" => prev_en = dep_pinid,
                                _ => {}
                            }
                        }
                        assert_ne!(prev_cp, usize::MAX);
                        assert_ne!(prev_en, usize::MAX);
                        // Push process then dependencies
                        work_stack.push(WorkItem::Process(pinid));
                        work_stack.push(WorkItem::Visit(prev_cp));
                        work_stack.push(WorkItem::Visit(prev_en));
                        continue;
                    }

                    // Handle GEM_ASSERT / GEM_DISPLAY
                    if celltype == "GEM_ASSERT" || celltype == "GEM_DISPLAY" {
                        // Push process then all input pin dependencies
                        work_stack.push(WorkItem::Process(pinid));
                        for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                            if netlistdb.pindirect[dep_pinid] == Direction::I {
                                work_stack.push(WorkItem::Visit(dep_pinid));
                            }
                        }
                        continue;
                    }

                    // Handle SKY130 cells
                    if is_sky130_cell(celltype) {
                        // Get dependencies based on cell type
                        let deps = self.get_sky130_dependencies(netlistdb, pinid, cellid, celltype);
                        // Do any pre-processing
                        self.sky130_preprocess(netlistdb, pinid, cellid, celltype);
                        // Push process then dependencies
                        work_stack.push(WorkItem::Process(pinid));
                        for dep in deps {
                            work_stack.push(WorkItem::Visit(dep));
                        }
                        continue;
                    }

                    // Handle AIGPDK combinational cells (AND2, INV, BUF)
                    let mut prev_a = usize::MAX;
                    let mut prev_b = usize::MAX;
                    for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                        match netlistdb.pinnames[dep_pinid].1.as_str() {
                            "A" => prev_a = dep_pinid,
                            "B" => prev_b = dep_pinid,
                            _ => {}
                        }
                    }
                    // Push process then dependencies
                    work_stack.push(WorkItem::Process(pinid));
                    if prev_b != usize::MAX {
                        work_stack.push(WorkItem::Visit(prev_b));
                    }
                    if prev_a != usize::MAX {
                        work_stack.push(WorkItem::Visit(prev_a));
                    }
                }

                WorkItem::Process(pinid) => {
                    let netid = netlistdb.pin2net[pinid];
                    let cellid = netlistdb.pin2cell[pinid];
                    let celltype = netlistdb.celltypes[cellid].as_str();

                    // Process input pins (copy from driver)
                    if netlistdb.pindirect[pinid] == Direction::I {
                        let root = netlistdb.net2pin.items[netlistdb.net2pin.start[netid]];
                        self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[root];
                        if cellid == 0 {
                            self.primary_outputs.insert(self.pin2aigpin_iv[pinid]);
                        }
                        topo_instack[pinid] = false;
                        continue;
                    }

                    // Process AIGPDK DFF/DFFSR
                    if matches!(celltype, "DFF" | "DFFSR") {
                        let dff = self.dffs.get(&cellid).unwrap();
                        let q = dff.q;
                        let mut ap_s_iv = 1;
                        let mut ap_r_iv = 1;
                        let mut q_out = q << 1;

                        for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                            let prev = self.pin2aigpin_iv[dep_pinid];
                            match netlistdb.pinnames[dep_pinid].1.as_str() {
                                "S" => ap_s_iv = prev,
                                "R" => ap_r_iv = prev,
                                _ => {}
                            }
                        }
                        // AIGPDK DFFSR .lib says: clear="(!R)", preset="(!S)"
                        // i.e. R=0 → Q=0 (clear), S=0 → Q=1 (preset)
                        // Same active-low semantics as SKY130 RESET_B/SET_B.
                        //   Q = AND(OR(Q_state, !S), R)
                        q_out = self.add_and_gate(q_out ^ 1, ap_s_iv) ^ 1;
                        q_out = self.add_and_gate(q_out, ap_r_iv);
                        self.pin2aigpin_iv[pinid] = q_out;
                        topo_instack[pinid] = false;
                        continue;
                    }

                    // Process CKLNQD (no output pin defined)
                    if celltype == "CKLNQD" {
                        // do not define pin2aigpin_iv[pinid] which is CKLNQD/Q and unused in logic.
                        topo_instack[pinid] = false;
                        continue;
                    }

                    // Process GEM_ASSERT / GEM_DISPLAY (no output pin defined)
                    if celltype == "GEM_ASSERT" || celltype == "GEM_DISPLAY" {
                        topo_instack[pinid] = false;
                        continue;
                    }

                    // Process SKY130 cells
                    if is_sky130_cell(celltype) {
                        self.sky130_postprocess(netlistdb, pinid, cellid, celltype, pdk_models);
                        topo_instack[pinid] = false;
                        continue;
                    }

                    // Process AIGPDK combinational cells
                    let mut prev_a = usize::MAX;
                    let mut prev_b = usize::MAX;
                    for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                        match netlistdb.pinnames[dep_pinid].1.as_str() {
                            "A" => prev_a = dep_pinid,
                            "B" => prev_b = dep_pinid,
                            _ => {}
                        }
                    }

                    match celltype {
                        "AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" | "AND2_11_1" => {
                            assert_ne!(prev_a, usize::MAX);
                            assert_ne!(prev_b, usize::MAX);
                            let name = netlistdb.celltypes[cellid].as_bytes();
                            let iv_a = name[5] - b'0';
                            let iv_b = name[6] - b'0';
                            let iv_y = name[8] - b'0';
                            let apid = self.add_and_gate(
                                self.pin2aigpin_iv[prev_a] ^ (iv_a as usize),
                                self.pin2aigpin_iv[prev_b] ^ (iv_b as usize),
                            ) ^ (iv_y as usize);
                            self.pin2aigpin_iv[pinid] = apid;
                        }
                        "INV" => {
                            assert_ne!(prev_a, usize::MAX);
                            self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a] ^ 1;
                        }
                        "BUF" => {
                            assert_ne!(prev_a, usize::MAX);
                            self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a];
                        }
                        _ => panic!("Unknown AIGPDK cell type: {}", celltype),
                    }
                    topo_instack[pinid] = false;
                }
            }
        }
    }

    /// Get dependencies for a SKY130 cell that need to be visited before processing.
    /// Uses SmallVec to avoid heap allocation for most cells (≤6 inputs).
    fn get_sky130_dependencies(
        &self,
        netlistdb: &NetlistDB,
        pinid: usize,
        cellid: usize,
        celltype: &str,
    ) -> SmallVec<[usize; 6]> {
        let cell_type = extract_cell_type(celltype);
        let output_pin_name = netlistdb.pinnames[pinid].1.as_str();

        // Tie cells have no dependencies
        if is_tie_cell(cell_type) {
            return SmallVec::new();
        }

        // SRAM has no dependencies for output creation
        if celltype.starts_with("CF_SRAM_") {
            return SmallVec::new();
        }

        // Sequential cells: depend on SET_B/RESET_B pins
        if is_sequential_cell(cell_type) {
            let mut deps = SmallVec::new();
            for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pin_name = netlistdb.pinnames[dep_pinid].1.as_str();
                if matches!(pin_name, "SET_B" | "RESET_B") {
                    deps.push(dep_pinid);
                }
            }
            return deps;
        }

        // Multi-output cells (ha, fa, dfbbp)
        if is_multi_output_cell(cell_type) {
            // dfbbp Q_N depends on Q being built first
            if cell_type == "dfbbp" && output_pin_name == "Q_N" {
                // Find Q pin
                for opin in netlistdb.cell2pin.iter_set(cellid) {
                    if netlistdb.pinnames[opin].1.as_str() == "Q" {
                        if self.pin2aigpin_iv[opin] == usize::MAX {
                            let mut deps = SmallVec::new();
                            deps.push(opin);
                            return deps;
                        }
                        return SmallVec::new();
                    }
                }
            }
            // For other multi-output cells, depend on all input pins
            let mut deps = SmallVec::new();
            for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                if netlistdb.pindirect[dep_pinid] == Direction::I {
                    deps.push(dep_pinid);
                }
            }
            return deps;
        }

        // Combinational cells: depend on all input pins
        let mut deps = SmallVec::new();
        for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
            if netlistdb.pindirect[dep_pinid] == Direction::I {
                deps.push(dep_pinid);
            }
        }
        deps
    }

    /// Pre-process for SKY130 cells (create output AIG pins before dependencies are visited).
    fn sky130_preprocess(
        &mut self,
        netlistdb: &NetlistDB,
        pinid: usize,
        cellid: usize,
        celltype: &str,
    ) {
        let cell_type = extract_cell_type(celltype);
        let output_pin_name = netlistdb.pinnames[pinid].1.as_str();

        // Tie cells - set output immediately
        if is_tie_cell(cell_type) {
            match output_pin_name {
                "HI" => self.pin2aigpin_iv[pinid] = 1,
                "LO" => self.pin2aigpin_iv[pinid] = 0,
                _ => panic!("Unknown tie cell output: {}", output_pin_name),
            }
            return;
        }

        // SRAM output creation
        if celltype.starts_with("CF_SRAM_") {
            let o = self.add_aigpin(DriverType::SRAM(cellid));
            self.pin2aigpin_iv[pinid] = o << 1;
            assert_eq!(output_pin_name, "DO", "Expected DO output pin for CF_SRAM");
            // Record cell origin for SRAM data output
            self.aigpin_cell_origins[o].push((
                cellid,
                celltype.to_string(),
                output_pin_name.to_string(),
            ));
            let sram = self.srams.entry(cellid).or_default();
            let bit_idx = netlistdb.pinnames[pinid]
                .2
                .expect("DO pin must have bit index") as usize;
            sram.port_r_rd_data[bit_idx] = o;
            return;
        }

        // Sequential cells: pre-create DFF Q output
        if is_sequential_cell(cell_type) {
            let q = self.add_aigpin(DriverType::DFF(cellid));
            // Record cell origin for DFF Q output (for SDF CLK→Q delay)
            self.aigpin_cell_origins[q].push((cellid, cell_type.to_string(), "Q".to_string()));
            let dff = self.dffs.entry(cellid).or_default();
            dff.q = q;
            return;
        }

        // dfbbp with Q_N output: no pre-processing needed
        // Other multi-output and combinational cells: no pre-processing needed
    }

    /// Post-process for SKY130 cells (compute output after dependencies are visited).
    fn sky130_postprocess(
        &mut self,
        netlistdb: &NetlistDB,
        pinid: usize,
        cellid: usize,
        celltype: &str,
        pdk_models: Option<&PdkModels>,
    ) {
        let cell_type = extract_cell_type(celltype);

        // Tie cells and SRAM were already handled in preprocess
        if is_tie_cell(cell_type) || celltype.starts_with("CF_SRAM_") {
            return;
        }

        // Sequential cells: apply reset/set logic to Q
        if is_sequential_cell(cell_type) {
            let dff = self.dffs.get(&cellid).unwrap();
            let q = dff.q;
            let mut ap_s_iv = 1;
            let mut ap_r_iv = 1;
            let mut q_out = q << 1;

            for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pin_name = netlistdb.pinnames[dep_pinid].1.as_str();
                let prev = self.pin2aigpin_iv[dep_pinid];
                match pin_name {
                    // SKY130 SET_B and RESET_B are active-low, same convention
                    // as AIGPDK's S and R pins. No inversion needed - the AIG
                    // formula OR(Q, NOT pin) / AND(result, pin) already handles
                    // active-low correctly:
                    //   RESET_B=0 (reset): AND(Q, 0) = 0  ✓
                    //   RESET_B=1 (normal): AND(Q, 1) = Q  ✓
                    "SET_B" => ap_s_iv = prev,
                    "RESET_B" => ap_r_iv = prev,
                    _ => {}
                }
            }
            q_out = self.add_and_gate(q_out ^ 1, ap_s_iv) ^ 1;
            q_out = self.add_and_gate(q_out, ap_r_iv);
            self.pin2aigpin_iv[pinid] = q_out;
            return;
        }

        // Multi-output cells (ha, fa, dfbbp)
        if is_multi_output_cell(cell_type) {
            self.build_sky130_multi_output_postprocess(
                netlistdb, pinid, cellid, cell_type, pdk_models,
            );
            return;
        }

        // Combinational cells: decompose and build
        // Use CellInputs struct instead of HashMap to avoid allocation
        let mut inputs = CellInputs::new();
        let mut input_count = 0;
        for ipin in netlistdb.cell2pin.iter_set(cellid) {
            // Include both explicit inputs (Direction::I) and unknown direction pins (Direction::U)
            // Unknown direction pins are treated as inputs for decomposition
            // Only skip explicit outputs (Direction::O)
            if netlistdb.pindirect[ipin] != Direction::O {
                let pin_name = netlistdb.pinnames[ipin].1.as_str();
                // Skip output pins by name (Y, X, Q, etc.)
                if !matches!(pin_name, "Y" | "X" | "Q" | "Q_N" | "SUM" | "COUT") {
                    inputs.set_pin(pin_name, self.pin2aigpin_iv[ipin]);
                    input_count += 1;
                }
            }
        }

        // Debug: check if we got enough inputs
        if cell_type == "nand2" && inputs.a == usize::MAX {
            use netlistdb::GeneralHierName;
            let cell_name = netlistdb.cellnames[cellid].dbg_fmt_hier();
            let mut pins_info: Vec<String> = Vec::new();
            for ipin in netlistdb.cell2pin.iter_set(cellid) {
                let pin_name = netlistdb.pinnames[ipin].1.as_str();
                let dir = format!("{:?}", netlistdb.pindirect[ipin]);
                pins_info.push(format!("{}:{}:{}", pin_name, dir, ipin));
            }
            clilog::warn!(
                "nand2 cell {} missing A input. Pins: {:?}",
                cell_name,
                pins_info
            );
        }
        if input_count == 0 {
            use netlistdb::GeneralHierName;
            let cell_name = netlistdb.cellnames[cellid].dbg_fmt_hier();
            let mut pins_info: Vec<String> = Vec::new();
            for ipin in netlistdb.cell2pin.iter_set(cellid) {
                let pin_name = netlistdb.pinnames[ipin].1.as_str();
                let dir = format!("{:?}", netlistdb.pindirect[ipin]);
                pins_info.push(format!("{}:{}", pin_name, dir));
            }
            panic!(
                "No input pins found for cell {} ({}), type='{}'. Pins: {:?}",
                cellid, cell_name, cell_type, pins_info
            );
        }

        let output_pin_name = netlistdb.pinnames[pinid].1.as_str();
        let pdk = pdk_models.expect(
            "PDK models required for SKY130 cell decomposition. \
             Ensure sky130_fd_sc_hd submodule is initialized.",
        );
        let decomp = decompose_with_pdk(cell_type, &inputs, output_pin_name, pdk);

        // Use SmallVec for gate outputs - most cells have ≤5 gates
        let mut gate_outputs: SmallVec<[usize; 5]> = SmallVec::new();
        for (i, (a_ref, b_ref)) in decomp.and_gates.iter().enumerate() {
            let a_iv = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.resolve_decomp_ref(*a_ref, &gate_outputs)
            })) {
                Ok(v) => v,
                Err(_) => {
                    use netlistdb::GeneralHierName;
                    let cell_name = netlistdb.cellnames[cellid].dbg_fmt_hier();
                    panic!(
                        "Failed resolve_decomp_ref for cell_type='{}', cell={}, gate {}, a_ref={}, gates_so_far={}, inputs={:?}",
                        cell_type, cell_name, i, a_ref, gate_outputs.len(), inputs
                    )
                }
            };
            let b_iv = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.resolve_decomp_ref(*b_ref, &gate_outputs)
            })) {
                Ok(v) => v,
                Err(_) => {
                    use netlistdb::GeneralHierName;
                    let cell_name = netlistdb.cellnames[cellid].dbg_fmt_hier();
                    panic!(
                        "Failed resolve_decomp_ref for cell_type='{}', cell={}, gate {}, b_ref={}, gates_so_far={}, inputs={:?}",
                        cell_type, cell_name, i, b_ref, gate_outputs.len(), inputs
                    )
                }
            };
            let gate_out = self.add_and_gate(a_iv, b_iv);
            gate_outputs.push(gate_out >> 1);
        }

        let output_iv = if decomp.output_idx < 0 {
            // Negative index = reference to a gate we just created
            let gate_idx = (-decomp.output_idx - 1) as usize;
            gate_outputs[gate_idx] << 1
        } else {
            // Positive index = passthrough to an existing AIG pin
            // Need to encode as (index << 1) since output_idx is a raw index
            (decomp.output_idx as usize) << 1
        };

        let final_iv = if decomp.output_inverted {
            output_iv ^ 1
        } else {
            output_iv
        };

        self.pin2aigpin_iv[pinid] = final_iv;

        // Record cell origin for SDF back-annotation.
        // The full IOPATH delay goes on the output AIG pin only;
        // internal AND gates from multi-gate decompositions keep None (zero delay).
        let output_aigpin = final_iv >> 1;
        if output_aigpin > 0 && output_aigpin < self.aigpin_cell_origins.len() {
            let output_pin_name = netlistdb.pinnames[pinid].1.as_str();
            self.aigpin_cell_origins[output_aigpin].push((
                cellid,
                cell_type.to_string(),
                output_pin_name.to_string(),
            ));
        }
    }

    /// Post-process for SKY130 multi-output cells (ha, fa, dfbbp).
    fn build_sky130_multi_output_postprocess(
        &mut self,
        netlistdb: &NetlistDB,
        output_pinid: usize,
        cellid: usize,
        cell_type: &str,
        pdk_models: Option<&PdkModels>,
    ) {
        let output_pin_name = netlistdb.pinnames[output_pinid].1.as_str();

        // dfbbp is sequential - handle inline (not in PDK combinational models)
        if cell_type == "dfbbp" {
            if output_pin_name == "Q_N" {
                // Q_N is inverted Q
                for opin in netlistdb.cell2pin.iter_set(cellid) {
                    if netlistdb.pinnames[opin].1.as_str() == "Q" {
                        self.pin2aigpin_iv[output_pinid] = self.pin2aigpin_iv[opin] ^ 1;
                        return;
                    }
                }
                panic!("dfbbp missing Q pin");
            } else {
                // Q output - build DFF logic
                let dff = self.dffs.get(&cellid).unwrap();
                let q = dff.q;
                let mut ap_s_iv = 1;
                let mut ap_r_iv = 1;
                let mut q_out = q << 1;

                for dep_pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_name = netlistdb.pinnames[dep_pinid].1.as_str();
                    let prev = self.pin2aigpin_iv[dep_pinid];
                    match pin_name {
                        "SET_B" => ap_s_iv = prev,
                        "RESET_B" => ap_r_iv = prev,
                        _ => {}
                    }
                }
                q_out = self.add_and_gate(q_out ^ 1, ap_s_iv) ^ 1;
                q_out = self.add_and_gate(q_out, ap_r_iv);
                self.pin2aigpin_iv[output_pinid] = q_out;
            }
            return;
        }

        // Use PDK models for ha/fa
        let pdk = pdk_models.expect(
            "PDK models required for SKY130 multi-output cell decomposition. \
             Ensure sky130_fd_sc_hd submodule is initialized.",
        );
        let mut inputs = CellInputs::new();
        for ipin in netlistdb.cell2pin.iter_set(cellid) {
            if netlistdb.pindirect[ipin] == Direction::I {
                let pin_name = netlistdb.pinnames[ipin].1.as_str();
                inputs.set_pin(pin_name, self.pin2aigpin_iv[ipin]);
            }
        }
        let decomp = decompose_with_pdk(cell_type, &inputs, output_pin_name, pdk);
        // Build the AIG gates from the decomposition
        let mut gate_outputs: SmallVec<[usize; 5]> = SmallVec::new();
        for (a_ref, b_ref) in &decomp.and_gates {
            let a_iv = self.resolve_decomp_ref(*a_ref, &gate_outputs);
            let b_iv = self.resolve_decomp_ref(*b_ref, &gate_outputs);
            let gate_out = self.add_and_gate(a_iv, b_iv);
            gate_outputs.push(gate_out >> 1);
        }
        let output_iv = if decomp.output_idx < 0 {
            let gate_idx = (-decomp.output_idx - 1) as usize;
            gate_outputs[gate_idx] << 1
        } else {
            (decomp.output_idx as usize) << 1
        };
        let final_iv = if decomp.output_inverted {
            output_iv ^ 1
        } else {
            output_iv
        };
        self.pin2aigpin_iv[output_pinid] = final_iv;

        // Record cell origin for SDF back-annotation
        let output_aigpin = final_iv >> 1;
        if output_aigpin > 0 && output_aigpin < self.aigpin_cell_origins.len() {
            self.aigpin_cell_origins[output_aigpin].push((
                cellid,
                cell_type.to_string(),
                output_pin_name.to_string(),
            ));
        }
    }

    /// Resolve a decomposition reference to an aigpin_iv value.
    fn resolve_decomp_ref(&self, ref_val: i64, gate_outputs: &[usize]) -> usize {
        if ref_val < 0 {
            // Reference to intermediate gate output
            // The inversion bit might be encoded in ref_val
            let base_ref = ref_val | 1; // Clear potential inversion bit for index calc
            let gate_idx = (-(base_ref >> 1) - 1) as usize;
            if gate_idx >= gate_outputs.len() {
                panic!(
                    "resolve_decomp_ref: ref_val={}, base_ref={}, gate_idx={}, gate_outputs.len()={}",
                    ref_val, base_ref, gate_idx, gate_outputs.len()
                );
            }
            let invert = (ref_val & 1) != (base_ref & 1);
            let gate_out = gate_outputs[gate_idx] << 1;
            if invert {
                gate_out ^ 1
            } else {
                gate_out
            }
        } else {
            // Direct input reference (already has inversion bit)
            ref_val as usize
        }
    }

    /// Build an AIG from a netlistdb, using explicit PDK behavioral models for decomposition.
    pub fn from_netlistdb_with_pdk(netlistdb: &NetlistDB, pdk_models: &PdkModels) -> AIG {
        Self::from_netlistdb_impl(netlistdb, Some(pdk_models))
    }

    /// Build an AIG from a netlistdb.
    ///
    /// For designs with SKY130 cells, automatically loads PDK models from the
    /// `vendor/sky130_fd_sc_hd/cells` submodule. Panics with a helpful message if the
    /// submodule is not initialized.
    pub fn from_netlistdb(netlistdb: &NetlistDB) -> AIG {
        // Check if the design uses SKY130 cells
        let has_sky130 =
            (1..netlistdb.num_cells).any(|cid| is_sky130_cell(netlistdb.celltypes[cid].as_str()));

        if has_sky130 {
            let pdk_path = std::path::PathBuf::from("vendor/sky130_fd_sc_hd/cells");
            assert!(
                pdk_path.exists(),
                "Design uses SKY130 cells but sky130_fd_sc_hd submodule not found. \
                 Run: git submodule update --init"
            );
            // Collect cell types
            let mut cell_types: Vec<String> = Vec::new();
            for cellid in 1..netlistdb.num_cells {
                let celltype = netlistdb.celltypes[cellid].as_str();
                if is_sky130_cell(celltype) {
                    let ct = extract_cell_type(celltype).to_string();
                    if !cell_types.contains(&ct) {
                        cell_types.push(ct);
                    }
                }
            }
            cell_types.sort();
            let pdk_models = crate::sky130_pdk::load_pdk_models(&pdk_path, &cell_types);
            Self::from_netlistdb_impl(netlistdb, Some(&pdk_models))
        } else {
            Self::from_netlistdb_impl(netlistdb, None)
        }
    }

    fn from_netlistdb_impl(netlistdb: &NetlistDB, pdk_models: Option<&PdkModels>) -> AIG {
        let mut aig = AIG {
            num_aigpins: 0,
            pin2aigpin_iv: vec![usize::MAX; netlistdb.num_pins],
            drivers: vec![DriverType::Tie0],
            aigpin_cell_origins: vec![Vec::new()], // Tie0 has no cell origin
            ..Default::default()
        };

        clilog::info!(
            "Starting clock tracing for {} cells...",
            netlistdb.num_cells
        );
        let clock_start = std::time::Instant::now();
        let mut seq_count = 0;
        let mut trace_count = 0;

        for cellid in 1..netlistdb.num_cells {
            let celltype = netlistdb.celltypes[cellid].as_str();

            // Check if this is a sequential element (AIGPDK or SKY130)
            let is_aigpdk_seq = matches!(celltype, "DFF" | "DFFSR" | "$__RAMGEM_SYNC_");
            let is_sky130_seq =
                is_sky130_cell(celltype) && is_sequential_cell(extract_cell_type(celltype));

            if !is_aigpdk_seq && !is_sky130_seq {
                continue;
            }

            seq_count += 1;
            if seq_count <= 5 {
                use netlistdb::GeneralHierName;
                clilog::info!(
                    "  Processing seq cell {} ({}): {}",
                    seq_count,
                    celltype,
                    netlistdb.cellnames[cellid].dbg_fmt_hier()
                );
            } else if seq_count % 500 == 0 {
                clilog::info!(
                    "  Processed {} sequential cells ({} clock traces) in {:?}...",
                    seq_count,
                    trace_count,
                    clock_start.elapsed()
                );
            }
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pin_name = netlistdb.pinnames[pinid].1.as_str();
                // Both AIGPDK and SKY130 use "CLK" for clock pins
                if !matches!(pin_name, "CLK" | "PORT_R_CLK" | "PORT_W_CLK") {
                    continue;
                }
                trace_count += 1;
                if seq_count <= 5 {
                    clilog::info!("    Tracing clock pin {} for cell {}...", pinid, cellid);
                }
                if let Err(pinid) = aig.trace_clock_pin(netlistdb, pinid, false, true) {
                    use netlistdb::GeneralHierName;
                    panic!(
                        "Tracing clock pin of cell {} error: \
                            there is a multi-input cell driving {} \
                            that clocks this sequential element. \
                            Clock gating need to be manually patched atm.",
                        netlistdb.cellnames[cellid].dbg_fmt_hier(),
                        netlistdb.pinnames[pinid].dbg_fmt_pin()
                    );
                }
            }
        }
        clilog::info!(
            "Clock tracing done in {:?}, {} sequential cells",
            clock_start.elapsed(),
            seq_count
        );

        for (&clk, &(flagr, flagf)) in &aig.clock_pin2aigpins {
            clilog::info!(
                "inferred clock port {} ({})",
                netlistdb.pinnames[clk].dbg_fmt_pin(),
                match (flagr, flagf) {
                    (_, usize::MAX) => "posedge",
                    (usize::MAX, _) => "negedge",
                    _ => "posedge & negedge",
                }
            );
        }

        clilog::info!("Starting AIG DFS build for {} pins...", netlistdb.num_pins);
        let dfs_start = std::time::Instant::now();

        let mut topo_vis = vec![false; netlistdb.num_pins];
        let mut topo_instack = vec![false; netlistdb.num_pins];

        for pinid in 0..netlistdb.num_pins {
            aig.dfs_netlistdb_build_aig(
                netlistdb,
                &mut topo_vis,
                &mut topo_instack,
                pinid,
                pdk_models,
            );
        }

        clilog::info!(
            "AIG DFS build done in {:?}, {} AIG pins created",
            dfs_start.elapsed(),
            aig.num_aigpins
        );

        for cellid in 0..netlistdb.num_cells {
            let celltype = netlistdb.celltypes[cellid].as_str();

            if matches!(celltype, "DFF" | "DFFSR") {
                let mut ap_s_iv = 1;
                let mut ap_r_iv = 1;
                let mut ap_d_iv = 0;
                let mut ap_clken_iv = 0;
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "D" => ap_d_iv = pin_iv,
                        "S" => ap_s_iv = pin_iv,
                        "R" => ap_r_iv = pin_iv,
                        "CLK" => {
                            ap_clken_iv =
                                aig.trace_clock_pin(netlistdb, pinid, false, false).unwrap()
                        }
                        _ => {}
                    }
                }
                let mut d_in = ap_d_iv;

                // AIGPDK DFFSR .lib says: clear="(!R)", preset="(!S)"
                // i.e. R=0 → Q=0 (clear), S=0 → Q=1 (preset)
                // Same active-low semantics as SKY130 RESET_B/SET_B:
                //   d_in = AND(OR(D, !S), R)
                //   en_iv = OR(posedge, !R, !S)  (latch whenever R or S is active)
                d_in = aig.add_and_gate(d_in ^ 1, ap_s_iv) ^ 1;
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_s_iv) ^ 1;
                d_in = aig.add_and_gate(d_in, ap_r_iv);
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_r_iv) ^ 1;
                let dff = aig.dffs.entry(cellid).or_default();
                dff.en_iv = ap_clken_iv;
                dff.d_iv = d_in;
                assert_ne!(dff.q, 0);
            } else if is_sky130_cell(celltype) && is_sequential_cell(extract_cell_type(celltype)) {
                // Handle SKY130 DFFs (dfxtp, edfxtp, dfrtp, etc.)
                let cell_type = extract_cell_type(celltype);
                let mut ap_d_iv = 0;
                let mut ap_clken_iv = 0;
                let mut ap_enable_iv = 1; // For edfxtp (data enable)
                let mut ap_s_iv = 1; // Default: no set (SET_B=1, inactive)
                let mut ap_r_iv = 1; // Default: no reset (RESET_B=1, inactive)

                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "D" => ap_d_iv = pin_iv,
                        "DE" => ap_enable_iv = pin_iv, // Data enable for edfxtp
                        "SET_B" => ap_s_iv = pin_iv, // Active-low: AND/OR formulas handle it directly
                        "RESET_B" => ap_r_iv = pin_iv, // Active-low: AND(d, RESET_B) → RESET_B=0 forces d=0
                        "CLK" => {
                            ap_clken_iv =
                                aig.trace_clock_pin(netlistdb, pinid, false, false).unwrap()
                        }
                        _ => {}
                    }
                }

                // For edfxtp, the clock enable is gated by DE (data enable)
                if cell_type == "edfxtp" {
                    ap_clken_iv = aig.add_and_gate(ap_clken_iv, ap_enable_iv);
                }

                let mut d_in = ap_d_iv;

                // Apply async set/reset to D input and clock enable (same as AIGPDK DFF)
                d_in = aig.add_and_gate(d_in ^ 1, ap_s_iv) ^ 1;
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_s_iv) ^ 1;
                d_in = aig.add_and_gate(d_in, ap_r_iv);
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_r_iv) ^ 1;

                let dff = aig.dffs.entry(cellid).or_default();
                dff.en_iv = ap_clken_iv;
                dff.d_iv = d_in;
                assert_ne!(dff.q, 0, "SKY130 DFF {} has no Q output built", cellid);
            } else if celltype == "$__RAMGEM_SYNC_" {
                let mut sram = aig.srams.entry(cellid).or_default().clone();
                let mut write_clken_iv = 0;
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let bit = netlistdb.pinnames[pinid].2.map(|i| i as usize);
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "PORT_R_ADDR" => {
                            sram.port_r_addr_iv[bit.unwrap()] = pin_iv;
                        }
                        "PORT_R_CLK" => {
                            sram.port_r_en_iv =
                                aig.trace_clock_pin(netlistdb, pinid, false, false).unwrap();
                        }
                        "PORT_W_ADDR" => {
                            sram.port_w_addr_iv[bit.unwrap()] = pin_iv;
                        }
                        "PORT_W_CLK" => {
                            write_clken_iv =
                                aig.trace_clock_pin(netlistdb, pinid, false, false).unwrap();
                        }
                        "PORT_W_WR_DATA" => {
                            sram.port_w_wr_data_iv[bit.unwrap()] = pin_iv;
                        }
                        "PORT_W_WR_EN" => {
                            sram.port_w_wr_en_iv[bit.unwrap()] = pin_iv;
                        }
                        _ => {}
                    }
                }
                for i in 0..32 {
                    let or_en = sram.port_w_wr_en_iv[i];
                    let or_en = aig.add_and_gate(or_en, write_clken_iv);
                    sram.port_w_wr_en_iv[i] = or_en;
                }
                *aig.srams.get_mut(&cellid).unwrap() = sram;
            } else if celltype.starts_with("CF_SRAM_") {
                // ChipFlow SRAM: single-port with shared address
                // Pins: CLKin, EN, R_WB, AD[9:0], BEN[31:0], DI[31:0], DO[31:0]
                let mut sram = aig.srams.entry(cellid).or_default().clone();
                let mut clken_iv = 0;
                let mut r_wb_iv = 0; // 0=write, 1=read

                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let bit = netlistdb.pinnames[pinid].2.map(|i| i as usize);
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "CLKin" => {
                            clken_iv = aig.trace_clock_pin(netlistdb, pinid, false, false).unwrap();
                        }
                        "EN" => {
                            // EN combined with clock for read enable
                            // (we'll combine later)
                        }
                        "R_WB" => {
                            r_wb_iv = pin_iv; // 0=write, 1=read
                        }
                        "AD" => {
                            // Shared address for read and write
                            let idx = bit.unwrap();
                            if idx < AIGPDK_SRAM_ADDR_WIDTH {
                                sram.port_r_addr_iv[idx] = pin_iv;
                                sram.port_w_addr_iv[idx] = pin_iv;
                            }
                        }
                        "DI" => {
                            sram.port_w_wr_data_iv[bit.unwrap()] = pin_iv;
                        }
                        "BEN" => {
                            // Byte enable becomes write enable when R_WB=0
                            sram.port_w_wr_en_iv[bit.unwrap()] = pin_iv;
                        }
                        _ => {} // Ignore control pins like SM, TM, Scan*, vpwr*
                    }
                }

                // Read enable: clken AND R_WB (read when R_WB=1)
                sram.port_r_en_iv = aig.add_and_gate(clken_iv, r_wb_iv);

                // Write enable: clken AND !R_WB (write when R_WB=0) AND BEN
                let write_clken_iv = aig.add_and_gate(clken_iv, r_wb_iv ^ 1);
                for i in 0..32 {
                    let ben_iv = sram.port_w_wr_en_iv[i];
                    sram.port_w_wr_en_iv[i] = aig.add_and_gate(write_clken_iv, ben_iv);
                }

                *aig.srams.get_mut(&cellid).unwrap() = sram;
            } else if netlistdb.celltypes[cellid].as_str() == "GEM_ASSERT" {
                // Parse GEM_ASSERT cells for assertion checking
                // GEM_ASSERT has: CLK (trigger), EN (enable), A (condition)
                // Assertion fails when EN is high and A is low
                let mut ap_en_iv = 1; // Default: always enabled
                let mut ap_a_iv = 1; // Default: always passing
                let mut ap_clken_iv = 1; // Default: always triggered

                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "EN" => ap_en_iv = pin_iv,
                        "A" => ap_a_iv = pin_iv,
                        "CLK" => {
                            // Try to trace clock, but if it's tied to 1, use constant
                            if let Ok(clken) = aig.trace_clock_pin(netlistdb, pinid, false, false) {
                                ap_clken_iv = clken;
                            }
                        }
                        _ => {}
                    }
                }

                // The condition for firing an assertion event:
                // fire = clk_enable && EN && !A
                // We store the "fire" condition (inverted A ANDed with EN and clock)
                let fire_condition = aig.add_and_gate(ap_en_iv, ap_a_iv ^ 1);
                let fire_condition = aig.add_and_gate(fire_condition, ap_clken_iv);

                let simcontrol = aig.simcontrols.entry(cellid).or_default();
                simcontrol.condition_iv = fire_condition;
                simcontrol.control_type = None; // Assertion, not $stop/$finish
                simcontrol.message_id = cellid as u32; // Use cell ID as message ID for now

                clilog::debug!(
                    "Found GEM_ASSERT cell {} (condition_iv={}, en_iv={}, a_iv={}, clken_iv={})",
                    cellid,
                    fire_condition,
                    ap_en_iv,
                    ap_a_iv,
                    ap_clken_iv
                );
            } else if netlistdb.celltypes[cellid].as_str() == "GEM_DISPLAY" {
                // Parse GEM_DISPLAY cells for $display/$write support
                // GEM_DISPLAY has: CLK (trigger), EN (enable), MSG_ID[31:0] (argument values)
                // Plus attributes: gem_format (format string), gem_args_width (arg width)
                let mut dp_en_iv = 1; // Default: always enabled
                let mut dp_clken_iv = 1; // Default: always triggered
                let mut dp_args_iv = Vec::new();

                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "EN" => dp_en_iv = pin_iv,
                        "CLK" => {
                            // Try to trace clock for edge detection
                            if let Ok(clken) = aig.trace_clock_pin(netlistdb, pinid, false, false) {
                                dp_clken_iv = clken;
                            }
                        }
                        "MSG_ID" => {
                            // MSG_ID is a 32-bit bus, collect all bit pins
                            dp_args_iv.push(pin_iv);
                        }
                        _ => {}
                    }
                }

                // The condition for firing a display event:
                // fire = clk_enable && EN
                let fire_condition = aig.add_and_gate(dp_en_iv, dp_clken_iv);

                // Get cell name for matching with JSON attributes later
                use netlistdb::GeneralHierName;
                let cell_name = netlistdb.cellnames[cellid].dbg_fmt_hier();

                // Create DisplayNode
                let display = aig.displays.entry(cellid).or_default();
                display.enable_iv = fire_condition;
                display.clken_iv = dp_clken_iv;
                // Format string will be populated from JSON attributes later
                display.format = format!("$display at cell {}", cellid);
                display.args_iv = dp_args_iv;
                display.arg_widths = vec![32]; // Default: 32-bit argument
                display.cell_name = cell_name.to_string();

                clilog::debug!(
                    "Found GEM_DISPLAY cell {} '{}' (enable_iv={}, clken_iv={}, args={})",
                    cellid,
                    display.cell_name,
                    fire_condition,
                    dp_clken_iv,
                    display.args_iv.len()
                );
            }
        }

        // Validate AIG: check for bad operand values
        for (i, driver) in aig.drivers.iter().enumerate() {
            if let DriverType::AndGate(a, b) = *driver {
                if (a >> 1) > aig.num_aigpins {
                    panic!("AIG validation: AND gate at pin {} has bad operand a={} (a>>1={}, num_aigpins={})", i, a, a >> 1, aig.num_aigpins);
                }
                if (b >> 1) > aig.num_aigpins {
                    panic!("AIG validation: AND gate at pin {} has bad operand b={} (b>>1={}, num_aigpins={})", i, b, b >> 1, aig.num_aigpins);
                }
            }
        }
        // Validate DFF d_iv and en_iv
        for (cellid, dff) in &aig.dffs {
            if (dff.d_iv >> 1) > aig.num_aigpins {
                use netlistdb::GeneralHierName;
                let cell_name = netlistdb.cellnames[*cellid].dbg_fmt_hier();
                panic!(
                    "AIG validation: DFF {} ({}) has bad d_iv={} (d_iv>>1={})",
                    cellid,
                    cell_name,
                    dff.d_iv,
                    dff.d_iv >> 1
                );
            }
            if (dff.en_iv >> 1) > aig.num_aigpins {
                use netlistdb::GeneralHierName;
                let cell_name = netlistdb.cellnames[*cellid].dbg_fmt_hier();
                panic!(
                    "AIG validation: DFF {} ({}) has bad en_iv={} (en_iv>>1={})",
                    cellid,
                    cell_name,
                    dff.en_iv,
                    dff.en_iv >> 1
                );
            }
        }
        // Validate primary_outputs
        for &po_iv in &aig.primary_outputs {
            if (po_iv >> 1) > aig.num_aigpins {
                // Find which pin has this value
                for pinid in 0..netlistdb.num_pins {
                    if aig.pin2aigpin_iv[pinid] == po_iv {
                        use netlistdb::GeneralHierName;
                        let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                        let cellid = netlistdb.pin2cell[pinid];
                        let cell_name = netlistdb.cellnames[cellid].dbg_fmt_hier();
                        let cell_type = &netlistdb.celltypes[cellid];
                        let dir = netlistdb.pindirect[pinid];
                        panic!("AIG validation: primary output has bad value iv={} from pin {} (cell {} type={}, dir={:?})", po_iv, pin_name, cell_name, cell_type, dir);
                    }
                }
                panic!("AIG validation: primary output has bad value iv={} (iv>>1={}) - could not find source pin", po_iv, po_iv >> 1);
            }
        }

        aig.fanouts_start = vec![0; aig.num_aigpins + 2];
        for (_i, driver) in aig.drivers.iter().enumerate() {
            if let DriverType::AndGate(a, b) = *driver {
                if (a >> 1) != 0 {
                    aig.fanouts_start[a >> 1] += 1;
                }
                if (b >> 1) != 0 {
                    aig.fanouts_start[b >> 1] += 1;
                }
            }
        }
        for i in 1..aig.num_aigpins + 2 {
            aig.fanouts_start[i] += aig.fanouts_start[i - 1];
        }
        aig.fanouts = vec![0; aig.fanouts_start[aig.num_aigpins + 1]];
        for (i, driver) in aig.drivers.iter().enumerate() {
            if let DriverType::AndGate(a, b) = *driver {
                if (a >> 1) != 0 {
                    let st = aig.fanouts_start[a >> 1] - 1;
                    aig.fanouts_start[a >> 1] = st;
                    aig.fanouts[st] = i;
                }
                if (b >> 1) != 0 {
                    let st = aig.fanouts_start[b >> 1] - 1;
                    aig.fanouts_start[b >> 1] = st;
                    aig.fanouts[st] = i;
                }
            }
        }

        aig
    }

    pub fn topo_traverse_generic(
        &self,
        endpoints: Option<&Vec<usize>>,
        is_primary_input: Option<&IndexSet<usize>>,
    ) -> Vec<usize> {
        let mut vis = IndexSet::new();
        let mut ret = Vec::new();
        fn dfs_topo(
            aig: &AIG,
            vis: &mut IndexSet<usize>,
            ret: &mut Vec<usize>,
            is_primary_input: Option<&IndexSet<usize>>,
            u: usize,
        ) {
            if vis.contains(&u) {
                return;
            }
            vis.insert(u);
            if let DriverType::AndGate(a, b) = aig.drivers[u] {
                if is_primary_input.map(|s| s.contains(&u)) != Some(true) {
                    if (a >> 1) != 0 {
                        dfs_topo(aig, vis, ret, is_primary_input, a >> 1);
                    }
                    if (b >> 1) != 0 {
                        dfs_topo(aig, vis, ret, is_primary_input, b >> 1);
                    }
                }
            }
            ret.push(u);
        }
        if let Some(endpoints) = endpoints {
            for &endpoint in endpoints {
                dfs_topo(self, &mut vis, &mut ret, is_primary_input, endpoint);
            }
        } else {
            for i in 1..self.num_aigpins + 1 {
                dfs_topo(self, &mut vis, &mut ret, is_primary_input, i);
            }
        }
        ret
    }

    pub fn num_endpoint_groups(&self) -> usize {
        self.primary_outputs.len()
            + self.dffs.len()
            + self.srams.len()
            + self.simcontrols.len()
            + self.displays.len()
    }

    pub fn get_endpoint_group(&self, endpt_id: usize) -> EndpointGroup<'_> {
        let po_len = self.primary_outputs.len();
        let dff_len = self.dffs.len();
        let sram_len = self.srams.len();
        let simctrl_len = self.simcontrols.len();

        if endpt_id < po_len {
            EndpointGroup::PrimaryOutput(*self.primary_outputs.get_index(endpt_id).unwrap())
        } else if endpt_id < po_len + dff_len {
            EndpointGroup::DFF(&self.dffs[endpt_id - po_len])
        } else if endpt_id < po_len + dff_len + sram_len {
            EndpointGroup::RAMBlock(&self.srams[endpt_id - po_len - dff_len])
        } else if endpt_id < po_len + dff_len + sram_len + simctrl_len {
            EndpointGroup::SimControl(&self.simcontrols[endpt_id - po_len - dff_len - sram_len])
        } else {
            EndpointGroup::Display(
                &self.displays[endpt_id - po_len - dff_len - sram_len - simctrl_len],
            )
        }
    }

    /// Populate display format information from JSON attributes.
    /// This should be called after from_netlistdb() with display info extracted from the JSON.
    pub fn populate_display_info(
        &mut self,
        display_info: &IndexMap<String, crate::display::DisplayCellInfo>,
    ) {
        for (_cell_id, display) in self.displays.iter_mut() {
            if let Some(info) = display_info.get(&display.cell_name) {
                display.format = info.format.clone();
                display.arg_widths = vec![info.args_width];
                clilog::debug!(
                    "Populated display info for '{}': format='{}', args_width={}",
                    display.cell_name,
                    info.format,
                    info.args_width
                );
            }
        }
    }

    // === Static Timing Analysis Methods ===

    /// Load timing information from a Liberty library.
    ///
    /// This initializes the gate_delays vector and dff_timing field.
    pub fn load_timing_library(&mut self, lib: &crate::liberty_parser::TimingLibrary) {
        // Initialize gate delays for all AIG pins
        // Index 0 is Tie0 which has zero delay
        self.gate_delays = vec![(0, 0); self.num_aigpins + 1];

        // Get default AND gate delay (all variants have same timing in AIGPDK)
        let and_delay = lib.and_gate_delay("AND2_00_0").unwrap_or((1, 1));

        // Assign delays based on driver type
        for i in 1..=self.num_aigpins {
            let delay = match &self.drivers[i] {
                DriverType::AndGate(_, _) => and_delay,
                DriverType::InputPort(_) => (0, 0), // Primary inputs have zero arrival
                DriverType::InputClockFlag(_, _) => (0, 0), // Clock flags
                DriverType::DFF(_) => {
                    // Clock-to-Q delay
                    lib.dff_timing()
                        .map(|t| (t.clk_to_q_rise_ps, t.clk_to_q_fall_ps))
                        .unwrap_or((0, 0))
                }
                DriverType::SRAM(_) => {
                    // SRAM read delay
                    lib.sram_timing()
                        .map(|t| (t.read_clk_to_data_rise_ps, t.read_clk_to_data_fall_ps))
                        .unwrap_or((1, 1))
                }
                DriverType::Tie0 => (0, 0),
            };
            self.gate_delays[i] = delay;
        }

        // Load DFF timing constraints
        self.dff_timing = lib.dff_timing();

        clilog::info!(
            "Loaded timing library: AND delay={}ps, DFF setup={}ps, hold={}ps",
            and_delay.0,
            self.dff_timing.as_ref().map(|t| t.max_setup()).unwrap_or(0),
            self.dff_timing.as_ref().map(|t| t.max_hold()).unwrap_or(0)
        );
    }

    /// Compute static timing analysis (STA) for the AIG.
    ///
    /// This computes arrival times at all nodes and calculates setup/hold
    /// slacks for all DFFs. Must call `load_timing_library` first.
    ///
    /// Returns a `TimingReport` with summary statistics.
    pub fn compute_timing(&mut self) -> TimingReport {
        // Initialize arrival times (min, max) for all pins
        self.arrival_times = vec![(0, 0); self.num_aigpins + 1];

        // Get topological order (already guaranteed in AIG)
        // Traverse in order 1..num_aigpins since drivers are in topo order
        for i in 1..=self.num_aigpins {
            let (min_arrival, max_arrival) = match &self.drivers[i] {
                DriverType::AndGate(a, b) => {
                    let (delay_rise, delay_fall) = self.gate_delays[i];
                    let delay = delay_rise.max(delay_fall);

                    // Get arrival times of inputs (strip inversion bit)
                    let a_idx = a >> 1;
                    let b_idx = b >> 1;

                    let (a_min, a_max) = if a_idx == 0 {
                        (0, 0)
                    } else {
                        self.arrival_times[a_idx]
                    };
                    let (b_min, b_max) = if b_idx == 0 {
                        (0, 0)
                    } else {
                        self.arrival_times[b_idx]
                    };

                    // Min arrival: min of inputs + delay
                    // Max arrival: max of inputs + delay
                    (a_min.min(b_min) + delay, a_max.max(b_max) + delay)
                }
                DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) | DriverType::Tie0 => {
                    // Primary inputs arrive at time 0
                    (0, 0)
                }
                DriverType::DFF(_) => {
                    // DFF output arrives after clock-to-Q delay
                    let (delay_rise, delay_fall) = self.gate_delays[i];
                    (delay_rise.max(delay_fall), delay_rise.max(delay_fall))
                }
                DriverType::SRAM(_) => {
                    // SRAM read data arrives after read delay
                    let (delay_rise, delay_fall) = self.gate_delays[i];
                    (delay_rise.max(delay_fall), delay_rise.max(delay_fall))
                }
            };

            self.arrival_times[i] = (min_arrival, max_arrival);
        }

        // Compute setup and hold slacks for each DFF
        self.setup_slacks = Vec::with_capacity(self.dffs.len());
        self.hold_slacks = Vec::with_capacity(self.dffs.len());

        let dff_timing = self.dff_timing.clone().unwrap_or_default();
        let setup_time = dff_timing.max_setup();
        let hold_time = dff_timing.max_hold();

        let mut report = TimingReport::default();
        report.num_endpoints = self.dffs.len();

        for (_cell_id, dff) in &self.dffs {
            // Get data arrival time at D input
            let d_idx = dff.d_iv >> 1;
            let (_, data_max_arrival) = if d_idx == 0 {
                (0, 0)
            } else {
                self.arrival_times[d_idx]
            };

            // Clock arrival is at time 0 (or clock_period for setup check)
            // Setup check: data must arrive before clock_period - setup_time
            let setup_slack =
                (self.clock_period_ps as i64) - (data_max_arrival as i64) - (setup_time as i64);

            // Hold check: data must be stable for hold_time after clock
            // For hold, we check min arrival vs hold_time
            let (data_min_arrival, _) = if d_idx == 0 {
                (0, 0)
            } else {
                self.arrival_times[d_idx]
            };
            let hold_slack = (data_min_arrival as i64) - (hold_time as i64);

            self.setup_slacks.push(setup_slack);
            self.hold_slacks.push(hold_slack);

            // Update report statistics
            report.worst_setup_slack = report.worst_setup_slack.min(setup_slack);
            report.worst_hold_slack = report.worst_hold_slack.min(hold_slack);

            if setup_slack < 0 {
                report.setup_violations += 1;
            }
            if hold_slack < 0 {
                report.hold_violations += 1;
            }
        }

        // Find critical path (longest arrival time at any endpoint)
        for &po_iv in &self.primary_outputs {
            let po = po_iv >> 1; // Strip inversion bit
            if po > 0 && po <= self.num_aigpins {
                let (_, max_arr) = self.arrival_times[po];
                report.critical_path_delay = report.critical_path_delay.max(max_arr);
            }
        }
        for (_cell_id, dff) in &self.dffs {
            let d_idx = dff.d_iv >> 1;
            if d_idx > 0 {
                let (_, max_arr) = self.arrival_times[d_idx];
                report.critical_path_delay = report.critical_path_delay.max(max_arr);
            }
        }

        report
    }

    /// Get the critical path endpoints (nodes with longest arrival times).
    ///
    /// Returns a list of (aigpin, arrival_time) tuples, sorted by arrival time descending.
    pub fn get_critical_paths(&self, limit: usize) -> Vec<(usize, u64)> {
        let mut endpoints: Vec<(usize, u64)> = Vec::new();

        // Collect all endpoints (primary outputs and DFF D inputs)
        for &po_iv in &self.primary_outputs {
            let po = po_iv >> 1; // Strip inversion bit
            if po > 0 && po <= self.num_aigpins {
                let (_, max_arr) = self.arrival_times[po];
                endpoints.push((po, max_arr));
            }
        }
        for (_cell_id, dff) in &self.dffs {
            let d_idx = dff.d_iv >> 1;
            if d_idx > 0 && d_idx <= self.num_aigpins {
                let (_, max_arr) = self.arrival_times[d_idx];
                endpoints.push((d_idx, max_arr));
            }
        }

        // Sort by arrival time descending
        endpoints.sort_by(|a, b| b.1.cmp(&a.1));
        endpoints.truncate(limit);
        endpoints
    }

    /// Trace back the critical path from an endpoint.
    ///
    /// Returns a list of (aigpin, arrival_time) tuples from endpoint back to a primary input.
    pub fn trace_critical_path(&self, endpoint: usize) -> Vec<(usize, u64)> {
        let mut path = Vec::new();
        let mut current = endpoint;

        while current > 0 {
            let (_, arrival) = self.arrival_times[current];
            path.push((current, arrival));

            match &self.drivers[current] {
                DriverType::AndGate(a, b) => {
                    let a_idx = a >> 1;
                    let b_idx = b >> 1;

                    // Follow the input with larger arrival time
                    let a_arr = if a_idx > 0 {
                        self.arrival_times[a_idx].1
                    } else {
                        0
                    };
                    let b_arr = if b_idx > 0 {
                        self.arrival_times[b_idx].1
                    } else {
                        0
                    };

                    current = if a_arr >= b_arr { a_idx } else { b_idx };
                }
                _ => break, // Reached a primary input or sequential element
            }
        }

        path
    }
}

/// A reusable topological traverser with dense visited buffer.
///
/// Uses a generation counter pattern to avoid clearing the visited buffer
/// between traversals. This is significantly faster than `IndexSet<usize>`
/// for repeated traversals on the same AIG.
pub struct TopoTraverser {
    visited: Vec<u32>,
    generation: u32,
}

impl TopoTraverser {
    /// Create a new traverser for an AIG with the given number of pins.
    pub fn new(num_aigpins: usize) -> Self {
        Self {
            visited: vec![0; num_aigpins + 1],
            generation: 1,
        }
    }

    /// Reset the generation counter, preparing for a new traversal.
    fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Overflow: clear the buffer and reset
            self.visited.fill(0);
            self.generation = 1;
        }
    }

    #[inline(always)]
    fn is_visited(&self, u: usize) -> bool {
        self.visited[u] == self.generation
    }

    #[inline(always)]
    fn mark_visited(&mut self, u: usize) {
        self.visited[u] = self.generation;
    }

    /// Perform a topological traversal, equivalent to `AIG::topo_traverse_generic`.
    ///
    /// Uses an iterative stack-based DFS with a dense visited buffer for speed.
    pub fn topo_traverse(
        &mut self,
        aig: &AIG,
        endpoints: Option<&Vec<usize>>,
        is_primary_input: Option<&IndexSet<usize>>,
    ) -> Vec<usize> {
        self.new_generation();
        let mut ret = Vec::new();
        // Two-phase iterative DFS: Visit checks + pushes deps, Process emits
        enum Phase {
            Visit(usize),
            Process(usize),
        }
        let mut stack = Vec::new();

        if let Some(endpoints) = endpoints {
            for &endpoint in endpoints {
                stack.push(Phase::Visit(endpoint));
            }
        } else {
            for i in (1..aig.num_aigpins + 1).rev() {
                stack.push(Phase::Visit(i));
            }
        }

        while let Some(item) = stack.pop() {
            match item {
                Phase::Visit(u) => {
                    if self.is_visited(u) {
                        continue;
                    }
                    self.mark_visited(u);
                    stack.push(Phase::Process(u));
                    if let DriverType::AndGate(a, b) = aig.drivers[u] {
                        if is_primary_input.map(|s| s.contains(&u)) != Some(true) {
                            // Push b first so a is processed first (stack order)
                            if (b >> 1) != 0 {
                                stack.push(Phase::Visit(b >> 1));
                            }
                            if (a >> 1) != 0 {
                                stack.push(Phase::Visit(a >> 1));
                            }
                        }
                    }
                }
                Phase::Process(u) => {
                    ret.push(u);
                }
            }
        }
        ret
    }

    /// Perform a topological traversal and also produce a bitset of visited nodes.
    ///
    /// Returns `(order, bitset)` where `bitset` has bit `node` set for each visited node.
    /// The bitset is `Vec<u64>` with `(num_aigpins + 64) / 64` words.
    pub fn topo_traverse_with_bitset(
        &mut self,
        aig: &AIG,
        endpoints: Option<&Vec<usize>>,
        is_primary_input: Option<&IndexSet<usize>>,
    ) -> (Vec<usize>, Vec<u64>) {
        let order = self.topo_traverse(aig, endpoints, is_primary_input);
        let num_words = (aig.num_aigpins + 64) / 64;
        let mut bitset = vec![0u64; num_words];
        for &node in &order {
            bitset[node / 64] |= 1u64 << (node % 64);
        }
        (order, bitset)
    }
}

/// Compute the popcount of the union of two bitsets.
#[inline]
pub fn bitset_union_popcount(a: &[u64], b: &[u64]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x | y).count_ones() as usize)
        .sum()
}

/// OR bitset `src` into `dst` in-place.
#[inline]
pub fn bitset_or_inplace(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d |= *s;
    }
}

/// Summary of timing analysis results.
#[derive(Debug, Clone, Default)]
pub struct TimingReport {
    /// Number of timing endpoints analyzed (DFFs + primary outputs)
    pub num_endpoints: usize,
    /// Critical path delay in picoseconds
    pub critical_path_delay: u64,
    /// Worst setup slack (negative = violation)
    pub worst_setup_slack: i64,
    /// Worst hold slack (negative = violation)
    pub worst_hold_slack: i64,
    /// Number of setup violations
    pub setup_violations: usize,
    /// Number of hold violations
    pub hold_violations: usize,
}

impl TimingReport {
    /// Returns true if there are any timing violations.
    pub fn has_violations(&self) -> bool {
        self.setup_violations > 0 || self.hold_violations > 0
    }
}

impl std::fmt::Display for TimingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Timing Report ===")?;
        writeln!(f, "Endpoints analyzed: {}", self.num_endpoints)?;
        writeln!(f, "Critical path delay: {} ps", self.critical_path_delay)?;
        writeln!(f, "Worst setup slack: {} ps", self.worst_setup_slack)?;
        writeln!(f, "Worst hold slack: {} ps", self.worst_hold_slack)?;
        writeln!(f, "Setup violations: {}", self.setup_violations)?;
        writeln!(f, "Hold violations: {}", self.hold_violations)?;
        if self.has_violations() {
            writeln!(f, "Status: TIMING VIOLATIONS DETECTED")?;
        } else {
            writeln!(f, "Status: All timing constraints met")?;
        }
        Ok(())
    }
}

// === X-Propagation Static Analysis ===

/// Statistics from X-propagation analysis.
#[derive(Debug, Clone)]
pub struct XPropStats {
    /// Number of X sources (DFF Q outputs + SRAM read data ports).
    pub num_x_sources: usize,
    /// Number of X-capable AIG pins after forward propagation.
    pub num_x_capable_pins: usize,
    /// Total number of AIG pins.
    pub total_pins: usize,
    /// Number of fixpoint iterations needed.
    pub fixpoint_iterations: usize,
}

impl AIG {
    /// Identify DFF Q outputs and SRAM read data ports as X sources.
    ///
    /// Returns a boolean vector indexed by aigpin (0..=num_aigpins).
    /// `x_sources[i] == true` means aigpin `i` is an X source at cycle 0.
    pub fn compute_x_sources(&self) -> Vec<bool> {
        let mut x_sources = vec![false; self.num_aigpins + 1];

        // DFF Q outputs are X sources (unknown at power-on)
        for dff in self.dffs.values() {
            assert!(dff.q >= 1 && dff.q <= self.num_aigpins);
            x_sources[dff.q] = true;
        }

        // SRAM read data ports are X sources (memory contents undefined)
        for sram in self.srams.values() {
            for &rd_pin in &sram.port_r_rd_data {
                assert!(rd_pin >= 1 && rd_pin <= self.num_aigpins);
                x_sources[rd_pin] = true;
            }
        }

        x_sources
    }

    /// Compute the full set of X-capable AIG pins via forward cone propagation
    /// and DFF fixpoint iteration.
    ///
    /// Returns `(x_capable, stats)` where `x_capable[i]` is true if aigpin `i`
    /// can carry an X value during simulation.
    ///
    /// Algorithm:
    /// 1. Mark all X sources (DFF Q, SRAM read data)
    /// 2. Forward pass through AND gates (pins are in topological order)
    /// 3. Fixpoint: if any DFF's D-input is X-capable but its Q is not,
    ///    mark Q and re-run forward pass from newly-marked pins.
    pub fn compute_x_capable_pins(&self) -> (Vec<bool>, XPropStats) {
        let mut x_capable = self.compute_x_sources();
        let num_x_sources = x_capable.iter().filter(|&&v| v).count();

        let mut fixpoint_iterations = 0;

        loop {
            // Forward pass: propagate X through AND gates.
            // AIG pins are guaranteed to be in topological order.
            for aigpin in 1..=self.num_aigpins {
                if x_capable[aigpin] {
                    continue; // already marked
                }
                if let DriverType::AndGate(a_iv, b_iv) = self.drivers[aigpin] {
                    let a = a_iv >> 1;
                    let b = b_iv >> 1;
                    if x_capable[a] || x_capable[b] {
                        x_capable[aigpin] = true;
                    }
                }
            }

            // Check fixpoint: do any DFF D-inputs feed X back to an
            // unmarked Q output?
            let mut changed = false;
            for dff in self.dffs.values() {
                let d_pin = dff.d_iv >> 1;
                if x_capable[d_pin] && !x_capable[dff.q] {
                    x_capable[dff.q] = true;
                    changed = true;
                }
            }

            fixpoint_iterations += 1;

            if !changed {
                break;
            }
        }

        let num_x_capable_pins = x_capable.iter().filter(|&&v| v).count();
        let stats = XPropStats {
            num_x_sources,
            num_x_capable_pins,
            total_pins: self.num_aigpins,
            fixpoint_iterations,
        };

        (x_capable, stats)
    }
}

#[cfg(test)]
mod xprop_tests {
    use super::*;

    /// Create an AIG with proper Tie0 initialization (matching from_netlistdb_impl).
    fn new_test_aig() -> AIG {
        AIG {
            drivers: vec![DriverType::Tie0],
            aigpin_cell_origins: vec![Vec::new()],
            ..Default::default()
        }
    }

    /// Build a minimal AIG with no DFFs or SRAMs (pure combinational).
    fn build_comb_only_aig() -> AIG {
        let mut aig = new_test_aig();
        // pin 1 = InputPort
        aig.add_aigpin(DriverType::InputPort(0));
        // pin 2 = InputPort
        aig.add_aigpin(DriverType::InputPort(1));
        // pin 3 = AND(pin1, pin2) — no inversion: a_iv=2, b_iv=4
        aig.add_aigpin(DriverType::AndGate(1 << 1, 2 << 1));
        aig
    }

    #[test]
    fn test_x_capable_no_dffs() {
        let aig = build_comb_only_aig();
        let (x_capable, stats) = aig.compute_x_capable_pins();
        // No DFFs, no SRAMs → no X sources, no X-capable pins
        assert_eq!(stats.num_x_sources, 0);
        assert_eq!(stats.num_x_capable_pins, 0);
        assert_eq!(stats.fixpoint_iterations, 1);
        for v in &x_capable {
            assert!(!v);
        }
    }

    #[test]
    fn test_x_capable_dff_forward_cone() {
        // Build AIG: InputPort(1), DFF Q(2), AND(1,2)=3, AND(3,1)=4
        let mut aig = new_test_aig();
        aig.add_aigpin(DriverType::InputPort(0)); // pin 1
        aig.add_aigpin(DriverType::DFF(0)); // pin 2
        aig.add_aigpin(DriverType::AndGate(1 << 1, 2 << 1)); // pin 3 = AND(1,2)
        aig.add_aigpin(DriverType::AndGate(3 << 1, 1 << 1)); // pin 4 = AND(3,1)

        // Register the DFF: D input is pin 3, Q output is pin 2
        aig.dffs.insert(
            0,
            DFF {
                d_iv: 3 << 1, // pin 3, no inversion
                en_iv: 0,     // always enabled
                q: 2,
            },
        );

        let (x_capable, stats) = aig.compute_x_capable_pins();

        assert_eq!(stats.num_x_sources, 1); // Just the DFF Q
        // Pin 2 (DFF Q) is X source, pin 3 (AND(1,2)) is X-capable,
        // pin 4 (AND(3,1)) is X-capable
        assert!(!x_capable[0]); // Tie0
        assert!(!x_capable[1]); // InputPort — not X
        assert!(x_capable[2]); // DFF Q — X source
        assert!(x_capable[3]); // AND(1,2) — b is X
        assert!(x_capable[4]); // AND(3,1) — a is X (via pin 3)
        assert_eq!(stats.num_x_capable_pins, 3);
    }

    #[test]
    fn test_x_capable_fixpoint() {
        // DFF feedback loop: DFF Q(1) -> AND(1, input) -> DFF2 Q(3) -> AND(3, input) -> back to DFF D
        // This tests that fixpoint converges when X propagates through DFF cycles.
        let mut aig = new_test_aig();
        aig.add_aigpin(DriverType::InputPort(0)); // pin 1
        aig.add_aigpin(DriverType::DFF(0)); // pin 2 (DFF0 Q)
        aig.add_aigpin(DriverType::AndGate(2 << 1, 1 << 1)); // pin 3 = AND(2,1)
        aig.add_aigpin(DriverType::DFF(1)); // pin 4 (DFF1 Q)
        aig.add_aigpin(DriverType::AndGate(4 << 1, 1 << 1)); // pin 5 = AND(4,1)

        // DFF0: D=pin5, Q=pin2
        aig.dffs.insert(
            0,
            DFF {
                d_iv: 5 << 1,
                en_iv: 0,
                q: 2,
            },
        );
        // DFF1: D=pin3, Q=pin4
        aig.dffs.insert(
            1,
            DFF {
                d_iv: 3 << 1,
                en_iv: 0,
                q: 4,
            },
        );

        let (x_capable, stats) = aig.compute_x_capable_pins();

        // Both DFFs are initially X sources, their forward cones too
        assert!(x_capable[2]); // DFF0 Q
        assert!(x_capable[3]); // AND(2,1)
        assert!(x_capable[4]); // DFF1 Q
        assert!(x_capable[5]); // AND(4,1)
        assert!(!x_capable[1]); // InputPort
        assert_eq!(stats.num_x_sources, 2);
        // Should converge in 1 iteration since both DFFs are already X sources
        assert_eq!(stats.fixpoint_iterations, 1);
    }

    #[test]
    fn test_x_capable_fixpoint_indirect() {
        // Test case where fixpoint needs >1 iteration:
        // InputPort(1), AND(1,1)=2, DFF0 Q=3 (D=2), AND(3,1)=4, DFF1 Q=5 (D=4)
        // DFF1 is initially NOT an X source if we only mark DFF0.
        // But wait — ALL DFFs are X sources by definition. So fixpoint always
        // converges in 1 iteration for the current algorithm.
        //
        // The fixpoint >1 case would only arise if we selectively excluded
        // some DFFs from X sources (e.g., reset-aware analysis). For now,
        // verify that the algorithm handles the general case correctly.
        let mut aig = new_test_aig();
        aig.add_aigpin(DriverType::InputPort(0)); // pin 1
        aig.add_aigpin(DriverType::DFF(0)); // pin 2 (initially X)
        aig.add_aigpin(DriverType::AndGate(2 << 1, 1 << 1)); // pin 3 = AND(2,1)
        aig.add_aigpin(DriverType::DFF(1)); // pin 4 (initially X)

        aig.dffs.insert(
            0,
            DFF {
                d_iv: 1 << 1,
                en_iv: 0,
                q: 2,
            },
        );
        aig.dffs.insert(
            1,
            DFF {
                d_iv: 3 << 1,
                en_iv: 0,
                q: 4,
            },
        );

        let (x_capable, stats) = aig.compute_x_capable_pins();
        assert!(x_capable[2]); // DFF0 Q — X source
        assert!(x_capable[3]); // AND(2,1) — X via DFF0
        assert!(x_capable[4]); // DFF1 Q — X source
        assert!(!x_capable[1]); // InputPort
        assert_eq!(stats.fixpoint_iterations, 1);
    }

    #[test]
    fn test_x_sources_sram() {
        // Test that SRAM read data ports are marked as X sources
        let mut aig = new_test_aig();
        aig.add_aigpin(DriverType::InputPort(0)); // pin 1

        // Create 32 SRAM read data pins
        let mut rd_data = [0usize; 32];
        for i in 0..32 {
            let pin = aig.add_aigpin(DriverType::SRAM(0)); // pins 2..33
            rd_data[i] = pin;
        }

        aig.srams.insert(
            0,
            RAMBlock {
                port_r_addr_iv: [0; AIGPDK_SRAM_ADDR_WIDTH],
                port_r_en_iv: 0,
                port_r_rd_data: rd_data,
                port_w_addr_iv: [0; AIGPDK_SRAM_ADDR_WIDTH],
                port_w_wr_en_iv: [0; 32],
                port_w_wr_data_iv: [0; 32],
            },
        );

        let x_sources = aig.compute_x_sources();
        assert!(!x_sources[0]); // Tie0
        assert!(!x_sources[1]); // InputPort
        for i in 0..32 {
            assert!(x_sources[rd_data[i]], "SRAM rd_data[{i}] should be X source");
        }

        let (x_capable, stats) = aig.compute_x_capable_pins();
        assert_eq!(stats.num_x_sources, 32);
        assert_eq!(stats.num_x_capable_pins, 32);
        // Verify forward propagation through AND gate
        // Add an AND gate that depends on SRAM read data
        // (already tested via the basic structure above)
        drop(x_capable);
    }

    #[test]
    fn test_x_prop_with_inv_chain() {
        // Integration test using the real inv_chain.v design
        let verilog_path =
            std::path::PathBuf::from("tests/timing_test/sky130_timing/inv_chain.v");
        if !verilog_path.exists() {
            eprintln!("Skipping test_x_prop_with_inv_chain: inv_chain.v not found");
            return;
        }
        use crate::sky130::SKY130LeafPins;
        let netlistdb = NetlistDB::from_sverilog_file(&verilog_path, None, &SKY130LeafPins)
            .expect("cannot build netlist");
        let aig = AIG::from_netlistdb(&netlistdb);

        let (x_capable, stats) = aig.compute_x_capable_pins();

        // inv_chain has 2 DFFs, so at least 2 X sources
        assert!(stats.num_x_sources >= 2, "Expected at least 2 X sources from DFFs, got {}", stats.num_x_sources);
        // The forward cone should propagate X through the inverter chain
        assert!(stats.num_x_capable_pins >= stats.num_x_sources,
            "X-capable pins ({}) should be >= X sources ({})",
            stats.num_x_capable_pins, stats.num_x_sources);
        // Sanity: not everything should be X-capable (inputs are known)
        assert!(stats.num_x_capable_pins < stats.total_pins,
            "Not all pins should be X-capable");

        println!(
            "inv_chain X-prop: {}/{} pins ({:.1}%) X-capable, {} fixpoint iterations",
            stats.num_x_capable_pins,
            stats.total_pins,
            100.0 * stats.num_x_capable_pins as f64 / stats.total_pins as f64,
            stats.fixpoint_iterations
        );

        // Verify no input ports are X-capable
        for (aigpin, driver) in aig.drivers.iter().enumerate() {
            if matches!(driver, DriverType::InputPort(_) | DriverType::InputClockFlag(_, _)) {
                assert!(!x_capable[aigpin], "Input port aigpin {} should not be X-capable", aigpin);
            }
        }
    }
}

#[cfg(test)]
mod sdf_integration_tests {
    use super::*;
    use crate::sdf_parser::{SdfCorner, SdfFile};
    use crate::sky130::SKY130LeafPins;
    use std::collections::HashMap;

    /// Helper: load inv_chain.v and build AIG.
    fn load_inv_chain_aig() -> (NetlistDB, AIG) {
        let verilog_path = std::path::PathBuf::from("tests/timing_test/sky130_timing/inv_chain.v");
        assert!(verilog_path.exists(), "inv_chain.v not found");

        let netlistdb = NetlistDB::from_sverilog_file(&verilog_path, None, &SKY130LeafPins)
            .expect("Failed to parse inv_chain.v");

        let aig = AIG::from_netlistdb(&netlistdb);
        (netlistdb, aig)
    }

    /// Helper: build cellid → SDF path map (same logic as load_timing_from_sdf).
    fn build_cellid_to_sdf_path(netlistdb: &NetlistDB) -> HashMap<usize, String> {
        let mut map = HashMap::new();
        for cellid in 1..netlistdb.num_cells {
            let parts: Vec<&str> = netlistdb.cellnames[cellid]
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>();
            let sdf_path: String = parts.iter().rev().cloned().collect::<Vec<_>>().join(".");
            map.insert(cellid, sdf_path);
        }
        map
    }

    // === Test 1: Cell origin population ===

    #[test]
    fn test_cell_origins_length() {
        let (_netlistdb, aig) = load_inv_chain_aig();
        assert_eq!(
            aig.aigpin_cell_origins.len(),
            aig.num_aigpins + 1,
            "aigpin_cell_origins should have num_aigpins + 1 entries (index 0 = Tie0)"
        );
    }

    #[test]
    fn test_cell_origin_tie0_is_empty() {
        let (_netlistdb, aig) = load_inv_chain_aig();
        assert!(
            aig.aigpin_cell_origins[0].is_empty(),
            "Index 0 (Tie0) must have no cell origins"
        );
    }

    #[test]
    fn test_cell_origins_accumulated() {
        let (_netlistdb, aig) = load_inv_chain_aig();

        // With accumulated origins, inverters no longer overwrite — they push.
        // For inv_chain: dff_in.Q → i0 → ... → i15 → dff_out.D
        // The shared AIG pin (dff_in.Q / inverter chain) accumulates:
        //   1 DFF origin (dff_in CLK→Q) + 16 inverter origins = 17 entries.
        // dff_out.Q gets its own AIG pin with 1 entry.
        let non_empty_count = aig
            .aigpin_cell_origins
            .iter()
            .filter(|o| !o.is_empty())
            .count();

        // 2 AIG pins with non-empty origins: shared chain pin + dff_out
        assert_eq!(
            non_empty_count, 2,
            "Expected 2 AIG pins with cell origins (chain pin + dff_out), got {}",
            non_empty_count
        );

        // The shared pin should have 17 entries (1 DFF + 16 inverters)
        let chain_pin = aig
            .aigpin_cell_origins
            .iter()
            .find(|origins| origins.len() > 1)
            .expect("Should have a pin with multiple origins (the inverter chain)");
        assert_eq!(
            chain_pin.len(),
            17,
            "Shared chain pin should have 17 origins (1 DFF + 16 inverters), got {}",
            chain_pin.len()
        );
    }

    #[test]
    fn test_cell_origin_dff_output_pin() {
        let (netlistdb, aig) = load_inv_chain_aig();

        // Both DFFs should now be present (dff_in is no longer overwritten)
        let dff_origins: Vec<_> = aig
            .aigpin_cell_origins
            .iter()
            .enumerate()
            .flat_map(|(i, origins)| {
                origins
                    .iter()
                    .map(move |(cid, ct, pin)| (i, *cid, ct.clone(), pin.clone()))
            })
            .filter(|(_, _, ct, _)| ct == "dfxtp")
            .collect();

        assert_eq!(
            dff_origins.len(),
            2,
            "Expected 2 DFF cell origins (dff_in + dff_out), got {}",
            dff_origins.len()
        );

        for (ref _aigpin, ref cellid, ref _cell_type, ref output_pin_name) in &dff_origins {
            assert_eq!(output_pin_name, "Q", "DFF output pin must be Q");
            assert!(
                netlistdb.celltypes[*cellid].contains("dfxtp"),
                "Cell {} should be dfxtp type, got {}",
                cellid,
                netlistdb.celltypes[*cellid]
            );
        }
    }

    #[test]
    fn test_all_inverters_present() {
        let (netlistdb, aig) = load_inv_chain_aig();

        // All 16 inverters should now have their origins accumulated (not just the last one)
        let inv_origins: Vec<_> = aig
            .aigpin_cell_origins
            .iter()
            .enumerate()
            .flat_map(|(i, origins)| {
                origins
                    .iter()
                    .map(move |(cid, ct, pin)| (i, *cid, ct.clone(), pin.clone()))
            })
            .filter(|(_, _, ct, _)| ct == "inv")
            .collect();

        assert_eq!(
            inv_origins.len(),
            16,
            "Expected 16 inverter cell origins (all inverters in chain), got {}",
            inv_origins.len()
        );

        for (ref _aigpin, ref cellid, ref _cell_type, ref output_pin_name) in &inv_origins {
            assert_eq!(output_pin_name, "Y", "Inverter output pin must be Y");
            assert!(
                netlistdb.celltypes[*cellid].contains("inv"),
                "Cell {} should be inv type, got {}",
                cellid,
                netlistdb.celltypes[*cellid]
            );
        }
    }

    #[test]
    fn test_no_and_gates_for_inverters() {
        let (_netlistdb, aig) = load_inv_chain_aig();

        // Inverters in AIG are just wire + invert bit, no AND gates needed.
        // The only AND gates should be none (inv_chain has no multi-input gates).
        let and_gate_count = (1..=aig.num_aigpins)
            .filter(|&ap| matches!(aig.drivers[ap], DriverType::AndGate(_, _)))
            .count();

        assert_eq!(
            and_gate_count, 0,
            "Inverter chain should produce 0 AND gates, got {}",
            and_gate_count
        );
    }

    // === Test 2: HierName → SDF path matching ===

    #[test]
    fn test_sdf_path_matching_all_cells() {
        let (netlistdb, _aig) = load_inv_chain_aig();
        let sdf_content = include_str!("../tests/timing_test/inv_chain_pnr/inv_chain_test.sdf");
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ).expect("Failed to parse SDF");

        let cellid_to_sdf_path = build_cellid_to_sdf_path(&netlistdb);

        let mut matched = 0;
        let mut unmatched = Vec::new();

        for cellid in 1..netlistdb.num_cells {
            let sdf_path = &cellid_to_sdf_path[&cellid];
            if sdf.get_cell(sdf_path).is_some() {
                matched += 1;
            } else {
                unmatched.push(format!(
                    "cellid={} type={} path={}",
                    cellid, netlistdb.celltypes[cellid], sdf_path
                ));
            }
        }

        assert!(
            unmatched.is_empty(),
            "All cells should match SDF instances. Unmatched:\n{}",
            unmatched.join("\n")
        );
        // 18 real cells (2 DFFs + 16 inverters)
        assert_eq!(matched, 18, "Expected 18 matched cells");
    }

    #[test]
    fn test_sdf_path_format_flat() {
        // For a flat design, cellnames should produce single-component paths
        let (netlistdb, _aig) = load_inv_chain_aig();
        let cellid_to_sdf_path = build_cellid_to_sdf_path(&netlistdb);

        // Verify expected instance names exist
        let expected_names = [
            "dff_in", "i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10", "i11",
            "i12", "i13", "i14", "i15", "dff_out",
        ];

        let paths: Vec<&str> = cellid_to_sdf_path.values().map(|s| s.as_str()).collect();
        for name in &expected_names {
            assert!(
                paths.contains(name),
                "Expected SDF path '{}' not found. Paths: {:?}",
                name,
                paths
            );
        }
    }
}
