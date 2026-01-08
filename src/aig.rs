// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! And-inverter graph format
//!
//! An AIG is derived from netlistdb synthesized in AIGPDK.

use netlistdb::{NetlistDB, GeneralPinName, Direction};
use indexmap::{IndexMap, IndexSet};
use crate::aigpdk::AIGPDK_SRAM_ADDR_WIDTH;

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
            if i >= 1 { f_nz(i); }
        };
        match self {
            Self::PrimaryOutput(idx) => f(idx >> 1),
            Self::DFF(dff) => {
                f(dff.en_iv >> 1);
                f(dff.d_iv >> 1);
            },
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
            },
            Self::SimControl(ctrl) => {
                f(ctrl.condition_iv >> 1);
            },
            Self::Display(disp) => {
                f(disp.enable_iv >> 1);
                for &arg_iv in &disp.args_iv {
                    f(arg_iv >> 1);
                }
            },
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
    Tie0
}

/// An AIG associated with a netlistdb.
#[derive(Debug, Default)]
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
}

impl AIG {
    fn add_aigpin(&mut self, driver: DriverType) -> usize {
        self.num_aigpins += 1;
        self.drivers.push(driver);
        self.num_aigpins
    }

    fn add_and_gate(&mut self, a: usize, b: usize) -> usize {
        assert_ne!(a | 1, usize::MAX);
        assert_ne!(b | 1, usize::MAX);
        if a == 0 || b == 0 {
            return 0
        }
        if a == 1 {
            return b
        }
        if b == 1 {
            return a
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
        pinid: usize, is_negedge: bool,
        // should we ignore cklnqd in this tracing.
        // if set to true, we will treat cklnqd as a simple buffer.
        // otherwise, we assert that cklnqd/en is already built in
        // our aig mapping (pin2aigpin_iv).
        ignore_cklnqd: bool,
    ) -> Result<usize, usize> {
        if netlistdb.pindirect[pinid] == Direction::I {
            let netid = netlistdb.pin2net[pinid];
            if Some(netid) == netlistdb.net_zero || Some(netid) == netlistdb.net_one {
                return Ok(0)
            }
            let root = netlistdb.net2pin.items[
                netlistdb.net2pin.start[netid]
            ];
            return self.trace_clock_pin(
                netlistdb, root, is_negedge,
                ignore_cklnqd
            )
        }
        let cellid = netlistdb.pin2cell[pinid];
        if cellid == 0 {
            let clkentry = self.clock_pin2aigpins.entry(pinid)
                .or_insert((usize::MAX, usize::MAX));
            let clksignal = match is_negedge {
                false => clkentry.0,
                true => clkentry.1
            };
            if clksignal != usize::MAX {
                return Ok(clksignal << 1)
            }
            let aigpin = self.add_aigpin(DriverType::InputClockFlag(pinid, is_negedge as u8));
            let clkentry = self.clock_pin2aigpins.get_mut(&pinid).unwrap();
            let clksignal = match is_negedge {
                false => &mut clkentry.0,
                true => &mut clkentry.1
            };
            *clksignal = aigpin;
            return Ok(aigpin << 1)
        }
        let mut pin_a = usize::MAX;
        let mut pin_cp = usize::MAX;
        let mut pin_en = usize::MAX;
        let celltype = netlistdb.celltypes[cellid].as_str();
        if !matches!(celltype, "INV" | "BUF" | "CKLNQD") {
            clilog::error!("cell type {} supported on clock path. expecting only INV, BUF, or CKLNQD", celltype);
            return Err(pinid)
        }
        for ipin in netlistdb.cell2pin.iter_set(cellid) {
            if netlistdb.pindirect[ipin] == Direction::I {
                match netlistdb.pinnames[ipin].1.as_str() {
                    "A" => pin_a = ipin,
                    "CP" => pin_cp = ipin,
                    "E" => pin_en = ipin,
                    i @ _ => {
                        clilog::error!("input pin {} unexpected for ck element {}", i, celltype);
                        return Err(ipin)
                    }
                }
            }
        }
        match celltype {
            "INV" => {
                assert_ne!(pin_a, usize::MAX);
                self.trace_clock_pin(
                    netlistdb, pin_a, !is_negedge,
                    ignore_cklnqd
                )
            },
            "BUF" => {
                assert_ne!(pin_a, usize::MAX);
                self.trace_clock_pin(
                    netlistdb, pin_a, is_negedge,
                    ignore_cklnqd
                )
            },
            "CKLNQD" => {
                assert_ne!(pin_cp, usize::MAX);
                assert_ne!(pin_en, usize::MAX);
                let ck_iv = self.trace_clock_pin(
                    netlistdb, pin_cp, is_negedge,
                    ignore_cklnqd
                )?;
                if ignore_cklnqd {
                    return Ok(ck_iv)
                }
                let en_iv = self.pin2aigpin_iv[pin_en];
                assert_ne!(en_iv, usize::MAX, "clken not built");
                Ok(self.add_and_gate(ck_iv, en_iv))
            },
            _ => unreachable!()
        }
    }

    /// recursively add aig pins for netlistdb pins
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
        pinid: usize
    ) {
        if topo_instack[pinid] {
            panic!("circuit has a loop around pin {}",
                   netlistdb.pinnames[pinid].dbg_fmt_pin());
        }
        if topo_vis[pinid] {
            return
        }
        topo_vis[pinid] = true;
        topo_instack[pinid] = true;
        let netid = netlistdb.pin2net[pinid];
        let cellid = netlistdb.pin2cell[pinid];
        let celltype = netlistdb.celltypes[cellid].as_str();
        if netlistdb.pindirect[pinid] == Direction::I {
            if Some(netid) == netlistdb.net_zero {
                self.pin2aigpin_iv[pinid] = 0;
            }
            else if Some(netid) == netlistdb.net_one {
                self.pin2aigpin_iv[pinid] = 1;
            }
            else {
                let root = netlistdb.net2pin.items[
                    netlistdb.net2pin.start[netid]
                ];
                self.dfs_netlistdb_build_aig(
                    netlistdb, topo_vis, topo_instack,
                    root
                );
                self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[root];
                if cellid == 0 {
                    self.primary_outputs.insert(self.pin2aigpin_iv[pinid]);
                }
            }
        }
        else if cellid == 0 {
            let aigpin = self.add_aigpin(
                DriverType::InputPort(pinid)
            );
            self.pin2aigpin_iv[pinid] = aigpin << 1;
        }
        else if matches!(celltype, "DFF" | "DFFSR") {
            let q = self.add_aigpin(DriverType::DFF(cellid));
            let dff = self.dffs.entry(cellid).or_default();
            dff.q = q;
            let mut ap_s_iv = 1;
            let mut ap_r_iv = 1;
            let mut q_out = q << 1;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if !matches!(netlistdb.pinnames[pinid].1.as_str(), "S" | "R") {
                    continue
                }
                self.dfs_netlistdb_build_aig(
                    netlistdb, topo_vis, topo_instack, pinid
                );
                let prev = self.pin2aigpin_iv[pinid];
                match netlistdb.pinnames[pinid].1.as_str() {
                    "S" => ap_s_iv = prev,
                    "R" => ap_r_iv = prev,
                    _ => unreachable!()
                }
            }
            q_out = self.add_and_gate(q_out ^ 1, ap_s_iv) ^ 1;
            q_out = self.add_and_gate(q_out, ap_r_iv);
            self.pin2aigpin_iv[pinid] = q_out;
        }
        else if celltype == "LATCH" {
            panic!("latches are intentionally UNSUPPORTED by GEM, \
                    except in identified gated clocks. \n\
                    you can link a FF&MUX-based LATCH module, \
                    but most likely that is NOT the right solution. \n\
                    check all your assignments inside always@(*) block \
                    to make sure they cover all scenarios.");
        }
        else if celltype == "$__RAMGEM_SYNC_" {
            let o = self.add_aigpin(DriverType::SRAM(cellid));
            self.pin2aigpin_iv[pinid] = o << 1;
            assert_eq!(netlistdb.pinnames[pinid].1.as_str(),
                       "PORT_R_RD_DATA");
            let sram = self.srams.entry(cellid).or_default();
            sram.port_r_rd_data[netlistdb.pinnames[pinid].2.unwrap() as usize] = o;
        }
        else if celltype == "CKLNQD" {
            let mut prev_cp = usize::MAX;
            let mut prev_en = usize::MAX;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                match netlistdb.pinnames[pinid].1.as_str() {
                    "CP" => prev_cp = pinid,
                    "E" => prev_en = pinid,
                    _ => {}
                }
            }
            assert_ne!(prev_cp, usize::MAX);
            assert_ne!(prev_en, usize::MAX);
            for prev in [prev_cp, prev_en] {
                self.dfs_netlistdb_build_aig(
                    netlistdb, topo_vis, topo_instack,
                    prev
                );
            }
            // do not define pin2aigpin_iv[pinid] which is CKLNQD/Q and unused in logic.
        }
        else if celltype == "GEM_ASSERT" || celltype == "GEM_DISPLAY" {
            // These are side-effect only cells with no outputs
            // Visit all input pins to build their AIG representations
            for input_pinid in netlistdb.cell2pin.iter_set(cellid) {
                if netlistdb.pindirect[input_pinid] == Direction::I {
                    self.dfs_netlistdb_build_aig(
                        netlistdb, topo_vis, topo_instack,
                        input_pinid
                    );
                }
            }
            // No output pin to define
        }
        else {
            let mut prev_a = usize::MAX;
            let mut prev_b = usize::MAX;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                match netlistdb.pinnames[pinid].1.as_str() {
                    "A" => prev_a = pinid,
                    "B" => prev_b = pinid,
                    _ => {}
                }
            }
            for prev in [prev_a, prev_b] {
                if prev != usize::MAX {
                    self.dfs_netlistdb_build_aig(
                        netlistdb, topo_vis, topo_instack,
                        prev
                    );
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
                },
                "INV" => {
                    assert_ne!(prev_a, usize::MAX);
                    self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a] ^ 1;
                },
                "BUF" => {
                    assert_ne!(prev_a, usize::MAX);
                    self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a];
                },
                _ => unreachable!()
            }
        }
        topo_instack[pinid] = false;
    }

    pub fn from_netlistdb(netlistdb: &NetlistDB) -> AIG {
        let mut aig = AIG {
            num_aigpins: 0,
            pin2aigpin_iv: vec![usize::MAX; netlistdb.num_pins],
            drivers: vec![DriverType::Tie0],
            ..Default::default()
        };

        for cellid in 1..netlistdb.num_cells {
            if !matches!(netlistdb.celltypes[cellid].as_str(),
                         "DFF" | "DFFSR" | "$__RAMGEM_SYNC_") {
                continue
            }
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if !matches!(netlistdb.pinnames[pinid].1.as_str(),
                            "CLK" | "PORT_R_CLK" | "PORT_W_CLK") {
                    continue
                }
                if let Err(pinid) = aig.trace_clock_pin(
                    netlistdb, pinid, false,
                    true
                ) {
                    use netlistdb::GeneralHierName;
                    panic!("Tracing clock pin of cell {} error: \
                            there is a multi-input cell driving {} \
                            that clocks this sequential element. \
                            Clock gating need to be manually patched atm.",
                           netlistdb.cellnames[cellid].dbg_fmt_hier(),
                           netlistdb.pinnames[pinid].dbg_fmt_pin());
                }
            }
        }
        for (&clk, &(flagr, flagf)) in &aig.clock_pin2aigpins {
            clilog::info!(
                "inferred clock port {} ({})",
                netlistdb.pinnames[clk].dbg_fmt_pin(),
                match (flagr, flagf) {
                    (_, usize::MAX) => "posedge",
                    (usize::MAX, _) => "negedge",
                    _ => "posedge & negedge"
                }
            );
        }

        let mut topo_vis = vec![false; netlistdb.num_pins];
        let mut topo_instack = vec![false; netlistdb.num_pins];

        for pinid in 0..netlistdb.num_pins {
            aig.dfs_netlistdb_build_aig(
                netlistdb, &mut topo_vis, &mut topo_instack,
                pinid
            );
        }

        for cellid in 0..netlistdb.num_cells {
            if matches!(netlistdb.celltypes[cellid].as_str(), "DFF" | "DFFSR") {
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
                        "CLK" => ap_clken_iv = aig.trace_clock_pin(
                            netlistdb, pinid, false,
                            false
                        ).unwrap(),
                        _ => {}
                    }
                }
                let mut d_in = ap_d_iv;

                d_in = aig.add_and_gate(d_in ^ 1, ap_s_iv) ^ 1;
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_s_iv) ^ 1;
                d_in = aig.add_and_gate(d_in, ap_r_iv);
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_r_iv) ^ 1;
                let dff = aig.dffs.entry(cellid).or_default();
                dff.en_iv = ap_clken_iv;
                dff.d_iv = d_in;
                assert_ne!(dff.q, 0);
            }
            else if netlistdb.celltypes[cellid].as_str() == "$__RAMGEM_SYNC_" {
                let mut sram = aig.srams.entry(cellid).or_default().clone();
                let mut write_clken_iv = 0;
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let bit = netlistdb.pinnames[pinid].2.map(|i| i as usize);
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "PORT_R_ADDR" => {
                            sram.port_r_addr_iv[bit.unwrap()] = pin_iv;
                        },
                        "PORT_R_CLK" => {
                            sram.port_r_en_iv = aig.trace_clock_pin(
                                netlistdb, pinid, false,
                                false
                            ).unwrap();
                        },
                        "PORT_W_ADDR" => {
                            sram.port_w_addr_iv[bit.unwrap()] = pin_iv;
                        }
                        "PORT_W_CLK" => {
                            write_clken_iv = aig.trace_clock_pin(
                                netlistdb, pinid, false,
                                false
                            ).unwrap();
                        },
                        "PORT_W_WR_DATA" => {
                            sram.port_w_wr_data_iv[bit.unwrap()] = pin_iv;
                        },
                        "PORT_W_WR_EN" => {
                            sram.port_w_wr_en_iv[bit.unwrap()] = pin_iv;
                        },
                        _ => {}
                    }
                }
                for i in 0..32 {
                    let or_en = sram.port_w_wr_en_iv[i];
                    let or_en = aig.add_and_gate(
                        or_en, write_clken_iv
                    );
                    sram.port_w_wr_en_iv[i] = or_en;
                }
                *aig.srams.get_mut(&cellid).unwrap() = sram;
            }
            else if netlistdb.celltypes[cellid].as_str() == "GEM_ASSERT" {
                // Parse GEM_ASSERT cells for assertion checking
                // GEM_ASSERT has: CLK (trigger), EN (enable), A (condition)
                // Assertion fails when EN is high and A is low
                let mut ap_en_iv = 1;  // Default: always enabled
                let mut ap_a_iv = 1;   // Default: always passing
                let mut ap_clken_iv = 1; // Default: always triggered

                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "EN" => ap_en_iv = pin_iv,
                        "A" => ap_a_iv = pin_iv,
                        "CLK" => {
                            // Try to trace clock, but if it's tied to 1, use constant
                            if let Ok(clken) = aig.trace_clock_pin(
                                netlistdb, pinid, false, false
                            ) {
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
                    cellid, fire_condition, ap_en_iv, ap_a_iv, ap_clken_iv
                );
            }
            else if netlistdb.celltypes[cellid].as_str() == "GEM_DISPLAY" {
                // Parse GEM_DISPLAY cells for $display/$write support
                // GEM_DISPLAY has: CLK (trigger), EN (enable), MSG_ID[31:0] (argument values)
                // Plus attributes: gem_format (format string), gem_args_width (arg width)
                let mut dp_en_iv = 1;      // Default: always enabled
                let mut dp_clken_iv = 1;   // Default: always triggered
                let mut dp_args_iv = Vec::new();

                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "EN" => dp_en_iv = pin_iv,
                        "CLK" => {
                            // Try to trace clock for edge detection
                            if let Ok(clken) = aig.trace_clock_pin(
                                netlistdb, pinid, false, false
                            ) {
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
                    cellid, display.cell_name, fire_condition, dp_clken_iv, display.args_iv.len()
                );
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
        fn dfs_topo(aig: &AIG, vis: &mut IndexSet<usize>, ret: &mut Vec<usize>, is_primary_input: Option<&IndexSet<usize>>, u: usize) {
            if vis.contains(&u) {
                return
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
        }
        else {
            for i in 1..self.num_aigpins + 1 {
                dfs_topo(self, &mut vis, &mut ret, is_primary_input, i);
            }
        }
        ret
    }

    pub fn num_endpoint_groups(&self) -> usize {
        self.primary_outputs.len() + self.dffs.len() + self.srams.len() +
        self.simcontrols.len() + self.displays.len()
    }

    pub fn get_endpoint_group(&self, endpt_id: usize) -> EndpointGroup {
        let po_len = self.primary_outputs.len();
        let dff_len = self.dffs.len();
        let sram_len = self.srams.len();
        let simctrl_len = self.simcontrols.len();

        if endpt_id < po_len {
            EndpointGroup::PrimaryOutput(*self.primary_outputs.get_index(endpt_id).unwrap())
        }
        else if endpt_id < po_len + dff_len {
            EndpointGroup::DFF(&self.dffs[endpt_id - po_len])
        }
        else if endpt_id < po_len + dff_len + sram_len {
            EndpointGroup::RAMBlock(&self.srams[endpt_id - po_len - dff_len])
        }
        else if endpt_id < po_len + dff_len + sram_len + simctrl_len {
            EndpointGroup::SimControl(&self.simcontrols[endpt_id - po_len - dff_len - sram_len])
        }
        else {
            EndpointGroup::Display(&self.displays[endpt_id - po_len - dff_len - sram_len - simctrl_len])
        }
    }

    /// Populate display format information from JSON attributes.
    /// This should be called after from_netlistdb() with display info extracted from the JSON.
    pub fn populate_display_info(&mut self, display_info: &IndexMap<String, crate::display::DisplayCellInfo>) {
        for (_cell_id, display) in self.displays.iter_mut() {
            if let Some(info) = display_info.get(&display.cell_name) {
                display.format = info.format.clone();
                display.arg_widths = vec![info.args_width];
                clilog::debug!(
                    "Populated display info for '{}': format='{}', args_width={}",
                    display.cell_name, info.format, info.args_width
                );
            }
        }
    }
}
