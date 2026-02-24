// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! VCD input/output utilities for simulation.
//!
//! Shared VCD I/O utilities for `loom sim`.

use compact_str::CompactString;
use netlistdb::{Direction, GeneralPinName, NetlistDB};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;
use vcd_ng::{Scope, ScopeItem, Var};

use crate::aig::{DriverType, AIG};
use crate::flatten::FlattenedScriptV1;

// ── VCDHier: hierarchical name representation in VCD ────────────────────────

/// Hierarchical name representation in VCD.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct VCDHier {
    pub cur: CompactString,
    pub prev: Option<Rc<VCDHier>>,
}

/// Reverse iterator of a [`VCDHier`], yielding cell names
/// from the bottom to the top module.
pub struct VCDHierRevIter<'i>(Option<&'i VCDHier>);

impl<'i> Iterator for VCDHierRevIter<'i> {
    type Item = &'i CompactString;

    #[inline]
    fn next(&mut self) -> Option<&'i CompactString> {
        let name = self.0?;
        if name.cur.is_empty() {
            return None;
        }
        let ret = &name.cur;
        self.0 = name.prev.as_ref().map(|a| a.as_ref());
        Some(ret)
    }
}

impl<'i> IntoIterator for &'i VCDHier {
    type Item = &'i CompactString;
    type IntoIter = VCDHierRevIter<'i>;

    #[inline]
    fn into_iter(self) -> VCDHierRevIter<'i> {
        VCDHierRevIter(Some(self))
    }
}

impl Hash for VCDHier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for s in self.iter() {
            s.hash(state);
        }
    }
}

#[allow(dead_code)]
impl VCDHier {
    #[inline]
    pub fn single(cur: CompactString) -> Self {
        VCDHier { cur, prev: None }
    }

    #[inline]
    pub fn empty() -> Self {
        VCDHier {
            cur: "".into(),
            prev: None,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cur.as_str() == "" && self.prev.is_none()
    }

    #[inline]
    pub fn iter(&self) -> VCDHierRevIter<'_> {
        (&self).into_iter()
    }
}

// ── Scope matching utilities ────────────────────────────────────────────────

/// Try to match one component in a scope path.
/// Returns the remaining scope on success, or None on failure.
pub fn match_scope_path<'i>(mut scope: &'i str, cur: &str) -> Option<&'i str> {
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

/// Find a scope by its path in the VCD hierarchy.
pub fn find_top_scope<'i>(items: &'i [ScopeItem], top_scope: &'_ str) -> Option<&'i Scope> {
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

/// Recursively collect all scope paths from VCD header.
pub fn collect_all_scopes<'a>(
    items: &'a [ScopeItem],
    prefix: &str,
    scopes: &mut Vec<(String, &'a Scope)>,
) {
    for item in items {
        if let ScopeItem::Scope(scope) = item {
            let scope_path = if prefix.is_empty() {
                scope.identifier.to_string()
            } else {
                format!("{}/{}", prefix, scope.identifier)
            };
            scopes.push((scope_path.clone(), scope));
            collect_all_scopes(&scope.children[..], &scope_path, scopes);
        }
    }
}

/// Get required input port names from netlistdb.
pub fn get_required_input_ports(netlistdb: &NetlistDB) -> HashSet<String> {
    let mut ports = HashSet::new();
    for i in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[i] != Direction::I {
            let port_name = netlistdb.pinnames[i].1.to_string();
            ports.insert(port_name);
        }
    }
    ports
}

/// Check if a VCD scope contains all required ports.
pub fn check_scope_contains_ports(scope: &Scope, required_ports: &HashSet<String>) -> bool {
    let mut found_ports = HashSet::new();
    for item in &scope.children {
        if let ScopeItem::Var(var) = item {
            found_ports.insert(var.reference.to_string());
        }
    }

    for port in required_ports {
        if !found_ports.contains(port) {
            return false;
        }
    }
    true
}

/// Auto-detect VCD scope containing the DUT.
pub fn auto_detect_vcd_scope<'i>(
    items: &'i [ScopeItem],
    netlistdb: &NetlistDB,
    top_module_name: &str,
) -> Option<(String, &'i Scope)> {
    let required_ports = get_required_input_ports(netlistdb);

    if required_ports.is_empty() {
        clilog::warn!("No input ports found in design - cannot auto-detect VCD scope");
        return None;
    }

    let mut all_scopes = Vec::new();
    collect_all_scopes(items, "", &mut all_scopes);

    if all_scopes.is_empty() {
        clilog::warn!("No scopes found in VCD file");
        return None;
    }

    clilog::debug!(
        "Searching for VCD scope containing {} input ports",
        required_ports.len()
    );
    clilog::debug!("Required ports: {:?}", required_ports);

    // Try common DUT scope names first
    let common_names = ["dut", "uut", "DUT", "UUT", top_module_name];
    for name in &common_names {
        for (path, scope) in &all_scopes {
            if path.ends_with(name) && check_scope_contains_ports(scope, &required_ports) {
                clilog::info!(
                    "Auto-detected VCD scope: {} (matched common pattern '{}')",
                    path,
                    name
                );
                return Some((path.clone(), *scope));
            }
        }
    }

    // Try any scope that contains all required ports
    for (path, scope) in &all_scopes {
        if check_scope_contains_ports(scope, &required_ports) {
            clilog::info!(
                "Auto-detected VCD scope: {} (contains all required ports)",
                path
            );
            return Some((path.clone(), *scope));
        }
    }

    clilog::error!("Could not auto-detect VCD scope. Available scopes:");
    for (path, _) in &all_scopes {
        clilog::error!("  - {}", path);
    }
    clilog::error!("Please specify scope manually with --input-vcd-scope");
    None
}

// ── VCD input parsing ───────────────────────────────────────────────────────

/// Result of parsing input VCD: the state vectors and cycle timestamps.
pub struct ParsedInputVCD {
    /// Flattened input state vectors, one per cycle + trailing state.
    pub input_states: Vec<u32>,
    /// (offset_into_input_states, vcd_timestamp) per cycle.
    pub offsets_timestamps: Vec<(usize, u64)>,
}

/// Resolve the top scope from VCD header, either from user-specified path or auto-detection.
pub fn resolve_vcd_scope<'i>(
    header_items: &'i [ScopeItem],
    input_vcd_scope: Option<&str>,
    netlistdb: &NetlistDB,
    top_module: Option<&str>,
) -> &'i Scope {
    if let Some(scope_path) = input_vcd_scope {
        clilog::info!("Using user-specified VCD scope: {}", scope_path);
        find_top_scope(header_items, scope_path).expect("Specified top scope not found in VCD.")
    } else {
        let top_module_name = top_module.unwrap_or("top");
        clilog::info!("No VCD scope specified - attempting auto-detection");

        match auto_detect_vcd_scope(header_items, netlistdb, top_module_name) {
            Some((_path, scope)) => scope,
            None => {
                panic!(
                    "Failed to auto-detect VCD scope. Please specify --input-vcd-scope manually."
                );
            }
        }
    }
}

/// Match VCD variables to netlist input ports.
///
/// Returns:
/// - `vcd2inp`: maps (vcd_code, bit_position) → netlist pin ID
/// - `inp_port_given`: set of netlist pin IDs that were found in VCD
pub fn match_vcd_inputs(
    top_scope: &Scope,
    netlistdb: &NetlistDB,
) -> (HashMap<(u64, usize), usize>, HashSet<usize>) {
    use sverilogparse::SVerilogRange;
    use vcd_ng::ReferenceIndex::*;

    let mut vcd2inp = HashMap::new();
    let mut inp_port_given = HashSet::new();

    let mut match_one_input = |var: &Var, i: Option<isize>, vcd_pos: usize| {
        let key = (VCDHier::empty(), var.reference.as_str(), i);
        if let Some(&id) = netlistdb.pinname2id.get(&key as &dyn GeneralPinName) {
            if netlistdb.pindirect[id] != Direction::O {
                return;
            }
            vcd2inp.insert((var.code.0, vcd_pos), id);
            inp_port_given.insert(id);
        }
    };

    for scope_item in &top_scope.children[..] {
        if let ScopeItem::Var(var) = scope_item {
            match var.index {
                None => match var.size {
                    1 => match_one_input(var, None, 0),
                    w => {
                        for (pos, i) in (0..w).rev().enumerate() {
                            match_one_input(var, Some(i as isize), pos)
                        }
                    }
                },
                Some(BitSelect(i)) => match_one_input(var, Some(i as isize), 0),
                Some(Range(a, b)) => {
                    for (pos, i) in SVerilogRange(a as isize, b as isize).enumerate() {
                        match_one_input(var, Some(i), pos);
                    }
                }
            }
        }
    }

    // Warn about missing primary inputs
    for i in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[i] != Direction::I && !inp_port_given.contains(&i) {
            clilog::warn!(
                GATESIM_VCDI_MISSING_PI,
                "Primary input port {:?} not present in the VCD input",
                netlistdb.pinnames[i]
            );
        }
    }

    (vcd2inp, inp_port_given)
}

/// Parse input VCD flow into state vectors for simulation.
pub fn parse_input_vcd(
    vcdflow: &mut vcd_ng::FastFlow<std::fs::File>,
    vcd2inp: &HashMap<(u64, usize), usize>,
    aig: &AIG,
    script: &FlattenedScriptV1,
    netlistdb: &NetlistDB,
    max_cycles: Option<usize>,
) -> ParsedInputVCD {
    use vcd_ng::{FFValueChange, FastFlowToken};

    let mut state = vec![0; script.reg_io_state_size as usize];
    let mut vcd_time_last_active = u64::MAX;
    let mut vcd_time = 0;
    let mut last_vcd_time_active = true;
    let mut delayed_bit_changes = HashSet::new();

    let mut input_states = Vec::new();
    let mut offsets_timestamps = Vec::new();

    while let Some(tok) = vcdflow.next_token().unwrap() {
        match tok {
            FastFlowToken::Timestamp(t) => {
                if t == vcd_time {
                    continue;
                }
                if last_vcd_time_active {
                    input_states.extend(state.iter().copied());
                    offsets_timestamps.push((input_states.len(), vcd_time_last_active));
                    for (_, &(pe, ne)) in &aig.clock_pin2aigpins {
                        if pe != usize::MAX {
                            let p = *script.input_map.get(&pe).unwrap();
                            state[p as usize >> 5] &= !(1 << (p & 31));
                        }
                        if ne != usize::MAX {
                            let p = *script.input_map.get(&ne).unwrap();
                            state[p as usize >> 5] &= !(1 << (p & 31));
                        }
                    }
                    if let Some(max_cycles) = max_cycles {
                        if offsets_timestamps.len() >= max_cycles {
                            clilog::info!("reached maximum cycles, stop reading input vcd");
                            break;
                        }
                    }
                }
                if last_vcd_time_active {
                    vcd_time_last_active = vcd_time;
                }
                vcd_time = t;
                last_vcd_time_active = false;

                for pos in std::mem::take(&mut delayed_bit_changes) {
                    state[(pos >> 5) as usize] ^= 1u32 << (pos & 31);
                }
            }
            FastFlowToken::Value(FFValueChange { id, bits }) => {
                for (pos, b) in bits.iter().enumerate() {
                    if let Some(&pin) = vcd2inp.get(&(id.0, pos)) {
                        let aigpin = aig.pin2aigpin_iv[pin];
                        assert_eq!(aigpin & 1, 0);
                        let aigpin = aigpin >> 1;
                        let pos = match script.input_map.get(&aigpin).copied() {
                            Some(pos) => pos,
                            None => {
                                panic!(
                                    "input pin {:?} (netlist id {}, aigpin {}) not found in output map.",
                                    netlistdb.pinnames[pin].dbg_fmt_pin(),
                                    pin,
                                    aigpin
                                );
                            }
                        };
                        let old_value = state[(pos >> 5) as usize] >> (pos & 31) & 1;
                        if old_value
                            == match b {
                                b'1' => 1,
                                _ => 0,
                            }
                        {
                            continue;
                        }
                        if let Some((pe, ne)) = aig.clock_pin2aigpins.get(&pin).copied() {
                            if pe != usize::MAX && old_value == 0 {
                                last_vcd_time_active = true;
                                let p = *script.input_map.get(&pe).unwrap();
                                state[p as usize >> 5] |= 1 << (p & 31);
                            }
                            if ne != usize::MAX && old_value == 1 {
                                last_vcd_time_active = true;
                                let p = *script.input_map.get(&ne).unwrap();
                                state[p as usize >> 5] |= 1 << (p & 31);
                            }
                            delayed_bit_changes.insert(pos);
                        } else {
                            state[(pos >> 5) as usize] ^= 1u32 << (pos & 31);
                        }
                    }
                }
            }
        }
    }
    input_states.extend(state.iter().copied());
    clilog::info!("total number of cycles: {}", offsets_timestamps.len());

    ParsedInputVCD {
        input_states,
        offsets_timestamps,
    }
}

// ── VCD output writing ──────────────────────────────────────────────────────

/// Information needed to write output VCD.
pub struct OutputVCDMapping {
    /// (aigpin, output_pos_in_state, vcd_variable_id)
    pub out2vcd: Vec<(usize, u32, vcd_ng::IdCode)>,
}

/// Set up output VCD writer: add wire definitions and build output mapping.
pub fn setup_output_vcd(
    writer: &mut vcd_ng::Writer<std::io::BufWriter<std::fs::File>>,
    header: &vcd_ng::Header,
    output_vcd_scope: Option<&str>,
    netlistdb: &NetlistDB,
    aig: &AIG,
    script: &FlattenedScriptV1,
) -> OutputVCDMapping {
    use vcd_ng::SimulationCommand;

    if let Some((ratio, unit)) = header.timescale {
        writer.timescale(ratio, unit).unwrap();
    }
    let output_vcd_scope_str = output_vcd_scope.unwrap_or("gem_top_module");
    let scope_parts = output_vcd_scope_str.split('/').collect::<Vec<_>>();
    for &scope in &scope_parts {
        writer.add_module(scope).unwrap();
    }

    let out2vcd = netlistdb
        .cell2pin
        .iter_set(0)
        .filter_map(|i| {
            if netlistdb.pindirect[i] == Direction::I {
                let aigpin = aig.pin2aigpin_iv[i];
                if matches!(aig.drivers[aigpin >> 1], DriverType::InputPort(_)) {
                    clilog::info!(
                        "skipped output for port {} as it is a pass-through of input port.",
                        netlistdb.pinnames[i].dbg_fmt_pin()
                    );
                    return None;
                }
                if aigpin <= 1 {
                    return Some((
                        aigpin,
                        u32::MAX,
                        writer
                            .add_wire(1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin()))
                            .unwrap(),
                    ));
                }
                Some((
                    aigpin,
                    *script.output_map.get(&aigpin).unwrap(),
                    writer
                        .add_wire(1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin()))
                        .unwrap(),
                ))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    for _ in 0..scope_parts.len() {
        writer.upscope().unwrap();
    }
    writer.enddefinitions().unwrap();
    writer.begin(SimulationCommand::Dumpvars).unwrap();

    OutputVCDMapping { out2vcd }
}

/// Write simulation results to output VCD.
pub fn write_output_vcd(
    writer: &mut vcd_ng::Writer<std::io::BufWriter<std::fs::File>>,
    output_mapping: &OutputVCDMapping,
    offsets_timestamps: &[(usize, u64)],
    states: &[u32],
) {
    use vcd_ng::Value;

    clilog::info!("write out vcd");
    let mut last_val = vec![2u32; output_mapping.out2vcd.len()];
    for &(offset, timestamp) in offsets_timestamps {
        if timestamp == u64::MAX {
            continue;
        }
        writer.timestamp(timestamp).unwrap();
        for (i, &(output_aigpin, output_pos, vid)) in output_mapping.out2vcd.iter().enumerate() {
            let value_new = match output_pos {
                u32::MAX => {
                    assert!(output_aigpin <= 1);
                    output_aigpin as u32
                }
                output_pos => states[offset + (output_pos >> 5) as usize] >> (output_pos & 31) & 1,
            };
            if value_new == last_val[i] {
                continue;
            }
            last_val[i] = value_new;
            writer
                .change_scalar(
                    vid,
                    match value_new {
                        1 => Value::V1,
                        _ => Value::V0,
                    },
                )
                .unwrap();
        }
    }
}
