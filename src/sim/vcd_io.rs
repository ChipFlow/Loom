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

// ── X-propagation state buffer helpers ───────────────────────────────────────

/// Expand value-only state buffer into a doubled buffer with X-mask for GPU dispatch.
///
/// The input `value_states` has `(num_cycles + 1)` snapshots of `reg_io_state_size` words.
/// The output has `(num_cycles + 1)` snapshots of `2 * reg_io_state_size` words:
/// `[values | xmask]` where X-mask is 0xFFFFFFFF for DFF output positions (unknown)
/// and 0 for primary input positions (known from VCD).
pub fn expand_states_for_xprop(
    value_states: &[u32],
    script: &FlattenedScriptV1,
) -> Vec<u32> {
    let rio = script.reg_io_state_size as usize;
    let num_snapshots = value_states.len() / rio;
    assert_eq!(value_states.len(), num_snapshots * rio);

    // Build X-mask template: 0xFFFFFFFF for DFF positions, 0 for inputs
    let mut xmask_template = vec![0xFFFF_FFFFu32; rio];
    for &pos in script.input_map.values() {
        xmask_template[(pos >> 5) as usize] &= !(1u32 << (pos & 31));
    }

    let mut expanded = Vec::with_capacity(num_snapshots * rio * 2);
    for snap_i in 0..num_snapshots {
        // Value portion
        expanded.extend_from_slice(&value_states[snap_i * rio..(snap_i + 1) * rio]);
        // X-mask portion
        expanded.extend_from_slice(&xmask_template);
    }
    expanded
}

/// Extract value-only states from a doubled GPU output buffer.
///
/// Input has `(num_cycles + 1)` snapshots of `2 * reg_io_state_size` words.
/// Returns separate value and X-mask arrays, each with the same layout as the
/// original value-only state buffer (indexed by `offsets_timestamps` offsets).
pub fn split_xprop_states(
    gpu_states: &[u32],
    reg_io_state_size: usize,
) -> (Vec<u32>, Vec<u32>) {
    let eff = reg_io_state_size * 2;
    let num_snapshots = gpu_states.len() / eff;
    assert_eq!(gpu_states.len(), num_snapshots * eff);

    let mut values = Vec::with_capacity(num_snapshots * reg_io_state_size);
    let mut xmasks = Vec::with_capacity(num_snapshots * reg_io_state_size);
    for snap_i in 0..num_snapshots {
        let base = snap_i * eff;
        values.extend_from_slice(&gpu_states[base..base + reg_io_state_size]);
        xmasks.extend_from_slice(&gpu_states[base + reg_io_state_size..base + eff]);
    }
    (values, xmasks)
}

/// Expand state buffer to include an arrival section for timing VCD output.
///
/// Appends `reg_io_state_size` words of zeros after each cycle's existing
/// state data (values + optional xmask). This reserves space for the GPU
/// kernel to write per-output arrival times.
pub fn expand_states_for_arrivals(
    states: &[u32],
    script: &FlattenedScriptV1,
) -> Vec<u32> {
    let old_eff = if script.xprop_enabled {
        script.reg_io_state_size as usize * 2
    } else {
        script.reg_io_state_size as usize
    };
    let rio = script.reg_io_state_size as usize;
    let new_eff = old_eff + rio;
    let num_snapshots = states.len() / old_eff;
    assert_eq!(states.len(), num_snapshots * old_eff);

    let mut expanded = Vec::with_capacity(num_snapshots * new_eff);
    for snap_i in 0..num_snapshots {
        // Copy existing data (values + optional xmask)
        expanded.extend_from_slice(&states[snap_i * old_eff..(snap_i + 1) * old_eff]);
        // Append zeroed arrival section
        expanded.extend(std::iter::repeat(0u32).take(rio));
    }
    expanded
}

/// Extract arrival times from GPU output buffer with timing arrivals enabled.
///
/// Returns a vector of u32 arrival times with the same snapshot/offset
/// structure as the value states. Each u32 contains a u16 arrival time
/// in picoseconds (stored in the low 16 bits).
pub fn split_arrival_states(
    gpu_states: &[u32],
    script: &FlattenedScriptV1,
) -> Vec<u32> {
    let eff = script.effective_state_size() as usize;
    let rio = script.reg_io_state_size as usize;
    let arrival_offset = script.arrival_state_offset as usize;
    let num_snapshots = gpu_states.len() / eff;
    assert_eq!(gpu_states.len(), num_snapshots * eff);

    let mut arrivals = Vec::with_capacity(num_snapshots * rio);
    for snap_i in 0..num_snapshots {
        let base = snap_i * eff + arrival_offset;
        arrivals.extend_from_slice(&gpu_states[base..base + rio]);
    }
    arrivals
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
    write_output_vcd_impl(writer, output_mapping, offsets_timestamps, states, None)
}

/// Write simulation results to output VCD with X-propagation support.
///
/// When `xmask_states` is provided, bits with X-mask=1 are emitted as `Value::X`
/// in the output VCD.
pub fn write_output_vcd_xprop(
    writer: &mut vcd_ng::Writer<std::io::BufWriter<std::fs::File>>,
    output_mapping: &OutputVCDMapping,
    offsets_timestamps: &[(usize, u64)],
    states: &[u32],
    xmask_states: &[u32],
) {
    write_output_vcd_impl(writer, output_mapping, offsets_timestamps, states, Some(xmask_states))
}

fn write_output_vcd_impl(
    writer: &mut vcd_ng::Writer<std::io::BufWriter<std::fs::File>>,
    output_mapping: &OutputVCDMapping,
    offsets_timestamps: &[(usize, u64)],
    states: &[u32],
    xmask_states: Option<&[u32]>,
) {
    use vcd_ng::Value;

    clilog::info!("write out vcd");
    // Use 3 to represent X in the last_val tracker (0=V0, 1=V1, 2=initial, 3=X)
    let mut last_val = vec![2u32; output_mapping.out2vcd.len()];
    let mut x_output_count = 0u64;
    for &(offset, timestamp) in offsets_timestamps {
        if timestamp == u64::MAX {
            continue;
        }
        writer.timestamp(timestamp).unwrap();
        for (i, &(output_aigpin, output_pos, vid)) in output_mapping.out2vcd.iter().enumerate() {
            let (value_new, is_x) = match output_pos {
                u32::MAX => {
                    assert!(output_aigpin <= 1);
                    (output_aigpin as u32, false)
                }
                output_pos => {
                    let v = states[offset + (output_pos >> 5) as usize] >> (output_pos & 31) & 1;
                    let x = xmask_states
                        .map(|xm| xm[offset + (output_pos >> 5) as usize] >> (output_pos & 31) & 1 != 0)
                        .unwrap_or(false);
                    (v, x)
                }
            };

            // Encode: 0=V0, 1=V1, 3=X
            let encoded = if is_x { 3 } else { value_new };
            if encoded == last_val[i] {
                continue;
            }
            last_val[i] = encoded;
            if is_x {
                x_output_count += 1;
            }
            writer
                .change_scalar(
                    vid,
                    match (is_x, value_new) {
                        (true, _) => Value::X,
                        (false, 1) => Value::V1,
                        _ => Value::V0,
                    },
                )
                .unwrap();
        }
    }
    if x_output_count > 0 {
        clilog::warn!(
            "VCD output contains {} X-value transitions across all signals",
            x_output_count
        );
    }
}

/// Write simulation results to output VCD with timing-accurate timestamps.
///
/// Instead of placing all output transitions at clock edges, each signal's
/// transition is offset by its arrival time (in picoseconds) from the GPU
/// kernel's arrival sideband.
///
/// `arrival_states` has the same snapshot/offset layout as `states`.
/// Each u32 word contains a u16 arrival time in picoseconds (low 16 bits).
/// `timescale` is the VCD timescale (ratio, unit) for converting ps to VCD time.
pub fn write_output_vcd_timed(
    writer: &mut vcd_ng::Writer<std::io::BufWriter<std::fs::File>>,
    output_mapping: &OutputVCDMapping,
    offsets_timestamps: &[(usize, u64)],
    states: &[u32],
    xmask_states: Option<&[u32]>,
    arrival_states: &[u32],
    timescale: Option<(u32, vcd_ng::TimescaleUnit)>,
) {
    use vcd_ng::Value;

    clilog::info!("write out timed vcd");

    // Compute how many picoseconds each VCD time unit represents.
    // E.g., timescale=(1, PS) → 1 ps/unit, (1, NS) → 1000 ps/unit.
    let ps_per_vcd_unit: f64 = match timescale {
        Some((ratio, unit)) => {
            // divisor: units-per-second. PS=1e12, NS=1e9.
            // ps_per_vcd_unit = ratio * (PS_divisor / unit_divisor)
            let ps_divisor = vcd_ng::TimescaleUnit::PS.divisor() as f64;
            let unit_divisor = unit.divisor() as f64;
            (ratio as f64) * (ps_divisor / unit_divisor)
        }
        None => 1.0, // Assume 1ps if no timescale
    };

    // Collect transitions: (actual_timestamp, output_index, value, is_x)
    let mut last_val = vec![2u32; output_mapping.out2vcd.len()]; // 2 = initial
    let mut timed_transitions: Vec<(u64, usize, u32, bool)> = Vec::new();
    let mut x_output_count = 0u64;

    for &(offset, timestamp) in offsets_timestamps {
        if timestamp == u64::MAX {
            continue;
        }

        for (i, &(output_aigpin, output_pos, _vid)) in output_mapping.out2vcd.iter().enumerate() {
            let (value_new, is_x) = match output_pos {
                u32::MAX => {
                    assert!(output_aigpin <= 1);
                    (output_aigpin as u32, false)
                }
                output_pos => {
                    let v = states[offset + (output_pos >> 5) as usize] >> (output_pos & 31) & 1;
                    let x = xmask_states
                        .map(|xm| xm[offset + (output_pos >> 5) as usize] >> (output_pos & 31) & 1 != 0)
                        .unwrap_or(false);
                    (v, x)
                }
            };

            let encoded = if is_x { 3 } else { value_new };
            if encoded == last_val[i] {
                continue;
            }
            last_val[i] = encoded;
            if is_x {
                x_output_count += 1;
            }

            // Get arrival time for this output position.
            // Arrivals are stored one u32 per word position (same indexing as values).
            // Each u32 word in the arrival section holds a u16 arrival in the low bits,
            // shared by all 32 output bits packed in the corresponding value word.
            let arrival_ps = if output_pos != u32::MAX {
                let word_idx = (output_pos >> 5) as usize;
                (arrival_states[offset + word_idx] & 0xFFFF) as u64
            } else {
                0u64
            };

            // Convert arrival_ps to VCD time units and add to base timestamp
            let arrival_vcd_units = if ps_per_vcd_unit > 0.0 {
                (arrival_ps as f64 / ps_per_vcd_unit).round() as u64
            } else {
                0
            };
            let actual_timestamp = timestamp + arrival_vcd_units;

            timed_transitions.push((actual_timestamp, i, value_new, is_x));
        }
    }

    // Sort by timestamp (stable sort preserves signal order within same timestamp)
    timed_transitions.sort_by_key(|&(ts, _, _, _)| ts);

    // Write sorted transitions
    let mut current_timestamp = u64::MAX;
    for &(ts, i, value_new, is_x) in &timed_transitions {
        if ts != current_timestamp {
            writer.timestamp(ts).unwrap();
            current_timestamp = ts;
        }
        let (_, _, vid) = output_mapping.out2vcd[i];
        writer
            .change_scalar(
                vid,
                match (is_x, value_new) {
                    (true, _) => Value::X,
                    (false, 1) => Value::V1,
                    _ => Value::V0,
                },
            )
            .unwrap();
    }

    clilog::info!(
        "Timed VCD: {} transitions written",
        timed_transitions.len()
    );
    if x_output_count > 0 {
        clilog::warn!(
            "VCD output contains {} X-value transitions across all signals",
            x_output_count
        );
    }
}

#[cfg(test)]
mod xprop_tests {
    use super::*;
    use indexmap::IndexMap;

    /// Build a minimal FlattenedScriptV1 for xprop buffer tests.
    ///
    /// `input_positions`: bit positions in the state that are primary inputs
    /// `output_positions`: bit positions in the state that are DFF outputs
    /// `rio`: reg_io_state_size in u32 words
    fn make_xprop_test_script(
        rio: u32,
        input_positions: &[u32],
        output_positions: &[u32],
    ) -> FlattenedScriptV1 {
        let mut input_map = IndexMap::new();
        for (i, &pos) in input_positions.iter().enumerate() {
            // Use distinct AIG pin IDs (even numbers for non-inverted)
            input_map.insert(i * 2 + 100, pos);
        }
        let mut output_map = IndexMap::new();
        for (i, &pos) in output_positions.iter().enumerate() {
            output_map.insert(i * 2 + 200, pos);
        }
        FlattenedScriptV1 {
            num_blocks: 0,
            num_major_stages: 0,
            blocks_start: Vec::<usize>::new().into(),
            blocks_data: Vec::<u32>::new().into(),
            reg_io_state_size: rio,
            sram_storage_size: 0,
            input_layout: Vec::new(),
            output_map,
            input_map,
            stages_blocks_parts: Vec::new(),
            assertion_positions: Vec::new(),
            display_positions: Vec::new(),
            gate_delays: Vec::new(),
            dff_constraints: Vec::new(),
            clock_period_ps: 1000,
            timing_enabled: false,
            delay_patch_map: Vec::new(),
            xprop_enabled: true,
            partition_x_capable: Vec::new(),
            xprop_state_offset: rio,
            timing_arrivals_enabled: false,
            arrival_state_offset: 0,
        }
    }

    #[test]
    fn test_expand_split_roundtrip() {
        // 2-word state, 1 input at bit 0, 1 DFF output at bit 32
        let rio = 2;
        let script = make_xprop_test_script(rio, &[0], &[32]);

        // 3 snapshots (2 cycles + trailing), value states only
        let value_states: Vec<u32> = vec![
            0x0000_0001, 0x0000_0000, // snap 0: input bit 0 = 1
            0x0000_0001, 0x0000_0001, // snap 1: input bit 0 = 1, dff bit 0 = 1
            0x0000_0000, 0x0000_0001, // snap 2: input bit 0 = 0, dff bit 0 = 1
        ];

        let expanded = expand_states_for_xprop(&value_states, &script);
        assert_eq!(expanded.len(), 3 * rio as usize * 2);

        // Split back
        let (values, xmasks) = split_xprop_states(&expanded, rio as usize);
        assert_eq!(values.len(), value_states.len());
        assert_eq!(values, value_states, "values should match original");

        // X-mask template: bit 0 = input → cleared (0), bit 32 = DFF → set (1)
        // Word 0 of xmask: bit 0 cleared → 0xFFFF_FFFE
        // Word 1 of xmask: all bits set → 0xFFFF_FFFF (DFF output at bit 0 of word 1)
        for snap in 0..3 {
            let xm_word0 = xmasks[snap * rio as usize];
            let xm_word1 = xmasks[snap * rio as usize + 1];
            assert_eq!(
                xm_word0, 0xFFFF_FFFE,
                "snap {}: word 0 should have bit 0 cleared (input)",
                snap
            );
            assert_eq!(
                xm_word1, 0xFFFF_FFFF,
                "snap {}: word 1 should be all-X (DFF output)",
                snap
            );
        }
    }

    #[test]
    fn test_split_preserves_cycle_structure() {
        let rio = 3usize;
        let num_snaps = 4usize;
        let eff = rio * 2;

        // Build a doubled buffer manually with known patterns
        let mut gpu_states = Vec::with_capacity(num_snaps * eff);
        for snap in 0..num_snaps {
            // Value section: fill with snap index
            for w in 0..rio {
                gpu_states.push((snap * 100 + w) as u32);
            }
            // Xmask section: fill with snap index + 1000
            for w in 0..rio {
                gpu_states.push((snap * 100 + w + 1000) as u32);
            }
        }

        let (values, xmasks) = split_xprop_states(&gpu_states, rio);
        assert_eq!(values.len(), num_snaps * rio);
        assert_eq!(xmasks.len(), num_snaps * rio);

        for snap in 0..num_snaps {
            for w in 0..rio {
                assert_eq!(
                    values[snap * rio + w],
                    (snap * 100 + w) as u32,
                    "snap {} word {} value mismatch",
                    snap,
                    w
                );
                assert_eq!(
                    xmasks[snap * rio + w],
                    (snap * 100 + w + 1000) as u32,
                    "snap {} word {} xmask mismatch",
                    snap,
                    w
                );
            }
        }
    }

    #[test]
    fn test_expand_xmask_template() {
        // 4-word state with various input/output positions
        let rio = 4;
        // Inputs at bits 0, 5, 64 (word 0 bits 0,5; word 2 bit 0)
        // DFF outputs at bits 32, 33, 96 (word 1 bits 0,1; word 3 bit 0)
        let script = make_xprop_test_script(rio, &[0, 5, 64], &[32, 33, 96]);

        let value_states = vec![0u32; rio as usize]; // 1 snapshot, all zeros
        let expanded = expand_states_for_xprop(&value_states, &script);

        assert_eq!(expanded.len(), 2 * rio as usize);

        // Check xmask template (second half of expanded)
        let xmask = &expanded[rio as usize..];

        // Word 0: bits 0 and 5 are inputs → should be cleared
        let expected_word0 = 0xFFFF_FFFF & !(1u32 << 0) & !(1u32 << 5);
        assert_eq!(
            xmask[0], expected_word0,
            "word 0: input bits 0,5 should be cleared"
        );

        // Word 1: no inputs → all 0xFFFFFFFF
        assert_eq!(xmask[1], 0xFFFF_FFFF, "word 1: no inputs, all X");

        // Word 2: bit 0 is input → cleared
        let expected_word2 = 0xFFFF_FFFF & !(1u32 << 0);
        assert_eq!(
            xmask[2], expected_word2,
            "word 2: input bit 0 should be cleared"
        );

        // Word 3: no inputs → all 0xFFFFFFFF
        assert_eq!(xmask[3], 0xFFFF_FFFF, "word 3: no inputs, all X");
    }
}
