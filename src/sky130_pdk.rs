// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! SKY130 PDK behavioral model parser and AIG decomposition.
//!
//! Parses gate-level functional Verilog models from the official SKY130 PDK
//! (google/skywater-pdk-libs-sky130_fd_sc_hd) and converts them to AIG
//! decompositions. This replaces hand-coded decompositions with vendor-verified
//! reference models.
//!
//! The functional models use only Verilog gate primitives:
//!   and, or, nand, nor, not, xor, xnor, buf
//! Some cells (mux2, mux2i, mux4) use UDP truth tables instead.

use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Core types (formerly in sky130_decomp.rs)
// ============================================================================

/// Fixed-size struct for collecting input pins during SKY130 decomposition.
/// Most SKY130 cells have at most 5 inputs. Using a fixed struct avoids heap allocation.
#[derive(Default, Clone, Copy, Debug)]
pub struct CellInputs {
    pub a: usize,
    pub a_n: usize,
    pub a0: usize,
    pub a1: usize,
    pub a1_n: usize,
    pub a2: usize,
    pub a2_n: usize,
    pub a3: usize,
    pub a4: usize,
    pub b: usize,
    pub b_n: usize,
    pub b1: usize,
    pub b1_n: usize,
    pub b2: usize,
    pub c: usize,
    pub c_n: usize,
    pub c1: usize,
    pub d: usize,
    pub d1: usize,
    pub s: usize,
    pub s0: usize,
    pub cin: usize,
    pub set_b: usize,
    pub reset_b: usize,
}

impl CellInputs {
    /// Create a new CellInputs with all pins set to MAX (unset).
    #[inline]
    pub fn new() -> Self {
        Self {
            a: usize::MAX,
            a_n: usize::MAX,
            a0: usize::MAX,
            a1: usize::MAX,
            a1_n: usize::MAX,
            a2: usize::MAX,
            a2_n: usize::MAX,
            a3: usize::MAX,
            a4: usize::MAX,
            b: usize::MAX,
            b_n: usize::MAX,
            b1: usize::MAX,
            b1_n: usize::MAX,
            b2: usize::MAX,
            c: usize::MAX,
            c_n: usize::MAX,
            c1: usize::MAX,
            d: usize::MAX,
            d1: usize::MAX,
            s: usize::MAX,
            s0: usize::MAX,
            cin: usize::MAX,
            set_b: usize::MAX,
            reset_b: usize::MAX,
        }
    }

    /// Set a pin value by name. Returns true if the pin was recognized.
    #[inline]
    pub fn set_pin(&mut self, pin_name: &str, value: usize) -> bool {
        match pin_name {
            "A" => self.a = value,
            "A_N" => self.a_n = value,
            "A0" => self.a0 = value,
            "A1" => self.a1 = value,
            "A1_N" => self.a1_n = value,
            "A2" => self.a2 = value,
            "A2_N" => self.a2_n = value,
            "A3" => self.a3 = value,
            "A4" => self.a4 = value,
            "B" => self.b = value,
            "B_N" => self.b_n = value,
            "B1" => self.b1 = value,
            "B1_N" => self.b1_n = value,
            "B2" => self.b2 = value,
            "C" => self.c = value,
            "C_N" => self.c_n = value,
            "C1" => self.c1 = value,
            "D" => self.d = value,
            "D1" => self.d1 = value,
            "S" => self.s = value,
            "S0" => self.s0 = value,
            "CIN" => self.cin = value,
            "SET_B" => self.set_b = value,
            "RESET_B" => self.reset_b = value,
            _ => return false,
        }
        true
    }
}

/// Result of decomposing a cell into AIG operations.
///
/// The decomposition produces a sequence of AND gates that must be built
/// in order, where later gates can reference earlier ones.
#[derive(Debug, Clone)]
pub struct DecompResult {
    /// Sequence of AND gate operations to build.
    /// Each entry is (input_a_iv, input_b_iv) where the lower bit is inversion.
    /// References to earlier gates use negative indices (-1 = first gate output, etc.)
    pub and_gates: Vec<(i64, i64)>,
    /// Index of the final output (-1 = first gate, -2 = second gate, etc.)
    /// Positive values reference original inputs.
    pub output_idx: i64,
    /// Whether to invert the final output
    pub output_inverted: bool,
}

/// Check if a cell type is a sequential element (DFF or latch).
pub fn is_sequential_cell(cell_type: &str) -> bool {
    matches!(
        cell_type,
        "dfxtp" | "dfrtp" | "dfrbp" | "dfstp" | "dfbbp" | "edfxtp" | "sdfxtp" | "dlxtp" | "dlat"
    )
}

/// Check if a cell is a tie cell (constant generator).
pub fn is_tie_cell(cell_type: &str) -> bool {
    cell_type == "conb"
}

/// Check if a cell has multiple outputs (like adders).
pub fn is_multi_output_cell(cell_type: &str) -> bool {
    matches!(cell_type, "ha" | "fa" | "dfbbp")
}

// ============================================================================
// Data structures
// ============================================================================

/// A single gate instantiation from a functional Verilog model.
#[derive(Debug, Clone)]
pub struct BehavioralGate {
    /// Gate type: "and", "or", "nand", "nor", "not", "xor", "xnor", "buf",
    /// or a UDP name like "sky130_fd_sc_hd__udp_mux_2to1_N"
    pub gate_type: String,
    /// Output wire name
    pub output: String,
    /// Input wire names (in port order)
    pub inputs: Vec<String>,
}

/// A parsed functional Verilog model for a cell.
#[derive(Debug, Clone)]
pub struct BehavioralModel {
    /// Module name (e.g. "sky130_fd_sc_hd__o21ai")
    pub module_name: String,
    /// Module input port names (e.g. ["A1", "A2", "B1"])
    pub inputs: Vec<String>,
    /// Module output port names (e.g. ["Y"])
    pub outputs: Vec<String>,
    /// Gate instantiations in order
    pub gates: Vec<BehavioralGate>,
}

/// A single row in a UDP truth table.
#[derive(Debug, Clone)]
pub struct UdpRow {
    /// Input values: Some(true)=1, Some(false)=0, None=don't-care (?)
    pub inputs: Vec<Option<bool>>,
    /// Output value
    pub output: bool,
}

/// A parsed Verilog UDP (User Defined Primitive).
#[derive(Debug, Clone)]
pub struct UdpModel {
    /// Primitive name
    pub name: String,
    /// Output port name
    pub output: String,
    /// Input port names in order
    pub inputs: Vec<String>,
    /// Truth table rows
    pub rows: Vec<UdpRow>,
}

/// Collection of loaded PDK models for a cell library.
pub struct PdkModels {
    /// Behavioral models indexed by cell type (e.g. "o21ai" -> model)
    pub models: HashMap<String, BehavioralModel>,
    /// UDP models indexed by primitive name
    pub udps: HashMap<String, UdpModel>,
}

// ============================================================================
// Parser: Functional Verilog models
// ============================================================================

/// Parse a functional Verilog model file (*.functional.v).
///
/// Handles:
/// - module declaration with port names and directions
/// - Gate instantiations: `gate_type name (output, input1, input2, ...);`
/// - Skips: comments, `supply`, `wire`, `timescale`, `celldefine`, etc.
pub fn parse_functional_model(verilog_src: &str) -> Option<BehavioralModel> {
    let mut module_name = String::new();
    let mut port_names: Vec<String> = Vec::new();
    let mut input_ports: Vec<String> = Vec::new();
    let mut output_ports: Vec<String> = Vec::new();
    let mut gates: Vec<BehavioralGate> = Vec::new();

    // Simple state machine
    enum State {
        LookingForModule,
        InModulePorts,
        InModuleBody,
    }
    let mut state = State::LookingForModule;

    // Accumulate multi-line tokens
    let mut accum = String::new();

    for line in verilog_src.lines() {
        let trimmed = line.trim();

        // Skip comments and preprocessor directives
        if trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed.starts_with("*/")
            || trimmed.starts_with('*')
            || trimmed.starts_with('`')
            || trimmed.is_empty()
        {
            continue;
        }

        match state {
            State::LookingForModule => {
                if trimmed.starts_with("module ") {
                    // Parse: module sky130_fd_sc_hd__o21ai (
                    accum.clear();
                    accum.push_str(trimmed);

                    if accum.contains(");") {
                        parse_module_header(&accum, &mut module_name, &mut port_names);
                        state = State::InModuleBody;
                    } else {
                        state = State::InModulePorts;
                    }
                }
            }
            State::InModulePorts => {
                accum.push(' ');
                accum.push_str(trimmed);
                if accum.contains(");") {
                    parse_module_header(&accum, &mut module_name, &mut port_names);
                    state = State::InModuleBody;
                }
            }
            State::InModuleBody => {
                if trimmed == "endmodule" {
                    break;
                }

                // Parse port direction declarations
                if trimmed.starts_with("output ") {
                    let names = parse_port_declaration(trimmed, "output");
                    output_ports.extend(names);
                    continue;
                }
                if trimmed.starts_with("input ") {
                    let names = parse_port_declaration(trimmed, "input");
                    input_ports.extend(names);
                    continue;
                }

                // Skip non-gate lines
                if trimmed.starts_with("wire ")
                    || trimmed.starts_with("supply")
                    || trimmed.starts_with("pullup")
                    || trimmed.starts_with("pulldown")
                    || trimmed.starts_with("reg ")
                {
                    continue;
                }

                // Try to parse gate instantiation
                // Could be multi-line, accumulate until we see ';'
                if !trimmed.starts_with("//") && !trimmed.is_empty() {
                    accum.clear();
                    accum.push_str(trimmed);
                    if !accum.contains(';') {
                        // Multi-line gate instantiation - keep accumulating
                        // (unusual in functional models but handle it)
                        continue;
                    }
                    if let Some(gate) = parse_gate_instantiation(&accum) {
                        gates.push(gate);
                    }
                }
            }
        }
    }

    if module_name.is_empty() {
        return None;
    }

    Some(BehavioralModel {
        module_name,
        inputs: input_ports,
        outputs: output_ports,
        gates,
    })
}

/// Parse module header: `module name ( port1, port2, ... );`
fn parse_module_header(header: &str, name: &mut String, ports: &mut Vec<String>) {
    // Find module name
    let after_module = header.strip_prefix("module ").unwrap_or(header);
    if let Some(paren_pos) = after_module.find('(') {
        *name = after_module[..paren_pos].trim().to_string();

        // Extract ports between ( and )
        let rest = &after_module[paren_pos + 1..];
        if let Some(close) = rest.find(')') {
            let port_str = &rest[..close];
            *ports = port_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
}

/// Parse port declarations like `output Y ;` or `input A1;`
fn parse_port_declaration(line: &str, keyword: &str) -> Vec<String> {
    let after_kw = line
        .strip_prefix(keyword)
        .unwrap_or(line)
        .trim()
        .trim_end_matches(';')
        .trim();
    after_kw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse a gate instantiation line like:
/// `or   or0   (or0_out    , A2, A1         );`
/// Returns BehavioralGate or None if not a gate.
fn parse_gate_instantiation(line: &str) -> Option<BehavioralGate> {
    let line = line.trim().trim_end_matches(';').trim();

    // Find the parenthesized port list
    let paren_start = line.find('(')?;
    let paren_end = line.rfind(')')?;

    // Everything before '(' is: gate_type [#delay] instance_name
    let prefix = line[..paren_start].trim();
    let port_list = &line[paren_start + 1..paren_end];

    // Split prefix into tokens
    let tokens: Vec<&str> = prefix.split_whitespace().collect();
    if tokens.is_empty() {
        return None;
    }

    let gate_type = tokens[0].to_string();

    // Skip delay specifications like `UNIT_DELAY
    // (used in DFF models - we skip those cells entirely)

    // Parse port list: output is first, then inputs
    let ports: Vec<String> = port_list
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if ports.is_empty() {
        return None;
    }

    // Skip named port connections (module instantiations like .A(wire))
    // These are hierarchical cell references, not gate primitives
    if ports[0].starts_with('.') {
        return None;
    }

    let output = ports[0].clone();
    let inputs = ports[1..].to_vec();

    Some(BehavioralGate {
        gate_type,
        output,
        inputs,
    })
}

// ============================================================================
// Parser: UDP truth tables
// ============================================================================

/// Parse a Verilog UDP primitive definition.
pub fn parse_udp(verilog_src: &str) -> Option<UdpModel> {
    let mut name = String::new();
    let mut output_port = String::new();
    let mut input_ports: Vec<String> = Vec::new();
    let mut rows: Vec<UdpRow> = Vec::new();

    enum State {
        LookingForPrimitive,
        InPrimitivePorts,
        InPrimitiveBody,
        InTable,
    }
    let mut state = State::LookingForPrimitive;
    let mut accum = String::new();

    for line in verilog_src.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed.starts_with("*/")
            || trimmed.starts_with('*')
            || trimmed.starts_with('`')
            || trimmed.is_empty()
        {
            continue;
        }

        match state {
            State::LookingForPrimitive => {
                if trimmed.starts_with("primitive ") {
                    accum.clear();
                    accum.push_str(trimmed);
                    if accum.contains(");") {
                        parse_primitive_header(&accum, &mut name);
                        state = State::InPrimitiveBody;
                    } else {
                        state = State::InPrimitivePorts;
                    }
                }
            }
            State::InPrimitivePorts => {
                accum.push(' ');
                accum.push_str(trimmed);
                if accum.contains(");") {
                    parse_primitive_header(&accum, &mut name);
                    state = State::InPrimitiveBody;
                }
            }
            State::InPrimitiveBody => {
                if trimmed == "endprimitive" {
                    break;
                }
                if trimmed.starts_with("output ") {
                    output_port = trimmed
                        .strip_prefix("output ")
                        .unwrap()
                        .trim()
                        .trim_end_matches(';')
                        .trim()
                        .to_string();
                } else if trimmed.starts_with("input ") {
                    let ports_str = trimmed
                        .strip_prefix("input ")
                        .unwrap()
                        .trim()
                        .trim_end_matches(';')
                        .trim();
                    // Accumulate input ports (may be on separate lines)
                    for port in ports_str.split(',') {
                        let p = port.trim().to_string();
                        if !p.is_empty() {
                            input_ports.push(p);
                        }
                    }
                } else if trimmed == "table" {
                    state = State::InTable;
                }
            }
            State::InTable => {
                if trimmed == "endtable" {
                    state = State::InPrimitiveBody;
                    continue;
                }

                // Parse table row like: `0   ?   0  :  1   ;`
                let row_str = trimmed.trim_end_matches(';').trim();
                // Remove comment prefix (//  A0  A1  S  :  Y)
                if row_str.starts_with("//") {
                    continue;
                }

                if let Some(colon_pos) = row_str.find(':') {
                    let input_str = row_str[..colon_pos].trim();
                    let output_str = row_str[colon_pos + 1..].trim();

                    let input_vals: Vec<Option<bool>> = input_str
                        .split_whitespace()
                        .map(|s| match s {
                            "0" => Some(false),
                            "1" => Some(true),
                            "?" | "x" | "X" => None,
                            _ => None,
                        })
                        .collect();

                    let output_val = match output_str {
                        "1" => true,
                        "0" => false,
                        _ => continue, // skip 'x' outputs
                    };

                    if input_vals.len() == input_ports.len() {
                        rows.push(UdpRow {
                            inputs: input_vals,
                            output: output_val,
                        });
                    }
                }
            }
        }
    }

    if name.is_empty() {
        return None;
    }

    Some(UdpModel {
        name,
        output: output_port,
        inputs: input_ports,
        rows,
    })
}

fn parse_primitive_header(header: &str, name: &mut String) {
    let after_prim = header.strip_prefix("primitive ").unwrap_or(header);
    if let Some(paren_pos) = after_prim.find('(') {
        *name = after_prim[..paren_pos].trim().to_string();
    }
}

// ============================================================================
// AIG builder from behavioral models
// ============================================================================

/// Evaluate a UDP truth table for specific input values.
/// Returns the output value. Panics if no matching row found.
fn eval_udp_for_inputs(udp: &UdpModel, input_vals: &[bool]) -> bool {
    assert_eq!(input_vals.len(), udp.inputs.len());

    for row in &udp.rows {
        let matches = row
            .inputs
            .iter()
            .zip(input_vals.iter())
            .all(|(pattern, &actual)| match pattern {
                None => true,           // don't-care matches anything
                Some(v) => *v == actual, // must match exactly
            });
        if matches {
            return row.output;
        }
    }

    panic!(
        "UDP '{}': no matching row for inputs {:?}",
        udp.name, input_vals
    );
}

/// Internal wire value during AIG construction from a behavioral model.
/// Can be either a reference to a module input (by aigpin_iv) or an intermediate
/// AND gate we built (by gate index).
#[derive(Clone, Copy, Debug)]
enum WireVal {
    /// An AIG pin with inversion bit (aigpin_iv). Bit 0 = inverted.
    AigPin(usize),
    /// Constant value
    Const(bool),
}

impl WireVal {
    /// Get the aigpin_iv value, creating const-0 = AigPin(0) convention.
    fn as_aigpin_iv(self) -> i64 {
        match self {
            WireVal::AigPin(iv) => iv as i64,
            WireVal::Const(false) => 0, // const-0
            WireVal::Const(true) => 1,  // const-1
        }
    }

    /// Invert this wire value.
    fn inverted(self) -> Self {
        match self {
            WireVal::AigPin(iv) => WireVal::AigPin(iv ^ 1),
            WireVal::Const(v) => WireVal::Const(!v),
        }
    }
}

/// Convert a parsed behavioral model to an AIG decomposition for a specific output pin.
///
/// # Arguments
/// * `model` - The parsed behavioral model
/// * `cell_inputs` - CellInputs with aigpin_iv values for each module input
/// * `output_pin` - Which output pin to build the logic cone for (e.g. "Y", "SUM", "COUT")
/// * `udps` - Available UDP models for resolving UDP instantiations
///
/// # Returns
/// A `DecompResult` describing the AIG for this output
pub fn decompose_from_behavioral(
    model: &BehavioralModel,
    cell_inputs: &CellInputs,
    output_pin: &str,
    udps: &HashMap<String, UdpModel>,
) -> DecompResult {
    // Map wire names to their values
    let mut wires: HashMap<String, WireVal> = HashMap::new();

    // Map module input port names to their aigpin_iv values from CellInputs
    for input_name in &model.inputs {
        let aigpin_iv = get_cell_input_by_name(cell_inputs, input_name);
        assert_ne!(
            aigpin_iv,
            usize::MAX,
            "Input pin '{}' not set in CellInputs for cell '{}'",
            input_name,
            model.module_name
        );
        wires.insert(input_name.clone(), WireVal::AigPin(aigpin_iv));
    }

    // Process gates in order (they're topologically sorted in the Verilog)
    let mut and_gates: Vec<(i64, i64)> = Vec::new();
    // Map from gate output index to AND gate index in and_gates
    // gate_idx_for_wire[wire_name] = index into and_gates (0-based)

    for gate in &model.gates {
        let gate_type = gate.gate_type.as_str();

        // Skip buf gates - they're just passthroughs
        if gate_type == "buf" {
            assert_eq!(gate.inputs.len(), 1, "buf must have exactly 1 input");
            let input_val = wires
                .get(&gate.inputs[0])
                .copied()
                .unwrap_or_else(|| panic!("Unknown wire '{}' in buf gate", gate.inputs[0]));
            wires.insert(gate.output.clone(), input_val);
            continue;
        }

        // Handle UDP instantiations
        if gate_type.starts_with("sky130_fd_sc_hd__udp_") {
            let result = build_udp_aig(gate, &wires, udps, &mut and_gates);
            wires.insert(gate.output.clone(), result);
            continue;
        }

        // Resolve all input wires
        let input_vals: Vec<WireVal> = gate
            .inputs
            .iter()
            .map(|name| {
                wires
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| panic!("Unknown wire '{}' in {} gate", name, gate_type))
            })
            .collect();

        let result = match gate_type {
            "not" => {
                assert_eq!(input_vals.len(), 1);
                input_vals[0].inverted()
            }
            "and" => build_chain_gate(&input_vals, false, false, &mut and_gates),
            "nand" => build_chain_gate(&input_vals, false, true, &mut and_gates),
            "or" => build_chain_gate(&input_vals, true, true, &mut and_gates),
            "nor" => build_chain_gate(&input_vals, true, false, &mut and_gates),
            "xor" => build_xor_chain(&input_vals, false, &mut and_gates),
            "xnor" => build_xor_chain(&input_vals, true, &mut and_gates),
            _ => panic!(
                "Unknown gate type '{}' in model '{}'",
                gate_type, model.module_name
            ),
        };

        wires.insert(gate.output.clone(), result);
    }

    // Get the output wire value
    let output_val = wires
        .get(output_pin)
        .copied()
        .unwrap_or_else(|| panic!("Output pin '{}' not found in model", output_pin));

    // Convert to DecompResult
    match output_val {
        WireVal::AigPin(iv) => {
            let pin_idx = iv >> 1;
            let inverted = (iv & 1) != 0;
            DecompResult {
                and_gates,
                output_idx: pin_idx as i64,
                output_inverted: inverted,
            }
        }
        WireVal::Const(v) => {
            // Constant output: use pin index 0 which is const-0
            DecompResult {
                and_gates,
                output_idx: 0,
                output_inverted: v,
            }
        }
    }
}

/// Build an AND/NAND/OR/NOR chain for multi-input gates.
///
/// For AND/NAND: compute AND of all inputs, optionally invert at the end.
/// For OR/NOR: invert all inputs, AND them, optionally invert at the end.
///   OR(a,b,c) = NOT(AND(NOT a, NOT b, NOT c))
///   NOR(a,b,c) = AND(NOT a, NOT b, NOT c)
fn build_chain_gate(
    inputs: &[WireVal],
    invert_inputs: bool,
    invert_output: bool,
    and_gates: &mut Vec<(i64, i64)>,
) -> WireVal {
    assert!(inputs.len() >= 2, "Gate must have at least 2 inputs");

    let inputs: Vec<WireVal> = if invert_inputs {
        inputs.iter().map(|v| v.inverted()).collect()
    } else {
        inputs.to_vec()
    };

    // Chain 2-input AND gates
    let mut accum = inputs[0];
    for input in &inputs[1..] {
        let a_ref = accum.as_aigpin_iv();
        let b_ref = input.as_aigpin_iv();
        and_gates.push((a_ref, b_ref));
        let gate_idx = and_gates.len() - 1;
        accum = WireVal::AigPin(GATE_MARKER | (gate_idx << 1));
    }

    if invert_output {
        accum.inverted()
    } else {
        accum
    }
}

/// Marker bit to distinguish gate references from pin references.
/// Gate outputs use bit 30 set. This limits us to ~500M gates (more than enough).
const GATE_MARKER: usize = 1 << 30;

/// Check if an aigpin_iv value is a gate reference.
fn is_gate_ref(aigpin_iv: usize) -> bool {
    aigpin_iv & GATE_MARKER != 0
}

/// Extract gate index from a gate-reference aigpin_iv.
fn gate_ref_index(aigpin_iv: usize) -> usize {
    (aigpin_iv & !GATE_MARKER & !1) >> 1
}

/// Build a 2-input XOR: A ^ B = !(!( A & !B) & !(!A & B))
fn build_xor_2(a: WireVal, b: WireVal, and_gates: &mut Vec<(i64, i64)>) -> WireVal {
    let a_iv = a.as_aigpin_iv();
    let b_iv = b.as_aigpin_iv();
    let a_inv_iv = a.inverted().as_aigpin_iv();
    let b_inv_iv = b.inverted().as_aigpin_iv();

    // gate0: A & !B
    and_gates.push((a_iv, b_inv_iv));
    let g0 = and_gates.len() - 1;
    let g0_val = WireVal::AigPin(GATE_MARKER | (g0 << 1));

    // gate1: !A & B
    and_gates.push((a_inv_iv, b_iv));
    let g1 = and_gates.len() - 1;
    let g1_val = WireVal::AigPin(GATE_MARKER | (g1 << 1));

    // gate2: !(A & !B) & !(!A & B)  -- this is XNOR, inverted gives XOR
    let g0_inv_iv = g0_val.inverted().as_aigpin_iv();
    let g1_inv_iv = g1_val.inverted().as_aigpin_iv();
    and_gates.push((g0_inv_iv, g1_inv_iv));
    let g2 = and_gates.len() - 1;
    // XOR = NOT(gate2), so return inverted
    WireVal::AigPin(GATE_MARKER | (g2 << 1) | 1)
}

/// Build XOR/XNOR chain for multi-input gates.
fn build_xor_chain(
    inputs: &[WireVal],
    invert_output: bool,
    and_gates: &mut Vec<(i64, i64)>,
) -> WireVal {
    assert!(inputs.len() >= 2);

    let mut accum = inputs[0];
    for input in &inputs[1..] {
        accum = build_xor_2(accum, *input, and_gates);
    }

    if invert_output {
        accum.inverted()
    } else {
        accum
    }
}

/// Build AIG for a UDP instantiation by converting truth table to sum-of-products.
fn build_udp_aig(
    gate: &BehavioralGate,
    wires: &HashMap<String, WireVal>,
    udps: &HashMap<String, UdpModel>,
    and_gates: &mut Vec<(i64, i64)>,
) -> WireVal {
    let udp_name = &gate.gate_type;
    let udp = udps
        .get(udp_name)
        .unwrap_or_else(|| panic!("UDP '{}' not found in loaded models", udp_name));

    // Get input wire values
    let input_vals: Vec<WireVal> = gate
        .inputs
        .iter()
        .map(|name| {
            wires
                .get(name)
                .copied()
                .unwrap_or_else(|| panic!("Unknown wire '{}' in UDP '{}'", name, udp_name))
        })
        .collect();

    assert_eq!(
        input_vals.len(),
        udp.inputs.len(),
        "UDP '{}' expects {} inputs, got {}",
        udp_name,
        udp.inputs.len(),
        input_vals.len()
    );

    // Build sum-of-products from truth table rows where output=1
    // Each row with output=1 becomes a product (AND) term.
    // Product terms are ORed together.
    //
    // For rows where output=0, we don't need to do anything explicitly.
    // Don't-care (?) inputs are omitted from the product term.

    let one_rows: Vec<&UdpRow> = udp.rows.iter().filter(|r| r.output).collect();

    if one_rows.is_empty() {
        // Output is always 0
        return WireVal::Const(false);
    }

    // Build each product term
    let mut product_terms: Vec<WireVal> = Vec::new();

    for row in &one_rows {
        // Collect non-don't-care inputs for this product term
        let mut term_inputs: Vec<WireVal> = Vec::new();
        for (i, pattern) in row.inputs.iter().enumerate() {
            match pattern {
                Some(true) => term_inputs.push(input_vals[i]),
                Some(false) => term_inputs.push(input_vals[i].inverted()),
                None => {} // don't-care - omit from product
            }
        }

        if term_inputs.is_empty() {
            // All inputs are don't-care: output is unconditionally 1
            return WireVal::Const(true);
        }

        if term_inputs.len() == 1 {
            product_terms.push(term_inputs[0]);
        } else {
            // Build AND chain for this product term
            let product = build_chain_gate(&term_inputs, false, false, and_gates);
            product_terms.push(product);
        }
    }

    if product_terms.len() == 1 {
        return product_terms[0];
    }

    // OR the product terms: OR(a,b,...) = NOT(AND(NOT a, NOT b, ...))
    build_chain_gate(&product_terms, true, true, and_gates)
}

/// Map a CellInputs pin name to its aigpin_iv value.
fn get_cell_input_by_name(inputs: &CellInputs, name: &str) -> usize {
    match name {
        "A" => inputs.a,
        "A_N" => inputs.a_n,
        "A0" => inputs.a0,
        "A1" => inputs.a1,
        "A1_N" => inputs.a1_n,
        "A2" => inputs.a2,
        "A2_N" => inputs.a2_n,
        "A3" => inputs.a3,
        "A4" => inputs.a4,
        "B" => inputs.b,
        "B_N" => inputs.b_n,
        "B1" => inputs.b1,
        "B1_N" => inputs.b1_n,
        "B2" => inputs.b2,
        "C" => inputs.c,
        "C_N" => inputs.c_n,
        "C1" => inputs.c1,
        "D" => inputs.d,
        "D1" => inputs.d1,
        "S" => inputs.s,
        "S0" => inputs.s0,
        "S1" => {
            // S1 not in CellInputs - we'd need to add it for mux4
            // For now, this will only be reached if mux4 is in the design
            panic!("S1 pin not yet supported in CellInputs - add it for mux4 support")
        }
        "CIN" => inputs.cin,
        "SET_B" => inputs.set_b,
        "RESET_B" => inputs.reset_b,
        _ => usize::MAX,
    }
}

// ============================================================================
// DecompResult conversion: GATE_MARKER encoding -> standard negative-index encoding
// ============================================================================


/// Post-process a DecompResult built with GATE_MARKER encoding to use
/// standard negative-index encoding for the and_gates references.
fn finalize_decomp_result(
    and_gates: Vec<(i64, i64)>,
    output: WireVal,
) -> DecompResult {
    // Convert gate references in and_gates from GATE_MARKER to negative indices
    let converted_gates: Vec<(i64, i64)> = and_gates
        .iter()
        .map(|(a, b)| {
            (
                convert_ref_to_standard(*a),
                convert_ref_to_standard(*b),
            )
        })
        .collect();

    match output {
        WireVal::AigPin(iv) if is_gate_ref(iv) => {
            let gate_idx = gate_ref_index(iv);
            let inverted = (iv & 1) != 0;
            DecompResult {
                and_gates: converted_gates,
                output_idx: -(gate_idx as i64) - 1,
                output_inverted: inverted,
            }
        }
        WireVal::AigPin(iv) => {
            let pin_idx = iv >> 1;
            let inverted = (iv & 1) != 0;
            DecompResult {
                and_gates: converted_gates,
                output_idx: pin_idx as i64,
                output_inverted: inverted,
            }
        }
        WireVal::Const(v) => DecompResult {
            and_gates: converted_gates,
            output_idx: 0,
            output_inverted: v,
        },
    }
}

/// Convert a single reference value from GATE_MARKER encoding to standard.
fn convert_ref_to_standard(ref_val: i64) -> i64 {
    let uval = ref_val as usize;
    if is_gate_ref(uval) {
        let gate_idx = gate_ref_index(uval);
        let inverted = (uval & 1) != 0;
        let base = -((gate_idx as i64) * 2 + 1);
        if inverted { base ^ 1 } else { base }
    } else {
        ref_val
    }
}

// ============================================================================
// Model loading
// ============================================================================

/// Validate that all gate inputs in a model reference known wires.
/// Returns false if any gate references a wire that isn't a module input
/// or the output of a previous gate.
fn validate_model(model: &BehavioralModel) -> bool {
    let mut known_wires: std::collections::HashSet<&str> = std::collections::HashSet::new();

    // Module inputs are known
    for name in &model.inputs {
        known_wires.insert(name.as_str());
    }

    for gate in &model.gates {
        // Check all inputs are known
        for input_name in &gate.inputs {
            if !known_wires.contains(input_name.as_str()) {
                return false;
            }
        }
        // Gate output becomes known
        known_wires.insert(gate.output.as_str());
    }

    // Check that all declared outputs are produced
    for output_name in &model.outputs {
        if !known_wires.contains(output_name.as_str()) {
            return false;
        }
    }

    true
}

/// Load all PDK models needed for a set of cell types.
///
/// # Arguments
/// * `pdk_cells_path` - Path to the `cells/` directory of the PDK
/// * `cell_types` - Set of cell type names that need models (e.g. ["o21ai", "nand2", ...])
///
/// # Returns
/// A PdkModels containing all loaded models and UDPs.
pub fn load_pdk_models(pdk_cells_path: &Path, cell_types: &[String]) -> PdkModels {
    let mut models = HashMap::new();
    let mut udps: HashMap<String, UdpModel> = HashMap::new();

    // Derive models_path from cells_path: ../models/
    let models_path = pdk_cells_path.parent().unwrap().join("models");

    for cell_type in cell_types {
        // Skip sequential and tie cells - handled elsewhere
        if is_sequential_cell(cell_type)
            || is_tie_cell(cell_type)
        {
            continue;
        }

        let cell_dir = pdk_cells_path.join(cell_type);
        let func_file = cell_dir.join(format!(
            "sky130_fd_sc_hd__{}.functional.v",
            cell_type
        ));

        if !func_file.exists() {
            clilog::warn!(
                "PDK functional model not found for cell '{}': {}",
                cell_type,
                func_file.display()
            );
            continue;
        }

        let src = std::fs::read_to_string(&func_file).unwrap_or_else(|e| {
            panic!(
                "Failed to read PDK model {}: {}",
                func_file.display(),
                e
            )
        });

        if let Some(model) = parse_functional_model(&src) {
            // Validate: check that all gate inputs reference known wires
            // (module inputs, or outputs of earlier gates)
            let valid = validate_model(&model);
            if !valid {
                clilog::debug!(
                    "Skipping PDK model '{}': has unresolvable wire references (macro cell?)",
                    cell_type
                );
                continue;
            }

            // Check if the model uses any UDPs and load them
            for gate in &model.gates {
                if gate.gate_type.starts_with("sky130_fd_sc_hd__udp_") {
                    let udp_name = &gate.gate_type;
                    if !udps.contains_key(udp_name) {
                        if let Some(udp) = load_udp(&models_path, udp_name) {
                            udps.insert(udp_name.clone(), udp);
                        }
                    }
                }
            }
            models.insert(cell_type.clone(), model);
        } else {
            clilog::warn!(
                "Failed to parse PDK functional model for cell '{}'",
                cell_type
            );
        }
    }

    clilog::info!(
        "Loaded {} PDK cell models and {} UDP models",
        models.len(),
        udps.len()
    );

    PdkModels { models, udps }
}

/// Load a single UDP model from the models directory.
fn load_udp(models_path: &Path, udp_name: &str) -> Option<UdpModel> {
    // UDP name format: sky130_fd_sc_hd__udp_mux_2to1_N
    // Directory format: udp_mux_2to1_n (lowercase)
    // File: sky130_fd_sc_hd__udp_mux_2to1_n.v (note: lowercase in filename sometimes differs)

    // Extract the udp part after "sky130_fd_sc_hd__"
    let udp_suffix = udp_name
        .strip_prefix("sky130_fd_sc_hd__")
        .unwrap_or(udp_name);

    // The directory name is lowercase version of the suffix
    let dir_name = udp_suffix.to_lowercase();
    let udp_dir = models_path.join(&dir_name);

    if !udp_dir.exists() {
        clilog::warn!("UDP directory not found: {}", udp_dir.display());
        return None;
    }

    // Find the .v file (not .blackbox.v or .tb.v)
    let udp_file = udp_dir.join(format!("sky130_fd_sc_hd__{}.v", dir_name));

    if !udp_file.exists() {
        clilog::warn!("UDP file not found: {}", udp_file.display());
        return None;
    }

    let src = std::fs::read_to_string(&udp_file).unwrap_or_else(|e| {
        panic!("Failed to read UDP file {}: {}", udp_file.display(), e)
    });

    parse_udp(&src)
}

/// Decompose a cell using PDK models.
///
/// This is the main entry point for PDK-based decomposition. Panics if no
/// PDK model is available for the given cell type.
pub fn decompose_with_pdk(
    cell_type: &str,
    inputs: &CellInputs,
    output_pin: &str,
    pdk: &PdkModels,
) -> DecompResult {
    let model = pdk.models.get(cell_type).unwrap_or_else(|| {
        panic!(
            "No PDK model found for cell type '{}'. Ensure the sky130_fd_sc_hd submodule is \
             initialized (git submodule update --init) and the cell model exists.",
            cell_type
        )
    });

    let mut and_gates = Vec::new();
    let mut wires: HashMap<String, WireVal> = HashMap::new();

    // Map module input port names to their aigpin_iv values
    for input_name in &model.inputs {
        let aigpin_iv = get_cell_input_by_name(inputs, input_name);
        if aigpin_iv == usize::MAX {
            panic!(
                "Input pin '{}' not set in CellInputs for cell type '{}'",
                input_name, cell_type
            );
        }
        wires.insert(input_name.clone(), WireVal::AigPin(aigpin_iv));
    }

    // Process gates in order
    for gate in &model.gates {
        let gate_type = gate.gate_type.as_str();

        if gate_type == "buf" {
            assert_eq!(gate.inputs.len(), 1);
            let input_val = wires.get(&gate.inputs[0]).copied().unwrap_or_else(|| {
                panic!("Unknown wire '{}' in buf gate", gate.inputs[0])
            });
            wires.insert(gate.output.clone(), input_val);
            continue;
        }

        if gate_type.starts_with("sky130_fd_sc_hd__udp_") {
            let result = build_udp_aig(gate, &wires, &pdk.udps, &mut and_gates);
            wires.insert(gate.output.clone(), result);
            continue;
        }

        let input_vals: Vec<WireVal> = gate
            .inputs
            .iter()
            .map(|name| {
                wires.get(name).copied().unwrap_or_else(|| {
                    panic!("Unknown wire '{}' in {} gate", name, gate_type)
                })
            })
            .collect();

        let result = match gate_type {
            "not" => {
                assert_eq!(input_vals.len(), 1);
                input_vals[0].inverted()
            }
            "and" => build_chain_gate(&input_vals, false, false, &mut and_gates),
            "nand" => build_chain_gate(&input_vals, false, true, &mut and_gates),
            "or" => build_chain_gate(&input_vals, true, true, &mut and_gates),
            "nor" => build_chain_gate(&input_vals, true, false, &mut and_gates),
            "xor" => build_xor_chain(&input_vals, false, &mut and_gates),
            "xnor" => build_xor_chain(&input_vals, true, &mut and_gates),
            _ => panic!(
                "Unknown gate type '{}' in model '{}'",
                gate_type, model.module_name
            ),
        };

        wires.insert(gate.output.clone(), result);
    }

    // Get the requested output
    let output_val = wires.get(output_pin).copied().unwrap_or_else(|| {
        panic!(
            "Output pin '{}' not found in model '{}'. Available: {:?}",
            output_pin, model.module_name, model.outputs
        )
    });

    finalize_decomp_result(and_gates, output_val)
}

// ============================================================================
// Gate-level evaluator (for testing)
// ============================================================================

/// Directly evaluate a behavioral model's gate network for given input values.
/// This doesn't go through AIG - it directly interprets the Verilog gates.
/// Used as a reference oracle in tests.
pub fn eval_behavioral_model(
    model: &BehavioralModel,
    input_values: &HashMap<String, bool>,
    output_pin: &str,
    udps: &HashMap<String, UdpModel>,
) -> bool {
    let mut wires: HashMap<String, bool> = HashMap::new();

    // Set input values
    for (name, &val) in input_values {
        wires.insert(name.clone(), val);
    }

    // Evaluate gates in order
    for gate in &model.gates {
        let gate_type = gate.gate_type.as_str();

        if gate_type == "buf" {
            let v = wires[&gate.inputs[0]];
            wires.insert(gate.output.clone(), v);
            continue;
        }

        if gate_type.starts_with("sky130_fd_sc_hd__udp_") {
            let udp = &udps[gate_type];
            let input_vals: Vec<bool> =
                gate.inputs.iter().map(|name| wires[name]).collect();
            let result = eval_udp_for_inputs(udp, &input_vals);
            wires.insert(gate.output.clone(), result);
            continue;
        }

        let input_vals: Vec<bool> =
            gate.inputs.iter().map(|name| {
                *wires.get(name).unwrap_or_else(|| {
                    panic!(
                        "Wire '{}' not found in model '{}' at gate '{}' (type '{}'). Available wires: {:?}",
                        name, model.module_name, gate.output, gate_type,
                        wires.keys().collect::<Vec<_>>()
                    )
                })
            }).collect();

        let result = match gate_type {
            "not" => !input_vals[0],
            "and" => input_vals.iter().all(|&v| v),
            "nand" => !input_vals.iter().all(|&v| v),
            "or" => input_vals.iter().any(|&v| v),
            "nor" => !input_vals.iter().any(|&v| v),
            "xor" => input_vals.iter().fold(false, |acc, &v| acc ^ v),
            "xnor" => !input_vals.iter().fold(false, |acc, &v| acc ^ v),
            _ => panic!("Unknown gate type: {}", gate_type),
        };

        wires.insert(gate.output.clone(), result);
    }

    wires[output_pin]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_o21ai() {
        let src = std::fs::read_to_string(
            "sky130_fd_sc_hd/cells/o21ai/sky130_fd_sc_hd__o21ai.functional.v",
        )
        .unwrap();
        let model = parse_functional_model(&src).unwrap();
        assert_eq!(model.module_name, "sky130_fd_sc_hd__o21ai");
        assert_eq!(model.inputs, vec!["A1", "A2", "B1"]);
        assert_eq!(model.outputs, vec!["Y"]);
        assert_eq!(model.gates.len(), 3); // or, nand, buf
        assert_eq!(model.gates[0].gate_type, "or");
        assert_eq!(model.gates[1].gate_type, "nand");
        assert_eq!(model.gates[2].gate_type, "buf");
    }

    #[test]
    fn test_parse_ha() {
        let src = std::fs::read_to_string(
            "sky130_fd_sc_hd/cells/ha/sky130_fd_sc_hd__ha.functional.v",
        )
        .unwrap();
        let model = parse_functional_model(&src).unwrap();
        assert_eq!(model.module_name, "sky130_fd_sc_hd__ha");
        assert_eq!(model.inputs, vec!["A", "B"]);
        assert_eq!(model.outputs, vec!["COUT", "SUM"]);
        assert_eq!(model.gates.len(), 4); // and, buf, xor, buf
    }

    #[test]
    fn test_parse_mux2i_udp() {
        let src = std::fs::read_to_string(
            "sky130_fd_sc_hd/cells/mux2i/sky130_fd_sc_hd__mux2i.functional.v",
        )
        .unwrap();
        let model = parse_functional_model(&src).unwrap();
        assert_eq!(model.module_name, "sky130_fd_sc_hd__mux2i");
        assert_eq!(model.inputs, vec!["A0", "A1", "S"]);
        assert_eq!(model.outputs, vec!["Y"]);
        // Should have UDP instantiation + buf
        assert_eq!(model.gates.len(), 2);
        assert!(model.gates[0]
            .gate_type
            .starts_with("sky130_fd_sc_hd__udp_mux_2to1"));
    }

    #[test]
    fn test_parse_udp_mux_2to1_n() {
        let src = std::fs::read_to_string(
            "sky130_fd_sc_hd/models/udp_mux_2to1_n/sky130_fd_sc_hd__udp_mux_2to1_n.v",
        )
        .unwrap();
        let udp = parse_udp(&src).unwrap();
        assert_eq!(udp.inputs, vec!["A0", "A1", "S"]);
        assert_eq!(udp.output, "Y");
        assert_eq!(udp.rows.len(), 6);

        // Verify: A0=0, S=0 -> Y=1 (inverted mux)
        assert!(eval_udp_for_inputs(&udp, &[false, false, false]));
        // A0=1, S=0 -> Y=0
        assert!(!eval_udp_for_inputs(&udp, &[true, false, false]));
        // A1=0, S=1 -> Y=1
        assert!(eval_udp_for_inputs(&udp, &[false, false, true]));
        // A1=1, S=1 -> Y=0
        assert!(!eval_udp_for_inputs(&udp, &[false, true, true]));
    }

    /// Helper: set up CellInputs and a bool->aigpin mapping for testing.
    /// Returns (CellInputs, input_name_to_bool_map) for the given model.
    fn setup_test_inputs(
        model: &BehavioralModel,
        values: &[bool],
    ) -> (CellInputs, HashMap<String, bool>) {
        assert_eq!(model.inputs.len(), values.len());
        let mut cell_inputs = CellInputs::new();
        let mut bool_map = HashMap::new();

        for (i, (name, &val)) in model.inputs.iter().zip(values.iter()).enumerate() {
            // Assign aigpin (i+1) to each input
            // aigpin_iv = ((i+1) << 1) | 0  for non-inverted
            let aigpin_iv = (i + 1) << 1;
            cell_inputs.set_pin(name, aigpin_iv);
            bool_map.insert(name.clone(), val);
        }

        (cell_inputs, bool_map)
    }

    /// Evaluate a DecompResult with concrete boolean inputs.
    /// `pin_values` maps aigpin (not aigpin_iv) to bool.
    fn eval_decomp_bool(decomp: &DecompResult, pin_values: &HashMap<usize, bool>) -> bool {
        let mut gate_outputs: Vec<bool> = Vec::new();

        for (a_ref, b_ref) in &decomp.and_gates {
            let a_val = resolve_decomp_ref_bool(*a_ref, pin_values, &gate_outputs);
            let b_val = resolve_decomp_ref_bool(*b_ref, pin_values, &gate_outputs);
            gate_outputs.push(a_val && b_val);
        }

        let output = if decomp.output_idx < 0 {
            let gate_idx = (-decomp.output_idx - 1) as usize;
            gate_outputs[gate_idx]
        } else {
            let aigpin = decomp.output_idx as usize;
            *pin_values.get(&aigpin).unwrap_or_else(|| {
                panic!("Pin {} not found in values map", aigpin)
            })
        };

        if decomp.output_inverted {
            !output
        } else {
            output
        }
    }

    fn resolve_decomp_ref_bool(
        ref_val: i64,
        pin_values: &HashMap<usize, bool>,
        gate_outputs: &[bool],
    ) -> bool {
        if ref_val < 0 {
            let abs_ref = -ref_val;
            let gate_idx = ((abs_ref - 1) / 2) as usize;
            let inverted = (abs_ref % 2) == 0;
            let val = gate_outputs[gate_idx];
            if inverted { !val } else { val }
        } else {
            // ref_val is aigpin_iv: pin = ref_val >> 1, inverted = ref_val & 1
            let aigpin = (ref_val >> 1) as usize;
            let inverted = (ref_val & 1) != 0;
            let val = *pin_values.get(&aigpin).unwrap_or_else(|| {
                panic!(
                    "Pin {} not found in values map (ref_val={})",
                    aigpin, ref_val
                )
            });
            if inverted { !val } else { val }
        }
    }

    /// Load all PDK models and UDPs from the submodule.
    fn load_test_pdk() -> PdkModels {
        let pdk_path = Path::new("sky130_fd_sc_hd/cells");
        if !pdk_path.exists() {
            panic!("sky130_fd_sc_hd submodule not initialized. Run: git submodule update --init");
        }

        // Get all cell types
        let mut cell_types: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(pdk_path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                cell_types.push(entry.file_name().to_string_lossy().to_string());
            }
        }
        cell_types.sort();

        load_pdk_models(pdk_path, &cell_types)
    }

    /// Check if all input pins of a model are recognized by CellInputs.
    fn all_inputs_supported(model: &BehavioralModel) -> bool {
        let supported = [
            "A", "A_N", "A0", "A1", "A1_N", "A2", "A2_N", "A3", "A4",
            "B", "B_N", "B1", "B1_N", "B2",
            "C", "C_N", "C1",
            "D", "D1",
            "S", "S0", "CIN",
            "SET_B", "RESET_B",
        ];
        model.inputs.iter().all(|name| supported.contains(&name.as_str()))
    }

    /// Exhaustive truth-table test for all combinational cells with supported pins.
    /// For each cell: evaluate all 2^N input combinations through both
    /// the AIG decomposition and the direct gate evaluator, verify they match.
    #[test]
    fn test_all_cells_vs_pdk() {
        let pdk = load_test_pdk();

        let mut tested = 0;
        let mut skipped_inputs = 0;
        let mut skipped_size = 0;

        // Sort keys for deterministic output
        let mut cell_types: Vec<&String> = pdk.models.keys().collect();
        cell_types.sort();

        for cell_type in cell_types {
            let model = &pdk.models[cell_type];
            let num_inputs = model.inputs.len();

            // Skip cells with unsupported input pin names
            if !all_inputs_supported(model) {
                skipped_inputs += 1;
                continue;
            }

            // Skip cells with too many inputs for exhaustive testing (> 8 inputs)
            if num_inputs > 8 {
                skipped_size += 1;
                continue;
            }

            let num_combos = 1u32 << num_inputs;

            for output_pin in &model.outputs {
                for combo in 0..num_combos {
                    // Convert combo to input booleans
                    let input_bools: Vec<bool> = (0..num_inputs)
                        .map(|i| ((combo >> i) & 1) != 0)
                        .collect();

                    // Set up CellInputs
                    let (cell_inputs, bool_map) = setup_test_inputs(model, &input_bools);

                    // Evaluate via direct gate interpretation (reference)
                    let expected =
                        eval_behavioral_model(model, &bool_map, output_pin, &pdk.udps);

                    // Evaluate via AIG decomposition
                    let decomp = decompose_with_pdk(
                        cell_type,
                        &cell_inputs,
                        output_pin,
                        &pdk,
                    );

                    // Build pin_values map for decomp evaluator
                    let mut pin_values: HashMap<usize, bool> = HashMap::new();
                    pin_values.insert(0, false); // const-0
                    for (i, &val) in input_bools.iter().enumerate() {
                        pin_values.insert(i + 1, val);
                    }

                    let actual = eval_decomp_bool(&decomp, &pin_values);

                    assert_eq!(
                        actual, expected,
                        "Mismatch for cell '{}' output '{}' with inputs {:?}: AIG={}, expected={}",
                        cell_type, output_pin, input_bools, actual, expected
                    );
                }
            }
            tested += 1;
        }

        println!(
            "Tested {} cell types exhaustively, skipped {} (unsupported pins), {} (too many inputs)",
            tested, skipped_inputs, skipped_size
        );
        assert!(tested > 50, "Expected to test at least 50 cell types, but only tested {}", tested);
    }
}
