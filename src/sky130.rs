// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! SKY130 standard cell library support for GEM.
//!
//! This module provides pin direction and width information for the SKY130
//! open-source PDK standard cells. SKY130 cells follow the naming convention:
//! `sky130_fd_sc_<variant>__<celltype>_<drive>`
//!
//! Supported variants: hd (high-density), hs (high-speed), ms (medium-speed)

use compact_str::CompactString;
use netlistdb::{Direction, LeafPinProvider};
use sverilogparse::SVerilogRange;

/// LeafPinProvider implementation for SKY130 standard cells.
pub struct SKY130LeafPins;

impl LeafPinProvider for SKY130LeafPins {
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
        _pin_idx: Option<isize>,
    ) -> Direction {
        let cell_type = extract_cell_type(macro_name);
        let pin = pin_name.as_str();

        match cell_type {
            // AIGPDK INV cell (synthesized from assign ~expr): A -> Y
            "INV" => match pin {
                "A" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Inverters: A -> Y
            t if t.starts_with("inv") => match pin {
                "A" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Buffers: A -> X
            t if t.starts_with("buf") || t.starts_with("clkbuf") || t.starts_with("clkdlybuf") => {
                match pin {
                    "A" => Direction::I,
                    "X" => Direction::O,
                    _ => unknown_pin(macro_name, pin_name),
                }
            }

            // Buffer-inverter: A -> Y
            t if t.starts_with("bufinv") => match pin {
                "A" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Clock inverters: A -> Y
            t if t.starts_with("clkinv") => match pin {
                "A" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Delay gate/metal delay: A -> X
            t if t.starts_with("dlygate") || t.starts_with("dlymetal") => match pin {
                "A" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Low-power isolation buffers: A, SLEEP/SLEEP_B -> X
            t if t.starts_with("lpflow_isobufsrc") => match pin {
                "A" | "SLEEP" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },
            t if t.starts_with("lpflow_inputiso") => match pin {
                "A" | "SLEEP" | "SLEEP_B" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 2-input AND: A, B -> X (and2b has A_N or B_N)
            t if t.starts_with("and2") => match pin {
                "A" | "A_N" | "B" | "B_N" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 3-input AND: A, B, C -> X (and3b has _N variants)
            t if t.starts_with("and3") => match pin {
                "A" | "A_N" | "B" | "B_N" | "C" | "C_N" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 4-input AND: A, B, C, D -> X (and4b/and4bb have _N variants)
            t if t.starts_with("and4") => match pin {
                "A" | "A_N" | "B" | "B_N" | "C" | "C_N" | "D" | "D_N" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 2-input OR: A, B -> X (or2b has _N variants)
            t if t.starts_with("or2") => match pin {
                "A" | "A_N" | "B" | "B_N" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 3-input OR: A, B, C -> X (or3b has C_N instead of C)
            t if t.starts_with("or3") => match pin {
                "A" | "A_N" | "B" | "B_N" | "C" | "C_N" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 4-input OR: A, B, C, D -> X (or4b/or4bb have _N variants)
            t if t.starts_with("or4") => match pin {
                "A" | "A_N" | "B" | "B_N" | "C" | "C_N" | "D" | "D_N" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 2-input NAND: A, B -> Y (nand2b has A_N or B_N)
            t if t.starts_with("nand2") => match pin {
                "A" | "A_N" | "B" | "B_N" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 3-input NAND: A, B, C -> Y (nand3b has A_N)
            t if t.starts_with("nand3") => match pin {
                "A" | "A_N" | "B" | "B_N" | "C" | "C_N" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 4-input NAND: A, B, C, D -> Y (nand4b/nand4bb have _N variants)
            t if t.starts_with("nand4") => match pin {
                "A" | "A_N" | "B" | "B_N" | "C" | "C_N" | "D" | "D_N" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 2-input NOR: A, B -> Y (nor2b has A_N or B_N)
            t if t.starts_with("nor2") => match pin {
                "A" | "A_N" | "B" | "B_N" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 3-input NOR: A, B, C -> Y (nor3b has C_N)
            t if t.starts_with("nor3") => match pin {
                "A" | "A_N" | "B" | "B_N" | "C" | "C_N" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 4-input NOR: A, B, C, D -> Y (nor4b/nor4bb have _N variants)
            t if t.starts_with("nor4") => match pin {
                "A" | "A_N" | "B" | "B_N" | "C" | "C_N" | "D" | "D_N" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Majority gate: A, B, C -> X
            t if t.starts_with("maj3") => match pin {
                "A" | "B" | "C" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 2-input XOR: A, B -> X
            t if t.starts_with("xor2") => match pin {
                "A" | "B" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 3-input XOR: A, B, C -> X
            t if t.starts_with("xor3") => match pin {
                "A" | "B" | "C" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 2-input XNOR: A, B -> Y
            t if t.starts_with("xnor2") => match pin {
                "A" | "B" => Direction::I,
                "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // 3-input XNOR: A, B, C -> X
            t if t.starts_with("xnor3") => match pin {
                "A" | "B" | "C" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // MUX2 (mux2, mux2i): A0, A1, S -> X or Y
            t if t.starts_with("mux2") => match pin {
                "A0" | "A1" | "S" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // MUX4: A0, A1, A2, A3, S0, S1 -> X
            t if t.starts_with("mux4") => match pin {
                "A0" | "A1" | "A2" | "A3" | "S0" | "S1" => Direction::I,
                "X" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // D flip-flop (dfxtp): CLK, D -> Q
            t if t == "dfxtp" => match pin {
                "CLK" | "D" => Direction::I,
                "Q" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // D flip-flop with reset (dfrtp, dfrbp): CLK, D, RESET_B -> Q
            t if t == "dfrtp" || t == "dfrbp" => match pin {
                "CLK" | "D" | "RESET_B" => Direction::I,
                "Q" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // D flip-flop with set (dfstp): CLK, D, SET_B -> Q
            t if t == "dfstp" => match pin {
                "CLK" | "D" | "SET_B" => Direction::I,
                "Q" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // D flip-flop with set and reset (dfbbp): CLK, D, RESET_B, SET_B -> Q, Q_N
            t if t == "dfbbp" => match pin {
                "CLK" | "D" | "RESET_B" | "SET_B" => Direction::I,
                "Q" | "Q_N" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // D flip-flop with enable (edfxtp): CLK, D, DE -> Q
            t if t == "edfxtp" => match pin {
                "CLK" | "D" | "DE" => Direction::I,
                "Q" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Scan D flip-flop (sdfxtp): CLK, D, SCD, SCE -> Q
            t if t == "sdfxtp" => match pin {
                "CLK" | "D" | "SCD" | "SCE" => Direction::I,
                "Q" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Latch (dlxtp): D, GATE -> Q
            t if t.starts_with("dlxtp") || t.starts_with("dlat") => match pin {
                "D" | "GATE" => Direction::I,
                "Q" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Tie cells (conb): -> HI, LO
            t if t == "conb" => match pin {
                "HI" | "LO" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Half adder: A, B -> COUT, SUM
            t if t == "ha" => match pin {
                "A" | "B" => Direction::I,
                "COUT" | "SUM" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Full adder: A, B, CIN -> COUT, SUM
            t if t == "fa" => match pin {
                "A" | "B" | "CIN" => Direction::I,
                "COUT" | "SUM" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // AOI cells - ORDER MATTERS: more specific patterns first!

            // a2111oi: Y = !((A1 & A2) | B1 | C1 | D1)
            t if t.starts_with("a2111") => match pin {
                "A1" | "A2" | "B1" | "C1" | "D1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a211oi: Y = !((A1 & A2) | B1 | C1)
            t if t.starts_with("a211") => match pin {
                "A1" | "A2" | "B1" | "C1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a221oi: Y = !((A1 & A2) | (B1 & B2) | C1)
            t if t.starts_with("a221") => match pin {
                "A1" | "A2" | "B1" | "B2" | "C1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a222oi: Y = !((A1 & A2) | (B1 & B2) | (C1 & C2))
            t if t.starts_with("a222") => match pin {
                "A1" | "A2" | "B1" | "B2" | "C1" | "C2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a22oi: Y = !((A1 & A2) | (B1 & B2))
            t if t.starts_with("a22") => match pin {
                "A1" | "A2" | "B1" | "B2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a2bb2oi: Y = !((!A1_N & !A2_N) | (B1 & B2))
            t if t.starts_with("a2bb2") => match pin {
                "A1_N" | "A2_N" | "B1" | "B2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a21oi Y = !((A1 & A2) | B1)
            // a21o  Y = (A1 & A2) | B1
            // a21boi: Y = !((A1 & A2) | !B1_N) = !((A1 & A2) | B1) with B1 pre-inverted
            t if t.starts_with("a21") => match pin {
                "A1" | "A2" | "B1" | "B1_N" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a311oi: Y = !((A1 & A2 & A3) | B1 | C1)
            t if t.starts_with("a311") => match pin {
                "A1" | "A2" | "A3" | "B1" | "C1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a31oi: Y = !((A1 & A2 & A3) | B1)
            t if t.starts_with("a31") => match pin {
                "A1" | "A2" | "A3" | "B1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a32oi: Y = !((A1 & A2 & A3) | (B1 & B2))
            t if t.starts_with("a32") => match pin {
                "A1" | "A2" | "A3" | "B1" | "B2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // a41oi: Y = !((A1 & A2 & A3 & A4) | B1)
            t if t.starts_with("a41") => match pin {
                "A1" | "A2" | "A3" | "A4" | "B1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // OAI cells - ORDER MATTERS: more specific patterns first!

            // o2111ai: Y = !((A1 | A2) & B1 & C1 & D1)
            t if t.starts_with("o2111") => match pin {
                "A1" | "A2" | "B1" | "C1" | "D1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o211ai: Y = !((A1 | A2) & B1 & C1)
            t if t.starts_with("o211") => match pin {
                "A1" | "A2" | "B1" | "C1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o221ai: Y = !((A1 | A2) & (B1 | B2) & C1)
            t if t.starts_with("o221") => match pin {
                "A1" | "A2" | "B1" | "B2" | "C1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o22ai: Y = !((A1 | A2) & (B1 | B2))
            t if t.starts_with("o22") => match pin {
                "A1" | "A2" | "B1" | "B2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o2bb2ai: Y = !((!A1_N | !A2_N) & (B1 | B2))
            t if t.starts_with("o2bb2") => match pin {
                "A1_N" | "A2_N" | "B1" | "B2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o21ai Y = !((A1 | A2) & B1)
            // o21a  Y = (A1 | A2) & B1
            // o21bai: Y = !((A1 | A2) & !B1_N)
            t if t.starts_with("o21") => match pin {
                "A1" | "A2" | "B1" | "B1_N" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o311ai: Y = !((A1 | A2 | A3) & B1 & C1)
            t if t.starts_with("o311") => match pin {
                "A1" | "A2" | "A3" | "B1" | "C1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o31ai: Y = !((A1 | A2 | A3) & B1)
            t if t.starts_with("o31") => match pin {
                "A1" | "A2" | "A3" | "B1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o32ai: Y = !((A1 | A2 | A3) & (B1 | B2))
            t if t.starts_with("o32") => match pin {
                "A1" | "A2" | "A3" | "B1" | "B2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o41ai: Y = !((A1 | A2 | A3 | A4) & B1)
            t if t.starts_with("o41") => match pin {
                "A1" | "A2" | "A3" | "A4" | "B1" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Complex BB cells (two inverted inputs)
            // a2bb2oi: Y = !((!A1_N & !A2_N) | (B1 & B2)) = !((A1_N | A2_N) | (B1 & B2))
            t if t.starts_with("a2bb2o") => match pin {
                "A1_N" | "A2_N" | "B1" | "B2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // o2bb2ai: Y = !((!A1_N | !A2_N) & (B1 | B2)) = !((A1_N & A2_N) & (B1 | B2))
            t if t.starts_with("o2bb2a") => match pin {
                "A1_N" | "A2_N" | "B1" | "B2" => Direction::I,
                "X" | "Y" => Direction::O,
                _ => unknown_pin(macro_name, pin_name),
            },

            // Filler/tap/decap/diode cells - no functional pins
            t if t.starts_with("fill") || t.starts_with("tap") || t.starts_with("decap") || t.starts_with("diode") => {
                // These cells have no functional pins, but may have power pins
                // which we treat as inputs (though they're typically not connected in logic)
                Direction::I
            }

            _ => {
                // Check for ChipFlow SRAM macros (CF_SRAM_*)
                if macro_name.starts_with("CF_SRAM_") {
                    match pin {
                        // Control inputs
                        "WLBI" | "WLOFF" | "CLKin" | "EN" | "R_WB" | "SM" | "TM" => Direction::I,
                        "ScanInDR" | "ScanInDL" | "ScanInCC" => Direction::I,
                        "vpwrpc" | "vpwrac" => Direction::I,
                        // Data/address inputs
                        "AD" | "BEN" | "DI" => Direction::I,
                        // Data/scan outputs
                        "DO" | "ScanOutCC" => Direction::O,
                        _ => unknown_pin(macro_name, pin_name),
                    }
                } else {
                    unknown_pin(macro_name, pin_name)
                }
            }
        }
    }

    fn width_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
    ) -> Option<SVerilogRange> {
        // Check for ChipFlow SRAM macros which have bus pins
        if macro_name.starts_with("CF_SRAM_") {
            match pin_name.as_str() {
                // Data buses are 32-bit
                "DO" | "DI" | "BEN" => Some(SVerilogRange(31, 0)),
                // Address bus - width depends on SRAM depth
                // CF_SRAM_1024x32 has 10-bit address (2^10 = 1024)
                "AD" => Some(SVerilogRange(9, 0)),
                // All other pins are single-bit
                _ => None,
            }
        } else {
            // All SKY130 standard cells have single-bit pins
            None
        }
    }
}

/// Extract the cell type from a SKY130 macro name.
///
/// Converts `sky130_fd_sc_hd__inv_2` to `inv`.
pub fn extract_cell_type(name: &str) -> &str {
    // First strip the library prefix
    let stripped = name
        .strip_prefix("sky130_fd_sc_hd__")
        .or_else(|| name.strip_prefix("sky130_fd_sc_hs__"))
        .or_else(|| name.strip_prefix("sky130_fd_sc_ms__"))
        .or_else(|| name.strip_prefix("sky130_fd_sc_ls__"))
        .or_else(|| name.strip_prefix("sky130_fd_sc_lp__"))
        .or_else(|| name.strip_prefix("sky130_fd_sc_hdll__"))
        .or_else(|| name.strip_prefix("sky130_fd_sc_hvl__"))
        .unwrap_or(name);

    // Then strip the drive strength suffix (e.g., _2, _4, _16)
    // The drive strength is always a number at the end after an underscore
    if let Some(last_underscore) = stripped.rfind('_') {
        let suffix = &stripped[last_underscore + 1..];
        // Check if suffix is purely numeric (drive strength)
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return &stripped[..last_underscore];
        }
    }

    stripped
}

/// Return Direction::I as fallback but log an error.
fn unknown_pin(macro_name: &CompactString, pin_name: &CompactString) -> Direction {
    panic!("Unknown SKY130 pin: macro={}, pin={}", macro_name, pin_name);
}

/// Check if a cell name is a SKY130 standard cell or associated macro.
pub fn is_sky130_cell(name: &str) -> bool {
    name.starts_with("sky130_fd_sc_") || name.starts_with("CF_SRAM_")
}

/// Check if a cell name is an AIGPDK cell.
pub fn is_aigpdk_cell(name: &str) -> bool {
    matches!(
        name,
        "INV"
            | "BUF"
            | "CKLNQD"
            | "AND2_00_0"
            | "AND2_01_0"
            | "AND2_10_0"
            | "AND2_11_0"
            | "AND2_11_1"
            | "DFF"
            | "DFFSR"
            | "LATCH"
            | "GEM_ASSERT"
            | "GEM_DISPLAY"
    ) || name.starts_with("$__RAMGEM_")
}

/// The detected cell library type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellLibrary {
    /// AIGPDK standard cells
    AIGPDK,
    /// SKY130 open-source PDK
    SKY130,
    /// Mixed cells from both libraries (error case)
    Mixed,
}

impl std::fmt::Display for CellLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellLibrary::AIGPDK => write!(f, "AIGPDK"),
            CellLibrary::SKY130 => write!(f, "SKY130"),
            CellLibrary::Mixed => write!(f, "Mixed"),
        }
    }
}

/// Detect which cell library is used in a netlist.
///
/// Scans all cells in the netlist and returns:
/// - `AIGPDK` if only AIGPDK cells are found
/// - `SKY130` if only SKY130 cells are found
/// - `Mixed` if both types are found (this is an error condition)
///
/// If no recognizable cells are found, defaults to AIGPDK.
pub fn detect_library(cell_types: impl Iterator<Item = impl AsRef<str>>) -> CellLibrary {
    let mut has_aigpdk = false;
    let mut has_sky130 = false;

    for cell_type in cell_types {
        let name = cell_type.as_ref();
        if is_sky130_cell(name) {
            has_sky130 = true;
        } else if is_aigpdk_cell(name) {
            has_aigpdk = true;
        }
        // Early exit if we've found both
        if has_aigpdk && has_sky130 {
            return CellLibrary::Mixed;
        }
    }

    match (has_aigpdk, has_sky130) {
        (_, true) => CellLibrary::SKY130,
        (true, false) => CellLibrary::AIGPDK,
        (false, false) => CellLibrary::AIGPDK, // Default
    }
}

/// Detect library from a Verilog file by scanning for cell types.
///
/// This does a quick scan of the file without full parsing.
pub fn detect_library_from_file(path: &std::path::Path) -> std::io::Result<CellLibrary> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut has_aigpdk = false;
    let mut has_sky130 = false;

    for line in reader.lines() {
        let line = line?;
        // Quick pattern matching for cell instantiations
        if line.contains("sky130_fd_sc_") {
            has_sky130 = true;
        }
        // Look for AIGPDK cells (typically at start of instantiation)
        for cell in [
            "AND2_",
            "INV ",
            "BUF ",
            "DFF ",
            "DFFSR ",
            "LATCH ",
            "CKLNQD ",
            "GEM_ASSERT ",
            "GEM_DISPLAY ",
            "$__RAMGEM_",
        ] {
            if line.contains(cell) {
                has_aigpdk = true;
                break;
            }
        }
        // Early exit if mixed
        if has_aigpdk && has_sky130 {
            return Ok(CellLibrary::Mixed);
        }
    }

    Ok(match (has_aigpdk, has_sky130) {
        (_, true) => CellLibrary::SKY130,
        (true, false) => CellLibrary::AIGPDK,
        (false, false) => CellLibrary::AIGPDK,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_cell_type() {
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__inv_2"), "inv");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__nand2_4"), "nand2");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__dfxtp_1"), "dfxtp");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__a21oi_1"), "a21oi");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__o21ai_0"), "o21ai");
        assert_eq!(extract_cell_type("sky130_fd_sc_hs__buf_16"), "buf");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__edfxtp_1"), "edfxtp");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__fa_1"), "fa");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__ha_1"), "ha");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__mux2i_1"), "mux2i");
        assert_eq!(extract_cell_type("sky130_fd_sc_hd__conb_1"), "conb");
    }

    #[test]
    fn test_is_sky130_cell() {
        assert!(is_sky130_cell("sky130_fd_sc_hd__inv_2"));
        assert!(is_sky130_cell("sky130_fd_sc_hs__nand2_4"));
        assert!(!is_sky130_cell("AND2_00_0"));
        assert!(!is_sky130_cell("INV"));
    }

    #[test]
    fn test_pin_directions() {
        let provider = SKY130LeafPins;

        // Test inverter
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__inv_2".into(), &"A".into(), None),
            Direction::I
        );
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__inv_2".into(), &"Y".into(), None),
            Direction::O
        );

        // Test buffer
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__buf_4".into(), &"A".into(), None),
            Direction::I
        );
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__buf_4".into(), &"X".into(), None),
            Direction::O
        );

        // Test NAND2
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__nand2_1".into(), &"A".into(), None),
            Direction::I
        );
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__nand2_1".into(), &"B".into(), None),
            Direction::I
        );
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__nand2_1".into(), &"Y".into(), None),
            Direction::O
        );

        // Test DFF
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__dfxtp_1".into(), &"CLK".into(), None),
            Direction::I
        );
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__dfxtp_1".into(), &"D".into(), None),
            Direction::I
        );
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__dfxtp_1".into(), &"Q".into(), None),
            Direction::O
        );

        // Test conb
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__conb_1".into(), &"HI".into(), None),
            Direction::O
        );
        assert_eq!(
            provider.direction_of(&"sky130_fd_sc_hd__conb_1".into(), &"LO".into(), None),
            Direction::O
        );
    }

    #[test]
    fn test_is_aigpdk_cell() {
        assert!(is_aigpdk_cell("AND2_00_0"));
        assert!(is_aigpdk_cell("INV"));
        assert!(is_aigpdk_cell("BUF"));
        assert!(is_aigpdk_cell("DFF"));
        assert!(is_aigpdk_cell("DFFSR"));
        assert!(is_aigpdk_cell("$__RAMGEM_SYNC_"));
        assert!(!is_aigpdk_cell("sky130_fd_sc_hd__inv_2"));
    }

    #[test]
    fn test_detect_library() {
        // SKY130 only
        let sky130_cells = ["sky130_fd_sc_hd__inv_2", "sky130_fd_sc_hd__nand2_1"];
        assert_eq!(detect_library(sky130_cells.iter()), CellLibrary::SKY130);

        // AIGPDK only
        let aigpdk_cells = ["AND2_00_0", "INV", "DFF"];
        assert_eq!(detect_library(aigpdk_cells.iter()), CellLibrary::AIGPDK);

        // Mixed
        let mixed_cells = ["sky130_fd_sc_hd__inv_2", "AND2_00_0"];
        assert_eq!(detect_library(mixed_cells.iter()), CellLibrary::Mixed);

        // Empty defaults to AIGPDK
        let empty: Vec<&str> = vec![];
        assert_eq!(detect_library(empty.iter()), CellLibrary::AIGPDK);
    }
}
