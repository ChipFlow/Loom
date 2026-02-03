// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! SKY130 cell decomposition to AIG primitives.
//!
//! This module decomposes SKY130 standard cells into And-Inverter Graph (AIG)
//! representations. Each complex cell is broken down into basic operations:
//! - AND gates with optional input inversions
//! - Passthrough with optional output inversion
//!
//! The decomposition uses De Morgan's laws to express OR as AND with inversions:
//! - OR(A,B) = !(!A & !B)
//! - NOR(A,B) = !A & !B
//! - NAND(A,B) = !(A & B)


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

    /// Get input, panicking if not set.
    #[inline]
    #[allow(dead_code)]
    fn get(&self, field: usize) -> usize {
        assert_ne!(field, usize::MAX, "Required input pin not set");
        field
    }

    /// Get input or return alternative if not set.
    #[inline]
    #[allow(dead_code)]
    fn get_or(&self, field: usize, alt: usize) -> usize {
        if field == usize::MAX { alt } else { field }
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

/// An input reference that can be either:
/// - A positive index referencing an original input (aigpin_iv with inversion bit)
/// - A negative index referencing an intermediate AND gate output (-1 = first gate)
type InputRef = i64;

impl DecompResult {
    /// Create a simple passthrough (no AND gates, just pass input with optional inversion)
    fn passthrough(input_idx: usize, inverted: bool) -> Self {
        DecompResult {
            and_gates: vec![],
            output_idx: input_idx as i64,
            output_inverted: inverted,
        }
    }

    /// Create a single AND gate result
    fn single_and(a: InputRef, b: InputRef, output_inverted: bool) -> Self {
        DecompResult {
            and_gates: vec![(a, b)],
            output_idx: -1,
            output_inverted,
        }
    }
}

/// Decompose a SKY130 cell into AIG operations.
///
/// # Arguments
/// * `cell_type` - The pre-extracted cell type (e.g., "inv", "nand2", "a21oi")
/// * `inputs` - CellInputs struct with pin values collected from netlist
///
/// # Returns
/// A `DecompResult` describing how to build the AIG for this cell
pub fn decompose_sky130_cell(
    cell_type: &str,
    inputs: &CellInputs,
) -> DecompResult {
    match cell_type {
        // === Simple gates ===

        // Inverter: Y = !A
        "inv" | "clkinv" | "clkinvlp" => {
            let a = inputs.a;
            DecompResult::passthrough(a >> 1, (a & 1) == 0) // flip inversion
        }

        // Buffer: X = A
        "buf" | "clkbuf" | "clkdlybuf4s50" | "dlygate4sd3" => {
            let a = inputs.a;
            DecompResult::passthrough(a >> 1, (a & 1) != 0)
        }

        // Buffer-inverter (bufinv): Y = !A
        "bufinv" => {
            let a = inputs.a;
            DecompResult::passthrough(a >> 1, (a & 1) == 0)
        }

        // 2-input AND: X = A & B
        "and2" => {
            let a = inputs.a;
            let b = inputs.b;
            DecompResult::single_and(a as i64, b as i64, false)
        }

        // 3-input AND: X = A & B & C
        "and3" => {
            let a = inputs.a;
            let b = inputs.b;
            let c = inputs.c;
            DecompResult {
                and_gates: vec![
                    (a as i64, b as i64),  // gate -1 = A & B
                    (-1, c as i64),        // gate -2 = (A & B) & C
                ],
                output_idx: -2,
                output_inverted: false,
            }
        }

        // 4-input AND: X = A & B & C & D
        "and4" => {
            let a = inputs.a;
            let b = inputs.b;
            let c = inputs.c;
            let d = inputs.d;
            DecompResult {
                and_gates: vec![
                    (a as i64, b as i64),  // gate -1 = A & B
                    (c as i64, d as i64),  // gate -2 = C & D
                    (-1, -2),              // gate -3 = (A & B) & (C & D)
                ],
                output_idx: -3,
                output_inverted: false,
            }
        }

        // 2-input OR: X = A | B = !(!A & !B)
        "or2" => {
            let a = inputs.a;
            let b = inputs.b;
            // OR = !(!A & !B), so invert inputs, AND them, invert output
            let a_inv = (a ^ 1) as i64;  // toggle inversion
            let b_inv = (b ^ 1) as i64;
            DecompResult::single_and(a_inv, b_inv, true) // output inverted
        }

        // 3-input OR: X = A | B | C = !(!A & !B & !C)
        "or3" => {
            let a = inputs.a;
            let b = inputs.b;
            let c = inputs.c;
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            let c_inv = (c ^ 1) as i64;
            DecompResult {
                and_gates: vec![
                    (a_inv, b_inv),  // gate -1 = !A & !B
                    (-1, c_inv),     // gate -2 = (!A & !B) & !C
                ],
                output_idx: -2,
                output_inverted: true,
            }
        }

        // or3b: X = A | B | !C_N = !(!A & !B & C_N)
        // SKY130 or3b has C_N instead of C (inverted third input)
        "or3b" => {
            let a = inputs.a;
            let b = inputs.b;
            let c_n = inputs.c_n;  // C_N is already inverted
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            DecompResult {
                and_gates: vec![
                    (a_inv, b_inv),      // gate -1 = !A & !B
                    (-1, c_n as i64),    // gate -2 = (!A & !B) & C_N
                ],
                output_idx: -2,
                output_inverted: true,
            }
        }

        // 4-input OR: X = A | B | C | D
        "or4" => {
            let a = inputs.a;
            let b = inputs.b;
            let c = inputs.c;
            let d = inputs.d;
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            let c_inv = (c ^ 1) as i64;
            let d_inv = (d ^ 1) as i64;
            DecompResult {
                and_gates: vec![
                    (a_inv, b_inv),  // gate -1 = !A & !B
                    (c_inv, d_inv),  // gate -2 = !C & !D
                    (-1, -2),        // gate -3 = (!A & !B) & (!C & !D)
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // 2-input NAND: Y = !(A & B)
        "nand2" => {
            let a = inputs.a;
            let b = inputs.b;
            DecompResult::single_and(a as i64, b as i64, true)
        }

        // 3-input NAND: Y = !(A & B & C)
        "nand3" => {
            let a = inputs.a;
            let b = inputs.b;
            let c = inputs.c;
            DecompResult {
                and_gates: vec![
                    (a as i64, b as i64),   // gate -1 = A & B
                    (-1, c as i64),         // gate -2 = (A & B) & C
                ],
                output_idx: -2,
                output_inverted: true,
            }
        }
        // nand3b: Y = !(!A_N & B & C) where A_N is the inverted input
        "nand3b" => {
            let a_n = inputs.a_n;  // This is the inverted input pin
            let b = inputs.b;
            let c = inputs.c;
            // !A_N gives the effective A value
            DecompResult {
                and_gates: vec![
                    ((a_n ^ 1) as i64, b as i64),   // gate -1 = !A_N & B
                    (-1, c as i64),                  // gate -2 = (!A_N & B) & C
                ],
                output_idx: -2,
                output_inverted: true,
            }
        }

        // 4-input NAND: Y = !(A & B & C & D)
        "nand4" => {
            let a = inputs.a;
            let b = inputs.b;
            let c = inputs.c;
            let d = inputs.d;
            DecompResult {
                and_gates: vec![
                    (a as i64, b as i64),  // gate -1 = A & B
                    (c as i64, d as i64),  // gate -2 = C & D
                    (-1, -2),              // gate -3 = (A & B) & (C & D)
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // 2-input NOR: Y = !(A | B) = !A & !B
        "nor2" => {
            let a = inputs.a;
            let b = inputs.b;
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            DecompResult::single_and(a_inv, b_inv, false)
        }

        // 3-input NOR: Y = !(A | B | C) = !A & !B & !C
        "nor3" => {
            let a = inputs.a;
            let b = inputs.b;
            let c = inputs.c;
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            let c_inv = (c ^ 1) as i64;
            DecompResult {
                and_gates: vec![
                    (a_inv, b_inv),  // gate -1 = !A & !B
                    (-1, c_inv),     // gate -2 = (!A & !B) & !C
                ],
                output_idx: -2,
                output_inverted: false,
            }
        }

        // nor3b: Y = !(A | B | !C_N) = !A & !B & C_N
        // SKY130 nor3b has C_N instead of C (inverted third input)
        "nor3b" => {
            let a = inputs.a;
            let b = inputs.b;
            let c_n = inputs.c_n;  // C_N is already inverted, use as-is
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            DecompResult {
                and_gates: vec![
                    (a_inv, b_inv),      // gate -1 = !A & !B
                    (-1, c_n as i64),    // gate -2 = (!A & !B) & C_N
                ],
                output_idx: -2,
                output_inverted: false,
            }
        }

        // 4-input NOR: Y = !(A | B | C | D) = !A & !B & !C & !D
        "nor4" => {
            let a = inputs.a;
            let b = inputs.b;
            let c = inputs.c;
            let d = inputs.d;
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            let c_inv = (c ^ 1) as i64;
            let d_inv = (d ^ 1) as i64;
            DecompResult {
                and_gates: vec![
                    (a_inv, b_inv),  // gate -1 = !A & !B
                    (c_inv, d_inv),  // gate -2 = !C & !D
                    (-1, -2),        // gate -3 = (!A & !B) & (!C & !D)
                ],
                output_idx: -3,
                output_inverted: false,
            }
        }

        // 2-input XOR: X = A ^ B = (A & !B) | (!A & B) = !(!A & !B) & !(A & B) | ...
        // Better: XOR = (A | B) & !(A & B) = (A | B) & (!A | !B)
        // But we need pure AND/INV. Use: A ^ B = !(!(A & !B) & !(!A & B))
        "xor2" => {
            let a = inputs.a;
            let b = inputs.b;
            // XOR = (A & !B) | (!A & B)
            // Using De Morgan: = !(!( A & !B) & !(!A & B))
            let a_iv = a as i64;
            let b_iv = b as i64;
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            DecompResult {
                and_gates: vec![
                    (a_iv, b_inv),   // gate -1 = A & !B
                    (a_inv, b_iv),   // gate -2 = !A & B
                    (-1 ^ 1, -2 ^ 1), // gate -3 = !(A & !B) & !(!A & B)
                ],
                output_idx: -3,
                output_inverted: true, // Final inversion for OR
            }
        }

        // 2-input XNOR: Y = !(A ^ B) = (A & B) | (!A & !B)
        "xnor2" => {
            let a = inputs.a;
            let b = inputs.b;
            let a_iv = a as i64;
            let b_iv = b as i64;
            let a_inv = (a ^ 1) as i64;
            let b_inv = (b ^ 1) as i64;
            // XNOR = (A & B) | (!A & !B)
            // = !( !(A & B) & !(!A & !B) )
            DecompResult {
                and_gates: vec![
                    (a_iv, b_iv),     // gate -1 = A & B
                    (a_inv, b_inv),   // gate -2 = !A & !B
                    (-1 ^ 1, -2 ^ 1), // gate -3 = !(A & B) & !(!A & !B)
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // === MUX cells ===

        // MUX2: Y = S ? A1 : A0 = (A0 & !S) | (A1 & S)
        "mux2" => {
            let a0 = inputs.a0;
            let a1 = inputs.a1;
            let s = inputs.s;
            let s_inv = (s ^ 1) as i64;
            // (A0 & !S) | (A1 & S) = !( !(A0 & !S) & !(A1 & S) )
            DecompResult {
                and_gates: vec![
                    (a0 as i64, s_inv),  // gate -1 = A0 & !S
                    (a1 as i64, s as i64), // gate -2 = A1 & S
                    (-1 ^ 1, -2 ^ 1),    // gate -3 = !(A0 & !S) & !(A1 & S)
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // MUX2I (inverted output): Y = !(S ? A1 : A0)
        "mux2i" => {
            let a0 = inputs.a0;
            let a1 = inputs.a1;
            let s = inputs.s;
            let s_inv = (s ^ 1) as i64;
            DecompResult {
                and_gates: vec![
                    (a0 as i64, s_inv),
                    (a1 as i64, s as i64),
                    (-1 ^ 1, -2 ^ 1),
                ],
                output_idx: -3,
                output_inverted: false,  // No final inversion = inverted MUX
            }
        }

        // === AOI cells (And-Or-Invert) ===

        // a21oi: Y = !((A1 & A2) | B1)
        "a21oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            // Y = !((A1 & A2) | B1) = !(A1 & A2) & !B1
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),  // gate -1 = A1 & A2
                    (-1 ^ 1, (b1 ^ 1) as i64), // gate -2 = !(A1 & A2) & !B1
                ],
                output_idx: -2,
                output_inverted: false,
            }
        }

        // a21o: Y = (A1 & A2) | B1
        "a21o" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (-1 ^ 1, (b1 ^ 1) as i64),
                ],
                output_idx: -2,
                output_inverted: true, // Invert NOR to get OR
            }
        }

        // a21boi: Y = !((A1 & A2) | !B1_N)
        "a21boi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1_n = inputs.b1_n;
            // B1 = !B1_N, so we use B1_N directly (already inverted)
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (-1 ^ 1, b1_n as i64), // !B1_N is B1, but input is B1_N = !B1
                ],
                output_idx: -2,
                output_inverted: false,
            }
        }

        // a22oi: Y = !((A1 & A2) | (B1 & B2))
        "a22oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),  // gate -1 = A1 & A2
                    (b1 as i64, b2 as i64),  // gate -2 = B1 & B2
                    (-1 ^ 1, -2 ^ 1),        // gate -3 = !(A1&A2) & !(B1&B2)
                ],
                output_idx: -3,
                output_inverted: false,
            }
        }

        // a22o: Y = (A1 & A2) | (B1 & B2)
        "a22o" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (b1 as i64, b2 as i64),
                    (-1 ^ 1, -2 ^ 1),
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // a211oi: Y = !((A1 & A2) | B1 | C1)
        "a211oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let c1 = inputs.c1;
            // = !(A1 & A2) & !B1 & !C1
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),         // -1 = A1 & A2
                    (-1 ^ 1, (b1 ^ 1) as i64),      // -2 = !(A1&A2) & !B1
                    (-2, (c1 ^ 1) as i64),          // -3 = above & !C1
                ],
                output_idx: -3,
                output_inverted: false,
            }
        }

        // a221oi: Y = !((A1 & A2) | (B1 & B2) | C1)
        "a221oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            let c1 = inputs.c1;
            // = !(A1&A2) & !(B1&B2) & !C1
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),    // -1 = A1 & A2
                    (b1 as i64, b2 as i64),    // -2 = B1 & B2
                    (-1 ^ 1, -2 ^ 1),          // -3 = !(A1&A2) & !(B1&B2)
                    (-3, (c1 ^ 1) as i64),     // -4 = above & !C1
                ],
                output_idx: -4,
                output_inverted: false,
            }
        }

        // a221o: Y = (A1 & A2) | (B1 & B2) | C1
        "a221o" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            let c1 = inputs.c1;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (b1 as i64, b2 as i64),
                    (-1 ^ 1, -2 ^ 1),
                    (-3, (c1 ^ 1) as i64),
                ],
                output_idx: -4,
                output_inverted: true,
            }
        }

        // a2111oi: Y = !((A1 & A2) | B1 | C1 | D1)
        "a2111oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let c1 = inputs.c1;
            let d1 = inputs.d1;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),         // -1 = A1 & A2
                    (-1 ^ 1, (b1 ^ 1) as i64),      // -2 = !(A1&A2) & !B1
                    (-2, (c1 ^ 1) as i64),          // -3 = above & !C1
                    (-3, (d1 ^ 1) as i64),          // -4 = above & !D1
                ],
                output_idx: -4,
                output_inverted: false,
            }
        }

        // a2bb2oi: Y = !((!A1_N & !A2_N) | (B1 & B2))
        // Note: A1_N and A2_N are the inverted inputs, so !A1_N is A1
        "a2bb2oi" => {
            let a1_n = inputs.a1_n;
            let a2_n = inputs.a2_n;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            // !A1_N = A1, so (!A1_N & !A2_N) = (A1 & A2)
            // We invert A1_N and A2_N
            DecompResult {
                and_gates: vec![
                    ((a1_n ^ 1) as i64, (a2_n ^ 1) as i64),  // -1 = !A1_N & !A2_N
                    (b1 as i64, b2 as i64),                  // -2 = B1 & B2
                    (-1 ^ 1, -2 ^ 1),                        // -3 = NOR
                ],
                output_idx: -3,
                output_inverted: false,
            }
        }

        // a31oi: Y = !((A1 & A2 & A3) | B1)
        "a31oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let b1 = inputs.b1;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),    // -1 = A1 & A2
                    (-1, a3 as i64),           // -2 = A1 & A2 & A3
                    (-2 ^ 1, (b1 ^ 1) as i64), // -3 = !(A1&A2&A3) & !B1
                ],
                output_idx: -3,
                output_inverted: false,
            }
        }

        // a31o: Y = (A1 & A2 & A3) | B1
        "a31o" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let b1 = inputs.b1;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (-1, a3 as i64),
                    (-2 ^ 1, (b1 ^ 1) as i64),
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // a311oi: Y = !((A1 & A2 & A3) | B1 | C1)
        "a311oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let b1 = inputs.b1;
            let c1 = inputs.c1;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (-1, a3 as i64),
                    (-2 ^ 1, (b1 ^ 1) as i64),
                    (-3, (c1 ^ 1) as i64),
                ],
                output_idx: -4,
                output_inverted: false,
            }
        }

        // a32oi: Y = !((A1 & A2 & A3) | (B1 & B2))
        "a32oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (-1, a3 as i64),
                    (b1 as i64, b2 as i64),
                    (-2 ^ 1, -3 ^ 1),
                ],
                output_idx: -4,
                output_inverted: false,
            }
        }

        // a32o: Y = (A1 & A2 & A3) | (B1 & B2)
        "a32o" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (-1, a3 as i64),
                    (b1 as i64, b2 as i64),
                    (-2 ^ 1, -3 ^ 1),
                ],
                output_idx: -4,
                output_inverted: true,
            }
        }

        // a41oi: Y = !((A1 & A2 & A3 & A4) | B1)
        "a41oi" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let a4 = inputs.a4;
            let b1 = inputs.b1;
            DecompResult {
                and_gates: vec![
                    (a1 as i64, a2 as i64),
                    (a3 as i64, a4 as i64),
                    (-1, -2),
                    (-3 ^ 1, (b1 ^ 1) as i64),
                ],
                output_idx: -4,
                output_inverted: false,
            }
        }

        // === OAI cells (Or-And-Invert) ===

        // o21ai: Y = !((A1 | A2) & B1)
        "o21ai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            // (A1 | A2) = !(!A1 & !A2)
            // Y = !((A1 | A2) & B1) = !(A1 | A2) | !B1 = (!A1 & !A2) | !B1
            // = !(!(!A1 & !A2) & B1) ... hmm, let's think differently
            // Actually: !((A1|A2) & B1) = !(A1|A2) | !B1 = (!A1 & !A2) | !B1
            // Build OR first: !A1 & !A2 gives !(A1|A2)
            // Then: result = !(A1|A2) | !B1 = !(!!( !(A1|A2) | !B1)) = !( (A1|A2) & B1 )
            // Direct: Y = !( !(!A1 & !A2) & B1 )
            //           = (!A1 & !A2) | !B1  -- but we need this as AND form
            //           = !( !(!A1 & !A2) & !!B1 ) = !( (A1|A2) & B1 )
            // So: build NOR of A1,A2, then AND with B1, then invert
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),  // -1 = !A1 & !A2 = !(A1|A2)
                    (-1 ^ 1, b1 as i64),                  // -2 = (A1|A2) & B1
                ],
                output_idx: -2,
                output_inverted: true,
            }
        }

        // o21a: Y = (A1 | A2) & B1
        "o21a" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    (-1 ^ 1, b1 as i64),
                ],
                output_idx: -2,
                output_inverted: false,
            }
        }

        // o21bai: Y = !((A1 | A2) & !B1_N)
        "o21bai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1_n = inputs.b1_n;
            // !B1_N = B1
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    (-1 ^ 1, (b1_n ^ 1) as i64),  // (A1|A2) & !B1_N = (A1|A2) & B1
                ],
                output_idx: -2,
                output_inverted: true,
            }
        }

        // o22ai: Y = !((A1 | A2) & (B1 | B2))
        "o22ai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),  // -1 = !(A1|A2)
                    ((b1 ^ 1) as i64, (b2 ^ 1) as i64),  // -2 = !(B1|B2)
                    (-1 ^ 1, -2 ^ 1),                     // -3 = (A1|A2) & (B1|B2)
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // o211ai: Y = !((A1 | A2) & B1 & C1)
        "o211ai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let c1 = inputs.c1;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),  // -1 = !(A1|A2)
                    (-1 ^ 1, b1 as i64),                  // -2 = (A1|A2) & B1
                    (-2, c1 as i64),                      // -3 = above & C1
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // o211a: Y = (A1 | A2) & B1 & C1
        "o211a" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let c1 = inputs.c1;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    (-1 ^ 1, b1 as i64),
                    (-2, c1 as i64),
                ],
                output_idx: -3,
                output_inverted: false,
            }
        }

        // o221ai: Y = !((A1 | A2) & (B1 | B2) & C1)
        "o221ai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            let c1 = inputs.c1;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    ((b1 ^ 1) as i64, (b2 ^ 1) as i64),
                    (-1 ^ 1, -2 ^ 1),
                    (-3, c1 as i64),
                ],
                output_idx: -4,
                output_inverted: true,
            }
        }

        // o221a: Y = (A1 | A2) & (B1 | B2) & C1
        "o221a" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            let c1 = inputs.c1;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    ((b1 ^ 1) as i64, (b2 ^ 1) as i64),
                    (-1 ^ 1, -2 ^ 1),
                    (-3, c1 as i64),
                ],
                output_idx: -4,
                output_inverted: false,
            }
        }

        // o2111ai: Y = !((A1 | A2) & B1 & C1 & D1)
        "o2111ai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let b1 = inputs.b1;
            let c1 = inputs.c1;
            let d1 = inputs.d1;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    (-1 ^ 1, b1 as i64),
                    (-2, c1 as i64),
                    (-3, d1 as i64),
                ],
                output_idx: -4,
                output_inverted: true,
            }
        }

        // o2bb2ai: Y = !((!A1_N | !A2_N) & (B1 | B2))
        "o2bb2ai" => {
            let a1_n = inputs.a1_n;
            let a2_n = inputs.a2_n;
            let b1 = inputs.b1;
            let b2 = inputs.b2;
            // !A1_N | !A2_N = !(A1_N & A2_N)
            DecompResult {
                and_gates: vec![
                    (a1_n as i64, a2_n as i64),           // -1 = A1_N & A2_N = !(!A1_N | !A2_N)
                    ((b1 ^ 1) as i64, (b2 ^ 1) as i64),   // -2 = !(B1|B2)
                    (-1 ^ 1, -2 ^ 1),                      // -3 = (!A1_N|!A2_N) & (B1|B2)
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // o31ai: Y = !((A1 | A2 | A3) & B1)
        "o31ai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let b1 = inputs.b1;
            // A1 | A2 | A3 = !(!A1 & !A2 & !A3)
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    (-1, (a3 ^ 1) as i64),
                    (-2 ^ 1, b1 as i64),
                ],
                output_idx: -3,
                output_inverted: true,
            }
        }

        // o311ai: Y = !((A1 | A2 | A3) & B1 & C1)
        "o311ai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let b1 = inputs.b1;
            let c1 = inputs.c1;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    (-1, (a3 ^ 1) as i64),
                    (-2 ^ 1, b1 as i64),
                    (-3, c1 as i64),
                ],
                output_idx: -4,
                output_inverted: true,
            }
        }

        // o41ai: Y = !((A1 | A2 | A3 | A4) & B1)
        "o41ai" => {
            let a1 = inputs.a1;
            let a2 = inputs.a2;
            let a3 = inputs.a3;
            let a4 = inputs.a4;
            let b1 = inputs.b1;
            DecompResult {
                and_gates: vec![
                    ((a1 ^ 1) as i64, (a2 ^ 1) as i64),
                    ((a3 ^ 1) as i64, (a4 ^ 1) as i64),
                    (-1, -2),
                    (-3 ^ 1, b1 as i64),
                ],
                output_idx: -4,
                output_inverted: true,
            }
        }

        // === Arithmetic cells ===

        // Half adder: SUM = A ^ B, COUT = A & B
        // Returns two outputs, handled specially by AIG builder
        "ha" => {
            // This cell has two outputs - handled specially
            panic!(
                "Half adder (ha) has multiple outputs and must be handled specially in AIG builder"
            );
        }

        // Full adder: SUM = A ^ B ^ CIN, COUT = (A & B) | (A & CIN) | (B & CIN)
        "fa" => {
            panic!(
                "Full adder (fa) has multiple outputs and must be handled specially in AIG builder"
            );
        }

        _ => panic!("Unknown SKY130 cell type for decomposition: {}", cell_type),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_inv() {
        let mut inputs = CellInputs::new();
        inputs.set_pin("A", 2usize); // aigpin 1, not inverted (2 = 1 << 1 | 0)

        let result = decompose_sky130_cell("inv", &inputs);
        assert!(result.and_gates.is_empty());
        assert_eq!(result.output_idx, 1); // aigpin 1
        assert!(result.output_inverted); // inverter inverts
    }

    #[test]
    fn test_decompose_buf() {
        let mut inputs = CellInputs::new();
        inputs.set_pin("A", 2usize);

        let result = decompose_sky130_cell("buf", &inputs);
        assert!(result.and_gates.is_empty());
        assert_eq!(result.output_idx, 1);
        assert!(!result.output_inverted);
    }

    #[test]
    fn test_decompose_nand2() {
        let mut inputs = CellInputs::new();
        inputs.set_pin("A", 2usize); // aigpin 1, not inverted
        inputs.set_pin("B", 4usize); // aigpin 2, not inverted

        let result = decompose_sky130_cell("nand2", &inputs);
        assert_eq!(result.and_gates.len(), 1);
        assert_eq!(result.and_gates[0], (2, 4));
        assert!(result.output_inverted);
    }

    #[test]
    fn test_decompose_nor2() {
        let mut inputs = CellInputs::new();
        inputs.set_pin("A", 2usize);
        inputs.set_pin("B", 4usize);

        let result = decompose_sky130_cell("nor2", &inputs);
        assert_eq!(result.and_gates.len(), 1);
        // NOR = !A & !B, inputs should be inverted (toggle bit 0)
        assert_eq!(result.and_gates[0], (3, 5)); // 2^1 = 3, 4^1 = 5
        assert!(!result.output_inverted);
    }

    #[test]
    fn test_decompose_or2() {
        let mut inputs = CellInputs::new();
        inputs.set_pin("A", 2usize);
        inputs.set_pin("B", 4usize);

        let result = decompose_sky130_cell("or2", &inputs);
        assert_eq!(result.and_gates.len(), 1);
        assert_eq!(result.and_gates[0], (3, 5)); // inverted inputs
        assert!(result.output_inverted); // OR = !(!A & !B)
    }

    #[test]
    fn test_decompose_a21oi() {
        let mut inputs = CellInputs::new();
        inputs.set_pin("A1", 2usize);
        inputs.set_pin("A2", 4usize);
        inputs.set_pin("B1", 6usize);

        let result = decompose_sky130_cell("a21oi", &inputs);
        // a21oi: Y = !((A1 & A2) | B1) = !(A1 & A2) & !B1
        assert_eq!(result.and_gates.len(), 2);
        assert_eq!(result.and_gates[0], (2, 4)); // A1 & A2
        // Second gate: !(A1&A2) & !B1 = (-1^1, 6^1) = (-2, 7)
        assert_eq!(result.and_gates[1], (-2, 7));
        assert!(!result.output_inverted);
    }

    #[test]
    fn test_decompose_o21ai() {
        let mut inputs = CellInputs::new();
        inputs.set_pin("A1", 2usize);
        inputs.set_pin("A2", 4usize);
        inputs.set_pin("B1", 6usize);

        let result = decompose_sky130_cell("o21ai", &inputs);
        // o21ai: Y = !((A1 | A2) & B1)
        // First build NOR: !A1 & !A2
        // Then: (!(!A1 & !A2)) & B1 = (A1|A2) & B1
        // Then invert
        assert_eq!(result.and_gates.len(), 2);
        assert_eq!(result.and_gates[0], (3, 5)); // !A1 & !A2
        assert_eq!(result.and_gates[1], (-2, 6)); // (A1|A2) & B1
        assert!(result.output_inverted);
    }

    #[test]
    fn test_decompose_mux2() {
        let mut inputs = CellInputs::new();
        inputs.set_pin("A0", 2usize);
        inputs.set_pin("A1", 4usize);
        inputs.set_pin("S", 6usize);

        let result = decompose_sky130_cell("mux2", &inputs);
        // MUX: Y = (A0 & !S) | (A1 & S)
        assert_eq!(result.and_gates.len(), 3);
        assert_eq!(result.and_gates[0], (2, 7)); // A0 & !S
        assert_eq!(result.and_gates[1], (4, 6)); // A1 & S
        // Third gate: OR = !( !(A0&!S) & !(A1&S) )
        assert!(result.output_inverted);
    }
}
