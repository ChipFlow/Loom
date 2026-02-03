// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Simple Liberty (.lib) parser for extracting timing data from AIGPDK.
//!
//! This parser is specialized for the scalar timing values used in aigpdk.lib.
//! It extracts:
//! - Cell delays (cell_rise, cell_fall) for combinational cells
//! - Setup/hold constraints for sequential cells (DFF, DFFSR)
//! - Clock-to-Q delays for sequential cells
//! - SRAM timing (read delays, setup/hold)

use indexmap::IndexMap;
use std::fs;
use std::path::Path;

/// Timing arc from an input pin to an output pin.
#[derive(Debug, Clone, Default)]
pub struct TimingArc {
    /// Related (input) pin name
    pub related_pin: String,
    /// Timing type (e.g., "setup_rising", "hold_rising", "rising_edge")
    pub timing_type: Option<String>,
    /// Cell rise delay in picoseconds
    pub cell_rise_ps: Option<u64>,
    /// Cell fall delay in picoseconds
    pub cell_fall_ps: Option<u64>,
    /// Rise constraint (for setup/hold) in picoseconds
    pub rise_constraint_ps: Option<u64>,
    /// Fall constraint (for setup/hold) in picoseconds
    pub fall_constraint_ps: Option<u64>,
}

/// Timing information for a single pin.
#[derive(Debug, Clone, Default)]
pub struct PinTiming {
    /// Pin name
    pub name: String,
    /// Pin direction ("input", "output", "internal")
    pub direction: String,
    /// Whether this pin is a clock
    pub is_clock: bool,
    /// Timing arcs from/to this pin
    pub timing_arcs: Vec<TimingArc>,
}

/// Timing information for a cell.
#[derive(Debug, Clone, Default)]
pub struct CellTiming {
    /// Cell name
    pub name: String,
    /// Pin timing information, keyed by pin name
    pub pins: IndexMap<String, PinTiming>,
}

impl CellTiming {
    /// Get the maximum propagation delay through this cell (for combinational cells).
    /// Returns (rise_delay_ps, fall_delay_ps).
    pub fn max_combinational_delay(&self) -> (u64, u64) {
        let mut max_rise = 0u64;
        let mut max_fall = 0u64;

        for pin in self.pins.values() {
            if pin.direction == "output" {
                for arc in &pin.timing_arcs {
                    if arc.timing_type.is_none()
                        || arc.timing_type.as_deref() == Some("combinational")
                    {
                        if let Some(rise) = arc.cell_rise_ps {
                            max_rise = max_rise.max(rise);
                        }
                        if let Some(fall) = arc.cell_fall_ps {
                            max_fall = max_fall.max(fall);
                        }
                    }
                }
            }
        }

        (max_rise, max_fall)
    }

    /// Get setup time for data pin relative to clock (for sequential cells).
    /// Returns (rise_setup_ps, fall_setup_ps) for rising edge of clock.
    pub fn setup_time(&self, data_pin: &str) -> Option<(u64, u64)> {
        let pin = self.pins.get(data_pin)?;
        for arc in &pin.timing_arcs {
            if arc.timing_type.as_deref() == Some("setup_rising") {
                return Some((
                    arc.rise_constraint_ps.unwrap_or(0),
                    arc.fall_constraint_ps.unwrap_or(0),
                ));
            }
        }
        None
    }

    /// Get hold time for data pin relative to clock (for sequential cells).
    /// Returns (rise_hold_ps, fall_hold_ps) for rising edge of clock.
    pub fn hold_time(&self, data_pin: &str) -> Option<(u64, u64)> {
        let pin = self.pins.get(data_pin)?;
        for arc in &pin.timing_arcs {
            if arc.timing_type.as_deref() == Some("hold_rising") {
                return Some((
                    arc.rise_constraint_ps.unwrap_or(0),
                    arc.fall_constraint_ps.unwrap_or(0),
                ));
            }
        }
        None
    }

    /// Get clock-to-Q delay for output pin (for sequential cells).
    /// Returns (rise_delay_ps, fall_delay_ps).
    pub fn clock_to_q(&self, output_pin: &str) -> Option<(u64, u64)> {
        let pin = self.pins.get(output_pin)?;
        for arc in &pin.timing_arcs {
            if arc.timing_type.as_deref() == Some("rising_edge") {
                return Some((
                    arc.cell_rise_ps.unwrap_or(0),
                    arc.cell_fall_ps.unwrap_or(0),
                ));
            }
        }
        None
    }
}

/// A Liberty timing library containing cell timing data.
#[derive(Debug, Clone, Default)]
pub struct TimingLibrary {
    /// Library name
    pub name: String,
    /// Time unit (e.g., "1ps")
    pub time_unit: String,
    /// Cells in the library, keyed by cell name
    pub cells: IndexMap<String, CellTiming>,
}

impl TimingLibrary {
    /// Load a Liberty library from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let content =
            fs::read_to_string(path.as_ref()).map_err(|e| format!("Failed to read file: {}", e))?;
        Self::parse(&content)
    }

    /// Load the default AIGPDK library from the standard location.
    pub fn load_aigpdk() -> Result<Self, String> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("aigpdk/aigpdk.lib");
        Self::from_file(path)
    }

    /// Create a TimingLibrary with default SKY130 timing values.
    ///
    /// These are approximate values based on typical SKY130 HD cell characteristics.
    /// For accurate timing, provide a proper Liberty file from the Skywater PDK.
    pub fn default_sky130() -> Self {
        // SKY130 typical timing values (in picoseconds):
        // - Inverter: ~30ps
        // - 2-input gates: ~50ps
        // - Complex gates (AOI/OAI): ~60-80ps
        // - DFF clk-to-Q: ~150ps
        // - DFF setup: ~80ps
        // - DFF hold: ~20ps
        // - SRAM read: ~500ps (conservative estimate)

        let mut lib = TimingLibrary {
            name: "sky130_default".to_string(),
            time_unit: "1ps".to_string(),
            cells: IndexMap::new(),
        };

        // Add generic AND gate timing (used for all combinational logic)
        let mut and_cell = CellTiming {
            name: "AND2".to_string(),
            pins: IndexMap::new(),
        };
        let mut and_out = PinTiming {
            name: "X".to_string(),
            direction: "output".to_string(),
            is_clock: false,
            timing_arcs: vec![TimingArc {
                related_pin: "A".to_string(),
                timing_type: None,
                cell_rise_ps: Some(50),
                cell_fall_ps: Some(50),
                rise_constraint_ps: None,
                fall_constraint_ps: None,
            }],
        };
        and_cell.pins.insert("X".to_string(), and_out);
        lib.cells.insert("AND2".to_string(), and_cell);

        // Also add as AND2_00_0 for compatibility with existing lookup
        let mut and_compat = CellTiming {
            name: "AND2_00_0".to_string(),
            pins: IndexMap::new(),
        };
        let and_compat_out = PinTiming {
            name: "Y".to_string(),
            direction: "output".to_string(),
            is_clock: false,
            timing_arcs: vec![TimingArc {
                related_pin: "A".to_string(),
                timing_type: None,
                cell_rise_ps: Some(50),
                cell_fall_ps: Some(50),
                rise_constraint_ps: None,
                fall_constraint_ps: None,
            }],
        };
        and_compat.pins.insert("Y".to_string(), and_compat_out);
        lib.cells.insert("AND2_00_0".to_string(), and_compat);

        // Add DFF timing
        let mut dff_cell = CellTiming {
            name: "DFF".to_string(),
            pins: IndexMap::new(),
        };
        // D pin with setup/hold
        let d_pin = PinTiming {
            name: "D".to_string(),
            direction: "input".to_string(),
            is_clock: false,
            timing_arcs: vec![
                TimingArc {
                    related_pin: "CLK".to_string(),
                    timing_type: Some("setup_rising".to_string()),
                    cell_rise_ps: None,
                    cell_fall_ps: None,
                    rise_constraint_ps: Some(80),
                    fall_constraint_ps: Some(80),
                },
                TimingArc {
                    related_pin: "CLK".to_string(),
                    timing_type: Some("hold_rising".to_string()),
                    cell_rise_ps: None,
                    cell_fall_ps: None,
                    rise_constraint_ps: Some(20),
                    fall_constraint_ps: Some(20),
                },
            ],
        };
        dff_cell.pins.insert("D".to_string(), d_pin);
        // Q pin with clk-to-Q
        let q_pin = PinTiming {
            name: "Q".to_string(),
            direction: "output".to_string(),
            is_clock: false,
            timing_arcs: vec![TimingArc {
                related_pin: "CLK".to_string(),
                timing_type: Some("rising_edge".to_string()),
                cell_rise_ps: Some(150),
                cell_fall_ps: Some(150),
                rise_constraint_ps: None,
                fall_constraint_ps: None,
            }],
        };
        dff_cell.pins.insert("Q".to_string(), q_pin);
        lib.cells.insert("DFF".to_string(), dff_cell);

        // Add SRAM timing (for CF_SRAM_* cells)
        let mut sram_cell = CellTiming {
            name: "$__RAMGEM_SYNC_".to_string(),
            pins: IndexMap::new(),
        };
        let rd_data_pin = PinTiming {
            name: "PORT_R_RD_DATA".to_string(),
            direction: "output".to_string(),
            is_clock: false,
            timing_arcs: vec![TimingArc {
                related_pin: "PORT_R_CLK".to_string(),
                timing_type: Some("rising_edge".to_string()),
                cell_rise_ps: Some(500),
                cell_fall_ps: Some(500),
                rise_constraint_ps: None,
                fall_constraint_ps: None,
            }],
        };
        sram_cell
            .pins
            .insert("PORT_R_RD_DATA".to_string(), rd_data_pin);
        lib.cells.insert("$__RAMGEM_SYNC_".to_string(), sram_cell);

        lib
    }

    /// Parse Liberty content from a string.
    pub fn parse(content: &str) -> Result<Self, String> {
        let mut lib = TimingLibrary::default();
        let mut parser = LibertyParser::new(content);
        parser.parse_library(&mut lib)?;
        Ok(lib)
    }

    /// Get timing for a cell by name.
    pub fn get_cell(&self, name: &str) -> Option<&CellTiming> {
        self.cells.get(name)
    }

    /// Get combinational delay for an AND gate variant.
    /// The AIGPDK uses AND2_XY_Z naming where X,Y are inversion flags.
    pub fn and_gate_delay(&self, cell_name: &str) -> Option<(u64, u64)> {
        self.cells.get(cell_name).map(|c| c.max_combinational_delay())
    }

    /// Get delay for the inverter cell (INV).
    pub fn inv_delay(&self) -> Option<(u64, u64)> {
        self.cells.get("INV").map(|c| c.max_combinational_delay())
    }

    /// Get delay for the buffer cell (BUF).
    pub fn buf_delay(&self) -> Option<(u64, u64)> {
        self.cells.get("BUF").map(|c| c.max_combinational_delay())
    }

    /// Get DFF timing information.
    pub fn dff_timing(&self) -> Option<DFFTiming> {
        let cell = self.cells.get("DFF")?;
        Some(DFFTiming {
            setup_rise_ps: cell.setup_time("D").map(|(r, _)| r).unwrap_or(0),
            setup_fall_ps: cell.setup_time("D").map(|(_, f)| f).unwrap_or(0),
            hold_rise_ps: cell.hold_time("D").map(|(r, _)| r).unwrap_or(0),
            hold_fall_ps: cell.hold_time("D").map(|(_, f)| f).unwrap_or(0),
            clk_to_q_rise_ps: cell.clock_to_q("Q").map(|(r, _)| r).unwrap_or(0),
            clk_to_q_fall_ps: cell.clock_to_q("Q").map(|(_, f)| f).unwrap_or(0),
        })
    }

    /// Get DFFSR (DFF with set/reset) timing information.
    pub fn dffsr_timing(&self) -> Option<DFFTiming> {
        let cell = self.cells.get("DFFSR")?;
        Some(DFFTiming {
            setup_rise_ps: cell.setup_time("D").map(|(r, _)| r).unwrap_or(0),
            setup_fall_ps: cell.setup_time("D").map(|(_, f)| f).unwrap_or(0),
            hold_rise_ps: cell.hold_time("D").map(|(r, _)| r).unwrap_or(0),
            hold_fall_ps: cell.hold_time("D").map(|(_, f)| f).unwrap_or(0),
            clk_to_q_rise_ps: cell.clock_to_q("Q").map(|(r, _)| r).unwrap_or(0),
            clk_to_q_fall_ps: cell.clock_to_q("Q").map(|(_, f)| f).unwrap_or(0),
        })
    }

    /// Get SRAM timing information.
    pub fn sram_timing(&self) -> Option<SRAMTiming> {
        let cell = self.cells.get("$__RAMGEM_SYNC_")?;

        // Get read data output timing (clock to Q)
        let read_delay = cell
            .pins
            .get("PORT_R_RD_DATA")
            .and_then(|p| {
                p.timing_arcs.iter().find_map(|arc| {
                    if arc.timing_type.as_deref() == Some("rising_edge") {
                        Some((
                            arc.cell_rise_ps.unwrap_or(0),
                            arc.cell_fall_ps.unwrap_or(0),
                        ))
                    } else {
                        None
                    }
                })
            })
            .unwrap_or((0, 0));

        Some(SRAMTiming {
            read_clk_to_data_rise_ps: read_delay.0,
            read_clk_to_data_fall_ps: read_delay.1,
            // Setup/hold for address and data inputs (simplified)
            addr_setup_ps: 0, // These use falling edge timing which is 0.0001ps
            addr_hold_ps: 0,
            write_data_setup_ps: 0,
            write_data_hold_ps: 0,
        })
    }
}

/// DFF timing parameters.
#[derive(Debug, Clone, Default)]
pub struct DFFTiming {
    /// Setup time for rising data transition (ps)
    pub setup_rise_ps: u64,
    /// Setup time for falling data transition (ps)
    pub setup_fall_ps: u64,
    /// Hold time for rising data transition (ps)
    pub hold_rise_ps: u64,
    /// Hold time for falling data transition (ps)
    pub hold_fall_ps: u64,
    /// Clock-to-Q delay for rising output (ps)
    pub clk_to_q_rise_ps: u64,
    /// Clock-to-Q delay for falling output (ps)
    pub clk_to_q_fall_ps: u64,
}

impl DFFTiming {
    /// Get maximum setup time (ps).
    pub fn max_setup(&self) -> u64 {
        self.setup_rise_ps.max(self.setup_fall_ps)
    }

    /// Get maximum hold time (ps).
    pub fn max_hold(&self) -> u64 {
        self.hold_rise_ps.max(self.hold_fall_ps)
    }

    /// Get maximum clock-to-Q delay (ps).
    pub fn max_clk_to_q(&self) -> u64 {
        self.clk_to_q_rise_ps.max(self.clk_to_q_fall_ps)
    }
}

/// SRAM timing parameters.
#[derive(Debug, Clone, Default)]
pub struct SRAMTiming {
    /// Read clock to data output rise delay (ps)
    pub read_clk_to_data_rise_ps: u64,
    /// Read clock to data output fall delay (ps)
    pub read_clk_to_data_fall_ps: u64,
    /// Address setup time (ps)
    pub addr_setup_ps: u64,
    /// Address hold time (ps)
    pub addr_hold_ps: u64,
    /// Write data setup time (ps)
    pub write_data_setup_ps: u64,
    /// Write data hold time (ps)
    pub write_data_hold_ps: u64,
}

/// Simple Liberty parser state machine.
struct LibertyParser<'a> {
    content: &'a str,
    pos: usize,
}

impl<'a> LibertyParser<'a> {
    fn new(content: &'a str) -> Self {
        Self { content, pos: 0 }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.content.len() {
            let ch = self.content.as_bytes()[self.pos];
            if ch == b' ' || ch == b'\t' || ch == b'\n' || ch == b'\r' {
                self.pos += 1;
            } else if self.content[self.pos..].starts_with("/*") {
                // Skip block comment
                if let Some(end) = self.content[self.pos..].find("*/") {
                    self.pos += end + 2;
                } else {
                    self.pos = self.content.len();
                }
            } else if self.content[self.pos..].starts_with("//") {
                // Skip line comment
                if let Some(end) = self.content[self.pos..].find('\n') {
                    self.pos += end + 1;
                } else {
                    self.pos = self.content.len();
                }
            } else {
                break;
            }
        }
    }

    fn peek_char(&mut self) -> Option<char> {
        self.skip_whitespace();
        self.content[self.pos..].chars().next()
    }

    fn expect_char(&mut self, ch: char) -> Result<(), String> {
        self.skip_whitespace();
        if self.content[self.pos..].starts_with(ch) {
            self.pos += ch.len_utf8();
            Ok(())
        } else {
            Err(format!(
                "Expected '{}' at position {}, found '{}'",
                ch,
                self.pos,
                self.content[self.pos..].chars().next().unwrap_or('?')
            ))
        }
    }

    fn read_identifier(&mut self) -> String {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.content.len() {
            let ch = self.content.as_bytes()[self.pos];
            if ch.is_ascii_alphanumeric() || ch == b'_' || ch == b'$' {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.content[start..self.pos].to_string()
    }

    fn read_string(&mut self) -> Result<String, String> {
        self.skip_whitespace();
        if !self.content[self.pos..].starts_with('"') {
            return Err(format!("Expected string at position {}", self.pos));
        }
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.content.len() && self.content.as_bytes()[self.pos] != b'"' {
            self.pos += 1;
        }
        let s = self.content[start..self.pos].to_string();
        if self.pos < self.content.len() {
            self.pos += 1; // Skip closing quote
        }
        Ok(s)
    }

    fn read_value(&mut self) -> Result<String, String> {
        self.skip_whitespace();
        if self.content[self.pos..].starts_with('"') {
            self.read_string()
        } else {
            // Read until semicolon, comma, or closing paren
            let start = self.pos;
            let mut depth = 0;
            while self.pos < self.content.len() {
                let ch = self.content.as_bytes()[self.pos];
                if ch == b'(' {
                    depth += 1;
                    self.pos += 1;
                } else if ch == b')' {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    self.pos += 1;
                } else if (ch == b';' || ch == b',') && depth == 0 {
                    break;
                } else {
                    self.pos += 1;
                }
            }
            Ok(self.content[start..self.pos].trim().to_string())
        }
    }

    fn parse_float_to_ps(&self, value: &str) -> u64 {
        // Parse a floating point value and convert to picoseconds (integer)
        // The Liberty file uses time_unit : "1ps", so values are already in ps
        let value = value.trim().trim_matches('"');
        if let Ok(f) = value.parse::<f64>() {
            // Round to nearest picosecond
            (f * 1.0).round() as u64
        } else {
            0
        }
    }

    fn skip_block(&mut self) -> Result<(), String> {
        let mut depth = 1;
        while self.pos < self.content.len() && depth > 0 {
            let ch = self.content.as_bytes()[self.pos];
            if ch == b'{' {
                depth += 1;
            } else if ch == b'}' {
                depth -= 1;
            }
            self.pos += 1;
        }
        Ok(())
    }

    fn parse_library(&mut self, lib: &mut TimingLibrary) -> Result<(), String> {
        self.skip_whitespace();
        let keyword = self.read_identifier();
        if keyword != "library" {
            return Err(format!("Expected 'library', found '{}'", keyword));
        }

        self.expect_char('(')?;
        lib.name = self.read_identifier();
        self.expect_char(')')?;
        self.expect_char('{')?;

        while self.peek_char() != Some('}') {
            self.skip_whitespace();
            let keyword = self.read_identifier();

            match keyword.as_str() {
                "time_unit" => {
                    self.expect_char(':')?;
                    lib.time_unit = self.read_value()?;
                    self.expect_char(';')?;
                }
                "cell" => {
                    let cell = self.parse_cell()?;
                    lib.cells.insert(cell.name.clone(), cell);
                }
                "type" | "operating_conditions" => {
                    // Skip these blocks
                    self.expect_char('(')?;
                    self.read_value()?;
                    self.expect_char(')')?;
                    self.expect_char('{')?;
                    self.skip_block()?;
                }
                _ => {
                    // Skip simple attribute
                    if self.peek_char() == Some(':') {
                        self.expect_char(':')?;
                        self.read_value()?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    } else if self.peek_char() == Some('(') {
                        // Consume entire parenthesized expression including commas
                        self.skip_parenthesized()?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    }
                }
            }
        }

        self.expect_char('}')?;
        Ok(())
    }

    /// Skip a parenthesized expression including its contents.
    fn skip_parenthesized(&mut self) -> Result<(), String> {
        self.expect_char('(')?;
        let mut depth = 1;
        while self.pos < self.content.len() && depth > 0 {
            let ch = self.content.as_bytes()[self.pos];
            if ch == b'(' {
                depth += 1;
            } else if ch == b')' {
                depth -= 1;
            }
            self.pos += 1;
        }
        Ok(())
    }

    fn parse_cell(&mut self) -> Result<CellTiming, String> {
        let mut cell = CellTiming::default();

        self.expect_char('(')?;
        cell.name = self.read_identifier();
        // Handle cell names with special characters like $__RAMGEM_SYNC_
        if self.peek_char() == Some('_') {
            while self.peek_char() == Some('_') {
                self.expect_char('_')?;
                cell.name.push('_');
                let extra = self.read_identifier();
                cell.name.push_str(&extra);
            }
        }
        self.expect_char(')')?;
        self.expect_char('{')?;

        while self.peek_char() != Some('}') {
            self.skip_whitespace();
            let keyword = self.read_identifier();

            match keyword.as_str() {
                "pin" => {
                    let pin = self.parse_pin()?;
                    cell.pins.insert(pin.name.clone(), pin);
                }
                "bus" => {
                    let pin = self.parse_bus()?;
                    cell.pins.insert(pin.name.clone(), pin);
                }
                "ff" | "memory" | "statetable" => {
                    // Skip these blocks - they have parenthesized args with commas
                    self.skip_parenthesized()?;
                    self.expect_char('{')?;
                    self.skip_block()?;
                }
                _ => {
                    // Skip simple attribute
                    if self.peek_char() == Some(':') {
                        self.expect_char(':')?;
                        self.read_value()?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    } else if self.peek_char() == Some('(') {
                        // Consume entire parenthesized expression including commas
                        self.skip_parenthesized()?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    }
                }
            }
        }

        self.expect_char('}')?;
        Ok(cell)
    }

    fn parse_pin(&mut self) -> Result<PinTiming, String> {
        let mut pin = PinTiming::default();

        self.expect_char('(')?;
        pin.name = self.read_identifier();
        self.expect_char(')')?;
        self.expect_char('{')?;

        while self.peek_char() != Some('}') {
            self.skip_whitespace();
            let keyword = self.read_identifier();

            match keyword.as_str() {
                "direction" => {
                    self.expect_char(':')?;
                    pin.direction = self.read_identifier();
                    self.expect_char(';')?;
                }
                "clock" => {
                    self.expect_char(':')?;
                    let val = self.read_identifier();
                    pin.is_clock = val == "true";
                    self.expect_char(';')?;
                }
                "timing" => {
                    let arc = self.parse_timing_arc()?;
                    pin.timing_arcs.push(arc);
                }
                _ => {
                    // Skip simple attribute
                    if self.peek_char() == Some(':') {
                        self.expect_char(':')?;
                        self.read_value()?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    } else if self.peek_char() == Some('(') {
                        self.expect_char('(')?;
                        self.read_value()?;
                        self.expect_char(')')?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    }
                }
            }
        }

        self.expect_char('}')?;
        Ok(pin)
    }

    fn parse_bus(&mut self) -> Result<PinTiming, String> {
        let mut pin = PinTiming::default();

        self.expect_char('(')?;
        pin.name = self.read_identifier();
        self.expect_char(')')?;
        self.expect_char('{')?;

        while self.peek_char() != Some('}') {
            self.skip_whitespace();
            let keyword = self.read_identifier();

            match keyword.as_str() {
                "direction" => {
                    self.expect_char(':')?;
                    pin.direction = self.read_identifier();
                    self.expect_char(';')?;
                }
                "timing" => {
                    let arc = self.parse_timing_arc()?;
                    pin.timing_arcs.push(arc);
                }
                "pin" | "memory_read" | "memory_write" => {
                    // Skip nested blocks
                    self.skip_parenthesized()?;
                    if self.peek_char() == Some('{') {
                        self.expect_char('{')?;
                        self.skip_block()?;
                    }
                }
                _ => {
                    // Skip simple attribute
                    if self.peek_char() == Some(':') {
                        self.expect_char(':')?;
                        self.read_value()?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    } else if self.peek_char() == Some('(') {
                        // Consume entire parenthesized expression including commas
                        self.skip_parenthesized()?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    }
                }
            }
        }

        self.expect_char('}')?;
        Ok(pin)
    }

    fn parse_timing_arc(&mut self) -> Result<TimingArc, String> {
        let mut arc = TimingArc::default();

        self.expect_char('(')?;
        self.expect_char(')')?;
        self.expect_char('{')?;

        while self.peek_char() != Some('}') {
            self.skip_whitespace();
            let keyword = self.read_identifier();

            match keyword.as_str() {
                "related_pin" => {
                    self.expect_char(':')?;
                    arc.related_pin = self.read_string()?;
                    self.expect_char(';')?;
                }
                "timing_type" => {
                    self.expect_char(':')?;
                    arc.timing_type = Some(self.read_identifier());
                    self.expect_char(';')?;
                }
                "cell_rise" | "cell_fall" | "rise_constraint" | "fall_constraint" => {
                    // Parse scalar value
                    self.expect_char('(')?;
                    let _table_type = self.read_identifier(); // "scalar"
                    self.expect_char(')')?;
                    self.expect_char('{')?;

                    // Find values
                    while self.peek_char() != Some('}') {
                        self.skip_whitespace();
                        let inner_kw = self.read_identifier();
                        if inner_kw == "values" {
                            self.expect_char('(')?;
                            let val_str = self.read_value()?;
                            self.expect_char(')')?;
                            self.expect_char(';')?;

                            let ps_val = self.parse_float_to_ps(&val_str);
                            match keyword.as_str() {
                                "cell_rise" => arc.cell_rise_ps = Some(ps_val),
                                "cell_fall" => arc.cell_fall_ps = Some(ps_val),
                                "rise_constraint" => arc.rise_constraint_ps = Some(ps_val),
                                "fall_constraint" => arc.fall_constraint_ps = Some(ps_val),
                                _ => {}
                            }
                        } else {
                            // Skip other attributes
                            if self.peek_char() == Some(':') {
                                self.expect_char(':')?;
                                self.read_value()?;
                                if self.peek_char() == Some(';') {
                                    self.expect_char(';')?;
                                }
                            }
                        }
                    }
                    self.expect_char('}')?;
                }
                _ => {
                    // Skip other timing attributes
                    if self.peek_char() == Some(':') {
                        self.expect_char(':')?;
                        self.read_value()?;
                        if self.peek_char() == Some(';') {
                            self.expect_char(';')?;
                        }
                    } else if self.peek_char() == Some('(') {
                        self.expect_char('(')?;
                        let _table = self.read_identifier();
                        self.expect_char(')')?;
                        self.expect_char('{')?;
                        self.skip_block()?;
                    }
                }
            }
        }

        self.expect_char('}')?;
        Ok(arc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_aigpdk() {
        let lib = TimingLibrary::load_aigpdk().expect("Failed to load AIGPDK library");

        assert_eq!(lib.name, "aigpdk");
        assert_eq!(lib.time_unit, "1ps");

        // Check AND gate timing
        let and_timing = lib.and_gate_delay("AND2_00_0");
        assert!(and_timing.is_some());
        let (rise, fall) = and_timing.unwrap();
        assert_eq!(rise, 1); // 1.0ps
        assert_eq!(fall, 1);

        // Check INV timing (near-zero delay)
        let inv_timing = lib.inv_delay();
        assert!(inv_timing.is_some());
        let (rise, fall) = inv_timing.unwrap();
        assert_eq!(rise, 0); // 0.0001ps rounds to 0
        assert_eq!(fall, 0);

        // Check DFF timing
        let dff = lib.dff_timing();
        assert!(dff.is_some());
        let dff = dff.unwrap();
        assert_eq!(dff.max_setup(), 0); // 0.0001ps rounds to 0
        assert_eq!(dff.max_hold(), 0);
        assert_eq!(dff.max_clk_to_q(), 0);

        // Check SRAM timing
        let sram = lib.sram_timing();
        assert!(sram.is_some());
        let sram = sram.unwrap();
        assert_eq!(sram.read_clk_to_data_rise_ps, 1); // 1.0ps
    }

    #[test]
    fn test_all_and_gates() {
        let lib = TimingLibrary::load_aigpdk().expect("Failed to load AIGPDK library");

        let and_variants = [
            "AND2_00_0",
            "AND2_01_0",
            "AND2_10_0",
            "AND2_11_0",
            "AND2_11_1",
        ];

        for name in and_variants {
            let timing = lib.and_gate_delay(name);
            assert!(timing.is_some(), "Missing timing for {}", name);
            let (rise, fall) = timing.unwrap();
            assert_eq!(rise, 1, "Wrong rise delay for {}", name);
            assert_eq!(fall, 1, "Wrong fall delay for {}", name);
        }
    }
}
