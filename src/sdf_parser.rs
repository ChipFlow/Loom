// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parser for Standard Delay Format (SDF) 3.0 files.
//!
//! SDF files contain per-instance timing delays from place-and-route tools.
//! This parser extracts IOPATH (cell) delays, INTERCONNECT (wire) delays,
//! and TIMINGCHECK (setup/hold) constraints.

use std::collections::HashMap;
use std::path::Path;

/// Delay value in picoseconds (rise and fall).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SdfDelay {
    pub rise_ps: u64,
    pub fall_ps: u64,
}

/// An IOPATH delay entry: input_pin → output_pin with rise/fall delay.
#[derive(Debug, Clone)]
pub struct SdfIopath {
    pub input_pin: String,
    pub output_pin: String,
    pub delay: SdfDelay,
}

/// An INTERCONNECT delay entry: source → dest with rise/fall delay.
#[derive(Debug, Clone)]
pub struct SdfInterconnect {
    /// Source port path (e.g., "dff_in.Q" or "u_cpu.alu.and_gate.Y")
    pub source: String,
    /// Destination port path (e.g., "i0.A")
    pub dest: String,
    pub delay: SdfDelay,
}

/// A TIMINGCHECK entry for setup or hold constraints.
#[derive(Debug, Clone)]
pub struct SdfTimingCheck {
    /// "SETUP" or "HOLD"
    pub check_type: TimingCheckType,
    /// Data pin name
    pub data_pin: String,
    /// Clock edge specification (e.g., "posedge CLK")
    pub clock_edge: String,
    /// Constraint value in picoseconds
    pub value_ps: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingCheckType {
    Setup,
    Hold,
}

/// A cell entry in the SDF file.
#[derive(Debug, Clone)]
pub struct SdfCell {
    pub cell_type: String,
    /// Instance path (empty string for top-level module)
    pub instance: String,
    pub iopaths: Vec<SdfIopath>,
    pub interconnects: Vec<SdfInterconnect>,
    pub timing_checks: Vec<SdfTimingCheck>,
}

/// Parsed SDF file.
#[derive(Debug, Clone)]
pub struct SdfFile {
    pub design: String,
    pub timescale_ps: f64,
    pub cells: Vec<SdfCell>,
    /// Quick lookup: instance path → index in cells vec
    pub instance_map: HashMap<String, usize>,
}

/// Which corner of (min:typ:max) triples to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfCorner {
    Min,
    Typ,
    Max,
}

impl std::fmt::Display for SdfCorner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SdfCorner::Min => write!(f, "min"),
            SdfCorner::Typ => write!(f, "typ"),
            SdfCorner::Max => write!(f, "max"),
        }
    }
}

impl std::str::FromStr for SdfCorner {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "min" => Ok(SdfCorner::Min),
            "typ" | "typical" => Ok(SdfCorner::Typ),
            "max" => Ok(SdfCorner::Max),
            _ => Err(format!("Unknown SDF corner '{}', expected min/typ/max", s)),
        }
    }
}

impl SdfFile {
    /// Parse an SDF file from a file path.
    pub fn parse_file(path: &Path, corner: SdfCorner) -> Result<Self, SdfParseError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| SdfParseError::Io(format!("{}: {}", path.display(), e)))?;
        Self::parse_str(&content, corner)
    }

    /// Parse an SDF file from a string.
    pub fn parse_str(input: &str, corner: SdfCorner) -> Result<Self, SdfParseError> {
        let mut parser = SdfParser::new(input, corner);
        parser.parse()
    }

    /// Look up a cell by instance path.
    pub fn get_cell(&self, instance: &str) -> Option<&SdfCell> {
        self.instance_map.get(instance).map(|&i| &self.cells[i])
    }

    /// Summary statistics for debug output.
    pub fn summary(&self) -> String {
        let num_iopaths: usize = self.cells.iter().map(|c| c.iopaths.len()).sum();
        let num_interconnects: usize = self.cells.iter().map(|c| c.interconnects.len()).sum();
        let num_timing_checks: usize = self.cells.iter().map(|c| c.timing_checks.len()).sum();
        format!(
            "SDF: {} cells, {} IOPATH delays, {} INTERCONNECT delays, {} timing checks",
            self.cells.len(), num_iopaths, num_interconnects, num_timing_checks
        )
    }
}

#[derive(Debug)]
pub enum SdfParseError {
    Io(String),
    Syntax(String, usize),
    UnexpectedEof,
}

impl std::fmt::Display for SdfParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SdfParseError::Io(msg) => write!(f, "SDF I/O error: {}", msg),
            SdfParseError::Syntax(msg, pos) => write!(f, "SDF syntax error at byte {}: {}", pos, msg),
            SdfParseError::UnexpectedEof => write!(f, "SDF unexpected end of file"),
        }
    }
}

impl std::error::Error for SdfParseError {}

/// S-expression token for SDF parsing.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    LParen,
    RParen,
    Str(String),
}

/// Tokenizer for SDF s-expression format.
struct Tokenizer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input: input.as_bytes(), pos: 0 }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch.is_ascii_whitespace() {
                self.pos += 1;
            } else if ch == b'/' && self.pos + 1 < self.input.len() {
                if self.input[self.pos + 1] == b'/' {
                    // Line comment
                    while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                } else if self.input[self.pos + 1] == b'*' {
                    // Block comment
                    self.pos += 2;
                    while self.pos + 1 < self.input.len() {
                        if self.input[self.pos] == b'*' && self.input[self.pos + 1] == b'/' {
                            self.pos += 2;
                            break;
                        }
                        self.pos += 1;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> Option<Token> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.input.len() {
            return None;
        }
        let ch = self.input[self.pos];
        match ch {
            b'(' => {
                self.pos += 1;
                Some(Token::LParen)
            }
            b')' => {
                self.pos += 1;
                Some(Token::RParen)
            }
            b'"' => {
                // Quoted string
                self.pos += 1;
                let start = self.pos;
                while self.pos < self.input.len() && self.input[self.pos] != b'"' {
                    self.pos += 1;
                }
                let s = std::str::from_utf8(&self.input[start..self.pos]).unwrap_or("").to_string();
                if self.pos < self.input.len() {
                    self.pos += 1; // skip closing quote
                }
                Some(Token::Str(s))
            }
            _ => {
                // Unquoted token (keyword, number, identifier)
                // SDF uses backslash to escape special characters in identifiers
                // (e.g., inst\$top\.soc → inst$top.soc). We strip escapes here
                // so that instance paths match the unescaped names from netlistdb.
                let start = self.pos;
                while self.pos < self.input.len() {
                    let c = self.input[self.pos];
                    if c.is_ascii_whitespace() || c == b'(' || c == b')' || c == b'"' {
                        break;
                    }
                    self.pos += 1;
                }
                let raw = std::str::from_utf8(&self.input[start..self.pos]).unwrap_or("");
                let s = if raw.contains('\\') {
                    raw.replace('\\', "")
                } else {
                    raw.to_string()
                };
                Some(Token::Str(s))
            }
        }
    }

    fn peek_token(&mut self) -> Option<Token> {
        let saved = self.pos;
        let tok = self.next_token();
        self.pos = saved;
        tok
    }
}

struct SdfParser<'a> {
    tokenizer: Tokenizer<'a>,
    corner: SdfCorner,
}

impl<'a> SdfParser<'a> {
    fn new(input: &'a str, corner: SdfCorner) -> Self {
        Self {
            tokenizer: Tokenizer::new(input),
            corner,
        }
    }

    fn expect_lparen(&mut self) -> Result<(), SdfParseError> {
        match self.tokenizer.next_token() {
            Some(Token::LParen) => Ok(()),
            Some(t) => Err(SdfParseError::Syntax(format!("expected '(', got {:?}", t), self.tokenizer.pos)),
            None => Err(SdfParseError::UnexpectedEof),
        }
    }

    fn expect_rparen(&mut self) -> Result<(), SdfParseError> {
        match self.tokenizer.next_token() {
            Some(Token::RParen) => Ok(()),
            Some(t) => Err(SdfParseError::Syntax(format!("expected ')', got {:?}", t), self.tokenizer.pos)),
            None => Err(SdfParseError::UnexpectedEof),
        }
    }

    fn expect_keyword(&mut self, kw: &str) -> Result<(), SdfParseError> {
        match self.tokenizer.next_token() {
            Some(Token::Str(s)) if s.eq_ignore_ascii_case(kw) => Ok(()),
            Some(t) => Err(SdfParseError::Syntax(format!("expected '{}', got {:?}", kw, t), self.tokenizer.pos)),
            None => Err(SdfParseError::UnexpectedEof),
        }
    }

    fn read_str(&mut self) -> Result<String, SdfParseError> {
        match self.tokenizer.next_token() {
            Some(Token::Str(s)) => Ok(s),
            Some(t) => Err(SdfParseError::Syntax(format!("expected string, got {:?}", t), self.tokenizer.pos)),
            None => Err(SdfParseError::UnexpectedEof),
        }
    }

    /// Skip balanced parentheses — consume everything until matching ')'.
    fn skip_balanced(&mut self) -> Result<(), SdfParseError> {
        let mut depth = 1u32;
        loop {
            match self.tokenizer.next_token() {
                Some(Token::LParen) => depth += 1,
                Some(Token::RParen) => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                Some(_) => {}
                None => return Err(SdfParseError::UnexpectedEof),
            }
        }
    }

    fn parse(&mut self) -> Result<SdfFile, SdfParseError> {
        // (DELAYFILE ...)
        self.expect_lparen()?;
        self.expect_keyword("DELAYFILE")?;

        let mut design = String::new();
        let mut timescale_ps: f64 = 1000.0; // default 1ns
        let mut cells = Vec::new();

        loop {
            match self.tokenizer.peek_token() {
                Some(Token::RParen) => {
                    self.tokenizer.next_token();
                    break;
                }
                Some(Token::LParen) => {
                    self.tokenizer.next_token();
                    let keyword = self.read_str()?;
                    match keyword.to_uppercase().as_str() {
                        "SDFVERSION" | "DATE" | "VENDOR" | "PROGRAM" | "VERSION"
                        | "DIVIDER" | "VOLTAGE" | "PROCESS" | "TEMPERATURE" => {
                            // Read and discard value(s) until matching ')'
                            self.skip_balanced()?;
                        }
                        "DESIGN" => {
                            design = self.read_str()?;
                            self.expect_rparen()?;
                        }
                        "TIMESCALE" => {
                            let ts = self.read_str()?;
                            timescale_ps = parse_timescale(&ts)?;
                            self.expect_rparen()?;
                        }
                        "CELL" => {
                            let cell = self.parse_cell(timescale_ps)?;
                            cells.push(cell);
                        }
                        _ => {
                            // Unknown top-level keyword, skip
                            self.skip_balanced()?;
                        }
                    }
                }
                Some(Token::Str(_)) => {
                    // Stray token at top level, skip
                    self.tokenizer.next_token();
                }
                None => return Err(SdfParseError::UnexpectedEof),
            }
        }

        let mut instance_map = HashMap::new();
        for (i, cell) in cells.iter().enumerate() {
            instance_map.insert(cell.instance.clone(), i);
        }

        Ok(SdfFile {
            design,
            timescale_ps,
            cells,
            instance_map,
        })
    }

    fn parse_cell(&mut self, timescale_ps: f64) -> Result<SdfCell, SdfParseError> {
        let mut cell_type = String::new();
        let mut instance = String::new();
        let mut iopaths = Vec::new();
        let mut interconnects = Vec::new();
        let mut timing_checks = Vec::new();

        loop {
            match self.tokenizer.peek_token() {
                Some(Token::RParen) => {
                    self.tokenizer.next_token();
                    break;
                }
                Some(Token::LParen) => {
                    self.tokenizer.next_token();
                    let keyword = self.read_str()?;
                    match keyword.to_uppercase().as_str() {
                        "CELLTYPE" => {
                            cell_type = self.read_str()?;
                            self.expect_rparen()?;
                        }
                        "INSTANCE" => {
                            // INSTANCE may be empty (top-level) or have a path
                            match self.tokenizer.peek_token() {
                                Some(Token::RParen) => {
                                    self.tokenizer.next_token();
                                    instance = String::new();
                                }
                                _ => {
                                    instance = self.read_str()?;
                                    self.expect_rparen()?;
                                }
                            }
                        }
                        "DELAY" => {
                            self.parse_delay_block(timescale_ps, &mut iopaths, &mut interconnects)?;
                        }
                        "TIMINGCHECK" => {
                            self.parse_timingcheck_block(timescale_ps, &mut timing_checks)?;
                        }
                        _ => {
                            self.skip_balanced()?;
                        }
                    }
                }
                _ => {
                    self.tokenizer.next_token();
                }
            }
        }

        Ok(SdfCell {
            cell_type,
            instance,
            iopaths,
            interconnects,
            timing_checks,
        })
    }

    fn parse_delay_block(
        &mut self,
        timescale_ps: f64,
        iopaths: &mut Vec<SdfIopath>,
        interconnects: &mut Vec<SdfInterconnect>,
    ) -> Result<(), SdfParseError> {
        // (DELAY (ABSOLUTE ...))
        loop {
            match self.tokenizer.peek_token() {
                Some(Token::RParen) => {
                    self.tokenizer.next_token();
                    return Ok(());
                }
                Some(Token::LParen) => {
                    self.tokenizer.next_token();
                    let keyword = self.read_str()?;
                    match keyword.to_uppercase().as_str() {
                        "ABSOLUTE" | "INCREMENT" => {
                            self.parse_delay_entries(timescale_ps, iopaths, interconnects)?;
                        }
                        _ => {
                            self.skip_balanced()?;
                        }
                    }
                }
                _ => {
                    self.tokenizer.next_token();
                }
            }
        }
    }

    fn parse_delay_entries(
        &mut self,
        timescale_ps: f64,
        iopaths: &mut Vec<SdfIopath>,
        interconnects: &mut Vec<SdfInterconnect>,
    ) -> Result<(), SdfParseError> {
        loop {
            match self.tokenizer.peek_token() {
                Some(Token::RParen) => {
                    self.tokenizer.next_token();
                    return Ok(());
                }
                Some(Token::LParen) => {
                    self.tokenizer.next_token();
                    let keyword = self.read_str()?;
                    match keyword.to_uppercase().as_str() {
                        "IOPATH" => {
                            let iopath = self.parse_iopath(timescale_ps)?;
                            iopaths.push(iopath);
                        }
                        "INTERCONNECT" => {
                            let ic = self.parse_interconnect(timescale_ps)?;
                            interconnects.push(ic);
                        }
                        _ => {
                            self.skip_balanced()?;
                        }
                    }
                }
                _ => {
                    self.tokenizer.next_token();
                }
            }
        }
    }

    fn parse_iopath(&mut self, timescale_ps: f64) -> Result<SdfIopath, SdfParseError> {
        // (IOPATH input_pin output_pin rise_delay fall_delay)
        // Input pin may be a simple name or an edge spec like (posedge CLK)
        let input_pin = self.read_pin_spec()?;
        let output_pin = self.read_str()?;
        let rise = self.parse_delay_value(timescale_ps)?;
        let fall = self.parse_delay_value(timescale_ps)?;
        // There might be additional delay values (e.g., 6-value or 12-value SDF)
        // Consume any remaining tokens before ')'
        loop {
            match self.tokenizer.peek_token() {
                Some(Token::RParen) => {
                    self.tokenizer.next_token();
                    break;
                }
                Some(Token::LParen) => {
                    self.tokenizer.next_token();
                    self.skip_balanced()?;
                }
                _ => {
                    self.tokenizer.next_token();
                }
            }
        }
        Ok(SdfIopath {
            input_pin,
            output_pin,
            delay: SdfDelay {
                rise_ps: rise,
                fall_ps: fall,
            },
        })
    }

    fn parse_interconnect(&mut self, timescale_ps: f64) -> Result<SdfInterconnect, SdfParseError> {
        // (INTERCONNECT source dest rise_delay fall_delay)
        let source = self.read_str()?;
        let dest = self.read_str()?;
        let rise = self.parse_delay_value(timescale_ps)?;
        let fall = self.parse_delay_value(timescale_ps)?;
        // Consume remaining tokens before ')'
        loop {
            match self.tokenizer.peek_token() {
                Some(Token::RParen) => {
                    self.tokenizer.next_token();
                    break;
                }
                Some(Token::LParen) => {
                    self.tokenizer.next_token();
                    self.skip_balanced()?;
                }
                _ => {
                    self.tokenizer.next_token();
                }
            }
        }
        Ok(SdfInterconnect {
            source,
            dest,
            delay: SdfDelay {
                rise_ps: rise,
                fall_ps: fall,
            },
        })
    }

    /// Read a pin spec, which may be a simple name or (posedge/negedge NAME).
    fn read_pin_spec(&mut self) -> Result<String, SdfParseError> {
        match self.tokenizer.peek_token() {
            Some(Token::LParen) => {
                // Edge-qualified pin: (posedge CLK)
                self.tokenizer.next_token();
                let _edge = self.read_str()?; // posedge/negedge
                let pin = self.read_str()?;
                self.expect_rparen()?;
                Ok(pin)
            }
            _ => self.read_str(),
        }
    }

    /// Parse a delay value, which can be:
    /// - A single number: 0.050
    /// - A triple: (0.040:0.050:0.060)
    /// Returns the selected corner value in picoseconds.
    fn parse_delay_value(&mut self, timescale_ps: f64) -> Result<u64, SdfParseError> {
        match self.tokenizer.peek_token() {
            Some(Token::LParen) => {
                self.tokenizer.next_token();
                let triple_str = self.read_str()?;
                self.expect_rparen()?;
                self.parse_triple(&triple_str, timescale_ps)
            }
            Some(Token::Str(s)) => {
                self.tokenizer.next_token();
                // Could be a simple number or a triple without parens
                if s.contains(':') {
                    self.parse_triple(&s, timescale_ps)
                } else {
                    let val: f64 = s.parse().map_err(|_| {
                        SdfParseError::Syntax(format!("invalid delay number '{}'", s), self.tokenizer.pos)
                    })?;
                    Ok((val * timescale_ps).round() as u64)
                }
            }
            _ => Ok(0), // Missing delay value = 0
        }
    }

    /// Parse a min:typ:max triple and select the appropriate corner value.
    fn parse_triple(&mut self, s: &str, timescale_ps: f64) -> Result<u64, SdfParseError> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() == 3 {
            let idx = match self.corner {
                SdfCorner::Min => 0,
                SdfCorner::Typ => 1,
                SdfCorner::Max => 2,
            };
            let val_str = parts[idx].trim();
            if val_str.is_empty() {
                // Empty slot in triple — common in OpenSTA output (min::max with no typ).
                // Fall back to any non-empty slot: try all three positions.
                let fallback = parts.iter()
                    .map(|p| p.trim())
                    .find(|p| !p.is_empty());
                if let Some(fb) = fallback {
                    let val: f64 = fb.parse().map_err(|_| {
                        SdfParseError::Syntax(format!("invalid delay triple '{}'", s), self.tokenizer.pos)
                    })?;
                    return Ok((val * timescale_ps).round() as u64);
                } else {
                    return Ok(0);
                }
            }
            let val: f64 = val_str.parse().map_err(|_| {
                SdfParseError::Syntax(format!("invalid delay number '{}' in triple '{}'", val_str, s), self.tokenizer.pos)
            })?;
            Ok((val * timescale_ps).round() as u64)
        } else if parts.len() == 1 {
            let val: f64 = s.parse().map_err(|_| {
                SdfParseError::Syntax(format!("invalid delay number '{}'", s), self.tokenizer.pos)
            })?;
            Ok((val * timescale_ps).round() as u64)
        } else {
            Err(SdfParseError::Syntax(format!("invalid delay triple '{}'", s), self.tokenizer.pos))
        }
    }

    fn parse_timingcheck_block(
        &mut self,
        timescale_ps: f64,
        checks: &mut Vec<SdfTimingCheck>,
    ) -> Result<(), SdfParseError> {
        loop {
            match self.tokenizer.peek_token() {
                Some(Token::RParen) => {
                    self.tokenizer.next_token();
                    return Ok(());
                }
                Some(Token::LParen) => {
                    self.tokenizer.next_token();
                    let keyword = self.read_str()?;
                    match keyword.to_uppercase().as_str() {
                        "SETUP" => {
                            let check = self.parse_timing_check_entry(TimingCheckType::Setup, timescale_ps)?;
                            checks.push(check);
                        }
                        "HOLD" => {
                            let check = self.parse_timing_check_entry(TimingCheckType::Hold, timescale_ps)?;
                            checks.push(check);
                        }
                        _ => {
                            self.skip_balanced()?;
                        }
                    }
                }
                _ => {
                    self.tokenizer.next_token();
                }
            }
        }
    }

    fn parse_timing_check_entry(
        &mut self,
        check_type: TimingCheckType,
        timescale_ps: f64,
    ) -> Result<SdfTimingCheck, SdfParseError> {
        // (SETUP data_pin (posedge CLK) value)
        // (HOLD data_pin (posedge CLK) value)
        // Data pin may also have edge spec: (posedge D) or just D
        let data_pin = self.read_pin_spec()?;

        // Clock edge spec
        let clock_edge = self.read_pin_spec()?;

        // Value - can be negative for hold
        let value_ps = self.parse_signed_delay_value(timescale_ps)?;

        self.expect_rparen()?;

        Ok(SdfTimingCheck {
            check_type,
            data_pin,
            clock_edge,
            value_ps,
        })
    }

    /// Parse a potentially signed delay value.
    fn parse_signed_delay_value(&mut self, timescale_ps: f64) -> Result<i64, SdfParseError> {
        match self.tokenizer.peek_token() {
            Some(Token::LParen) => {
                self.tokenizer.next_token();
                let triple_str = self.read_str()?;
                self.expect_rparen()?;
                self.parse_signed_triple(&triple_str, timescale_ps)
            }
            Some(Token::Str(s)) => {
                self.tokenizer.next_token();
                if s.contains(':') {
                    self.parse_signed_triple(&s, timescale_ps)
                } else {
                    let val: f64 = s.parse().map_err(|_| {
                        SdfParseError::Syntax(format!("invalid delay number '{}'", s), self.tokenizer.pos)
                    })?;
                    Ok((val * timescale_ps).round() as i64)
                }
            }
            _ => Ok(0),
        }
    }

    fn parse_signed_triple(&mut self, s: &str, timescale_ps: f64) -> Result<i64, SdfParseError> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() == 3 {
            let idx = match self.corner {
                SdfCorner::Min => 0,
                SdfCorner::Typ => 1,
                SdfCorner::Max => 2,
            };
            let val_str = parts[idx].trim();
            if val_str.is_empty() {
                // Empty slot — fall back to any non-empty slot (OpenSTA min::max format)
                let fallback = parts.iter()
                    .map(|p| p.trim())
                    .find(|p| !p.is_empty());
                if let Some(fb) = fallback {
                    let val: f64 = fb.parse().map_err(|_| {
                        SdfParseError::Syntax(format!("invalid delay triple '{}'", s), self.tokenizer.pos)
                    })?;
                    return Ok((val * timescale_ps).round() as i64);
                } else {
                    return Ok(0);
                }
            }
            let val: f64 = val_str.parse().map_err(|_| {
                SdfParseError::Syntax(format!("invalid number '{}' in triple '{}'", val_str, s), self.tokenizer.pos)
            })?;
            Ok((val * timescale_ps).round() as i64)
        } else if parts.len() == 1 {
            let val: f64 = s.parse().map_err(|_| {
                SdfParseError::Syntax(format!("invalid delay number '{}'", s), self.tokenizer.pos)
            })?;
            Ok((val * timescale_ps).round() as i64)
        } else {
            Err(SdfParseError::Syntax(format!("invalid delay triple '{}'", s), self.tokenizer.pos))
        }
    }
}

/// Parse a timescale string like "1ns", "100ps", "10us" into picoseconds.
fn parse_timescale(ts: &str) -> Result<f64, SdfParseError> {
    let ts = ts.trim();
    // Find where unit starts
    let num_end = ts.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(ts.len());
    let num_str = ts[..num_end].trim();
    let unit_str = ts[num_end..].trim().to_lowercase();

    let multiplier: f64 = if num_str.is_empty() {
        1.0
    } else {
        num_str.parse().map_err(|_| {
            SdfParseError::Syntax(format!("invalid timescale number '{}'", num_str), 0)
        })?
    };

    let unit_ps = match unit_str.as_str() {
        "s" => 1e12,
        "ms" => 1e9,
        "us" => 1e6,
        "ns" => 1e3,
        "ps" => 1.0,
        "fs" => 1e-3,
        _ => return Err(SdfParseError::Syntax(format!("unknown timescale unit '{}'", unit_str), 0)),
    };

    Ok(multiplier * unit_ps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_timescale() {
        assert!((parse_timescale("1ns").unwrap() - 1000.0).abs() < 0.001);
        assert!((parse_timescale("100ps").unwrap() - 100.0).abs() < 0.001);
        assert!((parse_timescale("10ns").unwrap() - 10000.0).abs() < 0.001);
        assert!((parse_timescale("1us").unwrap() - 1e6).abs() < 0.001);
    }

    #[test]
    fn test_parse_inv_chain_sdf() {
        let sdf_content = include_str!("../tests/timing_test/inv_chain_pnr/inv_chain_test.sdf");
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ).unwrap();

        assert_eq!(sdf.design, "inv_chain");
        // 19 cells: dff_in + 16 inverters + dff_out + top-level (interconnects)
        assert_eq!(sdf.cells.len(), 19);

        // Check dff_in IOPATH
        let dff_in = sdf.get_cell("dff_in").unwrap();
        assert_eq!(dff_in.cell_type, "sky130_fd_sc_hd__dfxtp_1");
        assert_eq!(dff_in.iopaths.len(), 1);
        assert_eq!(dff_in.iopaths[0].input_pin, "CLK");
        assert_eq!(dff_in.iopaths[0].output_pin, "Q");
        // 0.350ns = 350ps at typ corner (timescale 1ns)
        assert_eq!(dff_in.iopaths[0].delay.rise_ps, 350);
        assert_eq!(dff_in.iopaths[0].delay.fall_ps, 330);

        // Check timing checks
        assert_eq!(dff_in.timing_checks.len(), 2);
        let setup = dff_in.timing_checks.iter().find(|c| c.check_type == TimingCheckType::Setup).unwrap();
        assert_eq!(setup.value_ps, 80); // 0.080ns = 80ps
        let hold = dff_in.timing_checks.iter().find(|c| c.check_type == TimingCheckType::Hold).unwrap();
        assert_eq!(hold.value_ps, -30); // -0.030ns = -30ps

        // Check inverter i0
        let i0 = sdf.get_cell("i0").unwrap();
        assert_eq!(i0.cell_type, "sky130_fd_sc_hd__inv_1");
        assert_eq!(i0.iopaths.len(), 1);
        assert_eq!(i0.iopaths[0].delay.rise_ps, 50); // 0.050ns = 50ps
        assert_eq!(i0.iopaths[0].delay.fall_ps, 40); // 0.040ns = 40ps

        // Check top-level cell (interconnects)
        let top = sdf.get_cell("").unwrap();
        assert_eq!(top.interconnects.len(), 17);
        assert_eq!(top.interconnects[0].source, "dff_in.Q");
        assert_eq!(top.interconnects[0].dest, "i0.A");
        assert_eq!(top.interconnects[0].delay.rise_ps, 15); // 0.015ns = 15ps
    }

    #[test]
    fn test_corner_selection() {
        let sdf_content = include_str!("../tests/timing_test/inv_chain_pnr/inv_chain_test.sdf");

        let sdf_min = SdfFile::parse_str(sdf_content, SdfCorner::Min).unwrap();
        let sdf_typ = SdfFile::parse_str(sdf_content, SdfCorner::Typ).unwrap();
        let sdf_max = SdfFile::parse_str(sdf_content, SdfCorner::Max).unwrap();

        let i0_min = sdf_min.get_cell("i0").unwrap();
        let i0_typ = sdf_typ.get_cell("i0").unwrap();
        let i0_max = sdf_max.get_cell("i0").unwrap();

        // (0.040:0.050:0.060) rise, (0.030:0.040:0.055) fall
        assert_eq!(i0_min.iopaths[0].delay.rise_ps, 40);
        assert_eq!(i0_typ.iopaths[0].delay.rise_ps, 50);
        assert_eq!(i0_max.iopaths[0].delay.rise_ps, 60);

        assert_eq!(i0_min.iopaths[0].delay.fall_ps, 30);
        assert_eq!(i0_typ.iopaths[0].delay.fall_ps, 40);
        assert_eq!(i0_max.iopaths[0].delay.fall_ps, 55);
    }

    #[test]
    fn test_edge_qualified_timingcheck() {
        // OpenSTA outputs edge specifiers on both data and clock pins:
        // (SETUP (posedge D) (posedge CLK) (0.056::0.057))
        let sdf = r#"(DELAYFILE
            (SDFVERSION "3.0")
            (DESIGN "test")
            (TIMESCALE 1ns)
            (CELL (CELLTYPE "sky130_fd_sc_hd__dfxtp_1")
                (INSTANCE dff0)
                (TIMINGCHECK
                    (HOLD (posedge D) (posedge CLK) (-0.033::-0.034))
                    (HOLD (negedge D) (posedge CLK) (-0.051::-0.053))
                    (SETUP (posedge D) (posedge CLK) (0.056::0.057))
                    (SETUP (negedge D) (posedge CLK) (0.082::0.083))
                )
            )
        )"#;
        let parsed = SdfFile::parse_str(sdf, SdfCorner::Typ).unwrap();
        let cell = parsed.get_cell("dff0").unwrap();
        assert_eq!(cell.timing_checks.len(), 4);

        // Verify edge-qualified data pins are parsed correctly
        let setups: Vec<_> = cell.timing_checks.iter()
            .filter(|c| c.check_type == TimingCheckType::Setup)
            .collect();
        assert_eq!(setups.len(), 2);
        assert_eq!(setups[0].data_pin, "D");
        assert_eq!(setups[0].clock_edge, "CLK");
    }

    #[test]
    fn test_summary() {
        let sdf_content = include_str!("../tests/timing_test/inv_chain_pnr/inv_chain_test.sdf");
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ).unwrap();
        let summary = sdf.summary();
        assert!(summary.contains("19 cells"), "summary: {}", summary);
        assert!(summary.contains("18 IOPATH"), "summary: {}", summary);
        assert!(summary.contains("17 INTERCONNECT"), "summary: {}", summary);
        assert!(summary.contains("4 timing checks"), "summary: {}", summary);
    }
}
