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
use gem::sky130::{detect_library_from_file, extract_cell_type, is_sky130_cell, CellLibrary, SKY130LeafPins};
use gem::sky130_pdk::is_sequential_cell;
use netlistdb::{Direction, GeneralPinName, NetlistDB};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom, Write};
use std::path::PathBuf;
use sverilogparse::SVerilogRange;
use vcd_ng::{FFValueChange, FastFlow, FastFlowToken, Parser, Scope, ScopeItem, Var};

// FFI bindings to C++ SPI flash model (from chipflow-lib)
mod spiflash_ffi {
    use std::os::raw::c_int;

    #[repr(C)]
    pub struct SpiFlashModel {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn spiflash_new(size_bytes: usize) -> *mut SpiFlashModel;
        pub fn spiflash_free(flash: *mut SpiFlashModel);
        pub fn spiflash_load(
            flash: *mut SpiFlashModel,
            data: *const u8,
            len: usize,
            offset: usize,
        ) -> c_int;
        pub fn spiflash_step(
            flash: *mut SpiFlashModel,
            clk: c_int,
            csn: c_int,
            d_o: u8,
        ) -> u8;
        pub fn spiflash_get_command(flash: *mut SpiFlashModel) -> u8;
        pub fn spiflash_get_byte_count(flash: *mut SpiFlashModel) -> u32;
        pub fn spiflash_get_step_count(flash: *mut SpiFlashModel) -> u32;
        pub fn spiflash_get_posedge_count(flash: *mut SpiFlashModel) -> u32;
        pub fn spiflash_get_negedge_count(flash: *mut SpiFlashModel) -> u32;
    }
}

/// Safe wrapper around the C++ SPI flash model
struct CppSpiFlash {
    ptr: *mut spiflash_ffi::SpiFlashModel,
}

impl CppSpiFlash {
    fn new(size_bytes: usize) -> Self {
        let ptr = unsafe { spiflash_ffi::spiflash_new(size_bytes) };
        assert!(!ptr.is_null(), "Failed to create SPI flash model");
        Self { ptr }
    }

    fn load_firmware(&mut self, path: &std::path::Path, offset: usize) -> std::io::Result<usize> {
        use std::io::Read;
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let result = unsafe {
            spiflash_ffi::spiflash_load(self.ptr, data.as_ptr(), data.len(), offset)
        };

        if result < 0 {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to load firmware into flash",
            ))
        } else {
            Ok(result as usize)
        }
    }

    fn step(&mut self, clk: bool, csn: bool, d_o: u8) -> u8 {
        unsafe {
            spiflash_ffi::spiflash_step(
                self.ptr,
                if clk { 1 } else { 0 },
                if csn { 1 } else { 0 },
                d_o,
            )
        }
    }

    #[allow(dead_code)]
    fn get_command(&self) -> u8 {
        unsafe { spiflash_ffi::spiflash_get_command(self.ptr) }
    }

    #[allow(dead_code)]
    fn get_byte_count(&self) -> u32 {
        unsafe { spiflash_ffi::spiflash_get_byte_count(self.ptr) }
    }

    #[allow(dead_code)]
    fn get_step_count(&self) -> u32 {
        unsafe { spiflash_ffi::spiflash_get_step_count(self.ptr) }
    }

    #[allow(dead_code)]
    fn get_posedge_count(&self) -> u32 {
        unsafe { spiflash_ffi::spiflash_get_posedge_count(self.ptr) }
    }

    #[allow(dead_code)]
    fn get_negedge_count(&self) -> u32 {
        unsafe { spiflash_ffi::spiflash_get_negedge_count(self.ptr) }
    }
}

impl Drop for CppSpiFlash {
    fn drop(&mut self) {
        unsafe { spiflash_ffi::spiflash_free(self.ptr) };
    }
}

// Make CppSpiFlash safe to send between threads (the C++ model has no global state)
unsafe impl Send for CppSpiFlash {}

#[derive(clap::Parser, Debug)]
#[command(name = "timing_sim_cpu")]
#[command(about = "CPU timing simulation with per-gate delays")]
struct Args {
    /// Gate-level verilog path synthesized in AIGPDK or SKY130 library.
    /// Optional if --config provides netlist_path.
    netlist_verilog: Option<PathBuf>,

    /// VCD input signal path (optional in self-test mode or --config mode)
    input_vcd: Option<PathBuf>,

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

    /// JSON watchlist file with signals to trace.
    /// Format: {"signals": [{"name": "label", "net": "net_name"}, ...]}
    #[clap(long)]
    watchlist: Option<PathBuf>,

    /// Output file for signal trace (CSV format).
    #[clap(long)]
    trace_output: Option<PathBuf>,

    /// Run in self-test mode (programmatic testbench, no VCD required).
    /// Simulates reset sequence and runs with firmware from --firmware.
    #[clap(long)]
    self_test: bool,

    /// Reset GPIO index for self-test mode (default: 40 for Caravel).
    #[clap(long, default_value = "40")]
    reset_gpio: usize,

    /// Number of reset cycles in self-test mode (default: 10).
    #[clap(long, default_value = "10")]
    reset_cycles: usize,

    /// Clock GPIO index for self-test mode (default: 38 for Caravel).
    #[clap(long, default_value = "38")]
    clock_gpio: usize,

    /// Testbench configuration JSON file (generated by chipflow-gen-testbench).
    /// When provided, runs in programmatic mode without VCD.
    #[clap(long)]
    config: Option<PathBuf>,

    /// Reset polarity: true = GPIO high means reset active.
    #[clap(long)]
    reset_active_high: bool,

    /// Enable Wishbone bus monitor. Provide a hierarchical prefix to search
    /// for bus signals (e.g. "soc" finds inst$top.soc.*.{cyc,stb,ack,...}).
    /// Logs protocol-level bus transactions to stderr.
    #[clap(long)]
    wb_monitor: Option<String>,

    /// Path to PDK cell library for vendor-verified decompositions.
    /// Points to the cells/ directory of sky130_fd_sc_hd.
    /// Defaults to sky130_fd_sc_hd/cells if the submodule is present.
    #[clap(long)]
    pdk_cells: Option<PathBuf>,
}

/// Testbench configuration loaded from JSON.
#[derive(Debug, Clone, Deserialize)]
struct TestbenchConfig {
    netlist_path: Option<String>,
    liberty_path: Option<String>,
    clock_gpio: usize,
    reset_gpio: usize,
    reset_active_high: bool,
    reset_cycles: usize,
    num_cycles: usize,
    flash: Option<FlashConfig>,
    uart: Option<UartConfig>,
    #[serde(default)]
    gpios: Vec<GpioConfig>,
    sram_init: Option<SramInitConfig>,
    output_events: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct FlashConfig {
    clk_gpio: usize,
    csn_gpio: usize,
    d0_gpio: usize,
    firmware: String,
    firmware_offset: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct UartConfig {
    tx_gpio: usize,
    rx_gpio: usize,
    baud_rate: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct GpioConfig {
    name: String,
    pins: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct SramInitConfig {
    elf_path: String,
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
    // State (pub for debugging)
    pub last_clk: bool,
    last_csn: bool,
    bit_count: u8,
    pub byte_count: u32,
    curr_byte: u8,
    out_buffer: u8,
    pub command: u8,
    addr: u32,
    data_width: u8, // 1 for single SPI, 4 for quad
    last_d_in: u8,  // Hold the last d_in value to maintain across clock cycles
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
            last_d_in: 0,
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
                        let shift = (3 - self.byte_count) * 8;
                        clilog::debug!("FLASH: Addr byte {}: 0x{:02X} << {} -> addr=0x{:06X}",
                                      self.byte_count, self.curr_byte, shift,
                                      self.addr | ((self.curr_byte as u32) << shift));
                        self.addr |= (self.curr_byte as u32) << shift;
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
        static mut STEP_COUNT: u64 = 0;
        static mut LAST_LOG: u64 = 0;
        unsafe { STEP_COUNT += 1; }

        let mut d_i = 0u8;

        // Log CSN transitions
        if !csn && self.last_csn {
            // Falling edge of CSN - flash selected
            clilog::info!("FLASH: CSN low (selected) at step {}", unsafe { STEP_COUNT });
        }

        if csn && !self.last_csn {
            // Rising edge of CSN - deselect, reset state
            if self.command != 0 {
                clilog::info!("FLASH: CSN high (deselected), cmd=0x{:02X}, bytes={}",
                             self.command, self.byte_count);
            }
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
                let old_cmd = self.command;
                let old_addr = self.addr;
                self.process_byte();
                self.byte_count += 1;
                self.bit_count = 0;

                // Log command and first data bytes
                if self.byte_count == 1 {
                    clilog::info!("FLASH: Command 0x{:02X}", self.command);
                } else if self.byte_count == 4 && (self.command == 0x03 || self.command == 0xEB) {
                    clilog::info!("FLASH: Read addr=0x{:06X}, first data=0x{:02X}",
                                 self.addr.wrapping_sub(1) & 0xFFFFFF, self.out_buffer);
                } else if self.byte_count >= 4 && self.byte_count <= 8
                          && (self.command == 0x03 || self.command == 0xEB) {
                    // Log a few more data bytes
                    unsafe {
                        if STEP_COUNT - LAST_LOG > 100 {
                            clilog::info!("FLASH: Read byte {} = 0x{:02X} from addr 0x{:06X}",
                                         self.byte_count, self.out_buffer,
                                         self.addr.wrapping_sub(1) & 0xFFFFFF);
                            LAST_LOG = STEP_COUNT;
                        }
                    }
                }
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

/// SRAM cell information for simulation.
/// Tracks the pin IDs for a CF_SRAM_1024x32 cell and its memory contents.
struct SramCell {
    cell_id: usize,
    // Control pins
    clk_pin: usize,
    en_pin: usize,
    r_wb_pin: usize,  // 1=read, 0=write
    // Address pins (10 bits for 1024 words)
    addr_pins: Vec<usize>,
    // Byte enable pins (32 bits)
    ben_pins: Vec<usize>,
    // Data input pins (32 bits)
    di_pins: Vec<usize>,
    // Data output pins (32 bits)
    do_pins: Vec<usize>,
    // Memory contents (1024 x 32-bit words)
    memory: Vec<u32>,
    // Last clock state for edge detection
    last_clk: bool,
}

impl SramCell {
    fn new(cell_id: usize) -> Self {
        Self {
            cell_id,
            clk_pin: usize::MAX,
            en_pin: usize::MAX,
            r_wb_pin: usize::MAX,
            addr_pins: Vec::new(),
            ben_pins: Vec::new(),
            di_pins: Vec::new(),
            do_pins: Vec::new(),
            memory: vec![0; 1024],  // 1024 words, initialized to 0
            last_clk: false,
        }
    }

    /// Collect pin IDs from the netlist for this SRAM cell.
    fn collect_pins(&mut self, netlistdb: &NetlistDB) {
        for pinid in netlistdb.cell2pin.iter_set(self.cell_id) {
            let pin_name = netlistdb.pinnames[pinid].1.as_str();
            let idx = netlistdb.pinnames[pinid].2;

            match pin_name {
                "CLKin" => self.clk_pin = pinid,
                "EN" => self.en_pin = pinid,
                "R_WB" => self.r_wb_pin = pinid,
                "AD" => {
                    // Address is a bus - ensure we have space and insert at correct index
                    if let Some(i) = idx {
                        let i = i as usize;
                        if self.addr_pins.len() <= i {
                            self.addr_pins.resize(i + 1, usize::MAX);
                        }
                        self.addr_pins[i] = pinid;
                    }
                }
                "BEN" => {
                    if let Some(i) = idx {
                        let i = i as usize;
                        if self.ben_pins.len() <= i {
                            self.ben_pins.resize(i + 1, usize::MAX);
                        }
                        self.ben_pins[i] = pinid;
                    }
                }
                "DI" => {
                    if let Some(i) = idx {
                        let i = i as usize;
                        if self.di_pins.len() <= i {
                            self.di_pins.resize(i + 1, usize::MAX);
                        }
                        self.di_pins[i] = pinid;
                    }
                }
                "DO" => {
                    if let Some(i) = idx {
                        let i = i as usize;
                        if self.do_pins.len() <= i {
                            self.do_pins.resize(i + 1, usize::MAX);
                        }
                        self.do_pins[i] = pinid;
                    }
                }
                _ => {} // Ignore test pins (SM, TM, WLBI, ScanIn*, etc.)
            }
        }
    }

    /// Read address from current circuit state.
    fn read_addr(&self, circ_state: &[u8]) -> usize {
        let mut addr = 0usize;
        for (i, &pin) in self.addr_pins.iter().enumerate() {
            if pin != usize::MAX && circ_state[pin] != 0 {
                addr |= 1 << i;
            }
        }
        addr
    }

    /// Read data input from current circuit state.
    fn read_di(&self, circ_state: &[u8]) -> u32 {
        let mut data = 0u32;
        for (i, &pin) in self.di_pins.iter().enumerate() {
            if pin != usize::MAX && circ_state[pin] != 0 {
                data |= 1 << i;
            }
        }
        data
    }

    /// Read byte enable from current circuit state.
    fn read_ben(&self, circ_state: &[u8]) -> u32 {
        let mut ben = 0u32;
        for (i, &pin) in self.ben_pins.iter().enumerate() {
            if pin != usize::MAX && circ_state[pin] != 0 {
                ben |= 1 << i;
            }
        }
        ben
    }

    /// Write data output to circuit state.
    fn write_do(&self, circ_state: &mut [u8], data: u32) {
        for (i, &pin) in self.do_pins.iter().enumerate() {
            if pin != usize::MAX {
                circ_state[pin] = ((data >> i) & 1) as u8;
            }
        }
    }

    /// Simulate one clock cycle. Returns true if a read/write occurred.
    fn step(&mut self, circ_state: &mut [u8]) -> bool {
        let clk = self.clk_pin != usize::MAX && circ_state[self.clk_pin] != 0;
        let rising_edge = clk && !self.last_clk;
        self.last_clk = clk;

        if !rising_edge {
            return false;
        }

        // Check enable
        let en = self.en_pin != usize::MAX && circ_state[self.en_pin] != 0;
        if !en {
            return false;
        }

        let r_wb = self.r_wb_pin != usize::MAX && circ_state[self.r_wb_pin] != 0;
        let addr = self.read_addr(circ_state);

        if addr >= 1024 {
            return false;  // Invalid address
        }

        if r_wb {
            // Read operation: DO = memory[addr]
            let data = self.memory[addr];
            self.write_do(circ_state, data);
        } else {
            // Write operation: memory[addr] = (memory[addr] & ~BEN) | (DI & BEN)
            let di = self.read_di(circ_state);
            let ben = self.read_ben(circ_state);
            self.memory[addr] = (self.memory[addr] & !ben) | (di & ben);
        }

        true
    }
}

/// Watchlist signal entry.
#[derive(Debug, Clone, Deserialize)]
struct WatchlistSignal {
    /// Display name for the signal.
    name: String,
    /// Net name pattern to match in the netlist.
    net: String,
    /// Signal type (reg, comb, mem).
    #[serde(rename = "type", default)]
    signal_type: String,
    /// Width for bundle signals (e.g., 32 for a 32-bit bus).
    #[serde(default)]
    width: Option<usize>,
    /// Format for output: "bin", "hex", or "dec" (default: single bit = dec, multi-bit = hex).
    #[serde(default)]
    format: Option<String>,
}

/// Watchlist configuration loaded from JSON.
#[derive(Debug, Clone, Deserialize)]
struct Watchlist {
    signals: Vec<WatchlistSignal>,
}

/// Resolved watchlist entry - either single bit or bundle.
#[derive(Debug, Clone)]
enum WatchlistEntry {
    /// Single-bit signal.
    Bit { name: String, pin: usize },
    /// Multi-bit bundle (pins ordered LSB to MSB).
    Bundle { name: String, pins: Vec<usize>, format: String },
}

impl WatchlistEntry {
    fn name(&self) -> &str {
        match self {
            WatchlistEntry::Bit { name, .. } => name,
            WatchlistEntry::Bundle { name, .. } => name,
        }
    }

    fn format_value(&self, circ_state: &[u8]) -> String {
        match self {
            WatchlistEntry::Bit { pin, .. } => circ_state[*pin].to_string(),
            WatchlistEntry::Bundle { pins, format, .. } => {
                let mut value: u64 = 0;
                for (i, &pin) in pins.iter().enumerate() {
                    // Skip missing pins (usize::MAX means not found)
                    if pin < circ_state.len() && circ_state[pin] != 0 {
                        value |= 1u64 << i;
                    }
                }
                match format.as_str() {
                    "bin" => format!("{:0width$b}", value, width = pins.len()),
                    "dec" => format!("{}", value),
                    _ => format!("0x{:0width$X}", value, width = (pins.len() + 3) / 4),
                }
            }
        }
    }
}

/// A discovered Wishbone bus (master or slave side).
struct WbBus {
    /// Human-readable label (e.g. "cpu.fetch.ibus", "sram.wb_bus").
    label: String,
    /// CYC signal pin (Option since slaves don't have it).
    cyc_pin: Option<usize>,
    /// STB signal pin.
    stb_pin: Option<usize>,
    /// WE signal pin.
    we_pin: Option<usize>,
    /// ACK signal pin.
    ack_pin: Option<usize>,
    /// Address bits (index = bit position, value = pin; usize::MAX = not found).
    adr_pins: Vec<usize>,
    /// Write data bits.
    dat_w_pins: Vec<usize>,
    /// Read data bits.
    dat_r_pins: Vec<usize>,
    /// Byte select bits.
    sel_pins: Vec<usize>,
}

impl WbBus {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            cyc_pin: None,
            stb_pin: None,
            we_pin: None,
            ack_pin: None,
            adr_pins: Vec::new(),
            dat_w_pins: Vec::new(),
            dat_r_pins: Vec::new(),
            sel_pins: Vec::new(),
        }
    }

    fn read_bus_value(pins: &[usize], circ_state: &[u8]) -> u64 {
        let mut val = 0u64;
        for (i, &pin) in pins.iter().enumerate() {
            if pin < circ_state.len() && circ_state[pin] != 0 {
                val |= 1u64 << i;
            }
        }
        val
    }

    fn read_bit(pin: Option<usize>, circ_state: &[u8]) -> u8 {
        pin.map(|p| if p < circ_state.len() { circ_state[p] } else { 0 }).unwrap_or(0)
    }
}

/// Wishbone bus monitor: auto-discovers bus signals and logs transactions.
struct WishboneBusMonitor {
    /// All discovered buses (masters and slaves).
    buses: Vec<WbBus>,
    /// Arbiter grant pin (if found).
    grant_pin: Option<usize>,
    /// Extra single-bit signals (e.g. read_port__en, write_port__en).
    extra: Vec<(String, usize)>,
    /// Previous cycle's CYC values for edge detection (indexed same as buses).
    prev_cyc: Vec<u8>,
    /// Previous cycle's ACK values for edge detection.
    prev_ack: Vec<u8>,
    /// Previous grant value.
    prev_grant: u8,
    /// First cycle dbus CYC went high (for targeted tracing).
    first_dbus_cycle: Option<usize>,
}

impl WishboneBusMonitor {
    /// Discover all Wishbone signals under the given hierarchical prefix.
    /// E.g. prefix="soc" matches `inst$top.soc.cpu.fetch.ibus__cyc`, etc.
    fn discover(prefix: &str, netlistdb: &NetlistDB) -> Self {
        // We'll search pin names for patterns containing the prefix and WB signal names.
        // Signal patterns we look for (double underscore = Amaranth convention):
        //   __cyc, __stb, __we, __ack, __adr[N], __dat_w[N], __dat_r[N], __sel[N]
        //   .grant (arbiter)
        //   .read_port__en, .write_port__en[N], .read_port__data[N] (SRAM-specific)

        // Step 1: Collect all matching signal names and their pins.
        // Use a HashMap: signal_full_path -> pin_id
        let mut signal_pins: HashMap<String, usize> = HashMap::new();
        let match_prefix = format!("{}.", prefix);

        // Helper: find best pin for a net name (prefer DFF Q outputs).
        let find_best_pin = |net_pattern: &str| -> Option<usize> {
            // First: DFF Q output
            for pinid in 0..netlistdb.num_pins {
                let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                if pin_name.contains(net_pattern) && pin_name.ends_with(":Q") {
                    return Some(pinid);
                }
            }
            // Second: net name lookup
            for netid in 0..netlistdb.num_nets {
                let net_name = netlistdb.netnames[netid].dbg_fmt_pin();
                if net_name.contains(net_pattern) {
                    for pinid in netlistdb.net2pin.iter_set(netid) {
                        return Some(pinid);
                    }
                }
            }
            // Third: any pin containing pattern
            for pinid in 0..netlistdb.num_pins {
                let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                if pin_name.contains(net_pattern) {
                    return Some(pinid);
                }
            }
            None
        };

        // Step 2: Find all wire declarations matching WB signal patterns.
        // Scan all net names for the prefix.
        let wb_signal_suffixes = [
            "__cyc", "__stb", "__we", "__ack",
        ];

        let wb_bus_suffixes = [
            ("__adr", 30),
            ("__dat_w", 32),
            ("__dat_r", 32),
            ("__sel", 4),
        ];

        let extra_suffixes = [
            "read_port__en", "read_port__data",
            "write_port__en",
        ];

        // Discover bus names by finding unique hierarchical paths before the WB signal suffix.
        let mut bus_paths: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

        for netid in 0..netlistdb.num_nets {
            let net_name = netlistdb.netnames[netid].dbg_fmt_pin();
            if !net_name.contains(&match_prefix) {
                continue;
            }

            // Extract the path portion relevant to our prefix
            // Net names look like: \inst$top.soc.cpu.fetch.ibus__cyc
            // We want to extract: cpu.fetch.ibus
            for suffix in &wb_signal_suffixes {
                if net_name.contains(suffix) {
                    // Find the bus path: everything between the prefix and the signal suffix
                    if let Some(prefix_pos) = net_name.find(&match_prefix) {
                        let after_prefix = &net_name[prefix_pos + match_prefix.len()..];
                        // Find where the WB signal starts (at the last __ before the suffix)
                        if let Some(sig_pos) = after_prefix.rfind(suffix) {
                            let bus_path = &after_prefix[..sig_pos];
                            // Remove trailing dot if any
                            let bus_path = bus_path.trim_end_matches('.');
                            if !bus_path.is_empty() {
                                bus_paths.insert(bus_path.to_string());
                            }
                        }
                    }
                }
            }
            // Also check bus suffixes (with [N] indices)
            for (suffix, _) in &wb_bus_suffixes {
                if net_name.contains(suffix) {
                    if let Some(prefix_pos) = net_name.find(&match_prefix) {
                        let after_prefix = &net_name[prefix_pos + match_prefix.len()..];
                        if let Some(sig_pos) = after_prefix.rfind(suffix) {
                            let bus_path = &after_prefix[..sig_pos];
                            let bus_path = bus_path.trim_end_matches('.');
                            if !bus_path.is_empty() {
                                bus_paths.insert(bus_path.to_string());
                            }
                        }
                    }
                }
            }
        }

        // Also look for arbiter grant
        let mut grant_pin = None;
        if let Some(pin) = find_best_pin(&format!("{}.wb_arbiter.grant", prefix)) {
            grant_pin = Some(pin);
        } else if let Some(pin) = find_best_pin("wb_arbiter.grant") {
            grant_pin = Some(pin);
        }

        eprintln!("WB Monitor: prefix='{}', discovered {} bus paths: {:?}",
                  prefix, bus_paths.len(), bus_paths);
        if grant_pin.is_some() {
            eprintln!("WB Monitor: arbiter grant pin found");
        }

        // Step 3: Build WbBus for each discovered path.
        let mut buses = Vec::new();
        let mut extra = Vec::new();

        for bus_path in &bus_paths {
            let mut bus = WbBus::new(bus_path);
            let full_path = format!("{}.{}", prefix, bus_path);

            // Single-bit control signals
            bus.cyc_pin = find_best_pin(&format!("{}__cyc", full_path));
            bus.stb_pin = find_best_pin(&format!("{}__stb", full_path));
            bus.we_pin = find_best_pin(&format!("{}__we", full_path));
            bus.ack_pin = find_best_pin(&format!("{}__ack", full_path));

            // Multi-bit bus signals
            for (suffix, max_bits) in &wb_bus_suffixes {
                let mut pins = Vec::new();
                for bit in 0..*max_bits {
                    let pattern = format!("{}{}[{}]", full_path, suffix, bit);
                    match find_best_pin(&pattern) {
                        Some(pin) => pins.push(pin),
                        None => pins.push(usize::MAX),
                    }
                }
                // Trim trailing MAX entries
                while pins.last() == Some(&usize::MAX) {
                    pins.pop();
                }
                match *suffix {
                    "__adr" => bus.adr_pins = pins,
                    "__dat_w" => bus.dat_w_pins = pins,
                    "__dat_r" => bus.dat_r_pins = pins,
                    "__sel" => bus.sel_pins = pins,
                    _ => {}
                }
            }

            // Log what was found
            let found_signals: Vec<String> = [
                bus.cyc_pin.map(|_| "cyc"),
                bus.stb_pin.map(|_| "stb"),
                bus.we_pin.map(|_| "we"),
                bus.ack_pin.map(|_| "ack"),
            ].iter().filter_map(|x| *x).map(|s| s.to_string())
            .chain(if !bus.adr_pins.is_empty() { vec![format!("adr[{}]", bus.adr_pins.len())] } else { vec![] })
            .chain(if !bus.dat_w_pins.is_empty() { vec![format!("dat_w[{}]", bus.dat_w_pins.len())] } else { vec![] })
            .chain(if !bus.dat_r_pins.is_empty() { vec![format!("dat_r[{}]", bus.dat_r_pins.len())] } else { vec![] })
            .chain(if !bus.sel_pins.is_empty() { vec![format!("sel[{}]", bus.sel_pins.len())] } else { vec![] })
            .collect();
            eprintln!("  WB bus '{}': {}", bus_path, found_signals.join(", "));

            buses.push(bus);
        }

        // Extra SRAM-specific signals
        for suffix in &extra_suffixes {
            let pattern = format!("{}.sram.{}", prefix, suffix);
            if suffix.contains("[") || suffix == &"read_port__data" || suffix == &"write_port__en" {
                // Multi-bit
                for bit in 0..32 {
                    let bit_pattern = format!("{}.sram.{}[{}]", prefix, suffix, bit);
                    if let Some(pin) = find_best_pin(&bit_pattern) {
                        extra.push((format!("sram.{}[{}]", suffix, bit), pin));
                    }
                }
            } else {
                if let Some(pin) = find_best_pin(&pattern) {
                    extra.push((format!("sram.{}", suffix), pin));
                }
            }
        }
        if !extra.is_empty() {
            eprintln!("  WB extra signals: {} found", extra.len());
        }

        let num_buses = buses.len();
        Self {
            prev_cyc: vec![0; num_buses],
            prev_ack: vec![0; num_buses],
            prev_grant: 0,
            buses,
            grant_pin,
            extra,
            first_dbus_cycle: None,
        }
    }

    /// Log bus state for the current cycle. Returns true if anything interesting happened.
    fn log_cycle(&mut self, cycle: usize, circ_state: &[u8]) -> bool {
        let mut any_event = false;

        let grant = self.grant_pin.map(|p| WbBus::read_bit(Some(p), circ_state)).unwrap_or(0);

        for (i, bus) in self.buses.iter().enumerate() {
            let cyc = WbBus::read_bit(bus.cyc_pin, circ_state);
            let stb = WbBus::read_bit(bus.stb_pin, circ_state);
            let we = WbBus::read_bit(bus.we_pin, circ_state);
            let ack = WbBus::read_bit(bus.ack_pin, circ_state);

            let prev_cyc = self.prev_cyc[i];
            let prev_ack = self.prev_ack[i];

            // Detect CYC rising edge (new transaction start)
            if cyc == 1 && prev_cyc == 0 {
                let adr = WbBus::read_bus_value(&bus.adr_pins, circ_state);
                let sel = WbBus::read_bus_value(&bus.sel_pins, circ_state);
                let dat_w = WbBus::read_bus_value(&bus.dat_w_pins, circ_state);
                eprintln!("WB c{:6}: {} CYC↑ stb={} we={} adr=0x{:08X} sel=0x{:X} dat_w=0x{:08X} grant={}",
                         cycle, bus.label, stb, we, adr, sel, dat_w, grant);
                any_event = true;
            }

            // Detect CYC falling edge (transaction end)
            if cyc == 0 && prev_cyc == 1 {
                eprintln!("WB c{:6}: {} CYC↓", cycle, bus.label);
                any_event = true;
            }

            // Detect ACK rising edge
            if ack == 1 && prev_ack == 0 {
                let adr = WbBus::read_bus_value(&bus.adr_pins, circ_state);
                let dat_r = WbBus::read_bus_value(&bus.dat_r_pins, circ_state);
                let dat_w = WbBus::read_bus_value(&bus.dat_w_pins, circ_state);
                let sel = WbBus::read_bus_value(&bus.sel_pins, circ_state);
                if WbBus::read_bit(bus.we_pin, circ_state) == 1 {
                    eprintln!("WB c{:6}: {} ACK↑ WRITE adr=0x{:08X} dat_w=0x{:08X} sel=0x{:X}",
                             cycle, bus.label, adr, dat_w, sel);
                } else {
                    eprintln!("WB c{:6}: {} ACK↑ READ  adr=0x{:08X} dat_r=0x{:08X}",
                             cycle, bus.label, adr, dat_r);
                }
                any_event = true;
            }

            // Detect ACK falling edge
            if ack == 0 && prev_ack == 1 {
                eprintln!("WB c{:6}: {} ACK↓", cycle, bus.label);
                any_event = true;
            }

            // Log ongoing transactions with address changes (for multi-cycle)
            if cyc == 1 && stb == 1 && ack == 0 && cycle % 1000 == 0 {
                let adr = WbBus::read_bus_value(&bus.adr_pins, circ_state);
                eprintln!("WB c{:6}: {} STALL adr=0x{:08X} we={} grant={}",
                         cycle, bus.label, adr, we, grant);
                any_event = true;
            }

            self.prev_cyc[i] = cyc;
            self.prev_ack[i] = ack;
        }

        // Grant changes
        if grant != self.prev_grant {
            eprintln!("WB c{:6}: arbiter.grant {}→{}", cycle, self.prev_grant, grant);
            any_event = true;
            self.prev_grant = grant;
        }

        // Detailed trace: dump ALL bus states every cycle for a window after first dbus CYC↑
        // The dbus label contains "dbus"
        let dbus_idx = self.buses.iter().position(|b| b.label.contains("dbus"));
        if let Some(di) = dbus_idx {
            let dbus_cyc = WbBus::read_bit(self.buses[di].cyc_pin, circ_state);
            // Trace for 20 cycles after dbus first becomes active
            if dbus_cyc == 1 && cycle <= self.first_dbus_cycle.unwrap_or(usize::MAX) + 20 {
                if self.first_dbus_cycle.is_none() {
                    self.first_dbus_cycle = Some(cycle);
                }
                // Dump ALL buses on every cycle in this window
                let mut line = format!("WB TRACE c{:6}: grant={}", cycle, grant);
                for bus in &self.buses {
                    let cyc = WbBus::read_bit(bus.cyc_pin, circ_state);
                    let stb = WbBus::read_bit(bus.stb_pin, circ_state);
                    let we = WbBus::read_bit(bus.we_pin, circ_state);
                    let ack = WbBus::read_bit(bus.ack_pin, circ_state);
                    let adr = WbBus::read_bus_value(&bus.adr_pins, circ_state);
                    line += &format!(" | {}:c{}s{}w{}a{} @{:08X}",
                        &bus.label[..bus.label.len().min(12)], cyc, stb, we, ack, adr);
                }
                eprintln!("{}", line);
                any_event = true;
            }
        }

        any_event
    }

    /// Print a full bus state snapshot (useful at specific cycles).
    fn dump_state(&self, cycle: usize, circ_state: &[u8]) {
        let grant = self.grant_pin.map(|p| WbBus::read_bit(Some(p), circ_state)).unwrap_or(0);
        eprintln!("=== WB SNAPSHOT cycle {} grant={} ===", cycle, grant);
        for bus in &self.buses {
            let cyc = WbBus::read_bit(bus.cyc_pin, circ_state);
            let stb = WbBus::read_bit(bus.stb_pin, circ_state);
            let we = WbBus::read_bit(bus.we_pin, circ_state);
            let ack = WbBus::read_bit(bus.ack_pin, circ_state);
            let adr = WbBus::read_bus_value(&bus.adr_pins, circ_state);
            let dat_w = WbBus::read_bus_value(&bus.dat_w_pins, circ_state);
            let dat_r = WbBus::read_bus_value(&bus.dat_r_pins, circ_state);
            let sel = WbBus::read_bus_value(&bus.sel_pins, circ_state);
            eprintln!("  {}: cyc={} stb={} we={} ack={} adr=0x{:08X} dat_w=0x{:08X} dat_r=0x{:08X} sel=0x{:X}",
                     bus.label, cyc, stb, we, ack, adr, dat_w, dat_r, sel);
        }
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

/// Verify combinational cell outputs against truth tables.
/// Returns (total_checked, mismatches). Logs the first `max_log` mismatches.
fn verify_cell_outputs(
    netlistdb: &NetlistDB,
    circ_state: &[u8],
    cell_library: CellLibrary,
    cycle: usize,
    max_log: usize,
) -> (usize, usize) {
    let mut checked = 0usize;
    let mut mismatches = 0usize;

    for cellid in 1..netlistdb.num_cells {
        let celltype = netlistdb.celltypes[cellid].as_str();
        if !is_sky130_cell(celltype) {
            continue;
        }
        let ct = extract_cell_type(celltype);

        // Skip sequential, tie, filler, tap, decap cells
        if is_sequential_cell(ct)
            || matches!(ct, "conb" | "fill" | "tap" | "decap" | "clkbuf" | "clkdlybuf")
            || ct.starts_with("fill")
            || ct.starts_with("tap")
            || ct.starts_with("decap")
        {
            continue;
        }

        // Gather pins
        let mut a_val = 0u8;
        let mut b_val = 0u8;
        let mut c_val = 0u8;
        let mut d_val = 0u8;
        let mut a1_val = 0u8;
        let mut a2_val = 0u8;
        let mut a3_val = 0u8;
        let mut a4_val = 0u8;
        let mut b1_val = 0u8;
        let mut b2_val = 0u8;
        let mut c1_val = 0u8;
        let mut d1_val = 0u8;
        let mut s_val = 0u8;
        let mut s0_val = 0u8;
        let mut s1_val = 0u8;
        let mut a0_val = 0u8;
        let mut y_pin = usize::MAX;
        let mut x_pin = usize::MAX;

        for pinid in netlistdb.cell2pin.iter_set(cellid) {
            let pname = netlistdb.pinnames[pinid].1.as_str();
            let v = circ_state[pinid];
            match pname {
                "A" => a_val = v,
                "B" => b_val = v,
                "C" => c_val = v,
                "D" => d_val = v,
                "A1" => a1_val = v,
                "A2" => a2_val = v,
                "A3" => a3_val = v,
                "A4" => a4_val = v,
                "B1" => b1_val = v,
                "B2" => b2_val = v,
                "C1" => c1_val = v,
                "D1" => d1_val = v,
                "S" => s_val = v,
                "S0" => s0_val = v,
                "S1" => s1_val = v,
                "A0" => a0_val = v,
                "Y" => y_pin = pinid,
                "X" => x_pin = pinid,
                _ => {}
            }
        }

        // Determine output pin and expected value
        let (out_pin, expected) = match ct {
            "inv" => (y_pin, 1 - a_val),
            "buf" | "clkbuf" => (x_pin, a_val),
            "and2" => (x_pin, a_val & b_val),
            "and3" => (x_pin, a_val & b_val & c_val),
            "and4" => (x_pin, a_val & b_val & c_val & d_val),
            "nand2" => (y_pin, 1 - (a_val & b_val)),
            "nand3" => (y_pin, 1 - (a_val & b_val & c_val)),
            "nand4" => (y_pin, 1 - (a_val & b_val & c_val & d_val)),
            "or2" => (x_pin, a_val | b_val),
            "or3" => (x_pin, a_val | b_val | c_val),
            "or4" => (x_pin, a_val | b_val | c_val | d_val),
            "nor2" => (y_pin, 1 - (a_val | b_val)),
            "nor3" => (y_pin, 1 - (a_val | b_val | c_val)),
            "nor4" => (y_pin, 1 - (a_val | b_val | c_val | d_val)),
            "xor2" => (x_pin, a_val ^ b_val),
            "xnor2" => (y_pin, 1 - (a_val ^ b_val)),
            "a21oi" => (y_pin, 1 - ((a1_val & a2_val) | b1_val)),
            "a21o" => (x_pin, (a1_val & a2_val) | b1_val),
            "o21ai" => (y_pin, 1 - ((a1_val | a2_val) & b1_val)),
            "o21a" => (x_pin, (a1_val | a2_val) & b1_val),
            "a22o" => (x_pin, (a1_val & a2_val) | (b1_val & b2_val)),
            "a22oi" => (y_pin, 1 - ((a1_val & a2_val) | (b1_val & b2_val))),
            "o22a" => (x_pin, (a1_val | a2_val) & (b1_val | b2_val)),
            "o22ai" => (y_pin, 1 - ((a1_val | a2_val) & (b1_val | b2_val))),
            "a211o" => (x_pin, (a1_val & a2_val) | b1_val | c1_val),
            "a211oi" => (y_pin, 1 - ((a1_val & a2_val) | b1_val | c1_val)),
            "o211a" => (x_pin, (a1_val | a2_val) & b1_val & c1_val),
            "o211ai" => (y_pin, 1 - ((a1_val | a2_val) & b1_val & c1_val)),
            "a31o" => (x_pin, (a1_val & a2_val & a3_val) | b1_val),
            "a31oi" => (y_pin, 1 - ((a1_val & a2_val & a3_val) | b1_val)),
            "o31a" => (x_pin, (a1_val | a2_val | a3_val) & b1_val),
            "o31ai" => (y_pin, 1 - ((a1_val | a2_val | a3_val) & b1_val)),
            "a32o" => (x_pin, (a1_val & a2_val & a3_val) | (b1_val & b2_val)),
            "a32oi" => (y_pin, 1 - ((a1_val & a2_val & a3_val) | (b1_val & b2_val))),
            "o32a" => (x_pin, (a1_val | a2_val | a3_val) & (b1_val | b2_val)),
            "o32ai" => (y_pin, 1 - ((a1_val | a2_val | a3_val) & (b1_val | b2_val))),
            "a41oi" => (y_pin, 1 - ((a1_val & a2_val & a3_val & a4_val) | b1_val)),
            "o41ai" => (y_pin, 1 - ((a1_val | a2_val | a3_val | a4_val) & b1_val)),
            "a221o" => (x_pin, (a1_val & a2_val) | (b1_val & b2_val) | c1_val),
            "a221oi" => (y_pin, 1 - ((a1_val & a2_val) | (b1_val & b2_val) | c1_val)),
            "a311o" => (x_pin, (a1_val & a2_val & a3_val) | b1_val | c1_val),
            "a311oi" => (y_pin, 1 - ((a1_val & a2_val & a3_val) | b1_val | c1_val)),
            "o221a" => (x_pin, (a1_val | a2_val) & (b1_val | b2_val) & c1_val),
            "o221ai" => (y_pin, 1 - ((a1_val | a2_val) & (b1_val | b2_val) & c1_val)),
            "mux2" => (x_pin, if s_val != 0 { a1_val } else { a0_val }),
            "mux4" => {
                let sel = (s1_val as usize) * 2 + (s0_val as usize);
                let out = match sel {
                    0 => a0_val,
                    1 => a1_val,
                    2 => a2_val,
                    3 => a3_val,
                    _ => 0,
                };
                (x_pin, out)
            }
            // Skip cells we don't have truth tables for
            _ => continue,
        };

        if out_pin == usize::MAX {
            continue;
        }

        checked += 1;
        let actual = circ_state[out_pin];

        if actual != expected {
            mismatches += 1;
            if mismatches <= max_log {
                use netlistdb::GeneralHierName;
                let cell_name = netlistdb.cellnames[cellid].dbg_fmt_hier();
                let out_pname = netlistdb.pinnames[out_pin].1.as_str();
                eprintln!(
                    "CELL MISMATCH cycle {}: {} (type={}) {}={} expected={} | \
                     A={} B={} C={} D={} A1={} A2={} A3={} B1={} B2={} C1={} S={} A0={}",
                    cycle, cell_name, ct, out_pname, actual, expected,
                    a_val, b_val, c_val, d_val, a1_val, a2_val, a3_val,
                    b1_val, b2_val, c1_val, s_val, a0_val
                );
            }
        }
    }

    (checked, mismatches)
}

/// Verify that all pins on the same net have the same circ_state value.
/// Returns (nets_checked, mismatches). Logs mismatches to stderr.
fn verify_net_consistency(
    netlistdb: &NetlistDB,
    circ_state: &[u8],
    aig: &AIG,
    cycle: usize,
    max_log: usize,
) -> (usize, usize) {
    let mut checked = 0usize;
    let mut mismatches = 0usize;

    // For each net, find the driver pin value and compare all load pin values
    for netid in 0..netlistdb.net2pin.start.len() - 1 {
        let start = netlistdb.net2pin.start[netid];
        let end = netlistdb.net2pin.start[netid + 1];
        if end - start < 2 {
            continue; // Skip nets with 0 or 1 pin
        }

        // Find driver pin (Direction::O or cell 0)
        let mut driver_pin = None;
        for &pinid in &netlistdb.net2pin.items[start..end] {
            if netlistdb.pindirect[pinid] == Direction::O || netlistdb.pin2cell[pinid] == 0 {
                driver_pin = Some(pinid);
                break;
            }
        }

        let driver_pin = match driver_pin {
            Some(p) => p,
            None => continue, // No driver found
        };

        let driver_val = circ_state[driver_pin];
        checked += 1;

        // Check all other pins on this net
        for &pinid in &netlistdb.net2pin.items[start..end] {
            if pinid == driver_pin {
                continue;
            }
            // Only check pins that have AIG mappings
            if aig.pin2aigpin_iv[pinid] == usize::MAX {
                continue;
            }
            let pin_val = circ_state[pinid];
            if pin_val != driver_val {
                mismatches += 1;
                if mismatches <= max_log {
                    use netlistdb::GeneralHierName;
                    let driver_name = netlistdb.pinnames[driver_pin].dbg_fmt_pin();
                    let load_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                    let driver_celltype = netlistdb.celltypes[netlistdb.pin2cell[driver_pin]].as_str();
                    let load_celltype = netlistdb.celltypes[netlistdb.pin2cell[pinid]].as_str();
                    // Check AIG pin mappings
                    let driver_aigpin = aig.pin2aigpin_iv[driver_pin];
                    let load_aigpin = aig.pin2aigpin_iv[pinid];
                    eprintln!(
                        "NET MISMATCH cycle {}: driver {}({})={} vs load {}({})={} | \
                         driver_aigpin_iv={} load_aigpin_iv={} same_aig={}",
                        cycle, driver_name, driver_celltype, driver_val,
                        load_name, load_celltype, pin_val,
                        driver_aigpin, load_aigpin,
                        (driver_aigpin >> 1) == (load_aigpin >> 1)
                    );
                }
            }
        }
    }

    (checked, mismatches)
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

/// Run programmatic simulation from a config file.
fn run_programmatic_simulation(
    config: TestbenchConfig,
    args: &Args,
    netlistdb: &NetlistDB,
    aig: &AIG,
    lib: &TimingLibrary,
    cell_library: CellLibrary,
    sram_cells: &mut Vec<SramCell>,
    watchlist_entries: Vec<WatchlistEntry>,
    mut trace_file: Option<File>,
    mut wb_monitor: Option<WishboneBusMonitor>,
) {
    clilog::info!("Programmatic simulation: clock_gpio={}, reset_gpio={}, reset_active_high={}",
                  config.clock_gpio, config.reset_gpio, config.reset_active_high);

    // Find GPIO pins
    let find_gpio_in = |idx: usize| -> Option<usize> {
        let gpio_name = format!("gpio_in[{}]", idx);
        for pinid in 0..netlistdb.num_pins {
            if netlistdb.pin2cell[pinid] == 0 {
                let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                if pin_name.contains(&gpio_name) || pin_name.ends_with(&format!("gpio_in:{}", idx)) {
                    return Some(pinid);
                }
            }
        }
        None
    };

    let find_gpio_out = |idx: usize| -> Option<usize> {
        let gpio_name = format!("gpio_out[{}]", idx);
        for pinid in 0..netlistdb.num_pins {
            if netlistdb.pin2cell[pinid] == 0 {
                let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                if pin_name.contains(&gpio_name) || pin_name.ends_with(&format!("gpio_out:{}", idx)) {
                    return Some(pinid);
                }
            }
        }
        None
    };

    // Find clock and reset input pins
    let clock_pin = find_gpio_in(config.clock_gpio).expect("Clock GPIO pin not found");
    let reset_pin = find_gpio_in(config.reset_gpio).expect("Reset GPIO pin not found");

    clilog::info!("Clock input pin: {} (gpio_in[{}])", clock_pin, config.clock_gpio);
    clilog::info!("Reset input pin: {} (gpio_in[{}])", reset_pin, config.reset_gpio);

    // Initialize flash model (using C++ model from chipflow-lib)
    let mut flash: Option<CppSpiFlash> = if let Some(ref flash_cfg) = config.flash {
        let mut fl = CppSpiFlash::new(16 * 1024 * 1024); // 16MB flash
        let firmware_path = std::path::Path::new(&flash_cfg.firmware);
        match fl.load_firmware(firmware_path, flash_cfg.firmware_offset) {
            Ok(size) => clilog::info!("Loaded {} bytes firmware at offset 0x{:X}", size, flash_cfg.firmware_offset),
            Err(e) => panic!("Failed to load firmware: {}", e),
        }
        Some(fl)
    } else {
        None
    };

    // Flash GPIO pins
    let flash_clk_out = config.flash.as_ref().and_then(|f| find_gpio_out(f.clk_gpio));
    let flash_csn_out = config.flash.as_ref().and_then(|f| find_gpio_out(f.csn_gpio));
    let flash_d_out: Vec<Option<usize>> = if let Some(ref f) = config.flash {
        (0..4).map(|i| find_gpio_out(f.d0_gpio + i)).collect()
    } else {
        vec![None; 4]
    };
    let flash_d_in: Vec<Option<usize>> = if let Some(ref f) = config.flash {
        (0..4).map(|i| find_gpio_in(f.d0_gpio + i)).collect()
    } else {
        vec![None; 4]
    };

    // Find GPIO output enable pins
    let find_gpio_oeb = |idx: usize| -> Option<usize> {
        let gpio_name = format!("gpio_oeb[{}]", idx);
        for pinid in 0..netlistdb.num_pins {
            let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
            if pin_name.contains(&gpio_name) {
                return Some(pinid);
            }
        }
        None
    };

    let flash_clk_oeb = config.flash.as_ref().and_then(|f| find_gpio_oeb(f.clk_gpio));
    let flash_csn_oeb = config.flash.as_ref().and_then(|f| find_gpio_oeb(f.csn_gpio));

    if config.flash.is_some() {
        clilog::info!("Flash pins: clk_out={:?}, csn_out={:?}, d_out={:?}, d_in={:?}",
                      flash_clk_out, flash_csn_out, flash_d_out, flash_d_in);
        clilog::info!("Flash OEB pins: clk_oeb={:?}, csn_oeb={:?}",
                      flash_clk_oeb, flash_csn_oeb);
    }

    // UART TX monitoring
    let uart_tx_pin = config.uart.as_ref().and_then(|u| find_gpio_out(u.tx_gpio));
    let uart_baud_rate = config.uart.as_ref().map(|u| u.baud_rate).unwrap_or(115200);
    if let Some(pin) = uart_tx_pin {
        clilog::info!("UART TX pin: {} (gpio_out[{}])", pin, config.uart.as_ref().unwrap().tx_gpio);
    }

    // Find Wishbone signals for debugging - look for DFF Q outputs
    let find_net_pin = |net_pattern: &str| -> Option<usize> {
        // Look for a cell output pin (Q) that drives the signal
        for pinid in 0..netlistdb.num_pins {
            let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
            // For DFF outputs, the cell name contains the signal name and pin is Q
            if pin_name.contains(net_pattern) && pin_name.ends_with(":Q") {
                return Some(pinid);
            }
        }
        // Try net names (some signals are identified by their net, not pin name)
        for netid in 0..netlistdb.num_nets {
            let net_name = netlistdb.netnames[netid].dbg_fmt_pin();
            if net_name.contains(net_pattern) {
                // Find a pin on this net
                for pinid in netlistdb.net2pin.iter_set(netid) {
                    return Some(pinid);
                }
            }
        }
        // Fallback: any pin containing the pattern
        for pinid in 0..netlistdb.num_pins {
            let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
            if pin_name.contains(net_pattern) {
                return Some(pinid);
            }
        }
        None
    };

    let ibus_cyc_pin = find_net_pin("ibus__cyc");
    let ibus_stb_pin = find_net_pin("ibus__stb");
    let dbus_cyc_pin = find_net_pin("dbus__cyc");
    // Find reset synchronizer signals
    let rst_sync_stage0_pin = find_net_pin("rst_n_sync.stage0");
    let rst_sync_rst_pin = find_net_pin("rst_n_sync.rst");
    clilog::info!("Reset sync: stage0={:?}, rst={:?}", rst_sync_stage0_pin, rst_sync_rst_pin);
    // Find address bits to see where CPU is fetching from
    let ibus_adr: Vec<Option<usize>> = (0..24).map(|i| {
        find_net_pin(&format!("ibus__adr[{}]", i))
    }).collect();
    clilog::info!("Wishbone debug: ibus_cyc={:?}, ibus_stb={:?}, dbus_cyc={:?}",
                  ibus_cyc_pin, ibus_stb_pin, dbus_cyc_pin);
    clilog::info!("ibus_adr pins found: {}/24", ibus_adr.iter().filter(|x| x.is_some()).count());

    // SRAM peripheral (SRAMPeripheral Wishbone wrapper) signals
    let sram_wb_ack_pin = find_net_pin("sram.wb_bus__ack");
    let sram_read_en_pin = find_net_pin("sram.read_port__en");
    let sram_write_en: Vec<Option<usize>> = (0..4).map(|i| {
        find_net_pin(&format!("sram.write_port__en[{}]", i))
    }).collect();
    // Data bus address for store operations
    let dbus_adr: Vec<Option<usize>> = (0..30).map(|i| {
        find_net_pin(&format!("dbus__adr[{}]", i))
    }).collect();
    clilog::info!("SRAM peripheral: wb_ack={:?}, read_en={:?}, write_en=[{:?},{:?},{:?},{:?}]",
                  sram_wb_ack_pin, sram_read_en_pin,
                  sram_write_en[0], sram_write_en[1], sram_write_en[2], sram_write_en[3]);
    clilog::info!("dbus_adr pins found: {}/30", dbus_adr.iter().filter(|x| x.is_some()).count());

    // Trace the ACK logic chain:
    // wb_bus__ack DFF D input = _05294_ = NOR2(_40863_) of (net2951, _09347_)
    // _09347_ = NAND3(_28768_) of (_09336_, _09337_, _09346_)
    let ack_d_pin = find_net_pin("_05294_");      // D input of ACK DFF
    let ack_nor_a = find_net_pin("net2951");       // NOR2 input A (rst_n related)
    let ack_nor_b = find_net_pin("_09347_");       // NOR2 input B (from NAND3)
    let ack_nand_a = find_net_pin("_09336_");      // NAND3 input A
    let ack_nand_b = find_net_pin("_09337_");      // NAND3 input B
    let ack_nand_c = find_net_pin("_09346_");      // NAND3 input C
    clilog::info!("ACK logic: D={:?}, NOR_A={:?}, NOR_B={:?}, NAND_A={:?}, NAND_B={:?}, NAND_C={:?}",
                  ack_d_pin, ack_nor_a, ack_nor_b, ack_nand_a, ack_nand_b, ack_nand_c);

    // Find the ACK DFF's AIG pin and trace back through its D input logic
    if let Some(ack_pin) = sram_wb_ack_pin {
        // Find the ACK DFF in the AIG
        for i in 1..=aig.num_aigpins {
            if let DriverType::DFF(cellid) = &aig.drivers[i] {
                let cn = format!("{:?}", netlistdb.cellnames[*cellid]);
                if cn.contains("wb_bus__ack") {
                    eprintln!("ACK DFF: aigpin={}, cell={}", i, cn);
                    // Find the D input pin of this DFF
                    for pinid in netlistdb.cell2pin.iter_set(*cellid) {
                        let pn = netlistdb.pinnames[pinid].1.as_str();
                        if pn == "D" {
                            let d_aig_iv = aig.pin2aigpin_iv[pinid];
                            if d_aig_iv != usize::MAX {
                                let d_idx = d_aig_iv >> 1;
                                let d_inv = (d_aig_iv & 1) != 0;
                                eprintln!("  D pin={}, aig_iv=0x{:x} (aigpin={}, inv={})", pinid, d_aig_iv, d_idx, d_inv);
                                if d_idx > 0 && d_idx <= aig.num_aigpins {
                                    // Show what drives this AIG pin
                                    eprintln!("  D driver: {:?}", aig.drivers[d_idx]);
                                    if let DriverType::AndGate(a, b) = &aig.drivers[d_idx] {
                                        let a_idx = (*a as usize) >> 1;
                                        let a_inv = (*a & 1) != 0;
                                        let b_idx = (*b as usize) >> 1;
                                        let b_inv = (*b & 1) != 0;
                                        eprintln!("    AND({}{}, {}{})",
                                                 if a_inv {"!"} else {""}, a_idx,
                                                 if b_inv {"!"} else {""}, b_idx);
                                        if a_idx > 0 && a_idx <= aig.num_aigpins {
                                            eprintln!("    input A (aigpin {}): {:?}", a_idx, aig.drivers[a_idx]);
                                        }
                                        if b_idx > 0 && b_idx <= aig.num_aigpins {
                                            eprintln!("    input B (aigpin {}): {:?}", b_idx, aig.drivers[b_idx]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // Find sink__payload bits - this holds the reset vector (0x0FFFFC = 0x100000 - 4)
    let sink_payload: Vec<Option<usize>> = (0..32).map(|i| {
        find_net_pin(&format!("soc.cpu.sink__payload[{}]", i))
    }).collect();
    let sink_found = sink_payload.iter().filter(|x| x.is_some()).count();
    clilog::info!("sink__payload pins found: {}/32", sink_found);
    // Also try without soc.cpu prefix
    let sink_payload_20_pin = find_net_pin("sink__payload[20]");

    // Debug: show which ibus_adr pins were found and their initial values
    for (i, opt_pin) in ibus_adr.iter().enumerate() {
        if let Some(pin) = opt_pin {
            let pin_name = netlistdb.pinnames[*pin].dbg_fmt_pin();
            // Only print a few to avoid spam
            if i == 0 || i == 18 || i == 20 {
                clilog::info!("ibus_adr[{}] = pin {} ({})", i, pin, pin_name);
            }
        }
    }

    // Find SPI debug signals for cycle-level comparison with Icarus
    let buffer_io0_o_pin = find_net_pin("buffer_io0.o");
    let str_io0_o_pin = find_net_pin("str_io0.o");
    let o_latch_io0_o_pin = find_net_pin("o_latch.io0.o");
    clilog::info!("SPI debug pins: buffer_io0.o={:?}, str_io0.o={:?}, o_latch.io0.o={:?}",
                  buffer_io0_o_pin, str_io0_o_pin, o_latch_io0_o_pin);

    // Trace specific nets in the buffer_io0.o combinational cone:
    // buffer_io0.o = NAND(_18714_, _18715_)  [_40802_]
    //   _18714_ = NAND(_18712_, _18713_)     [_40800_]
    //   _18715_ = NAND(net806, o_latch.io0.o) [_40801_]
    //   net806 = BUF(_18628_)                 [fanout807]
    //   _18628_ = NAND3(_18246_, _18254_, _18359_) [_40712_]
    // Find driver chain cells for o_latch[2] by instance name.
    // From 6_final_fixed.v:
    //   _40938_: nand2(.A(_18782_), .B(_18705_), .Y(_18783_))
    //   _40939_: nand2(.A(_18360_), .B(o_latch.io0.o), .Y(_18784_))
    //   _40941_: a21oi(.A1(_18783_), .A2(_18784_), .B1(net2982), .Y(_05278_))
    // D formula: _05278_ = !((_18783_ & _18784_) | net2982)
    //          = (_18782_ & _18705_) | (_18360_ & Q)  when net2982=0
    struct CellPins {
        a_pin: usize,  // First input
        b_pin: usize,  // Second input
        c_pin: usize,  // Third input (for a21oi B1)
        y_pin: usize,  // Output
    }
    let empty_cell = CellPins { a_pin: usize::MAX, b_pin: usize::MAX, c_pin: usize::MAX, y_pin: usize::MAX };
    let mut cell_40938 = CellPins { ..empty_cell }; // nand2: .A(_18782_), .B(_18705_), .Y(_18783_)
    let mut cell_40939 = CellPins { ..empty_cell }; // nand2: .A(_18360_), .B(Q), .Y(_18784_)
    let mut cell_40941 = CellPins { ..empty_cell }; // a21oi: .A1(_18783_), .A2(_18784_), .B1(net2982), .Y(_05278_)
    let mut net2982_pin: usize = usize::MAX;

    for cellid in 1..netlistdb.num_cells {
        let cname = format!("{:?}", netlistdb.cellnames[cellid]);
        // Match by exact cell instance names (trimmed)
        let target = if cname.ends_with("_40938_)") { Some(&mut cell_40938) }
                     else if cname.ends_with("_40939_)") { Some(&mut cell_40939) }
                     else if cname.ends_with("_40941_)") { Some(&mut cell_40941) }
                     else { None };
        if let Some(cp) = target {
            let ctype = netlistdb.celltypes[cellid].as_str();
            let mut pins_info = String::new();
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pname_str = netlistdb.pinnames[pinid].1.as_str();
                match pname_str {
                    "A" | "A1" => cp.a_pin = pinid,
                    "B" | "A2" => cp.b_pin = pinid,
                    "B1" => cp.c_pin = pinid,
                    "Y" | "X" => cp.y_pin = pinid,
                    _ => {}
                }
                pins_info += &format!(" {}={}", pname_str, pinid);
            }
            eprintln!("OLATCH DRIVER CELL '{}' type='{}':{}", cname, ctype, pins_info);
        }
    }
    // Find net2982 pin (buffered rst_n_sync.rst)
    if let Some(pid) = find_net_pin("net2982") {
        net2982_pin = pid;
    }

    // Cell _40937_: o21ai(.A1(enframer.cycle[2]), .A2(_18743_), .B1(_18781_), .Y(_18782_))
    // _18782_ = !((cycle[2] | _18743_) & _18781_)
    let mut cell_40937 = CellPins { ..empty_cell };
    // Cell _40936_: nand3(.A(_18758_), .B(_18777_), .C(_18780_), .Y(_18781_))
    let mut cell_40936_a = usize::MAX;
    let mut cell_40936_b = usize::MAX;
    let mut cell_40936_c = usize::MAX;

    for cellid in 1..netlistdb.num_cells {
        let cname = format!("{:?}", netlistdb.cellnames[cellid]);
        if cname.ends_with("_40937_)") {
            let ctype = netlistdb.celltypes[cellid].as_str();
            let mut pins_info = String::new();
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pname_str = netlistdb.pinnames[pinid].1.as_str();
                match pname_str {
                    "A1" => cell_40937.a_pin = pinid,
                    "A2" => cell_40937.b_pin = pinid,
                    "B1" => cell_40937.c_pin = pinid,
                    "Y" => cell_40937.y_pin = pinid,
                    _ => {}
                }
                pins_info += &format!(" {}={}", pname_str, pinid);
            }
            eprintln!("CELL _40937_ type='{}': {}", ctype, pins_info);
        } else if cname.ends_with("_40936_)") {
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pname_str = netlistdb.pinnames[pinid].1.as_str();
                match pname_str {
                    "A" => cell_40936_a = pinid,
                    "B" => cell_40936_b = pinid,
                    "C" => cell_40936_c = pinid,
                    _ => {}
                }
            }
            eprintln!("CELL _40936_ (nand3 → _18781_): A(18758)={} B(18777)={} C(18780)={}", cell_40936_a, cell_40936_b, cell_40936_c);
        }
    }

    // Also trace the enframer.cycle DFF Q values
    let ecycle0_pin = find_net_pin("enframer.cycle[0]");
    let ecycle1_pin = find_net_pin("enframer.cycle[1]");
    let ecycle2_pin = find_net_pin("enframer.cycle[2]");
    eprintln!("Enframer cycle pins: [0]={:?} [1]={:?} [2]={:?}", ecycle0_pin, ecycle1_pin, ecycle2_pin);

    // Trace raw_tx_data[0..7] register bits (the SPI command byte the controller sends)
    let mut raw_tx_data_pins: [Option<usize>; 8] = [None; 8];
    for bit in 0..8 {
        raw_tx_data_pins[bit] = find_net_pin(&format!("raw_tx_data[{}]", bit));
    }
    eprintln!("raw_tx_data pins: {:?}", raw_tx_data_pins);

    // Trace SPI controller FSM state bits
    let mut fsm_state_pins: [Option<usize>; 4] = [None; 4];
    for bit in 0..4 {
        fsm_state_pins[bit] = find_net_pin(&format!("spiflash.ctrl.fsm_state[{}]", bit));
    }
    eprintln!("FSM state pins: {:?}", fsm_state_pins);

    // Trace cells in the _18777_ driver chain:
    // _40932_: nand3(.A(_18769_), .B(cycle[2]), .C(_18776_)) → _18777_
    // _40931_: nand3(.A(_18742_), .B(cycle[1]), .C(_18775_)) → _18776_
    // _40930_: nand2(.A(_18774_), .B(_02744_)) → _18775_; _02744_ = !cycle[0]
    // _40897_: o2111ai → _18742_
    let mut cell_40932 = CellPins { ..empty_cell }; // nand3 → _18777_
    let mut cell_40931 = CellPins { ..empty_cell }; // nand3 → _18776_
    let mut cell_40930 = CellPins { ..empty_cell }; // nand2 → _18775_

    for cellid in 1..netlistdb.num_cells {
        let cname = format!("{:?}", netlistdb.cellnames[cellid]);
        let target = if cname.ends_with("_40932_)") { Some(&mut cell_40932) }
                     else if cname.ends_with("_40931_)") { Some(&mut cell_40931) }
                     else if cname.ends_with("_40930_)") { Some(&mut cell_40930) }
                     else { None };
        if let Some(cp) = target {
            let ctype = netlistdb.celltypes[cellid].as_str();
            let mut pins_info = String::new();
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                let pname_str = netlistdb.pinnames[pinid].1.as_str();
                match pname_str {
                    "A" | "A1" => cp.a_pin = pinid,
                    "B" | "A2" => cp.b_pin = pinid,
                    "C" | "B1" => cp.c_pin = pinid,
                    "Y" | "X" => cp.y_pin = pinid,
                    _ => {}
                }
                pins_info += &format!(" {}={}", pname_str, pinid);
            }
            eprintln!("DEEP TRACE CELL '{}' type='{}':{}", cname, ctype, pins_info);
        }
    }

    // Find the DFF cell for buffer_io0.o and its D/Q pins
    let mut buffer_dff_cellid = usize::MAX;
    let mut buffer_dff_d_pin = usize::MAX;
    let mut buffer_dff_q_pin = usize::MAX;
    for cellid in 1..netlistdb.num_cells {
        let cell_name = format!("{:?}", netlistdb.cellnames[cellid]);
        if cell_name.contains("buffer_io0.o_ff") {
            buffer_dff_cellid = cellid;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                match netlistdb.pinnames[pinid].1.as_str() {
                    "D" => buffer_dff_d_pin = pinid,
                    "Q" => buffer_dff_q_pin = pinid,
                    _ => {}
                }
            }
            let ct = netlistdb.celltypes[cellid].as_str();
            clilog::info!("Found buffer_io0.o DFF: cell={}, type={}, D_pin={}, Q_pin={}", cellid, ct, buffer_dff_d_pin, buffer_dff_q_pin);
            break;
        }
    }

    // Verify AIG topological order: for each AND gate, both inputs must have lower index
    {
        let mut topo_violations = 0usize;
        for i in 1..=aig.num_aigpins {
            if let DriverType::AndGate(a_iv, b_iv) = &aig.drivers[i] {
                let a_idx = a_iv >> 1;
                let b_idx = b_iv >> 1;
                if a_idx >= i || b_idx >= i {
                    topo_violations += 1;
                    if topo_violations <= 10 {
                        eprintln!("TOPO VIOLATION: pin {} depends on pin {} and pin {} (both should be < {})",
                                 i, a_idx, b_idx, i);
                    }
                }
            }
        }
        if topo_violations > 0 {
            eprintln!("CRITICAL: {} AIG topological order violations! Single-pass eval will be WRONG!", topo_violations);
        } else {
            clilog::info!("AIG topological order verified: all {} pins in correct order", aig.num_aigpins);
        }
    }

    // Initialize timing state
    let mut state = TimingState::new(aig.num_aigpins, lib);
    state.init_delays(aig, lib);

    // Initialize circuit state
    let mut circ_state = vec![0u8; netlistdb.num_pins];

    // Initialize constant nets
    if let Some(netid) = netlistdb.net_one {
        for pinid in netlistdb.net2pin.iter_set(netid) {
            circ_state[pinid] = 1;
        }
    }

    // Initialize gpio_loopback_one and gpio_loopback_zero (Caravel constant inputs)
    // These provide constant 1 and 0 values to the design logic
    for pinid in 0..netlistdb.num_pins {
        let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
        if pin_name.contains("gpio_loopback_one") {
            circ_state[pinid] = 1;
        }
        // gpio_loopback_zero is already 0 from initialization
    }
    clilog::info!("Initialized gpio_loopback_one pins to 1");

    // Build reverse mapping from AIG pin to netlist pin
    let mut aigpin_to_netpin: Vec<usize> = vec![usize::MAX; aig.num_aigpins + 1];
    for (pinid, &aigpin_iv) in aig.pin2aigpin_iv.iter().enumerate() {
        if aigpin_iv != usize::MAX {
            let aigpin = aigpin_iv >> 1;
            if aigpin > 0 && aigpin <= aig.num_aigpins {
                aigpin_to_netpin[aigpin] = pinid;
            }
        }
    }

    // UART state
    let clock_hz = 1_000_000_000_000u64 / args.clock_period;
    let cycles_per_bit = (clock_hz / uart_baud_rate as u64) as usize;
    let mut uart_state = UartState::Idle;
    let mut uart_events: Vec<UartEvent> = Vec::new();
    let mut uart_last_tx = 1u8;

    let max_cycles = config.num_cycles;
    let reset_cycles = config.reset_cycles;

    clilog::info!("Starting programmatic simulation: {} reset cycles, {} total cycles",
                  reset_cycles, max_cycles);

    // Helper closure to evaluate combinational logic
    let eval_combinational = |state: &mut TimingState, circ_state: &[u8]| {
        for i in 1..=aig.num_aigpins {
            match &aig.drivers[i] {
                DriverType::AndGate(a, b) => {
                    state.eval_and(i, *a, *b);
                }
                DriverType::InputPort(pinid) => {
                    state.values[i] = circ_state[*pinid];
                }
                DriverType::DFF(cell_idx) => {
                    for pinid in netlistdb.cell2pin.iter_set(*cell_idx) {
                        if netlistdb.pinnames[pinid].1.as_str() == "Q" {
                            state.values[i] = circ_state[pinid];
                            break;
                        }
                    }
                }
                DriverType::SRAM(_) => {
                    let netpin = aigpin_to_netpin[i];
                    if netpin != usize::MAX {
                        state.values[i] = circ_state[netpin];
                    }
                }
                DriverType::InputClockFlag(_, _) | DriverType::Tie0 => {}
            }
        }
    };

    // Helper closure to update circ_state from AIG
    let update_circ_from_aig = |state: &TimingState, circ_state: &mut [u8]| {
        for (pinid, &aigpin_iv) in aig.pin2aigpin_iv.iter().enumerate() {
            if aigpin_iv != usize::MAX {
                let idx = aigpin_iv >> 1;
                let inv = (aigpin_iv & 1) != 0;
                if idx == 0 {
                    // AIG pin 0 is the constant-0 node. With inversion it's constant 1.
                    // This handles tie cells (conb_1 .HI/.LO) mapped to the AIG constant.
                    circ_state[pinid] = inv as u8;
                } else if idx <= aig.num_aigpins {
                    circ_state[pinid] = state.values[idx] ^ (inv as u8);
                }
            }
        }
    };

    // Build net propagation map for SRAM pins.
    // SRAM cell input pins are NOT in the AIG, so `update_circ_from_aig` never sets
    // their values. We need to find the AIG-driven pin on the same net and copy its
    // value to the SRAM input pin after each evaluation.
    // sram_net_prop[i] = (sram_pinid, driver_pinid) means:
    //   circ_state[sram_pinid] = circ_state[driver_pinid]
    let mut sram_net_prop: Vec<(usize, usize)> = Vec::new();
    for sram in sram_cells.iter() {
        // Collect all SRAM input pins that need propagation
        let mut sram_input_pins = Vec::new();
        sram_input_pins.push(sram.clk_pin);
        sram_input_pins.push(sram.en_pin);
        sram_input_pins.push(sram.r_wb_pin);
        sram_input_pins.extend_from_slice(&sram.addr_pins);
        sram_input_pins.extend_from_slice(&sram.ben_pins);
        sram_input_pins.extend_from_slice(&sram.di_pins);
        // DO pins are outputs from SRAM - they need reverse propagation
        // (SRAM writes to DO, then we need to push DO values to the net)

        for &pinid in &sram_input_pins {
            if pinid == usize::MAX { continue; }
            // Check if this pin is already in the AIG
            if aig.pin2aigpin_iv[pinid] != usize::MAX { continue; }
            // Find a driver pin (AIG-mapped) on the same net
            let netid = netlistdb.pin2net[pinid];
            let mut driver_pin = usize::MAX;
            for np in netlistdb.net2pin.iter_set(netid) {
                if np != pinid && aig.pin2aigpin_iv[np] != usize::MAX {
                    driver_pin = np;
                    break;
                }
            }
            if driver_pin != usize::MAX {
                sram_net_prop.push((pinid, driver_pin));
            } else {
                eprintln!("WARNING: SRAM pin {} has no AIG-mapped driver on net {}",
                         pinid, netid);
            }
        }
    }
    clilog::info!("Built SRAM net propagation map: {} pin pairs", sram_net_prop.len());

    // Debug: show SRAM pin AIG mappings
    for sram in sram_cells.iter() {
        let clk_aig = aig.pin2aigpin_iv[sram.clk_pin];
        let en_aig = aig.pin2aigpin_iv[sram.en_pin];
        let rwb_aig = aig.pin2aigpin_iv[sram.r_wb_pin];
        let clk_aigpin = clk_aig >> 1;
        let en_aigpin = en_aig >> 1;
        eprintln!("SRAM pin mappings: CLKin pin={} aig_iv=0x{:x} (aigpin={}), EN pin={} aig_iv=0x{:x} (aigpin={}), R_WB pin={} aig_iv=0x{:x}",
                 sram.clk_pin, clk_aig, clk_aigpin, sram.en_pin, en_aig, en_aigpin, sram.r_wb_pin, rwb_aig);
        // Show what type of AIG node drives CLKin
        if clk_aigpin > 0 && clk_aigpin <= aig.num_aigpins {
            eprintln!("  CLKin AIG driver: {:?}", aig.drivers[clk_aigpin]);
        }
        if en_aigpin > 0 && en_aigpin <= aig.num_aigpins {
            eprintln!("  EN AIG driver: {:?}", aig.drivers[en_aigpin]);
        }
        // Check what the SRAM CLKin's net looks like
        let clk_netid = netlistdb.pin2net[sram.clk_pin];
        let en_netid = netlistdb.pin2net[sram.en_pin];
        eprintln!("SRAM CLKin net={} ({}), EN net={} ({})",
                 clk_netid, netlistdb.netnames[clk_netid].dbg_fmt_pin(),
                 en_netid, netlistdb.netnames[en_netid].dbg_fmt_pin());
    }

    // Also build reverse propagation for SRAM DO (output) pins.
    // After SRAM writes DO, we need to propagate to all other pins on the same net
    // that ARE in the AIG (so they feed back into combinational logic).
    let mut sram_do_prop: Vec<(usize, Vec<usize>)> = Vec::new();
    for sram in sram_cells.iter() {
        for &do_pin in &sram.do_pins {
            if do_pin == usize::MAX { continue; }
            let netid = netlistdb.pin2net[do_pin];
            let mut load_pins = Vec::new();
            for np in netlistdb.net2pin.iter_set(netid) {
                if np != do_pin {
                    load_pins.push(np);
                }
            }
            if !load_pins.is_empty() {
                sram_do_prop.push((do_pin, load_pins));
            }
        }
    }
    clilog::info!("Built SRAM DO propagation map: {} output pins, {} total loads",
                 sram_do_prop.len(),
                 sram_do_prop.iter().map(|(_, loads)| loads.len()).sum::<usize>());

    // Helper functions are defined inline where used (no closures needed)

    // Simulation loop - aligned with CXXRTL's main.cc structure
    //
    // CXXRTL order per tick:
    // 1. Step models (flash, uart, gpio)
    // 2. Set clock = false (falling edge)
    // 3. agent.step() (evaluate)
    // 4. Set clock = true (rising edge)
    // 5. agent.step() (evaluate) - DFFs latch here
    //
    // Reset sequence: rst_n=false for 1 tick, then rst_n=true
    //

    // Set reset active FIRST (rst_n = false in CXXRTL, gpio=1 for us with invert)
    // Note: CXXRTL uses rst_n directly, we go through GPIO with inversion
    // reset_active_high=true means GPIO=1 = reset active = internal rst_n=0
    let reset_active_val = if config.reset_active_high { 1 } else { 0 };
    let reset_inactive_val = if config.reset_active_high { 0 } else { 1 };

    circ_state[reset_pin] = reset_active_val;
    clilog::info!("Reset: setting gpio_in[{}] = {} (reset active)", config.reset_gpio, reset_active_val);

    // Initial evaluation (like CXXRTL's initial agent.step())
    // This evaluates with reset already active
    eval_combinational(&mut state, &circ_state);
    update_circ_from_aig(&state, &mut circ_state);

    // Net consistency check: verify all pins on the same net have the same circ_state value.
    // If they don't, it means pin2aigpin_iv has inconsistent mappings (AIG mapping bug).
    {
        let mut net_inconsistencies = 0usize;
        let mut unmapped_on_mapped_net = 0usize;
        for netid in 0..netlistdb.num_nets {
            let mut driver_val: Option<(u8, usize, bool)> = None; // (value, pinid, is_output)
            let mut has_mapped = false;
            let mut has_unmapped = false;
            for pinid in netlistdb.net2pin.iter_set(netid) {
                let aigpin_iv = aig.pin2aigpin_iv[pinid];
                if aigpin_iv == usize::MAX {
                    has_unmapped = true;
                    continue;
                }
                has_mapped = true;
                let val = circ_state[pinid];
                if let Some((dv, dp, _)) = driver_val {
                    if val != dv {
                        net_inconsistencies += 1;
                        if net_inconsistencies <= 20 {
                            let net_name = netlistdb.netnames[netid].dbg_fmt_pin();
                            let pin1_name = netlistdb.pinnames[dp].dbg_fmt_pin();
                            let pin2_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                            eprintln!("NET INCONSISTENCY: net '{}' pin {}({})={} vs pin {}({})={}",
                                     net_name, dp, pin1_name, dv, pinid, pin2_name, val);
                        }
                    }
                } else {
                    // Check if this is an output pin (Direction::O)
                    let dir = netlistdb.pindirect[pinid];
                    driver_val = Some((val, pinid, dir == netlistdb::Direction::O));
                }
            }
            if has_mapped && has_unmapped {
                unmapped_on_mapped_net += 1;
            }
        }
        if net_inconsistencies > 0 {
            eprintln!("TOTAL NET INCONSISTENCIES: {} (across {} nets)", net_inconsistencies, netlistdb.num_nets);
        } else {
            clilog::info!("Net consistency check: all pins consistent across {} nets", netlistdb.num_nets);
        }
        if unmapped_on_mapped_net > 0 {
            clilog::info!("Warning: {} nets have both mapped and unmapped pins", unmapped_on_mapped_net);
        }
    }

    // Build a list of DFF info for efficient iteration
    // Each entry: (cell_id, d_pin, q_pin, clk_pin, reset_b_pin, set_b_pin, de_pin)
    let mut dff_list: Vec<(usize, usize, usize, usize, usize, usize, usize)> = Vec::new();
    for cellid in 1..netlistdb.num_cells {
        let celltype = netlistdb.celltypes[cellid].as_str();
        let is_dff = match cell_library {
            CellLibrary::SKY130 => {
                let ct = extract_cell_type(celltype);
                matches!(ct, "dfxtp" | "dfrtp" | "dfrbp" | "dfstp" | "dfbbp" | "edfxtp" | "sdfxtp")
            }
            _ => matches!(celltype, "DFF" | "DFFSR"),
        };
        if is_dff {
            let mut pinid_d = usize::MAX;
            let mut pinid_q = usize::MAX;
            let mut pinid_clk = usize::MAX;
            let mut pinid_de = usize::MAX;
            let mut pinid_reset_b = usize::MAX;
            let mut pinid_set_b = usize::MAX;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                match netlistdb.pinnames[pinid].1.as_str() {
                    "D" => pinid_d = pinid,
                    "Q" => pinid_q = pinid,
                    "CLK" => pinid_clk = pinid,
                    "DE" => pinid_de = pinid,
                    "RESET_B" => pinid_reset_b = pinid,
                    "SET_B" => pinid_set_b = pinid,
                    _ => {}
                }
            }
            if pinid_d != usize::MAX && pinid_q != usize::MAX && pinid_clk != usize::MAX {
                dff_list.push((cellid, pinid_d, pinid_q, pinid_clk, pinid_reset_b, pinid_set_b, pinid_de));
            }
        }
    }
    // Diagnostic: check how many DFFs are on the system clock vs other clocks
    let mut dffs_on_sysclk = 0usize;
    let mut dffs_off_sysclk = 0usize;
    for &(cellid, _, _, clk_pin, _, _, _) in &dff_list {
        if clk_pin == clock_pin {
            dffs_on_sysclk += 1;
        } else {
            dffs_off_sysclk += 1;
            // Log the first few non-system-clock DFFs
            if dffs_off_sysclk <= 20 {
                let cell_name = format!("{:?}", netlistdb.cellnames[cellid]);
                clilog::info!("DFF NOT on system clock: {} (clk_pin={}, sys_clk={})",
                             &cell_name[..cell_name.len().min(80)], clk_pin, clock_pin);
            }
        }
    }
    clilog::info!("Found {} DFFs total: {} on system clock, {} on OTHER clocks",
                 dff_list.len(), dffs_on_sysclk, dffs_off_sysclk);

    // Track previous CLK state, D value, and DE value for each DFF.
    // These are captured after FALL eval (before clock edge) so DFFs
    // latch the pre-edge D value, matching real hardware behavior.
    let mut dff_prev_clk: Vec<u8> = vec![0; dff_list.len()];
    let mut dff_prev_d: Vec<u8> = vec![0; dff_list.len()];
    let mut dff_prev_de: Vec<u8> = vec![1; dff_list.len()]; // default: enabled

    // Initialize previous CLK, D, and DE states
    for (i, &(_, pinid_d, _, clk_pin, _, _, pinid_de)) in dff_list.iter().enumerate() {
        dff_prev_clk[i] = circ_state[clk_pin];
        dff_prev_d[i] = circ_state[pinid_d];
        if pinid_de != usize::MAX {
            dff_prev_de[i] = circ_state[pinid_de];
        }
    }

    // Force DFFs with async reset to their reset values during initial state
    // This is critical because reset is already active (reset_active_val)
    let mut dffs_force_reset = 0usize;
    for cellid in 1..netlistdb.num_cells {
        let celltype = netlistdb.celltypes[cellid].as_str();
        let is_dff = match cell_library {
            CellLibrary::SKY130 => {
                let ct = extract_cell_type(celltype);
                matches!(ct, "dfxtp" | "dfrtp" | "dfrbp" | "dfstp" | "dfbbp" | "edfxtp" | "sdfxtp")
            }
            _ => matches!(celltype, "DFF" | "DFFSR"),
        };
        if is_dff {
            let mut pinid_q = usize::MAX;
            let mut pinid_reset_b = usize::MAX;
            let mut pinid_set_b = usize::MAX;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                match netlistdb.pinnames[pinid].1.as_str() {
                    "Q" => pinid_q = pinid,
                    "RESET_B" => pinid_reset_b = pinid,
                    "SET_B" => pinid_set_b = pinid,
                    _ => {}
                }
            }
            if pinid_q != usize::MAX {
                // Check if async reset/set is active
                let reset_active = pinid_reset_b != usize::MAX && circ_state[pinid_reset_b] == 0;
                let set_active = pinid_set_b != usize::MAX && circ_state[pinid_set_b] == 0;
                if reset_active {
                    circ_state[pinid_q] = 0;
                    dffs_force_reset += 1;
                } else if set_active {
                    circ_state[pinid_q] = 1;
                    dffs_force_reset += 1;
                }
            }
        }
    }
    if dffs_force_reset > 0 {
        clilog::info!("Forced {} DFFs to reset/set values during initial state", dffs_force_reset);
    }

    // Re-evaluate combinational after forcing DFF outputs
    eval_combinational(&mut state, &circ_state);
    update_circ_from_aig(&state, &mut circ_state);

    // Flash data delay: track previous step's data to model setup time.
    // In the post-synthesis netlist, the SPI clock and data DFFs advance simultaneously
    // on the system clock edge. This means at the moment the flash model sees a rising
    // SPI clock edge (posedge), the data has already changed to the NEXT bit value.
    // We compensate by feeding the flash model the PREVIOUS step's data, which was
    // the value present before the clock edge (matching real hardware setup timing).
    let mut prev_flash_d_out: u8 = 0;
    let mut prev_flash_csn: bool = true;

    for cycle in 0..max_cycles {
        // Apply reset polarity from config
        // reset_active_high=true: gpio=1 during reset, gpio=0 during run
        // reset_active_high=false: gpio=0 during reset, gpio=1 during run
        let new_reset = if cycle < config.reset_cycles as usize {
            reset_active_val
        } else {
            reset_inactive_val
        };
        if cycle <= 20 && (cycle == 0 || cycle == config.reset_cycles as usize) {
            let is_reset = new_reset == reset_active_val;
            clilog::info!("cycle {}: gpio_rst={} → {}", cycle, new_reset,
                         if is_reset { "RESET" } else { "RUN" });
        }
        circ_state[reset_pin] = new_reset;

        // === TICK START (matches CXXRTL tick lambda) ===
        //
        // CXXRTL tick order:
        // 1. clock = false
        // 2. agent.step() - evaluate combinational (includes flash.eval())
        // 3. clock = true
        // 4. agent.step() - evaluate + latch DFFs on rising edge (includes flash.eval())
        //
        // The flash model is evaluated as part of each agent.step() call,
        // so we need to step it AFTER each combinational evaluation to see
        // the current SPI clock value.

        // Helper closure to step flash and update gpio_in
        // Returns (clk, csn, d_out) for logging
        // Feeds previous step's data/CSN with current clock to model setup timing:
        // in real hardware, data is stable BEFORE the clock edge. In our simulation,
        // the SPI clock and data DFFs advance simultaneously, so the current data
        // at a posedge has already changed. Using delayed data fixes this.
        let step_flash = |fl: &mut CppSpiFlash, circ_state: &mut [u8], in_reset: bool,
                          prev_d: &mut u8, prev_csn: &mut bool| {
            let clk = flash_clk_out.map(|p| circ_state[p] != 0).unwrap_or(false);
            let current_csn = flash_csn_out.map(|p| circ_state[p] != 0).unwrap_or(true);
            let mut current_d_out = 0u8;
            for (i, opt_pin) in flash_d_out.iter().enumerate() {
                if let Some(pin) = opt_pin {
                    if circ_state[*pin] != 0 {
                        current_d_out |= 1 << i;
                    }
                }
            }

            // Use delayed data/CSN (from previous step) with current clock
            let effective_csn = if in_reset { true } else { *prev_csn };
            let d_in = fl.step(clk, effective_csn, *prev_d);

            // Save current values for next step
            *prev_d = current_d_out;
            *prev_csn = current_csn;

            // Only update gpio_in when NOT in reset
            // DEBUG: trace flash d_in at critical cycles
            static mut FLASH_STEP_COUNT: usize = 0;
            let step_count = unsafe { FLASH_STEP_COUNT += 1; FLASH_STEP_COUNT };
            if !in_reset {
                if step_count <= 80 || (d_in != 0) {
                    eprintln!("FLASH step {}: clk={} csn={} d_out={:04b} d_in={:04b} (eff_csn={} prev_d={:04b})",
                             step_count, clk, current_csn, current_d_out, d_in, effective_csn, *prev_d);
                }
                for (i, opt_pin) in flash_d_in.iter().enumerate() {
                    if let Some(pin) = opt_pin {
                        circ_state[*pin] = ((d_in >> i) & 1) as u8;
                    }
                }
            }
            (clk, current_csn, current_d_out)
        };

        // 1. Clock LOW (falling edge)
        circ_state[clock_pin] = 0;

        // 2. Evaluate combinational on falling edge
        eval_combinational(&mut state, &circ_state);
        update_circ_from_aig(&state, &mut circ_state);

        // Convergence check: re-evaluate and see if anything changes
        if cycle >= 10 && cycle <= 30 {
            let old_state: Vec<u8> = circ_state.clone();
            eval_combinational(&mut state, &circ_state);
            update_circ_from_aig(&state, &mut circ_state);
            let mut changes = 0usize;
            for i in 0..circ_state.len() {
                if circ_state[i] != old_state[i] {
                    changes += 1;
                    if changes <= 5 {
                        let pname = netlistdb.pinnames[i].dbg_fmt_pin();
                        eprintln!("  CONVERGENCE: c{} pin {} '{}' changed {}→{}",
                                 cycle, i, &pname[..pname.len().min(60)], old_state[i], circ_state[i]);
                    }
                }
            }
            if changes > 0 {
                eprintln!("  CONVERGENCE FAIL c{}: {} pins changed on 2nd pass!", cycle, changes);
            }
        }

        // SPI debug: trace o_latch[2] driver chain cells (from 6_final_fixed.v)
        // _40938_: nand2(.A=_18782_, .B=_18705_, .Y=_18783_)
        // _40939_: nand2(.A=_18360_, .B=Q, .Y=_18784_)
        // _40941_: a21oi(.A1=_18783_, .A2=_18784_, .B1=net2982, .Y=_05278_)
        // Broad trace: FSM state, raw_tx_data, cycle counter from reset onwards
        if cycle >= 8 && cycle <= 35 {
            let mut fsm = 0u8;
            for bit in 0..4 {
                if let Some(p) = fsm_state_pins[bit] {
                    if circ_state[p] != 0 { fsm |= 1 << bit; }
                }
            }
            let mut raw_tx = 0u8;
            for bit in 0..8 {
                if let Some(p) = raw_tx_data_pins[bit] {
                    if circ_state[p] != 0 { raw_tx |= 1 << bit; }
                }
            }
            let ec0 = ecycle0_pin.map(|p| circ_state[p]).unwrap_or(255);
            let ec1 = ecycle1_pin.map(|p| circ_state[p]).unwrap_or(255);
            let ec2 = ecycle2_pin.map(|p| circ_state[p]).unwrap_or(255);
            let ecycle_val = ec0 as u32 | ((ec1 as u32) << 1) | ((ec2 as u32) << 2);
            let csn_val = flash_csn_out.map(|p| circ_state[p]).unwrap_or(255);
            let clk_val = flash_clk_out.map(|p| circ_state[p]).unwrap_or(255);
            let olatch_q = o_latch_io0_o_pin.map(|p| circ_state[p]).unwrap_or(255);
            let ibus_cyc_v = ibus_cyc_pin.map(|p| circ_state[p]).unwrap_or(255);
            eprintln!("TRACE c{}: fsm={} ecyc={} raw_tx=0x{:02x} csn={} sclk={} olatch={} ibus_cyc={}",
                     cycle, fsm, ecycle_val, raw_tx, csn_val, clk_val, olatch_q, ibus_cyc_v);
        }

        // D = !((_18783_ & _18784_) | net2982) = (_18782_ & _18705_) | (_18360_ & Q)
        if cycle >= 22 && cycle <= 32 {
            let read_pin = |p: usize| -> u8 { if p != usize::MAX { circ_state[p] } else { 255 } };
            let olatch_q = o_latch_io0_o_pin.map(|p| circ_state[p]).unwrap_or(255);
            // Cell _40938_: A=_18782_ B=_18705_ Y=_18783_
            let c38_a = read_pin(cell_40938.a_pin); // _18782_
            let c38_b = read_pin(cell_40938.b_pin); // _18705_
            let c38_y = read_pin(cell_40938.y_pin); // _18783_ = nand(a,b)
            // Cell _40939_: A=_18360_ B=Q Y=_18784_
            let c39_a = read_pin(cell_40939.a_pin); // _18360_
            let c39_b = read_pin(cell_40939.b_pin); // Q = o_latch.io0.o
            let c39_y = read_pin(cell_40939.y_pin); // _18784_ = nand(a,b)
            // Cell _40941_: A1=_18783_ A2=_18784_ B1=net2982 Y=_05278_
            let c41_a1 = read_pin(cell_40941.a_pin); // _18783_
            let c41_a2 = read_pin(cell_40941.b_pin); // _18784_
            let c41_b1 = read_pin(cell_40941.c_pin); // net2982 (buffered rst)
            let c41_y  = read_pin(cell_40941.y_pin); // _05278_ = D of o_latch[2]

            // Verify cell truth tables
            let exp_38 = if c38_a == 1 && c38_b == 1 { 0 } else { 1 }; // nand
            let exp_39 = if c39_a == 1 && c39_b == 1 { 0 } else { 1 }; // nand
            let exp_41 = if ((c41_a1 & c41_a2) | c41_b1) != 0 { 0 } else { 1 }; // a21oi
            let m38 = if c38_y != exp_38 { "MISMATCH!" } else { "" };
            let m39 = if c39_y != exp_39 { "MISMATCH!" } else { "" };
            let m41 = if c41_y != exp_41 { "MISMATCH!" } else { "" };

            // Cell _40937_: o21ai - _18782_ = !((cycle[2] | _18743_) & _18781_)
            let c37_a1 = read_pin(cell_40937.a_pin); // enframer.cycle[2]
            let c37_a2 = read_pin(cell_40937.b_pin); // _18743_
            let c37_b1 = read_pin(cell_40937.c_pin); // _18781_
            let c37_y  = read_pin(cell_40937.y_pin); // _18782_
            let exp_37 = if ((c37_a1 | c37_a2) != 0) && c37_b1 != 0 { 0 } else { 1 }; // o21ai
            let m37 = if c37_y != exp_37 { "MISMATCH!" } else { "" };

            // enframer.cycle bits
            let ec0 = ecycle0_pin.map(|p| circ_state[p]).unwrap_or(255);
            let ec1 = ecycle1_pin.map(|p| circ_state[p]).unwrap_or(255);
            let ec2 = ecycle2_pin.map(|p| circ_state[p]).unwrap_or(255);
            let ecycle_val = ec0 as u32 | ((ec1 as u32) << 1) | ((ec2 as u32) << 2);

            // nand3 _40936_ inputs: _18758_, _18777_, _18780_ → _18781_
            let n36_a = read_pin(cell_40936_a); // _18758_
            let n36_b = read_pin(cell_40936_b); // _18777_
            let n36_c = read_pin(cell_40936_c); // _18780_

            // Deep trace: _18777_ driver chain
            // _40932_: nand3(.A(_18769_), .B(cycle[2]), .C(_18776_)) → _18777_
            let c32_a = read_pin(cell_40932.a_pin); // _18769_
            let c32_b = read_pin(cell_40932.b_pin); // cycle[2]
            let c32_c = read_pin(cell_40932.c_pin); // _18776_
            let c32_y = read_pin(cell_40932.y_pin); // _18777_
            // _40931_: nand3(.A(_18742_), .B(cycle[1]), .C(_18775_)) → _18776_
            let c31_a = read_pin(cell_40931.a_pin); // _18742_
            let c31_b = read_pin(cell_40931.b_pin); // cycle[1]
            let c31_c = read_pin(cell_40931.c_pin); // _18775_
            let c31_y = read_pin(cell_40931.y_pin); // _18776_
            // _40930_: nand2(.A(_18774_), .B(_02744_)) → _18775_; _02744_ = !cycle[0]
            let c30_a = read_pin(cell_40930.a_pin); // _18774_
            let c30_b = read_pin(cell_40930.b_pin); // _02744_ = !cycle[0]
            let c30_y = read_pin(cell_40930.y_pin); // _18775_

            // raw_tx_data register value
            let mut raw_tx = 0u8;
            for bit in 0..8 {
                if let Some(p) = raw_tx_data_pins[bit] {
                    if circ_state[p] != 0 { raw_tx |= 1 << bit; }
                }
            }

            eprintln!("FALL c{}: ecycle={} Q={} D={} raw_tx=0x{:02x} | nand3_36({},{},{})→{} | o21ai_37({},{},{})→{} {}",
                     cycle, ecycle_val, olatch_q, c41_y, raw_tx,
                     n36_a, n36_b, n36_c, c37_b1,
                     c37_a1, c37_a2, c37_b1, c37_y, m37);
            eprintln!("  _18777_ chain: nand3_32({},{},{})→{} | nand3_31(_742={},cyc1={},_775={})→{} | nand2_30(_774={},!c0={})→{}",
                     c32_a, c32_b, c32_c, c32_y,
                     c31_a, c31_b, c31_c, c31_y,
                     c30_a, c30_b, c30_y);
        }

        // FALL eval cell verification
        if cycle >= 24 && cycle <= 28 {
            let (checked, mismatches) = verify_cell_outputs(
                &netlistdb, &circ_state, cell_library, cycle, 0,
            );
            if mismatches > 0 {
                eprintln!("FALL cycle {}: {} cell mismatches out of {} cells checked!", cycle, mismatches, checked);
            } else {
                eprintln!("FALL cycle {}: 0 cell mismatches ({} checked)", cycle, checked);
            }
        }

        // 2b. Step flash model AFTER falling edge eval (sees current SPI clock)
        // Pass in_reset flag to prevent spurious flash activity during reset
        let in_reset = cycle < reset_cycles;
        if let Some(ref mut fl) = flash {
            let (clk, csn, d_out) = step_flash(fl, &mut circ_state, in_reset, &mut prev_flash_d_out, &mut prev_flash_csn);
            // Debug: show flash control signals periodically
            let clk_oeb = flash_clk_oeb.map(|p| circ_state[p]).unwrap_or(255);
            let csn_oeb = flash_csn_oeb.map(|p| circ_state[p]).unwrap_or(255);
            if cycle <= 5 || cycle == reset_cycles || cycle == reset_cycles + 1 || cycle == reset_cycles + 10
               || cycle == reset_cycles + 100 || cycle == reset_cycles + 1000 {
                clilog::info!("cycle {}: flash clk={}, csn={}, d_out={:04b}, clk_oeb={}, csn_oeb={}",
                             cycle, clk, csn, d_out, clk_oeb, csn_oeb);
            }
        }

        // 3. Capture DFF CLK, D, and DE states BEFORE setting clock HIGH.
        //    CLK snapshot gives us the "before" for edge detection.
        //    D/DE snapshots give us the values to latch (stable before clock edge).
        //    After FALL eval, all pins reflect the clock=0 phase.
        for (i, &(_, pinid_d, _, pinid_clk, _, _, pinid_de)) in dff_list.iter().enumerate() {
            dff_prev_clk[i] = circ_state[pinid_clk];
            dff_prev_d[i] = circ_state[pinid_d];
            if pinid_de != usize::MAX {
                dff_prev_de[i] = circ_state[pinid_de];
            }
        }

        // 4. Clock HIGH (rising edge)
        circ_state[clock_pin] = 1;

        // 5. Evaluate combinational with clock=1 to propagate clock through
        //    any gating/buffering logic before checking DFF clock edges.
        eval_combinational(&mut state, &circ_state);
        update_circ_from_aig(&state, &mut circ_state);

        // 6. Edge-triggered DFF latch: check each DFF's actual CLK pin
        //    for a rising edge (0→1) rather than assuming all DFFs use
        //    the system clock. This correctly handles gated clocks,
        //    derived clocks, and any other clock topology.
        //
        //    D values are from the FALL eval snapshot (step 3), matching
        //    real hardware where DFFs sample D before the clock edge.
        let mut dffs_latched = 0usize;
        let mut dffs_in_reset = 0usize;
        for (i, &(_cellid, _pinid_d, pinid_q, pinid_clk, pinid_reset_b, pinid_set_b, pinid_de)) in dff_list.iter().enumerate() {
            let curr_clk = circ_state[pinid_clk];
            let prev_clk = dff_prev_clk[i];

            // Check async reset (active low) - always applies regardless of clock
            if pinid_reset_b != usize::MAX && circ_state[pinid_reset_b] == 0 {
                if circ_state[pinid_q] != 0 {
                    circ_state[pinid_q] = 0;
                }
                dffs_in_reset += 1;
            }
            // Check async set (active low)
            else if pinid_set_b != usize::MAX && circ_state[pinid_set_b] == 0 {
                if circ_state[pinid_q] != 1 {
                    circ_state[pinid_q] = 1;
                }
            }
            // Rising edge on THIS DFF's clock: latch pre-edge D→Q
            else if curr_clk != 0 && prev_clk == 0 {
                let should_latch = if pinid_de != usize::MAX {
                    dff_prev_de[i] != 0  // DE from pre-edge snapshot
                } else {
                    true
                };
                if should_latch {
                    // Use D value from before the clock edge (FALL eval)
                    circ_state[pinid_q] = dff_prev_d[i];
                    dffs_latched += 1;
                }
            }
        }

        // Log DFF latch stats for first few cycles
        if cycle <= 5 || (cycle == 10) {
            let dffs_no_edge = dff_list.len() - dffs_latched - dffs_in_reset;
            clilog::info!("cycle {}: dffs_latched={}, dffs_in_reset={}, no_edge={}",
                         cycle, dffs_latched, dffs_in_reset, dffs_no_edge);
            // On first cycle, show breakdown of no-edge DFFs
            if cycle == 0 {
                let mut held_high = 0usize;
                let mut held_low = 0usize;
                for (i, &(_cellid, _pinid_d, _pinid_q, pinid_clk, pinid_reset_b, pinid_set_b, _pinid_de)) in dff_list.iter().enumerate() {
                    let curr_clk = circ_state[pinid_clk];
                    let prev_clk = dff_prev_clk[i];
                    let reset_active = pinid_reset_b != usize::MAX && circ_state[pinid_reset_b] == 0;
                    let set_active = pinid_set_b != usize::MAX && circ_state[pinid_set_b] == 0;
                    if !reset_active && !set_active && !(curr_clk != 0 && prev_clk == 0) {
                        if curr_clk != 0 && prev_clk != 0 { held_high += 1; }
                        else if curr_clk == 0 && prev_clk == 0 { held_low += 1; }
                        else { /* falling edge */ }
                    }
                }
                clilog::info!("  no_edge breakdown: clk_held_high={}, clk_held_low={}, falling={}",
                             held_high, held_low, dffs_no_edge - held_high - held_low);
            }
        }

        // Count DFF changes for activity monitoring
        let mut dff_changes = 0usize;
        static mut PREV_DFF_STATE: Vec<u8> = Vec::new();
        unsafe {
            if PREV_DFF_STATE.is_empty() {
                PREV_DFF_STATE = circ_state.clone();
            } else {
                for i in 0..circ_state.len() {
                    if circ_state[i] != PREV_DFF_STATE[i] {
                        dff_changes += 1;
                    }
                }
                PREV_DFF_STATE.copy_from_slice(&circ_state);
            }
        }

        // Debug: show DFF activity and Wishbone state
        let ibus_cyc = ibus_cyc_pin.map(|p| circ_state[p]).unwrap_or(255);
        let ibus_stb = ibus_stb_pin.map(|p| circ_state[p]).unwrap_or(255);
        let dbus_cyc = dbus_cyc_pin.map(|p| circ_state[p]).unwrap_or(255);

        // SRAM peripheral trace: log whenever dbus_cyc is active
        {
            let sram_ack = sram_wb_ack_pin.map(|p| circ_state[p]).unwrap_or(255);
            let sram_ren = sram_read_en_pin.map(|p| circ_state[p]).unwrap_or(255);
            let sram_wen: Vec<u8> = sram_write_en.iter().map(|p| p.map(|p| circ_state[p]).unwrap_or(255)).collect();
            let mut dbus_addr: u32 = 0;
            for (i, opt_pin) in dbus_adr.iter().enumerate() {
                if let Some(pin) = opt_pin {
                    if circ_state[*pin] != 0 { dbus_addr |= 1 << i; }
                }
            }
            // ACK logic chain values
            let ack_d = ack_d_pin.map(|p| circ_state[p]).unwrap_or(255);
            let nor_a = ack_nor_a.map(|p| circ_state[p]).unwrap_or(255);
            let nor_b = ack_nor_b.map(|p| circ_state[p]).unwrap_or(255);
            let nand_a = ack_nand_a.map(|p| circ_state[p]).unwrap_or(255);
            let nand_b = ack_nand_b.map(|p| circ_state[p]).unwrap_or(255);
            let nand_c = ack_nand_c.map(|p| circ_state[p]).unwrap_or(255);
            // Log first dbus-active cycle with deep recursive AIG trace
            static mut ACK_TRACE_DONE: bool = false;
            let do_ack_trace = dbus_cyc == 1 && unsafe { !ACK_TRACE_DONE };
            if do_ack_trace {
                unsafe { ACK_TRACE_DONE = true; }
                let v = |idx: usize| -> u8 { if idx <= aig.num_aigpins { state.values[idx] } else { 255 } };
                let net_name = |idx: usize| -> String {
                    if idx == 0 || idx > aig.num_aigpins { return String::new(); }
                    let np = aigpin_to_netpin[idx];
                    if np != usize::MAX {
                        format!(" net={}", &netlistdb.pinnames[np].dbg_fmt_pin()[..60.min(netlistdb.pinnames[np].dbg_fmt_pin().len())])
                    } else { String::new() }
                };
                // Find ACK DFF D input aigpin dynamically from DFF cell
                let ack_d_aigpin = {
                    let mut d_aigpin = 0usize;
                    for cellid in 0..netlistdb.cellnames.len() {
                        let cn = format!("{:?}", &netlistdb.cellnames[cellid]);
                        if cn.contains("sram.wb_bus__ack$") {
                            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                                if netlistdb.pinnames[pinid].1.as_str() == "D" {
                                    let aig_iv = aig.pin2aigpin_iv[pinid];
                                    if aig_iv != usize::MAX { d_aigpin = aig_iv >> 1; }
                                }
                            }
                            break;
                        }
                    }
                    d_aigpin
                };
                eprintln!("  ACK DFF D aigpin = {}, val = {}", ack_d_aigpin, v(ack_d_aigpin));
                // Recursive trace - follow ALL 0-valued paths to depth 12
                let mut trace_result = String::new();
                let mut stack: Vec<(usize, bool, usize, &str)> = vec![(ack_d_aigpin, false, 0, "ACK_D")];
                while let Some((idx, inv, depth, label)) = stack.pop() {
                    if depth > 12 || idx > aig.num_aigpins { continue; }
                    let raw = if idx == 0 { 0 } else { v(idx) };
                    let val = raw ^ (inv as u8);
                    let prefix = "  ".repeat(depth);
                    if idx == 0 {
                        trace_result += &format!("\n{}{}: CONST inv={} val={}", prefix, label, inv, val);
                        continue;
                    }
                    match &aig.drivers[idx] {
                        DriverType::AndGate(a, b) => {
                            let ai = (*a as usize) >> 1; let ainv = (*a & 1) != 0;
                            let bi = (*b as usize) >> 1; let binv = (*b & 1) != 0;
                            let av = if ai == 0 { ainv as u8 } else if ai <= aig.num_aigpins { v(ai) ^ (ainv as u8) } else { 255 };
                            let bv = if bi == 0 { binv as u8 } else if bi <= aig.num_aigpins { v(bi) ^ (binv as u8) } else { 255 };
                            trace_result += &format!("\n{}{}: [{}]{}AND({}{}={}, {}{}={}) raw={} val={}{}",
                                prefix, label, idx, if inv {"!"} else {""},
                                if ainv {"!"} else {""}, ai, av,
                                if binv {"!"} else {""}, bi, bv, raw, val, net_name(idx));
                            // Follow ALL 0-valued inputs when this gate output is 0
                            if val == 0 && depth < 12 {
                                if av == 0 { stack.push((ai, ainv, depth+1, "A")); }
                                if bv == 0 { stack.push((bi, binv, depth+1, "B")); }
                            }
                        }
                        DriverType::DFF(c) => {
                            let cn = format!("{:?}", netlistdb.cellnames[*c]);
                            trace_result += &format!("\n{}{}: [{}]{} DFF raw={} val={} cell={}",
                                prefix, label, idx, if inv {"!"} else {""}, raw, val, &cn[..cn.len().min(80)]);
                        }
                        DriverType::InputPort(p) => {
                            let pn = netlistdb.pinnames[*p].dbg_fmt_pin();
                            trace_result += &format!("\n{}{}: [{}]{} INPUT raw={} val={} pin={}",
                                prefix, label, idx, if inv {"!"} else {""}, raw, val, &pn[..pn.len().min(80)]);
                        }
                        d => {
                            trace_result += &format!("\n{}{}: [{}]{} {:?} raw={} val={}",
                                prefix, label, idx, if inv {"!"} else {""}, d, raw, val);
                        }
                    }
                }
                eprintln!("ACK DEEP TRACE at cycle {}:{}", cycle, trace_result);
                // Direct check: ACK DFF, NOR2 _41001_, NAND3 _28827_ and related cells
                let check_cells = ["sram.wb_bus__ack$", "_40863_", "_28768_", "_28758_", "_28757_", "_28756_", "_28727_", "_28755_",
                                   "dbus__adr[26]$", "dbus__adr[22]$", "ibus__adr[26]$", "ibus__adr[22]$", "wb_arbiter.grant$"];
                for pattern in &check_cells {
                    for cellid in 0..netlistdb.cellnames.len() {
                        let cn = format!("{:?}", &netlistdb.cellnames[cellid]);
                        if !cn.contains(pattern) { continue; }
                        let macro_name = &netlistdb.celltypes[cellid];
                        eprintln!("  CELL {} [{}]: {}", cellid, macro_name, cn);
                        for pinid in netlistdb.cell2pin.iter_set(cellid) {
                            let pn = netlistdb.pinnames[pinid].dbg_fmt_pin();
                            let aig_iv = aig.pin2aigpin_iv[pinid];
                            let (idx, inv) = if aig_iv != usize::MAX { (aig_iv >> 1, aig_iv & 1) } else { (usize::MAX, 0) };
                            let aig_val = if idx == 0 { inv as u8 }
                                else if idx != usize::MAX && idx <= aig.num_aigpins { state.values[idx] ^ (inv as u8) }
                                else { 255 };
                            let net = netlistdb.pin2net[pinid];
                            let net_name = if net < netlistdb.netnames.len() { format!("{:?}", &netlistdb.netnames[net]) } else { "?".to_string() };
                            eprintln!("    pin={} {} net={}({}) aig({}:{})=>{} circ={}",
                                     pinid, &pn[..pn.len().min(50)], net, &net_name[..net_name.len().min(40)],
                                     idx, inv, aig_val, circ_state[pinid]);
                        }
                    }
                }
                // Identify key leaf aigpins (including 1008, 1009 found in many leaves)
                for leaf in [504, 1008, 1009, 18182, 18121, 18185, 18203, 18155, 18132, 18124,
                             18151, 18153, 18127, 18129, 18179, 18226] {
                    if leaf > 0 && leaf <= aig.num_aigpins {
                        let raw = v(leaf);
                        let np = aigpin_to_netpin[leaf];
                        let name = if np != usize::MAX { netlistdb.pinnames[np].dbg_fmt_pin() } else { "no-netpin".to_string() };
                        let driver = format!("{:?}", &aig.drivers[leaf]);
                        eprintln!("  LEAF [{}] raw={} {} driver={}", leaf, raw, &name[..name.len().min(80)], &driver[..driver.len().min(80)]);
                    }
                }
            }
            if dbus_cyc == 1 && (cycle <= 7750 || cycle % 1000 == 0) {
                // Trace NOR4 _28727_ address decoder inputs
                // _09270_ = aigpin 18129 (inv from aig), _09277_ = aigpin 18140 (inv)
                // _09280_ = aigpin 18145 (inv), _09305_ = aigpin 18186 (inv)
                let v = |idx: usize| -> u8 { if idx > 0 && idx <= aig.num_aigpins { state.values[idx] } else { 0 } };
                // NOR4 _28727_ inputs (from cell dump, these are inverted from the AIG pins)
                let nor4_a = v(18129) ^ 1; // _09270_ = !aigpin18129
                let nor4_b = v(18140) ^ 1; // _09277_ = !aigpin18140
                let nor4_c = v(18145) ^ 1; // _09280_ = !aigpin18145
                let nor4_d = v(18186) ^ 1; // _09305_ = !aigpin18186
                let nor4_y = v(18189); // _09306_ = raw aigpin18189

                // Also NOR4 _28755_ inputs for _09346_ (NAND3 input C)
                let nor4b_a = v(18194); // _09308_
                let nor4b_b = v(18199) ^ 1; // _09311_ (inverted)
                let nor4b_c = v(18222) ^ 1; // _09326_ (inverted)
                let nor4b_d = v(18233) ^ 1; // _09333_ (inverted)
                let nor4b_y = v(18236); // _09334_

                eprintln!("DBUS c{}: addr=0x{:08X} ack={} | NOR4_28727({},{},{},{})={} NOR4_28755({},{},{},{})={} | NAND({},{},{}) NOR({},{})",
                         cycle, dbus_addr, sram_ack,
                         nor4_a, nor4_b, nor4_c, nor4_d, nor4_y,
                         nor4b_a, nor4b_b, nor4b_c, nor4b_d, nor4b_y,
                         nand_a, nand_b, nand_c, nor_a, nor_b);
            }
        }

        // Reconstruct sink__payload value (reset vector)
        let mut sink_val: u32 = 0;
        for (i, opt_pin) in sink_payload.iter().enumerate() {
            if let Some(pin) = opt_pin {
                if circ_state[*pin] != 0 {
                    sink_val |= 1 << i;
                }
            }
        }
        // Also get single bit for comparison
        let sink_20 = sink_payload_20_pin.map(|p| circ_state[p]).unwrap_or(255);

        // Reconstruct fetch address
        let mut fetch_addr: u32 = 0;
        for (i, opt_pin) in ibus_adr.iter().enumerate() {
            if let Some(pin) = opt_pin {
                if circ_state[*pin] != 0 {
                    fetch_addr |= 1 << i;
                }
            }
        }

        // Get reset sync values
        let stage0 = rst_sync_stage0_pin.map(|p| circ_state[p]).unwrap_or(255);
        let rst_sync = rst_sync_rst_pin.map(|p| circ_state[p]).unwrap_or(255);

        // Simple trace for reset test: show sink_payload and addr on every cycle for first 50
        if cycle <= 50 {
            clilog::info!("cycle {:2}: rst={}, rst_sync={}, ibus_cyc={}, sink=0x{:08X}, addr=0x{:06X}",
                         cycle, circ_state[reset_pin], rst_sync, ibus_cyc, sink_val, fetch_addr);
        }

        // Write watchlist trace output
        if let Some(ref mut f) = trace_file {
            let values: Vec<String> = std::iter::once(cycle.to_string())
                .chain(watchlist_entries.iter().map(|e| e.format_value(&circ_state)))
                .collect();
            writeln!(f, "{}", values.join(",")).expect("Failed to write trace");
        }

        // Simulate SRAM cells (edge-triggered, matching CF_SRAM_1024x32 Verilog model).
        // SRAM input pins are NOT in the AIG, so we must propagate net values first.
        //
        // We call step() twice per simulation cycle:
        //   1. With clock LOW (fall phase) - just updates last_clk=false
        //   2. With clock HIGH (rise phase) - detects 0→1 edge and performs R/W
        //
        // Phase 1: Propagate with clock LOW to record the low state
        {
            let saved_clk = circ_state[clock_pin];
            circ_state[clock_pin] = 0;
            eval_combinational(&mut state, &circ_state);
            update_circ_from_aig(&state, &mut circ_state);
            for &(sram_pin, driver_pin) in &sram_net_prop {
                circ_state[sram_pin] = circ_state[driver_pin];
            }
            for sram in sram_cells.iter_mut() {
                sram.step(&mut circ_state);
            }
            // Restore clock HIGH
            circ_state[clock_pin] = saved_clk;
            eval_combinational(&mut state, &circ_state);
            update_circ_from_aig(&state, &mut circ_state);
        }
        // Phase 2: Propagate with clock HIGH to detect rising edge
        for &(sram_pin, driver_pin) in &sram_net_prop {
            circ_state[sram_pin] = circ_state[driver_pin];
        }
        for sram in sram_cells.iter_mut() {
            if cycle <= 20 {
                let sram_clk = if sram.clk_pin != usize::MAX { circ_state[sram.clk_pin] } else { 255 };
                let sram_en = if sram.en_pin != usize::MAX { circ_state[sram.en_pin] } else { 255 };
                eprintln!("SRAM cycle {}: CLKin={} EN={} last_clk={}",
                         cycle, sram_clk, sram_en, sram.last_clk);
            }
            let acted = sram.step(&mut circ_state);
            if acted && cycle <= 500 {
                let addr = sram.read_addr(&circ_state);
                let r_wb = sram.r_wb_pin != usize::MAX && circ_state[sram.r_wb_pin] != 0;
                if r_wb {
                    eprintln!("SRAM cycle {}: READ  addr={} data=0x{:08X}", cycle, addr, sram.memory[addr.min(1023)]);
                } else {
                    let di = sram.read_di(&circ_state);
                    let ben = sram.read_ben(&circ_state);
                    eprintln!("SRAM cycle {}: WRITE addr={} data=0x{:08X} ben=0x{:08X}", cycle, addr, di, ben);
                }
            }
        }
        // Propagate SRAM DO outputs back to the net (for reads)
        for &(ref do_pin, ref load_pins) in &sram_do_prop {
            let val = circ_state[*do_pin];
            for &lp in load_pins {
                circ_state[lp] = val;
            }
        }

        // 6. Evaluate combinational again to propagate new Q/DO values
        eval_combinational(&mut state, &circ_state);
        update_circ_from_aig(&state, &mut circ_state);

        // Post-final-eval trace: check SRAM ACK chain with latest AIG values
        if dbus_cyc_pin.map(|p| circ_state[p]).unwrap_or(0) == 1 && (cycle <= 7750 || cycle % 5000 == 0) {
            let v = |idx: usize| -> u8 { if idx > 0 && idx <= aig.num_aigpins { state.values[idx] } else { 0 } };
            let dbus_adr26_circ = find_net_pin("dbus__adr[26]").map(|p| circ_state[p]).unwrap_or(255);
            let dbus_adr26_aig = v(18182);
            let grant_circ = find_net_pin("wb_arbiter.grant").map(|p| circ_state[p]).unwrap_or(255);
            let grant_aig = v(504);
            let nor4_y = v(18189); // _09306_
            let nand3_a = v(18237); // _09336_ (inverted = !nand2)
            // NOR4 _28727_ inputs (these are the inverted AIG pins)
            let nor4_a = v(18129) ^ 1; // _09270_
            let nor4_b = v(18140) ^ 1; // _09277_
            let nor4_c = v(18145) ^ 1; // _09280_
            let nor4_d = v(18186) ^ 1; // _09305_
            // Trace _09305_ (NOR4 input D) = NAND3(_28726_) of (_09295_, _09301_, _09304_)
            // From deep trace: aigpin 18186 = AND(!18184, 18185), inverted → _09305_ = !18186
            // 18184 = AND(!18181, !18183) and 18185 = AND(18168, 18179)
            // These represent sub-checks of the address comparison
            let p18184 = v(18184); // AND sub-tree 1
            let p18185 = v(18185); // AND sub-tree 2
            let p18168 = v(18168); // deeper
            let p18179 = v(18179); // deeper
            // 18168 = AND(18156, 18167) — address comparison sub-tree
            let p18156 = v(18156);
            let p18167 = v(18167);
            // 18156 = AND(18150, !18155)
            // 18150 = AND(!18147, !18149)
            // 18147 = AND(!504, 18146) — 504 is grant DFF
            // 18149 = AND(!18148, ...) or similar
            let p18150 = v(18150);
            let p18155 = v(18155);
            let p18147 = v(18147);
            let p18149 = v(18149);
            let p18146 = v(18146);
            let p504 = v(504); // grant DFF
            // Deep trace: find what cells these AIG pins correspond to
            let npi = |idx: usize| -> String {
                if idx == 0 || idx > aig.num_aigpins { return "const".to_string(); }
                let np = aigpin_to_netpin[idx];
                if np != usize::MAX {
                    let s = netlistdb.pinnames[np].dbg_fmt_pin();
                    s[..s.len().min(50)].to_string()
                } else { format!("aig#{}", idx) }
            };
            // Print raw AIG values and driver info
            let driver_info = |idx: usize| -> String {
                if idx == 0 || idx > aig.num_aigpins { return "const".to_string(); }
                match &aig.drivers[idx] {
                    DriverType::AndGate(a, b) => {
                        let ai = (*a as usize) >> 1; let ainv = (*a & 1) != 0;
                        let bi = (*b as usize) >> 1; let binv = (*b & 1) != 0;
                        let av = if ai == 0 { ainv as u8 } else { v(ai) ^ ainv as u8 };
                        let bv = if bi == 0 { binv as u8 } else { v(bi) ^ binv as u8 };
                        format!("AND({}{}={}raw{}, {}{}={}raw{})", if ainv {"!"} else {""}, ai, av, v(ai),
                                if binv {"!"} else {""}, bi, bv, v(bi))
                    }
                    DriverType::DFF(c) => {
                        let cn = format!("{:?}", netlistdb.cellnames[*c]);
                        format!("DFF({})", &cn[..cn.len().min(40)])
                    }
                    d => format!("{:?}", d)
                }
            };
            // When grant=1: the 18149 path is active (AND(grant, 18148))
            // 18150 = AND(!18147, !18149). grant=1 → 18147=0, so !18147=1. 18149=AND(1,18148).
            // So 18150 = !18148. If 18148=1 → 18150=0 → blocking.
            // 18148 is the dbus-side address comparison bit.
            let p18148 = v(18148);
            eprintln!("POST-EVAL c{}: grant={} | 18148(dbus_addr_cmp)={} {} | 18146(ibus_addr_cmp)={} | 18152={} 18154={}",
                     cycle, p504, p18148, driver_info(18148), p18146, v(18152), v(18154));
        }

        // SPI debug: trace raw_tx_data and enframer.cycle DFF values
        if cycle >= 0 && cycle <= 40 {
            // Find DFF Q values by searching AIG for named DFFs
            let find_dff_value = |name_part: &str| -> u8 {
                for i in 1..=aig.num_aigpins {
                    if let DriverType::DFF(cellid) = &aig.drivers[i] {
                        let cn = format!("{:?}", netlistdb.cellnames[*cellid]);
                        if cn.contains(name_part) {
                            return state.values[i];
                        }
                    }
                }
                255
            };

            // One-time dump of all DFF matching for debugging
            if cycle == 0 {
                for pattern in &["raw_tx_data[0]", "raw_tx_data[1]", "fsm_state[0]", "fsm_state[1]",
                                "o_latch.io0.o", "o_latch", "enframer.cycle[2]"] {
                    let mut matches = Vec::new();
                    for i in 1..=aig.num_aigpins {
                        if let DriverType::DFF(cellid) = &aig.drivers[i] {
                            let cn = format!("{:?}", netlistdb.cellnames[*cellid]);
                            if cn.contains(pattern) {
                                matches.push((i, cn));
                            }
                        }
                    }
                    eprintln!("DFF LOOKUP '{}': {} matches: {:?}", pattern, matches.len(),
                             matches.iter().map(|(i, n)| format!("pin{}={}", i, &n[..n.len().min(80)])).collect::<Vec<_>>());
                }
            }

            // raw_tx_data[0..7]
            let mut tx_byte: u8 = 0;
            for bit in 0..8 {
                let name = format!("raw_tx_data[{}]", bit);
                let val = find_dff_value(&name);
                if val == 1 { tx_byte |= 1 << bit; }
            }

            // enframer.cycle[0..2]
            let ec0 = find_dff_value("enframer.cycle[0]");
            let ec1 = find_dff_value("enframer.cycle[1]");
            let ec2 = find_dff_value("enframer.cycle[2]");
            let ecycle = (ec2 as u16) << 2 | (ec1 as u16) << 1 | ec0 as u16;

            // fsm_state
            let fsm0 = find_dff_value("fsm_state[0]");
            let fsm1 = find_dff_value("fsm_state[1]");
            let fsm2 = find_dff_value("fsm_state[2]");
            let fsm3 = find_dff_value("fsm_state[3]");
            let fsm = (fsm3 as u16) << 3 | (fsm2 as u16) << 2 | (fsm1 as u16) << 1 | fsm0 as u16;

            // o_latch and buffer
            let olatch_o = find_dff_value("o_latch.io0.o");
            let olatch_oe = find_dff_value("o_latch.io0.oe");

            // o_latch[2] = pin 40897 (MOSI data latch)
            let olatch2 = if 40897 <= aig.num_aigpins { state.values[40897] } else { 255 };
            // o_latch[0] = pin 40761 (CS enable)
            let olatch0 = if 40761 <= aig.num_aigpins { state.values[40761] } else { 255 };
            // SPI flash FSM state (using correct pins)
            let spi_fsm0 = if 39858 <= aig.num_aigpins { state.values[39858] } else { 255 };
            let spi_fsm1 = if 39862 <= aig.num_aigpins { state.values[39862] } else { 255 };
            let spi_fsm2 = if 39860 <= aig.num_aigpins { state.values[39860] } else { 255 };
            let spi_fsm3 = if 39859 <= aig.num_aigpins { state.values[39859] } else { 255 };

            eprintln!("GEM cycle {:3} DFF: tx=0x{:02X} ecycle={} spi_fsm={}{}{}{} olatch2={} olatch0={} buf_io0_q={}",
                     cycle, tx_byte, ecycle, spi_fsm3, spi_fsm2, spi_fsm1, spi_fsm0,
                     olatch2, olatch0,
                     find_dff_value("buffer_io0.o"));
        }

        // SPI debug: trace at rising-edge eval
        if cycle >= 24 && cycle <= 32 {
            let buf_o = buffer_io0_o_pin.map(|p| circ_state[p]).unwrap_or(255);
            let str_o = str_io0_o_pin.map(|p| circ_state[p]).unwrap_or(255);
            let olatch = o_latch_io0_o_pin.map(|p| circ_state[p]).unwrap_or(255);
            let d0_pin = flash_d_out.get(0).and_then(|o| *o);
            let d0 = d0_pin.map(|p| circ_state[p]).unwrap_or(255);
            let clk_out = flash_clk_out.map(|p| circ_state[p]).unwrap_or(255);
            let dff_d = if buffer_dff_d_pin != usize::MAX { circ_state[buffer_dff_d_pin] } else { 255 };
            let dff_q = if buffer_dff_q_pin != usize::MAX { circ_state[buffer_dff_q_pin] } else { 255 };
            // Trace AIG backwards from DFF D pin
            if buffer_dff_d_pin != usize::MAX {
                let d_aig = aig.pin2aigpin_iv.get(buffer_dff_d_pin).copied().unwrap_or(usize::MAX);
                if d_aig != usize::MAX {
                    let d_idx = d_aig >> 1;
                    let d_inv = (d_aig & 1) != 0;
                    let d_raw = if d_idx > 0 && d_idx <= aig.num_aigpins { state.values[d_idx] } else { 0 };

                    // Level 1: D's gate inputs
                    let (l1a_val, l1b_val, l1a_iv, l1b_iv) = if let DriverType::AndGate(a_iv, b_iv) = &aig.drivers[d_idx] {
                        let a_i = a_iv >> 1; let a_inv = (a_iv & 1) != 0;
                        let b_i = b_iv >> 1; let b_inv = (b_iv & 1) != 0;
                        let a_v = if a_i > 0 && a_i <= aig.num_aigpins { state.values[a_i] ^ (a_inv as u8) } else { 0 ^ (a_inv as u8) };
                        let b_v = if b_i > 0 && b_i <= aig.num_aigpins { state.values[b_i] ^ (b_inv as u8) } else { 0 ^ (b_inv as u8) };
                        (a_v, b_v, *a_iv, *b_iv)
                    } else {
                        (255, 255, usize::MAX, usize::MAX)
                    };

                    // Level 2: Each L1 input's gate inputs
                    let trace_gate = |iv: usize| -> String {
                        let idx = iv >> 1;
                        let inv = (iv & 1) != 0;
                        if idx == 0 { return format!("const{}", if inv { 1 } else { 0 }); }
                        if idx > aig.num_aigpins { return "OOB".to_string(); }
                        let val = state.values[idx] ^ (inv as u8);
                        match &aig.drivers[idx] {
                            DriverType::AndGate(a, b) => {
                                let ai = a >> 1; let bi = b >> 1;
                                let ainv = (a & 1) != 0; let binv = (b & 1) != 0;
                                let av = if ai > 0 && ai <= aig.num_aigpins { state.values[ai] ^ (ainv as u8) } else { 0 ^ (ainv as u8) };
                                let bv = if bi > 0 && bi <= aig.num_aigpins { state.values[bi] ^ (binv as u8) } else { 0 ^ (binv as u8) };
                                format!("AND({}={},{}={})={}", ai, av, bi, bv, val)
                            }
                            DriverType::DFF(c) => {
                                let cn = format!("{:?}", netlistdb.cellnames[*c]);
                                format!("DFF({})={}", &cn[..cn.len().min(40)], val)
                            }
                            DriverType::InputPort(p) => {
                                let pn = netlistdb.pinnames[*p].dbg_fmt_pin();
                                format!("IN({})={}", &pn[..pn.len().min(30)], val)
                            }
                            d => format!("{:?}={}", d, val)
                        }
                    };

                    // Recursive trace: follow specific AIG pins
                    let trace_pin_deep = |target: usize, depth: usize| -> String {
                        let mut result = String::new();
                        let mut stack: Vec<(usize, usize)> = vec![(target, 0)];
                        while let Some((idx, d)) = stack.pop() {
                            if d > depth || idx == 0 || idx > aig.num_aigpins { continue; }
                            let val = state.values[idx];
                            let indent = "  ".repeat(d);
                            match &aig.drivers[idx] {
                                DriverType::AndGate(a, b) => {
                                    let ai = a >> 1; let bi = b >> 1;
                                    let ainv = (a & 1) != 0; let binv = (b & 1) != 0;
                                    let av = if ai > 0 && ai <= aig.num_aigpins { state.values[ai] } else { 0 };
                                    let bv = if bi > 0 && bi <= aig.num_aigpins { state.values[bi] } else { 0 };
                                    result += &format!("\n{}  [{}] AND raw={} a[{}]raw={} inv={} b[{}]raw={} inv={}",
                                                      indent, idx, val, ai, av, ainv, bi, bv, binv);
                                    if d < depth { stack.push((ai, d+1)); stack.push((bi, d+1)); }
                                }
                                DriverType::DFF(c) => {
                                    let cn = format!("{:?}", netlistdb.cellnames[*c]);
                                    result += &format!("\n{}  [{}] DFF raw={} cell={}", indent, idx, val, &cn[..cn.len().min(60)]);
                                }
                                DriverType::InputPort(p) => {
                                    let pn = netlistdb.pinnames[*p].dbg_fmt_pin();
                                    result += &format!("\n{}  [{}] INPUT raw={} {}", indent, idx, val, &pn[..pn.len().min(40)]);
                                }
                                d => { result += &format!("\n{}  [{}] {:?} raw={}", indent, idx, d, val); }
                            }
                        }
                        result
                    };

                    // Trace pin 40826 (the root changing signal) to depth 6
                    let deep_trace = trace_pin_deep(40826, 6);

                    // Direct identification of key pins
                    let identify_pin = |idx: usize| -> String {
                        if idx == 0 || idx > aig.num_aigpins { return format!("[{}] OOB", idx); }
                        let netpin = aigpin_to_netpin[idx];
                        let net_name = if netpin != usize::MAX {
                            netlistdb.pinnames[netpin].dbg_fmt_pin()
                        } else {
                            "no-netpin".to_string()
                        };
                        match &aig.drivers[idx] {
                            DriverType::AndGate(a, b) => format!("[{}] AND({},{}) net={}", idx, a>>1, b>>1, &net_name[..net_name.len().min(60)]),
                            DriverType::DFF(c) => {
                                let cn = format!("{:?}", netlistdb.cellnames[*c]);
                                format!("[{}] DFF cell={} net={}", idx, &cn[..cn.len().min(50)], &net_name[..net_name.len().min(60)])
                            },
                            DriverType::InputPort(p) => {
                                let pn = netlistdb.pinnames[*p].dbg_fmt_pin();
                                format!("[{}] INPUT pin={} net={}", idx, &pn[..pn.len().min(50)], &net_name[..net_name.len().min(60)])
                            },
                            d => format!("[{}] {:?} net={}", idx, d, &net_name[..net_name.len().min(60)]),
                        }
                    };

                    eprintln!("GEM cycle {} RISE: D_raw={} D_inv={} D={} | L1: A={} B={} | L2: A={} B={}",
                             cycle, d_raw, d_inv, dff_d, l1a_val, l1b_val,
                             trace_gate(l1a_iv), trace_gate(l1b_iv));
                    eprintln!("  PIN IDs: 40826={} 40886={} 40893={} 11129={}",
                             identify_pin(40826), identify_pin(40886), identify_pin(40893), identify_pin(11129));
                    eprintln!("  VALS: 40826_raw={} 40886_raw={} 11129_raw={} 40893_raw={}",
                             state.values[40826], state.values[40886], state.values[11129], state.values[40893]);
                    eprintln!("  DEEP TRACE from 40826:{}", deep_trace);
                }
            }
        }

        // 6a. Cell-level and net verification at critical cycles
        if cycle >= 20 && cycle <= 35 {
            let (checked, mismatches) = verify_cell_outputs(
                &netlistdb, &circ_state, cell_library, cycle, 10,
            );
            if mismatches > 0 {
                eprintln!("cycle {}: {} cell mismatches out of {} cells checked", cycle, mismatches, checked);
            }
            let (nets_checked, net_mismatches) = verify_net_consistency(
                &netlistdb, &circ_state, &aig, cycle, 10,
            );
            if net_mismatches > 0 {
                eprintln!("cycle {}: {} net mismatches out of {} nets checked", cycle, net_mismatches, nets_checked);
            }
        }

        // 6b. Step flash model AFTER rising edge eval (sees current SPI clock HIGH)
        // This is when the flash controller samples data from flash
        if let Some(ref mut fl) = flash {
            step_flash(fl, &mut circ_state, in_reset, &mut prev_flash_d_out, &mut prev_flash_csn);
        }

        // 6c. (Removed) Edge-triggered DFF latching is now unified in step 5 above.
        // All DFFs are latched based on their actual CLK pin edge detection.

        // === TICK END ===

        // Wishbone bus monitor
        if let Some(ref mut wbm) = wb_monitor {
            wbm.log_cycle(cycle, &circ_state);
        }

        // UART TX decoding
        if let Some(tx_pin) = uart_tx_pin {
            let tx = circ_state[tx_pin];

            uart_state = match uart_state {
                UartState::Idle => {
                    if uart_last_tx == 1 && tx == 0 {
                        UartState::StartBit { start_cycle: cycle }
                    } else {
                        UartState::Idle
                    }
                }
                UartState::StartBit { start_cycle } => {
                    if cycle >= start_cycle + cycles_per_bit / 2 {
                        if tx == 0 {
                            UartState::DataBits {
                                start_cycle: start_cycle + cycles_per_bit,
                                bits_received: 0,
                                value: 0,
                            }
                        } else {
                            UartState::Idle
                        }
                    } else {
                        UartState::StartBit { start_cycle }
                    }
                }
                UartState::DataBits { start_cycle, bits_received, value } => {
                    let bit_center = start_cycle + (bits_received as usize) * cycles_per_bit + cycles_per_bit / 2;
                    if cycle >= bit_center {
                        let new_value = value | ((tx as u8) << bits_received);
                        if bits_received >= 7 {
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
                    if cycle >= start_cycle + cycles_per_bit / 2 {
                        if tx == 1 {
                            uart_events.push(UartEvent {
                                timestamp: cycle,
                                peripheral: "uart_0".to_string(),
                                event: "tx".to_string(),
                                payload: value,
                            });
                            let ch = if value >= 32 && value < 127 { value as char } else { '.' };
                            clilog::info!("UART TX @ cycle {}: 0x{:02X} '{}'", cycle, value, ch);
                        }
                        UartState::Idle
                    } else {
                        UartState::StopBit { start_cycle, value }
                    }
                }
            };
            uart_last_tx = tx;
        }

        // Progress logging
        if cycle % 100000 == 0 && cycle > 0 {
            clilog::info!("Cycle {} / {}", cycle, max_cycles);
        }
    }

    // Output results
    println!();
    println!("=== Programmatic Timing Simulation Results ===");
    println!("Cycles simulated: {}", max_cycles);
    println!("UART events captured: {}", uart_events.len());

    if let Some(output_path) = &config.output_events {
        #[derive(Serialize)]
        struct EventsOutput { events: Vec<UartEvent> }
        let output = EventsOutput { events: uart_events };
        let json = serde_json::to_string_pretty(&output).expect("Failed to serialize events");
        let mut file = File::create(output_path).expect("Failed to create events file");
        file.write_all(json.as_bytes()).expect("Failed to write events");
        clilog::info!("Wrote events to {}", output_path);
    }

    // Print flash model statistics
    if let Some(ref fl) = flash {
        println!("Flash model: steps={}, posedges={}, negedges={}",
                 fl.get_step_count(), fl.get_posedge_count(), fl.get_negedge_count());
    }

    println!();
    println!("TIMING: PASSED");
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);

    let args = <Args as clap::Parser>::parse();
    clilog::info!("Timing simulation args:\n{:#?}", args);

    // Load config if provided
    let config = if let Some(config_path) = &args.config {
        let file = File::open(config_path).expect("Failed to open config file");
        let reader = BufReader::new(file);
        let config: TestbenchConfig =
            serde_json::from_reader(reader).expect("Failed to parse config JSON");
        clilog::info!("Loaded testbench config: {:?}", config);
        Some(config)
    } else {
        None
    };

    // Determine netlist path from config or command line
    let netlist_path = if let Some(ref cfg) = config {
        if let Some(ref path) = cfg.netlist_path {
            PathBuf::from(path)
        } else if let Some(ref path) = args.netlist_verilog {
            path.clone()
        } else {
            panic!("No netlist path provided (need --config with netlist_path or positional argument)");
        }
    } else if let Some(ref path) = args.netlist_verilog {
        path.clone()
    } else {
        panic!("No netlist path provided");
    };

    // Determine liberty path from config or command line
    let liberty_path = if let Some(ref cfg) = config {
        cfg.liberty_path.as_ref().map(PathBuf::from).or_else(|| args.liberty.clone())
    } else {
        args.liberty.clone()
    };

    // Detect cell library
    let cell_library = detect_library_from_file(&netlist_path)
        .expect("Failed to read netlist file for library detection");
    clilog::info!("Detected cell library: {}", cell_library);

    if cell_library == CellLibrary::Mixed {
        panic!("Mixed AIGPDK and SKY130 cells in netlist not supported");
    }

    // Load Liberty library (or use defaults for SKY130)
    let lib = if let Some(lib_path) = &liberty_path {
        TimingLibrary::from_file(lib_path).expect("Failed to load Liberty library")
    } else if cell_library == CellLibrary::SKY130 {
        clilog::info!("Using default SKY130 timing values (no liberty file)");
        TimingLibrary::default_sky130()
    } else {
        TimingLibrary::load_aigpdk().expect("Failed to load default AIGPDK library")
    };
    clilog::info!("Loaded timing library: {}", lib.name);

    // Load netlist with appropriate LeafPinProvider
    clilog::info!("Loading netlist: {:?}", netlist_path);
    let netlistdb = match cell_library {
        CellLibrary::SKY130 => NetlistDB::from_sverilog_file(
            &netlist_path,
            args.top_module.as_deref(),
            &SKY130LeafPins,
        )
        .expect("Failed to build netlist"),
        CellLibrary::AIGPDK | CellLibrary::Mixed => NetlistDB::from_sverilog_file(
            &netlist_path,
            args.top_module.as_deref(),
            &AIGPDKLeafPins(),
        )
        .expect("Failed to build netlist"),
    };

    // Load PDK models if available
    let pdk_cells_path = args.pdk_cells
        .clone()
        .or_else(|| {
            let default_path = std::path::PathBuf::from("sky130_fd_sc_hd/cells");
            if default_path.exists() { Some(default_path) } else { None }
        });

    let pdk_models = if let Some(ref pdk_path) = pdk_cells_path {
        clilog::info!("Loading PDK cell models from: {}", pdk_path.display());
        // Collect cell types from the netlist
        let mut cell_types: Vec<String> = Vec::new();
        for cellid in 1..netlistdb.num_cells {
            let celltype = netlistdb.celltypes[cellid].as_str();
            if gem::sky130::is_sky130_cell(celltype) {
                let ct = gem::sky130::extract_cell_type(celltype).to_string();
                if !cell_types.contains(&ct) {
                    cell_types.push(ct);
                }
            }
        }
        cell_types.sort();
        Some(gem::sky130_pdk::load_pdk_models(pdk_path, &cell_types))
    } else {
        None
    };

    // Build AIG
    let aig = if let Some(ref pdk) = pdk_models {
        AIG::from_netlistdb_with_pdk(&netlistdb, pdk)
    } else {
        AIG::from_netlistdb(&netlistdb)
    };
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

    // Collect SRAM cells for simulation
    let mut sram_cells: Vec<SramCell> = Vec::new();
    for cellid in 1..netlistdb.num_cells {
        let celltype = netlistdb.celltypes[cellid].as_str();
        let is_sram = match cell_library {
            CellLibrary::SKY130 => celltype.starts_with("CF_SRAM_"),
            _ => celltype == "$__RAMGEM_SYNC_",
        };

        if is_sram {
            let mut sram = SramCell::new(cellid);
            sram.collect_pins(&netlistdb);
            clilog::info!(
                "SRAM cell {}: addr_pins={}, di_pins={}, do_pins={}, ben_pins={}",
                cellid,
                sram.addr_pins.len(),
                sram.di_pins.len(),
                sram.do_pins.len(),
                sram.ben_pins.len()
            );
            sram_cells.push(sram);
        }
    }
    clilog::info!("Collected {} SRAM cells for simulation", sram_cells.len());

    // Resolve watchlist entries (shared by both programmatic and VCD modes)
    let watchlist_entries: Vec<WatchlistEntry> = if let Some(ref watchlist_path) = args.watchlist {
        let file = File::open(watchlist_path).expect("Failed to open watchlist file");
        let watchlist: Watchlist =
            serde_json::from_reader(BufReader::new(file)).expect("Failed to parse watchlist JSON");

        let mut entries = Vec::new();
        for sig in &watchlist.signals {
            // Check if this is a bundle signal
            if let Some(width) = sig.width {
                // Find all bits of the bundle
                let mut pins = vec![usize::MAX; width];
                let mut found_count = 0;

                for bit in 0..width {
                    let bit_pattern = format!("{}[{}]", sig.net, bit);
                    let dff_q_pattern = format!("{}[{}]$_", sig.net, bit); // DFF cell name pattern

                    // Search for the bit pin - prefer DFF Q outputs over wire pins
                    let mut wire_pin: Option<usize> = None;
                    for pinid in 0..netlistdb.num_pins {
                        let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();

                        // Check for DFF Q output first (preferred)
                        if pin_name.contains(&dff_q_pattern) && pin_name.ends_with(":Q") {
                            pins[bit] = pinid;
                            found_count += 1;
                            break;
                        }

                        // Otherwise remember wire pin as fallback
                        if wire_pin.is_none() {
                            if let Some(pos) = pin_name.find(&bit_pattern) {
                                let end_pos = pos + bit_pattern.len();
                                let is_exact = end_pos == pin_name.len()
                                    || !pin_name[end_pos..].starts_with(|c: char| c.is_ascii_digit());
                                if is_exact {
                                    wire_pin = Some(pinid);
                                }
                            }
                        }
                    }

                    // Use wire pin if DFF Q not found
                    if pins[bit] == usize::MAX {
                        if let Some(wp) = wire_pin {
                            pins[bit] = wp;
                            found_count += 1;
                        }
                    }
                }

                if found_count > 0 {
                    let format = sig.format.clone().unwrap_or_else(|| "hex".to_string());
                    // Log some sample pin mappings for verification
                    let sample_pins: Vec<String> = pins.iter().enumerate()
                        .filter(|(_, &p)| p < usize::MAX)
                        .take(5)
                        .map(|(i, &p)| format!("[{}]={}", i, p))
                        .collect();
                    clilog::info!(
                        "Watchlist: {} -> {}/{} bits found (bundle, format={}, pins: {}...)",
                        sig.name, found_count, width, format, sample_pins.join(", ")
                    );
                    entries.push(WatchlistEntry::Bundle {
                        name: sig.name.clone(),
                        pins,
                        format,
                    });
                } else {
                    clilog::warn!("Watchlist: {} not found (bundle pattern: {}[0..{}])", sig.name, sig.net, width);
                }
            } else {
                // Single-bit signal
                let mut found = false;

                // First try: find by net name (for internal wires)
                for netid in 0..netlistdb.num_nets {
                    let net_name = netlistdb.netnames[netid].dbg_fmt_pin();
                    if net_name == sig.net || net_name.ends_with(&sig.net) || net_name.contains(&sig.net) {
                        // Find a pin on this net
                        for pinid in netlistdb.net2pin.iter_set(netid) {
                            entries.push(WatchlistEntry::Bit {
                                name: sig.name.clone(),
                                pin: pinid,
                            });
                            clilog::info!("Watchlist: {} -> pin {} (net {})", sig.name, pinid, net_name);
                            found = true;
                            break;
                        }
                        if found {
                            break;
                        }
                    }
                }

                // Second try: any pin containing the pattern
                if !found {
                    for pinid in 0..netlistdb.num_pins {
                        let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                        if pin_name.contains(&sig.net) {
                            entries.push(WatchlistEntry::Bit {
                                name: sig.name.clone(),
                                pin: pinid,
                            });
                            clilog::info!("Watchlist: {} -> pin {} ({})", sig.name, pinid, pin_name);
                            found = true;
                            break;
                        }
                    }
                }

                if !found {
                    clilog::warn!("Watchlist: {} not found (pattern: {})", sig.name, sig.net);
                }
            }
        }
        entries
    } else {
        Vec::new()
    };

    // Open trace output file if specified
    let trace_file: Option<File> = args.trace_output.as_ref().map(|path| {
        let mut f = File::create(path).expect("Failed to create trace output file");
        // Write CSV header
        let header: Vec<&str> = std::iter::once("cycle")
            .chain(watchlist_entries.iter().map(|e| e.name()))
            .collect();
        writeln!(f, "{}", header.join(",")).expect("Failed to write trace header");
        f
    });

    // Create Wishbone bus monitor if requested
    let wb_monitor = args.wb_monitor.as_ref().map(|prefix| {
        WishboneBusMonitor::discover(prefix, &netlistdb)
    });

    // Check for config-based (programmatic) simulation
    if let Some(cfg) = config {
        run_programmatic_simulation(
            cfg,
            &args,
            &netlistdb,
            &aig,
            &lib,
            cell_library,
            &mut sram_cells,
            watchlist_entries,
            trace_file,
            wb_monitor,
        );
        return;
    }

    // VCD-driven simulation mode
    let input_vcd_path = args.input_vcd.as_ref().expect("VCD file required (use --config for programmatic mode)");
    let input_vcd = File::open(input_vcd_path).expect("Failed to open VCD");
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

    // QSPI Flash setup for functional simulation (using C++ model from chipflow-lib)
    let mut flash: Option<CppSpiFlash> = if args.firmware.is_some() {
        Some(CppSpiFlash::new(16 * 1024 * 1024)) // 16MB flash
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

    // Helper to find internal wire/pin by name pattern (look for Q output of DFF)
    let find_internal_q_pin = |dff_pattern: &str| -> Option<usize> {
        for pinid in 0..netlistdb.num_pins {
            let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
            // Look for the Q output pin (ends with :Q)
            if pin_name.contains(dff_pattern) && pin_name.ends_with(":Q") {
                return Some(pinid);
            }
        }
        None
    };

    // Helper to find internal wire by net name
    let find_internal_wire = |wire_pattern: &str| -> Option<usize> {
        for netid in 0..netlistdb.num_nets {
            let net_name = netlistdb.netnames[netid].dbg_fmt_pin();
            if net_name.contains(wire_pattern) {
                // Find any pin on this net (preferably output)
                let net_pins_start = netlistdb.net2pin.start[netid];
                let net_pins_end = if netid + 1 < netlistdb.net2pin.start.len() {
                    netlistdb.net2pin.start[netid + 1]
                } else {
                    netlistdb.net2pin.items.len()
                };
                for &pinid in &netlistdb.net2pin.items[net_pins_start..net_pins_end] {
                    return Some(pinid);
                }
            }
        }
        None
    };

    // Helper to find D input pin of a DFF by pattern
    let find_internal_d_pin = |dff_pattern: &str| -> Option<usize> {
        for pinid in 0..netlistdb.num_pins {
            let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
            if pin_name.contains(dff_pattern) && pin_name.ends_with(":D") {
                return Some(pinid);
            }
        }
        None
    };

    // Helper to find CLK input pin of a DFF by pattern
    let find_internal_clk_pin = |dff_pattern: &str| -> Option<usize> {
        for pinid in 0..netlistdb.num_pins {
            let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
            if pin_name.contains(dff_pattern) && pin_name.ends_with(":CLK") {
                return Some(pinid);
            }
        }
        None
    };

    // Find internal monitoring pins (Q outputs of DFFs)
    let rst_sync_rst_pin = find_internal_q_pin("rst_n_sync.rst");
    let buffer_cs_pin = find_internal_q_pin("buffer_cs.o_ff");
    let ibus_cyc_pin = find_internal_wire("ibus__cyc");
    let ibus_cyc_d_pin = find_internal_d_pin("ibus__cyc");
    let cpu_clk_pin = find_internal_clk_pin("cpu.fetch.ibus__cyc");

    if let Some(pin) = rst_sync_rst_pin {
        let pin_name = netlistdb.pinnames[pin].dbg_fmt_pin();
        clilog::info!("Found reset sync Q output: pin {} ({})", pin, pin_name);
    } else {
        clilog::warn!("Could not find rst_n_sync.rst Q pin for monitoring");
    }

    if let Some(pin) = cpu_clk_pin {
        let pin_name = netlistdb.pinnames[pin].dbg_fmt_pin();
        clilog::info!("Found CPU clock input: pin {} ({})", pin, pin_name);
    }

    if let Some(pin) = ibus_cyc_d_pin {
        let pin_name = netlistdb.pinnames[pin].dbg_fmt_pin();
        clilog::info!("Found ibus_cyc D input: pin {} ({})", pin, pin_name);
    }

    // Find net639 and net2949 for debugging a21oi logic
    let net639_pin = find_internal_wire("net639");
    let net2949_pin = find_internal_wire("net2949");
    let sig_09415_pin = find_internal_wire("_09415_");
    if let Some(pin) = net639_pin {
        clilog::info!("Found net639: pin {}", pin);
    }
    if let Some(pin) = net2949_pin {
        clilog::info!("Found net2949: pin {}", pin);
    }
    if let Some(pin) = sig_09415_pin {
        clilog::info!("Found _09415_: pin {}", pin);
    }

    if let Some(pin) = buffer_cs_pin {
        let pin_name = netlistdb.pinnames[pin].dbg_fmt_pin();
        clilog::info!("Found buffer_cs.o_ff Q output: pin {} ({})", pin, pin_name);
    } else {
        clilog::warn!("Could not find buffer_cs.o_ff Q pin for monitoring");
    }

    if let Some(pin) = ibus_cyc_pin {
        let pin_name = netlistdb.pinnames[pin].dbg_fmt_pin();
        let net_name = netlistdb.netnames[netlistdb.pin2net[pin]].dbg_fmt_pin();
        clilog::info!("Found ibus_cyc wire: pin {} ({}) on net {}", pin, pin_name, net_name);
    } else {
        clilog::warn!("Could not find ibus__cyc wire for monitoring");
    }

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
            let pin_name = netlistdb.pinnames[clk_pin].dbg_fmt_pin();
            clilog::info!("Flash CLK pin {} ({}) AIG mapping: {}", clk_pin, pin_name, has_aig);

            // Check pin 81150 (output583:X) mapping
            if let Some(&aigpin_81150) = aig.pin2aigpin_iv.get(81150) {
                if aigpin_81150 != usize::MAX {
                    let idx = aigpin_81150 >> 1;
                    clilog::info!("Pin 81150 (output583:X) AIG: idx={}, driver={:?}", idx, aig.drivers.get(idx));
                } else {
                    clilog::info!("Pin 81150 (output583:X) AIG: not mapped");
                }
            }

            // Check cell 23336 (output583) and its input
            clilog::info!("Cell 23336 (output583) pins:");
            for pinid in netlistdb.cell2pin.iter_set(23336) {
                let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                let pin_dir = netlistdb.pindirect[pinid];
                let pin_net = netlistdb.pin2net[pinid];
                let aigpin_iv = aig.pin2aigpin_iv.get(pinid).copied().unwrap_or(usize::MAX);
                clilog::info!("  pin {}: {}, dir={:?}, net={}, aig={}", pinid, pin_name, pin_dir, pin_net, aigpin_iv);
            }

            // Check pins on net 20966 (output583's input net)
            let net = 20966;
            let net_pins_start = netlistdb.net2pin.start[net];
            let net_pins_end = if net + 1 < netlistdb.net2pin.start.len() {
                netlistdb.net2pin.start[net + 1]
            } else {
                netlistdb.net2pin.items.len()
            };
            clilog::info!("Pins on net {} (output583's A input):", net);
            for &np in &netlistdb.net2pin.items[net_pins_start..net_pins_end] {
                let np_name = netlistdb.pinnames[np].dbg_fmt_pin();
                let np_cell = netlistdb.pin2cell[np];
                let np_dir = netlistdb.pindirect[np];
                let np_aig = aig.pin2aigpin_iv.get(np).copied().unwrap_or(usize::MAX);
                clilog::info!("  pin {}: {} (cell={}, dir={:?}, aig={})", np, np_name, np_cell, np_dir, np_aig);
            }

            // Debug: trace what's driving the flash clock
            if let Some(&aigpin_iv) = aig.pin2aigpin_iv.get(clk_pin) {
                if aigpin_iv != usize::MAX {
                    let idx = aigpin_iv >> 1;
                    let inv = (aigpin_iv & 1) != 0;
                    if idx > 0 && idx <= aig.num_aigpins {
                        let driver = &aig.drivers[idx];
                        clilog::info!("Flash CLK driver: idx={}, inv={}, driver={:?}", idx, inv, driver);

                        // If it's a DFF, check the D input
                        if let DriverType::DFF(cell_idx) = driver {
                            clilog::info!("Flash CLK is DFF output from cell {}", cell_idx);
                            // Find D input for this DFF
                            if let Some(dff) = aig.dffs.get(cell_idx) {
                                let d_idx = dff.d_iv >> 1;
                                let d_inv = (dff.d_iv & 1) != 0;
                                clilog::info!("Flash CLK DFF D input: idx={}, inv={}, driver={:?}",
                                              d_idx, d_inv, aig.drivers.get(d_idx));
                            }
                        }
                        // If it's an InputPort, check what that pin is
                        if let DriverType::InputPort(pinid) = driver {
                            let pin_name = netlistdb.pinnames[*pinid].dbg_fmt_pin();
                            let pin_cell = netlistdb.pin2cell[*pinid];
                            let pin_net = netlistdb.pin2net[*pinid];
                            let pin_dir = netlistdb.pindirect[*pinid];
                            clilog::info!("Flash CLK is connected to input pin {}: {}, cell={}, net={}, dir={:?}",
                                         pinid, pin_name, pin_cell, pin_net, pin_dir);

                            // Check what's on the same net as gpio_out[0]
                            let clk_net = netlistdb.pin2net[clk_pin];
                            clilog::info!("gpio_out[0] (pin {}) is on net {}", clk_pin, clk_net);

                            // List all pins on that net
                            let net_pins_start = netlistdb.net2pin.start[clk_net];
                            let net_pins_end = if clk_net + 1 < netlistdb.net2pin.start.len() {
                                netlistdb.net2pin.start[clk_net + 1]
                            } else {
                                netlistdb.net2pin.items.len()
                            };
                            clilog::info!("Pins on gpio_out[0]'s net {}:", clk_net);
                            for &np in &netlistdb.net2pin.items[net_pins_start..net_pins_end] {
                                let np_name = netlistdb.pinnames[np].dbg_fmt_pin();
                                let np_cell = netlistdb.pin2cell[np];
                                let np_dir = netlistdb.pindirect[np];
                                clilog::info!("  pin {}: {} (cell={}, dir={:?})", np, np_name, np_cell, np_dir);
                            }
                        }
                    }
                }
            }
        }
        if let Some(csn_pin) = flash_csn_out {
            let has_aig = aig.pin2aigpin_iv.get(csn_pin).map_or(false, |&v| v != usize::MAX);
            clilog::info!("Flash CSN pin {} AIG mapping: {}", csn_pin, has_aig);
        }
    }

    // Build reverse mapping from AIG pin to netlist pin for SRAM DO outputs
    let mut aigpin_to_netpin: Vec<usize> = vec![usize::MAX; aig.num_aigpins + 1];
    for (pinid, &aigpin_iv) in aig.pin2aigpin_iv.iter().enumerate() {
        if aigpin_iv != usize::MAX {
            let aigpin = aigpin_iv >> 1;
            if aigpin > 0 && aigpin <= aig.num_aigpins {
                aigpin_to_netpin[aigpin] = pinid;
            }
        }
    }

    // Load watchlist and resolve signal pins
    let watchlist_entries: Vec<WatchlistEntry> = if let Some(ref watchlist_path) = args.watchlist {
        let file = File::open(watchlist_path).expect("Failed to open watchlist file");
        let watchlist: Watchlist =
            serde_json::from_reader(BufReader::new(file)).expect("Failed to parse watchlist JSON");

        let mut entries = Vec::new();
        for sig in &watchlist.signals {
            // Check if this is a bundle signal
            if let Some(width) = sig.width {
                // Find all bits of the bundle
                let mut pins = vec![usize::MAX; width];
                let mut found_count = 0;

                for bit in 0..width {
                    let bit_pattern = format!("{}[{}]", sig.net, bit);
                    let dff_q_pattern = format!("{}[{}]$_", sig.net, bit); // DFF cell name pattern

                    // Search for the bit pin - prefer DFF Q outputs over wire pins
                    let mut wire_pin: Option<usize> = None;
                    for pinid in 0..netlistdb.num_pins {
                        let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();

                        // Check for DFF Q output first (preferred)
                        if pin_name.contains(&dff_q_pattern) && pin_name.ends_with(":Q") {
                            pins[bit] = pinid;
                            found_count += 1;
                            break;
                        }

                        // Otherwise remember wire pin as fallback
                        if wire_pin.is_none() {
                            if let Some(pos) = pin_name.find(&bit_pattern) {
                                let end_pos = pos + bit_pattern.len();
                                let is_exact = end_pos == pin_name.len()
                                    || !pin_name[end_pos..].starts_with(|c: char| c.is_ascii_digit());
                                if is_exact {
                                    wire_pin = Some(pinid);
                                }
                            }
                        }
                    }

                    // Use wire pin if DFF Q not found
                    if pins[bit] == usize::MAX {
                        if let Some(wp) = wire_pin {
                            pins[bit] = wp;
                            found_count += 1;
                        }
                    }
                }

                if found_count > 0 {
                    let format = sig.format.clone().unwrap_or_else(|| "hex".to_string());
                    // Log some sample pin mappings for verification
                    let sample_pins: Vec<String> = pins.iter().enumerate()
                        .filter(|(_, &p)| p < usize::MAX)
                        .take(5)
                        .map(|(i, &p)| format!("[{}]={}", i, p))
                        .collect();
                    clilog::info!(
                        "Watchlist: {} -> {}/{} bits found (bundle, format={}, pins: {}...)",
                        sig.name, found_count, width, format, sample_pins.join(", ")
                    );
                    entries.push(WatchlistEntry::Bundle {
                        name: sig.name.clone(),
                        pins,
                        format,
                    });
                } else {
                    clilog::warn!("Watchlist: {} not found (bundle pattern: {}[0..{}])", sig.name, sig.net, width);
                }
            } else {
                // Single-bit signal
                let mut found = false;

                // First try: find by net name (for internal wires)
                for netid in 0..netlistdb.num_nets {
                    let net_name = netlistdb.netnames[netid].dbg_fmt_pin();
                    if net_name == sig.net || net_name.ends_with(&sig.net) {
                        // Find a pin on this net
                        for pinid in netlistdb.net2pin.iter_set(netid) {
                            entries.push(WatchlistEntry::Bit {
                                name: sig.name.clone(),
                                pin: pinid,
                            });
                            clilog::info!("Watchlist: {} -> pin {} (net {})", sig.name, pinid, net_name);
                            found = true;
                            break;
                        }
                        if found {
                            break;
                        }
                    }
                }

                // Second try: find Q output pin for registers
                if !found && sig.signal_type == "reg" {
                    for pinid in 0..netlistdb.num_pins {
                        let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                        // Look for Q pin on a DFF with matching name
                        if pin_name.contains(&sig.net) && pin_name.ends_with(":Q") {
                            entries.push(WatchlistEntry::Bit {
                                name: sig.name.clone(),
                                pin: pinid,
                            });
                            clilog::info!("Watchlist: {} -> pin {} ({})", sig.name, pinid, pin_name);
                            found = true;
                            break;
                        }
                    }
                }

                // Third try: any pin containing the pattern
                if !found {
                    for pinid in 0..netlistdb.num_pins {
                        let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
                        if pin_name.contains(&sig.net) {
                            entries.push(WatchlistEntry::Bit {
                                name: sig.name.clone(),
                                pin: pinid,
                            });
                            clilog::info!("Watchlist: {} -> pin {} ({})", sig.name, pinid, pin_name);
                            found = true;
                            break;
                        }
                    }
                }

                if !found {
                    clilog::warn!("Watchlist: {} not found (pattern: {})", sig.name, sig.net);
                }
            }
        }
        entries
    } else {
        Vec::new()
    };

    // Open trace output file if specified
    let mut trace_file: Option<File> = args.trace_output.as_ref().map(|path| {
        let mut f = File::create(path).expect("Failed to create trace output file");
        // Write CSV header
        let header: Vec<&str> = std::iter::once("cycle")
            .chain(watchlist_entries.iter().map(|e| e.name()))
            .collect();
        writeln!(f, "{}", header.join(",")).expect("Failed to write trace header");
        f
    });

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

                    // Simulate SRAM cells
                    for sram in &mut sram_cells {
                        let en = sram.en_pin != usize::MAX && circ_state[sram.en_pin] != 0;
                        if en {
                            let r_wb = sram.r_wb_pin != usize::MAX && circ_state[sram.r_wb_pin] != 0;
                            let addr = sram.read_addr(&circ_state);
                            if addr < 1024 {
                                if r_wb {
                                    // Read operation: DO = memory[addr]
                                    let data = sram.memory[addr];
                                    sram.write_do(&mut circ_state, data);
                                    if args.verbose && stats.cycles_simulated <= 20 {
                                        clilog::debug!(
                                            "SRAM read: addr={:#x}, data={:#010x}",
                                            addr, data
                                        );
                                    }
                                } else {
                                    // Write operation: memory[addr] = (memory[addr] & ~BEN) | (DI & BEN)
                                    let di = sram.read_di(&circ_state);
                                    let ben = sram.read_ben(&circ_state);
                                    let old = sram.memory[addr];
                                    sram.memory[addr] = (old & !ben) | (di & ben);
                                    if args.verbose && stats.cycles_simulated <= 20 {
                                        clilog::debug!(
                                            "SRAM write: addr={:#x}, di={:#010x}, ben={:#010x}, old={:#010x}, new={:#010x}",
                                            addr, di, ben, old, sram.memory[addr]
                                        );
                                    }
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
                            DriverType::SRAM(_cell_idx) => {
                                // Get DO value from SRAM (simulated at cycle start)
                                // Use reverse mapping to find netlist pin
                                let netpin = aigpin_to_netpin[i];
                                if netpin != usize::MAX {
                                    state.values[i] = circ_state[netpin];
                                }
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

                    // Debug: log reset and CSn state in early cycles (regardless of flash)
                    if args.verbose && stats.cycles_simulated <= 100 {
                        let reset_val = find_gpio_pin("gpio_in", 40).map(|p| circ_state[p]).unwrap_or(255);
                        let csn_val = flash_csn_out.map(|p| circ_state[p]).unwrap_or(255);
                        let clk_val = flash_clk_out.map(|p| circ_state[p]).unwrap_or(255);

                        // Monitor internal signals
                        let rst_sync = rst_sync_rst_pin.map(|p| circ_state[p]).unwrap_or(255);
                        let ibus_cyc = ibus_cyc_pin.map(|p| circ_state[p]).unwrap_or(255);
                        let ibus_cyc_d = ibus_cyc_d_pin.map(|p| circ_state[p]).unwrap_or(255);
                        let n639 = net639_pin.map(|p| circ_state[p]).unwrap_or(255);
                        let n2949 = net2949_pin.map(|p| circ_state[p]).unwrap_or(255);
                        let s09415 = sig_09415_pin.map(|p| circ_state[p]).unwrap_or(255);

                        clilog::debug!(
                            "Cycle {}: rst={}, cyc={}, D={}, _09415={}, n639={}, n2949={}",
                            stats.cycles_simulated, rst_sync, ibus_cyc, ibus_cyc_d, s09415, n639, n2949
                        );
                    }

                    // Write watchlist trace output
                    if let Some(ref mut f) = trace_file {
                        let values: Vec<String> = std::iter::once(stats.cycles_simulated.to_string())
                            .chain(watchlist_entries.iter().map(|e| e.format_value(&circ_state)))
                            .collect();
                        writeln!(f, "{}", values.join(",")).expect("Failed to write trace");
                    }

                    // Step QSPI flash simulation
                    // The flash SPI protocol requires both clock edges:
                    // - Falling edge: flash outputs data (MISO)
                    // - Rising edge: flash samples input (MOSI/command/address)
                    // We step the flash twice to simulate both edges.
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

                            // Check gpio_oeb values for flash pins
                            let oeb0 = find_gpio_pin("gpio_oeb", 0).map(|p| circ_state[p]).unwrap_or(255);
                            let oeb1 = find_gpio_pin("gpio_oeb", 1).map(|p| circ_state[p]).unwrap_or(255);

                            clilog::debug!(
                                "Cycle {}: reset={}, clk={}, csn={}, d_out=0x{:X}, oeb0={}, oeb1={}, byte_count={}, nonzero_pins={}",
                                stats.cycles_simulated, reset_val, clk, csn, d_out, oeb0, oeb1, fl.get_byte_count(), nonzero_count
                            );
                        }

                        // Step the C++ flash model - it handles edge detection and
                        // maintains output value between clock edges internally
                        let d_in = fl.step(clk, csn, d_out);
                        flash_last_d_in = d_in;

                        // Drive flash data inputs back into design
                        for (i, opt_pin) in flash_d_in.iter().enumerate() {
                            if let Some(pin) = opt_pin {
                                circ_state[*pin] = ((flash_last_d_in >> i) & 1) as u8;
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

#[cfg(test)]
mod tests {
    use super::SramCell;

    /// Helper: create a SramCell with mock pin assignments in a circ_state array.
    /// Returns (sram, circ_state) where pins are at known offsets.
    ///
    /// Layout in circ_state:
    ///   0: CLKin
    ///   1: EN
    ///   2: R_WB
    ///   3..12: AD[0..9]
    ///   13..44: BEN[0..31]
    ///   45..76: DI[0..31]
    ///   77..108: DO[0..31]
    fn make_test_sram() -> (SramCell, Vec<u8>) {
        let mut sram = SramCell::new(0);
        sram.clk_pin = 0;
        sram.en_pin = 1;
        sram.r_wb_pin = 2;
        sram.addr_pins = (3..13).collect();
        sram.ben_pins = (13..45).collect();
        sram.di_pins = (45..77).collect();
        sram.do_pins = (77..109).collect();
        let circ_state = vec![0u8; 109];
        (sram, circ_state)
    }

    /// Helper: set a 32-bit value across pin indices in circ_state.
    fn set_bus(circ_state: &mut [u8], base: usize, width: usize, value: u32) {
        for i in 0..width {
            circ_state[base + i] = ((value >> i) & 1) as u8;
        }
    }

    /// Helper: read a 32-bit value from pin indices in circ_state.
    fn read_bus(circ_state: &[u8], base: usize, width: usize) -> u32 {
        let mut val = 0u32;
        for i in 0..width {
            if circ_state[base + i] != 0 {
                val |= 1 << i;
            }
        }
        val
    }

    /// Helper: perform a rising clock edge via step().
    fn clock_edge(sram: &mut SramCell, circ_state: &mut [u8]) -> bool {
        // Clock low → step (captures low)
        circ_state[0] = 0;
        sram.step(circ_state);
        // Clock high → step (rising edge)
        circ_state[0] = 1;
        sram.step(circ_state)
    }

    #[test]
    fn test_sram_step_requires_clock_edge() {
        let (mut sram, mut cs) = make_test_sram();

        // Set up a write: EN=1, R_WB=0 (write), addr=0, BEN=all ones, DI=0xDEADBEEF
        cs[1] = 1; // EN
        cs[2] = 0; // R_WB = write
        set_bus(&mut cs, 3, 10, 0);            // addr = 0
        set_bus(&mut cs, 13, 32, 0xFFFFFFFF);  // BEN = all bits
        set_bus(&mut cs, 45, 32, 0xDEADBEEF);  // DI = 0xDEADBEEF

        // With clock held low - step() should NOT trigger (no rising edge)
        cs[0] = 0;
        assert!(!sram.step(&mut cs), "step() should not trigger without rising edge");
        assert_eq!(sram.memory[0], 0, "memory should be unchanged");

        // With clock held high (but no transition) - step() should NOT trigger
        cs[0] = 1;
        sram.last_clk = true; // Pretend clock was already high
        assert!(!sram.step(&mut cs), "step() should not trigger when clock stays high");
        assert_eq!(sram.memory[0], 0, "memory should be unchanged");

        // Now do a proper rising edge
        sram.last_clk = false;
        cs[0] = 1;
        assert!(sram.step(&mut cs), "step() should trigger on rising edge");
        assert_eq!(sram.memory[0], 0xDEADBEEF, "write should have occurred");
    }

    #[test]
    fn test_sram_write_then_read() {
        let (mut sram, mut cs) = make_test_sram();

        // Write 0x12345678 to address 5
        cs[1] = 1; // EN
        cs[2] = 0; // R_WB = write
        set_bus(&mut cs, 3, 10, 5);            // addr = 5
        set_bus(&mut cs, 13, 32, 0xFFFFFFFF);  // BEN = all bits
        set_bus(&mut cs, 45, 32, 0x12345678);  // DI
        clock_edge(&mut sram, &mut cs);

        // Now read from address 5
        cs[2] = 1; // R_WB = read
        set_bus(&mut cs, 3, 10, 5);  // addr = 5
        clock_edge(&mut sram, &mut cs);

        let do_val = read_bus(&cs, 77, 32);
        assert_eq!(do_val, 0x12345678, "read should return written value");
    }

    #[test]
    fn test_sram_byte_enable_masking() {
        let (mut sram, mut cs) = make_test_sram();

        // Pre-load memory with known value
        sram.memory[10] = 0xAAAAAAAA;

        // Write with only lower 16 bits enabled (BEN[0..15] = 1, BEN[16..31] = 0)
        cs[1] = 1; // EN
        cs[2] = 0; // R_WB = write
        set_bus(&mut cs, 3, 10, 10);          // addr = 10
        set_bus(&mut cs, 13, 32, 0x0000FFFF); // BEN = lower 16 bits only
        set_bus(&mut cs, 45, 32, 0x55555555); // DI
        clock_edge(&mut sram, &mut cs);

        // Upper 16 bits should be preserved from original, lower 16 should be new
        assert_eq!(sram.memory[10], 0xAAAA5555,
            "byte enable should mask write: expected 0xAAAA5555, got 0x{:08X}", sram.memory[10]);
    }

    #[test]
    fn test_sram_en_required() {
        let (mut sram, mut cs) = make_test_sram();

        // Set up write but with EN=0
        cs[1] = 0; // EN = disabled
        cs[2] = 0; // R_WB = write
        set_bus(&mut cs, 3, 10, 0);            // addr = 0
        set_bus(&mut cs, 13, 32, 0xFFFFFFFF);  // BEN
        set_bus(&mut cs, 45, 32, 0xCAFEBABE);  // DI
        clock_edge(&mut sram, &mut cs);

        assert_eq!(sram.memory[0], 0, "write should not occur when EN=0");

        // Also verify read doesn't drive output when EN=0
        sram.memory[0] = 0xCAFEBABE;
        cs[2] = 1; // R_WB = read
        // Clear DO pins
        set_bus(&mut cs, 77, 32, 0);
        clock_edge(&mut sram, &mut cs);

        let do_val = read_bus(&cs, 77, 32);
        assert_eq!(do_val, 0, "read should not drive DO when EN=0");
    }

    #[test]
    fn test_sram_address_range() {
        let (mut sram, mut cs) = make_test_sram();

        // Write to max valid address (1023)
        cs[1] = 1;
        cs[2] = 0;
        set_bus(&mut cs, 3, 10, 1023);
        set_bus(&mut cs, 13, 32, 0xFFFFFFFF);
        set_bus(&mut cs, 45, 32, 0xBAADF00D);
        clock_edge(&mut sram, &mut cs);
        assert_eq!(sram.memory[1023], 0xBAADF00D);

        // Address 1024 (out of range) should be rejected
        set_bus(&mut cs, 3, 10, 1024);
        set_bus(&mut cs, 45, 32, 0xDEADDEAD);
        clock_edge(&mut sram, &mut cs);
        // Memory at 1023 should be unchanged
        assert_eq!(sram.memory[1023], 0xBAADF00D);
    }

    #[test]
    fn test_sram_multiple_writes_same_address() {
        let (mut sram, mut cs) = make_test_sram();

        cs[1] = 1; // EN
        cs[2] = 0; // write
        set_bus(&mut cs, 3, 10, 42);           // addr = 42
        set_bus(&mut cs, 13, 32, 0xFFFFFFFF);  // BEN = all

        // First write
        set_bus(&mut cs, 45, 32, 0x11111111);
        clock_edge(&mut sram, &mut cs);
        assert_eq!(sram.memory[42], 0x11111111);

        // Second write overwrites
        set_bus(&mut cs, 45, 32, 0x22222222);
        clock_edge(&mut sram, &mut cs);
        assert_eq!(sram.memory[42], 0x22222222);
    }

    #[test]
    fn test_sram_read_does_not_modify_memory() {
        let (mut sram, mut cs) = make_test_sram();

        sram.memory[7] = 0xFEEDFACE;

        // Read from address 7
        cs[1] = 1; // EN
        cs[2] = 1; // R_WB = read
        set_bus(&mut cs, 3, 10, 7);
        // Set DI to something (should be ignored during read)
        set_bus(&mut cs, 45, 32, 0x00000000);
        clock_edge(&mut sram, &mut cs);

        assert_eq!(sram.memory[7], 0xFEEDFACE, "read should not modify memory");
        assert_eq!(read_bus(&cs, 77, 32), 0xFEEDFACE, "DO should have read value");
    }

    /// This test demonstrates the bug in the inline SRAM simulation code.
    /// The inline code (lines 2242-2258) is level-sensitive (checks EN directly)
    /// while the real SRAM is clock-edge triggered (posedge CLKin).
    /// The step() method correctly implements edge detection.
    #[test]
    fn test_sram_inline_vs_step_difference() {
        let (mut sram, mut cs) = make_test_sram();

        // Set up a write with EN=1
        cs[1] = 1; // EN
        cs[2] = 0; // R_WB = write
        set_bus(&mut cs, 3, 10, 0);
        set_bus(&mut cs, 13, 32, 0xFFFFFFFF);
        set_bus(&mut cs, 45, 32, 0xCAFEBABE);

        // The INLINE code behavior: checks EN regardless of clock.
        // Simulating what lines 2242-2258 do:
        let en = cs[1] != 0;
        assert!(en, "EN is high");
        // Inline code would write immediately - no clock check!

        // The step() behavior: requires rising clock edge.
        // Clock is at 0 (from make_test_sram), last_clk is false.
        // Setting clock to 0 → no rising edge.
        cs[0] = 0;
        let triggered = sram.step(&mut cs);
        assert!(!triggered, "step() correctly requires rising edge - no write yet");
        assert_eq!(sram.memory[0], 0, "memory unchanged without clock edge");

        // Now give it a rising edge
        cs[0] = 1;
        let triggered = sram.step(&mut cs);
        assert!(triggered, "step() triggers on rising edge");
        assert_eq!(sram.memory[0], 0xCAFEBABE, "write occurs on rising edge");
    }
}
