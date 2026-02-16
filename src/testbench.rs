// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Shared testbench infrastructure for timing and GPU simulation.
//!
//! Contains testbench configuration, SPI flash model (FFI), UART decoder,
//! and watchlist types used by both `timing_sim_cpu` and `gpu_sim`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

// ── FFI bindings to C++ SPI flash model (from chipflow-lib) ──────────────────

pub mod spiflash_ffi {
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
        pub fn spiflash_set_verbose(flash: *mut SpiFlashModel, verbose: c_int);
    }
}

// ── Safe wrapper around the C++ SPI flash model ──────────────────────────────

/// Safe wrapper around the C++ SPI flash model.
pub struct CppSpiFlash {
    ptr: *mut spiflash_ffi::SpiFlashModel,
}

impl CppSpiFlash {
    pub fn new(size_bytes: usize) -> Self {
        let ptr = unsafe { spiflash_ffi::spiflash_new(size_bytes) };
        assert!(!ptr.is_null(), "Failed to create SPI flash model");
        Self { ptr }
    }

    pub fn load_firmware(&mut self, path: &Path, offset: usize) -> std::io::Result<usize> {
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

    pub fn step(&mut self, clk: bool, csn: bool, d_o: u8) -> u8 {
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
    pub fn get_command(&self) -> u8 {
        unsafe { spiflash_ffi::spiflash_get_command(self.ptr) }
    }

    #[allow(dead_code)]
    pub fn get_byte_count(&self) -> u32 {
        unsafe { spiflash_ffi::spiflash_get_byte_count(self.ptr) }
    }

    #[allow(dead_code)]
    pub fn get_step_count(&self) -> u32 {
        unsafe { spiflash_ffi::spiflash_get_step_count(self.ptr) }
    }

    #[allow(dead_code)]
    pub fn get_posedge_count(&self) -> u32 {
        unsafe { spiflash_ffi::spiflash_get_posedge_count(self.ptr) }
    }

    #[allow(dead_code)]
    pub fn get_negedge_count(&self) -> u32 {
        unsafe { spiflash_ffi::spiflash_get_negedge_count(self.ptr) }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        unsafe { spiflash_ffi::spiflash_set_verbose(self.ptr, if verbose { 1 } else { 0 }) }
    }
}

impl Drop for CppSpiFlash {
    fn drop(&mut self) {
        unsafe { spiflash_ffi::spiflash_free(self.ptr) };
    }
}

// Make CppSpiFlash safe to send between threads (the C++ model has no global state)
unsafe impl Send for CppSpiFlash {}

// ── Peripheral Monitor Types (GPU↔CPU callback interface) ────────────────────

/// Describes a GPIO output to watch for edges on the GPU side.
/// Used to build the MonitorConfig array passed to the `state_prep_monitored` kernel.
pub struct PeripheralMonitor {
    pub name: String,
    /// State bit positions to watch for edges.
    pub watch_positions: Vec<u32>,
    /// Edge type per position: 0 = any, 1 = rising, 2 = falling.
    pub edge_type: u32,
}

/// GPU-side monitor descriptor (must match Metal `MonitorConfig` struct).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MonitorConfig {
    pub position: u32,   // bit position in state buffer to monitor
    pub edge_type: u32,  // 0 = any edge, 1 = rising, 2 = falling
}

/// GPU↔CPU peripheral callback control block (must match Metal `PeripheralControl` struct).
/// Lives in shared memory; GPU writes callback request, CPU reads and responds.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PeripheralControl {
    // GPU → CPU
    pub needs_callback: u32,
    pub monitor_id: u32,
    pub tick_number: u32,
    pub _pad: u32,
    // GPU → CPU: output snapshot
    pub flash_clk: u32,
    pub flash_csn: u32,
    pub flash_d_out: u32,
    // CPU → GPU: response
    pub flash_d_in: u32,
    // Tracked previous output values for edge detection (max 16 monitors)
    pub prev_values: [u32; 16],
    // Autonomous mode fields
    pub autonomous_break_tick: u32,       // tick offset where monitor first fired (0 = none, 1-indexed)
    pub autonomous_ticks_completed: u32,  // counter incremented per autonomous tick
}

impl Default for PeripheralControl {
    fn default() -> Self {
        Self {
            needs_callback: 0,
            monitor_id: 0,
            tick_number: 0,
            _pad: 0,
            flash_clk: 0,
            flash_csn: 0,
            flash_d_out: 0,
            flash_d_in: 0,
            prev_values: [0; 16],
            autonomous_break_tick: 0,
            autonomous_ticks_completed: 0,
        }
    }
}

// ── Testbench configuration (loaded from JSON) ──────────────────────────────

/// Testbench configuration loaded from JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct TestbenchConfig {
    pub netlist_path: Option<String>,
    pub liberty_path: Option<String>,
    pub clock_gpio: usize,
    pub reset_gpio: usize,
    pub reset_active_high: bool,
    pub reset_cycles: usize,
    pub num_cycles: usize,
    /// Clock period in picoseconds (e.g. 40000 for 25MHz). Used for UART baud rate calculation.
    pub clock_period_ps: Option<u64>,
    pub flash: Option<FlashConfig>,
    pub uart: Option<UartConfig>,
    #[serde(default)]
    pub gpios: Vec<GpioConfig>,
    pub sram_init: Option<SramInitConfig>,
    pub output_events: Option<String>,
    pub events_reference: Option<String>,
    /// Timing simulation configuration for post-layout SDF back-annotation.
    pub timing: Option<TimingSimConfig>,
    /// Maps GPIO indices to actual port names in the netlist.
    /// Required for designs that don't use gpio_in[N]/gpio_out[N] naming
    /// (e.g. ChipFlow designs with io$signal$i / io$signal$o naming).
    pub port_mapping: Option<PortMapping>,
    /// Constant input values: GPIO index → value (0 or 1).
    /// These are driven every tick on both clock edges.
    #[serde(default)]
    pub constant_inputs: HashMap<String, u8>,
}

/// Configuration for post-layout timing simulation with SDF back-annotation.
#[derive(Debug, Clone, Deserialize)]
pub struct TimingSimConfig {
    /// Path to SDF file from OpenLane2 (e.g. 6_final.sdf).
    pub sdf_file: String,
    /// SDF corner selection: "min", "typ", or "max".
    #[serde(default = "default_sdf_corner")]
    pub sdf_corner: String,
    /// Clock period in picoseconds for timing checks.
    pub clock_period_ps: u64,
    /// Maximum number of timing violations before stopping simulation.
    pub max_violations_before_stop: Option<usize>,
}

fn default_sdf_corner() -> String {
    "typ".to_string()
}

/// Maps GPIO indices to netlist port names for designs with non-standard naming.
///
/// Keys are GPIO index strings ("0", "1", etc.), values are port names
/// as they appear in the netlist (matching `dbg_fmt_pin()` output).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PortMapping {
    /// GPIO index → input port name (external signals entering the design)
    #[serde(default)]
    pub inputs: HashMap<String, String>,
    /// GPIO index → output port name (signals driven by the design)
    #[serde(default)]
    pub outputs: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FlashConfig {
    pub clk_gpio: usize,
    pub csn_gpio: usize,
    pub d0_gpio: usize,
    pub firmware: String,
    pub firmware_offset: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UartConfig {
    pub tx_gpio: usize,
    pub rx_gpio: usize,
    pub baud_rate: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GpioConfig {
    pub name: String,
    pub pins: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SramInitConfig {
    pub elf_path: String,
}

// ── UART TX decoder ─────────────────────────────────────────────────────────

/// UART TX decoder state machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UartState {
    Idle,
    StartBit { start_cycle: usize },
    DataBits { start_cycle: usize, bits_received: u8, value: u8 },
    StopBit { start_cycle: usize, value: u8 },
}

/// Decoded UART event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UartEvent {
    pub timestamp: usize,
    pub peripheral: String,
    pub event: String,
    pub payload: u8,
}

/// UART TX monitor that decodes serial output into bytes.
pub struct UartMonitor {
    pub state: UartState,
    pub last_tx: u8,
    pub cycles_per_bit: usize,
    pub events: Vec<UartEvent>,
}

impl UartMonitor {
    /// Create a new UART monitor.
    /// `clock_hz` is the system clock frequency, `baud_rate` is the UART baud rate.
    pub fn new(clock_hz: u64, baud_rate: u32) -> Self {
        Self {
            state: UartState::Idle,
            last_tx: 1,
            cycles_per_bit: (clock_hz / baud_rate as u64) as usize,
            events: Vec::new(),
        }
    }

    /// Step the UART monitor with the current TX pin value and cycle number.
    /// Returns Some(byte) if a complete byte was received.
    pub fn step(&mut self, tx: u8, cycle: usize) -> Option<u8> {
        let mut received = None;
        self.state = match self.state {
            UartState::Idle => {
                if self.last_tx == 1 && tx == 0 {
                    UartState::StartBit { start_cycle: cycle }
                } else {
                    UartState::Idle
                }
            }
            UartState::StartBit { start_cycle } => {
                if cycle >= start_cycle + self.cycles_per_bit / 2 {
                    if tx == 0 {
                        UartState::DataBits {
                            start_cycle: start_cycle + self.cycles_per_bit,
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
                let bit_center = start_cycle + (bits_received as usize) * self.cycles_per_bit + self.cycles_per_bit / 2;
                if cycle >= bit_center {
                    let new_value = value | ((tx as u8) << bits_received);
                    if bits_received >= 7 {
                        UartState::StopBit {
                            start_cycle: start_cycle + 8 * self.cycles_per_bit,
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
                if cycle >= start_cycle + self.cycles_per_bit / 2 {
                    if tx == 1 {
                        self.events.push(UartEvent {
                            timestamp: cycle,
                            peripheral: "uart_0".to_string(),
                            event: "tx".to_string(),
                            payload: value,
                        });
                        let ch = if value >= 32 && value < 127 { value as char } else { '.' };
                        clilog::info!("UART TX @ cycle {}: 0x{:02X} '{}'", cycle, value, ch);
                        received = Some(value);
                    }
                    UartState::Idle
                } else {
                    UartState::StopBit { start_cycle, value }
                }
            }
        };
        self.last_tx = tx;
        received
    }
}

// ── Watchlist types ─────────────────────────────────────────────────────────

/// Watchlist signal entry (from JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct WatchlistSignal {
    /// Display name for the signal.
    pub name: String,
    /// Net name pattern to match in the netlist.
    pub net: String,
    /// Signal type (reg, comb, mem).
    #[serde(rename = "type", default)]
    pub signal_type: String,
    /// Width for bundle signals (e.g., 32 for a 32-bit bus).
    #[serde(default)]
    pub width: Option<usize>,
    /// Format for output: "bin", "hex", or "dec".
    #[serde(default)]
    pub format: Option<String>,
}

/// Watchlist configuration loaded from JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct Watchlist {
    pub signals: Vec<WatchlistSignal>,
}

/// Resolved watchlist entry - either single bit or bundle.
#[derive(Debug, Clone)]
pub enum WatchlistEntry {
    /// Single-bit signal.
    Bit { name: String, pin: usize },
    /// Multi-bit bundle (pins ordered LSB to MSB).
    Bundle { name: String, pins: Vec<usize>, format: String },
}

impl WatchlistEntry {
    pub fn name(&self) -> &str {
        match self {
            WatchlistEntry::Bit { name, .. } => name,
            WatchlistEntry::Bundle { name, .. } => name,
        }
    }

    pub fn format_value(&self, circ_state: &[u8]) -> String {
        match self {
            WatchlistEntry::Bit { pin, .. } => circ_state[*pin].to_string(),
            WatchlistEntry::Bundle { pins, format, .. } => {
                let mut value: u64 = 0;
                for (i, &pin) in pins.iter().enumerate() {
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
