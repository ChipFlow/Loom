// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Hybrid GPU/CPU co-simulation using Metal for gate evaluation
//! and CPU for peripheral models (SPI flash, UART).
//!
//! The GPU evaluates all combinational + sequential logic per clock tick,
//! while the CPU steps peripheral models (SPI flash, UART) between ticks,
//! feeding GPIO values back via shared memory.
//!
//! Usage:
//!   cargo run -r --features metal --bin gpu_sim -- <netlist.gv> <gemparts> \
//!     --config <testbench.json> [--max-cycles N] [--num-blocks N]

use gem::aig::{DriverType, AIG};
use gem::aigpdk::AIGPDKLeafPins;
use gem::flatten::FlattenedScriptV1;
use gem::staging::build_staged_aigs;
use gem::pe::Partition;
use gem::testbench::{CppSpiFlash, TestbenchConfig, UartMonitor};
use netlistdb::{Direction, GeneralPinName, NetlistDB};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use ulib::{AsUPtr, AsUPtrMut, Device, UVec};

use metal::{Device as MTLDevice, MTLSize, ComputePipelineState, CommandQueue, MTLResourceOptions};

// ── CLI Arguments ────────────────────────────────────────────────────────────

#[derive(clap::Parser, Debug)]
#[command(name = "gpu_sim")]
#[command(about = "Hybrid GPU/CPU co-simulation with Metal")]
struct Args {
    /// Gate-level verilog path synthesized in AIGPDK library.
    netlist_verilog: PathBuf,

    /// Pre-compiled partition mapping (.gemparts file).
    gemparts: PathBuf,

    /// Testbench configuration JSON file.
    #[clap(long)]
    config: PathBuf,

    /// Top module type in netlist.
    #[clap(long)]
    top_module: Option<String>,

    /// Level split thresholds (comma-separated).
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,

    /// Number of GPU threadgroups (blocks). Should be ~2x GPU SM count.
    #[clap(long, default_value = "64")]
    num_blocks: usize,

    /// Maximum system clock ticks to simulate.
    #[clap(long)]
    max_cycles: Option<usize>,

    /// Enable verbose flash model debug output.
    #[clap(long)]
    flash_verbose: bool,

    /// Clock period in picoseconds (default: 1000 = 1ns, for UART baud calc).
    #[clap(long, default_value = "1000")]
    clock_period: u64,

    /// Verify GPU results against CPU baseline.
    #[clap(long)]
    check_with_cpu: bool,
}

// ── Simulation Parameters (must match Metal shader) ──────────────────────────

#[repr(C)]
struct SimParams {
    num_blocks: u64,
    num_major_stages: u64,
    num_cycles: u64,
    state_size: u64,
    current_cycle: u64,
    current_stage: u64,
}

// ── Metal Simulator ──────────────────────────────────────────────────────────

struct MetalSimulator {
    device: metal::Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}

impl MetalSimulator {
    fn new() -> Self {
        let device = MTLDevice::system_default().expect("No Metal device found");
        clilog::info!("Using Metal device: {}", device.name());

        let metallib_path = env!("METALLIB_PATH");
        let library = device
            .new_library_with_file(metallib_path)
            .expect("Failed to load metallib");

        let kernel_function = library
            .get_function("simulate_v1_stage", None)
            .expect("Failed to get kernel function");

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel_function)
            .expect("Failed to create pipeline state");

        let command_queue = device.new_command_queue();

        Self {
            device,
            command_queue,
            pipeline_state,
        }
    }

    /// Dispatch a single stage of the simulation kernel.
    fn dispatch_stage(
        &self,
        num_blocks: usize,
        num_major_stages: usize,
        state_size: usize,
        cycle_i: usize,
        stage_i: usize,
        blocks_start_buffer: &metal::Buffer,
        blocks_data_buffer: &metal::Buffer,
        sram_data_buffer: &metal::Buffer,
        states_buffer: &metal::Buffer,
        event_buffer_metal: &metal::Buffer,
    ) {
        let params = SimParams {
            num_blocks: num_blocks as u64,
            num_major_stages: num_major_stages as u64,
            num_cycles: 1, // hybrid mode: always 1 cycle at a time
            state_size: state_size as u64,
            current_cycle: cycle_i as u64,
            current_stage: stage_i as u64,
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const SimParams as *const _,
            std::mem::size_of::<SimParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_state);
        encoder.set_buffer(0, Some(blocks_start_buffer), 0);
        encoder.set_buffer(1, Some(blocks_data_buffer), 0);
        encoder.set_buffer(2, Some(sram_data_buffer), 0);
        encoder.set_buffer(3, Some(states_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.set_buffer(5, Some(event_buffer_metal), 0);

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(num_blocks as u64, 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

// ── GPIO ↔ State Buffer Mapping ──────────────────────────────────────────────

/// Maps GPIO pin indices to bit positions in the packed u32 state buffer.
struct GpioMapping {
    /// gpio_in[idx] → (aigpin, state bit position)
    input_bits: HashMap<usize, u32>,
    /// gpio_out[idx] → state bit position in output_map
    output_bits: HashMap<usize, u32>,
    /// Posedge clock flag bit positions (one per clock pin)
    posedge_flag_bits: Vec<u32>,
    /// Negedge clock flag bit positions
    negedge_flag_bits: Vec<u32>,
}

/// Read a single bit from a packed u32 state buffer.
#[inline]
fn read_bit(state: &[u32], pos: u32) -> u8 {
    ((state[(pos >> 5) as usize] >> (pos & 31)) & 1) as u8
}

/// Set a single bit in a packed u32 state buffer.
#[inline]
fn set_bit(state: &mut [u32], pos: u32, val: u8) {
    let word = &mut state[(pos >> 5) as usize];
    let mask = 1u32 << (pos & 31);
    if val != 0 {
        *word |= mask;
    } else {
        *word &= !mask;
    }
}

/// Clear a single bit in a packed u32 state buffer.
#[inline]
fn clear_bit(state: &mut [u32], pos: u32) {
    state[(pos >> 5) as usize] &= !(1u32 << (pos & 31));
}

/// Build GPIO-to-state-buffer mapping from AIG + FlattenedScript.
fn build_gpio_mapping(
    aig: &AIG,
    netlistdb: &NetlistDB,
    script: &FlattenedScriptV1,
) -> GpioMapping {
    let mut input_bits: HashMap<usize, u32> = HashMap::new();
    let mut output_bits: HashMap<usize, u32> = HashMap::new();
    let mut posedge_flag_bits: Vec<u32> = Vec::new();
    let mut negedge_flag_bits: Vec<u32> = Vec::new();

    // Map input ports (gpio_in) → state buffer positions
    for (aigpin_idx, driv) in aig.drivers.iter().enumerate() {
        match driv {
            DriverType::InputPort(pinid) => {
                let pin_name = netlistdb.pinnames[*pinid].dbg_fmt_pin();
                // Parse gpio_in[N] from the pin name
                if let Some(gpio_idx) = parse_gpio_index(&pin_name, "gpio_in") {
                    if let Some(&pos) = script.input_map.get(&aigpin_idx) {
                        input_bits.insert(gpio_idx, pos);
                    }
                }
            }
            DriverType::InputClockFlag(pinid, is_negedge) => {
                if let Some(&pos) = script.input_map.get(&aigpin_idx) {
                    if *is_negedge == 0 {
                        posedge_flag_bits.push(pos);
                    } else {
                        negedge_flag_bits.push(pos);
                    }
                    // Also check if this is associated with a GPIO
                    let pin_name = netlistdb.pinnames[*pinid].dbg_fmt_pin();
                    clilog::debug!("ClockFlag aigpin={} pin={} negedge={} pos={}",
                                   aigpin_idx, pin_name, is_negedge, pos);
                }
            }
            _ => {}
        }
    }

    // Map output ports (gpio_out) → state buffer positions
    for i in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[i] == Direction::I {
            let aigpin_iv = aig.pin2aigpin_iv[i];
            if aigpin_iv == usize::MAX || aigpin_iv <= 1 {
                continue;
            }
            let aigpin = aigpin_iv >> 1;
            if let Some(&pos) = script.output_map.get(&aigpin) {
                let pin_name = netlistdb.pinnames[i].dbg_fmt_pin();
                if let Some(gpio_idx) = parse_gpio_index(&pin_name, "gpio_out") {
                    output_bits.insert(gpio_idx, pos);
                }
            }
        }
    }

    clilog::info!("GPIO mapping: {} inputs, {} outputs, {} posedge flags, {} negedge flags",
                  input_bits.len(), output_bits.len(),
                  posedge_flag_bits.len(), negedge_flag_bits.len());

    GpioMapping {
        input_bits,
        output_bits,
        posedge_flag_bits,
        negedge_flag_bits,
    }
}

/// Parse a GPIO index from a pin name like "gpio_in[38]" or "gpio_in:38".
fn parse_gpio_index(pin_name: &str, prefix: &str) -> Option<usize> {
    // Try "gpio_in[N]" format
    if let Some(start) = pin_name.find(&format!("{}[", prefix)) {
        let after = &pin_name[start + prefix.len() + 1..];
        if let Some(end) = after.find(']') {
            return after[..end].parse().ok();
        }
    }
    // Try "gpio_in:N" format
    if let Some(start) = pin_name.find(&format!("{}:", prefix)) {
        let after = &pin_name[start + prefix.len() + 1..];
        let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        return num_str.parse().ok();
    }
    None
}

/// Read a 4-bit nibble from flash data GPIO outputs.
fn read_flash_nibble(state: &[u32], gpio_map: &GpioMapping, d0_gpio: usize) -> u8 {
    let mut val = 0u8;
    for i in 0..4 {
        if let Some(&pos) = gpio_map.output_bits.get(&(d0_gpio + i)) {
            val |= read_bit(state, pos) << i;
        }
    }
    val
}

/// Write flash data input to GPIO state.
fn set_flash_din(state: &mut [u32], gpio_map: &GpioMapping, d0_gpio: usize, din: u8) {
    for i in 0..4 {
        if let Some(&pos) = gpio_map.input_bits.get(&(d0_gpio + i)) {
            set_bit(state, pos, (din >> i) & 1);
        }
    }
}

// ── CPU baseline for verification ────────────────────────────────────────────

/// CPU prototype partition executor (from metal_test.rs).
/// Used for --check-with-cpu verification.
#[allow(dead_code)]
fn simulate_block_v1(
    script: &[u32],
    input_state: &[u32],
    output_state: &mut [u32],
    sram_data: &mut [u32],
) {
    use gem::aigpdk::AIGPDK_SRAM_SIZE;
    let mut script_pi = 0;
    loop {
        let num_stages = script[script_pi];
        let is_last_part = script[script_pi + 1];
        let num_ios = script[script_pi + 2];
        let io_offset = script[script_pi + 3];
        let num_srams = script[script_pi + 4];
        let sram_offset = script[script_pi + 5];
        let num_global_read_rounds = script[script_pi + 6];
        let num_output_duplicates = script[script_pi + 7];
        let mut writeout_hooks = vec![0; 256];
        for i in 0..128 {
            let t = script[script_pi + 128 + i];
            writeout_hooks[i * 2] = (t & ((1 << 16) - 1)) as u16;
            writeout_hooks[i * 2 + 1] = (t >> 16) as u16;
        }
        if num_stages == 0 {
            script_pi += 256;
            break;
        }
        script_pi += 256;
        let mut writeouts = vec![0u32; num_ios as usize];
        let mut state = vec![0u32; 256];

        for _gr_i in 0..num_global_read_rounds {
            for i in 0..256 {
                let mut cur_state = state[i];
                let idx = script[script_pi + (i * 2)];
                let mut mask = script[script_pi + (i * 2 + 1)];
                if mask == 0 { continue; }
                let value = match (idx >> 31) != 0 {
                    false => input_state[idx as usize],
                    true => output_state[(idx ^ (1 << 31)) as usize],
                };
                while mask != 0 {
                    cur_state <<= 1;
                    let lowbit = mask & (-(mask as i32)) as u32;
                    if (value & lowbit) != 0 { cur_state |= 1; }
                    mask ^= lowbit;
                }
                state[i] = cur_state;
            }
            script_pi += 256 * 2;
        }

        for bs_i in 0..num_stages {
            let mut hier_inputs = vec![0; 256];
            let mut hier_flag_xora = vec![0; 256];
            let mut hier_flag_xorb = vec![0; 256];
            let mut hier_flag_orb = vec![0; 256];
            for k_outer in 0..4 {
                for i in 0..256 {
                    for k_inner in 0..4 {
                        let k = k_outer * 4 + k_inner;
                        let t_shuffle = script[script_pi + i * 4 + k_inner];
                        let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u16;
                        let t_shuffle_2_idx = (t_shuffle >> 16) as u16;
                        hier_inputs[i] |= (state[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1) << (k * 2);
                        hier_inputs[i] |= (state[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);
                    }
                }
                script_pi += 256 * 4;
            }
            for i in 0..256 {
                hier_flag_xora[i] = script[script_pi + i * 4];
                hier_flag_xorb[i] = script[script_pi + i * 4 + 1];
                hier_flag_orb[i] = script[script_pi + i * 4 + 2];
            }
            script_pi += 256 * 4;

            for i in 0..128 {
                let a = hier_inputs[i];
                let b = hier_inputs[128 + i];
                let xora = hier_flag_xora[128 + i];
                let xorb = hier_flag_xorb[128 + i];
                let orb = hier_flag_orb[128 + i];
                hier_inputs[128 + i] = (a ^ xora) & ((b ^ xorb) | orb);
            }
            for hi in 1..=7 {
                let hier_width = 1 << (7 - hi);
                for i in 0..hier_width {
                    let a = hier_inputs[hier_width * 2 + i];
                    let b = hier_inputs[hier_width * 3 + i];
                    let xora = hier_flag_xora[hier_width + i];
                    let xorb = hier_flag_xorb[hier_width + i];
                    let orb = hier_flag_orb[hier_width + i];
                    hier_inputs[hier_width + i] = (a ^ xora) & ((b ^ xorb) | orb);
                }
            }
            let v1 = hier_inputs[1];
            let xora = hier_flag_xora[0];
            let xorb = hier_flag_xorb[0];
            let orb = hier_flag_orb[0];
            let r8 = ((v1 << 16) ^ xora) & ((v1 ^ xorb) | orb) & 0xffff0000;
            let r9 = ((r8 >> 8) ^ xora) & (((r8 >> 16) ^ xorb) | orb) & 0xff00;
            let r10 = ((r9 >> 4) ^ xora) & (((r9 >> 8) ^ xorb) | orb) & 0xf0;
            let r11 = ((r10 >> 2) ^ xora) & (((r10 >> 4) ^ xorb) | orb) & 0b1100;
            let r12 = ((r11 >> 1) ^ xora) & (((r11 >> 2) ^ xorb) | orb) & 0b10;
            hier_inputs[0] = r8 | r9 | r10 | r11 | r12;
            state = hier_inputs;

            for i in 0..256 {
                let hooki = writeout_hooks[i];
                if (hooki >> 8) as u32 == bs_i {
                    writeouts[i] = state[(hooki & 255) as usize];
                }
            }
        }

        let mut sram_duplicate_perm = vec![0u32; (num_srams * 4 + num_output_duplicates) as usize];
        for k_outer in 0..4 {
            for i in 0..(num_srams * 4 + num_output_duplicates) {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle = script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u32;
                    let t_shuffle_2_idx = (t_shuffle >> 16) as u32;
                    sram_duplicate_perm[i as usize] |= (writeouts[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1) << (k * 2);
                    sram_duplicate_perm[i as usize] |= (writeouts[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..(num_srams * 4 + num_output_duplicates) as usize {
            sram_duplicate_perm[i] &= !script[script_pi + i * 4 + 1];
            sram_duplicate_perm[i] ^= script[script_pi + i * 4];
        }
        script_pi += 256 * 4;

        for sram_i_u32 in 0..num_srams {
            let sram_i = sram_i_u32 as usize;
            let addrs = sram_duplicate_perm[sram_i * 4];
            let port_r_addr_iv = addrs & 0xffff;
            let port_w_addr_iv = (addrs & 0xffff0000) >> 16;
            let port_w_wr_en = sram_duplicate_perm[sram_i * 4 + 1];
            let port_w_wr_data_iv = sram_duplicate_perm[sram_i * 4 + 2];
            let sram_st = sram_offset as usize + sram_i * AIGPDK_SRAM_SIZE;
            let sram_ed = sram_st + AIGPDK_SRAM_SIZE;
            let ram = &mut sram_data[sram_st..sram_ed];
            let r = ram[port_r_addr_iv as usize];
            let w0 = ram[port_w_addr_iv as usize];
            writeouts[(num_ios - num_srams + sram_i_u32) as usize] = r;
            ram[port_w_addr_iv as usize] = (w0 & !port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
        }

        for i in 0..num_output_duplicates {
            writeouts[(num_ios - num_srams - num_output_duplicates + i) as usize] =
                sram_duplicate_perm[(num_srams * 4 + i) as usize];
        }

        let mut clken_perm = vec![0u32; num_ios as usize];
        let writeouts_for_clken = writeouts.clone();
        for k_outer in 0..4 {
            for i in 0..num_ios {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle = script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u32;
                    let t_shuffle_2_idx = (t_shuffle >> 16) as u32;
                    clken_perm[i as usize] |= (writeouts_for_clken[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1) << (k * 2);
                    clken_perm[i as usize] |= (writeouts_for_clken[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..num_ios as usize {
            clken_perm[i] &= !script[script_pi + i * 4 + 1];
            clken_perm[i] ^= script[script_pi + i * 4];
            writeouts[i] ^= script[script_pi + i * 4 + 2];
        }
        script_pi += 256 * 4;

        for i in 0..num_ios {
            let old_wo = input_state[(io_offset + i) as usize];
            let clken = clken_perm[i as usize];
            let wo = (old_wo & !clken) | (writeouts[i as usize] & clken);
            output_state[(io_offset + i) as usize] = wo;
        }

        if is_last_part != 0 { break; }
    }
    assert_eq!(script_pi, script.len());
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("gpu_sim");
    clilog::enable_timer("gem");
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);

    let args = <Args as clap::Parser>::parse();
    clilog::info!("gpu_sim args:\n{:#?}", args);

    // ── Load testbench config ────────────────────────────────────────────

    let file = File::open(&args.config).expect("Failed to open config file");
    let reader = BufReader::new(file);
    let config: TestbenchConfig =
        serde_json::from_reader(reader).expect("Failed to parse config JSON");
    clilog::info!("Loaded testbench config: {:?}", config);

    let max_ticks = args.max_cycles.unwrap_or(config.num_cycles);

    // ── Load netlist and build AIG ───────────────────────────────────────

    let timer_load = clilog::stimer!("load_netlist");
    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins(),
    )
    .expect("cannot build netlist");

    let aig = AIG::from_netlistdb(&netlistdb);
    clilog::info!(
        "AIG: {} pins, {} DFFs, {} SRAMs",
        aig.num_aigpins,
        aig.dffs.len(),
        aig.srams.len()
    );
    clilog::finish!(timer_load);

    // ── Build staged AIGs and load partitions ────────────────────────────

    let timer_script = clilog::stimer!("build_script");
    let stageds = build_staged_aigs(&aig, &args.level_split);

    let f = std::fs::File::open(&args.gemparts).unwrap();
    let mut buf = std::io::BufReader::new(f);
    let parts_in_stages: Vec<Vec<Partition>> = serde_bare::from_reader(&mut buf).unwrap();
    clilog::info!(
        "Partitions per stage: {:?}",
        parts_in_stages.iter().map(|ps| ps.len()).collect::<Vec<_>>()
    );

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    let script = FlattenedScriptV1::from(
        &aig,
        &stageds.iter().map(|(_, _, staged)| staged).collect::<Vec<_>>(),
        &parts_in_stages.iter().map(|ps| ps.as_slice()).collect::<Vec<_>>(),
        args.num_blocks,
        input_layout,
    );
    clilog::info!(
        "Script: state_size={}, sram_storage={}, blocks={}, stages={}",
        script.reg_io_state_size,
        script.sram_storage_size,
        script.num_blocks,
        script.num_major_stages
    );
    clilog::finish!(timer_script);

    // ── Build GPIO mapping ───────────────────────────────────────────────

    let gpio_map = build_gpio_mapping(&aig, &netlistdb, &script);

    // Verify we found the expected GPIO pins
    let clock_gpio = config.clock_gpio;
    let reset_gpio = config.reset_gpio;
    assert!(
        gpio_map.input_bits.contains_key(&clock_gpio),
        "Clock GPIO {} not found in input mapping. Available: {:?}",
        clock_gpio,
        gpio_map.input_bits.keys().collect::<Vec<_>>()
    );
    assert!(
        gpio_map.input_bits.contains_key(&reset_gpio),
        "Reset GPIO {} not found in input mapping. Available: {:?}",
        reset_gpio,
        gpio_map.input_bits.keys().collect::<Vec<_>>()
    );

    if let Some(ref flash_cfg) = config.flash {
        for i in 0..4 {
            let gpio = flash_cfg.d0_gpio + i;
            if gpio_map.input_bits.contains_key(&gpio) {
                clilog::info!("Flash D{} input GPIO {} -> state pos {}", i, gpio, gpio_map.input_bits[&gpio]);
            }
            if gpio_map.output_bits.contains_key(&gpio) {
                clilog::info!("Flash D{} output GPIO {} -> state pos {}", i, gpio, gpio_map.output_bits[&gpio]);
            }
        }
    }

    // ── Initialize peripheral models ─────────────────────────────────────

    let mut flash: Option<CppSpiFlash> = if let Some(ref flash_cfg) = config.flash {
        let mut fl = CppSpiFlash::new(16 * 1024 * 1024);
        fl.set_verbose(args.flash_verbose);
        let firmware_path = std::path::Path::new(&flash_cfg.firmware);
        match fl.load_firmware(firmware_path, flash_cfg.firmware_offset) {
            Ok(size) => clilog::info!("Loaded {} bytes firmware at offset 0x{:X}", size, flash_cfg.firmware_offset),
            Err(e) => panic!("Failed to load firmware: {}", e),
        }
        Some(fl)
    } else {
        None
    };

    let clock_hz = 1_000_000_000_000u64 / args.clock_period;
    let uart_baud = config.uart.as_ref().map(|u| u.baud_rate).unwrap_or(115200);
    let mut uart_monitor = UartMonitor::new(clock_hz, uart_baud);
    let uart_tx_gpio = config.uart.as_ref().map(|u| u.tx_gpio);

    // ── Initialize Metal simulator and GPU state buffers ─────────────────

    let timer_init = clilog::stimer!("init_gpu");
    let simulator = MetalSimulator::new();
    let device = Device::Metal(0);

    let state_size = script.reg_io_state_size as usize;

    // The Metal kernel expects states laid out as:
    //   [cycle 0 input (state_size)] [cycle 0 output (state_size)] [cycle 1 input] ...
    // For hybrid mode with num_cycles=1, we need exactly 2 * state_size:
    //   [input state] [output state]
    let mut states = UVec::new_zeroed(2 * state_size, device);
    let mut sram_storage: UVec<u32> = UVec::new_zeroed(script.sram_storage_size as usize, device);

    // Initialize: set reset active
    let reset_val = if config.reset_active_high { 1u8 } else { 0u8 };
    set_bit(&mut states[..state_size], gpio_map.input_bits[&reset_gpio], reset_val);
    clilog::info!("Initial state: reset GPIO {} = {} (active)", reset_gpio, reset_val);

    // Set flash D_IN defaults (high = no data)
    if let Some(ref flash_cfg) = config.flash {
        set_flash_din(&mut states[..state_size], &gpio_map, flash_cfg.d0_gpio, 0x0F);
    }

    // Create Metal buffers (shared memory = zero-copy on Apple Silicon)
    let states_ptr = states.as_mut_uptr(device);
    let blocks_start_ptr = script.blocks_start.as_uptr(device);
    let blocks_data_ptr = script.blocks_data.as_uptr(device);
    let sram_ptr = sram_storage.as_mut_uptr(device);

    let blocks_start_buffer = simulator.device.new_buffer_with_bytes_no_copy(
        blocks_start_ptr as *const _,
        (script.blocks_start.len() * std::mem::size_of::<usize>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let blocks_data_buffer = simulator.device.new_buffer_with_bytes_no_copy(
        blocks_data_ptr as *const _,
        (script.blocks_data.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let sram_data_buffer = simulator.device.new_buffer_with_bytes_no_copy(
        sram_ptr as *mut _ as *const _,
        (sram_storage.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let states_buffer = simulator.device.new_buffer_with_bytes_no_copy(
        states_ptr as *mut _ as *const _,
        (states.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );

    // Event buffer (for $stop/$finish/assertions - allocated but mostly unused in hybrid mode)
    let event_buffer = Box::new(gem::event_buffer::EventBuffer::new());
    let event_buffer_ptr = Box::into_raw(event_buffer);
    let event_buffer_metal = simulator.device.new_buffer_with_bytes_no_copy(
        event_buffer_ptr as *const _,
        std::mem::size_of::<gem::event_buffer::EventBuffer>() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );

    clilog::finish!(timer_init);

    // ── Simulation loop ──────────────────────────────────────────────────
    //
    // Each tick = 1 system clock cycle = 2 GPU evaluations (fall + rise).
    //
    // Tick order (matching CXXRTL):
    //   1. Read gpio_out from previous tick's output
    //   2. Step peripheral models (flash, UART)
    //   3. Copy output → input, set falling edge inputs
    //   4. GPU dispatch all stages (falling edge)
    //   5. Copy output → input, set rising edge inputs + posedge flag
    //   6. GPU dispatch all stages (rising edge, DFFs latch)
    //   7. Clear posedge flags in output

    let timer_sim = clilog::stimer!("simulation");
    let sim_start = std::time::Instant::now();
    let reset_cycles = config.reset_cycles;
    let num_major_stages = script.num_major_stages;
    let num_blocks = script.num_blocks;

    // Track previous flash data for setup timing delay (same as timing_sim_cpu)
    let mut prev_flash_d_out: u8 = 0;
    let mut prev_flash_csn: bool = true;

    for tick in 0..max_ticks {
        // Reset event buffer
        unsafe { (*event_buffer_ptr).reset(); }

        let in_reset = tick < reset_cycles;

        // ── 1. Read gpio_out from output state ──────────────────────────
        // Read all values we need into locals before any mutable borrows.
        let flash_clk = config.flash.as_ref().map(|f| {
            gpio_map.output_bits.get(&f.clk_gpio)
                .map(|&pos| read_bit(&states[state_size..2 * state_size], pos) != 0)
                .unwrap_or(false)
        }).unwrap_or(false);

        let current_flash_csn = config.flash.as_ref().map(|f| {
            gpio_map.output_bits.get(&f.csn_gpio)
                .map(|&pos| read_bit(&states[state_size..2 * state_size], pos) != 0)
                .unwrap_or(true)
        }).unwrap_or(true);

        let current_flash_d_out = config.flash.as_ref().map(|f| {
            read_flash_nibble(&states[state_size..2 * state_size], &gpio_map, f.d0_gpio)
        }).unwrap_or(0);

        let uart_tx_val = uart_tx_gpio.and_then(|tx_gpio| {
            gpio_map.output_bits.get(&tx_gpio)
                .map(|&pos| read_bit(&states[state_size..2 * state_size], pos))
        });

        // ── 2. Step peripheral models ───────────────────────────────────

        // Flash: use delayed data/CSN for setup timing (same as timing_sim_cpu)
        if let Some(ref mut fl) = flash {
            let effective_csn = if in_reset { true } else { prev_flash_csn };
            let din = fl.step(flash_clk, effective_csn, prev_flash_d_out);

            // Update GPIO inputs with flash response (only when not in reset)
            if !in_reset {
                if let Some(ref flash_cfg) = config.flash {
                    set_flash_din(&mut states[state_size..2 * state_size], &gpio_map, flash_cfg.d0_gpio, din);
                }
            }

            prev_flash_d_out = current_flash_d_out;
            prev_flash_csn = current_flash_csn;

            if tick <= 5 || tick == reset_cycles || tick == reset_cycles + 1 || tick == reset_cycles + 100 {
                clilog::info!("tick {}: flash clk={}, csn={}, d_out={:04b}, din={:04b}",
                             tick, flash_clk, current_flash_csn, current_flash_d_out, din);
            }
        }

        // UART: read TX from output state
        if let Some(tx) = uart_tx_val {
            uart_monitor.step(tx, tick);
        }

        // ── 3. Copy output → input, set falling edge ───────────────────
        // For the first tick, output state is all zeros (initial).
        // After that, we copy the previous tick's output to become the new input.
        states.copy_within(state_size..2 * state_size, 0);

        // Set reset GPIO
        let reset_val = if in_reset {
            if config.reset_active_high { 1u8 } else { 0u8 }
        } else {
            if config.reset_active_high { 0u8 } else { 1u8 }
        };
        set_bit(&mut states[..state_size], gpio_map.input_bits[&reset_gpio], reset_val);

        // Set clock LOW (falling edge)
        set_bit(&mut states[..state_size], gpio_map.input_bits[&clock_gpio], 0);

        // Clear clock flags for falling edge
        for &pos in &gpio_map.posedge_flag_bits {
            clear_bit(&mut states[..state_size], pos);
        }
        // Set negedge flag if design has negedge-triggered elements
        for &pos in &gpio_map.negedge_flag_bits {
            set_bit(&mut states[..state_size], pos, 1);
        }

        // Note: flash D_IN values were set in the output state (step 2 above).
        // The copy_within propagated them to the input state automatically.

        // ── 4. GPU dispatch: falling edge ───────────────────────────────
        for stage_i in 0..num_major_stages {
            simulator.dispatch_stage(
                num_blocks,
                num_major_stages,
                state_size,
                0, // cycle_i = 0 (single-cycle mode)
                stage_i,
                &blocks_start_buffer,
                &blocks_data_buffer,
                &sram_data_buffer,
                &states_buffer,
                &event_buffer_metal,
            );
        }

        // ── 5. Copy output → input, set rising edge ────────────────────
        states.copy_within(state_size..2 * state_size, 0);

        // Re-set reset GPIO (it was overwritten by copy)
        set_bit(&mut states[..state_size], gpio_map.input_bits[&reset_gpio], reset_val);

        // Set clock HIGH (rising edge)
        set_bit(&mut states[..state_size], gpio_map.input_bits[&clock_gpio], 1);

        // Set posedge flag (DFFs latch on this evaluation)
        for &pos in &gpio_map.posedge_flag_bits {
            set_bit(&mut states[..state_size], pos, 1);
        }
        // Clear negedge flags
        for &pos in &gpio_map.negedge_flag_bits {
            clear_bit(&mut states[..state_size], pos);
        }

        // ── 6. GPU dispatch: rising edge (DFFs latch) ───────────────────
        for stage_i in 0..num_major_stages {
            simulator.dispatch_stage(
                num_blocks,
                num_major_stages,
                state_size,
                0,
                stage_i,
                &blocks_start_buffer,
                &blocks_data_buffer,
                &sram_data_buffer,
                &states_buffer,
                &event_buffer_metal,
            );
        }

        // ── 7. Clear posedge flags in output (ready for next tick) ──────
        for &pos in &gpio_map.posedge_flag_bits {
            clear_bit(&mut states[state_size..], pos);
        }

        // Step flash again after rising edge (sees SPI clock HIGH)
        if let Some(ref mut fl) = flash {
            // Re-read output after rising edge evaluation
            let out2 = &states[state_size..2 * state_size];
            let flash_clk2 = config.flash.as_ref().map(|f| {
                gpio_map.output_bits.get(&f.clk_gpio).map(|&pos| read_bit(out2, pos) != 0).unwrap_or(false)
            }).unwrap_or(false);

            let effective_csn = if in_reset { true } else { prev_flash_csn };
            let din = fl.step(flash_clk2, effective_csn, prev_flash_d_out);

            // Update flash D_IN in output state for next tick
            if !in_reset {
                if let Some(ref flash_cfg) = config.flash {
                    set_flash_din(&mut states[state_size..2 * state_size], &gpio_map, flash_cfg.d0_gpio, din);
                }
            }

            // Update delayed state
            let out2 = &states[state_size..2 * state_size];
            prev_flash_d_out = config.flash.as_ref().map(|f| {
                read_flash_nibble(out2, &gpio_map, f.d0_gpio)
            }).unwrap_or(0);
            prev_flash_csn = config.flash.as_ref().map(|f| {
                gpio_map.output_bits.get(&f.csn_gpio).map(|&pos| read_bit(out2, pos) != 0).unwrap_or(true)
            }).unwrap_or(true);
        }

        // Progress logging
        if tick > 0 && tick % 100000 == 0 {
            clilog::info!("Tick {} / {}", tick, max_ticks);
        }

        // Log transition out of reset
        if tick == reset_cycles {
            clilog::info!("Reset released at tick {}", tick);
        }
    }

    let sim_elapsed = sim_start.elapsed();
    clilog::finish!(timer_sim);

    // ── Results ──────────────────────────────────────────────────────────

    println!();
    println!("=== GPU Hybrid Simulation Results ===");
    println!("Ticks simulated: {}", max_ticks);
    println!("UART bytes received: {}", uart_monitor.events.len());
    if max_ticks > 0 {
        let us_per_tick = sim_elapsed.as_micros() as f64 / max_ticks as f64;
        println!("Time per tick: {:.1}μs ({:.1}s total)", us_per_tick, sim_elapsed.as_secs_f64());
    }

    // Print UART output as string
    if !uart_monitor.events.is_empty() {
        let uart_str: String = uart_monitor
            .events
            .iter()
            .map(|e| {
                if e.payload >= 32 && e.payload < 127 {
                    e.payload as char
                } else if e.payload == b'\n' {
                    '\n'
                } else if e.payload == b'\r' {
                    '\r'
                } else {
                    '.'
                }
            })
            .collect();
        println!("UART output:\n{}", uart_str);
    }

    // Print flash stats
    if let Some(ref fl) = flash {
        println!(
            "Flash model: steps={}, posedges={}, negedges={}",
            fl.get_step_count(),
            fl.get_posedge_count(),
            fl.get_negedge_count()
        );
    }

    // Output events to JSON
    if let Some(ref output_path) = config.output_events {
        #[derive(serde::Serialize)]
        struct EventsOutput {
            events: Vec<gem::testbench::UartEvent>,
        }
        let output = EventsOutput {
            events: uart_monitor.events,
        };
        let json = serde_json::to_string_pretty(&output).expect("Failed to serialize events");
        let mut file = File::create(output_path).expect("Failed to create events file");
        use std::io::Write;
        file.write_all(json.as_bytes())
            .expect("Failed to write events");
        clilog::info!("Wrote events to {}", output_path);
    }

    // ── Optional CPU verification ────────────────────────────────────────

    if args.check_with_cpu {
        clilog::info!("CPU verification not yet implemented for hybrid mode");
        // TODO: Run the same tick sequence through simulate_block_v1 and compare
        // This would need to save/replay all GPIO state changes
    }

    // Clean up event buffer
    unsafe {
        drop(Box::from_raw(event_buffer_ptr));
    }

    println!();
    println!("SIMULATION: PASSED");
}
