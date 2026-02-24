// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Metal GPU co-simulation engine with SPI flash, UART, and Wishbone bus trace models.
//!
//! Extracted from the `gpu_sim` binary. All IO models (flash, UART, bus trace)
//! run as GPU kernels — no per-tick CPU interaction needed.

use std::collections::HashMap;

use crate::aig::{DriverType, AIG};
use crate::flatten::FlattenedScriptV1;
use crate::sim::setup::{self, LoadedDesign};
use crate::testbench::{CppSpiFlash, PortMapping, TestbenchConfig, UartEvent};
use metal::{
    CommandQueue, ComputePipelineState, Device as MTLDevice, MTLResourceOptions, MTLSize,
    SharedEvent,
};
use netlistdb::{GeneralPinName, NetlistDB};
use ulib::{AsUPtr, Device};

/// Runtime options for co-simulation.
pub struct CosimOpts {
    pub max_cycles: Option<usize>,
    pub num_blocks: usize,
    pub flash_verbose: bool,
    pub check_with_cpu: bool,
    pub gpu_profile: bool,
    pub clock_period: Option<u64>,
}

/// Result of a co-simulation run.
pub struct CosimResult {
    pub passed: bool,
    pub uart_events: Vec<UartEvent>,
    pub ticks_simulated: usize,
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

/// Bit set/clear operation (must match Metal shader BitOp struct).
#[repr(C)]
#[derive(Clone, Copy)]
struct BitOp {
    position: u32, // bit position in state buffer
    value: u32,    // 0 = clear, 1 = set
}

/// Parameters for the state_prep kernel (must match Metal shader StatePrepParams struct).
#[repr(C)]
struct StatePrepParams {
    state_size: u32,   // number of u32 words per state slot
    num_ops: u32,      // number of bit set/clear operations
    num_monitors: u32, // number of peripheral monitors to check (0 = skip)
    tick_number: u32,  // current tick number
}

/// GPU-side flash state (must match Metal FlashState struct exactly).
#[repr(C)]
#[derive(Clone, Copy)]
struct FlashState {
    bit_count: i32,
    byte_count: i32,
    data_width: u32,
    addr: u32,
    curr_byte: u8,
    command: u8,
    out_buffer: u8,
    _pad1: u8,
    prev_clk: u32,
    prev_csn: u32,
    d_i: u8,
    _pad2: [u8; 3],
    prev_d_out: u8,
    _pad3: [u8; 3],
    in_reset: u32,
    last_error_cmd: u32,
    model_prev_csn: u32,
}

/// Parameters for gpu_apply_flash_din kernel (must match Metal FlashDinParams).
#[repr(C)]
struct FlashDinParams {
    d_in_pos: [u32; 4],
    has_flash: u32,
}

/// Parameters for gpu_flash_model_step kernel (must match Metal FlashModelParams).
#[repr(C)]
struct FlashModelParams {
    state_size: u32,
    clk_out_pos: u32,
    csn_out_pos: u32,
    d_out_pos: [u32; 4],
    flash_data_size: u32,
}

/// GPU-side UART decoder state (must match Metal UartDecoderState).
#[repr(C)]
struct UartDecoderState {
    state: u32, // 0=IDLE, 1=START, 2=DATA, 3=STOP
    last_tx: u32,
    start_cycle: u32,
    bits_received: u32,
    value: u32,
    current_cycle: u32,
}

/// Parameters for UART in gpu_io_step kernel (must match Metal UartParams).
#[repr(C)]
struct UartParams {
    state_size: u32,
    tx_out_pos: u32,
    cycles_per_bit: u32,
    has_uart: u32,
}

/// GPU-side UART channel (must match Metal UartChannel).
#[repr(C)]
struct UartChannel {
    write_head: u32,
    capacity: u32,
    _pad: [u32; 2],
    data: [u8; 4096],
}

const WB_TRACE_MAX_ADR_BITS: usize = 30;
const WB_TRACE_MAX_DAT_BITS: usize = 32;
const WB_TRACE_CHANNEL_CAP: usize = 16384;

/// Parameters for Wishbone bus trace (must match Metal WbTraceParams).
#[repr(C)]
struct WbTraceParams {
    ibus_cyc_pos: u32,
    ibus_stb_pos: u32,
    ibus_adr_pos: [u32; WB_TRACE_MAX_ADR_BITS],
    ibus_rdata_pos: [u32; WB_TRACE_MAX_DAT_BITS],
    dbus_cyc_pos: u32,
    dbus_stb_pos: u32,
    dbus_we_pos: u32,
    dbus_adr_pos: [u32; WB_TRACE_MAX_ADR_BITS],
    spiflash_ack_pos: u32,
    sram_ack_pos: u32,
    csr_ack_pos: u32,
    has_trace: u32,
}

/// Per-tick bus snapshot (must match Metal WbTraceEntry).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WbTraceEntry {
    tick: u32,
    flags: u32,
    ibus_adr: u32,
    ibus_rdata: u32,
    dbus_adr: u32,
}

/// GPU→CPU bus trace ring buffer header (must match Metal WbTraceChannel).
#[repr(C)]
struct WbTraceChannel {
    write_head: u32,
    capacity: u32,
    current_tick: u32,
    prev_flags: u32,
    // entries[capacity] follow in memory
}

/// Batch size for GPU-only simulation (no per-tick CPU interaction).
const BATCH_SIZE: usize = 1024;

/// Pre-allocated Metal buffers for each schedule position's fall/rise ops.
struct ScheduleBuffers {
    /// Per-schedule-tick: (fall_params, fall_ops, rise_params, rise_ops).
    tick_buffers: Vec<(metal::Buffer, metal::Buffer, metal::Buffer, metal::Buffer)>,
    /// Number of fall ops per schedule tick (for CPU verification).
    fall_ops_lens: Vec<usize>,
    /// Number of rise ops per schedule tick (for CPU verification).
    rise_ops_lens: Vec<usize>,
}

// ── Metal Simulator ──────────────────────────────────────────────────────────

struct MetalSimulator {
    device: metal::Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
    /// Pipeline state for the state_prep kernel (no monitors).
    state_prep_pipeline: ComputePipelineState,
    /// Pipeline state for gpu_apply_flash_din kernel.
    gpu_apply_flash_din_pipeline: ComputePipelineState,
    /// Pipeline state for gpu_flash_model_step kernel.
    gpu_flash_model_step_pipeline: ComputePipelineState,
    /// Pipeline state for gpu_io_step kernel (UART + bus trace combined).
    gpu_io_step_pipeline: ComputePipelineState,
    /// Pre-allocated params buffers for each stage (shared memory, rewritten each dispatch).
    /// We need one per stage since multi-stage designs encode all stages before commit.
    params_buffers: Vec<metal::Buffer>,
    /// Shared event for GPU↔CPU synchronization within a single command buffer.
    shared_event: SharedEvent,
    /// Monotonic counter for shared event signaling
    event_counter: std::cell::Cell<u64>,
}

impl MetalSimulator {
    fn new(num_stages: usize) -> Self {
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

        let state_prep_fn = library
            .get_function("state_prep", None)
            .expect("Failed to get state_prep function");
        let state_prep_pipeline = device
            .new_compute_pipeline_state_with_function(&state_prep_fn)
            .expect("Failed to create state_prep pipeline");

        let flash_din_fn = library
            .get_function("gpu_apply_flash_din", None)
            .expect("Failed to get gpu_apply_flash_din function");
        let gpu_apply_flash_din_pipeline = device
            .new_compute_pipeline_state_with_function(&flash_din_fn)
            .expect("Failed to create gpu_apply_flash_din pipeline");

        let flash_step_fn = library
            .get_function("gpu_flash_model_step", None)
            .expect("Failed to get gpu_flash_model_step function");
        let gpu_flash_model_step_pipeline = device
            .new_compute_pipeline_state_with_function(&flash_step_fn)
            .expect("Failed to create gpu_flash_model_step pipeline");

        let io_step_fn = library
            .get_function("gpu_io_step", None)
            .expect("Failed to get gpu_io_step function");
        let gpu_io_step_pipeline = device
            .new_compute_pipeline_state_with_function(&io_step_fn)
            .expect("Failed to create gpu_io_step pipeline");

        let command_queue = device.new_command_queue();

        // Pre-allocate one params buffer per stage
        let params_buffers: Vec<_> = (0..num_stages.max(1))
            .map(|_| {
                device.new_buffer(
                    std::mem::size_of::<SimParams>() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();

        let shared_event = device.new_shared_event();

        Self {
            device,
            command_queue,
            pipeline_state,
            state_prep_pipeline,
            gpu_apply_flash_din_pipeline,
            gpu_flash_model_step_pipeline,
            gpu_io_step_pipeline,
            params_buffers,
            shared_event,
            event_counter: std::cell::Cell::new(0),
        }
    }

    /// Dispatch a single stage (standalone, with own command buffer).
    /// Used as fallback when dual-tick pattern isn't applicable.
    #[allow(dead_code)]
    #[inline]
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
        timing_constraints_buffer: Option<&metal::Buffer>,
    ) {
        self.write_params(stage_i, num_blocks, num_major_stages, state_size, cycle_i);

        let command_buffer = self.command_queue.new_command_buffer();
        self.encode_dispatch(
            &command_buffer,
            num_blocks,
            stage_i,
            blocks_start_buffer,
            blocks_data_buffer,
            sram_data_buffer,
            states_buffer,
            event_buffer_metal,
            timing_constraints_buffer,
        );
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Write params for a given stage into the pre-allocated shared memory buffer.
    #[inline]
    fn write_params(
        &self,
        stage_i: usize,
        num_blocks: usize,
        num_major_stages: usize,
        state_size: usize,
        cycle_i: usize,
    ) {
        let params = SimParams {
            num_blocks: num_blocks as u64,
            num_major_stages: num_major_stages as u64,
            num_cycles: 1,
            state_size: state_size as u64,
            current_cycle: cycle_i as u64,
            current_stage: stage_i as u64,
        };
        unsafe {
            std::ptr::write(
                self.params_buffers[stage_i].contents() as *mut SimParams,
                params,
            );
        }
    }

    /// Encode a compute dispatch into an existing command buffer (no commit).
    #[inline]
    fn encode_dispatch(
        &self,
        command_buffer: &metal::CommandBufferRef,
        num_blocks: usize,
        stage_i: usize,
        blocks_start_buffer: &metal::Buffer,
        blocks_data_buffer: &metal::Buffer,
        sram_data_buffer: &metal::Buffer,
        states_buffer: &metal::Buffer,
        event_buffer_metal: &metal::Buffer,
        timing_constraints_buffer: Option<&metal::Buffer>,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline_state);
        encoder.set_buffer(0, Some(blocks_start_buffer), 0);
        encoder.set_buffer(1, Some(blocks_data_buffer), 0);
        encoder.set_buffer(2, Some(sram_data_buffer), 0);
        encoder.set_buffer(3, Some(states_buffer), 0);
        encoder.set_buffer(4, Some(&self.params_buffers[stage_i]), 0);
        encoder.set_buffer(5, Some(event_buffer_metal), 0);
        encoder.set_buffer(6, timing_constraints_buffer.map(|v| &**v), 0);

        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(num_blocks as u64, 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
    }

    /// Spin-wait for shared event to reach target value.
    #[inline]
    fn spin_wait(&self, target: u64) {
        while self.shared_event.signaled_value() < target {
            std::hint::spin_loop();
        }
    }

    /// Encode a state_prep dispatch into an existing command buffer.
    #[inline]
    fn encode_state_prep(
        &self,
        command_buffer: &metal::CommandBufferRef,
        states_buffer: &metal::Buffer,
        prep_params_buffer: &metal::Buffer,
        ops_buffer: &metal::Buffer,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.state_prep_pipeline);
        encoder.set_buffer(0, Some(states_buffer), 0);
        encoder.set_buffer(1, Some(prep_params_buffer), 0);
        encoder.set_buffer(2, Some(ops_buffer), 0);
        let tpg = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tpg);
        encoder.end_encoding();
    }

    /// Encode a gpu_apply_flash_din dispatch into an existing command buffer.
    #[inline]
    fn encode_apply_flash_din(
        &self,
        command_buffer: &metal::CommandBufferRef,
        states_buffer: &metal::Buffer,
        flash_state_buffer: &metal::Buffer,
        flash_din_params_buffer: &metal::Buffer,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gpu_apply_flash_din_pipeline);
        encoder.set_buffer(0, Some(states_buffer), 0);
        encoder.set_buffer(1, Some(flash_state_buffer), 0);
        encoder.set_buffer(2, Some(flash_din_params_buffer), 0);
        let tpg = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tpg);
        encoder.end_encoding();
    }

    /// Encode a gpu_flash_model_step dispatch into an existing command buffer.
    #[inline]
    fn encode_flash_model_step(
        &self,
        command_buffer: &metal::CommandBufferRef,
        states_buffer: &metal::Buffer,
        flash_state_buffer: &metal::Buffer,
        flash_model_params_buffer: &metal::Buffer,
        flash_data_buffer: &metal::Buffer,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gpu_flash_model_step_pipeline);
        encoder.set_buffer(0, Some(states_buffer), 0);
        encoder.set_buffer(1, Some(flash_state_buffer), 0);
        encoder.set_buffer(2, Some(flash_model_params_buffer), 0);
        encoder.set_buffer(3, Some(flash_data_buffer), 0);
        let tpg = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tpg);
        encoder.end_encoding();
    }

    /// Encode a gpu_io_step dispatch (UART + bus trace) into an existing command buffer.
    #[inline]
    fn encode_io_step(
        &self,
        command_buffer: &metal::CommandBufferRef,
        states_buffer: &metal::Buffer,
        uart_state_buffer: &metal::Buffer,
        uart_params_buffer: &metal::Buffer,
        uart_channel_buffer: &metal::Buffer,
        wb_trace_channel_buffer: &metal::Buffer,
        wb_trace_params_buffer: &metal::Buffer,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gpu_io_step_pipeline);
        encoder.set_buffer(0, Some(states_buffer), 0);
        encoder.set_buffer(1, Some(uart_state_buffer), 0);
        encoder.set_buffer(2, Some(uart_params_buffer), 0);
        encoder.set_buffer(3, Some(uart_channel_buffer), 0);
        encoder.set_buffer(4, Some(wb_trace_channel_buffer), 0);
        encoder.set_buffer(5, Some(wb_trace_params_buffer), 0);
        let tpg = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), tpg);
        encoder.end_encoding();
    }

    /// Encode and commit a GPU-only batch of K ticks in a single command buffer.
    ///
    /// No per-tick CPU interaction — flash and UART are handled entirely on GPU.
    /// A single signal at the end notifies the CPU that the batch is complete.
    ///
    /// Each tick encodes:
    ///   1. state_prep (falling edge ops)
    ///   2. gpu_apply_flash_din
    ///   3. simulate_v1_stage × num_stages (falling edge)
    ///   4. state_prep (rising edge ops)
    ///   5. gpu_apply_flash_din
    ///   6. simulate_v1_stage × num_stages (rising edge)
    ///   7. gpu_flash_model_step (after rising edge — sees posedge SPI CLK)
    ///   8. gpu_io_step (UART + bus trace)
    ///
    /// The flash model runs TWICE per tick (after each half-cycle simulate),
    /// matching timing_sim_cpu behavior. This is critical because the SPI CLK
    /// signal passes through clock gating logic (Q = system_CLK & EN_latch),
    /// so the flash sees CLK=0 after the falling edge and CLK=EN_latch after
    /// the rising edge. The d_in from the falling-edge flash step is picked up
    /// by the rising-edge apply_flash_din, allowing the flash controller to
    /// see the MISO response within the same tick.
    ///
    /// Returns the event value that signals batch completion.
    #[allow(clippy::too_many_arguments)]
    fn encode_and_commit_gpu_batch(
        &self,
        batch_size: usize,
        num_blocks: usize,
        num_major_stages: usize,
        state_size: usize,
        blocks_start_buffer: &metal::Buffer,
        blocks_data_buffer: &metal::Buffer,
        sram_data_buffer: &metal::Buffer,
        states_buffer: &metal::Buffer,
        event_buffer_metal: &metal::Buffer,
        schedule_buffers: &ScheduleBuffers,
        schedule_offset: usize,
        flash_state_buffer: &metal::Buffer,
        flash_din_params_buffer: &metal::Buffer,
        flash_model_params_buffer: &metal::Buffer,
        flash_data_buffer: &metal::Buffer,
        uart_state_buffer: &metal::Buffer,
        uart_params_buffer: &metal::Buffer,
        uart_channel_buffer: &metal::Buffer,
        wb_trace_channel_buffer: &metal::Buffer,
        wb_trace_params_buffer: &metal::Buffer,
        timing_constraints_buffer: Option<&metal::Buffer>,
    ) -> u64 {
        let batch_done = self.event_counter.get() + 1;
        let cb = self.command_queue.new_command_buffer();
        let schedule_len = schedule_buffers.tick_buffers.len();

        for tick_offset in 0..batch_size {
            let sched_idx = (schedule_offset + tick_offset) % schedule_len;
            let (ref fall_params, ref fall_ops, ref rise_params, ref rise_ops) =
                schedule_buffers.tick_buffers[sched_idx];

            // ── Falling edge: state_prep + flash_din + simulate ──
            self.encode_state_prep(cb, states_buffer, fall_params, fall_ops);
            self.encode_apply_flash_din(
                cb,
                states_buffer,
                flash_state_buffer,
                flash_din_params_buffer,
            );
            for stage_i in 0..num_major_stages {
                self.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
                self.encode_dispatch(
                    cb,
                    num_blocks,
                    stage_i,
                    blocks_start_buffer,
                    blocks_data_buffer,
                    sram_data_buffer,
                    states_buffer,
                    event_buffer_metal,
                    timing_constraints_buffer,
                );
            }

            // ── Flash model step after falling edge ──
            // Flash sees SPI CLK=0 (clock gated low when system CLK=0).
            // Produces d_in that will be applied before rising-edge simulate.
            self.encode_flash_model_step(
                cb,
                states_buffer,
                flash_state_buffer,
                flash_model_params_buffer,
                flash_data_buffer,
            );

            // ── Rising edge: state_prep + flash_din + simulate ──
            self.encode_state_prep(cb, states_buffer, rise_params, rise_ops);
            self.encode_apply_flash_din(
                cb,
                states_buffer,
                flash_state_buffer,
                flash_din_params_buffer,
            );
            for stage_i in 0..num_major_stages {
                self.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
                self.encode_dispatch(
                    cb,
                    num_blocks,
                    stage_i,
                    blocks_start_buffer,
                    blocks_data_buffer,
                    sram_data_buffer,
                    states_buffer,
                    event_buffer_metal,
                    timing_constraints_buffer,
                );
            }

            // ── Flash model step after rising edge + IO step ──
            // Flash sees SPI CLK after DFFs have latched (posedge system CLK).
            self.encode_flash_model_step(
                cb,
                states_buffer,
                flash_state_buffer,
                flash_model_params_buffer,
                flash_data_buffer,
            );
            self.encode_io_step(
                cb,
                states_buffer,
                uart_state_buffer,
                uart_params_buffer,
                uart_channel_buffer,
                wb_trace_channel_buffer,
                wb_trace_params_buffer,
            );
        }

        // Single signal at end of entire batch
        cb.encode_signal_event(&self.shared_event, batch_done);
        cb.commit();

        self.event_counter.set(batch_done);
        batch_done
    }

    /// Run GPU kernel profiling: dispatch each kernel in its own command buffer,
    /// wait for completion, and measure GPU execution time.
    #[allow(clippy::too_many_arguments)]
    fn profile_gpu_kernels(
        &self,
        num_ticks: usize,
        num_blocks: usize,
        num_major_stages: usize,
        state_size: usize,
        blocks_start_buffer: &metal::Buffer,
        blocks_data_buffer: &metal::Buffer,
        sram_data_buffer: &metal::Buffer,
        states_buffer: &metal::Buffer,
        event_buffer_metal: &metal::Buffer,
        schedule_buffers: &ScheduleBuffers,
        flash_state_buffer: &metal::Buffer,
        flash_din_params_buffer: &metal::Buffer,
        flash_model_params_buffer: &metal::Buffer,
        flash_data_buffer: &metal::Buffer,
        uart_state_buffer: &metal::Buffer,
        uart_params_buffer: &metal::Buffer,
        uart_channel_buffer: &metal::Buffer,
        wb_trace_channel_buffer: &metal::Buffer,
        wb_trace_params_buffer: &metal::Buffer,
        timing_constraints_buffer: Option<&metal::Buffer>,
    ) {
        // Use schedule position 0 for profiling (all patterns have same kernel cost)
        let (ref fall_prep_params_buffer, ref fall_ops_buffer, ref rise_prep_params_buffer, ref rise_ops_buffer) =
            schedule_buffers.tick_buffers[0];
        #[inline]
        fn gpu_times(cb: &metal::CommandBufferRef) -> (f64, f64) {
            unsafe {
                let obj: *mut objc::runtime::Object =
                    &*(cb as *const _ as *const objc::runtime::Object) as *const _ as *mut _;
                let start: f64 = msg_send![obj, GPUStartTime];
                let end: f64 = msg_send![obj, GPUEndTime];
                (start, end)
            }
        }

        println!("\n=== GPU Kernel Profiling ({} ticks) ===\n", num_ticks);

        // Warmup: 10 ticks (with dual flash stepping)
        for _ in 0..10 {
            let cb = self.command_queue.new_command_buffer();
            self.encode_state_prep(cb, states_buffer, fall_prep_params_buffer, fall_ops_buffer);
            self.encode_apply_flash_din(
                cb,
                states_buffer,
                flash_state_buffer,
                flash_din_params_buffer,
            );
            for stage_i in 0..num_major_stages {
                self.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
                self.encode_dispatch(
                    cb,
                    num_blocks,
                    stage_i,
                    blocks_start_buffer,
                    blocks_data_buffer,
                    sram_data_buffer,
                    states_buffer,
                    event_buffer_metal,
                    timing_constraints_buffer,
                );
            }
            self.encode_flash_model_step(
                cb,
                states_buffer,
                flash_state_buffer,
                flash_model_params_buffer,
                flash_data_buffer,
            );
            self.encode_state_prep(cb, states_buffer, rise_prep_params_buffer, rise_ops_buffer);
            self.encode_apply_flash_din(
                cb,
                states_buffer,
                flash_state_buffer,
                flash_din_params_buffer,
            );
            for stage_i in 0..num_major_stages {
                self.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
                self.encode_dispatch(
                    cb,
                    num_blocks,
                    stage_i,
                    blocks_start_buffer,
                    blocks_data_buffer,
                    sram_data_buffer,
                    states_buffer,
                    event_buffer_metal,
                    timing_constraints_buffer,
                );
            }
            self.encode_flash_model_step(
                cb,
                states_buffer,
                flash_state_buffer,
                flash_model_params_buffer,
                flash_data_buffer,
            );
            self.encode_io_step(
                cb,
                states_buffer,
                uart_state_buffer,
                uart_params_buffer,
                uart_channel_buffer,
                wb_trace_channel_buffer,
                wb_trace_params_buffer,
            );
            cb.commit();
            cb.wait_until_completed();
        }

        let mut time_state_prep_fall = 0.0f64;
        let mut time_flash_din = 0.0f64;
        let mut time_simulate_fall = 0.0f64;
        let mut time_flash_step_fall = 0.0f64;
        let mut time_state_prep_rise = 0.0f64;
        let mut time_simulate_rise = 0.0f64;
        let mut time_flash_step_rise = 0.0f64;
        let mut time_io_step = 0.0f64;
        let mut time_full_tick = 0.0f64;

        let wall_start = std::time::Instant::now();

        for _tick in 0..num_ticks {
            // 1. state_prep (falling) — isolated
            let cb1 = self.command_queue.new_command_buffer();
            self.encode_state_prep(cb1, states_buffer, fall_prep_params_buffer, fall_ops_buffer);
            cb1.commit();
            cb1.wait_until_completed();
            let (s, e) = gpu_times(cb1);
            time_state_prep_fall += e - s;

            // 2. gpu_apply_flash_din — isolated
            let cb1b = self.command_queue.new_command_buffer();
            self.encode_apply_flash_din(
                cb1b,
                states_buffer,
                flash_state_buffer,
                flash_din_params_buffer,
            );
            cb1b.commit();
            cb1b.wait_until_completed();
            let (s, e) = gpu_times(cb1b);
            time_flash_din += e - s;

            // 3. simulate (falling) — isolated per stage
            for stage_i in 0..num_major_stages {
                self.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
                let cb2 = self.command_queue.new_command_buffer();
                self.encode_dispatch(
                    cb2,
                    num_blocks,
                    stage_i,
                    blocks_start_buffer,
                    blocks_data_buffer,
                    sram_data_buffer,
                    states_buffer,
                    event_buffer_metal,
                    timing_constraints_buffer,
                );
                cb2.commit();
                cb2.wait_until_completed();
                let (s, e) = gpu_times(cb2);
                time_simulate_fall += e - s;
            }

            // 3b. gpu_flash_model_step after falling edge — isolated
            let cb2b = self.command_queue.new_command_buffer();
            self.encode_flash_model_step(
                cb2b,
                states_buffer,
                flash_state_buffer,
                flash_model_params_buffer,
                flash_data_buffer,
            );
            cb2b.commit();
            cb2b.wait_until_completed();
            let (s, e) = gpu_times(cb2b);
            time_flash_step_fall += e - s;

            // 4. state_prep (rising) — isolated
            let cb3 = self.command_queue.new_command_buffer();
            self.encode_state_prep(cb3, states_buffer, rise_prep_params_buffer, rise_ops_buffer);
            cb3.commit();
            cb3.wait_until_completed();
            let (s, e) = gpu_times(cb3);
            time_state_prep_rise += e - s;

            // 5. gpu_apply_flash_din — isolated
            let cb3b = self.command_queue.new_command_buffer();
            self.encode_apply_flash_din(
                cb3b,
                states_buffer,
                flash_state_buffer,
                flash_din_params_buffer,
            );
            cb3b.commit();
            cb3b.wait_until_completed();
            let (s, e) = gpu_times(cb3b);
            time_flash_din += e - s;

            // 6. simulate (rising) — isolated per stage
            for stage_i in 0..num_major_stages {
                self.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
                let cb4 = self.command_queue.new_command_buffer();
                self.encode_dispatch(
                    cb4,
                    num_blocks,
                    stage_i,
                    blocks_start_buffer,
                    blocks_data_buffer,
                    sram_data_buffer,
                    states_buffer,
                    event_buffer_metal,
                    timing_constraints_buffer,
                );
                cb4.commit();
                cb4.wait_until_completed();
                let (s, e) = gpu_times(cb4);
                time_simulate_rise += e - s;
            }

            // 7. gpu_flash_model_step after rising edge — isolated
            let cb5 = self.command_queue.new_command_buffer();
            self.encode_flash_model_step(
                cb5,
                states_buffer,
                flash_state_buffer,
                flash_model_params_buffer,
                flash_data_buffer,
            );
            cb5.commit();
            cb5.wait_until_completed();
            let (s, e) = gpu_times(cb5);
            time_flash_step_rise += e - s;

            // 8. gpu_io_step — isolated
            let cb6 = self.command_queue.new_command_buffer();
            self.encode_io_step(
                cb6,
                states_buffer,
                uart_state_buffer,
                uart_params_buffer,
                uart_channel_buffer,
                wb_trace_channel_buffer,
                wb_trace_params_buffer,
            );
            cb6.commit();
            cb6.wait_until_completed();
            let (s, e) = gpu_times(cb6);
            time_io_step += e - s;

            // Full tick in single CB for comparison
            let cb_full = self.command_queue.new_command_buffer();
            self.encode_state_prep(
                cb_full,
                states_buffer,
                fall_prep_params_buffer,
                fall_ops_buffer,
            );
            self.encode_apply_flash_din(
                cb_full,
                states_buffer,
                flash_state_buffer,
                flash_din_params_buffer,
            );
            for stage_i in 0..num_major_stages {
                self.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
                self.encode_dispatch(
                    cb_full,
                    num_blocks,
                    stage_i,
                    blocks_start_buffer,
                    blocks_data_buffer,
                    sram_data_buffer,
                    states_buffer,
                    event_buffer_metal,
                    timing_constraints_buffer,
                );
            }
            self.encode_flash_model_step(
                cb_full,
                states_buffer,
                flash_state_buffer,
                flash_model_params_buffer,
                flash_data_buffer,
            );
            self.encode_state_prep(
                cb_full,
                states_buffer,
                rise_prep_params_buffer,
                rise_ops_buffer,
            );
            self.encode_apply_flash_din(
                cb_full,
                states_buffer,
                flash_state_buffer,
                flash_din_params_buffer,
            );
            for stage_i in 0..num_major_stages {
                self.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
                self.encode_dispatch(
                    cb_full,
                    num_blocks,
                    stage_i,
                    blocks_start_buffer,
                    blocks_data_buffer,
                    sram_data_buffer,
                    states_buffer,
                    event_buffer_metal,
                    timing_constraints_buffer,
                );
            }
            self.encode_flash_model_step(
                cb_full,
                states_buffer,
                flash_state_buffer,
                flash_model_params_buffer,
                flash_data_buffer,
            );
            self.encode_io_step(
                cb_full,
                states_buffer,
                uart_state_buffer,
                uart_params_buffer,
                uart_channel_buffer,
                wb_trace_channel_buffer,
                wb_trace_params_buffer,
            );
            cb_full.commit();
            cb_full.wait_until_completed();
            let (s, e) = gpu_times(cb_full);
            time_full_tick += e - s;
        }

        let wall_elapsed = wall_start.elapsed();
        let n = num_ticks as f64;

        let time_flash_step = time_flash_step_fall + time_flash_step_rise;
        let total_isolated = time_state_prep_fall
            + time_flash_din
            + time_simulate_fall
            + time_flash_step_fall
            + time_state_prep_rise
            + time_simulate_rise
            + time_flash_step_rise
            + time_io_step;

        let print_kernel = |name: &str, t: f64| {
            let us = t / n * 1e6;
            let pct = if total_isolated > 0.0 {
                100.0 * t / total_isolated
            } else {
                0.0
            };
            println!("  {:<36} {:>8.2}μs/tick  {:>5.1}%", name, us, pct);
        };

        println!("Per-kernel GPU time (isolated command buffers):");
        print_kernel("state_prep (falling)", time_state_prep_fall);
        print_kernel("gpu_apply_flash_din (2×)", time_flash_din);
        print_kernel("simulate_v1_stage (falling)", time_simulate_fall);
        print_kernel("gpu_flash_model_step (falling)", time_flash_step_fall);
        print_kernel("state_prep (rising)", time_state_prep_rise);
        print_kernel("simulate_v1_stage (rising)", time_simulate_rise);
        print_kernel("gpu_flash_model_step (rising)", time_flash_step_rise);
        print_kernel("gpu_io_step", time_io_step);
        println!(
            "  {:<36} {:>8.2}μs/tick",
            "TOTAL (isolated sum)",
            total_isolated / n * 1e6
        );
        println!();
        println!(
            "  {:<36} {:>8.2}μs/tick",
            "Full tick (single CB)",
            time_full_tick / n * 1e6
        );
        println!(
            "  {:<36} {:>8.2}μs/tick",
            "Wall clock (2× ticks, profiling)",
            wall_elapsed.as_secs_f64() / n * 1e6
        );
        println!();

        let sim_us = time_simulate_fall / n * 1e6 + time_simulate_rise / n * 1e6;
        let io_us = time_flash_din / n * 1e6 + time_flash_step / n * 1e6 + time_io_step / n * 1e6;
        let prep_us = time_state_prep_fall / n * 1e6 + time_state_prep_rise / n * 1e6;
        println!(
            "  Simulation kernels total:     {:>8.2}μs/tick  ({:.1}%)",
            sim_us,
            100.0 * (time_simulate_fall + time_simulate_rise) / total_isolated
        );
        println!(
            "  IO model kernels total:       {:>8.2}μs/tick  ({:.1}%)",
            io_us,
            100.0 * (time_flash_din + time_flash_step + time_io_step) / total_isolated
        );
        println!(
            "  State prep total:             {:>8.2}μs/tick  ({:.1}%)",
            prep_us,
            100.0 * (time_state_prep_fall + time_state_prep_rise) / total_isolated
        );
        println!();
        println!(
            "  CB submission overhead:        {:>8.2}μs/tick (wall - GPU)",
            (wall_elapsed.as_secs_f64() - total_isolated - time_full_tick) / n * 1e6
        );
    }
}

// ── GPIO ↔ State Buffer Mapping ──────────────────────────────────────────────

/// Per-clock-domain flag bit positions in the packed state buffer.
///
/// Each clock domain (identified by its netlistdb pin ID) has its own set of
/// posedge/negedge flag bits that gate DFF writeout for that domain.
struct ClockDomainFlags {
    /// Netlistdb pin ID for this clock (key from `clock_pin2aigpins`).
    #[allow(dead_code)]
    clock_pinid: usize,
    /// Human-readable name from pin (e.g. "io$clk$i").
    name: String,
    /// State positions for this domain's posedge flags.
    posedge_flag_bits: Vec<u32>,
    /// State positions for this domain's negedge flags.
    negedge_flag_bits: Vec<u32>,
    /// State position of the clock input port itself (if it's a primary input).
    clock_input_pos: Option<u32>,
    /// GPIO index if this clock is driven from a GPIO.
    clock_gpio: Option<usize>,
}

/// Maps GPIO pin indices to bit positions in the packed u32 state buffer.
pub(crate) struct GpioMapping {
    /// gpio_in[idx] → (aigpin, state bit position)
    input_bits: HashMap<usize, u32>,
    /// gpio_out[idx] → state bit position in output_map
    output_bits: HashMap<usize, u32>,
    /// Per-clock-domain flag bit positions (grouped by netlistdb pin ID).
    clock_domains: Vec<ClockDomainFlags>,
    /// Named input port → state bit position (for non-GPIO ports like por_l, resetb_h)
    named_input_bits: HashMap<String, u32>,
}

impl GpioMapping {
    /// All posedge flag bit positions across all clock domains.
    fn all_posedge_flag_bits(&self) -> Vec<u32> {
        self.clock_domains
            .iter()
            .flat_map(|d| d.posedge_flag_bits.iter().copied())
            .collect()
    }

    /// All negedge flag bit positions across all clock domains.
    fn all_negedge_flag_bits(&self) -> Vec<u32> {
        self.clock_domains
            .iter()
            .flat_map(|d| d.negedge_flag_bits.iter().copied())
            .collect()
    }
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
#[allow(dead_code)]
#[inline]
fn clear_bit(state: &mut [u32], pos: u32) {
    state[(pos >> 5) as usize] &= !(1u32 << (pos & 31));
}

/// Build GPIO-to-state-buffer mapping from AIG + FlattenedScript.
///
/// When `port_mapping` is provided, maps GPIO indices to port names explicitly
/// (for designs like ChipFlow that use named ports instead of gpio_in[N]/gpio_out[N]).
/// Falls back to parsing gpio_in[N]/gpio_out[N] from pin names when no mapping given.
pub(crate) fn build_gpio_mapping(
    aig: &AIG,
    netlistdb: &NetlistDB,
    script: &FlattenedScriptV1,
    port_mapping: Option<&PortMapping>,
) -> GpioMapping {
    let mut input_bits: HashMap<usize, u32> = HashMap::new();
    let mut output_bits: HashMap<usize, u32> = HashMap::new();
    let mut named_input_bits: HashMap<String, u32> = HashMap::new();

    // Group clock flags by pinid → ClockDomainFlags
    let mut clock_domain_map: HashMap<usize, (Vec<u32>, Vec<u32>)> = HashMap::new();

    // Build reverse lookup: port_name → gpio_index for port_mapping mode
    let input_name_to_gpio: HashMap<String, usize> = port_mapping
        .map(|pm| {
            pm.inputs
                .iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (v.clone(), idx)))
                .collect()
        })
        .unwrap_or_default();
    let output_name_to_gpio: HashMap<String, usize> = port_mapping
        .map(|pm| {
            pm.outputs
                .iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (v.clone(), idx)))
                .collect()
        })
        .unwrap_or_default();

    // Map input ports → state buffer positions
    for (aigpin_idx, driv) in aig.drivers.iter().enumerate() {
        match driv {
            DriverType::InputPort(pinid) => {
                let pin_name = netlistdb.pinnames[*pinid].dbg_fmt_pin();
                // Try port_mapping first, then fall back to gpio_in[N] parsing
                let gpio_idx = input_name_to_gpio
                    .get(&pin_name)
                    .copied()
                    .or_else(|| parse_gpio_index(&pin_name, "gpio_in"));
                if let Some(gpio_idx) = gpio_idx {
                    if let Some(&pos) = script.input_map.get(&aigpin_idx) {
                        input_bits.insert(gpio_idx, pos);
                        clilog::debug!(
                            "input[{}] = pin '{}' → state pos {}",
                            gpio_idx,
                            pin_name,
                            pos
                        );
                    }
                }
                // Also store by name for constant_ports
                if let Some(&pos) = script.input_map.get(&aigpin_idx) {
                    named_input_bits.insert(pin_name.clone(), pos);
                }
            }
            DriverType::InputClockFlag(pinid, is_negedge) => {
                if let Some(&pos) = script.input_map.get(&aigpin_idx) {
                    let entry = clock_domain_map
                        .entry(*pinid)
                        .or_insert_with(|| (Vec::new(), Vec::new()));
                    if *is_negedge == 0 {
                        entry.0.push(pos);
                    } else {
                        entry.1.push(pos);
                    }
                    let pin_name = netlistdb.pinnames[*pinid].dbg_fmt_pin();
                    clilog::debug!(
                        "ClockFlag aigpin={} pin={} (pinid={}) negedge={} pos={}",
                        aigpin_idx,
                        pin_name,
                        pinid,
                        is_negedge,
                        pos
                    );
                }
            }
            _ => {}
        }
    }

    // Map output ports → state buffer positions
    for i in netlistdb.cell2pin.iter_set(0) {
        let pin_name = netlistdb.pinnames[i].dbg_fmt_pin();
        // Try port_mapping first, then fall back to gpio_out[N] parsing
        let gpio_idx = output_name_to_gpio
            .get(&pin_name)
            .copied()
            .or_else(|| parse_gpio_index(&pin_name, "gpio_out"));
        if let Some(gpio_idx) = gpio_idx {
            let aigpin_iv = aig.pin2aigpin_iv[i];
            if aigpin_iv == usize::MAX {
                clilog::info!(
                    "output[{}] pin '{}' has no AIG connection (usize::MAX)",
                    gpio_idx,
                    pin_name
                );
                continue;
            }
            if aigpin_iv <= 1 {
                clilog::info!(
                    "output[{}] pin '{}' is constant (aigpin_iv={})",
                    gpio_idx,
                    pin_name,
                    aigpin_iv
                );
                continue;
            }
            if let Some(&pos) = script.output_map.get(&aigpin_iv) {
                output_bits.insert(gpio_idx, pos);
                clilog::debug!(
                    "output[{}] = pin '{}' → state pos {}",
                    gpio_idx,
                    pin_name,
                    pos
                );
            } else if let Some(&pos) = script.output_map.get(&(aigpin_iv ^ 1)) {
                output_bits.insert(gpio_idx, pos);
                clilog::debug!(
                    "output[{}] = pin '{}' → state pos {} (flipped inv)",
                    gpio_idx,
                    pin_name,
                    pos
                );
            } else {
                clilog::info!(
                    "output[{}] pin '{}' aigpin_iv={} (aigpin={}) not in output_map (dir={:?})",
                    gpio_idx,
                    pin_name,
                    aigpin_iv,
                    aigpin_iv >> 1,
                    netlistdb.pindirect[i]
                );
            }
        }
    }

    // Build ClockDomainFlags from grouped clock flags
    // Also resolve GPIO index for each clock domain by looking up the clock pin
    // in the input_bits mapping.
    let mut clock_domains: Vec<ClockDomainFlags> = Vec::new();
    // Build reverse: pinid → (pin_name, gpio_idx, state_pos) from input_bits
    let mut pinid_to_input_info: HashMap<usize, (String, Option<usize>, Option<u32>)> =
        HashMap::new();
    for (aigpin_idx, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(pinid) = driv {
            let pin_name = netlistdb.pinnames[*pinid].dbg_fmt_pin();
            let gpio_idx = input_name_to_gpio
                .get(&pin_name)
                .copied()
                .or_else(|| parse_gpio_index(&pin_name, "gpio_in"));
            let pos = script.input_map.get(&aigpin_idx).copied();
            pinid_to_input_info.insert(*pinid, (pin_name, gpio_idx, pos));
        }
    }

    for (pinid, (posedge_bits, negedge_bits)) in &clock_domain_map {
        let pin_name = netlistdb.pinnames[*pinid].dbg_fmt_pin();
        // Try to find the GPIO index and state position for this clock pin
        let (clock_gpio, clock_input_pos) =
            if let Some((_, gpio_idx, pos)) = pinid_to_input_info.get(pinid) {
                (*gpio_idx, *pos)
            } else {
                (None, None)
            };
        clock_domains.push(ClockDomainFlags {
            clock_pinid: *pinid,
            name: pin_name.clone(),
            posedge_flag_bits: posedge_bits.clone(),
            negedge_flag_bits: negedge_bits.clone(),
            clock_input_pos,
            clock_gpio,
        });
        clilog::info!(
            "Clock domain '{}' (pinid={}): {} posedge flags, {} negedge flags, gpio={:?}",
            pin_name,
            pinid,
            posedge_bits.len(),
            negedge_bits.len(),
            clock_gpio
        );
    }

    let total_posedge: usize = clock_domains.iter().map(|d| d.posedge_flag_bits.len()).sum();
    let total_negedge: usize = clock_domains.iter().map(|d| d.negedge_flag_bits.len()).sum();
    clilog::info!(
        "GPIO mapping: {} inputs, {} outputs, {} clock domains ({} posedge flags, {} negedge flags)",
        input_bits.len(),
        output_bits.len(),
        clock_domains.len(),
        total_posedge,
        total_negedge
    );

    clilog::info!("Named input ports: {} mapped", named_input_bits.len());

    GpioMapping {
        input_bits,
        output_bits,
        clock_domains,
        named_input_bits,
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

/// Resolve an internal signal name to its output state bit position.
///
/// Searches DFF Q output pins (which appear as cell output pins driving named nets)
/// for names containing `pattern`. Returns the state buffer bit position, or 0xFFFFFFFF.
fn resolve_signal_pos(
    aig: &AIG,
    netlistdb: &NetlistDB,
    script: &FlattenedScriptV1,
    pattern: &str,
) -> u32 {
    // Strategy: scan ALL cell pins looking for names matching pattern.
    // DFF Q outputs are cell output pins whose names match the net they drive.
    for cellid in 0..netlistdb.num_cells {
        for pinid in netlistdb.cell2pin.iter_set(cellid) {
            let pin_name = netlistdb.pinnames[pinid].dbg_fmt_pin();
            if !pin_name.contains(pattern) {
                continue;
            }
            // Only care about output pins (Q of DFFs) — check for output_map presence
            let aigpin_iv = aig.pin2aigpin_iv[pinid];
            if aigpin_iv == usize::MAX || aigpin_iv <= 1 {
                continue;
            }
            if let Some(&pos) = script.output_map.get(&aigpin_iv) {
                return pos;
            }
            if let Some(&pos) = script.output_map.get(&(aigpin_iv ^ 1)) {
                return pos;
            }
        }
    }
    0xFFFFFFFF
}

/// Resolve a bus signal with bit index, e.g. "ibus__adr" with bit 5 → "ibus__adr[5]".
fn resolve_bus_signal(
    aig: &AIG,
    netlistdb: &NetlistDB,
    script: &FlattenedScriptV1,
    base_pattern: &str,
    bit: usize,
) -> u32 {
    let pattern = format!("{}[{}]", base_pattern, bit);
    resolve_signal_pos(aig, netlistdb, script, &pattern)
}

/// Build WbTraceParams by resolving all bus signal names to state positions.
fn build_wb_trace_params(
    aig: &AIG,
    netlistdb: &NetlistDB,
    script: &FlattenedScriptV1,
) -> WbTraceParams {
    let mut params = WbTraceParams {
        ibus_cyc_pos: 0xFFFFFFFF,
        ibus_stb_pos: 0xFFFFFFFF,
        ibus_adr_pos: [0xFFFFFFFF; WB_TRACE_MAX_ADR_BITS],
        ibus_rdata_pos: [0xFFFFFFFF; WB_TRACE_MAX_DAT_BITS],
        dbus_cyc_pos: 0xFFFFFFFF,
        dbus_stb_pos: 0xFFFFFFFF,
        dbus_we_pos: 0xFFFFFFFF,
        dbus_adr_pos: [0xFFFFFFFF; WB_TRACE_MAX_ADR_BITS],
        spiflash_ack_pos: 0xFFFFFFFF,
        sram_ack_pos: 0xFFFFFFFF,
        csr_ack_pos: 0xFFFFFFFF,
        has_trace: 0,
    };

    // ibus (instruction bus)
    params.ibus_cyc_pos = resolve_signal_pos(aig, netlistdb, script, "cpu.fetch.ibus__cyc");
    params.ibus_stb_pos = resolve_signal_pos(aig, netlistdb, script, "cpu.fetch.ibus__stb");
    for i in 0..WB_TRACE_MAX_ADR_BITS {
        params.ibus_adr_pos[i] =
            resolve_bus_signal(aig, netlistdb, script, "cpu.fetch.ibus__adr", i);
    }
    for i in 0..WB_TRACE_MAX_DAT_BITS {
        params.ibus_rdata_pos[i] =
            resolve_bus_signal(aig, netlistdb, script, "cpu.fetch.ibus_rdata", i);
    }

    // dbus (data bus)
    params.dbus_cyc_pos = resolve_signal_pos(aig, netlistdb, script, "cpu.loadstore.dbus__cyc");
    params.dbus_stb_pos = resolve_signal_pos(aig, netlistdb, script, "cpu.loadstore.dbus__stb");
    params.dbus_we_pos = resolve_signal_pos(aig, netlistdb, script, "cpu.loadstore.dbus__we");
    for i in 0..WB_TRACE_MAX_ADR_BITS {
        params.dbus_adr_pos[i] =
            resolve_bus_signal(aig, netlistdb, script, "cpu.loadstore.dbus__adr", i);
    }

    // peripheral acks
    params.spiflash_ack_pos =
        resolve_signal_pos(aig, netlistdb, script, "spiflash.ctrl.wb_bus__ack");
    params.sram_ack_pos = resolve_signal_pos(aig, netlistdb, script, "sram.wb_bus__ack");
    params.csr_ack_pos = resolve_signal_pos(aig, netlistdb, script, "wb_to_csr.wb_bus__ack");

    // Check if we resolved any signals
    let found = [
        params.ibus_cyc_pos,
        params.ibus_stb_pos,
        params.dbus_cyc_pos,
    ]
    .iter()
    .filter(|&&p| p != 0xFFFFFFFF)
    .count();
    if found > 0 {
        params.has_trace = 1;
        let ibus_adr_count = params
            .ibus_adr_pos
            .iter()
            .filter(|&&p| p != 0xFFFFFFFF)
            .count();
        let ibus_rdata_count = params
            .ibus_rdata_pos
            .iter()
            .filter(|&&p| p != 0xFFFFFFFF)
            .count();
        let dbus_adr_count = params
            .dbus_adr_pos
            .iter()
            .filter(|&&p| p != 0xFFFFFFFF)
            .count();
        clilog::info!(
            "WB trace: ibus_cyc={} ibus_stb={} ibus_adr={}/30 ibus_rdata={}/32",
            params.ibus_cyc_pos != 0xFFFFFFFF,
            params.ibus_stb_pos != 0xFFFFFFFF,
            ibus_adr_count,
            ibus_rdata_count
        );
        clilog::info!(
            "WB trace: dbus_cyc={} dbus_stb={} dbus_we={} dbus_adr={}/30",
            params.dbus_cyc_pos != 0xFFFFFFFF,
            params.dbus_stb_pos != 0xFFFFFFFF,
            params.dbus_we_pos != 0xFFFFFFFF,
            dbus_adr_count
        );
        clilog::info!(
            "WB trace: spiflash_ack={} sram_ack={} csr_ack={}",
            params.spiflash_ack_pos != 0xFFFFFFFF,
            params.sram_ack_pos != 0xFFFFFFFF,
            params.csr_ack_pos != 0xFFFFFFFF
        );
    } else {
        clilog::info!("WB trace: no bus signals found in netlist, tracing disabled");
    }

    params
}

/// Write flash data input to GPIO state.
fn set_flash_din(state: &mut [u32], gpio_map: &GpioMapping, d0_gpio: usize, din: u8) {
    for i in 0..4 {
        if let Some(&pos) = gpio_map.input_bits.get(&(d0_gpio + i)) {
            set_bit(state, pos, (din >> i) & 1);
        }
    }
}

// ── BitOp Builder ────────────────────────────────────────────────────────────

/// Build the BitOp array for a falling edge state_prep.
/// Sets: clock=0, negedge_flag=1, clears: posedge_flag, plus reset.
/// Flash D_IN is now handled by gpu_apply_flash_din kernel.
fn build_falling_edge_ops(
    gpio_map: &GpioMapping,
    clock_gpio: usize,
    reset_gpio: usize,
    reset_val: u8,
    constant_inputs: &HashMap<String, u8>,
    constant_ports: &HashMap<String, u8>,
) -> Vec<BitOp> {
    let mut ops = Vec::new();
    // Clock = 0
    ops.push(BitOp {
        position: gpio_map.input_bits[&clock_gpio],
        value: 0,
    });
    // Reset
    ops.push(BitOp {
        position: gpio_map.input_bits[&reset_gpio],
        value: reset_val as u32,
    });
    // Negedge flag = 1 (all domains)
    for pos in gpio_map.all_negedge_flag_bits() {
        ops.push(BitOp {
            position: pos,
            value: 1,
        });
    }
    // Posedge flag = 0 (all domains)
    for pos in gpio_map.all_posedge_flag_bits() {
        ops.push(BitOp {
            position: pos,
            value: 0,
        });
    }
    // Constant inputs
    for (gpio_str, val) in constant_inputs {
        if let Ok(gpio_idx) = gpio_str.parse::<usize>() {
            if let Some(&pos) = gpio_map.input_bits.get(&gpio_idx) {
                ops.push(BitOp {
                    position: pos,
                    value: *val as u32,
                });
            }
        }
    }
    // Named constant ports (e.g. por_l, resetb_h)
    for (port_name, val) in constant_ports {
        if let Some(&pos) = gpio_map.named_input_bits.get(port_name) {
            ops.push(BitOp {
                position: pos,
                value: *val as u32,
            });
            clilog::debug!("constant_port '{}' → pos {} = {}", port_name, pos, val);
        } else {
            clilog::warn!(
                "constant_port '{}' not found in named_input_bits",
                port_name
            );
        }
    }
    ops
}

/// Build the BitOp array for a rising edge state_prep.
/// Sets: clock=1, posedge_flag=1, clears: negedge_flag, plus reset.
/// Flash D_IN is now handled by gpu_apply_flash_din kernel.
fn build_rising_edge_ops(
    gpio_map: &GpioMapping,
    clock_gpio: usize,
    reset_gpio: usize,
    reset_val: u8,
    constant_inputs: &HashMap<String, u8>,
    constant_ports: &HashMap<String, u8>,
) -> Vec<BitOp> {
    let mut ops = Vec::new();
    // Clock = 1
    ops.push(BitOp {
        position: gpio_map.input_bits[&clock_gpio],
        value: 1,
    });
    // Reset
    ops.push(BitOp {
        position: gpio_map.input_bits[&reset_gpio],
        value: reset_val as u32,
    });
    // Posedge flag = 1 (all domains)
    for pos in gpio_map.all_posedge_flag_bits() {
        ops.push(BitOp {
            position: pos,
            value: 1,
        });
    }
    // Negedge flag = 0 (all domains)
    for pos in gpio_map.all_negedge_flag_bits() {
        ops.push(BitOp {
            position: pos,
            value: 0,
        });
    }
    // Constant inputs
    for (gpio_str, val) in constant_inputs {
        if let Ok(gpio_idx) = gpio_str.parse::<usize>() {
            if let Some(&pos) = gpio_map.input_bits.get(&gpio_idx) {
                ops.push(BitOp {
                    position: pos,
                    value: *val as u32,
                });
            }
        }
    }
    // Named constant ports
    for (port_name, val) in constant_ports {
        if let Some(&pos) = gpio_map.named_input_bits.get(port_name) {
            ops.push(BitOp {
                position: pos,
                value: *val as u32,
            });
        }
    }
    ops
}

// ── Multi-Clock Scheduler ────────────────────────────────────────────────────

/// Per-tick edge activity for each clock domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TickEdges {
    /// For each domain index: (has_falling_edge, has_rising_edge).
    domain_edges: Vec<(bool, bool)>,
}

/// Schedules clock edges across multiple domains at GCD granularity.
///
/// For N clock domains with half-periods H1, H2, ..., the GCD tick is
/// `gcd(H1, H2, ...)` and the schedule repeats every `lcm(H1, H2, ...) / gcd` ticks.
/// Each tick in the schedule records which domains have falling/rising edges.
struct MultiClockScheduler {
    /// GCD of all half-periods in picoseconds.
    #[allow(dead_code)]
    gcd_ps: u64,
    /// One entry per GCD tick in the LCM cycle.
    schedule: Vec<TickEdges>,
}

/// Compute GCD of two numbers.
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Compute LCM of two numbers.
fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// A clock domain's timing parameters for scheduling.
struct ClockDomainTiming {
    /// Half-period in picoseconds (period_ps / 2).
    half_period_ps: u64,
    /// Phase offset in picoseconds.
    phase_offset_ps: u64,
    /// Index into GpioMapping::clock_domains.
    domain_index: usize,
}

impl MultiClockScheduler {
    /// Build a scheduler from clock domain timings.
    ///
    /// For a single clock with no phase offset, produces a schedule of length 2
    /// alternating between falling (posedge_flag=0) and rising (posedge_flag=1)
    /// edges.
    fn new(timings: &[ClockDomainTiming]) -> Self {
        assert!(!timings.is_empty(), "At least one clock domain is required");

        // Compute GCD of all half-periods and phase offsets
        let mut gcd_ps = timings[0].half_period_ps;
        for t in &timings[1..] {
            gcd_ps = gcd(gcd_ps, t.half_period_ps);
        }
        // Include phase offsets in GCD (non-zero offsets need finer granularity)
        for t in timings {
            if t.phase_offset_ps > 0 {
                gcd_ps = gcd(gcd_ps, t.phase_offset_ps);
            }
        }

        // Compute LCM of all full periods (= 2 * half_period)
        // We need a full period to capture both falling and rising edges.
        let mut lcm_ps = timings[0].half_period_ps * 2;
        for t in &timings[1..] {
            lcm_ps = lcm(lcm_ps, t.half_period_ps * 2);
        }

        let schedule_len = (lcm_ps / gcd_ps) as usize;
        assert!(
            schedule_len <= 1_000_000,
            "Multi-clock schedule too large ({} ticks). \
             Clock periods may not be commensurable at this resolution.",
            schedule_len
        );

        let num_domains = timings.len();
        let mut schedule = Vec::with_capacity(schedule_len);

        for tick in 0..schedule_len {
            let tick_ps = tick as u64 * gcd_ps;
            let mut domain_edges = vec![(false, false); num_domains];

            for (i, timing) in timings.iter().enumerate() {
                let hp = timing.half_period_ps;
                let offset = timing.phase_offset_ps;
                // Adjust tick_ps by phase offset
                if tick_ps < offset {
                    continue;
                }
                let adjusted = tick_ps - offset;
                if adjusted % hp != 0 {
                    continue;
                }
                let edge_count = adjusted / hp;
                // Even edge_count = falling, odd = rising
                if edge_count % 2 == 0 {
                    domain_edges[i].0 = true; // falling edge
                } else {
                    domain_edges[i].1 = true; // rising edge
                }
            }

            schedule.push(TickEdges { domain_edges });
        }

        clilog::info!(
            "MultiClockScheduler: gcd={}ps, schedule_len={} ticks, {} domains",
            gcd_ps,
            schedule_len,
            num_domains
        );

        Self { gcd_ps, schedule }
    }

    /// Build the falling-edge BitOps for a given schedule tick.
    ///
    /// Only toggles flags for domains that have a falling edge at this tick.
    /// Also sets clock input port values and constants.
    fn build_tick_fall_ops(
        &self,
        tick_idx: usize,
        gpio_map: &GpioMapping,
        reset_gpio: usize,
        reset_val: u8,
        constant_inputs: &HashMap<String, u8>,
        constant_ports: &HashMap<String, u8>,
    ) -> Vec<BitOp> {
        let tick = &self.schedule[tick_idx % self.schedule.len()];
        let mut ops = Vec::new();

        // Reset
        ops.push(BitOp {
            position: gpio_map.input_bits[&reset_gpio],
            value: reset_val as u32,
        });

        // Per-domain clock flags
        for (i, domain) in gpio_map.clock_domains.iter().enumerate() {
            let (has_fall, _has_rise) = tick.domain_edges[i];

            // Set clock input port value: falling = 0
            if has_fall {
                if let Some(pos) = domain.clock_input_pos {
                    ops.push(BitOp {
                        position: pos,
                        value: 0,
                    });
                }
            }

            // Set negedge flags = 1 for domains with falling edge, 0 otherwise
            for &pos in &domain.negedge_flag_bits {
                ops.push(BitOp {
                    position: pos,
                    value: if has_fall { 1 } else { 0 },
                });
            }
            // Clear posedge flags for domains with falling edge
            for &pos in &domain.posedge_flag_bits {
                ops.push(BitOp {
                    position: pos,
                    value: 0,
                });
            }
        }

        // Constant inputs
        for (gpio_str, val) in constant_inputs {
            if let Ok(gpio_idx) = gpio_str.parse::<usize>() {
                if let Some(&pos) = gpio_map.input_bits.get(&gpio_idx) {
                    ops.push(BitOp {
                        position: pos,
                        value: *val as u32,
                    });
                }
            }
        }
        // Named constant ports
        for (port_name, val) in constant_ports {
            if let Some(&pos) = gpio_map.named_input_bits.get(port_name) {
                ops.push(BitOp {
                    position: pos,
                    value: *val as u32,
                });
            }
        }

        ops
    }

    /// Build the rising-edge BitOps for a given schedule tick.
    fn build_tick_rise_ops(
        &self,
        tick_idx: usize,
        gpio_map: &GpioMapping,
        reset_gpio: usize,
        reset_val: u8,
        constant_inputs: &HashMap<String, u8>,
        constant_ports: &HashMap<String, u8>,
    ) -> Vec<BitOp> {
        let tick = &self.schedule[tick_idx % self.schedule.len()];
        let mut ops = Vec::new();

        // Reset
        ops.push(BitOp {
            position: gpio_map.input_bits[&reset_gpio],
            value: reset_val as u32,
        });

        // Per-domain clock flags
        for (i, domain) in gpio_map.clock_domains.iter().enumerate() {
            let (_has_fall, has_rise) = tick.domain_edges[i];

            // Set clock input port value: rising = 1
            if has_rise {
                if let Some(pos) = domain.clock_input_pos {
                    ops.push(BitOp {
                        position: pos,
                        value: 1,
                    });
                }
            }

            // Set posedge flags = 1 for domains with rising edge, 0 otherwise
            for &pos in &domain.posedge_flag_bits {
                ops.push(BitOp {
                    position: pos,
                    value: if has_rise { 1 } else { 0 },
                });
            }
            // Clear negedge flags for domains with rising edge
            for &pos in &domain.negedge_flag_bits {
                ops.push(BitOp {
                    position: pos,
                    value: 0,
                });
            }
        }

        // Constant inputs
        for (gpio_str, val) in constant_inputs {
            if let Ok(gpio_idx) = gpio_str.parse::<usize>() {
                if let Some(&pos) = gpio_map.input_bits.get(&gpio_idx) {
                    ops.push(BitOp {
                        position: pos,
                        value: *val as u32,
                    });
                }
            }
        }
        // Named constant ports
        for (port_name, val) in constant_ports {
            if let Some(&pos) = gpio_map.named_input_bits.get(port_name) {
                ops.push(BitOp {
                    position: pos,
                    value: *val as u32,
                });
            }
        }

        ops
    }
}

/// Create a Metal buffer containing a StatePrepParams struct.
fn create_prep_params_buffer(
    device: &metal::Device,
    state_size: u32,
    num_ops: u32,
    num_monitors: u32,
    tick_number: u32,
) -> metal::Buffer {
    let buf = device.new_buffer(
        std::mem::size_of::<StatePrepParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::write(
            buf.contents() as *mut StatePrepParams,
            StatePrepParams {
                state_size,
                num_ops,
                num_monitors,
                tick_number,
            },
        );
    }
    buf
}

/// Create a Metal buffer containing a BitOp array.
fn create_ops_buffer(device: &metal::Device, ops: &[BitOp]) -> metal::Buffer {
    let size = if ops.is_empty() {
        std::mem::size_of::<BitOp>() as u64 // minimum 1 element
    } else {
        (ops.len() * std::mem::size_of::<BitOp>()) as u64
    };
    let buf = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
    if !ops.is_empty() {
        unsafe {
            std::ptr::copy_nonoverlapping(ops.as_ptr(), buf.contents() as *mut BitOp, ops.len());
        }
    }
    buf
}

// ── CPU baseline for verification ────────────────────────────────────────────

/// Direct AIG evaluator: evaluates all AIG pins by traversing the AND-gate
/// tree directly, bypassing the boomerang/FlattenedScript entirely.
/// Returns a map from AIG pin → value (0 or 1).
///
/// This is used to diagnose whether the AIG construction is correct
/// (if direct eval matches FlattenedScript → bug is in AIG; if they differ → bug in flattening).
fn eval_aig_direct(aig: &AIG, input_state: &[u32], script: &FlattenedScriptV1) -> Vec<u8> {
    let mut pin_values = vec![0u8; aig.num_aigpins + 1];
    // pin 0 = constant 0 (Tie0)

    // Initialize all primary inputs from input_state using input_map
    for (&aigpin, &pos) in &script.input_map {
        let word = (pos / 32) as usize;
        let bit = pos & 31;
        if word < input_state.len() {
            pin_values[aigpin] = ((input_state[word] >> bit) & 1) as u8;
        }
    }

    // Topological traversal using topo_traverse_generic
    // endpoints=None means traverse ALL AIG pins
    // is_primary_input = set of pins that are inputs (InputPort, InputClockFlag, DFF Q, SRAM)
    let primary_input_set: indexmap::IndexSet<usize> = script.input_map.keys().copied().collect();
    let order = aig.topo_traverse_generic(None, Some(&primary_input_set));

    // Evaluate each AND gate in topological order
    for &pin in &order {
        if primary_input_set.contains(&pin) {
            // Already initialized from input_state
            continue;
        }
        if let DriverType::AndGate(a_iv, b_iv) = aig.drivers[pin] {
            let a_pin = a_iv >> 1;
            let a_inv = (a_iv & 1) as u8;
            let b_pin = b_iv >> 1;
            let b_inv = (b_iv & 1) as u8;
            let a_val = if a_pin == 0 {
                0
            } else {
                pin_values[a_pin] ^ a_inv
            };
            let b_val = if b_pin == 0 {
                0
            } else {
                pin_values[b_pin] ^ b_inv
            };
            pin_values[pin] = a_val & b_val;
        }
        // Non-AndGate, non-primary-input pins (shouldn't happen in traversal) stay 0
    }

    pin_values
}

/// Compare direct AIG evaluation with FlattenedScript output at a given tick.
/// Prints diagnostic information about any mismatches.
fn compare_aig_vs_flattened(
    aig: &AIG,
    input_state: &[u32],
    output_state: &[u32],
    script: &FlattenedScriptV1,
    state_size: usize,
    tick: usize,
    netlistdb: Option<&NetlistDB>,
) {
    let pin_values = eval_aig_direct(aig, input_state, script);

    // Compare DFF D inputs: direct AIG eval vs FlattenedScript output
    let mut dff_mismatch_count = 0;
    let mut dff_match_count = 0;
    let mut dff_en_active_count = 0;
    let mut dff_d_ones_direct = 0;
    let mut dff_d_ones_flat = 0;
    let mut first_mismatches: Vec<String> = Vec::new();

    for (_cellid, dff) in aig.dffs.iter() {
        let d_pin = dff.d_iv >> 1;
        let d_inv = (dff.d_iv & 1) as u8;
        let en_pin = dff.en_iv >> 1;
        let en_inv = (dff.en_iv & 1) as u8;

        // Direct AIG evaluation
        let d_val_direct = if d_pin == 0 {
            0
        } else {
            pin_values[d_pin] ^ d_inv
        };
        let en_val_direct = if en_pin == 0 {
            0
        } else {
            pin_values[en_pin] ^ en_inv
        };

        if en_val_direct != 0 {
            dff_en_active_count += 1;
        }
        if d_val_direct != 0 {
            dff_d_ones_direct += 1;
        }

        // FlattenedScript result: read from output_state
        if let Some(&pos) = script.output_map.get(&dff.d_iv) {
            let word = (pos / 32) as usize;
            let bit = pos & 31;
            let flat_val = if word < state_size {
                ((output_state[word] >> bit) & 1) as u8
            } else {
                0
            };
            if flat_val != 0 {
                dff_d_ones_flat += 1;
            }

            if d_val_direct != flat_val {
                dff_mismatch_count += 1;
                if first_mismatches.len() < 10 {
                    first_mismatches.push(format!(
                        "DFF d_iv={} pos={} en={}: direct={} flat={} (d_pin={} en_pin={} en_val={})",
                        dff.d_iv, pos, dff.en_iv, d_val_direct, flat_val, d_pin, en_pin, en_val_direct
                    ));
                }
            } else {
                dff_match_count += 1;
            }
        }
    }

    // Also compare primary outputs
    let mut po_mismatch_count = 0;
    for (&aigpin_iv, &pos) in &script.output_map {
        let pin = aigpin_iv >> 1;
        let inv = (aigpin_iv & 1) as u8;
        let direct_val = if pin == 0 { 0u8 } else { pin_values[pin] ^ inv };

        let word = (pos / 32) as usize;
        let bit = pos & 31;
        let flat_val = if word < state_size {
            ((output_state[word] >> bit) & 1) as u8
        } else {
            0
        };

        if direct_val != flat_val {
            po_mismatch_count += 1;
        }
    }

    eprintln!("=== AIG vs FlattenedScript comparison at tick {} ===", tick);
    eprintln!(
        "  DFFs: {} match, {} MISMATCH (out of {} total)",
        dff_match_count,
        dff_mismatch_count,
        aig.dffs.len()
    );
    eprintln!(
        "  DFF en_active={}, d_ones_direct={}, d_ones_flat={}",
        dff_en_active_count, dff_d_ones_direct, dff_d_ones_flat
    );
    eprintln!("  All output_map entries: {} mismatches", po_mismatch_count);
    for m in &first_mismatches {
        eprintln!("  MISMATCH: {}", m);
    }

    // Also check: how many DFF Q values are 1 in the input state?
    let mut q_ones = 0;
    let mut q_hash: u64 = 0;
    let mut q_one_entries: Vec<(usize, u32, String)> = Vec::new(); // (cellid, pos, name)
    for (&cellid, dff) in aig.dffs.iter() {
        if let Some(&pos) = script.input_map.get(&dff.q) {
            let word = (pos / 32) as usize;
            let bit = pos & 31;
            if word < input_state.len() && ((input_state[word] >> bit) & 1) != 0 {
                q_ones += 1;
                let name = if let Some(ndb) = netlistdb {
                    format!("{:?}", ndb.cellnames[cellid])
                } else {
                    format!("cell_{}", cellid)
                };
                q_one_entries.push((cellid, pos, name));
                q_hash = q_hash
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(pos as u64);
            }
        }
    }
    q_one_entries.sort_by_key(|&(cellid, _, _)| cellid);
    eprintln!(
        "  DFF Q values = 1 in input_state: {}/{} (hash=0x{:016X})",
        q_ones,
        aig.dffs.len(),
        q_hash
    );
    if q_ones <= 100 {
        for (cellid, pos, name) in &q_one_entries {
            eprintln!("  Q=1: cell {} pos={} = {}", cellid, pos, name);
        }
    }
}

/// CPU prototype partition executor.
/// Used for --check-with-cpu verification.
fn simulate_block_v1(
    script: &[u32],
    input_state: &[u32],
    output_state: &mut [u32],
    sram_data: &mut [u32],
) {
    simulate_block_v1_inner(script, input_state, output_state, sram_data, false);
}

fn simulate_block_v1_diag(
    script: &[u32],
    input_state: &[u32],
    output_state: &mut [u32],
    sram_data: &mut [u32],
) {
    simulate_block_v1_inner(script, input_state, output_state, sram_data, true);
}

fn simulate_block_v1_inner(
    script: &[u32],
    input_state: &[u32],
    output_state: &mut [u32],
    sram_data: &mut [u32],
    diag: bool,
) {
    use crate::aigpdk::AIGPDK_SRAM_SIZE;
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
                if mask == 0 {
                    continue;
                }
                let value = match (idx >> 31) != 0 {
                    false => input_state[idx as usize],
                    true => output_state[(idx ^ (1 << 31)) as usize],
                };
                while mask != 0 {
                    cur_state <<= 1;
                    let lowbit = mask & (-(mask as i32)) as u32;
                    if (value & lowbit) != 0 {
                        cur_state |= 1;
                    }
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
                        hier_inputs[i] |=
                            (state[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1)
                                << (k * 2);
                        hier_inputs[i] |=
                            (state[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1)
                                << (k * 2 + 1);
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
                    sram_duplicate_perm[i as usize] |=
                        (writeouts[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1)
                            << (k * 2);
                    sram_duplicate_perm[i as usize] |=
                        (writeouts[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1)
                            << (k * 2 + 1);
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
            ram[port_w_addr_iv as usize] =
                (w0 & !port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
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
                    clken_perm[i as usize] |= (writeouts_for_clken
                        [(t_shuffle_1_idx >> 5) as usize]
                        >> (t_shuffle_1_idx & 31)
                        & 1)
                        << (k * 2);
                    clken_perm[i as usize] |= (writeouts_for_clken
                        [(t_shuffle_2_idx >> 5) as usize]
                        >> (t_shuffle_2_idx & 31)
                        & 1)
                        << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..num_ios as usize {
            clken_perm[i] &= !script[script_pi + i * 4 + 1];
            clken_perm[i] ^= script[script_pi + i * 4];
            writeouts[i] ^= script[script_pi + i * 4 + 2];
        }

        if diag {
            // Count non-zero clken_perm bits across all DFF positions
            let clken_nonzero: usize = clken_perm[..num_ios as usize]
                .iter()
                .map(|c| c.count_ones() as usize)
                .sum();
            let clken_set0_all_ones: usize = (0..num_ios as usize)
                .filter(|&i| script[script_pi + i * 4 + 1] == u32::MAX)
                .count();
            let clken_set0_zero: usize = (0..num_ios as usize)
                .filter(|&i| script[script_pi + i * 4 + 1] == 0)
                .count();
            let changed_bits: usize = (0..num_ios as usize)
                .map(|i| {
                    let old_wo = input_state[io_offset as usize + i];
                    let clken = clken_perm[i];
                    let wo = (old_wo & !clken) | (writeouts[i] & clken);
                    (old_wo ^ wo).count_ones() as usize
                })
                .sum();
            eprintln!("  DIAG partition: num_ios={}, io_offset={}, clken_nonzero_bits={}, set0_all={}, set0_zero={}, changed_bits={}",
                num_ios, io_offset, clken_nonzero, clken_set0_all_ones, clken_set0_zero, changed_bits);
            // Show first few positions with non-zero clken_perm
            let mut shown = 0;
            for i in 0..num_ios as usize {
                if clken_perm[i] != 0 && shown < 3 {
                    let set0 = script[script_pi + i * 4 + 1];
                    let inv = script[script_pi + i * 4];
                    eprintln!("    clken_perm[{}]=0x{:08X} (set0=0x{:08X}, inv=0x{:08X}, writeout=0x{:08X})",
                        i, clken_perm[i], set0, inv, writeouts[i]);
                    shown += 1;
                }
            }
            // Also show posedge flag in input_state
            eprintln!(
                "    input_state[0]=0x{:08X} (bit0=posedge_flag={})",
                input_state[0],
                input_state[0] & 1
            );
        }

        script_pi += 256 * 4;

        for i in 0..num_ios {
            let old_wo = input_state[(io_offset + i) as usize];
            let clken = clken_perm[i as usize];
            let wo = (old_wo & !clken) | (writeouts[i as usize] & clken);
            output_state[(io_offset + i) as usize] = wo;
        }

        if is_last_part != 0 {
            break;
        }
    }
    assert_eq!(script_pi, script.len());
}

// ── Public Entry Point ───────────────────────────────────────────────────────

/// Run a GPU co-simulation with testbench config.
///
/// `design` should already have basic SDF loaded if `--sdf` was passed on CLI.
/// This function also checks `config.timing` for additional SDF configuration.
pub fn run_cosim(
    design: &mut LoadedDesign,
    config: &TestbenchConfig,
    opts: &CosimOpts,
    timing_constraints: &Option<Vec<u32>>,
) -> CosimResult {
    // Load SDF from testbench config if not already loaded via CLI --sdf
    if !design.script.timing_enabled {
        let sdf_path_from_config = config
            .timing
            .as_ref()
            .map(|t| std::path::PathBuf::from(&t.sdf_file));
        if let Some(ref sdf_path) = sdf_path_from_config {
            let sdf_corner_str = config
                .timing
                .as_ref()
                .map(|t| t.sdf_corner.as_str())
                .unwrap_or("typ");
            let clock_ps = config
                .timing
                .as_ref()
                .map(|t| t.clock_period_ps)
                .or(config.clock_period_ps)
                .unwrap_or(25000);
            setup::load_sdf(
                &mut design.script,
                &design.aig,
                &design.netlistdb,
                sdf_path,
                sdf_corner_str,
                false,
                Some(clock_ps),
            );
        }
    }

    let max_ticks = opts.max_cycles.unwrap_or(config.num_cycles);
    let script = &design.script;
    let aig = &design.aig;
    let netlistdb = &design.netlistdb;
    let num_blocks = script.num_blocks;
    let num_major_stages = script.num_major_stages;
    let state_size = script.reg_io_state_size as usize;

    // ── Build GPIO mapping ───────────────────────────────────────────────

    let gpio_map = build_gpio_mapping(aig, netlistdb, script, config.port_mapping.as_ref());

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
                clilog::info!(
                    "Flash D{} input GPIO {} -> state pos {}",
                    i,
                    gpio,
                    gpio_map.input_bits[&gpio]
                );
            }
            if gpio_map.output_bits.contains_key(&gpio) {
                clilog::info!(
                    "Flash D{} output GPIO {} -> state pos {}",
                    i,
                    gpio,
                    gpio_map.output_bits[&gpio]
                );
            }
        }
    }

    // ── Diagnostic: resolve key internal signals for debugging ──────────

    let diag_signals = [
        "fetch_enable",
        "ibus__cyc",
        "ibus__stb",
        "dbus__cyc",
        "spiflash",
        "flash_clk",
        "flash_csn",
    ];
    for sig in &diag_signals {
        let pos = resolve_signal_pos(aig, netlistdb, script, sig);
        if pos != 0xFFFFFFFF {
            clilog::info!("Diagnostic signal '{}' → output state pos {}", sig, pos);
        } else {
            clilog::debug!("Diagnostic signal '{}' not found in output_map", sig);
        }
    }

    // ── Initialize peripheral models (CPU-side, kept for --check-with-cpu) ──

    let _flash: Option<CppSpiFlash> = if let Some(ref flash_cfg) = config.flash {
        let mut fl = CppSpiFlash::new(16 * 1024 * 1024);
        fl.set_verbose(opts.flash_verbose);
        let firmware_path = std::path::Path::new(&flash_cfg.firmware);
        match fl.load_firmware(firmware_path, flash_cfg.firmware_offset) {
            Ok(size) => clilog::info!(
                "Loaded {} bytes firmware at offset 0x{:X}",
                size,
                flash_cfg.firmware_offset
            ),
            Err(e) => panic!("Failed to load firmware: {}", e),
        }
        Some(fl)
    } else {
        None
    };

    // CLI --clock-period overrides config file clock_period_ps; default 1000ps (1GHz) if neither set
    let clock_period_ps = opts.clock_period.or(config.clock_period_ps).unwrap_or(1000);
    let clock_hz = 1_000_000_000_000u64 / clock_period_ps;
    clilog::info!(
        "Clock period: {} ps ({} MHz), UART cycles_per_bit: {}",
        clock_period_ps,
        clock_hz / 1_000_000,
        clock_hz / config.uart.as_ref().map(|u| u.baud_rate).unwrap_or(115200) as u64
    );
    let uart_baud = config.uart.as_ref().map(|u| u.baud_rate).unwrap_or(115200);
    let uart_tx_gpio = config.uart.as_ref().map(|u| u.tx_gpio);

    // ── Initialize Metal simulator and GPU state buffers ─────────────────

    let timer_init = clilog::stimer!("init_gpu");
    let simulator = MetalSimulator::new(num_major_stages);

    // States: [input state (state_size)] [output state (state_size)]
    let states_buffer = simulator.device.new_buffer(
        (2 * state_size * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let states: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(states_buffer.contents() as *mut u32, 2 * state_size)
    };
    states.fill(0);

    // SRAM storage
    let sram_data_buffer = simulator.device.new_buffer(
        (script.sram_storage_size as usize * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sram_data: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(
            sram_data_buffer.contents() as *mut u32,
            script.sram_storage_size as usize,
        )
    };
    sram_data.fill(0);
    let _ = sram_data;

    // Initialize: set reset active
    let reset_val = if config.reset_active_high { 1u8 } else { 0u8 };
    set_bit(
        &mut states[..state_size],
        gpio_map.input_bits[&reset_gpio],
        reset_val,
    );
    clilog::info!(
        "Initial state: reset GPIO {} = {} (active)",
        reset_gpio,
        reset_val
    );

    // Initialize constant ports (e.g. por_l=1, resetb_h=1 for Caravel wrapper)
    for (port_name, val) in &config.constant_ports {
        if let Some(&pos) = gpio_map.named_input_bits.get(port_name) {
            set_bit(&mut states[..state_size], pos, *val);
            clilog::info!(
                "Initial state: port '{}' = {} (pos {})",
                port_name,
                val,
                pos
            );
        } else {
            clilog::warn!("constant_port '{}' not found in named inputs", port_name);
        }
    }

    // Set flash D_IN defaults (high = no data) — initial state before GPU flash takes over
    if let Some(ref flash_cfg) = config.flash {
        set_flash_din(
            &mut states[..state_size],
            &gpio_map,
            flash_cfg.d0_gpio,
            0x0F,
        );
    }

    // Create Metal buffers for script data (read-only, use UVec's Metal path)
    let device = Device::Metal(0);
    let blocks_start_ptr = script.blocks_start.as_uptr(device);
    let blocks_data_ptr = script.blocks_data.as_uptr(device);

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

    // Event buffer (for $stop/$finish/assertions)
    let event_buffer = Box::new(crate::event_buffer::EventBuffer::new());
    let event_buffer_ptr = Box::into_raw(event_buffer);
    let event_buffer_metal = simulator.device.new_buffer_with_bytes_no_copy(
        event_buffer_ptr as *const _,
        std::mem::size_of::<crate::event_buffer::EventBuffer>() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );

    // Timing constraint buffer for GPU-side setup/hold checking.
    let timing_constraints_buffer = timing_constraints.as_ref().map(|buf| {
        simulator.device.new_buffer_with_data(
            buf.as_ptr() as *const _,
            (buf.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    clilog::finish!(timer_init);

    // ── Build GPU-side state_prep + IO model buffers ──────────────────────

    let timer_prep = clilog::stimer!("build_state_prep_buffers");
    let reset_cycles = config.reset_cycles;

    // Initial reset value
    let reset_val_active = if config.reset_active_high { 1u8 } else { 0u8 };
    let reset_val_inactive = if config.reset_active_high { 0u8 } else { 1u8 };

    // ── Build multi-clock scheduler ────────────────────────────────────────
    let effective_clocks = config.effective_clocks();
    let clock_timings: Vec<ClockDomainTiming> = effective_clocks
        .iter()
        .enumerate()
        .filter_map(|(cfg_idx, clk_cfg)| {
            // Find the clock domain in gpio_map that matches this clock's GPIO
            let domain_idx = gpio_map
                .clock_domains
                .iter()
                .position(|d| d.clock_gpio == Some(clk_cfg.gpio));
            if let Some(di) = domain_idx {
                Some(ClockDomainTiming {
                    half_period_ps: clk_cfg.period_ps / 2,
                    phase_offset_ps: clk_cfg.phase_offset_ps,
                    domain_index: di,
                })
            } else {
                clilog::warn!(
                    "Clock config[{}] gpio={} ({:?}): no matching clock domain found, skipping",
                    cfg_idx,
                    clk_cfg.gpio,
                    clk_cfg.name
                );
                None
            }
        })
        .collect();

    // If no clock domains were matched, fall back to legacy single-clock behavior
    // using all flag bits (same as before multi-clock support)
    let use_legacy_single_clock = clock_timings.is_empty();
    if use_legacy_single_clock {
        clilog::info!(
            "No clock domains matched from config; using legacy single-clock mode \
             (all {} posedge + {} negedge flags toggle together)",
            gpio_map.all_posedge_flag_bits().len(),
            gpio_map.all_negedge_flag_bits().len()
        );
    } else {
        for (i, clk_cfg) in effective_clocks.iter().enumerate() {
            if let Some(timing) = clock_timings.iter().find(|t| {
                gpio_map.clock_domains[t.domain_index].clock_gpio == Some(clk_cfg.gpio)
            }) {
                let domain = &gpio_map.clock_domains[timing.domain_index];
                clilog::info!(
                    "Clock '{}' (gpio {}, period {}ps, phase {}ps) → domain '{}' \
                     ({} posedge flags, {} negedge flags)",
                    clk_cfg.name.as_deref().unwrap_or("?"),
                    clk_cfg.gpio,
                    clk_cfg.period_ps,
                    clk_cfg.phase_offset_ps,
                    domain.name,
                    domain.posedge_flag_bits.len(),
                    domain.negedge_flag_bits.len()
                );
            }
            let _ = i; // suppress unused warning
        }
    }

    // Build per-schedule-tick BitOp buffers
    let schedule_buffers = if use_legacy_single_clock {
        // Legacy: single schedule entry, all flags toggle together
        let fall_ops = build_falling_edge_ops(
            &gpio_map,
            clock_gpio,
            reset_gpio,
            reset_val_active,
            &config.constant_inputs,
            &config.constant_ports,
        );
        let rise_ops = build_rising_edge_ops(
            &gpio_map,
            clock_gpio,
            reset_gpio,
            reset_val_active,
            &config.constant_inputs,
            &config.constant_ports,
        );
        let fall_len = fall_ops.len();
        let rise_len = rise_ops.len();
        let fall_ops_buf = create_ops_buffer(&simulator.device, &fall_ops);
        let rise_ops_buf = create_ops_buffer(&simulator.device, &rise_ops);
        let fall_params = create_prep_params_buffer(
            &simulator.device,
            state_size as u32,
            fall_ops.len() as u32,
            0,
            0,
        );
        let rise_params = create_prep_params_buffer(
            &simulator.device,
            state_size as u32,
            rise_ops.len() as u32,
            0,
            0,
        );
        ScheduleBuffers {
            tick_buffers: vec![(fall_params, fall_ops_buf, rise_params, rise_ops_buf)],
            fall_ops_lens: vec![fall_len],
            rise_ops_lens: vec![rise_len],
        }
    } else {
        let scheduler = MultiClockScheduler::new(&clock_timings);
        let mut tick_buffers = Vec::with_capacity(scheduler.schedule.len());
        let mut fall_ops_lens = Vec::with_capacity(scheduler.schedule.len());
        let mut rise_ops_lens = Vec::with_capacity(scheduler.schedule.len());

        for tick_idx in 0..scheduler.schedule.len() {
            let fall_ops = scheduler.build_tick_fall_ops(
                tick_idx,
                &gpio_map,
                reset_gpio,
                reset_val_active,
                &config.constant_inputs,
                &config.constant_ports,
            );
            let rise_ops = scheduler.build_tick_rise_ops(
                tick_idx,
                &gpio_map,
                reset_gpio,
                reset_val_active,
                &config.constant_inputs,
                &config.constant_ports,
            );
            let fall_len = fall_ops.len();
            let rise_len = rise_ops.len();
            let fall_ops_buf = create_ops_buffer(&simulator.device, &fall_ops);
            let rise_ops_buf = create_ops_buffer(&simulator.device, &rise_ops);
            let fall_params = create_prep_params_buffer(
                &simulator.device,
                state_size as u32,
                fall_ops.len() as u32,
                0,
                0,
            );
            let rise_params = create_prep_params_buffer(
                &simulator.device,
                state_size as u32,
                rise_ops.len() as u32,
                0,
                0,
            );
            tick_buffers.push((fall_params, fall_ops_buf, rise_params, rise_ops_buf));
            fall_ops_lens.push(fall_len);
            rise_ops_lens.push(rise_len);
        }

        clilog::info!(
            "Multi-clock schedule: {} ticks, {} Metal buffer sets",
            scheduler.schedule.len(),
            tick_buffers.len()
        );

        ScheduleBuffers {
            tick_buffers,
            fall_ops_lens,
            rise_ops_lens,
        }
    };

    // ── GPU Flash IO buffers ────────────────────────────────────────────

    // FlashState (shared, persistent across ticks)
    let flash_state_buffer = simulator.device.new_buffer(
        std::mem::size_of::<FlashState>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let fs = &mut *(flash_state_buffer.contents() as *mut FlashState);
        *fs = std::mem::zeroed();
        fs.data_width = 1; // SPI single-bit mode (reset by posedge_csn, but first tx has none)
        fs.prev_csn = 1; // CSN starts high (deselected)
        fs.model_prev_csn = 1; // Model internal edge detection starts high
        fs.d_i = 0x0F; // Flash output starts high
        fs.in_reset = 1; // Start in reset
                         // Verify write
        let verify = std::ptr::read_volatile(&fs.d_i);
        assert_eq!(
            verify, 0x0F,
            "FlashState.d_i not written correctly: got 0x{:02X}",
            verify
        );
        clilog::info!(
            "FlashState init: d_i=0x{:02X}, data_width={}, prev_csn={}, in_reset={}",
            fs.d_i,
            fs.data_width,
            fs.prev_csn,
            fs.in_reset
        );
    }

    // FlashDinParams (constant)
    let flash_din_params_buffer = simulator.device.new_buffer(
        std::mem::size_of::<FlashDinParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let p = &mut *(flash_din_params_buffer.contents() as *mut FlashDinParams);
        p.has_flash = if config.flash.is_some() { 1 } else { 0 };
        for i in 0..4 {
            p.d_in_pos[i] = config
                .flash
                .as_ref()
                .and_then(|f| gpio_map.input_bits.get(&(f.d0_gpio + i)).copied())
                .unwrap_or(0xFFFFFFFF);
        }
    }

    // FlashModelParams (constant)
    let flash_model_params_buffer = simulator.device.new_buffer(
        std::mem::size_of::<FlashModelParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let p = &mut *(flash_model_params_buffer.contents() as *mut FlashModelParams);
        p.state_size = state_size as u32;
        p.clk_out_pos = config
            .flash
            .as_ref()
            .and_then(|f| gpio_map.output_bits.get(&f.clk_gpio).copied())
            .unwrap_or(0);
        p.csn_out_pos = config
            .flash
            .as_ref()
            .and_then(|f| gpio_map.output_bits.get(&f.csn_gpio).copied())
            .unwrap_or(0);
        for i in 0..4 {
            p.d_out_pos[i] = config
                .flash
                .as_ref()
                .and_then(|f| gpio_map.output_bits.get(&(f.d0_gpio + i)).copied())
                .unwrap_or(0xFFFFFFFF);
        }
        p.flash_data_size = 16 * 1024 * 1024; // 16 MB
        clilog::info!("FlashModelParams: state_size={}, clk_out_pos={}, csn_out_pos={}, d_out_pos={:?}, flash_data_size={}",
            p.state_size, p.clk_out_pos, p.csn_out_pos, p.d_out_pos, p.flash_data_size);
    }

    // Log flash din params too
    unsafe {
        let p = &*(flash_din_params_buffer.contents() as *const FlashDinParams);
        clilog::info!(
            "FlashDinParams: has_flash={}, d_in_pos={:?}",
            p.has_flash,
            p.d_in_pos
        );
    }

    // Flash data buffer (16 MB, loaded with firmware)
    let flash_data_buffer = simulator
        .device
        .new_buffer(16 * 1024 * 1024, MTLResourceOptions::StorageModeShared);
    unsafe {
        // Fill with 0xFF (erased flash state)
        std::ptr::write_bytes(
            flash_data_buffer.contents() as *mut u8,
            0xFF,
            16 * 1024 * 1024,
        );
    }
    // Load firmware into flash data buffer
    if let Some(ref flash_cfg) = config.flash {
        use std::io::Read;
        let firmware_path = std::path::Path::new(&flash_cfg.firmware);
        let mut file = std::fs::File::open(firmware_path).expect("Failed to open firmware file");
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .expect("Failed to read firmware");
        let offset = flash_cfg.firmware_offset;
        assert!(
            offset + data.len() <= 16 * 1024 * 1024,
            "Firmware too large for flash buffer"
        );
        unsafe {
            let dest = (flash_data_buffer.contents() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dest, data.len());
        }
        clilog::info!(
            "Loaded {} bytes firmware into GPU flash buffer at offset 0x{:X}",
            data.len(),
            offset
        );
    }

    // ── GPU UART IO buffers ─────────────────────────────────────────────

    // UartDecoderState (shared, persistent across ticks)
    let uart_state_buffer = simulator.device.new_buffer(
        std::mem::size_of::<UartDecoderState>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let us = &mut *(uart_state_buffer.contents() as *mut UartDecoderState);
        us.state = 0; // IDLE
        us.last_tx = 1; // TX line idle high
        us.start_cycle = 0;
        us.bits_received = 0;
        us.value = 0;
        us.current_cycle = 0;
    }

    // UartParams (constant)
    let uart_params_buffer = simulator.device.new_buffer(
        std::mem::size_of::<UartParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let p = &mut *(uart_params_buffer.contents() as *mut UartParams);
        p.state_size = state_size as u32;
        p.has_uart = if config.uart.is_some() { 1 } else { 0 };
        p.tx_out_pos = uart_tx_gpio
            .and_then(|tx| gpio_map.output_bits.get(&tx).copied())
            .unwrap_or(0);
        p.cycles_per_bit = (clock_hz / uart_baud as u64) as u32;
    }

    // UartChannel (shared ring buffer, CPU drains after each batch)
    let uart_channel_buffer = simulator.device.new_buffer(
        std::mem::size_of::<UartChannel>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let ch = &mut *(uart_channel_buffer.contents() as *mut UartChannel);
        ch.write_head = 0;
        ch.capacity = 4096;
        ch._pad = [0; 2];
        // data doesn't need to be zeroed (ring buffer semantics)
    }

    // ── GPU Wishbone Bus Trace buffers ────────────────────────────────

    let wb_trace_params = build_wb_trace_params(aig, netlistdb, script);
    let wb_trace_params_buffer = simulator.device.new_buffer(
        std::mem::size_of::<WbTraceParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let p = &mut *(wb_trace_params_buffer.contents() as *mut WbTraceParams);
        *p = wb_trace_params;
    }

    // WbTraceChannel: header (16 bytes) + entries array
    let wb_channel_byte_size = std::mem::size_of::<WbTraceChannel>()
        + WB_TRACE_CHANNEL_CAP * std::mem::size_of::<WbTraceEntry>();
    let wb_trace_channel_buffer = simulator.device.new_buffer(
        wb_channel_byte_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let ch = &mut *(wb_trace_channel_buffer.contents() as *mut WbTraceChannel);
        ch.write_head = 0;
        ch.capacity = WB_TRACE_CHANNEL_CAP as u32;
        ch.current_tick = 0;
        ch.prev_flags = 0;
    }

    // Pre-write params for all simulation stages (they don't change between ticks)
    for stage_i in 0..num_major_stages {
        simulator.write_params(stage_i, num_blocks, num_major_stages, state_size, 0);
    }

    clilog::finish!(timer_prep);

    // ── GPU Kernel Profiling (optional) ──────────────────────────────────

    if opts.gpu_profile {
        let profile_ticks = opts.max_cycles.unwrap_or(1000).min(5000);
        simulator.profile_gpu_kernels(
            profile_ticks,
            num_blocks,
            num_major_stages,
            state_size,
            &blocks_start_buffer,
            &blocks_data_buffer,
            &sram_data_buffer,
            &states_buffer,
            &event_buffer_metal,
            &schedule_buffers,
            &flash_state_buffer,
            &flash_din_params_buffer,
            &flash_model_params_buffer,
            &flash_data_buffer,
            &uart_state_buffer,
            &uart_params_buffer,
            &uart_channel_buffer,
            &wb_trace_channel_buffer,
            &wb_trace_params_buffer,
            timing_constraints_buffer.as_ref(),
        );

        // Clean up event buffer
        unsafe {
            drop(Box::from_raw(event_buffer_ptr));
        }
        return CosimResult {
            passed: true,
            uart_events: Vec::new(),
            ticks_simulated: 0,
        };
    }

    // ── GPU-only simulation loop ─────────────────────────────────────────
    //
    // All IO models (flash + UART) run on GPU. No per-tick CPU interaction.
    // CPU just drains decoded UART bytes from the ring buffer after each batch.

    let timer_sim = clilog::stimer!("simulation");
    let sim_start = std::time::Instant::now();

    // Helper: update reset value in all schedule ops buffers
    let update_reset_in_ops = |reset_val: u8| {
        let reset_pos = gpio_map.input_bits[&reset_gpio];
        for (sched_idx, (ref fall_params, ref fall_buf, ref rise_params, ref rise_buf)) in
            schedule_buffers.tick_buffers.iter().enumerate()
        {
            let _ = (fall_params, rise_params); // params don't contain reset
            for (buf, len) in [
                (fall_buf, schedule_buffers.fall_ops_lens[sched_idx]),
                (rise_buf, schedule_buffers.rise_ops_lens[sched_idx]),
            ] {
                let ops: &mut [BitOp] =
                    unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut BitOp, len) };
                for op in ops.iter_mut() {
                    if op.position == reset_pos {
                        op.value = reset_val as u32;
                    }
                }
            }
        }
    };

    // Track schedule position across batches
    let mut schedule_offset: usize = 0;

    // Verify flash state hasn't been corrupted before main loop
    unsafe {
        let fs = &*(flash_state_buffer.contents() as *const FlashState);
        clilog::info!(
            "FlashState before main loop: d_i=0x{:02X}, data_width={}, prev_csn={}, in_reset={}",
            fs.d_i,
            fs.data_width,
            fs.prev_csn,
            fs.in_reset
        );
        assert_eq!(
            fs.d_i, 0x0F,
            "FlashState.d_i corrupted before main loop: got 0x{:02X}",
            fs.d_i
        );
    }

    // UART event collection (CPU-side, populated from channel drain)
    let mut uart_events: Vec<UartEvent> = Vec::new();
    let mut uart_read_head: u32 = 0;
    let mut wb_trace_read_head: u32 = 0;

    // Profiling accumulators
    let mut prof_batch_encode: u64 = 0;
    let mut prof_gpu_wait: u64 = 0;
    let mut prof_drain: u64 = 0;
    let mut total_batches: u64 = 0;

    // Per-tick tracing: run 1 tick at a time for first N ticks after reset
    let trace_ticks = if std::env::var("FLASH_TRACE").is_ok() {
        200
    } else {
        0
    };
    let deep_diag = std::env::var("GEM_DIAG").is_ok();
    let mut diag_prev_flash_addr: u32 = 0;
    let mut diag_prev_flash_cmd: u8 = 0;
    let mut diag_prev_csn: u32 = 1;
    let mut diag_sram_write_count: usize = 0;

    // CPU verification state (--check-with-cpu)
    let mut cpu_states: Vec<u32> = if opts.check_with_cpu {
        vec![0u32; 2 * state_size]
    } else {
        Vec::new()
    };
    let mut cpu_sram: Vec<u32> = if opts.check_with_cpu {
        vec![0u32; script.sram_storage_size as usize]
    } else {
        Vec::new()
    };
    let mut cpu_check_mismatches: usize = 0;
    let cpu_check_max_ticks = if opts.check_with_cpu { 500 } else { 0 };

    let mut post_reset_state_snapshot: Option<Vec<u32>> = None;

    let mut tick: usize = 0;
    while tick < max_ticks {
        let batch = if opts.check_with_cpu && tick < cpu_check_max_ticks {
            1 // single tick for CPU comparison
        } else if trace_ticks > 0 && tick < reset_cycles + trace_ticks {
            1 // single tick for tracing
        } else if deep_diag {
            1 // single tick for deep diagnostics
        } else {
            BATCH_SIZE.min(max_ticks - tick)
        };

        // Don't cross reset boundary within a batch
        let batch = if tick < reset_cycles && tick + batch > reset_cycles {
            reset_cycles - tick
        } else {
            batch
        };

        // Update reset value and flash in_reset for this batch
        let in_reset = tick < reset_cycles;
        let current_reset_val = if in_reset {
            reset_val_active
        } else {
            reset_val_inactive
        };
        update_reset_in_ops(current_reset_val);

        // Update flash_state.in_reset on GPU
        unsafe {
            let fs = &mut *(flash_state_buffer.contents() as *mut FlashState);
            fs.in_reset = if in_reset { 1 } else { 0 };
        }

        // Save pre-tick state for CPU verification
        let saved_flash_d_i: u8;
        if opts.check_with_cpu && tick < cpu_check_max_ticks && batch == 1 {
            let gpu_states: &[u32] = unsafe {
                std::slice::from_raw_parts(states_buffer.contents() as *const u32, 2 * state_size)
            };
            cpu_states.copy_from_slice(gpu_states);
            let gpu_sram: &[u32] = unsafe {
                std::slice::from_raw_parts(
                    sram_data_buffer.contents() as *const u32,
                    script.sram_storage_size as usize,
                )
            };
            cpu_sram.copy_from_slice(gpu_sram);
            // Save flash d_i before GPU modifies it (apply_flash_din reads this)
            saved_flash_d_i = unsafe {
                let fs = &*(flash_state_buffer.contents() as *const FlashState);
                if tick == 0 {
                    // Dump raw bytes at flash state location
                    let raw = std::slice::from_raw_parts(
                        flash_state_buffer.contents() as *const u8,
                        std::mem::size_of::<FlashState>(),
                    );
                    eprintln!("  FlashState raw bytes (tick 0): {:02X?}", raw);
                    eprintln!("  FlashState fields: bit_count={}, byte_count={}, data_width={}, addr=0x{:08X}",
                        fs.bit_count, fs.byte_count, fs.data_width, fs.addr);
                    eprintln!("  FlashState fields: curr_byte=0x{:02X}, command=0x{:02X}, out_buffer=0x{:02X}",
                        fs.curr_byte, fs.command, fs.out_buffer);
                    eprintln!(
                        "  FlashState fields: prev_clk={}, prev_csn={}, d_i=0x{:02X}",
                        fs.prev_clk, fs.prev_csn, fs.d_i
                    );
                    eprintln!("  FlashState fields: prev_d_out=0x{:02X}, in_reset={}, last_error_cmd={}, model_prev_csn={}",
                        fs.prev_d_out, fs.in_reset, fs.last_error_cmd, fs.model_prev_csn);
                    eprintln!(
                        "  FlashState offsetof d_i = {}",
                        (&fs.d_i as *const u8 as usize) - (fs as *const FlashState as usize)
                    );
                }
                fs.d_i
            };
            if tick == 0 {
                eprintln!(
                    "  saved_flash_d_i = 0x{:02X} (right after assignment)",
                    saved_flash_d_i
                );
            }
        } else {
            saved_flash_d_i = 0;
        }

        // Encode and commit GPU batch
        let t_encode = std::time::Instant::now();
        let batch_done = simulator.encode_and_commit_gpu_batch(
            batch,
            num_blocks,
            num_major_stages,
            state_size,
            &blocks_start_buffer,
            &blocks_data_buffer,
            &sram_data_buffer,
            &states_buffer,
            &event_buffer_metal,
            &schedule_buffers,
            schedule_offset,
            &flash_state_buffer,
            &flash_din_params_buffer,
            &flash_model_params_buffer,
            &flash_data_buffer,
            &uart_state_buffer,
            &uart_params_buffer,
            &uart_channel_buffer,
            &wb_trace_channel_buffer,
            &wb_trace_params_buffer,
            timing_constraints_buffer.as_ref(),
        );
        schedule_offset = (schedule_offset + batch) % schedule_buffers.tick_buffers.len();
        prof_batch_encode += t_encode.elapsed().as_nanos() as u64;

        // Wait for GPU batch to complete
        let t_wait = std::time::Instant::now();
        simulator.spin_wait(batch_done);
        prof_gpu_wait += t_wait.elapsed().as_nanos() as u64;

        // Per-tick flash signal diagnostic
        if trace_ticks > 0 && tick + batch <= reset_cycles + trace_ticks as usize {
            let output_state: &[u32] = unsafe {
                std::slice::from_raw_parts(
                    (states_buffer.contents() as *const u32).add(state_size),
                    state_size,
                )
            };
            let fmp =
                unsafe { &*(flash_model_params_buffer.contents() as *const FlashModelParams) };
            let flash_clk_pos = fmp.clk_out_pos;
            let flash_csn_pos = fmp.csn_out_pos;
            let flash_clk =
                (output_state[(flash_clk_pos >> 5) as usize] >> (flash_clk_pos & 31)) & 1;
            let flash_csn =
                (output_state[(flash_csn_pos >> 5) as usize] >> (flash_csn_pos & 31)) & 1;
            let fs = unsafe { &*(flash_state_buffer.contents() as *const FlashState) };
            clilog::debug!("FLASH_TRACE tick {}: clk={} csn={} d_i=0x{:02X} cmd=0x{:02X} addr=0x{:06X} in_reset={}",
                tick + batch, flash_clk, flash_csn, fs.d_i, fs.command, fs.addr, fs.in_reset);
        }

        // ── CPU verification: simulate same tick on CPU and compare ──
        if opts.check_with_cpu && tick < cpu_check_max_ticks && batch == 1 {
            // CPU state_prep(fall): copy output → input, apply fall_ops
            // Use schedule position 0 for CPU verification (single-clock backward compat)
            let cpu_sched_idx = 0;
            cpu_states.copy_within(state_size..2 * state_size, 0);
            let (_, ref cpu_fall_buf, _, _) = schedule_buffers.tick_buffers[cpu_sched_idx];
            let cpu_fall_ops: &[BitOp] = unsafe {
                std::slice::from_raw_parts(
                    cpu_fall_buf.contents() as *const BitOp,
                    schedule_buffers.fall_ops_lens[cpu_sched_idx],
                )
            };
            for op in cpu_fall_ops {
                let word_idx = op.position as usize >> 5;
                let bit_mask = 1u32 << (op.position & 31);
                if op.value != 0 {
                    cpu_states[word_idx] |= bit_mask;
                } else {
                    cpu_states[word_idx] &= !bit_mask;
                }
            }

            // CPU apply_flash_din
            {
                let p = unsafe { &*(flash_din_params_buffer.contents() as *const FlashDinParams) };
                if tick == 0 {
                    eprintln!(
                        "  CPU flash_din: has_flash={}, d_i=0x{:02X}, d_in_pos={:?}",
                        p.has_flash, saved_flash_d_i, p.d_in_pos
                    );
                }
                if p.has_flash != 0 {
                    for i in 0..4usize {
                        let pos = p.d_in_pos[i];
                        if pos == 0xFFFFFFFF {
                            continue;
                        }
                        let word_idx = (pos >> 5) as usize;
                        let bit_mask = 1u32 << (pos & 31);
                        if (saved_flash_d_i >> i) & 1 != 0 {
                            cpu_states[word_idx] |= bit_mask;
                        } else {
                            cpu_states[word_idx] &= !bit_mask;
                        }
                    }
                }
                if tick == 0 {
                    eprintln!(
                        "  CPU after flash_din(fall): word[4]=0x{:08X}",
                        cpu_states[4]
                    );
                }
            }

            // CPU simulate(fall): run all partitions
            for stage_i in 0..num_major_stages {
                for block_i in 0..num_blocks {
                    let start = script.blocks_start[stage_i * num_blocks + block_i];
                    let end = script.blocks_start[stage_i * num_blocks + block_i + 1];
                    if start == end {
                        continue;
                    }
                    let (input_half, output_half) = cpu_states.split_at_mut(state_size);
                    simulate_block_v1(
                        &script.blocks_data[start..end],
                        input_half,
                        output_half,
                        &mut cpu_sram,
                    );
                }
            }

            // CPU state_prep(rise): copy output → input, apply rise_ops
            cpu_states.copy_within(state_size..2 * state_size, 0);
            let (_, _, _, ref cpu_rise_buf) = schedule_buffers.tick_buffers[cpu_sched_idx];
            let cpu_rise_ops: &[BitOp] = unsafe {
                std::slice::from_raw_parts(
                    cpu_rise_buf.contents() as *const BitOp,
                    schedule_buffers.rise_ops_lens[cpu_sched_idx],
                )
            };
            for op in cpu_rise_ops {
                let word_idx = op.position as usize >> 5;
                let bit_mask = 1u32 << (op.position & 31);
                if op.value != 0 {
                    cpu_states[word_idx] |= bit_mask;
                } else {
                    cpu_states[word_idx] &= !bit_mask;
                }
            }

            // CPU apply_flash_din (same d_i for both edges within a tick)
            unsafe {
                let p = &*(flash_din_params_buffer.contents() as *const FlashDinParams);
                if p.has_flash != 0 {
                    for i in 0..4 {
                        let pos = p.d_in_pos[i];
                        if pos == 0xFFFFFFFF {
                            continue;
                        }
                        let word_idx = (pos >> 5) as usize;
                        let bit_mask = 1u32 << (pos & 31);
                        if (saved_flash_d_i >> i) & 1 != 0 {
                            cpu_states[word_idx] |= bit_mask;
                        } else {
                            cpu_states[word_idx] &= !bit_mask;
                        }
                    }
                }
            }

            // CPU simulate(rise): run all partitions
            let use_diag = tick == reset_cycles; // first tick after reset release
            if use_diag {
                eprintln!(
                    "=== DIAG: Rising-edge simulate at tick {} (first post-reset) ===",
                    tick
                );
                eprintln!(
                    "  input_state[0]=0x{:08X} (bit0=posedge={})",
                    cpu_states[0],
                    cpu_states[0] & 1
                );
            }
            for stage_i in 0..num_major_stages {
                for block_i in 0..num_blocks {
                    let start = script.blocks_start[stage_i * num_blocks + block_i];
                    let end = script.blocks_start[stage_i * num_blocks + block_i + 1];
                    if start == end {
                        continue;
                    }
                    let (input_half, output_half) = cpu_states.split_at_mut(state_size);
                    if use_diag && block_i < 3 {
                        eprintln!(
                            "  Block {}/{} (stage {}): script range {}..{} ({} words)",
                            block_i,
                            num_blocks,
                            stage_i,
                            start,
                            end,
                            end - start
                        );
                        simulate_block_v1_diag(
                            &script.blocks_data[start..end],
                            input_half,
                            output_half,
                            &mut cpu_sram,
                        );
                    } else {
                        simulate_block_v1(
                            &script.blocks_data[start..end],
                            input_half,
                            output_half,
                            &mut cpu_sram,
                        );
                    }
                }
            }

            // Direct AIG evaluation comparison at first few post-reset ticks
            if tick >= reset_cycles && tick <= reset_cycles + 5 {
                let (input_half, output_half) = cpu_states.split_at(state_size);
                compare_aig_vs_flattened(
                    aig,
                    input_half,
                    output_half,
                    script,
                    state_size,
                    tick,
                    Some(netlistdb),
                );
            }

            // Compare GPU input with CPU input (should match after state_prep+flash_din)
            let gpu_states: &[u32] = unsafe {
                std::slice::from_raw_parts(states_buffer.contents() as *const u32, 2 * state_size)
            };
            let mut input_mismatches = 0;
            for i in 0..state_size {
                if gpu_states[i] != cpu_states[i] {
                    if input_mismatches < 5 {
                        let diff = gpu_states[i] ^ cpu_states[i];
                        let mut bits = Vec::new();
                        for b in 0..32 {
                            if (diff >> b) & 1 != 0 {
                                bits.push(i as u32 * 32 + b);
                            }
                        }
                        eprintln!(
                            "  INPUT MISMATCH word[{}]: GPU=0x{:08X} CPU=0x{:08X} bits={:?}",
                            i, gpu_states[i], cpu_states[i], bits
                        );
                    }
                    input_mismatches += 1;
                }
            }

            // Compare GPU output with CPU output
            let gpu_output = &gpu_states[state_size..2 * state_size];
            let cpu_output = &cpu_states[state_size..2 * state_size];
            let mut mismatches = 0;
            let mut _first_mismatch_word = 0;
            for i in 0..state_size {
                if gpu_output[i] != cpu_output[i] {
                    if mismatches < 5 {
                        let diff = gpu_output[i] ^ cpu_output[i];
                        let mut bits = Vec::new();
                        for b in 0..32 {
                            if (diff >> b) & 1 != 0 {
                                bits.push(i as u32 * 32 + b);
                            }
                        }
                        eprintln!(
                            "  OUTPUT MISMATCH word[{}]: GPU=0x{:08X} CPU=0x{:08X} bits={:?}",
                            i, gpu_output[i], cpu_output[i], bits
                        );
                    }
                    if mismatches == 0 {
                        _first_mismatch_word = i;
                    }
                    mismatches += 1;
                }
            }
            // Also compare SRAM
            let gpu_sram: &[u32] = unsafe {
                std::slice::from_raw_parts(
                    sram_data_buffer.contents() as *const u32,
                    script.sram_storage_size as usize,
                )
            };
            let mut sram_mismatches = 0;
            for i in 0..script.sram_storage_size as usize {
                if gpu_sram[i] != cpu_sram[i] {
                    if sram_mismatches < 3 {
                        eprintln!(
                            "  SRAM MISMATCH [{}]: GPU=0x{:08X} CPU=0x{:08X}",
                            i, gpu_sram[i], cpu_sram[i]
                        );
                    }
                    sram_mismatches += 1;
                }
            }
            if input_mismatches > 0 || mismatches > 0 || sram_mismatches > 0 {
                eprintln!(
                    "CHECK-WITH-CPU tick {}: {} input, {} output, {} SRAM mismatches",
                    tick, input_mismatches, mismatches, sram_mismatches
                );
                cpu_check_mismatches += mismatches;
                if cpu_check_mismatches > 200 {
                    eprintln!("Too many mismatches, stopping CPU check");
                }
            } else if tick <= reset_cycles + 5 || tick % 50 == 0 {
                eprintln!(
                    "CHECK-WITH-CPU tick {}: OK (state_size={}, sram_size={})",
                    tick, state_size, script.sram_storage_size
                );
            }

            // Per-tick output state change tracking
            if tick >= reset_cycles.saturating_sub(2) && tick <= reset_cycles + 15 {
                let gpu_output = &gpu_states[state_size..2 * state_size];
                let _changed_words: usize = gpu_output
                    .iter()
                    .zip(cpu_output.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                let output_bits_set: usize =
                    gpu_output.iter().map(|w| w.count_ones() as usize).sum();
                let changed_from_prev: usize = if let Some(ref snap) = post_reset_state_snapshot {
                    gpu_output
                        .iter()
                        .zip(snap.iter())
                        .map(|(a, b)| (a ^ b).count_ones() as usize)
                        .sum()
                } else {
                    0
                };
                eprintln!(
                    "TICK-TRACE {}: output_bits_set={}, changed_from_prev={}",
                    tick, output_bits_set, changed_from_prev
                );
                post_reset_state_snapshot = Some(gpu_output.to_vec());
            }
        }

        // Drain UART channel (CPU reads decoded bytes from GPU ring buffer)
        let t_drain = std::time::Instant::now();
        unsafe {
            let channel = &*(uart_channel_buffer.contents() as *const UartChannel);
            while uart_read_head < channel.write_head {
                let byte = channel.data[(uart_read_head % channel.capacity) as usize];
                let ch = if byte >= 32 && byte < 127 {
                    byte as char
                } else {
                    '.'
                };
                clilog::info!("UART TX: 0x{:02X} '{}'", byte, ch);
                uart_events.push(UartEvent {
                    timestamp: tick, // approximate tick
                    peripheral: "uart_0".to_string(),
                    event: "tx".to_string(),
                    payload: byte,
                });
                uart_read_head += 1;
            }
        }
        // Drain WB trace channel
        unsafe {
            let ch = &*(wb_trace_channel_buffer.contents() as *const WbTraceChannel);
            let entries_ptr =
                (wb_trace_channel_buffer.contents() as *const u8).add(16) as *const WbTraceEntry;
            while wb_trace_read_head < ch.write_head {
                let idx = (wb_trace_read_head % ch.capacity) as usize;
                let e = &*entries_ptr.add(idx);
                let ibus_cyc = e.flags & 1;
                let ibus_stb = (e.flags >> 1) & 1;
                let dbus_cyc = (e.flags >> 2) & 1;
                let dbus_stb = (e.flags >> 3) & 1;
                let dbus_we = (e.flags >> 4) & 1;
                let spi_ack = (e.flags >> 5) & 1;
                let sram_ack = (e.flags >> 6) & 1;
                let _csr_ack = (e.flags >> 7) & 1;
                if ibus_cyc != 0 && ibus_stb != 0 {
                    eprintln!(
                        "WB T{:>5}: IBUS adr=0x{:08X} rdata=0x{:08X} spi_ack={} sram_ack={}",
                        e.tick, e.ibus_adr, e.ibus_rdata, spi_ack, sram_ack
                    );
                }
                if dbus_cyc != 0 && dbus_stb != 0 {
                    eprintln!(
                        "WB T{:>5}: DBUS adr=0x{:08X} we={} sram_ack={}",
                        e.tick, e.dbus_adr, dbus_we, sram_ack
                    );
                }
                if ibus_cyc == 0 && ibus_stb == 0 && dbus_cyc == 0 && dbus_stb == 0 {
                    eprintln!("WB T{:>5}: bus idle (flags=0x{:02X})", e.tick, e.flags);
                }
                wb_trace_read_head += 1;
            }
        }
        prof_drain += t_drain.elapsed().as_nanos() as u64;

        total_batches += 1;
        tick += batch;

        // Deep diagnostics: SRAM activity + flash transaction tracking
        if deep_diag && tick > reset_cycles && batch == 1 {
            unsafe {
                let fs = &*(flash_state_buffer.contents() as *const FlashState);
                let fmp = &*(flash_model_params_buffer.contents() as *const FlashModelParams);
                let st = std::slice::from_raw_parts(
                    states_buffer.contents() as *const u32,
                    2 * state_size,
                );
                let read_out_bit = |pos: u32| -> u32 {
                    let word_idx = state_size + (pos as usize >> 5);
                    let bit_idx = pos & 31;
                    (st[word_idx] >> bit_idx) & 1
                };
                let csn_val = read_out_bit(fmp.csn_out_pos);

                // Detect CSN transitions (transaction boundaries)
                if csn_val == 1 && diag_prev_csn == 0 {
                    // CSN rising edge = transaction complete
                    eprintln!(
                        "DIAG T{}: Flash transaction end: cmd=0x{:02X} addr=0x{:06X} bytes={}",
                        tick, fs.command, fs.addr, fs.byte_count
                    );
                }
                if csn_val == 0 && diag_prev_csn == 1 {
                    eprintln!("DIAG T{}: Flash transaction start", tick);
                }
                diag_prev_csn = csn_val;

                // Track address changes
                if fs.addr != diag_prev_flash_addr || fs.command != diag_prev_flash_cmd {
                    if fs.command != 0 {
                        eprintln!("DIAG T{}: Flash addr changed: cmd=0x{:02X} addr=0x{:06X} bytes={} bitc={}",
                            tick, fs.command, fs.addr, fs.byte_count, fs.bit_count);
                    }
                    diag_prev_flash_addr = fs.addr;
                    diag_prev_flash_cmd = fs.command;
                }

                // Check SRAM activity every 100 ticks
                if tick % 100 == 0 {
                    let sram = std::slice::from_raw_parts(
                        sram_data_buffer.contents() as *const u32,
                        script.sram_storage_size as usize,
                    );
                    let nonzero = sram.iter().filter(|&&w| w != 0).count();
                    if nonzero != diag_sram_write_count {
                        eprintln!(
                            "DIAG T{}: SRAM non-zero words: {} (was {})",
                            tick, nonzero, diag_sram_write_count
                        );
                        diag_sram_write_count = nonzero;
                    }
                }
            }
        }

        // Diagnostic: dump flash-related signals
        if trace_ticks > 0 && tick <= reset_cycles + trace_ticks && tick > reset_cycles {
            unsafe {
                let st = std::slice::from_raw_parts(
                    states_buffer.contents() as *const u32,
                    2 * state_size,
                );
                let read_out_bit = |pos: u32| -> u32 {
                    let word_idx = state_size + (pos as usize >> 5);
                    let bit_idx = pos & 31;
                    (st[word_idx] >> bit_idx) & 1
                };
                let read_in_bit = |pos: u32| -> u32 {
                    let word_idx = (pos as usize) >> 5;
                    let bit_idx = pos & 31;
                    (st[word_idx] >> bit_idx) & 1
                };
                let fmp = &*(flash_model_params_buffer.contents() as *const FlashModelParams);
                let fdp = &*(flash_din_params_buffer.contents() as *const FlashDinParams);
                let fs = &*(flash_state_buffer.contents() as *const FlashState);
                let clk_val = read_out_bit(fmp.clk_out_pos);
                let csn_val = read_out_bit(fmp.csn_out_pos);
                let mut d_out_vals = [0u32; 4];
                for i in 0..4 {
                    if fmp.d_out_pos[i] != 0xFFFFFFFF {
                        d_out_vals[i] = read_out_bit(fmp.d_out_pos[i]);
                    }
                }
                let mut d_in_vals = [0u32; 4];
                for i in 0..4 {
                    if fdp.d_in_pos[i] != 0xFFFFFFFF {
                        d_in_vals[i] = read_in_bit(fdp.d_in_pos[i]);
                    }
                }
                eprintln!("T{:>4}: clk={} csn={} d_o={}{}{}{} d_i={}{}{}{} | fs: d_i=0x{:X} cmd=0x{:02X} bc={} bitc={} dw={} addr=0x{:06X} pclk={} pcsn={} mcsn={} inr={}",
                    tick, clk_val, csn_val,
                    d_out_vals[3], d_out_vals[2], d_out_vals[1], d_out_vals[0],
                    d_in_vals[3], d_in_vals[2], d_in_vals[1], d_in_vals[0],
                    fs.d_i, fs.command, fs.byte_count, fs.bit_count, fs.data_width, fs.addr,
                    fs.prev_clk, fs.prev_csn, fs.model_prev_csn, fs.in_reset);
            }
        }

        // Progress logging
        if tick > 0 && tick % 100000 < BATCH_SIZE {
            let elapsed = sim_start.elapsed();
            let us_per_tick = elapsed.as_micros() as f64 / tick as f64;
            // Read UART TX bit and decoder state for diagnostics
            let (uart_tx_val, uart_dec_state, uart_dec_cycle) = unsafe {
                let up = &*(uart_params_buffer.contents() as *const UartParams);
                let st = std::slice::from_raw_parts(
                    states_buffer.contents() as *const u32,
                    2 * state_size,
                );
                let tx_word = state_size + (up.tx_out_pos as usize >> 5);
                let tx_bit = up.tx_out_pos & 31;
                let tx_val = (st[tx_word] >> tx_bit) & 1;
                let us = &*(uart_state_buffer.contents() as *const UartDecoderState);
                (tx_val, us.state, us.current_cycle)
            };
            clilog::info!(
                "Tick {} / {} ({:.1}μs/tick, batches={}, UART bytes={}, tx={}, uart_st={}, uart_cyc={})",
                tick, max_ticks, us_per_tick, total_batches, uart_events.len(),
                uart_tx_val, uart_dec_state, uart_dec_cycle
            );
        }
        if tick >= reset_cycles && tick - batch < reset_cycles {
            clilog::info!("Reset released at tick {}", reset_cycles);
            // Snapshot output state just after reset for change detection
            if post_reset_state_snapshot.is_none() {
                let st = unsafe {
                    std::slice::from_raw_parts(
                        states_buffer.contents() as *const u32,
                        2 * state_size,
                    )
                };
                post_reset_state_snapshot = Some(st[state_size..2 * state_size].to_vec());
            }
        }
    }

    // Print profiling results
    let total_ns = prof_batch_encode + prof_gpu_wait + prof_drain;
    let print_prof = |name: &str, ns: u64| {
        let us = ns as f64 / 1000.0 / max_ticks as f64;
        let pct = if total_ns > 0 {
            100.0 * ns as f64 / total_ns as f64
        } else {
            0.0
        };
        println!("  {:<32} {:>8.1}μs/tick  {:>5.1}%", name, us, pct);
    };
    println!();
    println!("=== Profiling Breakdown ===");
    print_prof("Batch encode + commit", prof_batch_encode);
    print_prof("GPU wait (spin)", prof_gpu_wait);
    print_prof("UART channel drain", prof_drain);
    println!(
        "  {:<32} {:>8.1}μs/tick  100.0%",
        "TOTAL (instrumented)",
        total_ns as f64 / 1000.0 / max_ticks as f64
    );
    println!();
    println!("  Total batches:                 {}", total_batches);

    let sim_elapsed = sim_start.elapsed();
    clilog::finish!(timer_sim);

    // ── State buffer diagnostics ─────────────────────────────────────────

    unsafe {
        let st = std::slice::from_raw_parts(states_buffer.contents() as *const u32, 2 * state_size);
        let input_state = &st[..state_size];
        let output_state = &st[state_size..2 * state_size];
        let in_nonzero = input_state.iter().filter(|&&w| w != 0).count();
        let out_nonzero = output_state.iter().filter(|&&w| w != 0).count();
        let in_popcount: u32 = input_state.iter().map(|w| w.count_ones()).sum();
        let out_popcount: u32 = output_state.iter().map(|w| w.count_ones()).sum();
        println!();
        println!("=== State Buffer Diagnostics ===");
        println!(
            "State size: {} words ({} bits)",
            state_size,
            state_size * 32
        );
        println!(
            "Input state:  {} non-zero words, {} bits set",
            in_nonzero, in_popcount
        );
        println!(
            "Output state: {} non-zero words, {} bits set",
            out_nonzero, out_popcount
        );
        // Compare with post-reset snapshot if available
        if let Some(ref snapshot) = post_reset_state_snapshot {
            let mut changed_words = 0;
            let mut changed_bits = 0u32;
            for (i, (&cur, &snap)) in output_state.iter().zip(snapshot.iter()).enumerate() {
                let diff = cur ^ snap;
                if diff != 0 {
                    changed_words += 1;
                    changed_bits += diff.count_ones();
                    if changed_words <= 10 {
                        println!(
                            "  CHANGED output[{}]: 0x{:08X} → 0x{:08X} (diff=0x{:08X})",
                            i, snap, cur, diff
                        );
                    }
                }
            }
            println!(
                "Output state changes since tick {}: {} words, {} bits changed",
                config.reset_cycles + 1,
                changed_words,
                changed_bits
            );
        }
    }

    // ── Check GPU flash state for errors ─────────────────────────────────

    let last_error_cmd = unsafe {
        let fs = &*(flash_state_buffer.contents() as *const FlashState);
        fs.last_error_cmd
    };
    if last_error_cmd != 0 {
        clilog::warn!(
            "GPU flash model encountered unknown command: 0x{:02X}",
            last_error_cmd
        );
    }

    // ── Results ──────────────────────────────────────────────────────────

    println!();
    println!("=== GPU Simulation Results ===");
    println!("Ticks simulated: {}", max_ticks);
    println!("UART bytes received: {}", uart_events.len());

    if max_ticks > 0 {
        let us_per_tick = sim_elapsed.as_micros() as f64 / max_ticks as f64;
        println!(
            "Time per tick: {:.1}μs ({:.1}s total)",
            us_per_tick,
            sim_elapsed.as_secs_f64()
        );
    }

    // Print UART output as string
    if !uart_events.is_empty() {
        let uart_str: String = uart_events
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

    // Print flash model stats (GPU-side state)
    if config.flash.is_some() {
        let fs = unsafe { &*(flash_state_buffer.contents() as *const FlashState) };
        println!(
            "GPU Flash model: command=0x{:02X}, byte_count={}, addr=0x{:06X}, error_cmd=0x{:X}",
            fs.command, fs.byte_count, fs.addr, fs.last_error_cmd
        );
    }

    // Output events to JSON
    if let Some(ref output_path) = config.output_events {
        #[derive(serde::Serialize)]
        struct EventsOutput {
            events: Vec<UartEvent>,
        }
        let output = EventsOutput {
            events: uart_events.clone(),
        };
        let json = serde_json::to_string_pretty(&output).expect("Failed to serialize events");
        let mut file = std::fs::File::create(output_path).expect("Failed to create events file");
        use std::io::Write;
        file.write_all(json.as_bytes())
            .expect("Failed to write events");
        clilog::info!("Wrote events to {}", output_path);
    }

    // ── Event reference comparison ───────────────────────────────────────

    let mut events_passed = true;
    if let Some(ref ref_path) = config.events_reference {
        #[derive(serde::Deserialize)]
        struct EventsFile {
            events: Vec<UartEvent>,
        }
        let ref_file = std::fs::read_to_string(ref_path)
            .unwrap_or_else(|e| panic!("Failed to read events reference {}: {}", ref_path, e));
        let reference: EventsFile = serde_json::from_str(&ref_file)
            .unwrap_or_else(|e| panic!("Failed to parse events reference {}: {}", ref_path, e));

        let ref_events = &reference.events;
        let ref_payloads: Vec<u8> = ref_events.iter().map(|e| e.payload).collect();
        let actual_payloads: Vec<u8> = uart_events.iter().map(|e| e.payload).collect();

        println!();
        println!("=== Event Reference Check ===");
        println!("Reference: {} events from {}", ref_events.len(), ref_path);
        println!("Actual:    {} events", uart_events.len());

        if ref_payloads.len() > actual_payloads.len() {
            println!(
                "FAIL: missing {} events (got {}, expected {})",
                ref_payloads.len() - actual_payloads.len(),
                actual_payloads.len(),
                ref_payloads.len()
            );
            println!(
                "  Hint: increase --max-cycles (last reference event at timestamp {})",
                ref_events.last().map(|e| e.timestamp).unwrap_or(0)
            );
            events_passed = false;
        } else {
            let mut mismatches = 0;
            for (i, (expected, actual)) in
                ref_payloads.iter().zip(actual_payloads.iter()).enumerate()
            {
                if expected != actual {
                    if mismatches < 10 {
                        let ref_ts = ref_events[i].timestamp;
                        let act_ts = uart_events[i].timestamp;
                        println!("  MISMATCH event {}: expected 0x{:02X} (ref ts={}), got 0x{:02X} (tick={})",
                            i, expected, ref_ts, actual, act_ts);
                    }
                    mismatches += 1;
                }
            }

            if mismatches > 0 {
                println!(
                    "FAIL: {} payload mismatches out of {} events",
                    mismatches,
                    ref_payloads.len()
                );
                events_passed = false;
            } else {
                // Decode the matched message for display
                let decoded: String = actual_payloads
                    .iter()
                    .map(|&b| if b >= 32 && b < 127 { b as char } else { '.' })
                    .collect();
                println!("PASS: all {} event payloads match", ref_payloads.len());
                println!("  Decoded: \"{}\"", decoded);
            }
        }
    }

    // ── Optional CPU verification ────────────────────────────────────────

    if opts.check_with_cpu {
        if cpu_check_mismatches == 0 {
            clilog::info!(
                "CPU verification: PASSED ({} ticks checked)",
                cpu_check_max_ticks.min(max_ticks)
            );
        } else {
            clilog::warn!(
                "CPU verification: {} total mismatches in {} ticks",
                cpu_check_mismatches,
                cpu_check_max_ticks.min(max_ticks)
            );
        }
    }

    // Clean up event buffer
    unsafe {
        drop(Box::from_raw(event_buffer_ptr));
    }

    println!();
    if events_passed {
        println!("SIMULATION: PASSED");
    } else {
        println!("SIMULATION: FAILED (event mismatch)");
    }

    CosimResult {
        passed: events_passed,
        uart_events,
        ticks_simulated: max_ticks,
    }
}
