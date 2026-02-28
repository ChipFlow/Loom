// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Unified CLI for the Loom GPU-accelerated RTL simulator.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use gem::aig::AIG;
use gem::aigpdk::AIGPDKLeafPins;
use gem::sim::setup::{self, DesignArgs};
use gem::sky130::{detect_library_from_file, CellLibrary, SKY130LeafPins};
use gem::staging::build_staged_aigs;
use netlistdb::NetlistDB;

#[derive(Parser)]
#[command(name = "loom", about = "Loom — GPU-accelerated RTL logic simulator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Map a synthesized gate-level netlist to a .gemparts partition file.
    ///
    /// This is the first step in the Loom workflow. The resulting .gemparts file
    /// is then used by `loom sim` to run the design on a GPU.
    Map(MapArgs),

    /// Run a GPU simulation with VCD input/output.
    ///
    /// Reads a gate-level netlist and partition file, processes a VCD input
    /// waveform through the GPU simulator, and writes the output VCD.
    /// Requires building with `--features metal` (macOS), `--features cuda` (Linux/NVIDIA),
    /// or `--features hip` (Linux/AMD).
    Sim(SimArgs),

    /// Run a GPU co-simulation with SPI flash and UART models (Metal only).
    ///
    /// Reads a gate-level netlist and testbench configuration JSON, then runs
    /// a cycle-accurate co-simulation with GPU-side SPI flash and UART models.
    /// Requires building with `--features metal`.
    Cosim(CosimArgs),
}

#[derive(Parser)]
struct MapArgs {
    /// Gate-level Verilog path synthesized with the AIGPDK or SKY130 library.
    ///
    /// If your design is still at RTL level, you must synthesize it first.
    /// See usage.md for synthesis instructions.
    netlist_verilog: PathBuf,

    /// Output path for the serialized partition file (.gemparts).
    parts_out: PathBuf,

    /// Top module name in the netlist.
    ///
    /// If not specified, Loom guesses the top module from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,

    /// Level split thresholds for deep circuits.
    ///
    /// If mapping fails because a single endpoint cannot be partitioned,
    /// add level-split thresholds (e.g. --level-split 30 or --level-split 20,40).
    /// Remember to pass the same thresholds when simulating.
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,

    /// Maximum stage degradation layers allowed during partition merging.
    ///
    /// Default is 0, meaning no degradation is allowed.
    #[clap(long, default_value_t = 0)]
    max_stage_degrad: usize,

    /// Enable selective X-propagation analysis (informational only at map time).
    ///
    /// Reports how many pins and partitions would be X-capable. The actual
    /// X-propagation is enabled at simulation time with `loom sim --xprop`.
    #[clap(long)]
    xprop: bool,
}

#[derive(Parser)]
struct SimArgs {
    /// Gate-level Verilog path synthesized with AIGPDK or SKY130 library.
    netlist_verilog: PathBuf,

    /// Pre-compiled partition mapping (.gemparts file).
    /// If omitted, partitions are generated inline (adds ~20s).
    gemparts: Option<PathBuf>,

    /// VCD input signal path.
    input_vcd: String,

    /// Output VCD path (must be writable).
    output_vcd: String,

    /// Number of GPU blocks to use.
    ///
    /// For CUDA: set to 2x the number of GPU SMs.
    /// For Metal: set to 1.
    num_blocks: usize,

    /// Top module name in the netlist.
    #[clap(long)]
    top_module: Option<String>,

    /// Level split thresholds (must match values used during mapping).
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,

    /// The scope path of top module in the input VCD.
    #[clap(long)]
    input_vcd_scope: Option<String>,

    /// The scope path of top module in the output VCD.
    #[clap(long)]
    output_vcd_scope: Option<String>,

    /// Verify GPU results against CPU baseline.
    #[clap(long)]
    check_with_cpu: bool,

    /// Limit the number of simulated cycles.
    #[clap(long)]
    max_cycles: Option<usize>,

    /// JSON file path for extracting display format strings.
    #[clap(long)]
    json_path: Option<PathBuf>,

    /// Path to SDF file for per-instance back-annotated delays.
    #[clap(long)]
    sdf: Option<PathBuf>,

    /// SDF corner selection: min, typ, or max.
    #[clap(long, default_value = "typ")]
    sdf_corner: String,

    /// Enable SDF debug output.
    #[clap(long)]
    sdf_debug: bool,

    /// Enable selective X-propagation.
    ///
    /// Tracks unknown (X) values through DFF and SRAM outputs. Only partitions
    /// containing X-capable signals pay the overhead. X values appear in the
    /// output VCD as 'x'.
    #[clap(long)]
    xprop: bool,

    /// Enable timing analysis after simulation.
    #[clap(long)]
    enable_timing: bool,

    /// Clock period in picoseconds for timing analysis.
    #[clap(long, default_value = "1000")]
    timing_clock_period: u64,

    /// Report all timing violations (not just summary).
    #[clap(long)]
    timing_report_violations: bool,

    /// Path to Liberty library file for timing data.
    #[clap(long)]
    liberty: Option<PathBuf>,

    /// Enable timing-accurate VCD output with per-signal arrival times.
    ///
    /// Requires --sdf. Signal transitions in the output VCD are offset from
    /// clock edges by their computed arrival times rather than placed at the
    /// clock edge.
    #[clap(long)]
    timing_vcd: bool,
}

#[derive(Parser)]
struct CosimArgs {
    /// Gate-level Verilog path synthesized with AIGPDK or SKY130 library.
    netlist_verilog: PathBuf,

    /// Pre-compiled partition mapping (.gemparts file).
    /// If omitted, partitions are generated inline (adds ~20s).
    gemparts: Option<PathBuf>,

    /// Testbench configuration JSON file.
    #[clap(long)]
    config: PathBuf,

    /// Top module name in the netlist.
    #[clap(long)]
    top_module: Option<String>,

    /// Level split thresholds (comma-separated).
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,

    /// Number of GPU threadgroups (blocks).
    #[clap(long, default_value = "64")]
    num_blocks: usize,

    /// Maximum system clock ticks to simulate.
    #[clap(long)]
    max_cycles: Option<usize>,

    /// Enable verbose flash model debug output.
    #[clap(long)]
    flash_verbose: bool,

    /// Clock period in picoseconds.
    #[clap(long)]
    clock_period: Option<u64>,

    /// Verify GPU results against CPU baseline.
    #[clap(long)]
    check_with_cpu: bool,

    /// Run GPU kernel profiling.
    #[clap(long)]
    gpu_profile: bool,

    /// Path to SDF file for per-instance back-annotated delays.
    #[clap(long)]
    sdf: Option<PathBuf>,

    /// SDF corner selection: min, typ, or max.
    #[clap(long, default_value = "typ")]
    sdf_corner: String,

    /// Enable SDF debug output.
    #[clap(long)]
    sdf_debug: bool,

    /// Path to write stimulus VCD (all primary inputs driven by cosim).
    /// Forces single-tick mode for accurate per-cycle capture.
    #[clap(long)]
    stimulus_vcd: Option<PathBuf>,
}

fn cmd_map(args: MapArgs) {
    clilog::info!("Loom map args:\n{:#?}", args.netlist_verilog);

    // Detect cell library
    let lib = detect_library_from_file(&args.netlist_verilog).expect("Failed to read netlist file");
    clilog::info!("Detected cell library: {}", lib);

    if lib == CellLibrary::Mixed {
        panic!("Mixed AIGPDK and SKY130 cells in netlist not supported");
    }

    let netlistdb = match lib {
        CellLibrary::SKY130 => NetlistDB::from_sverilog_file(
            &args.netlist_verilog,
            args.top_module.as_deref(),
            &SKY130LeafPins,
        )
        .expect("cannot build netlist"),
        CellLibrary::AIGPDK | CellLibrary::Mixed => NetlistDB::from_sverilog_file(
            &args.netlist_verilog,
            args.top_module.as_deref(),
            &AIGPDKLeafPins(),
        )
        .expect("cannot build netlist"),
    };

    let aig = AIG::from_netlistdb(&netlistdb);
    println!(
        "netlist has {} pins, {} aig pins, {} and gates",
        netlistdb.num_pins,
        aig.num_aigpins,
        aig.and_gate_cache.len()
    );

    if args.xprop {
        let (_x_capable, stats) = aig.compute_x_capable_pins();
        println!(
            "X-propagation analysis: {}/{} pins ({:.1}%) X-capable, {} X-sources, {} fixpoint iterations",
            stats.num_x_capable_pins,
            stats.total_pins,
            if stats.total_pins > 0 {
                stats.num_x_capable_pins as f64 / stats.total_pins as f64 * 100.0
            } else {
                0.0
            },
            stats.num_x_sources,
            stats.fixpoint_iterations,
        );
    }

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let stages_effective_parts =
        setup::generate_partitions(&aig, &stageds, args.max_stage_degrad);

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stages_effective_parts).unwrap();
}

#[allow(unused_variables)]
fn cmd_sim(args: SimArgs) {
    use gem::sim::setup;
    use gem::sim::vcd_io;

    let design_args = DesignArgs {
        netlist_verilog: args.netlist_verilog.clone(),
        top_module: args.top_module.clone(),
        level_split: args.level_split.clone(),
        gemparts: args.gemparts.clone(),
        num_blocks: args.num_blocks,
        json_path: args.json_path.clone(),
        sdf: args.sdf.clone(),
        sdf_corner: args.sdf_corner.clone(),
        sdf_debug: args.sdf_debug,
        clock_period_ps: None,
        xprop: args.xprop,
    };

    #[allow(unused_mut)]
    let mut design = setup::load_design(&design_args);

    // Enable timing arrival readback if --timing-vcd is set
    if args.timing_vcd {
        if args.sdf.is_none() {
            eprintln!("Error: --timing-vcd requires --sdf");
            std::process::exit(1);
        }
        design.script.enable_timing_arrivals();
    }

    let timing_constraints = setup::build_timing_constraints(&design.script);

    // Parse input VCD
    let input_vcd = std::fs::File::open(&args.input_vcd).unwrap();
    let mut bufrd = std::io::BufReader::with_capacity(65536, input_vcd);
    let mut vcd_parser = vcd_ng::Parser::new(&mut bufrd);
    let header = vcd_parser.parse_header().unwrap();
    drop(vcd_parser);
    use std::io::{Seek, SeekFrom};
    let mut vcd_file = bufrd.into_inner();
    vcd_file.seek(SeekFrom::Start(0)).unwrap();
    let mut vcdflow = vcd_ng::FastFlow::new(vcd_file, 65536);

    // Resolve VCD scope
    let top_scope = vcd_io::resolve_vcd_scope(
        &header.items[..],
        args.input_vcd_scope.as_deref(),
        &design.netlistdb,
        args.top_module.as_deref(),
    );

    // Match VCD inputs to netlist ports
    let (vcd2inp, _) = vcd_io::match_vcd_inputs(top_scope, &design.netlistdb);

    // Parse input VCD into state vectors
    let parsed = vcd_io::parse_input_vcd(
        &mut vcdflow,
        &vcd2inp,
        &design.aig,
        &design.script,
        &design.netlistdb,
        args.max_cycles,
    );

    // Set up output VCD writer
    let write_buf = std::fs::File::create(&args.output_vcd).unwrap();
    let write_buf = std::io::BufWriter::new(write_buf);
    let mut writer = vcd_ng::Writer::new(write_buf);
    let output_mapping = vcd_io::setup_output_vcd(
        &mut writer,
        &header,
        args.output_vcd_scope.as_deref(),
        &design.netlistdb,
        &design.aig,
        &design.script,
    );

    // GPU dispatch
    let input_states = parsed.input_states;
    let offsets_timestamps = parsed.offsets_timestamps;
    let num_cycles = offsets_timestamps.len();

    #[cfg(not(any(feature = "metal", feature = "cuda", feature = "hip")))]
    {
        eprintln!(
            "loom sim requires GPU support. Build with:\n\
             \n  cargo build -r --features metal --bin loom   (macOS)\n\
             \n  cargo build -r --features cuda --bin loom    (Linux/NVIDIA)\n\
             \n  cargo build -r --features hip --bin loom     (Linux/AMD)\n"
        );
        std::process::exit(1);
    }

    #[cfg(feature = "metal")]
    let gpu_states = {
        sim_metal(
            &design,
            &input_states,
            &offsets_timestamps,
            &timing_constraints,
        )
    };

    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    let gpu_states = {
        sim_cuda(
            &design,
            &input_states,
            &offsets_timestamps,
            &timing_constraints,
        )
    };

    #[cfg(all(feature = "hip", not(feature = "metal"), not(feature = "cuda")))]
    let gpu_states = {
        sim_hip(
            &design,
            &input_states,
            &offsets_timestamps,
            &timing_constraints,
        )
    };

    // CPU sanity check
    #[cfg(any(feature = "metal", feature = "cuda", feature = "hip"))]
    if args.check_with_cpu {
        if design.script.xprop_enabled {
            let rio = design.script.reg_io_state_size as usize;
            // When timing arrivals are enabled, effective size is 3*rio instead of 2*rio.
            // Extract values and xmasks manually using the actual effective size.
            let eff = design.script.effective_state_size() as usize;
            let num_snapshots = gpu_states.len() / eff;
            let mut gpu_values_vec = Vec::with_capacity(num_snapshots * rio);
            let mut gpu_xmasks_vec = Vec::with_capacity(num_snapshots * rio);
            for snap_i in 0..num_snapshots {
                let base = snap_i * eff;
                gpu_values_vec.extend_from_slice(&gpu_states[base..base + rio]);
                gpu_xmasks_vec.extend_from_slice(&gpu_states[base + rio..base + 2 * rio]);
            }
            let (gpu_values, gpu_xmasks) = (gpu_values_vec, gpu_xmasks_vec);
            // Build input X-masks: same initial template as expand_states_for_xprop
            let num_input_snaps = input_states.len() / rio;
            let mut xmask_template = vec![0xFFFF_FFFFu32; rio];
            for &pos in design.script.input_map.values() {
                xmask_template[(pos >> 5) as usize] &= !(1u32 << (pos & 31));
            }
            let mut input_xmasks = Vec::with_capacity(num_input_snaps * rio);
            for _ in 0..num_input_snaps {
                input_xmasks.extend_from_slice(&xmask_template);
            }
            gem::sim::cpu_reference::sanity_check_cpu_xprop(
                &design.script,
                &input_states,
                &gpu_values,
                &input_xmasks,
                &gpu_xmasks,
                num_cycles,
            );
        } else if design.script.timing_arrivals_enabled {
            // Extract value-only states for CPU comparison
            let rio = design.script.reg_io_state_size as usize;
            let eff = design.script.effective_state_size() as usize;
            let num_snapshots = gpu_states.len() / eff;
            let mut values = Vec::with_capacity(num_snapshots * rio);
            for snap_i in 0..num_snapshots {
                values.extend_from_slice(&gpu_states[snap_i * eff..snap_i * eff + rio]);
            }
            gem::sim::cpu_reference::sanity_check_cpu(
                &design.script,
                &input_states,
                &values,
                num_cycles,
            );
        } else {
            gem::sim::cpu_reference::sanity_check_cpu(
                &design.script,
                &input_states,
                &gpu_states[..],
                num_cycles,
            );
        }
    }

    // Post-simulation timing analysis
    #[cfg(any(feature = "metal", feature = "cuda", feature = "hip"))]
    if args.enable_timing {
        run_timing_analysis(&mut design.aig, &args);
    }

    // Write output VCD
    #[cfg(any(feature = "metal", feature = "cuda", feature = "hip"))]
    if args.timing_vcd && design.script.timing_arrivals_enabled {
        // Timed VCD: extract arrivals and use timed writer
        let rio = design.script.reg_io_state_size as usize;
        let arrival_states = vcd_io::split_arrival_states(&gpu_states[..], &design.script);
        // Debug: show arrival statistics
        let nonzero_count = arrival_states.iter().filter(|&&v| v != 0).count();
        clilog::info!("Arrival states: {} total, {} non-zero",
            arrival_states.len(), nonzero_count);
        let xmask_states = if design.script.xprop_enabled {
            let eff = design.script.effective_state_size() as usize;
            let num_snapshots = gpu_states.len() / eff;
            let mut xmasks = Vec::with_capacity(num_snapshots * rio);
            for snap_i in 0..num_snapshots {
                let base = snap_i * eff + rio;
                xmasks.extend_from_slice(&gpu_states[base..base + rio]);
            }
            Some(xmasks)
        } else {
            None
        };
        // Extract value-only states (same as split_xprop but works for any layout)
        let eff = design.script.effective_state_size() as usize;
        let num_snapshots = gpu_states.len() / eff;
        let mut values = Vec::with_capacity(num_snapshots * rio);
        for snap_i in 0..num_snapshots {
            let base = snap_i * eff;
            values.extend_from_slice(&gpu_states[base..base + rio]);
        }
        vcd_io::write_output_vcd_timed(
            &mut writer,
            &output_mapping,
            &offsets_timestamps,
            &values,
            xmask_states.as_deref(),
            &arrival_states,
            header.timescale,
        );
    } else if design.script.xprop_enabled {
        let rio = design.script.reg_io_state_size as usize;
        let (values, xmasks) = vcd_io::split_xprop_states(&gpu_states[..], rio);
        vcd_io::write_output_vcd_xprop(
            &mut writer,
            &output_mapping,
            &offsets_timestamps,
            &values,
            &xmasks,
        );

        // X-propagation report: count X bits at primary outputs
        let eff = design.script.effective_state_size() as usize;
        let num_snapshots = gpu_states.len() / eff;
        let mut first_x_free_cycle: Option<usize> = None;
        for snap_i in 1..num_snapshots {
            let xmask_base = snap_i * eff + rio;
            let mut has_x = false;
            for &(_aigpin, pos, _vid) in &output_mapping.out2vcd {
                if pos == u32::MAX {
                    continue;
                }
                let x = gpu_states[xmask_base + (pos >> 5) as usize] >> (pos & 31) & 1;
                if x != 0 {
                    has_x = true;
                    break;
                }
            }
            if !has_x && first_x_free_cycle.is_none() {
                first_x_free_cycle = Some(snap_i - 1);
            }
        }
        if let Some(cycle) = first_x_free_cycle {
            clilog::info!("All primary outputs X-free at cycle {}", cycle);
        } else if num_snapshots > 1 {
            clilog::warn!(
                "Primary outputs still have X values at final cycle {}",
                num_snapshots - 2
            );
        }
    } else {
        vcd_io::write_output_vcd(
            &mut writer,
            &output_mapping,
            &offsets_timestamps,
            &gpu_states[..],
        );
    }
}

#[cfg(feature = "metal")]
fn sim_metal(
    design: &gem::sim::setup::LoadedDesign,
    input_states: &[u32],
    offsets_timestamps: &[(usize, u64)],
    timing_constraints: &Option<Vec<u32>>,
) -> Vec<u32> {
    use gem::aig::SimControlType;
    use gem::display::format_display_message;
    use gem::event_buffer::{
        process_events, AssertConfig, EventBuffer, EventType, SimControl, SimStats, MAX_EVENTS,
    };
    use metal::{Device as MTLDevice, MTLResourceOptions, MTLSize};
    use ulib::{AsUPtr, AsUPtrMut, Device, UVec};

    let script = &design.script;

    // Initialize Metal
    let mtl_device = MTLDevice::system_default().expect("No Metal device found");
    clilog::info!("Using Metal device: {}", mtl_device.name());

    let metallib_path = env!("METALLIB_PATH");
    let library = mtl_device
        .new_library_with_file(metallib_path)
        .expect("Failed to load metallib");
    let kernel_function = library
        .get_function("simulate_v1_stage", None)
        .expect("Failed to get kernel function");
    let pipeline_state = mtl_device
        .new_compute_pipeline_state_with_function(&kernel_function)
        .expect("Failed to create pipeline state");
    let command_queue = mtl_device.new_command_queue();

    let device = Device::Metal(0);
    let num_cycles = offsets_timestamps.len();

    // When xprop is enabled, expand the value-only state buffer to include X-mask
    let mut expanded_states = if script.xprop_enabled {
        gem::sim::vcd_io::expand_states_for_xprop(input_states, script)
    } else {
        input_states.to_vec()
    };
    // When timing arrivals are enabled, expand further to include arrival section
    if script.timing_arrivals_enabled {
        expanded_states = gem::sim::vcd_io::expand_states_for_arrivals(&expanded_states, script);
    }
    let mut input_states_uvec: UVec<_> = expanded_states.into();
    input_states_uvec.as_mut_uptr(device);
    let mut sram_storage: UVec<u32> = UVec::new_zeroed(script.sram_storage_size as usize, device);
    // SRAM X-mask shadow: all 0xFFFFFFFF (unknown) initially when xprop enabled
    let sram_xmask_size = if script.xprop_enabled {
        script.sram_storage_size as usize
    } else {
        1 // Metal requires non-zero buffer; kernel checks is_x_capable before reading
    };
    let mut sram_xmask: UVec<u32> = if script.xprop_enabled {
        // Build on CPU then transfer to GPU — UVec::new_filled() doesn't support Metal/CUDA devices
        let v: UVec<u32> = vec![0xFFFF_FFFFu32; sram_xmask_size].into();
        v
    } else {
        UVec::new_zeroed(sram_xmask_size, device)
    };

    // Get Metal buffer pointers
    let blocks_start_ptr = script.blocks_start.as_uptr(device);
    let blocks_data_ptr = script.blocks_data.as_uptr(device);
    let sram_data_ptr = sram_storage.as_mut_uptr(device);
    let sram_xmask_ptr = sram_xmask.as_mut_uptr(device);
    let states_ptr = input_states_uvec.as_mut_uptr(device);

    let blocks_start_buffer = mtl_device.new_buffer_with_bytes_no_copy(
        blocks_start_ptr as *const _,
        (script.blocks_start.len() * std::mem::size_of::<usize>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let blocks_data_buffer = mtl_device.new_buffer_with_bytes_no_copy(
        blocks_data_ptr as *const _,
        (script.blocks_data.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let sram_data_buffer = mtl_device.new_buffer_with_bytes_no_copy(
        sram_data_ptr as *mut _ as *const _,
        (sram_storage.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let states_buffer = mtl_device.new_buffer_with_bytes_no_copy(
        states_ptr as *mut _ as *const _,
        (input_states_uvec.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let sram_xmask_buffer = mtl_device.new_buffer_with_bytes_no_copy(
        sram_xmask_ptr as *mut _ as *const _,
        (sram_xmask.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );

    // Event buffer
    let event_buffer = Box::new(EventBuffer::new());
    let event_buffer_ptr = Box::into_raw(event_buffer);
    let event_buffer_metal = mtl_device.new_buffer_with_bytes_no_copy(
        event_buffer_ptr as *const _,
        std::mem::size_of::<EventBuffer>() as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );

    // Timing constraint buffer (THE GAP FIX)
    let timing_buffer = timing_constraints.as_ref().map(|buf| {
        mtl_device.new_buffer_with_data(
            buf.as_ptr() as *const _,
            (buf.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    #[repr(C)]
    struct SimParams {
        num_blocks: u64,
        num_major_stages: u64,
        num_cycles: u64,
        state_size: u64,
        current_cycle: u64,
        current_stage: u64,
        arrival_state_offset: u64,
    }

    let assert_config = AssertConfig::default();
    let mut sim_stats = SimStats::default();
    let mut cycles_completed = 0;
    let mut final_control = SimControl::Continue;

    let timer_sim = clilog::stimer!("simulation");

    for cycle_i in 0..num_cycles {
        unsafe {
            (*event_buffer_ptr).reset();
        }

        for stage_i in 0..script.num_major_stages {
            let params = SimParams {
                num_blocks: script.num_blocks as u64,
                num_major_stages: script.num_major_stages as u64,
                num_cycles: num_cycles as u64,
                state_size: script.effective_state_size() as u64,
                current_cycle: cycle_i as u64,
                current_stage: stage_i as u64,
                arrival_state_offset: script.arrival_state_offset as u64,
            };

            let params_buffer = mtl_device.new_buffer_with_data(
                &params as *const SimParams as *const _,
                std::mem::size_of::<SimParams>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&blocks_start_buffer), 0);
            encoder.set_buffer(1, Some(&blocks_data_buffer), 0);
            encoder.set_buffer(2, Some(&sram_data_buffer), 0);
            encoder.set_buffer(3, Some(&states_buffer), 0);
            encoder.set_buffer(4, Some(&params_buffer), 0);
            encoder.set_buffer(5, Some(&event_buffer_metal), 0);
            // Buffer slot 6: timing constraints (THE GAP FIX)
            encoder.set_buffer(6, timing_buffer.as_ref().map(|v| &**v), 0);
            // Buffer slot 7: SRAM X-mask shadow
            encoder.set_buffer(7, Some(&sram_xmask_buffer), 0);

            let threads_per_threadgroup = MTLSize::new(256, 1, 1);
            let threadgroups = MTLSize::new(script.num_blocks as u64, 1, 1);

            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Check assertions
        if !script.assertion_positions.is_empty() {
            let states_slice =
                unsafe { std::slice::from_raw_parts(states_ptr, input_states_uvec.len()) };
            let cycle_output_offset = (cycle_i + 1) * script.effective_state_size() as usize;

            for &(cell_id, pos, message_id, control_type) in &script.assertion_positions {
                let word_idx = (pos >> 5) as usize;
                let bit_idx = pos & 31;
                let abs_word_idx = cycle_output_offset + word_idx;
                if abs_word_idx < states_slice.len() {
                    let condition = (states_slice[abs_word_idx] >> bit_idx) & 1;
                    if condition == 1 {
                        let event_type = match control_type {
                            None => EventType::AssertFail,
                            Some(SimControlType::Stop) => EventType::Stop,
                            Some(SimControlType::Finish) => EventType::Finish,
                        };

                        unsafe {
                            let count = (*event_buffer_ptr)
                                .count
                                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                                as usize;
                            if count < MAX_EVENTS {
                                let event = &mut (*event_buffer_ptr).events[count];
                                event.event_type = event_type as u32;
                                event.message_id = message_id;
                                event.cycle = cycle_i as u32;
                            }
                        }

                        clilog::debug!(
                            "[cycle {}] Assertion condition fired: cell={}, pos={}, type={:?}",
                            cycle_i,
                            cell_id,
                            pos,
                            control_type
                        );
                    }
                }
            }
        }

        // Check display conditions
        if !script.display_positions.is_empty() {
            let states_slice =
                unsafe { std::slice::from_raw_parts(states_ptr, input_states_uvec.len()) };
            let cycle_output_offset = (cycle_i + 1) * script.effective_state_size() as usize;

            for (cell_id, enable_pos, format, arg_positions, arg_widths) in
                &script.display_positions
            {
                let word_idx = (*enable_pos >> 5) as usize;
                let bit_idx = *enable_pos & 31;
                let abs_word_idx = cycle_output_offset + word_idx;
                if abs_word_idx < states_slice.len() {
                    let enable = (states_slice[abs_word_idx] >> bit_idx) & 1;
                    if enable == 1 {
                        let mut display_args: Vec<u64> = Vec::new();
                        for &arg_pos in arg_positions {
                            let arg_word_idx = (arg_pos >> 5) as usize;
                            let arg_bit_idx = arg_pos & 31;
                            let abs_arg_idx = cycle_output_offset + arg_word_idx;
                            if abs_arg_idx < states_slice.len() {
                                let val = ((states_slice[abs_arg_idx] >> arg_bit_idx) & 1) as u64;
                                display_args.push(val);
                            }
                        }

                        let message = format_display_message(format, &display_args, arg_widths);
                        print!("{}", message);

                        unsafe {
                            let count = (*event_buffer_ptr)
                                .count
                                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                                as usize;
                            if count < MAX_EVENTS {
                                let event = &mut (*event_buffer_ptr).events[count];
                                event.event_type = EventType::Display as u32;
                                event.message_id = *cell_id as u32;
                                event.cycle = cycle_i as u32;
                            }
                        }

                        clilog::debug!(
                            "[cycle {}] Display fired: cell={}, format='{}'",
                            cycle_i,
                            cell_id,
                            format
                        );
                    }
                }
            }
        }

        // Process events
        let control = unsafe {
            process_events(
                &*event_buffer_ptr,
                &assert_config,
                &mut sim_stats,
                |msg_id, cycle, _data| {
                    clilog::debug!("[cycle {}] Event processed: message id={}", cycle, msg_id);
                },
            )
        };

        cycles_completed = cycle_i + 1;

        match control {
            SimControl::Continue => {}
            SimControl::Pause => {
                final_control = SimControl::Pause;
            }
            SimControl::Terminate => {
                final_control = SimControl::Terminate;
                break;
            }
        }
    }

    clilog::finish!(timer_sim);

    // Report simulation result
    match final_control {
        SimControl::Continue => {
            clilog::info!("Simulation completed {} cycles", cycles_completed);
        }
        SimControl::Pause => {
            clilog::info!(
                "Simulation paused at cycle {} ($stop encountered)",
                cycles_completed
            );
        }
        SimControl::Terminate => {
            clilog::info!(
                "Simulation terminated at cycle {} ($finish encountered)",
                cycles_completed
            );
        }
    }

    if sim_stats.assertion_failures > 0 {
        clilog::warn!("Total assertion failures: {}", sim_stats.assertion_failures);
    }

    // Clean up event buffer
    unsafe {
        drop(Box::from_raw(event_buffer_ptr));
    }

    input_states_uvec[..].to_vec()
}

#[cfg(feature = "cuda")]
fn sim_cuda(
    design: &gem::sim::setup::LoadedDesign,
    input_states: &[u32],
    offsets_timestamps: &[(usize, u64)],
    timing_constraints: &Option<Vec<u32>>,
) -> Vec<u32> {
    use gem::aig::SimControlType;
    use gem::display::format_display_message;
    use gem::event_buffer::{AssertAction, AssertConfig, EventType, SimStats};
    use ulib::{AsUPtrMut, Device, UVec};

    mod ucci {
        include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v1.rs"));
    }

    let script = &design.script;
    let device = Device::CUDA(0);
    let num_cycles = offsets_timestamps.len();

    // When xprop is enabled, expand the value-only state buffer to include X-mask
    let expanded_states = if script.xprop_enabled {
        gem::sim::vcd_io::expand_states_for_xprop(input_states, script)
    } else {
        input_states.to_vec()
    };
    let mut input_states_uvec: UVec<_> = expanded_states.into();
    input_states_uvec.as_mut_uptr(device);
    let mut sram_storage = UVec::new_zeroed(script.sram_storage_size as usize, device);
    // SRAM X-mask shadow: all 0xFFFFFFFF (unknown) initially when xprop enabled
    let sram_xmask_size = if script.xprop_enabled {
        script.sram_storage_size as usize
    } else {
        1 // Kernel checks is_x_capable before reading
    };
    let mut sram_xmask: UVec<u32> = if script.xprop_enabled {
        // Build on CPU then transfer to GPU — UVec::new_filled() doesn't support Metal/CUDA devices
        let v: UVec<u32> = vec![0xFFFF_FFFFu32; sram_xmask_size].into();
        v
    } else {
        UVec::new_zeroed(sram_xmask_size, device)
    };

    // Launch GPU simulation
    device.synchronize();
    let timer_sim = clilog::stimer!("simulation");

    if timing_constraints.is_some() {
        // TODO: Wire timing constraints to CUDA timed kernel variant.
        // The EventBuffer struct contains AtomicU32, which doesn't impl Copy,
        // so UVec<EventBuffer> can't satisfy the UniversalCopy trait bound.
        // For now, the simple_scan variant passes nullptr for both timing_constraints
        // and event_buffer on the C side.
        clilog::warn!(
            "Timing constraints requested but CUDA timed kernel not yet wired; \
             running without GPU-side timing checks"
        );
    }
    ucci::simulate_v1_noninteractive_simple_scan(
        script.num_blocks,
        script.num_major_stages,
        &script.blocks_start,
        &script.blocks_data,
        &mut sram_storage,
        &mut sram_xmask,
        num_cycles,
        script.effective_state_size() as usize,
        &mut input_states_uvec,
        device,
    );

    device.synchronize();
    clilog::finish!(timer_sim);

    // Process display outputs (post-sim scan)
    if !script.display_positions.is_empty() {
        clilog::info!(
            "Processing {} display nodes",
            script.display_positions.len()
        );

        let eff_size = script.effective_state_size() as usize;
        let states_slice = &input_states_uvec[eff_size..];
        for cycle_i in 0..num_cycles {
            let cycle_offset = cycle_i * eff_size;
            for (_cell_id, enable_pos, format, arg_positions, arg_widths) in
                &script.display_positions
            {
                let word_idx = (*enable_pos >> 5) as usize;
                let bit_idx = *enable_pos & 31;
                let abs_word_idx = cycle_offset + word_idx;
                if abs_word_idx < states_slice.len() {
                    let enable = (states_slice[abs_word_idx] >> bit_idx) & 1;
                    if enable == 1 {
                        let mut args: Vec<u64> = Vec::new();
                        for &arg_pos in arg_positions {
                            let arg_word_idx = (arg_pos >> 5) as usize;
                            let arg_bit_idx = arg_pos & 31;
                            let abs_arg_idx = cycle_offset + arg_word_idx;
                            if abs_arg_idx < states_slice.len() {
                                let val = ((states_slice[abs_arg_idx] >> arg_bit_idx) & 1) as u64;
                                args.push(val);
                            }
                        }
                        let message = format_display_message(format, &args, arg_widths);
                        print!("{}", message);
                    }
                }
            }
        }
    }

    // Process assertion conditions (post-sim scan)
    if !script.assertion_positions.is_empty() {
        clilog::info!(
            "Processing {} assertion nodes",
            script.assertion_positions.len()
        );

        let assert_config = AssertConfig::default();
        let mut sim_stats = SimStats::default();

        let eff_size = script.effective_state_size() as usize;
        let states_slice = &input_states_uvec[eff_size..];
        for cycle_i in 0..num_cycles {
            let cycle_offset = cycle_i * eff_size;
            for &(_cell_id, pos, _message_id, control_type) in &script.assertion_positions {
                let word_idx = (pos >> 5) as usize;
                let bit_idx = pos & 31;
                let abs_word_idx = cycle_offset + word_idx;
                if abs_word_idx < states_slice.len() {
                    let condition = (states_slice[abs_word_idx] >> bit_idx) & 1;
                    if condition == 1 {
                        let event_type = match control_type {
                            None => EventType::AssertFail,
                            Some(SimControlType::Stop) => EventType::Stop,
                            Some(SimControlType::Finish) => EventType::Finish,
                        };
                        clilog::warn!(
                            "[cycle {}] Assertion condition fired: pos={}, type={:?}",
                            cycle_i,
                            pos,
                            control_type
                        );
                        match (event_type, assert_config.on_failure) {
                            (EventType::AssertFail, AssertAction::Log) => {
                                sim_stats.assertion_failures += 1;
                            }
                            (EventType::AssertFail, AssertAction::Terminate) => {
                                clilog::error!("Assertion failed - terminating simulation");
                                sim_stats.assertion_failures += 1;
                                std::process::exit(1);
                            }
                            (EventType::Stop, _) => {
                                clilog::info!("$stop encountered at cycle {}", cycle_i);
                                sim_stats.stop_count += 1;
                                break;
                            }
                            (EventType::Finish, _) => {
                                clilog::info!("$finish encountered at cycle {}", cycle_i);
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        if sim_stats.assertion_failures > 0 {
            clilog::warn!(
                "Simulation completed with {} assertion failures",
                sim_stats.assertion_failures
            );
        }
    }

    input_states_uvec[..].to_vec()
}

#[cfg(feature = "hip")]
fn sim_hip(
    design: &gem::sim::setup::LoadedDesign,
    input_states: &[u32],
    offsets_timestamps: &[(usize, u64)],
    timing_constraints: &Option<Vec<u32>>,
) -> Vec<u32> {
    use gem::aig::SimControlType;
    use gem::display::format_display_message;
    use gem::event_buffer::{AssertAction, AssertConfig, EventType, SimStats};
    use ulib::{AsUPtrMut, Device, UVec};

    mod ucci_hip {
        include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v1_hip.rs"));
    }

    let script = &design.script;
    let device = Device::HIP(0);
    let num_cycles = offsets_timestamps.len();

    // When xprop is enabled, expand the value-only state buffer to include X-mask
    let expanded_states = if script.xprop_enabled {
        gem::sim::vcd_io::expand_states_for_xprop(input_states, script)
    } else {
        input_states.to_vec()
    };
    let mut input_states_uvec: UVec<_> = expanded_states.into();
    input_states_uvec.as_mut_uptr(device);
    let mut sram_storage = UVec::new_zeroed(script.sram_storage_size as usize, device);
    // SRAM X-mask shadow: all 0xFFFFFFFF (unknown) initially when xprop enabled
    let sram_xmask_size = if script.xprop_enabled {
        script.sram_storage_size as usize
    } else {
        1 // Kernel checks is_x_capable before reading
    };
    let mut sram_xmask: UVec<u32> = if script.xprop_enabled {
        let v: UVec<u32> = vec![0xFFFF_FFFFu32; sram_xmask_size].into();
        v
    } else {
        UVec::new_zeroed(sram_xmask_size, device)
    };

    // Launch GPU simulation
    device.synchronize();
    let timer_sim = clilog::stimer!("simulation");

    if timing_constraints.is_some() {
        clilog::warn!(
            "Timing constraints requested but HIP timed kernel not yet wired; \
             running without GPU-side timing checks"
        );
    }
    ucci_hip::simulate_v1_noninteractive_simple_scan(
        script.num_blocks,
        script.num_major_stages,
        &script.blocks_start,
        &script.blocks_data,
        &mut sram_storage,
        &mut sram_xmask,
        num_cycles,
        script.effective_state_size() as usize,
        &mut input_states_uvec,
        device,
    );

    device.synchronize();
    clilog::finish!(timer_sim);

    // Process display outputs (post-sim scan)
    if !script.display_positions.is_empty() {
        clilog::info!(
            "Processing {} display nodes",
            script.display_positions.len()
        );

        let eff_size = script.effective_state_size() as usize;
        let states_slice = &input_states_uvec[eff_size..];
        for cycle_i in 0..num_cycles {
            let cycle_offset = cycle_i * eff_size;
            for (_cell_id, enable_pos, format, arg_positions, arg_widths) in
                &script.display_positions
            {
                let word_idx = (*enable_pos >> 5) as usize;
                let bit_idx = *enable_pos & 31;
                let abs_word_idx = cycle_offset + word_idx;
                if abs_word_idx < states_slice.len() {
                    let enable = (states_slice[abs_word_idx] >> bit_idx) & 1;
                    if enable == 1 {
                        let mut args: Vec<u64> = Vec::new();
                        for &arg_pos in arg_positions {
                            let arg_word_idx = (arg_pos >> 5) as usize;
                            let arg_bit_idx = arg_pos & 31;
                            let abs_arg_idx = cycle_offset + arg_word_idx;
                            if abs_arg_idx < states_slice.len() {
                                let val = ((states_slice[abs_arg_idx] >> arg_bit_idx) & 1) as u64;
                                args.push(val);
                            }
                        }
                        let message = format_display_message(format, &args, arg_widths);
                        print!("{}", message);
                    }
                }
            }
        }
    }

    // Process assertion conditions (post-sim scan)
    if !script.assertion_positions.is_empty() {
        clilog::info!(
            "Processing {} assertion nodes",
            script.assertion_positions.len()
        );

        let assert_config = AssertConfig::default();
        let mut sim_stats = SimStats::default();

        let eff_size = script.effective_state_size() as usize;
        let states_slice = &input_states_uvec[eff_size..];
        for cycle_i in 0..num_cycles {
            let cycle_offset = cycle_i * eff_size;
            for &(_cell_id, pos, _message_id, control_type) in &script.assertion_positions {
                let word_idx = (pos >> 5) as usize;
                let bit_idx = pos & 31;
                let abs_word_idx = cycle_offset + word_idx;
                if abs_word_idx < states_slice.len() {
                    let condition = (states_slice[abs_word_idx] >> bit_idx) & 1;
                    if condition == 1 {
                        let event_type = match control_type {
                            None => EventType::AssertFail,
                            Some(SimControlType::Stop) => EventType::Stop,
                            Some(SimControlType::Finish) => EventType::Finish,
                        };
                        clilog::warn!(
                            "[cycle {}] Assertion condition fired: pos={}, type={:?}",
                            cycle_i,
                            pos,
                            control_type
                        );
                        match (event_type, assert_config.on_failure) {
                            (EventType::AssertFail, AssertAction::Log) => {
                                sim_stats.assertion_failures += 1;
                            }
                            (EventType::AssertFail, AssertAction::Terminate) => {
                                clilog::error!("Assertion failed - terminating simulation");
                                sim_stats.assertion_failures += 1;
                                std::process::exit(1);
                            }
                            (EventType::Stop, _) => {
                                clilog::info!("$stop encountered at cycle {}", cycle_i);
                                sim_stats.stop_count += 1;
                                break;
                            }
                            (EventType::Finish, _) => {
                                clilog::info!("$finish encountered at cycle {}", cycle_i);
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        if sim_stats.assertion_failures > 0 {
            clilog::warn!(
                "Simulation completed with {} assertion failures",
                sim_stats.assertion_failures
            );
        }
    }

    input_states_uvec[..].to_vec()
}

#[cfg(any(feature = "metal", feature = "cuda", feature = "hip"))]
fn run_timing_analysis(aig: &mut AIG, args: &SimArgs) {
    use gem::liberty_parser::TimingLibrary;

    clilog::info!("Running timing analysis on GPU simulation results...");
    let timer_timing = clilog::stimer!("timing_analysis");

    let lib = if let Some(lib_path) = &args.liberty {
        TimingLibrary::from_file(lib_path).expect("Failed to load Liberty library")
    } else {
        TimingLibrary::load_aigpdk().expect("Failed to load default AIGPDK library")
    };
    clilog::info!("Loaded Liberty library: {}", lib.name);

    aig.load_timing_library(&lib);
    aig.clock_period_ps = args.timing_clock_period;

    let report = aig.compute_timing();
    println!();
    println!("{}", report);
    println!(
        "Clock period: {} ps ({:.3} ns)",
        args.timing_clock_period,
        args.timing_clock_period as f64 / 1000.0
    );
    println!();

    println!("=== Critical Paths (Top 5) ===");
    let critical_paths = aig.get_critical_paths(5);
    for (i, (endpoint, arrival)) in critical_paths.iter().enumerate() {
        let slack = args.timing_clock_period as i64 - *arrival as i64;
        println!(
            "#{}: endpoint aigpin {} arrival={} ps, slack={} ps",
            i + 1,
            endpoint,
            arrival,
            slack
        );
    }
    println!();

    if args.timing_report_violations && report.has_violations() {
        println!("=== Timing Violations ===");
        for (i, ((_cell_id, dff), (&setup_slack, &hold_slack))) in aig
            .dffs
            .iter()
            .zip(aig.setup_slacks.iter().zip(aig.hold_slacks.iter()))
            .enumerate()
        {
            if setup_slack < 0 || hold_slack < 0 {
                println!("DFF #{}: D aigpin {}", i, dff.d_iv >> 1);
                if setup_slack < 0 {
                    println!("  SETUP VIOLATION: slack = {} ps", setup_slack);
                }
                if hold_slack < 0 {
                    println!("  HOLD VIOLATION: slack = {} ps", hold_slack);
                }
            }
        }
        println!();
    }

    if report.has_violations() {
        clilog::warn!(
            "TIMING ANALYSIS: FAILED ({} setup, {} hold violations)",
            report.setup_violations,
            report.hold_violations
        );
    } else {
        clilog::info!("TIMING ANALYSIS: PASSED");
    }

    clilog::finish!(timer_timing);
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let cli = Cli::parse();

    match cli.command {
        Commands::Map(args) => cmd_map(args),
        Commands::Sim(args) => cmd_sim(args),
        Commands::Cosim(args) => cmd_cosim(args),
    }
}

#[allow(unused_variables)]
fn cmd_cosim(args: CosimArgs) {
    #[cfg(not(feature = "metal"))]
    {
        eprintln!(
            "loom cosim requires Metal support (macOS only). Build with:\n\
             \n  cargo build -r --features metal --bin loom\n"
        );
        std::process::exit(1);
    }

    #[cfg(feature = "metal")]
    {
        use gem::sim::cosim_metal::CosimOpts;
        use gem::sim::setup;
        use gem::testbench::TestbenchConfig;

        // Load testbench config
        let file = std::fs::File::open(&args.config).expect("Failed to open config file");
        let reader = std::io::BufReader::new(file);
        let config: TestbenchConfig =
            serde_json::from_reader(reader).expect("Failed to parse config JSON");
        clilog::info!("Loaded testbench config: {:?}", config);

        // Determine clock period for SDF loading
        let clock_period_ps = args
            .clock_period
            .or(config.clock_period_ps)
            .or(config.timing.as_ref().map(|t| t.clock_period_ps));

        // Determine SDF path: CLI --sdf takes priority, then config.timing.sdf_file
        let sdf = args.sdf.clone().or_else(|| {
            config
                .timing
                .as_ref()
                .map(|t| std::path::PathBuf::from(&t.sdf_file))
        });
        let sdf_corner = if args.sdf.is_some() {
            args.sdf_corner.clone()
        } else if let Some(ref t) = config.timing {
            t.sdf_corner.clone()
        } else {
            "typ".to_string()
        };
        let sdf_debug = args.sdf_debug;

        let design_args = DesignArgs {
            netlist_verilog: args.netlist_verilog.clone(),
            top_module: args.top_module.clone(),
            level_split: args.level_split.clone(),
            gemparts: args.gemparts.clone(),
            num_blocks: args.num_blocks,
            json_path: None,
            sdf,
            sdf_corner,
            sdf_debug,
            clock_period_ps,
            xprop: false, // cosim doesn't support xprop yet
        };

        let mut design = setup::load_design(&design_args);
        let timing_constraints = setup::build_timing_constraints(&design.script);

        let opts = CosimOpts {
            max_cycles: args.max_cycles,
            num_blocks: args.num_blocks,
            flash_verbose: args.flash_verbose,
            check_with_cpu: args.check_with_cpu,
            gpu_profile: args.gpu_profile,
            clock_period: args.clock_period,
            stimulus_vcd: args.stimulus_vcd.clone(),
        };

        let result =
            gem::sim::cosim_metal::run_cosim(&mut design, &config, &opts, &timing_constraints);
        std::process::exit(if result.passed { 0 } else { 1 });
    }
}
