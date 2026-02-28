// CVC SDF-annotated testbench for MCU SoC timing validation.
//
// Replays stimulus captured from Loom cosim and compares against CVC's
// event-driven SDF simulation as a reference.
//
// Setup:
//   1. Generate stimulus: loom cosim ... --stimulus-vcd stimulus.vcd
//   2. Convert: uv run convert_stimulus.py stimulus.vcd stimulus_gen.v
//   3. Generate cell models: uv run gen_cell_models.py
//   4. Run CVC:
//        cvc64 +typdelays tb_cvc.v sky130_cells.v 6_final.v && ./cvcsim
//
// Outputs:
//   cvc_output.vcd  — GPIO output waveforms for comparison with Loom

`timescale 1ps/1ps

module tb_cvc;

  // ---- Input registers (driven by stimulus) ----
  reg por_l;
  reg porb_h;
  reg porb_l;
  reg resetb_h;
  reg resetb_l;
  reg [43:0] gpio_in;
  reg [43:0] gpio_in_h;
  reg [43:0] gpio_loopback_one;
  reg [43:0] gpio_loopback_zero;
  reg [31:0] mask_rev;

  // ---- Output wires ----
  wire [43:0] gpio_out;
  wire [43:0] gpio_oeb;
  wire [43:0] gpio_analog_en;
  wire [43:0] gpio_analog_pol;
  wire [43:0] gpio_analog_sel;
  wire [43:0] gpio_dm0;
  wire [43:0] gpio_dm1;
  wire [43:0] gpio_dm2;
  wire [43:0] gpio_holdover;
  wire [43:0] gpio_ib_mode_sel;
  wire [43:0] gpio_inp_dis;
  wire [43:0] gpio_slow_sel;
  wire [43:0] gpio_vtrip_sel;

  // ---- Inout wires (unused, active-low/high tied off) ----
  wire [43:0] analog_io;
  wire [43:0] analog_noesd_io;

  // ---- DUT ----
  openframe_project_wrapper uut (
    .por_l(por_l),
    .porb_h(porb_h),
    .porb_l(porb_l),
    .resetb_h(resetb_h),
    .resetb_l(resetb_l),
    .analog_io(analog_io),
    .analog_noesd_io(analog_noesd_io),
    .gpio_analog_en(gpio_analog_en),
    .gpio_analog_pol(gpio_analog_pol),
    .gpio_analog_sel(gpio_analog_sel),
    .gpio_dm0(gpio_dm0),
    .gpio_dm1(gpio_dm1),
    .gpio_dm2(gpio_dm2),
    .gpio_holdover(gpio_holdover),
    .gpio_ib_mode_sel(gpio_ib_mode_sel),
    .gpio_in(gpio_in),
    .gpio_in_h(gpio_in_h),
    .gpio_inp_dis(gpio_inp_dis),
    .gpio_loopback_one(gpio_loopback_one),
    .gpio_loopback_zero(gpio_loopback_zero),
    .gpio_oeb(gpio_oeb),
    .gpio_out(gpio_out),
    .gpio_slow_sel(gpio_slow_sel),
    .gpio_vtrip_sel(gpio_vtrip_sel),
    .mask_rev(mask_rev)
  );

  // ---- SDF annotation ----
  // The SDF targets (CELLTYPE "top") (INSTANCE), so scope it to the inner
  // `top` module instance inside the openframe_project_wrapper.
  initial begin
    $sdf_annotate("6_final_nocheck.sdf", uut.top_inst, , , "TYPICAL");
  end

  // ---- VCD output (only GPIO outputs, not full hierarchy) ----
  initial begin
    $dumpfile("cvc_output.vcd");
    // Dump specific output signals, not full hierarchy (too large)
    $dumpvars(0, gpio_out);
    $dumpvars(0, gpio_oeb);
    $dumpvars(0, gpio_in);
  end

  // ---- Initial values ----
  initial begin
    por_l = 1'b1;
    porb_h = 1'b1;
    porb_l = 1'b1;
    resetb_h = 1'b1;
    resetb_l = 1'b1;
    gpio_in = 44'h0;
    gpio_in_h = 44'h0;
    gpio_loopback_one = 44'h0;
    gpio_loopback_zero = 44'h0;
    mask_rev = 32'h0;
  end

  // ---- Stimulus (generated from Loom cosim VCD) ----
  `include "stimulus_gen.v"

endmodule
