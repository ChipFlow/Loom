// CVC SDF-annotated testbench for multi_depth timing validation.
//
// This testbench validates timing of the multi_depth design with
// SDF back-annotation. 5 groups of 8 outputs at different logic depths
// exercise Source 3 overestimation in timing-aware bit packing.
//
// Run with CVC:
//   cvc64 +typdelays tb_cvc.v multi_depth.v
//   ./cvcsim
//
// Expected typ-corner arrivals (from CLK posedge):
//   Group A out[0..7]:   depth 3  -> ~530ps
//   Group B out[8..15]:  depth 5  -> ~650ps
//   Group C out[16..23]: depth 9  -> ~890ps
//   Group D out[24..31]: depth 13 -> ~1130ps
//   Group E out[32..39]: depth 17 -> ~1370ps

`timescale 1ps/1ps

// DFF UDP primitive -- CVC requires gate/UDP-based outputs for specify
// path delays to take effect (procedural always blocks don't work).
primitive dff_udp(Q, CLK, D);
  output Q;
  reg Q;
  input CLK, D;
  table
    // CLK  D : Q : Q+
       r    0 : ? : 0;
       r    1 : ? : 1;
       n    ? : ? : -;
       ?    * : ? : -;
  endtable
endprimitive

module sky130_fd_sc_hd__dfxtp_1 (CLK, D, Q);
  input CLK, D;
  output Q;

  dff_udp u(Q, CLK, D);

  specify
    (posedge CLK => (Q +: D)) = (0, 0);
    $setup(D, posedge CLK, 0);
    $hold(posedge CLK, D, 0);
  endspecify
endmodule

module sky130_fd_sc_hd__inv_1 (A, Y);
  input A;
  output Y;

  assign Y = ~A;

  specify
    (A => Y) = (0, 0);
  endspecify
endmodule

module tb_cvc;
  reg CLK, D;
  wire [39:0] out;

  multi_depth uut (
    .CLK(CLK),
    .D(D),
    .out(out)
  );

  // VCD output for comparison with Loom
  initial begin
    $dumpfile("cvc_multi_depth_output.vcd");
    $dumpvars(0, uut);
  end

  // SDF annotation
  initial begin
    $sdf_annotate("multi_depth.sdf", uut, , , "TYPICAL");
  end

  // Clock: 10ns period (5000ps high, 5000ps low)
  initial begin
    CLK = 0;
    forever #5000 CLK = ~CLK;
  end

  // Stimulus: toggle D to exercise transitions
  initial begin
    D = 0;

    // Wait for first rising clock edge (cycle 0 latches D=0)
    @(posedge CLK);
    #100;
    D = 1;  // Will be latched at next posedge (cycle 1)

    @(posedge CLK);
    #100;
    D = 0;  // Will be latched at next posedge (cycle 2)

    @(posedge CLK);
    #100;
    D = 1;  // Will be latched at next posedge (cycle 3)

    @(posedge CLK);
    #100;
    D = 0;  // Will be latched at next posedge (cycle 4)

    // Let simulation run for output propagation
    repeat(4) @(posedge CLK);
  end

  // Track per-group arrival times
  realtime t_clk_edge;
  realtime t_q1_rise;
  realtime t_grp_a, t_grp_b, t_grp_c, t_grp_d, t_grp_e;

  initial begin
    t_clk_edge = 0;
    t_q1_rise = 0;
    t_grp_a = 0;
    t_grp_b = 0;
    t_grp_c = 0;
    t_grp_d = 0;
    t_grp_e = 0;
  end

  // Detect the clock edge that latches D=1
  always @(posedge CLK) begin
    if (D === 1'b1 && uut.q1 === 1'b0 && t_clk_edge == 0) begin
      t_clk_edge = $realtime;
      $display("TIMING: CLK posedge at %0t ps (latching D=1)", $realtime);
    end
  end

  // DFF output
  always @(uut.q1) begin
    if (uut.q1 === 1'b1 && t_clk_edge > 0 && t_q1_rise == 0) begin
      t_q1_rise = $realtime;
      $display("TIMING: dff_in.Q -> 1 at %0t ps (CLK->Q = %0t ps)",
               $realtime, $realtime - t_clk_edge);
    end
  end

  // Group A: out[0] (representative of depth-3 group)
  always @(out[0]) begin
    if (t_clk_edge > 0 && t_grp_a == 0) begin
      t_grp_a = $realtime;
      $display("TIMING: out[0] (grp A, depth 3) -> %b at %0t ps (from CLK: %0t ps)",
               out[0], $realtime, $realtime - t_clk_edge);
    end
  end

  // Group B: out[8] (representative of depth-5 group)
  always @(out[8]) begin
    if (t_clk_edge > 0 && t_grp_b == 0) begin
      t_grp_b = $realtime;
      $display("TIMING: out[8] (grp B, depth 5) -> %b at %0t ps (from CLK: %0t ps)",
               out[8], $realtime, $realtime - t_clk_edge);
    end
  end

  // Group C: out[16] (representative of depth-9 group)
  always @(out[16]) begin
    if (t_clk_edge > 0 && t_grp_c == 0) begin
      t_grp_c = $realtime;
      $display("TIMING: out[16] (grp C, depth 9) -> %b at %0t ps (from CLK: %0t ps)",
               out[16], $realtime, $realtime - t_clk_edge);
    end
  end

  // Group D: out[24] (representative of depth-13 group)
  always @(out[24]) begin
    if (t_clk_edge > 0 && t_grp_d == 0) begin
      t_grp_d = $realtime;
      $display("TIMING: out[24] (grp D, depth 13) -> %b at %0t ps (from CLK: %0t ps)",
               out[24], $realtime, $realtime - t_clk_edge);
    end
  end

  // Group E: out[32] (representative of depth-17 group)
  always @(out[32]) begin
    if (t_clk_edge > 0 && t_grp_e == 0) begin
      t_grp_e = $realtime;
      $display("TIMING: out[32] (grp E, depth 17) -> %b at %0t ps (from CLK: %0t ps)",
               out[32], $realtime, $realtime - t_clk_edge);
    end
  end

  // Summary and finish
  initial begin
    #80000;

    $display("");
    $display("=== CVC Multi-Depth Delay Summary ===");
    if (t_grp_e > 0 && t_clk_edge > 0) begin
      $display("RESULT: clk_to_q=%0t", t_q1_rise - t_clk_edge);
      $display("RESULT: grp_a_delay=%0t", t_grp_a - t_clk_edge);
      $display("RESULT: grp_b_delay=%0t", t_grp_b - t_clk_edge);
      $display("RESULT: grp_c_delay=%0t", t_grp_c - t_clk_edge);
      $display("RESULT: grp_d_delay=%0t", t_grp_d - t_clk_edge);
      $display("RESULT: grp_e_delay=%0t", t_grp_e - t_clk_edge);
      $display("");
      $display("RESULT: grp_a_from_q=%0t", t_grp_a - t_q1_rise);
      $display("RESULT: grp_b_from_q=%0t", t_grp_b - t_q1_rise);
      $display("RESULT: grp_c_from_q=%0t", t_grp_c - t_q1_rise);
      $display("RESULT: grp_d_from_q=%0t", t_grp_d - t_q1_rise);
      $display("RESULT: grp_e_from_q=%0t", t_grp_e - t_q1_rise);
    end else begin
      $display("ERROR: Signals did not propagate as expected");
      $display("  t_clk_edge=%0t t_q1_rise=%0t", t_clk_edge, t_q1_rise);
      $display("  t_grp_a=%0t t_grp_b=%0t t_grp_c=%0t t_grp_d=%0t t_grp_e=%0t",
               t_grp_a, t_grp_b, t_grp_c, t_grp_d, t_grp_e);
    end

    $finish;
  end
endmodule
