// Iverilog SDF-annotated testbench for inv_chain timing validation.
//
// This testbench applies SDF timing to the inv_chain design and measures
// signal propagation delays through the 16-inverter chain between two DFFs.
//
// Run:
//   iverilog -o sim.vvp -g2005-sv -gspecify -Ttyp \
//     tb_sdf.v inv_chain.v
//   vvp sim.vvp
//
// NOTE: -gspecify is required for specify blocks / SDF annotation.
//       Do NOT use -DFUNCTIONAL — that disables specify blocks.
//
// Expected output (IOPATH-only, no INTERCONNECT wire delays):
//   CLK→Q (dff_in):  350ps
//   q1→c[15] (inverter chain, alternating rise/fall): 747ps
//   Total CLK→c[15]: 1097ps
//
// GEM analytical value (IOPATH + wire delays, pessimistic): 1323ps rise, 1125ps fall.
// The difference (226ps) is INTERCONNECT wire delays, which iverilog's specify-based
// simulation does not apply (known limitation). The iverilog result validates
// the IOPATH component of our analytical computation.

`timescale 1ps/1ps

// Minimal behavioral cell models for SDF annotation.
// These have specify blocks that accept SDF back-annotation.

module sky130_fd_sc_hd__dfxtp_1 (CLK, D, Q);
  input CLK, D;
  output Q;
  reg Q;

  always @(posedge CLK) Q <= D;

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

module tb_sdf;
  reg CLK, D;
  wire Q;
  wire q1;          // dff_in.Q
  wire [15:0] c;    // inverter chain outputs

  inv_chain uut (
    .CLK(CLK),
    .D(D),
    .Q(Q)
  );

  // Hierarchical access to internal signals
  assign q1 = uut.q1;
  assign c  = uut.c;

  initial begin
    // Use the ps-timescale SDF (inv_chain_test_ps.sdf) because iverilog 12
    // truncates fractional values when converting from ns to ps timescale.
    // The ps SDF has identical delays but with integer picosecond values.
    $sdf_annotate("inv_chain_test_ps.sdf", uut);
  end

  // Clock: 10ns period (5ns high, 5ns low) = 10000ps
  initial begin
    CLK = 0;
    forever #5000 CLK = ~CLK;
  end

  // Stimulus: toggle D at different times to observe propagation
  initial begin
    D = 0;
    // Wait for first rising clock edge to latch D=0
    @(posedge CLK);
    #100;  // small offset after clock edge

    // Set D=1 before next clock edge
    D = 1;

    // Wait for the clock edge that latches D=1
    @(posedge CLK);

    // Now dff_in.Q should go high after CLK→Q delay (~350ps typ)
    // Then the inverter chain propagates through c[0]..c[15]

    // Monitor chain propagation
    $display("=== SDF Timing Validation ===");
    $display("Clock period: 10000ps");
    $display("");
  end

  // Track signal transitions with timestamps
  realtime t_clk_edge;
  realtime t_q1_change;
  realtime t_c0_change;
  realtime t_c15_change;

  always @(posedge CLK) begin
    if (D === 1'b1 && q1 === 1'b0) begin
      // This is the clock edge that will latch D=1
      t_clk_edge = $realtime;
      $display("CLK posedge at %0t ps (latching D=1)", $realtime);
    end
  end

  always @(q1) begin
    if (q1 === 1'b1 && t_clk_edge > 0) begin
      t_q1_change = $realtime;
      $display("dff_in.Q  -> 1 at %0t ps (CLK->Q = %0t ps)", $realtime, $realtime - t_clk_edge);
    end
  end

  always @(c[0]) begin
    if (t_q1_change > 0) begin
      t_c0_change = $realtime;
      $display("c[0]      -> %b at %0t ps (from q1: %0t ps)", c[0], $realtime, $realtime - t_q1_change);
    end
  end

  always @(c[15]) begin
    if (t_clk_edge > 0) begin
      t_c15_change = $realtime;
      $display("c[15]     -> %b at %0t ps (from CLK: %0t ps)", c[15], $realtime, $realtime - t_clk_edge);
    end
  end

  always @(Q) begin
    if (t_clk_edge > 0) begin
      $display("Q (output)-> %b at %0t ps", Q, $realtime);
    end
  end

  // Summary after signals settle
  initial begin
    // Wait enough time for full propagation (2 clock cycles)
    #25000;

    $display("");
    $display("=== Delay Summary ===");
    if (t_c15_change > 0 && t_clk_edge > 0) begin
      $display("Total combo delay (CLK edge → c[15]): %0t ps", t_c15_change - t_clk_edge);
      $display("  CLK→Q (dff_in):  %0t ps", t_q1_change - t_clk_edge);
      $display("  q1→c[15] (inv chain): %0t ps", t_c15_change - t_q1_change);
    end else begin
      $display("ERROR: Signals did not propagate as expected");
    end

    $display("");
    $display("Expected IOPATH-only (iverilog): CLK→c[15] = 1097ps (350 + 747)");
    $display("GEM analytical (IOPATH+wire):    CLK→c[15] = 1323ps (350 + 973)");
    $display("Actual measured:                 CLK→c[15] = %0t ps", t_c15_change - t_clk_edge);

    $finish;
  end
endmodule
