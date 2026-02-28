// CVC SDF-annotated testbench for inv_chain timing validation.
//
// This testbench is designed for CVC (open-src-cvc) and validates timing
// of the inv_chain design with SDF back-annotation. Results can be compared
// against Loom's timing_sim_cpu for correctness verification.
//
// Run with CVC:
//   cvc64 +typdelays tb_cvc.v inv_chain.v
//   ./cvcsim
//
// Expected output (IOPATH + INTERCONNECT, typ corner):
//   CLK->Q (dff_in):   350ps
//   q1->c[15] (chain): ~973ps (IOPATH 747ps + INTERCONNECT 226ps)
//   Total CLK->c[15]:  ~1323ps
//
// CVC handles both IOPATH and INTERCONNECT delays from SDF, unlike iverilog
// which only applies IOPATH delays through specify blocks.

`timescale 1ps/1ps

// DFF UDP primitive — CVC requires gate/UDP-based outputs for specify
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
  wire Q;

  inv_chain uut (
    .CLK(CLK),
    .D(D),
    .Q(Q)
  );

  // VCD output for comparison with timing_sim_cpu
  initial begin
    $dumpfile("cvc_inv_chain_output.vcd");
    $dumpvars(0, uut);
  end

  // SDF annotation
  initial begin
    $sdf_annotate("inv_chain_test_ps.sdf", uut, , , "TYPICAL");
  end

  // Clock: 10ns period (5000ps high, 5000ps low)
  initial begin
    CLK = 0;
    forever #5000 CLK = ~CLK;
  end

  // Stimulus: matches the pattern used by inv_chain_stimulus.vcd for Loom
  // Cycle 0: D=0 latched
  // Cycle 1: D=1 latched (set D=1 shortly after cycle 0 clock edge)
  // Cycle 2: D=0 latched (set D=0 shortly after cycle 1 clock edge)
  // Cycle 3: D=1 latched
  // Continue toggling for several cycles
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

    // Let simulation run a few more cycles for output DFF to capture
    repeat(4) @(posedge CLK);
  end

  // Track signal transitions for timing measurement.
  // Use a single always block to avoid race conditions between
  // concurrent always blocks on the same edge.
  realtime t_clk_edge;
  realtime t_q1_rise;
  realtime t_c15_change;

  initial begin
    t_clk_edge = 0;
    t_q1_rise = 0;
    t_c15_change = 0;
  end

  // Detect the clock edge that latches D=1: D is 1 and q1 is still 0
  always @(posedge CLK) begin
    if (D === 1'b1 && uut.q1 === 1'b0 && t_clk_edge == 0) begin
      t_clk_edge = $realtime;
      $display("TIMING: CLK posedge at %0t ps (latching D=1)", $realtime);
    end
  end

  always @(uut.q1) begin
    if (uut.q1 === 1'b1 && t_clk_edge > 0 && t_q1_rise == 0) begin
      t_q1_rise = $realtime;
      $display("TIMING: dff_in.Q -> 1 at %0t ps (CLK->Q = %0t ps)",
               $realtime, $realtime - t_clk_edge);
    end
  end

  always @(uut.c[15]) begin
    if (t_clk_edge > 0 && t_c15_change == 0) begin
      t_c15_change = $realtime;
      $display("TIMING: c[15] -> %b at %0t ps (from CLK: %0t ps)",
               uut.c[15], $realtime, $realtime - t_clk_edge);
    end
  end

  always @(Q) begin
    if (t_clk_edge > 0) begin
      $display("TIMING: Q -> %b at %0t ps", Q, $realtime);
    end
  end

  // Summary and finish
  initial begin
    // Wait long enough for all propagation (8 full clock cycles)
    #80000;

    $display("");
    $display("=== CVC Delay Summary ===");
    if (t_c15_change > 0 && t_clk_edge > 0) begin
      $display("RESULT: clk_to_q=%0t", t_q1_rise - t_clk_edge);
      $display("RESULT: chain_delay=%0t", t_c15_change - t_q1_rise);
      $display("RESULT: total_delay=%0t", t_c15_change - t_clk_edge);
    end else begin
      $display("ERROR: Signals did not propagate as expected");
      $display("  t_clk_edge=%0t t_q1_rise=%0t t_c15_change=%0t",
               t_clk_edge, t_q1_rise, t_c15_change);
    end

    $finish;
  end
endmodule
