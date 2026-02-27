/* SKY130 Multi-Depth Timing Test: 40 outputs at 5 logic depths
 *
 * Architecture:
 *                +-- inv x2  -- 8 buf invs -- out[0..7]    (depth 3)
 *                +-- inv x4  -- 8 buf invs -- out[8..15]   (depth 5)
 *   D -> DFF(q1) +-- inv x8  -- 8 buf invs -- out[16..23]  (depth 9)
 *                +-- inv x12 -- 8 buf invs -- out[24..31]  (depth 13)
 *                +-- inv x16 -- 8 buf invs -- out[32..39]  (depth 17)
 *
 * Each output is a distinct sky130_fd_sc_hd__inv_1 buffer cell, so
 * each is a distinct AIG pin. This exercises Source 3 overestimation
 * in timing-aware bit packing (32-bit thread groups).
 *
 * Total: 1 dfxtp + 42 chain inverters + 40 buffer inverters = 83 cells.
 */

module multi_depth(CLK, D, out);
  input CLK;
  wire CLK;
  input D;
  wire D;
  output [39:0] out;
  wire [39:0] out;

  wire q1;

  // Chain wires
  wire [1:0] c2;    // 2-inverter chain
  wire [3:0] c4;    // 4-inverter chain
  wire [7:0] c8;    // 8-inverter chain
  wire [11:0] c12;  // 12-inverter chain
  wire [15:0] c16;  // 16-inverter chain

  // Input DFF
  sky130_fd_sc_hd__dfxtp_1 dff_in (
    .CLK(CLK),
    .D(D),
    .Q(q1)
  );

  // === Chain A: 2 inverters (depth 3 with buffer) ===
  sky130_fd_sc_hd__inv_1 ca0 (.A(q1),    .Y(c2[0]));
  sky130_fd_sc_hd__inv_1 ca1 (.A(c2[0]), .Y(c2[1]));

  // === Chain B: 4 inverters (depth 5 with buffer) ===
  sky130_fd_sc_hd__inv_1 cb0 (.A(q1),    .Y(c4[0]));
  sky130_fd_sc_hd__inv_1 cb1 (.A(c4[0]), .Y(c4[1]));
  sky130_fd_sc_hd__inv_1 cb2 (.A(c4[1]), .Y(c4[2]));
  sky130_fd_sc_hd__inv_1 cb3 (.A(c4[2]), .Y(c4[3]));

  // === Chain C: 8 inverters (depth 9 with buffer) ===
  sky130_fd_sc_hd__inv_1 cc0 (.A(q1),    .Y(c8[0]));
  sky130_fd_sc_hd__inv_1 cc1 (.A(c8[0]), .Y(c8[1]));
  sky130_fd_sc_hd__inv_1 cc2 (.A(c8[1]), .Y(c8[2]));
  sky130_fd_sc_hd__inv_1 cc3 (.A(c8[2]), .Y(c8[3]));
  sky130_fd_sc_hd__inv_1 cc4 (.A(c8[3]), .Y(c8[4]));
  sky130_fd_sc_hd__inv_1 cc5 (.A(c8[4]), .Y(c8[5]));
  sky130_fd_sc_hd__inv_1 cc6 (.A(c8[5]), .Y(c8[6]));
  sky130_fd_sc_hd__inv_1 cc7 (.A(c8[6]), .Y(c8[7]));

  // === Chain D: 12 inverters (depth 13 with buffer) ===
  sky130_fd_sc_hd__inv_1 cd0  (.A(q1),     .Y(c12[0]));
  sky130_fd_sc_hd__inv_1 cd1  (.A(c12[0]), .Y(c12[1]));
  sky130_fd_sc_hd__inv_1 cd2  (.A(c12[1]), .Y(c12[2]));
  sky130_fd_sc_hd__inv_1 cd3  (.A(c12[2]), .Y(c12[3]));
  sky130_fd_sc_hd__inv_1 cd4  (.A(c12[3]), .Y(c12[4]));
  sky130_fd_sc_hd__inv_1 cd5  (.A(c12[4]), .Y(c12[5]));
  sky130_fd_sc_hd__inv_1 cd6  (.A(c12[5]), .Y(c12[6]));
  sky130_fd_sc_hd__inv_1 cd7  (.A(c12[6]), .Y(c12[7]));
  sky130_fd_sc_hd__inv_1 cd8  (.A(c12[7]), .Y(c12[8]));
  sky130_fd_sc_hd__inv_1 cd9  (.A(c12[8]), .Y(c12[9]));
  sky130_fd_sc_hd__inv_1 cd10 (.A(c12[9]), .Y(c12[10]));
  sky130_fd_sc_hd__inv_1 cd11 (.A(c12[10]),.Y(c12[11]));

  // === Chain E: 16 inverters (depth 17 with buffer) ===
  sky130_fd_sc_hd__inv_1 ce0  (.A(q1),     .Y(c16[0]));
  sky130_fd_sc_hd__inv_1 ce1  (.A(c16[0]), .Y(c16[1]));
  sky130_fd_sc_hd__inv_1 ce2  (.A(c16[1]), .Y(c16[2]));
  sky130_fd_sc_hd__inv_1 ce3  (.A(c16[2]), .Y(c16[3]));
  sky130_fd_sc_hd__inv_1 ce4  (.A(c16[3]), .Y(c16[4]));
  sky130_fd_sc_hd__inv_1 ce5  (.A(c16[4]), .Y(c16[5]));
  sky130_fd_sc_hd__inv_1 ce6  (.A(c16[5]), .Y(c16[6]));
  sky130_fd_sc_hd__inv_1 ce7  (.A(c16[6]), .Y(c16[7]));
  sky130_fd_sc_hd__inv_1 ce8  (.A(c16[7]), .Y(c16[8]));
  sky130_fd_sc_hd__inv_1 ce9  (.A(c16[8]), .Y(c16[9]));
  sky130_fd_sc_hd__inv_1 ce10 (.A(c16[9]), .Y(c16[10]));
  sky130_fd_sc_hd__inv_1 ce11 (.A(c16[10]),.Y(c16[11]));
  sky130_fd_sc_hd__inv_1 ce12 (.A(c16[11]),.Y(c16[12]));
  sky130_fd_sc_hd__inv_1 ce13 (.A(c16[12]),.Y(c16[13]));
  sky130_fd_sc_hd__inv_1 ce14 (.A(c16[13]),.Y(c16[14]));
  sky130_fd_sc_hd__inv_1 ce15 (.A(c16[14]),.Y(c16[15]));

  // === Buffer inverters: 8 per group, each a distinct output pin ===
  // Group 0: depth 3 (chain A endpoint c2[1] -> buffer -> out[0..7])
  sky130_fd_sc_hd__inv_1 buf_a0 (.A(c2[1]), .Y(out[0]));
  sky130_fd_sc_hd__inv_1 buf_a1 (.A(c2[1]), .Y(out[1]));
  sky130_fd_sc_hd__inv_1 buf_a2 (.A(c2[1]), .Y(out[2]));
  sky130_fd_sc_hd__inv_1 buf_a3 (.A(c2[1]), .Y(out[3]));
  sky130_fd_sc_hd__inv_1 buf_a4 (.A(c2[1]), .Y(out[4]));
  sky130_fd_sc_hd__inv_1 buf_a5 (.A(c2[1]), .Y(out[5]));
  sky130_fd_sc_hd__inv_1 buf_a6 (.A(c2[1]), .Y(out[6]));
  sky130_fd_sc_hd__inv_1 buf_a7 (.A(c2[1]), .Y(out[7]));

  // Group 1: depth 5 (chain B endpoint c4[3] -> buffer -> out[8..15])
  sky130_fd_sc_hd__inv_1 buf_b0 (.A(c4[3]), .Y(out[8]));
  sky130_fd_sc_hd__inv_1 buf_b1 (.A(c4[3]), .Y(out[9]));
  sky130_fd_sc_hd__inv_1 buf_b2 (.A(c4[3]), .Y(out[10]));
  sky130_fd_sc_hd__inv_1 buf_b3 (.A(c4[3]), .Y(out[11]));
  sky130_fd_sc_hd__inv_1 buf_b4 (.A(c4[3]), .Y(out[12]));
  sky130_fd_sc_hd__inv_1 buf_b5 (.A(c4[3]), .Y(out[13]));
  sky130_fd_sc_hd__inv_1 buf_b6 (.A(c4[3]), .Y(out[14]));
  sky130_fd_sc_hd__inv_1 buf_b7 (.A(c4[3]), .Y(out[15]));

  // Group 2: depth 9 (chain C endpoint c8[7] -> buffer -> out[16..23])
  sky130_fd_sc_hd__inv_1 buf_c0 (.A(c8[7]), .Y(out[16]));
  sky130_fd_sc_hd__inv_1 buf_c1 (.A(c8[7]), .Y(out[17]));
  sky130_fd_sc_hd__inv_1 buf_c2 (.A(c8[7]), .Y(out[18]));
  sky130_fd_sc_hd__inv_1 buf_c3 (.A(c8[7]), .Y(out[19]));
  sky130_fd_sc_hd__inv_1 buf_c4 (.A(c8[7]), .Y(out[20]));
  sky130_fd_sc_hd__inv_1 buf_c5 (.A(c8[7]), .Y(out[21]));
  sky130_fd_sc_hd__inv_1 buf_c6 (.A(c8[7]), .Y(out[22]));
  sky130_fd_sc_hd__inv_1 buf_c7 (.A(c8[7]), .Y(out[23]));

  // Group 3: depth 13 (chain D endpoint c12[11] -> buffer -> out[24..31])
  sky130_fd_sc_hd__inv_1 buf_d0 (.A(c12[11]), .Y(out[24]));
  sky130_fd_sc_hd__inv_1 buf_d1 (.A(c12[11]), .Y(out[25]));
  sky130_fd_sc_hd__inv_1 buf_d2 (.A(c12[11]), .Y(out[26]));
  sky130_fd_sc_hd__inv_1 buf_d3 (.A(c12[11]), .Y(out[27]));
  sky130_fd_sc_hd__inv_1 buf_d4 (.A(c12[11]), .Y(out[28]));
  sky130_fd_sc_hd__inv_1 buf_d5 (.A(c12[11]), .Y(out[29]));
  sky130_fd_sc_hd__inv_1 buf_d6 (.A(c12[11]), .Y(out[30]));
  sky130_fd_sc_hd__inv_1 buf_d7 (.A(c12[11]), .Y(out[31]));

  // Group 4: depth 17 (chain E endpoint c16[15] -> buffer -> out[32..39])
  sky130_fd_sc_hd__inv_1 buf_e0 (.A(c16[15]), .Y(out[32]));
  sky130_fd_sc_hd__inv_1 buf_e1 (.A(c16[15]), .Y(out[33]));
  sky130_fd_sc_hd__inv_1 buf_e2 (.A(c16[15]), .Y(out[34]));
  sky130_fd_sc_hd__inv_1 buf_e3 (.A(c16[15]), .Y(out[35]));
  sky130_fd_sc_hd__inv_1 buf_e4 (.A(c16[15]), .Y(out[36]));
  sky130_fd_sc_hd__inv_1 buf_e5 (.A(c16[15]), .Y(out[37]));
  sky130_fd_sc_hd__inv_1 buf_e6 (.A(c16[15]), .Y(out[38]));
  sky130_fd_sc_hd__inv_1 buf_e7 (.A(c16[15]), .Y(out[39]));

endmodule
