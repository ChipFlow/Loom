/* SKY130 Timing Test 2: Convergent Logic Cone
 * 4 dfxtp -> tree of nand2/nor2/and2 -> dfxtp
 *
 * Critical path: a_q -> nand2 -> and2 -> nand2 -> inv (depth 4)
 * Expected combo delay (default): 4 x 50ps = 200ps
 * Expected arrival: 150ps + 200ps = 350ps
 */

module logic_cone(CLK, A, B, C, D_IN, Q);
  input CLK;
  wire CLK;
  input A;
  wire A;
  input B;
  wire B;
  input C;
  wire C;
  input D_IN;
  wire D_IN;
  output Q;
  wire Q;
  wire a_q, b_q, c_q, d_q;
  wire n1, n2, a1, n3, result;

  sky130_fd_sc_hd__dfxtp_1 dff_a (.CLK(CLK), .D(A),    .Q(a_q));
  sky130_fd_sc_hd__dfxtp_1 dff_b (.CLK(CLK), .D(B),    .Q(b_q));
  sky130_fd_sc_hd__dfxtp_1 dff_c (.CLK(CLK), .D(C),    .Q(c_q));
  sky130_fd_sc_hd__dfxtp_1 dff_d (.CLK(CLK), .D(D_IN), .Q(d_q));

  sky130_fd_sc_hd__nand2_1 g1 (.A(a_q), .B(b_q), .Y(n1));
  sky130_fd_sc_hd__nor2_1  g2 (.A(c_q), .B(d_q), .Y(n2));
  sky130_fd_sc_hd__and2_1  g3 (.A(n1),  .B(n2),  .X(a1));
  sky130_fd_sc_hd__nand2_1 g4 (.A(a1),  .B(a_q), .Y(n3));
  sky130_fd_sc_hd__inv_1   g5 (.A(n3),  .Y(result));

  sky130_fd_sc_hd__dfxtp_1 dff_out (.CLK(CLK), .D(result), .Q(Q));

endmodule
