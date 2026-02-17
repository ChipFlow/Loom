/* SKY130 Timing Test 1: Inverter Chain
 * dfxtp -> 16 x inv_1 -> dfxtp
 *
 * Expected combo delay (default SKY130): 16 x 50ps = 800ps
 * Expected arrival at D2: clk_to_q(150ps) + 800ps = 950ps
 * Min clock period: 950ps + setup(80ps) = 1030ps
 */

module inv_chain(CLK, D, Q);
  input CLK;
  wire CLK;
  input D;
  wire D;
  output Q;
  wire Q;
  wire q1;
  wire [15:0] c;

  sky130_fd_sc_hd__dfxtp_1 dff_in (
    .CLK(CLK),
    .D(D),
    .Q(q1)
  );

  sky130_fd_sc_hd__inv_1 i0  (.A(q1),    .Y(c[0]));
  sky130_fd_sc_hd__inv_1 i1  (.A(c[0]),  .Y(c[1]));
  sky130_fd_sc_hd__inv_1 i2  (.A(c[1]),  .Y(c[2]));
  sky130_fd_sc_hd__inv_1 i3  (.A(c[2]),  .Y(c[3]));
  sky130_fd_sc_hd__inv_1 i4  (.A(c[3]),  .Y(c[4]));
  sky130_fd_sc_hd__inv_1 i5  (.A(c[4]),  .Y(c[5]));
  sky130_fd_sc_hd__inv_1 i6  (.A(c[5]),  .Y(c[6]));
  sky130_fd_sc_hd__inv_1 i7  (.A(c[6]),  .Y(c[7]));
  sky130_fd_sc_hd__inv_1 i8  (.A(c[7]),  .Y(c[8]));
  sky130_fd_sc_hd__inv_1 i9  (.A(c[8]),  .Y(c[9]));
  sky130_fd_sc_hd__inv_1 i10 (.A(c[9]),  .Y(c[10]));
  sky130_fd_sc_hd__inv_1 i11 (.A(c[10]), .Y(c[11]));
  sky130_fd_sc_hd__inv_1 i12 (.A(c[11]), .Y(c[12]));
  sky130_fd_sc_hd__inv_1 i13 (.A(c[12]), .Y(c[13]));
  sky130_fd_sc_hd__inv_1 i14 (.A(c[13]), .Y(c[14]));
  sky130_fd_sc_hd__inv_1 i15 (.A(c[14]), .Y(c[15]));

  sky130_fd_sc_hd__dfxtp_1 dff_out (
    .CLK(CLK),
    .D(c[15]),
    .Q(Q)
  );

endmodule
