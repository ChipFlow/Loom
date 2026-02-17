// DFF edge-triggering test netlist
// Tests 4 types of DFF clocking:
//   1. System clock DFF (CLK = buffered system clock)
//   2. Gated clock DFF (CLK = AND(system_clock, enable))
//   3. External clock DFF (CLK = separate clock input)
//   4. Non-clock DFF (CLK tied to a data signal - should only latch on that signal's edges)
//
// GPIO mapping:
//   gpio_in[0]  = system clock
//   gpio_in[1]  = external clock
//   gpio_in[2]  = data input for DFF1 and DFF2
//   gpio_in[3]  = gate enable for DFF2's clock
//   gpio_in[4]  = data input for DFF3
//   gpio_in[5]  = data input for DFF4
//   gpio_in[6]  = "clock" for DFF4 (actually a data signal)
//   gpio_out[0] = DFF1 Q (system clock)
//   gpio_out[1] = DFF2 Q (gated clock)
//   gpio_out[2] = DFF3 Q (external clock)
//   gpio_out[3] = DFF4 Q (non-clock)

module dff_edge_test (
    input  [6:0] gpio_in,
    output [3:0] gpio_out
);

    // Internal wires
    wire sys_clk;      // buffered system clock
    wire gated_clk;    // gated clock
    wire ext_clk;      // external clock
    wire data_clk;     // data signal used as "clock"

    // Clock buffer (system clock goes through a buffer, like in real designs)
    sky130_fd_sc_hd__buf_2 clk_buf (
        .A(gpio_in[0]),
        .X(sys_clk)
    );

    // Gated clock: AND(sys_clk, enable)
    sky130_fd_sc_hd__and2_1 clk_gate (
        .A(sys_clk),
        .B(gpio_in[3]),
        .X(gated_clk)
    );

    // External clock buffer
    sky130_fd_sc_hd__buf_1 ext_clk_buf (
        .A(gpio_in[1]),
        .X(ext_clk)
    );

    // Data "clock" buffer
    sky130_fd_sc_hd__buf_1 data_clk_buf (
        .A(gpio_in[6]),
        .X(data_clk)
    );

    // DFF1: System clock - latches gpio_in[2] on posedge sys_clk
    sky130_fd_sc_hd__dfxtp_1 dff_sysclk (
        .CLK(sys_clk),
        .D(gpio_in[2]),
        .Q(gpio_out[0])
    );

    // DFF2: Gated clock - latches gpio_in[2] on posedge gated_clk
    //        Should NOT latch when enable (gpio_in[3]) is low
    sky130_fd_sc_hd__dfxtp_1 dff_gated (
        .CLK(gated_clk),
        .D(gpio_in[2]),
        .Q(gpio_out[1])
    );

    // DFF3: External clock - latches gpio_in[4] on posedge ext_clk
    sky130_fd_sc_hd__dfxtp_1 dff_extclk (
        .CLK(ext_clk),
        .D(gpio_in[4]),
        .Q(gpio_out[2])
    );

    // DFF4: Non-clock signal as CLK - latches gpio_in[5] on posedge data_clk
    sky130_fd_sc_hd__dfxtp_1 dff_dataclk (
        .CLK(data_clk),
        .D(gpio_in[5]),
        .Q(gpio_out[3])
    );

endmodule
