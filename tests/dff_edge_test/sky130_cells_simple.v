// Simplified SKY130 cell models for testing (behavioral, no timing)

module sky130_fd_sc_hd__buf_2 (
    input  A,
    output X
);
    assign X = A;
endmodule

module sky130_fd_sc_hd__buf_1 (
    input  A,
    output X
);
    assign X = A;
endmodule

module sky130_fd_sc_hd__and2_1 (
    input  A,
    input  B,
    output X
);
    assign X = A & B;
endmodule

module sky130_fd_sc_hd__dfxtp_1 (
    input  CLK,
    input  D,
    output reg Q
);
    initial Q = 0;
    always @(posedge CLK)
        Q <= D;
endmodule
