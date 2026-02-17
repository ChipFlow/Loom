// Techmap wrapper: maps Yosys memory_libmap ports to CF_SRAM_1024x32 pins.
module \$__CF_SRAM_1024x32_wrapper (
    PORT_1P_ADDR,
    PORT_1P_CLK,
    PORT_1P_CLK_EN,
    PORT_1P_RD_DATA,
    PORT_1P_WR_DATA,
    PORT_1P_WR_EN,
    PORT_1P_WR_BE
);

    input PORT_1P_CLK;
    input PORT_1P_CLK_EN;
    input PORT_1P_WR_EN;
    input [9:0] PORT_1P_ADDR;
    output [31:0] PORT_1P_RD_DATA;
    input [31:0] PORT_1P_WR_DATA;
    input [31:0] PORT_1P_WR_BE;

    CF_SRAM_1024x32 _TECHMAP_REPLACE_ (
        .DI(PORT_1P_WR_DATA),
        .AD(PORT_1P_ADDR),
        .BEN(PORT_1P_WR_BE),
        .CLKin(PORT_1P_CLK),
        .DO(PORT_1P_RD_DATA),
        .EN(PORT_1P_CLK_EN),
        .R_WB(~PORT_1P_WR_EN),
        .ScanInCC   (1'b0),
        .ScanInDL   (1'b0),
        .ScanInDR   (1'b0),
        .SM         (1'b0),
        .TM         (1'b0),
        .WLBI       (1'b0),
        .WLOFF      (1'b0),
        .vpwrac     (1'b1),
        .vpwrpc     (1'b1)
    );

endmodule
