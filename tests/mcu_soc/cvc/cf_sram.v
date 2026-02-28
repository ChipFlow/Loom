// Behavioral model for CF_SRAM_1024x32 (ChipFlow SRAM macro).
// Simple synchronous SRAM: read on R_WB=1, write on R_WB=0, with byte enables.

`timescale 1ps/1ps

module CF_SRAM_1024x32 (
    WLBI, WLOFF, CLKin, EN, R_WB, SM, TM,
    ScanInDR, ScanInDL, ScanInCC,
    vpwrpc, vpwrac,
    AD, BEN, DI, DO
);
    input WLBI, WLOFF, CLKin, EN, R_WB, SM, TM;
    input ScanInDR, ScanInDL, ScanInCC;
    input vpwrpc, vpwrac;
    input [9:0] AD;
    input [31:0] BEN;
    input [31:0] DI;
    output reg [31:0] DO;

    reg [31:0] mem [0:1023];

    integer i;
    initial begin
        DO = 32'b0;
        for (i = 0; i < 1024; i = i + 1)
            mem[i] = 32'b0;
    end

    always @(posedge CLKin) begin
        if (EN) begin
            if (R_WB) begin
                // Read
                DO <= mem[AD];
            end else begin
                // Write with byte enables
                for (i = 0; i < 32; i = i + 1) begin
                    if (BEN[i])
                        mem[AD][i] <= DI[i];
                end
            end
        end
    end

    // SDF path delays for CLKin -> DO (one per bit for CVC)
    specify
        (posedge CLKin => (DO[0] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[1] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[2] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[3] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[4] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[5] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[6] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[7] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[8] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[9] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[10] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[11] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[12] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[13] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[14] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[15] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[16] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[17] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[18] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[19] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[20] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[21] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[22] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[23] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[24] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[25] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[26] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[27] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[28] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[29] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[30] : CLKin)) = (0, 0);
        (posedge CLKin => (DO[31] : CLKin)) = (0, 0);
    endspecify
endmodule
