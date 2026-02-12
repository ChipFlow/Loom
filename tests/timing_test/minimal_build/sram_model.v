// Simple behavioral model for CF_SRAM_1024x32
// Matches the port names from the post-synthesis netlist
module CF_SRAM_1024x32 (
    input WLBI,
    input CLKin,
    input EN,
    input R_WB,  // 1=read, 0=write
    input SM,
    input TM,
    input ScanInDR,
    input ScanInDL,
    input ScanInCC,
    input vpwrpc,
    input vpwrac,
    input [9:0] AD,
    input [31:0] BEN,  // bit-level enables
    input [31:0] DI,
    output reg [31:0] DO
);
    reg [31:0] memory [0:1023];

    always @(posedge CLKin) begin
        if (EN) begin
            if (R_WB) begin
                // Read
                DO <= memory[AD];
            end else begin
                // Write with bit enables
                if (BEN[0]) memory[AD][0] <= DI[0];
                if (BEN[1]) memory[AD][1] <= DI[1];
                if (BEN[2]) memory[AD][2] <= DI[2];
                if (BEN[3]) memory[AD][3] <= DI[3];
                if (BEN[4]) memory[AD][4] <= DI[4];
                if (BEN[5]) memory[AD][5] <= DI[5];
                if (BEN[6]) memory[AD][6] <= DI[6];
                if (BEN[7]) memory[AD][7] <= DI[7];
                if (BEN[8]) memory[AD][8] <= DI[8];
                if (BEN[9]) memory[AD][9] <= DI[9];
                if (BEN[10]) memory[AD][10] <= DI[10];
                if (BEN[11]) memory[AD][11] <= DI[11];
                if (BEN[12]) memory[AD][12] <= DI[12];
                if (BEN[13]) memory[AD][13] <= DI[13];
                if (BEN[14]) memory[AD][14] <= DI[14];
                if (BEN[15]) memory[AD][15] <= DI[15];
                if (BEN[16]) memory[AD][16] <= DI[16];
                if (BEN[17]) memory[AD][17] <= DI[17];
                if (BEN[18]) memory[AD][18] <= DI[18];
                if (BEN[19]) memory[AD][19] <= DI[19];
                if (BEN[20]) memory[AD][20] <= DI[20];
                if (BEN[21]) memory[AD][21] <= DI[21];
                if (BEN[22]) memory[AD][22] <= DI[22];
                if (BEN[23]) memory[AD][23] <= DI[23];
                if (BEN[24]) memory[AD][24] <= DI[24];
                if (BEN[25]) memory[AD][25] <= DI[25];
                if (BEN[26]) memory[AD][26] <= DI[26];
                if (BEN[27]) memory[AD][27] <= DI[27];
                if (BEN[28]) memory[AD][28] <= DI[28];
                if (BEN[29]) memory[AD][29] <= DI[29];
                if (BEN[30]) memory[AD][30] <= DI[30];
                if (BEN[31]) memory[AD][31] <= DI[31];
            end
        end
    end
endmodule
