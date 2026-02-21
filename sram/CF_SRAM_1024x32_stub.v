// Blackbox stub for CF_SRAM_1024x32 (ChipFoundry sky130 SRAM macro).
// Used during Yosys synthesis so the cell is preserved as a blackbox.
// Includes power pins so OpenLane2 PDN can connect them.
module CF_SRAM_1024x32 (DO, ScanOutCC, AD, BEN, CLKin, DI, EN, R_WB,
    ScanInCC, ScanInDL, ScanInDR, SM, TM, WLBI, WLOFF,
    vgnd, vnb, vpb, vpwra, vpwrac, vpwrm, vpwrp, vpwrpc);
    output [31:0] DO;
    output ScanOutCC;
    input [31:0] DI;
    input [31:0] BEN;
    input [9:0] AD;
    input EN;
    input R_WB;
    input CLKin;
    input WLBI;
    input WLOFF;
    input TM;
    input SM;
    input ScanInCC;
    input ScanInDL;
    input ScanInDR;
    input vpwrac;
    input vpwrpc;
    input vgnd;
    input vnb;
    input vpb;
    input vpwra;
    input vpwrm;
    input vpwrp;
endmodule
