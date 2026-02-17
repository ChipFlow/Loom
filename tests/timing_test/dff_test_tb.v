// Testbench for DFF timing test
// Critical test case: d changes at the SAME timestamp as clock posedge
// In Verilog semantics, the new value of d should be captured

`timescale 1ns/1ps

module dff_test_tb;

reg clk;
reg d;
wire q;

dff_test dut (
    .clk(clk),
    .d(d),
    .q(q)
);

initial begin
    $dumpfile("dff_test.vcd");
    $dumpvars(0, dff_test_tb);

    // Initialize - q will be X initially
    clk = 0;
    d = 0;
    #10;

    // Cycle 1: Capture d=0
    clk = 1;  // posedge at t=10
    #10;
    clk = 0;
    #10;
    // After cycle 1: q should be 0

    // Cycle 2: d=1 set up before clock (normal case)
    d = 1;    // t=30, d changes to 1
    #10;
    clk = 1;  // t=40, posedge - should capture d=1
    #10;
    clk = 0;
    #10;
    // After cycle 2: q should be 1

    // CRITICAL TEST - Cycle 3: d changes 0->1 AT SAME TIME as clock posedge
    // This is the bug case in GEM
    d = 0;    // t=60, d back to 0
    #10;      // t=70, d=0, clk=0, q=1

    // At t=70: BOTH d and clk change simultaneously
    // d: 0 -> 1
    // clk: 0 -> 1 (posedge)
    // Expected: q captures NEW value d=1 (remains 1)
    d = 1;
    clk = 1;
    #10;
    clk = 0;
    #10;
    // After cycle 3: q should be 1

    // CRITICAL TEST - Cycle 4: d changes 1->0 AT SAME TIME as clock posedge
    // Start with d=1, q=1
    // Then simultaneous: d: 1->0, clk: 0->1
    // Expected: q captures NEW value d=0
    d = 1;    // Ensure d=1
    #10;      // t=100, d=1, clk=0, q=1

    // At t=100: BOTH d and clk change simultaneously
    d = 0;
    clk = 1;  // posedge with d=0
    #10;
    clk = 0;
    #10;
    // After cycle 4: q should be 0 (captured d=0)

    // Cycle 5: Verify q=0 persists
    d = 0;
    #10;
    clk = 1;  // t=130, posedge - captures d=0
    #10;
    clk = 0;
    #10;
    // After cycle 5: q should be 0

    // End of test
    #20;
    $display("Test completed - check VCD for timing");
    $finish;
end

// Detailed monitor to trace timing
initial begin
    $monitor("t=%0t clk=%b d=%b q=%b", $time, clk, d, q);
end

endmodule
