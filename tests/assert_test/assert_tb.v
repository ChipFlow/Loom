// Testbench for assert_simple module
`timescale 1ns/1ps

module assert_tb;

reg clk;
reg rst;
reg [3:0] data_in;
wire [3:0] data_out;
wire overflow_flag;

// Instantiate the DUT
assert_simple dut (
    .clk(clk),
    .rst(rst),
    .data_in(data_in),
    .data_out(data_out),
    .overflow_flag(overflow_flag)
);

// Clock generation
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

// Test sequence
initial begin
    $dumpfile("assert_test.vcd");
    $dumpvars(0, assert_tb);

    // Initialize
    rst = 1;
    data_in = 4'h0;
    #20;
    rst = 0;

    // Normal operation - counter runs from 0 to 15
    // At cycle 11+ (counter > 10), if data_in == 0xF, assertion should fail

    // Run for a while with safe data
    repeat(10) begin
        data_in = $random % 14;  // 0-13, never 15
        #10;
    end

    // Now trigger assertion failure: counter > 10 and data_in = 0xF
    data_in = 4'hF;  // This should trigger assertion at cycle 11+
    #10;

    // Continue with more cycles
    data_in = 4'h5;
    #50;

    $display("Test completed");
    $finish;
end

endmodule
