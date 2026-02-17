// Testbench for finish_test module
// Demonstrates $finish behavior for GEM event buffer testing
`timescale 1ns/1ps

module finish_test_tb;

reg clk;
reg rst;
wire [3:0] count;
wire done;

// Instantiate the DUT
finish_test dut (
    .clk(clk),
    .rst(rst),
    .count(count),
    .done(done)
);

// Clock generation
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

// Test sequence
initial begin
    $dumpfile("finish_test.vcd");
    $dumpvars(0, finish_test_tb);

    // Reset
    rst = 1;
    #20;
    rst = 0;

    // Wait for done signal or timeout
    @(posedge done);
    $display("Done signal received at time %t, count = %d", $time, count);

    // In GEM, this $finish should trigger an event
    $finish;
end

// Timeout watchdog
initial begin
    #1000;
    $display("TIMEOUT: Test did not complete");
    $finish;
end

endmodule
