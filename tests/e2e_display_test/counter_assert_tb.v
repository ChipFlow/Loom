// Testbench for counter_with_assert that generates VCD
module testbench;

reg clk;
reg rst;
wire [3:0] count;

// Instantiate DUT (just for VCD generation, not used by GEM)
counter_with_assert dut (
    .clk(clk),
    .rst(rst),
    .count(count)
);

// Clock generation
initial begin
    clk = 0;
    forever #10000 clk = ~clk;  // 20ns period
end

// Test sequence
initial begin
    $dumpfile("counter_with_assert.vcd");
    $dumpvars(0, testbench);

    rst = 1;
    #20000;  // Hold reset for 1 cycle
    rst = 0;

    // Run for 20 cycles to see assertion fire at count=10
    repeat(20) @(posedge clk);

    $finish;
end

endmodule
