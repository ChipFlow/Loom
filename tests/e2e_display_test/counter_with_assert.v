// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Simple counter with assertions to test Phase 3b
// This design asserts that the counter never exceeds 10

module counter_with_assert(
    input wire clk,
    input wire rst,
    output reg [3:0] count
);

always @(posedge clk or posedge rst) begin
    if (rst) begin
        count <= 4'h0;
    end else begin
        count <= count + 1;

        // Display current count
        $display("Count: %d", count + 1);

        // Stop at count 15
        if (count == 4'hf) begin
            $display("Counter complete");
        end
    end
end

// Assertion: count should never exceed 10
// This will be converted to $check/$assert cell by Yosys proc pass
`ifdef FORMAL
always @(posedge clk) begin
    if (!rst) begin
        assert (count < 10);
    end
end
`endif

endmodule


`ifdef SYNTHESIS
// Empty - don't synthesize testbench
`else
module testbench;

reg clk;
reg rst;
wire [3:0] count;

counter_with_assert dut (
    .clk(clk),
    .rst(rst),
    .count(count)
);

// Clock generation
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

// Test sequence
initial begin
    $display("=== Counter with Assertions Test Starting ===");
    rst = 1;
    #10;
    rst = 0;

    // Wait for counter to complete
    wait(count == 4'hf);
    #20;
    $display("=== Test Complete ===");
    $finish;
end

endmodule
`endif
