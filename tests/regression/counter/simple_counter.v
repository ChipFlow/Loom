// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Simple 4-bit counter with $display output
// Tests: basic $display, counter logic, $finish

module simple_counter(
    input wire clk,
    input wire rst,
    output reg [3:0] count
);

always @(posedge clk or posedge rst) begin
    if (rst) begin
        count <= 4'h0;
    end else begin
        count <= count + 1;

        // Display every count value
        $display("Count: %d (0x%h)", count + 1, count + 1);

        // Test milestone displays
        if (count == 4'h7) begin
            $display("Halfway point reached!");
        end

        if (count == 4'hf) begin
            $display("Counter rolled over, test complete");
            $finish;
        end
    end
end

endmodule


`ifdef SYNTHESIS
// Empty - don't synthesize testbench
`else
// Self-checking testbench
module testbench;

reg clk;
reg rst;
wire [3:0] count;

simple_counter dut (
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
    $display("=== Simple Counter Test Starting ===");
    rst = 1;
    #10;
    rst = 0;

    // Let counter run to compilation
    #200;

    // If we get here, $finish didn't trigger
    $display("ERROR: Counter did not finish");
    $finish;
end

endmodule
`endif
