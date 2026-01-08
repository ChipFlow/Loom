// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Test various $display format specifiers
// Tests: decimal, hex, binary, multiple arguments

module format_test(
    input wire clk,
    input wire rst,
    output reg [7:0] counter,
    output reg [7:0] data
);

always @(posedge clk or posedge rst) begin
    if (rst) begin
        counter <= 8'h00;
        data <= 8'hAA;
    end else begin
        counter <= counter + 1;
        data <= data ^ 8'h55;  // Toggle pattern

        // Test different format specifiers
        case (counter)
            8'h00: $display("Test 0: Decimal format: counter=%d, data=%d", counter, data);
            8'h01: $display("Test 1: Hex format: counter=%h, data=%h", counter, data);
            8'h02: $display("Test 2: Binary format: counter=%b, data=%b", counter, data);
            8'h03: $display("Test 3: Mixed formats: dec=%d, hex=%h, bin=%b", counter, counter, counter);
            8'h04: $display("Test 4: Multiple args: %d %h %d %h", counter, counter, data, data);
            8'h05: begin
                $display("Test 5: All tests passed!");
                $finish;
            end
        endcase
    end
end

endmodule


`ifdef SYNTHESIS
// Empty - don't synthesize testbench
`else
module testbench;

reg clk;
reg rst;
wire [7:0] counter;
wire [7:0] data;

format_test dut (
    .clk(clk),
    .rst(rst),
    .counter(counter),
    .data(data)
);

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

initial begin
    $display("=== Format Specifier Test Starting ===");
    rst = 1;
    #10;
    rst = 0;

    #100;

    $display("ERROR: Test did not complete");
    $finish;
end

endmodule
`endif
