// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Simple 4-bit ALU with assertions and $display
// Tests: ALU operations, assertions, $display with multiple values

module simple_alu(
    input wire [3:0] a,
    input wire [3:0] b,
    input wire [1:0] op,
    output reg [4:0] result
);

always @(*) begin
    case (op)
        2'b00: result = a + b;      // ADD
        2'b01: result = a - b;      // SUB
        2'b10: result = a & b;      // AND
        2'b11: result = a | b;      // OR
        default: result = 5'h00;
    endcase
end

endmodule


`ifdef SYNTHESIS
// Empty - don't synthesize testbench
`else
// Self-checking testbench with assertions
module testbench;

reg [3:0] a, b;
reg [1:0] op;
wire [4:0] result;
reg [4:0] expected;
integer errors;

simple_alu dut (
    .a(a),
    .b(b),
    .op(op),
    .result(result)
);

task check_result;
    input [4:0] exp;
    input [7:0] test_name;
    begin
        #1;  // Wait for combinational logic
        if (result !== exp) begin
            $display("FAIL Test %d: a=%h, b=%h, op=%b, expected=%h, got=%h",
                     test_name, a, b, op, exp, result);
            errors = errors + 1;
        end else begin
            $display("PASS Test %d: a=%h, b=%h, op=%b, result=%h",
                     test_name, a, b, op, result);
        end

        // Assertion check
        assert (result == exp) else begin
            $display("ASSERTION FAILED at test %d", test_name);
        end
    end
endtask

initial begin
    errors = 0;
    $display("=== ALU Test Starting ===");

    // Test ADD
    op = 2'b00;
    a = 4'h3; b = 4'h5; check_result(5'h08, 0);  // 3 + 5 = 8
    a = 4'hF; b = 4'h1; check_result(5'h10, 1);  // 15 + 1 = 16 (overflow)
    a = 4'h0; b = 4'h0; check_result(5'h00, 2);  // 0 + 0 = 0

    // Test SUB
    op = 2'b01;
    a = 4'h7; b = 4'h3; check_result(5'h04, 3);  // 7 - 3 = 4
    a = 4'h3; b = 4'h7; check_result(5'h1C, 4);  // 3 - 7 = -4 (underflow)

    // Test AND
    op = 2'b10;
    a = 4'hF; b = 4'h7; check_result(5'h07, 5);  // F & 7 = 7
    a = 4'hA; b = 4'h5; check_result(5'h00, 6);  // A & 5 = 0

    // Test OR
    op = 2'b11;
    a = 4'hC; b = 4'h3; check_result(5'h0F, 7);  // C | 3 = F
    a = 4'h0; b = 4'h0; check_result(5'h00, 8);  // 0 | 0 = 0

    #10;

    if (errors == 0) begin
        $display("=== ALL TESTS PASSED ===");
    end else begin
        $display("=== %d TESTS FAILED ===", errors);
    end

    $finish;
end

endmodule
`endif
