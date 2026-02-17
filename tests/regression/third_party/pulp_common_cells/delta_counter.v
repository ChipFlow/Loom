// SPDX-FileCopyrightText: Copyright 2018 ETH Zurich and University of Bologna
// SPDX-License-Identifier: SHL-0.51
//
// Original source: https://github.com/pulp-platform/common_cells
// File: src/delta_counter.sv
// Adapted for GEM regression testing
//
// Up/down counter with variable delta - adapted to Verilog for GEM

module delta_counter #(
    parameter WIDTH = 4,
    parameter STICKY_OVERFLOW = 0
)(
    input  wire             clk_i,
    input  wire             rst_ni,
    input  wire             clear_i,
    input  wire             en_i,
    input  wire             load_i,
    input  wire             down_i,
    input  wire [WIDTH-1:0] delta_i,
    input  wire [WIDTH-1:0] d_i,
    output wire [WIDTH-1:0] q_o,
    output wire             overflow_o
);
    reg [WIDTH:0] counter_q;
    reg [WIDTH:0] counter_d;

    // Overflow detection
    generate
        if (STICKY_OVERFLOW) begin : gen_sticky_overflow
            reg overflow_q;
            reg overflow_d;

            always @(posedge clk_i or negedge rst_ni) begin
                if (!rst_ni)
                    overflow_q <= 1'b0;
                else
                    overflow_q <= overflow_d;
            end

            always @(*) begin
                overflow_d = overflow_q;
                if (clear_i || load_i)
                    overflow_d = 1'b0;
                else if (!overflow_q && en_i) begin
                    if (down_i)
                        overflow_d = delta_i > counter_q[WIDTH-1:0];
                    else
                        overflow_d = counter_q[WIDTH-1:0] > ({WIDTH{1'b1}} - delta_i);
                end
            end
            assign overflow_o = overflow_q;
        end else begin : gen_transient_overflow
            assign overflow_o = counter_q[WIDTH];
        end
    endgenerate

    assign q_o = counter_q[WIDTH-1:0];

    // Counter logic
    always @(*) begin
        counter_d = counter_q;
        if (clear_i)
            counter_d = 0;
        else if (load_i)
            counter_d = {1'b0, d_i};
        else if (en_i) begin
            if (down_i)
                counter_d = counter_q - delta_i;
            else
                counter_d = counter_q + delta_i;
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni)
            counter_q <= 0;
        else
            counter_q <= counter_d;
    end

endmodule


// Self-contained testbench for regression testing
`ifndef SYNTHESIS
module testbench;

    parameter WIDTH = 8;

    reg clk = 0;
    reg rst_n;
    reg clear;
    reg en;
    reg load;
    reg down;
    reg [WIDTH-1:0] delta;
    reg [WIDTH-1:0] d_in;
    wire [WIDTH-1:0] q_out;
    wire overflow;

    // Expected values for checking
    reg [WIDTH-1:0] expected_q;
    integer errors = 0;

    delta_counter #(.WIDTH(WIDTH), .STICKY_OVERFLOW(0)) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .clear_i(clear),
        .en_i(en),
        .load_i(load),
        .down_i(down),
        .delta_i(delta),
        .d_i(d_in),
        .q_o(q_out),
        .overflow_o(overflow)
    );

    // Clock generation
    always #5 clk = !clk;

    initial begin
        $dumpfile("delta_counter.vcd");
        $dumpvars(0, testbench);

        // Initialize
        rst_n = 0;
        clear = 0;
        en = 0;
        load = 0;
        down = 0;
        delta = 1;
        d_in = 0;
        expected_q = 0;

        // Reset
        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Test 1: Count up by 1
        $display("=== Test 1: Count up by 1 ===");
        en = 1;
        delta = 1;
        down = 0;
        repeat(5) begin
            @(posedge clk);
            #1;
            expected_q = expected_q + 1;
            if (q_out !== expected_q) begin
                $display("FAIL: Expected %d, got %d", expected_q, q_out);
                errors = errors + 1;
            end
        end
        $display("Counter at %d after 5 cycles", q_out);

        // Test 2: Count up by 3
        $display("=== Test 2: Count up by 3 ===");
        delta = 3;
        repeat(4) begin
            @(posedge clk);
            #1;
            expected_q = expected_q + 3;
        end
        $display("Counter at %d after 4 more cycles (delta=3)", q_out);
        if (q_out !== expected_q) begin
            $display("FAIL: Expected %d, got %d", expected_q, q_out);
            errors = errors + 1;
        end else begin
            $display("PASS: Counter value correct");
        end

        // Test 3: Count down
        $display("=== Test 3: Count down by 2 ===");
        down = 1;
        delta = 2;
        repeat(3) begin
            @(posedge clk);
            #1;
            expected_q = expected_q - 2;
        end
        $display("Counter at %d after counting down", q_out);
        if (q_out !== expected_q) begin
            $display("FAIL: Expected %d, got %d", expected_q, q_out);
            errors = errors + 1;
        end else begin
            $display("PASS: Down-count correct");
        end

        // Test 4: Load value
        $display("=== Test 4: Load value 100 ===");
        en = 0;
        load = 1;
        d_in = 100;
        @(posedge clk);
        #1;
        load = 0;
        expected_q = 100;
        if (q_out !== expected_q) begin
            $display("FAIL: Expected %d, got %d", expected_q, q_out);
            errors = errors + 1;
        end else begin
            $display("PASS: Load value correct");
        end

        // Test 5: Clear
        $display("=== Test 5: Clear counter ===");
        clear = 1;
        @(posedge clk);
        #1;
        clear = 0;
        expected_q = 0;
        if (q_out !== expected_q) begin
            $display("FAIL: Expected %d, got %d", expected_q, q_out);
            errors = errors + 1;
        end else begin
            $display("PASS: Clear correct");
        end

        // Summary
        repeat(2) @(posedge clk);
        if (errors == 0) begin
            $display("=== All tests PASSED ===");
        end else begin
            $display("=== FAILED with %d errors ===", errors);
        end

        $finish;
    end

endmodule
`endif
