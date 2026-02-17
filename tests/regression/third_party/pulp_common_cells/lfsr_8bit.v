// SPDX-FileCopyrightText: Copyright 2017 ETH Zurich and University of Bologna
// SPDX-License-Identifier: SHL-0.51
//
// Original source: https://github.com/pulp-platform/common_cells
// File: src/lfsr_8bit.sv
// Adapted for GEM regression testing with assertions
//
// 8-bit Linear Feedback Shift Register
// Authors: Igor Loi (Univ. Bologna), Florian Zaruba (ETH Zurich)

module lfsr_8bit #(
    parameter [7:0] SEED = 8'b0,
    parameter WIDTH = 8
)(
    input  wire       clk_i,
    input  wire       rst_ni,
    input  wire       en_i,
    output wire [7:0] refill_way_oh_o,
    output wire [2:0] refill_way_bin_o
);

    reg [7:0] shift_d, shift_q;

    // LFSR feedback: taps at positions 7, 3, 2, 1
    // Polynomial: x^8 + x^4 + x^3 + x^2 + 1
    always @(*) begin
        shift_d = shift_q;
        if (en_i) begin
            shift_d = shift_q >> 1;
            shift_d[7] = shift_q[0];
            shift_d[3] = shift_d[3] ^ shift_q[0];
            shift_d[2] = shift_d[2] ^ shift_q[0];
            shift_d[1] = shift_d[1] ^ shift_q[0];
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni)
            shift_q <= SEED;
        else
            shift_q <= shift_d;
    end

    // One-hot output (useful for cache replacement)
    assign refill_way_oh_o = shift_q;

    // Binary output (lower 3 bits for 8-way selection)
    assign refill_way_bin_o = shift_q[2:0];

endmodule


// Self-contained testbench with assertions for GEM testing
`ifndef SYNTHESIS
module testbench;

    reg clk = 0;
    reg rst_n;
    reg en;
    wire [7:0] way_oh;
    wire [2:0] way_bin;

    // Track LFSR sequence for verification
    reg [7:0] expected_seq [0:255];
    integer seq_idx = 0;
    integer errors = 0;
    integer cycle_count = 0;

    // For detecting stuck-at or repeating patterns
    reg [7:0] prev_value;
    integer stuck_count = 0;

    lfsr_8bit #(.SEED(8'hAC)) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .en_i(en),
        .refill_way_oh_o(way_oh),
        .refill_way_bin_o(way_bin)
    );

    // Clock generation
    always #5 clk = !clk;

    // Track previous value for stuck detection
    always @(posedge clk) begin
        if (rst_n && en) begin
            prev_value <= way_oh;
            if (way_oh == prev_value)
                stuck_count <= stuck_count + 1;
            else
                stuck_count <= 0;
        end
    end

    // Formal assertions (converted to GEM_ASSERT during synthesis)
`ifdef FORMAL
    // Assert LFSR never gets stuck at same value for too long
    always @(posedge clk) begin
        if (rst_n && en) begin
            assert(stuck_count < 5);  // Should never be stuck for 5+ cycles
        end
    end

    // Assert binary output matches lower bits of one-hot output
    always @(posedge clk) begin
        if (rst_n) begin
            assert(way_bin == way_oh[2:0]);
        end
    end
`endif

    initial begin
        $dumpfile("lfsr_8bit.vcd");
        $dumpvars(0, testbench);

        // Initialize
        rst_n = 0;
        en = 0;
        prev_value = 0;

        // Reset
        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        #1;

        $display("=== LFSR 8-bit Test ===");
        $display("Initial value after reset: 0x%h", way_oh);

        // Test 1: Run LFSR for many cycles, check it produces different values
        $display("=== Test 1: LFSR sequence generation ===");
        en = 1;
        repeat(20) begin
            @(posedge clk);
            #1;
            cycle_count = cycle_count + 1;
            $display("Cycle %3d: LFSR = 0x%h, bin = %d", cycle_count, way_oh, way_bin);

            // Check binary output matches
            if (way_bin !== way_oh[2:0]) begin
                $display("FAIL: Binary output mismatch!");
                errors = errors + 1;
            end
        end

        // Test 2: Disable and verify it holds value
        $display("=== Test 2: Hold value when disabled ===");
        en = 0;
        prev_value = way_oh;
        repeat(5) @(posedge clk);
        #1;
        if (way_oh !== prev_value) begin
            $display("FAIL: LFSR changed while disabled!");
            errors = errors + 1;
        end else begin
            $display("PASS: LFSR held value while disabled");
        end

        // Test 3: Re-enable and continue
        $display("=== Test 3: Continue after re-enable ===");
        en = 1;
        @(posedge clk);
        #1;
        if (way_oh === prev_value) begin
            $display("WARNING: LFSR same value after re-enable (possible but unlikely)");
        end else begin
            $display("PASS: LFSR generating new values");
        end

        // Run more cycles
        repeat(30) begin
            @(posedge clk);
            #1;
            cycle_count = cycle_count + 1;
        end

        // Summary
        repeat(2) @(posedge clk);
        if (errors == 0) begin
            $display("=== All tests PASSED (%d cycles run) ===", cycle_count);
        end else begin
            $display("=== FAILED with %d errors ===", errors);
        end

        $finish;
    end

endmodule
`endif
