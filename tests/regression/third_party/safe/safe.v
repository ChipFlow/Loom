// SPDX-FileCopyrightText: Copyright (c) 2020 Christian Svensson
// SPDX-License-Identifier: MIT
//
// Original source: https://github.com/bluecmd/sva-playground/blob/master/backdoor/safe.sv
// Adapted for GEM by: NVIDIA GEM Team
// Modifications:
//   - Converted from SystemVerilog (.sv) to Verilog (.v)
//   - Simplified enum to parameters for broader tool support
//   - Added ifdef SYNTHESIS/FORMAL blocks

module safe (
    input clk,
    input reset,
    input [3:0] din,
    input din_valid,
    output wire unlocked
  );

  // State machine encoding
  parameter PIN_0        = 4'd0;
  parameter PIN_1        = 4'd1;
  parameter PIN_2        = 4'd2;
  parameter PIN_3        = 4'd3;
  parameter SECRET_PIN_1 = 4'd4;
  parameter SECRET_PIN_2 = 4'd5;
  parameter SECRET_PIN_3 = 4'd6;
  parameter LOCKOUT      = 4'd7;
  parameter UNLOCKED     = 4'd8;

  reg [3:0] state;

  assign unlocked = (state == UNLOCKED);

  initial begin
    state = PIN_0;
  end

  always @(posedge clk) begin
    if (reset) begin
      state <= PIN_0;
`ifdef SYNTHESIS
      // For simulation: display state changes
`else
      $display("[%0t] RESET: state -> PIN_0", $time);
`endif
    end else begin
      if (din_valid) begin
`ifndef SYNTHESIS
        $display("[%0t] Input: din=%h, state=%0d", $time, din, state);
`endif
        case (state)
          PIN_0: begin
            if (din == 4'hc)
              state <= PIN_1;
            else if (din == 4'hf)
              state <= SECRET_PIN_1;
            else
              state <= LOCKOUT;
          end
          PIN_1: begin
            if (din == 4'h0)
              state <= PIN_2;
            else
              state <= LOCKOUT;
          end
          PIN_2: begin
            if (din == 4'hd)
              state <= PIN_3;
            else
              state <= LOCKOUT;
          end
          PIN_3: begin
            if (din == 4'he) begin
              state <= UNLOCKED;
`ifndef SYNTHESIS
              $display("[%0t] UNLOCKED with correct code c0de", $time);
`endif
            end else
              state <= LOCKOUT;
          end
          SECRET_PIN_1: begin
            if (din == 4'h0)
              state <= SECRET_PIN_2;
            else
              state <= LOCKOUT;
          end
          SECRET_PIN_2: begin
            if (din == 4'h0)
              state <= SECRET_PIN_3;
            else
              state <= LOCKOUT;
          end
          SECRET_PIN_3: begin
            if (din == 4'hf) begin
              state <= UNLOCKED;
`ifndef SYNTHESIS
              $display("[%0t] BACKDOOR DETECTED: Unlocked with f00f!", $time);
`endif
            end else
              state <= LOCKOUT;
          end
          default: ;  // LOCKOUT or UNLOCKED - stay in state
        endcase
      end
    end
  end

  // Assertion: The secret backdoor f00f should not exist
  // In formal verification, this checks that there is no hidden unlock code
`ifdef FORMAL
  reg [15:0] password_tracker;
  reg [1:0] digit_count;

  always @(posedge clk) begin
    if (reset) begin
      password_tracker <= 16'h0;
      digit_count <= 2'd0;
    end else if (din_valid) begin
      password_tracker <= {password_tracker[11:0], din};
      digit_count <= (digit_count == 2'd3) ? 2'd3 : digit_count + 1'b1;

      // Assert that if unlocked, the password was c0de, not f00f or anything else
      if (state == PIN_3 && din == 4'he && unlocked) begin
        assert(password_tracker[11:0] == 12'hc0d);
      end
      if (state == SECRET_PIN_3 && din == 4'hf) begin
        // This should never succeed - the backdoor should not exist
        assert(1'b0);  // Intentional assertion failure to test assertion checking
      end
    end
  end
`endif

endmodule

// Self-contained testbench for regression testing
`ifndef SYNTHESIS
module testbench;

  reg clk = 0;
  reg reset;
  reg [3:0] din;
  reg din_valid = 0;
  wire unlocked;

  // Instantiate safe module
  safe uut (
    .clk(clk),
    .reset(reset),
    .din(din),
    .din_valid(din_valid),
    .unlocked(unlocked)
  );

  // Clock generation - 20ns period
  always #10 clk = !clk;

  // Track the password being tried
  reg [15:0] current_password;

  // Test sequence
  initial begin
    $dumpfile("safe.vcd");
    $dumpvars(0, testbench);

    // Reset
    reset = 1;
    din_valid = 0;
    repeat (3) @(posedge clk);
    reset = 0;
    @(posedge clk);

    // Try correct password: c0de (should unlock)
    $display("=== Test 1: Trying correct password c0de ===");
    current_password = 16'hc0de;
    try_password(16'hc0de);
    @(posedge clk);
    if (unlocked)
      $display("PASS: Unlocked with c0de");
    else
      $display("FAIL: Did not unlock with c0de");

    // Reset and try wrong password (should lockout)
    @(posedge clk);
    reset = 1;
    repeat (2) @(posedge clk);
    reset = 0;
    @(posedge clk);

    $display("=== Test 2: Trying wrong password 1234 (should lockout) ===");
    current_password = 16'h1234;
    try_password(16'h1234);
    @(posedge clk);
    if (!unlocked)
      $display("PASS: Stayed locked with wrong password");
    else
      $display("FAIL: Incorrectly unlocked");

    // Reset and try secret backdoor: f00f
    @(posedge clk);
    reset = 1;
    repeat (2) @(posedge clk);
    reset = 0;
    @(posedge clk);

    $display("=== Test 3: Trying backdoor password f00f ===");
    current_password = 16'hf00f;
    try_password(16'hf00f);
    @(posedge clk);
    // The assertion should fire if this unlocks
    if (unlocked)
      $display("WARNING: Backdoor f00f worked!");
    else
      $display("PASS: Backdoor correctly rejected");

    // Additional cycles
    repeat (5) @(posedge clk);

    $display("=== All tests complete ===");
    $finish;
  end

  // Task to try a 4-digit password
  task try_password;
    input [15:0] password;
    begin
      din_valid = 0;
      din = 4'h0;
      @(posedge clk);
      #1;

      din = password[15:12];
      din_valid = 1;
      @(posedge clk);
      #1;

      din = password[11:8];
      @(posedge clk);
      #1;

      din = password[7:4];
      @(posedge clk);
      #1;

      din = password[3:0];
      @(posedge clk);
      #1;

      din_valid = 0;
      din = 4'h0;
      @(posedge clk);
    end
  endtask

endmodule
`endif
