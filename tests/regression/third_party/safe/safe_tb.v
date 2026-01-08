// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
//
// Testbench for safe module adapted for GEM simulation
// Tests assertion checking with PIN-based safe cracker FSM

`timescale 1ns/1ns

module safe_tb;

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
  integer digit_count = 0;

  // Test sequence
  initial begin
    $dumpfile("safe.vcd");
    $dumpvars(0, safe_tb);

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

    // Reset and try secret backdoor: f00f (should NOT exist per assertion)
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
      $display("ASSERTION VIOLATION: Backdoor f00f worked!");
    else
      $display("PASS: Backdoor correctly rejected");

    // Additional cycles to see state
    repeat (10) @(posedge clk);

    $display("=== Test Complete ===");
    $finish;
  end

  // Task to try a 4-digit password
  task try_password;
    input [15:0] password;
    begin
      digit_count = 0;
      din_valid = 1;

      // Send 4 digits
      din = password[15:12];
      $display("  Digit 0: %h", password[15:12]);
      @(posedge clk);

      din = password[11:8];
      $display("  Digit 1: %h", password[11:8]);
      @(posedge clk);

      din = password[7:4];
      $display("  Digit 2: %h", password[7:4]);
      @(posedge clk);

      din = password[3:0];
      $display("  Digit 3: %h", password[3:0]);
      @(posedge clk);

      din_valid = 0;
      @(posedge clk);
    end
  endtask

  // Assertion: Only c0de should unlock
  // This will be converted to GEM_ASSERT during synthesis
`ifdef FORMAL
  always @(posedge clk) begin
    if (!reset) begin
      assert(!unlocked || current_password == 16'hc0de);
    end
  end
`endif

endmodule
