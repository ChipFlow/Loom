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
