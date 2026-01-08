// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Techmap rules for converting yosys formal verification cells to GEM cells
//
// Usage in yosys:
//   techmap -map gem_formal.v

// Convert $check cells (assertions) to GEM_ASSERT
// $check has: TRG (trigger), EN (enable), A (condition), ARGS (format args)
// FLAVOR parameter: "assert", "assume", "cover"

(* techmap_celltype = "$check" *)
module _techmap_check (TRG, EN, A, ARGS);
    parameter TRG_WIDTH = 1;
    parameter TRG_ENABLE = 1;
    parameter TRG_POLARITY = 1'b1;
    parameter PRIORITY = 0;
    parameter ARGS_WIDTH = 0;
    parameter FORMAT = "";
    parameter FLAVOR = "assert";

    input [TRG_WIDTH-1:0] TRG;
    input EN;
    input A;
    input [ARGS_WIDTH-1:0] ARGS;

    // Only map assertion-type checks (not cover statements)
    // Cover statements are for verification, not runtime checking
    generate
        if (FLAVOR == "assert" || FLAVOR == "assume") begin
            // For triggered assertions, use the trigger as clock
            // TRG[0] is the clock signal (if TRG_WIDTH >= 1)
            if (TRG_ENABLE && TRG_WIDTH >= 1) begin
                GEM_ASSERT _TECHMAP_REPLACE_ (
                    .CLK(TRG[0]),
                    .EN(EN),
                    .A(A)
                );
            end else begin
                // For non-triggered assertions, tie CLK high (always active)
                GEM_ASSERT _TECHMAP_REPLACE_ (
                    .CLK(1'b1),
                    .EN(EN),
                    .A(A)
                );
            end
        end
        // Cover statements are silently dropped (not needed for simulation)
    endgenerate
endmodule

// Convert $assert cells directly
(* techmap_celltype = "$assert" *)
module _techmap_assert (A, EN);
    input A;
    input EN;

    GEM_ASSERT _TECHMAP_REPLACE_ (
        .CLK(1'b1),
        .EN(EN),
        .A(A)
    );
endmodule

// Convert $assume cells (treat as assertions for simulation)
(* techmap_celltype = "$assume" *)
module _techmap_assume (A, EN);
    input A;
    input EN;

    GEM_ASSERT _TECHMAP_REPLACE_ (
        .CLK(1'b1),
        .EN(EN),
        .A(A)
    );
endmodule

// Convert $print cells to GEM_DISPLAY
// $print has: TRG (trigger), EN (enable), ARGS (format arguments)
// FORMAT parameter contains the format string
// GEM_DISPLAY stores the format in an attribute and passes ARGS as MSG_ID (truncated to 32 bits)
// The full ARGS and FORMAT are reconstructed from the original $print cell
//
// Note: For full $display support, we need to track:
// 1. Format string (from FORMAT parameter)
// 2. Argument signals (from ARGS connection)
// 3. Argument widths (from ARGS_WIDTH or inferred from FORMAT)
//
// The GEM_DISPLAY cell uses MSG_ID to index into a message table built at compile time.
// For now, we use a simplified approach: MSG_ID[31:0] = first 32 bits of ARGS

(* techmap_celltype = "$print" *)
module _techmap_print (TRG, EN, ARGS);
    parameter TRG_WIDTH = 1;
    parameter TRG_ENABLE = 1;
    parameter TRG_POLARITY = 1'b1;
    parameter PRIORITY = 0;
    parameter ARGS_WIDTH = 32;
    parameter FORMAT = "";

    input [TRG_WIDTH-1:0] TRG;
    input EN;
    input [ARGS_WIDTH-1:0] ARGS;

    // For GEM_DISPLAY, we pass the first 32 bits of ARGS as MSG_ID
    // The actual formatting is done on the CPU side using the message table
    wire [31:0] msg_id;

    generate
        if (TRG_ENABLE && TRG_WIDTH >= 1) begin
            if (ARGS_WIDTH >= 32) begin
                assign msg_id = ARGS[31:0];
            end else if (ARGS_WIDTH > 0) begin
                assign msg_id = {{(32-ARGS_WIDTH){1'b0}}, ARGS};
            end else begin
                assign msg_id = 32'h0;
            end

            // Create GEM_DISPLAY with FORMAT preserved as attribute
            // For multi-bit triggers (e.g., {rst, clk}), use TRG[0] as the clock
            (* gem_format = FORMAT *)
            (* gem_args_width = ARGS_WIDTH *)
            GEM_DISPLAY _TECHMAP_REPLACE_ (
                .CLK(TRG[0]),
                .EN(EN),
                .MSG_ID(msg_id)
            );
        end
        // Non-triggered $print cells are not supported yet
    endgenerate
endmodule
