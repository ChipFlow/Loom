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
            if (TRG_ENABLE && TRG_WIDTH == 1) begin
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
