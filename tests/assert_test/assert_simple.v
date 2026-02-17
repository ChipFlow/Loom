// Simple assertion test for GEM
// Uses immediate assertions which synthesize to $assert/$check cells

module assert_simple (
    input wire clk,
    input wire rst,
    input wire [3:0] data_in,
    output reg [3:0] data_out,
    output reg overflow_flag
);

// Simple counter to generate test conditions
reg [3:0] counter;

always @(posedge clk) begin
    if (rst) begin
        counter <= 4'd0;
        data_out <= 4'd0;
        overflow_flag <= 1'b0;
    end else begin
        counter <= counter + 1'b1;
        data_out <= data_in;

        // Track overflow condition
        if (data_in == 4'hF && counter > 4'd10) begin
            overflow_flag <= 1'b1;
        end
    end
end

// Wire for assertion condition
wire no_overflow_condition = !(data_in == 4'hF && counter > 4'd10) || rst;

// For formal verification - this should create an $assert cell
// We'll use the (* keep *) attribute to prevent optimization

`ifdef FORMAL
always @(posedge clk) begin
    // Assertion: no overflow when reset is active
    assert(rst || !(data_in == 4'hF && counter > 4'd10));
end
`endif

endmodule
