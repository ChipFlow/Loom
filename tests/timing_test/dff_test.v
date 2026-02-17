// Simple D flip-flop test for VCD timing validation
// This test exposes the issue where input changes at the same timestamp
// as clock edges are not captured correctly due to delayed_bit_changes

module dff_test (
    input  wire clk,
    input  wire d,
    output reg  q
);

// Simple D flip-flop - should capture d on posedge clk
always @(posedge clk) begin
    q <= d;
end

endmodule
