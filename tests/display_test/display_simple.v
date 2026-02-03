// Simple $display test for GEM
// Tests $display synthesis through yosys-slang

module display_simple (
    input wire clk,
    input wire rst,
    input wire [7:0] data_in,
    output reg [7:0] data_out
);

// Simple counter
reg [3:0] counter;

always @(posedge clk) begin
    if (rst) begin
        counter <= 4'd0;
        data_out <= 8'd0;
    end else begin
        counter <= counter + 1'b1;
        data_out <= data_in;

        // Display message when counter reaches certain values
        if (counter == 4'd5) begin
            $display("Counter reached 5, data_in = %h", data_in);
        end

        if (counter == 4'd10) begin
            $display("Counter reached 10, data_in = %h, data_out = %h", data_in, data_out);
        end
    end
end

endmodule
