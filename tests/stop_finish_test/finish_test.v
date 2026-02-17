// Simple test for $finish system task
// This module has a counter that triggers $finish when it reaches 5
module finish_test(
    input clk,
    input rst,
    output reg [3:0] count,
    output reg done
);

always @(posedge clk or posedge rst) begin
    if (rst) begin
        count <= 4'b0;
        done <= 1'b0;
    end else begin
        if (count == 4'd5) begin
            done <= 1'b1;
            // Note: In a real testbench, we'd call $finish here
            // But for synthesis, we just set the done flag
        end else begin
            count <= count + 1;
        end
    end
end

endmodule
