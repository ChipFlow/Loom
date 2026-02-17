`timescale 1ns/1ps

module tb_trace;
    reg [43:0] gpio_in;
    wire [43:0] gpio_out;
    wire [43:0] gpio_oeb;

    // Instantiate the design
    openframe_project_wrapper dut (
        .gpio_in(gpio_in),
        .gpio_out(gpio_out),
        .gpio_oeb(gpio_oeb)
    );

    // Clock on gpio_in[38]
    initial begin
        gpio_in = 44'h0;
        gpio_in[40] = 0;  // Reset asserted (active low: 0 = reset)
    end

    // Clock generation - 25MHz = 40ns period
    always begin
        #20 gpio_in[38] = ~gpio_in[38];
    end

    integer cycle = 0;

    // Trace at every rising edge (matches GEM's cycle numbering)
    always @(posedge gpio_in[38]) begin
        cycle <= cycle + 1;
        // Print CSV: cycle, flash_clk, flash_csn, flash_d0, buffer_io0_o
        $display("TRACE,%0d,%b,%b,%b,%b",
            cycle,
            gpio_out[0],   // flash_clk
            gpio_out[1],   // flash_csn
            gpio_out[2],   // flash_d0 (MOSI)
            dut.\inst$top.soc.spiflash.phy.io_streamer.buffer_io0.o  // D input to DFF
        );
    end

    initial begin
        // Hold reset for 10 cycles
        #400;
        gpio_in[40] = 1;  // Release reset (active low: 1 = run)

        // Run for 50 cycles total
        #2000;

        $display("=== Done ===");
        $finish;
    end
endmodule
