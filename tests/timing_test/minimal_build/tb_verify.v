`timescale 1ns/1ps

module tb_verify;
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

    // SPI signal aliases
    wire flash_clk = gpio_out[0];
    wire flash_csn = gpio_out[1];
    wire flash_d0  = gpio_out[2];

    // Track SPI command byte
    reg prev_flash_clk = 0;
    reg prev_flash_csn = 1;
    reg [7:0] cmd_byte = 0;
    integer spi_bit = 0;
    integer spi_byte = 0;
    integer cycle = 0;

    always @(posedge gpio_in[38]) begin
        cycle <= cycle + 1;

        // Per-cycle trace for first 50 cycles
        if (cycle < 50)
            $display("cycle %3d: flash_clk=%b csn=%b d0=%b", cycle, flash_clk, flash_csn, flash_d0);

        // Edge detection
        if (prev_flash_csn && !flash_csn) begin
            // CSN falling edge - transaction start
            spi_bit <= 0;
            spi_byte <= 0;
            cmd_byte <= 0;
        end

        if (!prev_flash_clk && flash_clk && !flash_csn) begin
            // Flash clock posedge while selected
            cmd_byte <= {cmd_byte[6:0], flash_d0};
            spi_bit <= spi_bit + 1;
            $display("  SPI posedge #%0d: MOSI=%b, byte=0x%02h", spi_bit+1, flash_d0, {cmd_byte[6:0], flash_d0});
            if (spi_bit == 7) begin
                $display("  >>> BYTE COMPLETE: 0x%02h (byte #%0d)", {cmd_byte[6:0], flash_d0}, spi_byte);
                spi_byte <= spi_byte + 1;
                spi_bit <= 0;
            end
        end

        prev_flash_clk <= flash_clk;
        prev_flash_csn <= flash_csn;
    end

    initial begin
        // Hold reset for 10 cycles
        #400;
        gpio_in[40] = 1;  // Release reset (active low: 1 = run)

        // Run for 300 cycles total
        #12000;

        $display("=== Simulation complete: %0d cycles ===", cycle);
        $finish;
    end
endmodule
