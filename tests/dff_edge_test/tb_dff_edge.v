`timescale 1ns/1ps

module tb_dff_edge;
    reg [6:0] gpio_in;
    wire [3:0] gpio_out;

    dff_edge_test dut (
        .gpio_in(gpio_in),
        .gpio_out(gpio_out)
    );

    // gpio_out[0]=DFF1(sysclk), [1]=DFF2(gated), [2]=DFF3(ext), [3]=DFF4(data)

    integer cycle = 0;
    integer errors = 0;

    task check;
        input [3:0] expected;
        input [255:0] desc;
        begin
            if (gpio_out !== expected) begin
                $display("FAIL cycle %0d: got %b expected %b - %0s", cycle, gpio_out, expected, desc);
                errors = errors + 1;
            end else begin
                $display("OK   cycle %0d: %b - %0s", cycle, gpio_out, desc);
            end
        end
    endtask

    initial begin
        $dumpfile("dff_edge_test.vcd");
        $dumpvars(0, tb_dff_edge);

        // Initialize: all inputs low, all Q=x initially
        gpio_in = 7'b0;
        #10;

        // === Test 1: System clock + gated clock latch ===
        // D=1 for DFF1/DFF2, gate=1 (enable gated clock), then sysclk posedge
        gpio_in[2] = 1;  // D for DFF1(sysclk) and DFF2(gated)
        gpio_in[3] = 1;  // Enable for gated clock
        #10;
        gpio_in[0] = 1;  // System clock rising edge
        #10;
        cycle = 1;
        // DFF1: CLK=buf(1)=1, was 0 → posedge, D=1 → Q=1
        // DFF2: CLK=AND(1,1)=1, was 0 → posedge, D=1 → Q=1
        // DFF3: no edge (ext_clk=0) → Q=0
        // DFF4: no edge (data_clk=0) → Q=0
        check(4'b0011, "sysclk+gated posedge, D=1, gate=1");

        gpio_in[0] = 0;  // Falling edge
        #10;

        // === Test 2: Gated clock disabled - DFF2 should NOT latch ===
        gpio_in[2] = 0;  // D=0 for DFF1 and DFF2
        gpio_in[3] = 0;  // Disable gate
        #10;
        gpio_in[0] = 1;  // System clock rising edge
        #10;
        cycle = 2;
        // DFF1: posedge, D=0 → Q=0
        // DFF2: CLK=AND(1,0)=0, was AND(0,0)=0 → no edge! → Q=1 (held)
        // DFF3: no edge → Q=0
        // DFF4: no edge → Q=0
        check(4'b0010, "sysclk posedge, gate=0: DFF2 holds");

        gpio_in[0] = 0;
        #10;

        // === Test 3: External clock only ===
        gpio_in[4] = 1;  // D for DFF3
        #10;
        gpio_in[1] = 1;  // External clock rising edge (sysclk stays 0)
        #10;
        cycle = 3;
        // DFF1: no edge (sysclk=0) → Q=0
        // DFF2: no edge → Q=1
        // DFF3: posedge, D=1 → Q=1
        // DFF4: no edge → Q=0
        check(4'b0110, "ext clock posedge only");

        gpio_in[1] = 0;
        #10;

        // === Test 4: Data signal as clock ===
        gpio_in[5] = 1;  // D for DFF4
        #10;
        gpio_in[6] = 1;  // "data clock" rising edge
        #10;
        cycle = 4;
        // DFF1: no edge → Q=0
        // DFF2: no edge → Q=1
        // DFF3: no edge → Q=1
        // DFF4: posedge, D=1 → Q=1
        check(4'b1110, "data clock posedge only");

        gpio_in[6] = 0;
        #10;

        // === Test 5: Only sysclk toggles, others hold ===
        gpio_in[2] = 1;  // D=1 for DFF1
        gpio_in[3] = 0;  // Gate stays disabled
        gpio_in[4] = 0;  // D=0 for DFF3 (but ext_clk doesn't toggle)
        gpio_in[5] = 0;  // D=0 for DFF4 (but data_clk doesn't toggle)
        #10;
        gpio_in[0] = 1;  // System clock rising edge
        #10;
        cycle = 5;
        // DFF1: posedge, D=1 → Q=1
        // DFF2: CLK=AND(1,0)=0, no edge → Q=1 (held from test 1)
        // DFF3: no edge → Q=1 (held from test 3)
        // DFF4: no edge → Q=1 (held from test 4)
        check(4'b1111, "sysclk only, others hold previous values");

        gpio_in[0] = 0;
        #10;

        // === Test 6: Re-enable gate with D=0, verify DFF2 latches ===
        gpio_in[2] = 0;  // D=0 for DFF1/DFF2
        gpio_in[3] = 1;  // Re-enable gate
        #10;
        gpio_in[0] = 1;  // System clock rising edge
        #10;
        cycle = 6;
        // DFF1: posedge, D=0 → Q=0
        // DFF2: CLK=AND(1,1)=1, was AND(0,1)=0 → posedge, D=0 → Q=0
        // DFF3: no edge → Q=1
        // DFF4: no edge → Q=1
        check(4'b1100, "gate re-enabled, DFF2 latches D=0");

        gpio_in[0] = 0;
        #10;

        // === Test 7: Simultaneous edges on ext and data clocks ===
        gpio_in[4] = 0;  // D=0 for DFF3
        gpio_in[5] = 0;  // D=0 for DFF4
        #10;
        gpio_in[1] = 1;  // External clock posedge
        gpio_in[6] = 1;  // Data clock posedge
        #10;
        cycle = 7;
        // DFF1: no edge → Q=0
        // DFF2: no edge (sysclk=0) → Q=0
        // DFF3: posedge, D=0 → Q=0
        // DFF4: posedge, D=0 → Q=0
        check(4'b0000, "ext+data clocks posedge, D=0");

        // === Summary ===
        #10;
        if (errors == 0)
            $display("=== ALL %0d TESTS PASSED ===", cycle);
        else
            $display("=== %0d of %0d TESTS FAILED ===", errors, cycle);

        $finish;
    end
endmodule
