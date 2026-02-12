#!/usr/bin/env python3
"""Generate a VCD with corrected reset polarity."""

# GPIO[7] = UART RX idle
# GPIO[38] = clock
# GPIO[40] = reset (with invert=true in pins.lock, so 1 = held, 0 = released)

header = r"""$date
	Test VCD with corrected reset polarity
$end
$version
	Manual test
$end
$timescale
	1ps
$end
$scope module openframe_project_wrapper $end
$var wire 1 ! por_l $end
$var wire 1 " porb_h $end
$var wire 1 # porb_l $end
$var wire 1 $ resetb_h $end
$var wire 1 % resetb_l $end
$var wire 44 & gpio_in $end
$var wire 44 ' gpio_in_h $end
$var wire 44 ( gpio_loopback_one $end
$var wire 44 ) gpio_loopback_zero $end
$var wire 32 * mask_rev $end
$var wire 1 + clk_in $end
$upscope $end
$enddefinitions $end
#0
$dumpvars
1!
1"
1#
0$
0%
b00000000000000000000000000000000000010000000 &
b00000000000000000000000000000000000000000000 '
b00000000000000000000000000000000000000000000 (
b00000000000000000000000000000000000000000000 )
b00000000000000000000000000000000 *
0+
$end
"""

# GPIO bit positions
uart_rx = 1 << 7
clk = 1 << 38
rst = 1 << 40

clock_period = 40000  # ps
reset_cycles = 10
total_cycles = 10000  # Run longer for firmware execution

lines = [header]

t = 0
for cycle in range(total_cycles):
    # Rising edge of clock
    t += clock_period // 2

    gpio_val = uart_rx | clk  # Clock high
    if cycle >= reset_cycles:
        gpio_val |= rst  # After reset: gpio=1 → rst_n_sync.rst=0 → CPU runs

    lines.append(f"#{t}")
    lines.append("1+")
    lines.append(f"b{gpio_val:044b} &")

    if cycle == reset_cycles:
        lines.append("1$")
        lines.append("1%")

    # Falling edge of clock
    t += clock_period // 2

    gpio_val = uart_rx  # Clock low
    if cycle >= reset_cycles:
        gpio_val |= rst  # After reset: gpio=1 → rst_n_sync.rst=0 → CPU runs

    lines.append(f"#{t}")
    lines.append("0+")
    lines.append(f"b{gpio_val:044b} &")

with open("test_reset_fix.vcd", "w") as f:
    f.write("\n".join(lines))

print(f"Generated VCD with {total_cycles} cycles")
