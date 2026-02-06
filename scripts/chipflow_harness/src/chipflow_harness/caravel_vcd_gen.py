"""VCD generator for Caravel OpenFrame wrapper designs.

The Caravel OpenFrame wrapper uses a different GPIO interface:
- gpio_in[43:0]: Input values to the design
- gpio_out[43:0]: Output values from the design
- gpio_oeb[43:0]: Output enable (active low) for each GPIO

This generator maps ChipFlow pin indices to Caravel GPIO indices.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chipflow_harness.pin_map import PinsLock, load_pins_lock
from chipflow_harness.vcd_writer import VCDWriter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


@dataclass
class CaravelVCDGenerator:
    """Generates VCD stimulus for Caravel OpenFrame wrapper designs."""

    pins_lock: PinsLock
    clock_period_ps: int = 40_000_000  # 25 MHz = 40ns
    clock_hz: int = 25_000_000
    reset_cycles: int = 10
    max_cycles: int = 1_000_000
    baud_rate: int = 115200
    gpio_width: int = 44  # Caravel uses 44 GPIOs

    # Caravel-specific pin mappings
    # These map the SoC wrapper pins to Caravel GPIO indices
    # Based on typical Caravel/OpenFrame pinout:
    # - GPIO[2] = clock input
    # - GPIO[3] = reset_n
    # GPIO indices for peripherals come from pins.lock

    # Internal state
    _gpio_in: int = field(default=0, init=False)
    _current_cycle: int = field(default=0, init=False)

    def _pin_to_gpio(self, pin_index: int) -> int:
        """Convert ChipFlow pin index to Caravel GPIO index.

        This is a direct mapping - the pins.lock pin indices
        should correspond to Caravel GPIO indices.
        """
        return pin_index

    def generate(
        self,
        input_path: Path,
        output_path: Path,
        module_name: str = "openframe_project_wrapper",
    ) -> None:
        """Generate VCD with Caravel GPIO interface."""
        with open(input_path) as f:
            data = json.load(f)

        commands = data.get("commands", [])

        with VCDWriter(output_path, timescale="1ps", module_name=module_name) as vcd:
            # Register Caravel signals
            vcd.register_signal("por_l", 1, module_name)
            vcd.register_signal("porb_h", 1, module_name)
            vcd.register_signal("porb_l", 1, module_name)
            vcd.register_signal("resetb_h", 1, module_name)
            vcd.register_signal("resetb_l", 1, module_name)
            vcd.register_signal("gpio_in", self.gpio_width, module_name)
            vcd.register_signal("gpio_in_h", self.gpio_width, module_name)
            vcd.register_signal("gpio_loopback_one", self.gpio_width, module_name)
            vcd.register_signal("gpio_loopback_zero", self.gpio_width, module_name)
            vcd.register_signal("mask_rev", 32, module_name)
            # Also register clk_in as a separate signal for timing simulation
            # (The netlist may have `wire clk_in` that's not directly connected to gpio_in)
            vcd.register_signal("clk_in", 1, module_name)

            # Collect all events
            all_events: list[tuple[int, str, int]] = []

            # Get clock and reset GPIO indices from pins.lock
            clock_gpio = self._pin_to_gpio(self.pins_lock.clock_pin)
            reset_gpio = self._pin_to_gpio(self.pins_lock.reset_pin)

            log.info(f"Clock on GPIO[{clock_gpio}], Reset on GPIO[{reset_gpio}]")

            # Initialize at time 0
            vcd.set_time(0)

            # Power-on reset signals (active high for por_l, porb_h, porb_l)
            vcd.set_value("por_l", 1)
            vcd.set_value("porb_h", 1)
            vcd.set_value("porb_l", 1)
            vcd.set_value("resetb_h", 0)  # Assert reset (active high for resetb means 0)
            vcd.set_value("resetb_l", 0)
            vcd.set_value("gpio_in", 0)
            vcd.set_value("gpio_in_h", 0)
            vcd.set_value("gpio_loopback_one", 0)
            vcd.set_value("gpio_loopback_zero", 0)
            vcd.set_value("mask_rev", 0)
            vcd.set_value("clk_in", 0)

            # Initial gpio_in value - set clock low, reset asserted
            self._gpio_in = 0

            # Reset sequence - hold reset, generate clock edges
            for cycle in range(self.reset_cycles):
                time_ps = cycle * self.clock_period_ps

                # Clock low (falling edge)
                gpio_clk_low = self._gpio_in & ~(1 << clock_gpio)
                all_events.append((time_ps, "gpio_in", gpio_clk_low))
                all_events.append((time_ps, "clk_in", 0))

                # Clock high (rising edge at half period)
                gpio_clk_high = self._gpio_in | (1 << clock_gpio)
                all_events.append((time_ps + self.clock_period_ps // 2, "gpio_in", gpio_clk_high))
                all_events.append((time_ps + self.clock_period_ps // 2, "clk_in", 1))

            # Deassert reset - set both legacy signals AND gpio_in[reset_gpio]
            reset_time = self.reset_cycles * self.clock_period_ps
            all_events.append((reset_time, "resetb_h", 1))
            all_events.append((reset_time, "resetb_l", 1))

            # Also set the reset bit in gpio_in (active-low reset, so 1 = deasserted)
            # The reset_gpio from pins.lock indicates which GPIO bit controls reset
            self._gpio_in |= (1 << reset_gpio)
            log.info(f"Reset deasserted at cycle {self.reset_cycles} (gpio_in[{reset_gpio}] = 1)")

            self._current_cycle = self.reset_cycles

            # Process commands
            for cmd in commands:
                cmd_type = cmd.get("type")

                if cmd_type == "action":
                    peripheral = cmd["peripheral"]
                    event = cmd["event"]
                    payload = cmd.get("payload", "")

                    self._process_action(peripheral, event, payload, all_events)

                elif cmd_type == "wait":
                    peripheral = cmd["peripheral"]
                    event = cmd["event"]

                    # Estimate cycles to wait
                    wait_cycles = self._estimate_wait_cycles(peripheral, event)
                    self._current_cycle += wait_cycles

            # Generate clock waveform for remaining cycles
            # Use max_cycles if specified and larger than current cycle count
            total_cycles = max(self._current_cycle + 100, self.max_cycles)
            log.info(f"Generating {total_cycles} clock cycles")

            for cycle in range(self.reset_cycles, total_cycles):
                time_ps = cycle * self.clock_period_ps

                # Clock low
                gpio_clk_low = self._gpio_in & ~(1 << clock_gpio)
                all_events.append((time_ps, "gpio_in", gpio_clk_low))
                all_events.append((time_ps, "clk_in", 0))

                # Clock high
                gpio_clk_high = self._gpio_in | (1 << clock_gpio)
                all_events.append((time_ps + self.clock_period_ps // 2, "gpio_in", gpio_clk_high))
                all_events.append((time_ps + self.clock_period_ps // 2, "clk_in", 1))

            # Sort and write events
            all_events.sort(key=lambda x: (x[0], x[1]))

            for time_ps, signal, value in all_events:
                vcd.set_value(signal, value, time_ps)

        log.info(f"Generated VCD: {output_path}")

    def _process_action(
        self, peripheral: str, event: str, payload: Any, events: list[tuple[int, str, int]]
    ) -> None:
        """Process an action command and generate GPIO events."""
        mappings = self.pins_lock.get_peripheral_mappings(peripheral)

        if peripheral.startswith("gpio"):
            # GPIO set - update gpio_in bits
            gpio_mapping = next((m for m in mappings if m.signal == "gpio"), None)
            if gpio_mapping:
                if isinstance(payload, str):
                    # Parse string like "10101010" or "1010ZZZZ"
                    for i, char in enumerate(reversed(payload)):
                        if i < len(gpio_mapping.pin_indices):
                            pin_idx = gpio_mapping.pin_indices[i]
                        else:
                            pin_idx = -1
                        if pin_idx >= 0:
                            gpio_idx = self._pin_to_gpio(pin_idx)
                            if char == "1":
                                self._gpio_in |= (1 << gpio_idx)
                            elif char == "0":
                                self._gpio_in &= ~(1 << gpio_idx)
                            # Z leaves unchanged
                else:
                    for i, pin_idx in enumerate(gpio_mapping.pin_indices):
                        gpio_idx = self._pin_to_gpio(pin_idx)
                        if (payload >> i) & 1:
                            self._gpio_in |= (1 << gpio_idx)
                        else:
                            self._gpio_in &= ~(1 << gpio_idx)

                time_ps = self._current_cycle * self.clock_period_ps
                events.append((time_ps, "gpio_in", self._gpio_in))

        elif peripheral.startswith("uart"):
            # UART TX (to DUT RX) - generate bit waveform
            rx_mapping = next((m for m in mappings if m.signal == "rx"), None)
            if rx_mapping and event == "tx":
                byte_val = int(payload)
                gpio_idx = self._pin_to_gpio(rx_mapping.pin_indices[0])

                # UART timing
                cycles_per_bit = self.clock_hz // self.baud_rate
                cycle = self._current_cycle

                # Start bit (low)
                self._gpio_in &= ~(1 << gpio_idx)
                events.append((cycle * self.clock_period_ps, "gpio_in", self._gpio_in))
                cycle += cycles_per_bit

                # Data bits (LSB first)
                for i in range(8):
                    bit = (byte_val >> i) & 1
                    if bit:
                        self._gpio_in |= (1 << gpio_idx)
                    else:
                        self._gpio_in &= ~(1 << gpio_idx)
                    events.append((cycle * self.clock_period_ps, "gpio_in", self._gpio_in))
                    cycle += cycles_per_bit

                # Stop bit (high)
                self._gpio_in |= (1 << gpio_idx)
                events.append((cycle * self.clock_period_ps, "gpio_in", self._gpio_in))

    def _estimate_wait_cycles(self, peripheral: str, event: str) -> int:
        """Estimate cycles needed to wait for an event."""
        if event == "tx" and peripheral.startswith("uart"):
            cycles_per_bit = self.clock_hz // self.baud_rate
            return cycles_per_bit * 10

        elif event == "change" and peripheral.startswith("gpio"):
            return 100

        elif event in ("select", "deselect", "data") and "spi" in peripheral:
            return 500

        elif event in ("start", "stop", "address") and peripheral.startswith("i2c"):
            cycles_per_bit = self.clock_hz // 100_000
            return cycles_per_bit * 9

        return 1000


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate VCD stimulus for Caravel OpenFrame wrapper"
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to input.json test file",
    )
    parser.add_argument(
        "--pins-lock",
        type=Path,
        required=True,
        help="Path to pins.lock file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output VCD file path",
    )
    parser.add_argument(
        "--clock-period-ps",
        type=int,
        default=40_000_000,
        help="Clock period in picoseconds (default: 40000000 = 25MHz)",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=115200,
        help="UART baud rate (default: 115200)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=1_000_000,
        help="Maximum simulation cycles (default: 1000000)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        pins_lock = load_pins_lock(args.pins_lock)
    except Exception as e:
        log.error(f"Failed to load pins.lock: {e}")
        return 1

    clock_hz = 1_000_000_000_000 // args.clock_period_ps

    generator = CaravelVCDGenerator(
        pins_lock=pins_lock,
        clock_period_ps=args.clock_period_ps,
        clock_hz=clock_hz,
        baud_rate=args.baud_rate,
        max_cycles=args.max_cycles,
    )

    try:
        generator.generate(args.input_json, args.output)
    except Exception as e:
        log.error(f"Failed to generate VCD: {e}")
        if args.verbose:
            raise
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
