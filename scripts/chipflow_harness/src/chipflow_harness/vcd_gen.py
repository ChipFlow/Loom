"""VCD generator from ChipFlow input.json test format."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chipflow_harness.models.gpio import GPIOModel
from chipflow_harness.models.i2c import I2CModel
from chipflow_harness.models.spi import SPIModel
from chipflow_harness.models.uart import UARTModel
from chipflow_harness.pin_map import PinMapping, PinsLock, load_pins_lock
from chipflow_harness.vcd_writer import VCDWriter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


@dataclass
class PeripheralState:
    """State for a peripheral instance."""

    name: str
    model: GPIOModel | UARTModel | SPIModel | I2CModel
    mappings: dict[str, PinMapping] = field(default_factory=dict)


@dataclass
class VCDGenerator:
    """Generates VCD stimulus from ChipFlow input.json commands."""

    pins_lock: PinsLock
    clock_period_ps: int = 40_000_000  # 25 MHz = 40ns = 40,000,000 ps
    clock_hz: int = 25_000_000
    reset_cycles: int = 10
    max_cycles: int = 1_000_000
    baud_rate: int = 115200

    # Internal state
    _peripherals: dict[str, PeripheralState] = field(default_factory=dict, init=False)
    _vcd: VCDWriter | None = field(default=None, init=False)
    _current_cycle: int = field(default=0, init=False)
    _pending_actions: list[dict[str, Any]] = field(default_factory=list, init=False)
    _signal_ids: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize peripheral models."""
        self._init_peripherals()

    def _init_peripherals(self) -> None:
        """Create peripheral models from pins.lock."""
        for mapping in self.pins_lock.all_io_pins():
            periph_name = mapping.peripheral

            if periph_name not in self._peripherals:
                # Determine peripheral type from name
                if periph_name.startswith("gpio"):
                    model = GPIOModel(width=8)
                elif periph_name.startswith("uart"):
                    model = UARTModel(baud_rate=self.baud_rate, clock_hz=self.clock_hz)
                elif periph_name.startswith("user_spi") or periph_name.startswith("spi"):
                    model = SPIModel(clock_hz=self.clock_hz)
                elif periph_name.startswith("i2c"):
                    model = I2CModel(clock_hz=self.clock_hz)
                else:
                    # Generic model - use GPIO
                    model = GPIOModel(width=mapping.width)

                self._peripherals[periph_name] = PeripheralState(
                    name=periph_name, model=model
                )

            self._peripherals[periph_name].mappings[mapping.signal] = mapping

    def _register_signals(self, vcd: VCDWriter, module_name: str) -> None:
        """Register all signals in the VCD file."""
        # Register clock and reset
        vcd.register_signal("clk", 1, module_name)
        vcd.register_signal("rst_n", 1, module_name)

        # Register all IO pins
        for mapping in self.pins_lock.all_io_pins():
            signal_name = mapping.port_name
            vcd.register_signal(signal_name, mapping.width, module_name)

    def _write_clock_edge(self, vcd: VCDWriter, cycle: int, rising: bool) -> None:
        """Write a clock edge."""
        time_ps = cycle * self.clock_period_ps
        if rising:
            time_ps += self.clock_period_ps // 2  # Rising edge at half period
        vcd.set_value("clk", 1 if rising else 0, time_ps)

    def _write_reset_sequence(self, vcd: VCDWriter) -> int:
        """Write initial reset sequence. Returns ending cycle."""
        # Initialize clock low
        vcd.set_time(0)
        vcd.set_value("clk", 0)
        vcd.set_value("rst_n", 0)

        # Initialize all inputs to safe defaults
        for mapping in self.pins_lock.get_input_pins():
            if mapping.direction in ("i", "io"):
                if mapping.width == 1:
                    vcd.set_value(mapping.port_name, 1)  # Default high (idle)
                else:
                    vcd.set_value(mapping.port_name, 0)

        # Run reset cycles
        for cycle in range(self.reset_cycles):
            # Falling edge
            time_ps = cycle * self.clock_period_ps
            vcd.set_value("clk", 0, time_ps)

            # Rising edge
            time_ps += self.clock_period_ps // 2
            vcd.set_value("clk", 1, time_ps)

        # Deassert reset
        time_ps = self.reset_cycles * self.clock_period_ps
        vcd.set_value("rst_n", 1, time_ps)

        return self.reset_cycles

    def _process_action(self, cmd: dict[str, Any]) -> None:
        """Process an action command."""
        peripheral = cmd["peripheral"]
        event = cmd["event"]
        payload = cmd.get("payload", "")

        if peripheral not in self._peripherals:
            log.warning(f"Unknown peripheral: {peripheral}")
            return

        state = self._peripherals[peripheral]
        model = state.model

        if isinstance(model, GPIOModel):
            if event == "set":
                model.set_value(payload)
        elif isinstance(model, UARTModel):
            if event == "tx":
                model.queue_transmit(int(payload))
        elif isinstance(model, SPIModel):
            if event == "set_data":
                model.set_tx_data(int(payload))
            elif event == "set_width":
                model.set_tx_width(int(payload))
        elif isinstance(model, I2CModel):
            if event == "ack":
                model.on_ack()
            elif event == "set_data":
                model.set_tx_data(int(payload))

    def _check_wait_condition(self, cmd: dict[str, Any]) -> bool:
        """Check if a wait condition would be satisfied.

        For VCD generation, we don't actually check - we just generate
        enough stimulus and let the simulation determine if it passes.
        Returns True to proceed.
        """
        # In VCD generation mode, we just move forward
        return True

    def _write_peripheral_stimulus(
        self, vcd: VCDWriter, cycle: int
    ) -> list[tuple[int, str, int]]:
        """Generate stimulus from all peripheral models.

        Returns list of (cycle, signal, value) tuples.
        """
        events: list[tuple[int, str, int]] = []

        for state in self._peripherals.values():
            model = state.model

            if isinstance(model, UARTModel):
                # Generate UART TX waveform
                rx_mapping = state.mappings.get("rx")
                if rx_mapping and model._tx_queue:
                    for offset, value in model.get_tx_waveform(cycle):
                        events.append((offset, rx_mapping.port_name, value))

            elif isinstance(model, GPIOModel):
                # Set GPIO values
                gpio_mapping = state.mappings.get("gpio")
                if gpio_mapping:
                    value = model.get_value()
                    events.append((cycle, gpio_mapping.port_name, value))

        return events

    def generate(self, input_path: Path, output_path: Path, module_name: str = "testbench") -> None:
        """Generate VCD from input.json commands."""
        with open(input_path) as f:
            data = json.load(f)

        commands = data.get("commands", [])

        with VCDWriter(output_path, timescale="1ps", module_name=module_name) as vcd:
            self._vcd = vcd
            self._register_signals(vcd, module_name)

            # Initialize at time 0
            vcd.set_time(0)
            vcd.set_value("clk", 0)
            vcd.set_value("rst_n", 0)

            # Initialize all inputs to safe defaults
            for mapping in self.pins_lock.get_input_pins():
                if mapping.direction in ("i", "io"):
                    if mapping.width == 1:
                        vcd.set_value(mapping.port_name, 1)  # Default high (idle)
                    else:
                        vcd.set_value(mapping.port_name, 0)

            # Collect all events: (time_ps, signal, value)
            all_events: list[tuple[int, str, int | str]] = []

            # Add reset sequence events
            for cycle in range(self.reset_cycles):
                time_ps = cycle * self.clock_period_ps
                all_events.append((time_ps, "clk", 0))
                all_events.append((time_ps + self.clock_period_ps // 2, "clk", 1))

            # Deassert reset
            reset_time = self.reset_cycles * self.clock_period_ps
            all_events.append((reset_time, "rst_n", 1))

            self._current_cycle = self.reset_cycles

            # Process commands and collect peripheral events
            for cmd in commands:
                cmd_type = cmd.get("type")

                if cmd_type == "action":
                    self._process_action(cmd)
                    events = self._write_peripheral_stimulus(vcd, self._current_cycle)
                    for cycle, signal, value in events:
                        time_ps = cycle * self.clock_period_ps
                        all_events.append((time_ps, signal, value))

                elif cmd_type == "wait":
                    peripheral = cmd["peripheral"]
                    event = cmd["event"]

                    # Estimate cycles to wait based on event type
                    wait_cycles = 1000  # Default

                    if event == "tx" and peripheral.startswith("uart"):
                        cycles_per_bit = self.clock_hz // self.baud_rate
                        wait_cycles = cycles_per_bit * 10

                    elif event == "change" and peripheral.startswith("gpio"):
                        wait_cycles = 100

                    elif event in ("select", "deselect", "data") and "spi" in peripheral:
                        wait_cycles = 500

                    elif event in ("start", "stop", "address") and peripheral.startswith("i2c"):
                        cycles_per_bit = self.clock_hz // 100_000
                        wait_cycles = cycles_per_bit * 9

                    self._current_cycle += wait_cycles

            # Add clock waveform for all cycles
            total_cycles = self._current_cycle + 100
            log.info(f"Generating {total_cycles} clock cycles")

            for cycle in range(self.reset_cycles, total_cycles):
                time_ps = cycle * self.clock_period_ps
                all_events.append((time_ps, "clk", 0))
                all_events.append((time_ps + self.clock_period_ps // 2, "clk", 1))

            # Sort all events by time and write them
            all_events.sort(key=lambda x: (x[0], x[1]))  # Sort by time, then signal name

            for time_ps, signal, value in all_events:
                vcd.set_value(signal, value, time_ps)

        log.info(f"Generated VCD: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate VCD stimulus from ChipFlow input.json"
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
        "--module-name",
        type=str,
        default="testbench",
        help="VCD module name (default: testbench)",
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

    # Load pins.lock
    try:
        pins_lock = load_pins_lock(args.pins_lock)
    except Exception as e:
        log.error(f"Failed to load pins.lock: {e}")
        return 1

    # Calculate clock_hz from period
    clock_hz = 1_000_000_000_000 // args.clock_period_ps  # ps to Hz

    # Create generator
    generator = VCDGenerator(
        pins_lock=pins_lock,
        clock_period_ps=args.clock_period_ps,
        clock_hz=clock_hz,
        baud_rate=args.baud_rate,
        max_cycles=args.max_cycles,
    )

    # Generate VCD
    try:
        generator.generate(args.input_json, args.output, args.module_name)
    except Exception as e:
        log.error(f"Failed to generate VCD: {e}")
        if args.verbose:
            raise
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
