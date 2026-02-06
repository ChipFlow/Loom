"""Parser for ChipFlow pins.lock files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class PinMapping:
    """Mapping of a peripheral signal to package pins."""

    peripheral: str  # e.g., "uart_0", "gpio_1"
    signal: str  # e.g., "tx", "rx", "gpio"
    pin_indices: list[int]  # Package pin numbers
    direction: str  # "i", "o", "io"
    width: int
    port_name: str  # e.g., "soc_uart_0_tx"
    invert: list[bool] = field(default_factory=list)

    def is_input(self) -> bool:
        """Returns True if this signal accepts input (i or io)."""
        return self.direction in ("i", "io")

    def is_output(self) -> bool:
        """Returns True if this signal produces output (o or io)."""
        return self.direction in ("o", "io")


@dataclass
class PinsLock:
    """Parsed pins.lock file."""

    process: str  # "ihp_sg13g2", "sky130"
    port_map: dict[str, Any]  # Full port mapping structure
    clock_pin: int  # Clock pin number
    reset_pin: int  # Reset pin number
    clock_port: str  # Clock port name
    reset_port: str  # Reset port name
    _mappings: dict[tuple[str, str], PinMapping] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> PinsLock:
        """Load pins.lock from file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PinsLock:
        """Parse pins.lock from JSON dict."""
        process = data["process"]
        port_map = data["port_map"]["ports"]

        # Extract clock and reset from _core.bringup_pins
        core = port_map.get("_core", {}).get("bringup_pins", {})
        clock_info = core.get("clk", {})
        reset_info = core.get("rst_n", {})

        # Pin format: [package_pin, type, gpio_index, ...]
        # For Caravel OpenFrame, the gpio_index (index 2) is what we need
        clock_pins = clock_info.get("pins", [[0, "gpio", 0]])[0]
        reset_pins = reset_info.get("pins", [[0, "gpio", 0]])[0]
        clock_pin = clock_pins[2] if len(clock_pins) > 2 else clock_pins[0]
        reset_pin = reset_pins[2] if len(reset_pins) > 2 else reset_pins[0]
        clock_port = clock_info.get("port_name", "clk")
        reset_port = reset_info.get("port_name", "rst_n")

        pins_lock = cls(
            process=process,
            port_map=port_map,
            clock_pin=clock_pin,
            reset_pin=reset_pin,
            clock_port=clock_port,
            reset_port=reset_port,
        )

        # Build mappings for soc peripherals
        soc = port_map.get("soc", {})
        for periph_name, periph_signals in soc.items():
            for signal_name, signal_info in periph_signals.items():
                if signal_info.get("type") != "io":
                    continue

                iomodel = signal_info.get("iomodel", {})
                # Extract GPIO indices from pin tuples [package_pin, type, gpio_index, ...]
                raw_pins = signal_info.get("pins", [])
                gpio_indices = []
                for pin in raw_pins:
                    if isinstance(pin, list) and len(pin) > 2:
                        gpio_indices.append(pin[2])  # gpio_index is at index 2
                    else:
                        gpio_indices.append(pin[0] if isinstance(pin, list) else pin)
                mapping = PinMapping(
                    peripheral=periph_name,
                    signal=signal_name,
                    pin_indices=gpio_indices,
                    direction=iomodel.get("direction", "io"),
                    width=iomodel.get("width", 1),
                    port_name=signal_info.get("port_name", ""),
                    invert=iomodel.get("invert", []),
                )
                pins_lock._mappings[(periph_name, signal_name)] = mapping

        return pins_lock

    def get_mapping(self, peripheral: str, signal: str) -> PinMapping | None:
        """Get mapping for a specific peripheral signal."""
        return self._mappings.get((peripheral, signal))

    def get_peripheral_mappings(self, peripheral: str) -> list[PinMapping]:
        """Get all mappings for a peripheral."""
        return [m for (p, _), m in self._mappings.items() if p == peripheral]

    def all_io_pins(self) -> Iterator[PinMapping]:
        """Iterate over all IO pin mappings."""
        yield from self._mappings.values()

    def get_input_pins(self) -> list[PinMapping]:
        """Get all input-capable pin mappings."""
        return [m for m in self._mappings.values() if m.is_input()]

    def get_output_pins(self) -> list[PinMapping]:
        """Get all output-capable pin mappings."""
        return [m for m in self._mappings.values() if m.is_output()]


def load_pins_lock(path: Path) -> PinsLock:
    """Load and parse a pins.lock file."""
    return PinsLock.from_file(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m chipflow_harness.pin_map <pins.lock>")
        sys.exit(1)

    pins = load_pins_lock(Path(sys.argv[1]))
    print(f"Process: {pins.process}")
    print(f"Clock: pin {pins.clock_pin} ({pins.clock_port})")
    print(f"Reset: pin {pins.reset_pin} ({pins.reset_port})")
    print("\nIO Mappings:")
    for mapping in pins.all_io_pins():
        print(f"  {mapping.peripheral}.{mapping.signal}: pins={mapping.pin_indices} "
              f"dir={mapping.direction} width={mapping.width}")
