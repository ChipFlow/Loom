#!/usr/bin/env python3
"""Generate a Rust testbench from pins.lock for timing simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from .pin_map import PinsLock, load_pins_lock


def find_flash_mapping(pins: PinsLock) -> dict[str, Any] | None:
    """Find QSPI flash interface mapping."""
    flash = pins.get_peripheral_mappings("flash")
    if not flash:
        return None

    mapping = {}
    for m in flash:
        if m.signal == "clk":
            mapping["clk_gpio"] = m.pin_indices[0]
        elif m.signal == "csn":
            mapping["csn_gpio"] = m.pin_indices[0]
        elif m.signal == "d" and m.width >= 1:
            mapping["d0_gpio"] = m.pin_indices[0]

    if len(mapping) >= 3:
        return mapping
    return None


def find_uart_mapping(pins: PinsLock, name: str = "uart_0") -> dict[str, Any] | None:
    """Find UART interface mapping."""
    uart = pins.get_peripheral_mappings(name)
    if not uart:
        return None

    mapping = {"baud_rate": 115200}
    for m in uart:
        if m.signal == "tx":
            mapping["tx_gpio"] = m.pin_indices[0]
        elif m.signal == "rx":
            mapping["rx_gpio"] = m.pin_indices[0]

    if "tx_gpio" in mapping:
        return mapping
    return None


def find_gpio_mappings(pins: PinsLock) -> list[dict[str, Any]]:
    """Find all GPIO interface mappings."""
    result = []
    seen = set()

    for mapping in pins.all_io_pins():
        if mapping.peripheral.startswith("gpio"):
            if mapping.peripheral not in seen:
                seen.add(mapping.peripheral)
                result.append({
                    "name": mapping.peripheral,
                    "pins": mapping.pin_indices,
                })

    return result


def generate_testbench_config(
    pins_path: Path,
    netlist_path: Path,
    liberty_path: Path,
    firmware_path: Path | None = None,
    firmware_offset: int = 0x100000,
    elf_path: Path | None = None,
    num_cycles: int = 1_000_000,
    reset_cycles: int = 10,
    output_events: Path | None = None,
) -> dict[str, Any]:
    """Generate testbench configuration from pins.lock."""
    pins = load_pins_lock(pins_path)

    # Get reset polarity from iomodel
    reset_active_high = pins.get_bringup_invert("rst_n")

    config = {
        "netlist_path": str(netlist_path),
        "liberty_path": str(liberty_path),
        "clock_gpio": pins.clock_pin,
        "reset_gpio": pins.reset_pin,
        "reset_active_high": reset_active_high,
        "reset_cycles": reset_cycles,
        "num_cycles": num_cycles,
    }

    # Flash configuration
    flash = find_flash_mapping(pins)
    if flash and firmware_path:
        flash["firmware"] = str(firmware_path)
        flash["firmware_offset"] = firmware_offset
        config["flash"] = flash

    # UART configuration
    uart = find_uart_mapping(pins)
    if uart:
        config["uart"] = uart

    # GPIO configuration
    gpios = find_gpio_mappings(pins)
    if gpios:
        config["gpios"] = gpios

    # SRAM initialization
    if elf_path:
        config["sram_init"] = {"elf_path": str(elf_path)}

    # Output events
    if output_events:
        config["output_events"] = str(output_events)

    return config


def generate_rust_testbench(config: dict[str, Any]) -> str:
    """Generate Rust testbench code from configuration."""
    env = Environment(
        loader=PackageLoader("chipflow_harness", "templates"),
    )
    template = env.get_template("timing_main.rs.jinja")
    return template.render(**config)


def generate_json_config(config: dict[str, Any]) -> str:
    """Generate JSON testbench configuration."""
    return json.dumps(config, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate testbench for timing simulation"
    )
    parser.add_argument("pins_lock", type=Path, help="Path to pins.lock")
    parser.add_argument("--netlist", type=Path, required=True, help="Path to netlist")
    parser.add_argument("--liberty", type=Path, required=True, help="Path to Liberty file")
    parser.add_argument("--firmware", type=Path, help="Path to firmware binary")
    parser.add_argument("--firmware-offset", type=int, default=0x100000,
                       help="Firmware offset in flash")
    parser.add_argument("--elf", type=Path, help="Path to ELF for SRAM init")
    parser.add_argument("--cycles", type=int, default=1_000_000, help="Number of cycles")
    parser.add_argument("--reset-cycles", type=int, default=10, help="Reset cycles")
    parser.add_argument("--output-events", type=Path, help="Output events JSON")
    parser.add_argument("--format", choices=["rust", "json"], default="json",
                       help="Output format")
    parser.add_argument("-o", "--output", type=Path, help="Output file")

    args = parser.parse_args()

    config = generate_testbench_config(
        pins_path=args.pins_lock,
        netlist_path=args.netlist,
        liberty_path=args.liberty,
        firmware_path=args.firmware,
        firmware_offset=args.firmware_offset,
        elf_path=args.elf,
        num_cycles=args.cycles,
        reset_cycles=args.reset_cycles,
        output_events=args.output_events,
    )

    if args.format == "rust":
        output = generate_rust_testbench(config)
    else:
        output = generate_json_config(config)

    if args.output:
        args.output.write_text(output)
        print(f"Wrote {args.format} to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
