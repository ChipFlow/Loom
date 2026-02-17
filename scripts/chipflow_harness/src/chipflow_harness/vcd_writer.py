"""Simple VCD file writer for test stimulus generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


@dataclass
class VCDSignal:
    """A signal registered in the VCD file."""

    name: str
    width: int
    identifier: str
    scope: str = ""


class VCDWriter:
    """Writes VCD (Value Change Dump) files."""

    def __init__(
        self,
        output: TextIO | Path,
        timescale: str = "1ps",
        module_name: str = "testbench",
    ):
        self._output: TextIO
        self._owns_file = False

        if isinstance(output, Path):
            self._output = open(output, "w")
            self._owns_file = True
        else:
            self._output = output

        self._timescale = timescale
        self._module_name = module_name
        self._signals: dict[str, VCDSignal] = {}
        self._next_id = ord("!")
        self._header_written = False
        self._last_time: int | None = None
        self._current_values: dict[str, int | str] = {}

    def __enter__(self) -> VCDWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the VCD file."""
        if self._owns_file and self._output:
            self._output.close()

    def _get_next_identifier(self) -> str:
        """Generate next VCD signal identifier."""
        if self._next_id > ord("~"):
            # Use two-character identifiers
            first = (self._next_id - ord("~") - 1) // 94 + ord("!")
            second = (self._next_id - ord("~") - 1) % 94 + ord("!")
            self._next_id += 1
            return chr(first) + chr(second)
        result = chr(self._next_id)
        self._next_id += 1
        return result

    def register_signal(self, name: str, width: int = 1, scope: str = "") -> str:
        """Register a signal for the VCD file. Returns the identifier."""
        if self._header_written:
            raise RuntimeError("Cannot register signals after header is written")

        identifier = self._get_next_identifier()
        self._signals[name] = VCDSignal(
            name=name,
            width=width,
            identifier=identifier,
            scope=scope or self._module_name,
        )
        return identifier

    def _write_header(self) -> None:
        """Write VCD header."""
        if self._header_written:
            return

        out = self._output
        now = datetime.now()

        out.write("$date\n")
        out.write(f"\t{now.strftime('%a %b %d %H:%M:%S %Y')}\n")
        out.write("$end\n")

        out.write("$version\n")
        out.write("\tChipFlow Harness VCD Generator\n")
        out.write("$end\n")

        out.write("$timescale\n")
        out.write(f"\t{self._timescale}\n")
        out.write("$end\n")

        # Group signals by scope
        scopes: dict[str, list[VCDSignal]] = {}
        for sig in self._signals.values():
            scopes.setdefault(sig.scope, []).append(sig)

        # Write signal declarations
        for scope_name, signals in scopes.items():
            out.write(f"$scope module {scope_name} $end\n")
            for sig in signals:
                if sig.width == 1:
                    out.write(f"$var wire 1 {sig.identifier} {sig.name} $end\n")
                else:
                    out.write(f"$var wire {sig.width} {sig.identifier} {sig.name} $end\n")
            out.write("$upscope $end\n")

        out.write("$enddefinitions $end\n")
        self._header_written = True

    def set_time(self, time_ps: int) -> None:
        """Set current simulation time in picoseconds."""
        if not self._header_written:
            self._write_header()

        if self._last_time is None:
            # First time - write initial values
            self._output.write(f"#{time_ps}\n")
            self._output.write("$dumpvars\n")
            for sig in self._signals.values():
                if sig.width == 1:
                    self._output.write(f"x{sig.identifier}\n")
                else:
                    self._output.write(f"bx {sig.identifier}\n")
            self._output.write("$end\n")
        elif time_ps > self._last_time:
            self._output.write(f"#{time_ps}\n")
        elif time_ps < self._last_time:
            raise ValueError(f"Time cannot go backwards: {time_ps} < {self._last_time}")

        self._last_time = time_ps

    def set_value(self, name: str, value: int | str, time_ps: int | None = None) -> None:
        """Set a signal value. If time_ps is provided, sets time first."""
        if time_ps is not None:
            self.set_time(time_ps)

        if not self._header_written:
            raise RuntimeError("Must call set_time before set_value")

        sig = self._signals.get(name)
        if sig is None:
            raise KeyError(f"Unknown signal: {name}")

        # Skip if value hasn't changed
        old_value = self._current_values.get(name)
        if old_value == value:
            return

        self._current_values[name] = value

        if sig.width == 1:
            if isinstance(value, str):
                self._output.write(f"{value}{sig.identifier}\n")
            else:
                self._output.write(f"{value & 1}{sig.identifier}\n")
        else:
            if isinstance(value, str):
                self._output.write(f"b{value} {sig.identifier}\n")
            else:
                binary = format(value, f"0{sig.width}b")
                self._output.write(f"b{binary} {sig.identifier}\n")

    def format_binary(self, value: int, width: int) -> str:
        """Format an integer as a binary string of given width."""
        return format(value & ((1 << width) - 1), f"0{width}b")


def create_vcd(path: Path, timescale: str = "1ps", module_name: str = "testbench") -> VCDWriter:
    """Create a new VCD file for writing."""
    return VCDWriter(path, timescale, module_name)
