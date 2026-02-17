"""GPIO peripheral model for VCD generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GPIOModel:
    """GPIO model for VCD generation.

    Handles set/change events for GPIO pins.
    """

    width: int = 8
    _current_value: int = 0
    _output_enable: int = 0  # Bitmask of which pins are outputs
    _pending_changes: list[tuple[int, int]] = field(default_factory=list, repr=False)

    def set_value(self, value: int | str) -> int:
        """Set GPIO output value.

        Value can be an integer or a string like "10101010" or "1010ZZZZ".
        Returns the numeric value (Z bits become 0 in numeric representation).
        """
        if isinstance(value, str):
            # Parse string value - may contain Z for high-impedance
            numeric = 0
            for i, char in enumerate(reversed(value)):
                if char == "1":
                    numeric |= (1 << i)
                elif char == "Z":
                    pass  # Leave as 0 in numeric, but track for VCD
            self._current_value = numeric
        else:
            self._current_value = value & ((1 << self.width) - 1)

        return self._current_value

    def get_value(self) -> int:
        """Get current GPIO value."""
        return self._current_value

    def set_output_enable(self, mask: int) -> None:
        """Set which GPIO pins are outputs (1 = output, 0 = input)."""
        self._output_enable = mask & ((1 << self.width) - 1)

    def get_output_enable(self) -> int:
        """Get output enable mask."""
        return self._output_enable

    def format_value(self, value: int | str | None = None) -> str:
        """Format value as binary string for VCD."""
        if value is None:
            value = self._current_value

        if isinstance(value, str):
            # Pad or truncate to width
            if len(value) < self.width:
                value = "0" * (self.width - len(value)) + value
            elif len(value) > self.width:
                value = value[-self.width:]
            return value
        else:
            return format(value & ((1 << self.width) - 1), f"0{self.width}b")

    def check_change(self, expected: str, actual: int) -> bool:
        """Check if GPIO matches expected pattern.

        Expected can contain 'Z' for don't-care bits.
        """
        if len(expected) != self.width:
            raise ValueError(f"Expected pattern length {len(expected)} != width {self.width}")

        for i, char in enumerate(reversed(expected)):
            if char == "Z":
                continue  # Don't care
            bit_value = (actual >> i) & 1
            expected_bit = 1 if char == "1" else 0
            if bit_value != expected_bit:
                return False
        return True

    def reset(self) -> None:
        """Reset GPIO state."""
        self._current_value = 0
        self._output_enable = 0
        self._pending_changes.clear()
