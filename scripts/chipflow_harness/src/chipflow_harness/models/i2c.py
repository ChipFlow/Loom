"""I2C peripheral model for VCD generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class I2CState(Enum):
    """I2C bus state."""

    IDLE = "idle"
    START = "start"
    ADDRESS = "address"
    ACK_ADDR = "ack_addr"
    DATA = "data"
    ACK_DATA = "ack_data"
    STOP = "stop"


@dataclass
class I2CModel:
    """I2C model for VCD generation.

    Models an I2C peripheral (slave) that the DUT communicates with.
    """

    clock_hz: int = 25_000_000
    i2c_clock_hz: int = 100_000  # Standard mode
    address: int = 0x50  # 7-bit address

    # State
    _state: I2CState = field(default=I2CState.IDLE, repr=False)
    _bit_count: int = 0
    _current_byte: int = 0
    _is_read: bool = False
    _tx_data: int = 0  # Data to send on read
    _received_bytes: list[int] = field(default_factory=list, repr=False)
    _events: list[tuple[str, int | str]] = field(default_factory=list, repr=False)
    _prev_scl: int = 1
    _prev_sda: int = 1
    _ack_pending: bool = False

    @property
    def cycles_per_bit(self) -> int:
        """Clock cycles per I2C bit."""
        return self.clock_hz // self.i2c_clock_hz

    def set_tx_data(self, data: int) -> None:
        """Set data to transmit on read."""
        self._tx_data = data

    def on_start(self) -> None:
        """Called when START condition detected."""
        self._state = I2CState.ADDRESS
        self._bit_count = 0
        self._current_byte = 0
        self._events.append(("start", ""))

    def on_stop(self) -> None:
        """Called when STOP condition detected."""
        self._state = I2CState.IDLE
        self._events.append(("stop", ""))

    def on_ack(self) -> None:
        """Send ACK on next clock."""
        self._ack_pending = True
        self._events.append(("ack", ""))

    def sample(self, scl: int, sda: int) -> int:
        """Sample SCL and SDA, return SDA output (for peripheral to drive).

        Returns 1 for high-Z (release), 0 to pull low.
        """
        # Detect START: SDA falls while SCL high
        if self._prev_scl == 1 and scl == 1 and self._prev_sda == 1 and sda == 0:
            self.on_start()
            self._prev_scl = scl
            self._prev_sda = sda
            return 1

        # Detect STOP: SDA rises while SCL high
        if self._prev_scl == 1 and scl == 1 and self._prev_sda == 0 and sda == 1:
            self.on_stop()
            self._prev_scl = scl
            self._prev_sda = sda
            return 1

        # Rising edge of SCL - sample data
        if self._prev_scl == 0 and scl == 1:
            result = self._on_scl_rise(sda)
            self._prev_scl = scl
            self._prev_sda = sda
            return result

        # Falling edge of SCL - change data
        if self._prev_scl == 1 and scl == 0:
            result = self._on_scl_fall()
            self._prev_scl = scl
            self._prev_sda = sda
            return result

        self._prev_scl = scl
        self._prev_sda = sda
        return 1 if not self._ack_pending else 0

    def _on_scl_rise(self, sda: int) -> int:
        """Handle SCL rising edge."""
        if self._state == I2CState.ADDRESS:
            self._current_byte = (self._current_byte << 1) | sda
            self._bit_count += 1

            if self._bit_count == 8:
                # Full address byte received (address is bits 7:1, R/W is bit 0)
                self._is_read = bool(self._current_byte & 1)
                self._events.append(("address", self._current_byte))
                self._state = I2CState.ACK_ADDR
                self._bit_count = 0
                self._current_byte = 0
                return 1

        elif self._state == I2CState.DATA:
            if not self._is_read:
                # Receiving data from master
                self._current_byte = (self._current_byte << 1) | sda
                self._bit_count += 1

                if self._bit_count == 8:
                    self._received_bytes.append(self._current_byte)
                    self._events.append(("data", self._current_byte))
                    self._state = I2CState.ACK_DATA
                    self._bit_count = 0
                    self._current_byte = 0

        return 1

    def _on_scl_fall(self) -> int:
        """Handle SCL falling edge - time to change SDA."""
        if self._ack_pending:
            self._ack_pending = False
            if self._state == I2CState.ACK_ADDR:
                self._state = I2CState.DATA
            elif self._state == I2CState.ACK_DATA:
                self._state = I2CState.DATA
            return 0  # Pull SDA low for ACK

        if self._state == I2CState.DATA and self._is_read:
            # Sending data to master
            bit = (self._tx_data >> (7 - self._bit_count)) & 1
            self._bit_count += 1

            if self._bit_count == 8:
                self._bit_count = 0
                self._state = I2CState.ACK_DATA  # Wait for master ACK
            return bit

        return 1  # Release SDA

    def get_events(self) -> list[tuple[str, int | str]]:
        """Get recorded I2C events."""
        return list(self._events)

    def get_received_bytes(self) -> list[int]:
        """Get bytes received from master."""
        return list(self._received_bytes)

    def generate_start(self, cycle: int) -> Iterator[tuple[int, str, int]]:
        """Generate START condition waveform."""
        half_bit = self.cycles_per_bit // 2

        # SCL and SDA both high initially
        yield (cycle, "scl", 1)
        yield (cycle, "sda", 1)
        cycle += half_bit

        # SDA falls while SCL high (START)
        yield (cycle, "sda", 0)
        cycle += half_bit

        # SCL falls
        yield (cycle, "scl", 0)

    def generate_stop(self, cycle: int) -> Iterator[tuple[int, str, int]]:
        """Generate STOP condition waveform."""
        half_bit = self.cycles_per_bit // 2

        # SDA low, SCL low
        yield (cycle, "sda", 0)
        yield (cycle, "scl", 0)
        cycle += half_bit

        # SCL rises
        yield (cycle, "scl", 1)
        cycle += half_bit

        # SDA rises while SCL high (STOP)
        yield (cycle, "sda", 1)

    def generate_byte(
        self, byte: int, cycle: int, need_ack: bool = True
    ) -> Iterator[tuple[int, str, int]]:
        """Generate I2C byte transmission waveform."""
        half_bit = self.cycles_per_bit // 2

        for i in range(8):
            bit = (byte >> (7 - i)) & 1

            # SDA changes while SCL low
            yield (cycle, "sda", bit)
            cycle += half_bit

            # SCL rises
            yield (cycle, "scl", 1)
            cycle += half_bit

            # SCL falls
            yield (cycle, "scl", 0)
            cycle += half_bit

        if need_ack:
            # Release SDA for ACK
            yield (cycle, "sda", 1)
            cycle += half_bit
            yield (cycle, "scl", 1)
            cycle += half_bit
            yield (cycle, "scl", 0)

    def reset(self) -> None:
        """Reset I2C state."""
        self._state = I2CState.IDLE
        self._bit_count = 0
        self._current_byte = 0
        self._is_read = False
        self._tx_data = 0
        self._received_bytes.clear()
        self._events.clear()
        self._prev_scl = 1
        self._prev_sda = 1
        self._ack_pending = False
