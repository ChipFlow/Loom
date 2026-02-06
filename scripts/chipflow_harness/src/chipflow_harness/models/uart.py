"""UART peripheral model for VCD generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class UARTModel:
    """UART bit-level model for VCD generation.

    Generates timing for UART transmit (to DUT RX) and
    decodes UART receive (from DUT TX).
    """

    baud_rate: int = 115200
    clock_hz: int = 25_000_000
    data_bits: int = 8
    stop_bits: int = 1
    parity: str | None = None  # None, "odd", or "even"

    # Internal state
    _tx_queue: list[int] = field(default_factory=list, repr=False)
    _rx_samples: list[int] = field(default_factory=list, repr=False)
    _rx_state: str = field(default="idle", repr=False)
    _rx_bit_count: int = field(default=0, repr=False)
    _rx_sample_count: int = field(default=0, repr=False)
    _rx_current_byte: int = field(default=0, repr=False)
    _received_bytes: list[int] = field(default_factory=list, repr=False)

    @property
    def cycles_per_bit(self) -> int:
        """Number of clock cycles per UART bit."""
        return self.clock_hz // self.baud_rate

    def queue_transmit(self, byte: int) -> None:
        """Queue a byte for transmission to DUT."""
        assert 0 <= byte <= 255
        self._tx_queue.append(byte)

    def get_tx_waveform(self, start_cycle: int) -> Iterator[tuple[int, int]]:
        """Generate TX waveform for all queued bytes.

        Yields (cycle, tx_value) pairs. TX is held high (idle) initially,
        then generates start bit (0), data bits (LSB first), and stop bit (1).
        """
        cycle = start_cycle

        for byte in self._tx_queue:
            # Start bit (low)
            yield (cycle, 0)
            cycle += self.cycles_per_bit

            # Data bits (LSB first)
            for i in range(self.data_bits):
                bit = (byte >> i) & 1
                yield (cycle, bit)
                cycle += self.cycles_per_bit

            # Parity bit if enabled
            if self.parity is not None:
                parity_bit = bin(byte).count("1") % 2
                if self.parity == "even":
                    yield (cycle, parity_bit)
                else:  # odd
                    yield (cycle, 1 - parity_bit)
                cycle += self.cycles_per_bit

            # Stop bit(s) (high)
            for _ in range(self.stop_bits):
                yield (cycle, 1)
                cycle += self.cycles_per_bit

        self._tx_queue.clear()

    def transmit_byte(self, byte: int) -> Iterator[tuple[int, int]]:
        """Yield (cycle_offset, tx_value) for transmitting a single byte.

        Offset is relative to start of transmission.
        """
        offset = 0

        # Start bit (low)
        yield (offset, 0)
        offset += self.cycles_per_bit

        # Data bits (LSB first)
        for i in range(self.data_bits):
            bit = (byte >> i) & 1
            yield (offset, bit)
            offset += self.cycles_per_bit

        # Parity bit if enabled
        if self.parity is not None:
            parity_bit = bin(byte).count("1") % 2
            if self.parity == "even":
                yield (offset, parity_bit)
            else:
                yield (offset, 1 - parity_bit)
            offset += self.cycles_per_bit

        # Stop bit(s) (high)
        for _ in range(self.stop_bits):
            yield (offset, 1)
            offset += self.cycles_per_bit

    def sample_rx(self, value: int) -> int | None:
        """Sample RX line and decode bytes.

        Call this once per clock cycle with the RX pin value.
        Returns decoded byte when complete, None otherwise.
        """
        self._rx_sample_count += 1

        if self._rx_state == "idle":
            if value == 0:  # Start bit detected
                self._rx_state = "start"
                self._rx_sample_count = 1
                self._rx_current_byte = 0
                self._rx_bit_count = 0
            return None

        elif self._rx_state == "start":
            # Sample at middle of bit
            if self._rx_sample_count >= self.cycles_per_bit // 2:
                if value == 0:  # Valid start bit
                    self._rx_state = "data"
                    self._rx_sample_count = 0
                else:  # False start
                    self._rx_state = "idle"
            return None

        elif self._rx_state == "data":
            if self._rx_sample_count >= self.cycles_per_bit:
                self._rx_sample_count = 0
                self._rx_current_byte |= (value << self._rx_bit_count)
                self._rx_bit_count += 1

                if self._rx_bit_count >= self.data_bits:
                    if self.parity is not None:
                        self._rx_state = "parity"
                    else:
                        self._rx_state = "stop"
            return None

        elif self._rx_state == "parity":
            if self._rx_sample_count >= self.cycles_per_bit:
                self._rx_sample_count = 0
                # Could verify parity here
                self._rx_state = "stop"
            return None

        elif self._rx_state == "stop":
            if self._rx_sample_count >= self.cycles_per_bit:
                self._rx_sample_count = 0
                byte = self._rx_current_byte
                self._received_bytes.append(byte)
                self._rx_state = "idle"
                return byte
            return None

        return None

    def get_received_bytes(self) -> list[int]:
        """Get all bytes received so far."""
        return list(self._received_bytes)

    def clear_received(self) -> None:
        """Clear received bytes buffer."""
        self._received_bytes.clear()

    def reset(self) -> None:
        """Reset the UART model state."""
        self._tx_queue.clear()
        self._rx_samples.clear()
        self._rx_state = "idle"
        self._rx_bit_count = 0
        self._rx_sample_count = 0
        self._rx_current_byte = 0
        self._received_bytes.clear()
