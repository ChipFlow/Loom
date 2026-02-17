"""SPI peripheral model for VCD generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class SPIState(Enum):
    """SPI peripheral state."""

    IDLE = "idle"
    SELECTED = "selected"
    TRANSFERRING = "transferring"


@dataclass
class SPIModel:
    """SPI model for VCD generation.

    Models an SPI peripheral (slave) that the DUT communicates with.
    """

    clock_hz: int = 25_000_000
    spi_clock_div: int = 4  # SPI clock = clock_hz / spi_clock_div
    cpol: int = 0  # Clock polarity (0 = idle low, 1 = idle high)
    cpha: int = 0  # Clock phase (0 = sample on first edge, 1 = sample on second edge)
    bit_order: str = "msb"  # "msb" or "lsb" first
    word_size: int = 8  # Bits per transfer

    # State
    _state: SPIState = field(default=SPIState.IDLE, repr=False)
    _tx_data: int = 0  # Data to send to DUT (on CIPO)
    _tx_width: int = 8  # Width of current transfer
    _rx_data: int = 0  # Data received from DUT (on COPI)
    _bit_count: int = 0
    _received_bytes: list[int] = field(default_factory=list, repr=False)
    _pending_tx: list[tuple[int, int]] = field(default_factory=list, repr=False)  # (data, width)

    def set_tx_data(self, data: int, width: int | None = None) -> None:
        """Set data to transmit when selected."""
        self._tx_data = data
        self._tx_width = width if width is not None else self.word_size

    def set_tx_width(self, width: int) -> None:
        """Set transfer width in bits."""
        self._tx_width = width

    def queue_tx(self, data: int, width: int | None = None) -> None:
        """Queue data for transmission."""
        w = width if width is not None else self.word_size
        self._pending_tx.append((data, w))

    def on_select(self) -> None:
        """Called when CS goes low (peripheral selected)."""
        self._state = SPIState.SELECTED
        self._rx_data = 0
        self._bit_count = 0

    def on_deselect(self) -> None:
        """Called when CS goes high (peripheral deselected)."""
        if self._bit_count > 0:
            self._received_bytes.append(self._rx_data)
        self._state = SPIState.IDLE
        self._bit_count = 0

    def on_clock_edge(self, sck: int, copi: int) -> int:
        """Process clock edge, return CIPO value.

        Args:
            sck: Current SCK value
            copi: Current COPI (data from DUT) value

        Returns:
            CIPO value to send to DUT
        """
        if self._state == SPIState.IDLE:
            return 1  # High-Z represented as 1

        sample_edge = (sck == 1) if (self.cpol ^ self.cpha) == 0 else (sck == 0)

        if sample_edge:
            # Sample COPI
            if self.bit_order == "msb":
                self._rx_data = (self._rx_data << 1) | (copi & 1)
            else:
                self._rx_data |= (copi & 1) << self._bit_count

            self._bit_count += 1

            if self._bit_count >= self._tx_width:
                self._received_bytes.append(self._rx_data)
                self._rx_data = 0
                self._bit_count = 0

                # Get next tx data if queued
                if self._pending_tx:
                    self._tx_data, self._tx_width = self._pending_tx.pop(0)

        # Return CIPO bit
        if self.bit_order == "msb":
            bit_idx = self._tx_width - 1 - self._bit_count
        else:
            bit_idx = self._bit_count

        if bit_idx < 0 or bit_idx >= self._tx_width:
            return 1
        return (self._tx_data >> bit_idx) & 1

    def get_received_bytes(self) -> list[int]:
        """Get all bytes received."""
        return list(self._received_bytes)

    def generate_transaction(
        self, data_to_send: int, width: int, start_cycle: int
    ) -> Iterator[tuple[int, str, int]]:
        """Generate SPI transaction waveform.

        Yields (cycle, signal, value) tuples for CSN, SCK, CIPO.
        """
        cycles_per_bit = self.clock_hz // (self.clock_hz // self.spi_clock_div) // 2
        cycle = start_cycle

        # Assert CS (active low)
        yield (cycle, "csn", 0)
        cycle += cycles_per_bit

        # Transfer bits
        for i in range(width):
            if self.bit_order == "msb":
                bit_idx = width - 1 - i
            else:
                bit_idx = i

            bit = (data_to_send >> bit_idx) & 1

            # Rising edge
            yield (cycle, "sck", 1)
            yield (cycle, "cipo", bit)
            cycle += cycles_per_bit

            # Falling edge
            yield (cycle, "sck", 0)
            cycle += cycles_per_bit

        # Deassert CS
        yield (cycle, "csn", 1)

    def reset(self) -> None:
        """Reset SPI state."""
        self._state = SPIState.IDLE
        self._tx_data = 0
        self._tx_width = self.word_size
        self._rx_data = 0
        self._bit_count = 0
        self._received_bytes.clear()
        self._pending_tx.clear()
