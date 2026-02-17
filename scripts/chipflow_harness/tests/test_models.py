"""Tests for peripheral models."""

from chipflow_harness.models.uart import UARTModel
from chipflow_harness.models.gpio import GPIOModel
from chipflow_harness.models.spi import SPIModel
from chipflow_harness.models.i2c import I2CModel


def test_uart_cycles_per_bit():
    """Test UART timing calculation."""
    uart = UARTModel(baud_rate=115200, clock_hz=25_000_000)
    # 25MHz / 115200 = 217 cycles per bit
    assert uart.cycles_per_bit == 217


def test_uart_transmit_byte():
    """Test UART byte transmission waveform."""
    uart = UARTModel(baud_rate=115200, clock_hz=25_000_000)

    # Transmit 0x55 = 0b01010101
    waveform = list(uart.transmit_byte(0x55))

    # Should have: start bit + 8 data bits + stop bit = 10 transitions
    assert len(waveform) == 10

    # Start bit is 0
    assert waveform[0] == (0, 0)

    # Data bits (LSB first): 1,0,1,0,1,0,1,0
    for i, expected in enumerate([1, 0, 1, 0, 1, 0, 1, 0]):
        offset, value = waveform[1 + i]
        assert value == expected, f"Bit {i}: expected {expected}, got {value}"

    # Stop bit is 1
    assert waveform[9][1] == 1


def test_gpio_set_value_int():
    """Test GPIO set value with integer."""
    gpio = GPIOModel(width=8)
    result = gpio.set_value(0xAB)
    assert result == 0xAB
    assert gpio.get_value() == 0xAB


def test_gpio_set_value_string():
    """Test GPIO set value with binary string."""
    gpio = GPIOModel(width=8)
    result = gpio.set_value("10101010")
    assert result == 0xAA
    assert gpio.get_value() == 0xAA


def test_gpio_set_value_with_z():
    """Test GPIO set value with Z (high-impedance) bits."""
    gpio = GPIOModel(width=8)
    gpio.set_value(0xFF)  # Start with all 1s
    result = gpio.set_value("1010ZZZZ")
    # Z bits become 0 in numeric, but existing value not changed for those bits
    assert result == 0xA0  # Only top 4 bits set


def test_gpio_check_change():
    """Test GPIO pattern matching."""
    gpio = GPIOModel(width=8)
    gpio.set_value(0xA5)

    # Exact match
    assert gpio.check_change("10100101", 0xA5)

    # With don't care bits
    assert gpio.check_change("1010ZZZZ", 0xA5)
    assert gpio.check_change("ZZZZ0101", 0xA5)

    # Mismatch
    assert not gpio.check_change("10100100", 0xA5)


def test_spi_set_data():
    """Test SPI data setup."""
    spi = SPIModel()
    spi.set_tx_data(0x55)
    spi.set_tx_width(8)
    # Just verify no exceptions


def test_i2c_cycles_per_bit():
    """Test I2C timing calculation."""
    i2c = I2CModel(clock_hz=25_000_000, i2c_clock_hz=100_000)
    # 25MHz / 100kHz = 250 cycles per bit
    assert i2c.cycles_per_bit == 250


def test_i2c_model_reset():
    """Test I2C model reset."""
    i2c = I2CModel()
    i2c.on_start()
    i2c.reset()
    assert len(i2c.get_events()) == 0
    assert len(i2c.get_received_bytes()) == 0
