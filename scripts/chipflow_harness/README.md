# ChipFlow Test Harness

VCD stimulus generator for ChipFlow post-synthesis simulation using GEM.

## Overview

This tool converts ChipFlow's `input.json` test format into VCD waveforms suitable
for gate-level timing simulation with GEM's `timing_sim_cpu`.

## Installation

```bash
cd scripts/chipflow_harness
uv sync
```

## Usage

### Standard ChipFlow VCD Generation

Generate VCD from input.json using ChipFlow's peripheral-based pin structure:

```bash
uv run chipflow-vcd-gen \
  --input-json /path/to/input.json \
  --pins-lock /path/to/pins.lock \
  --output test.vcd
```

### Caravel OpenFrame VCD Generation

For netlists using the Caravel OpenFrame wrapper (gpio_in[43:0] interface):

```bash
uv run chipflow-caravel-vcd-gen \
  --input-json /path/to/input.json \
  --pins-lock /path/to/pins.lock \
  --output test.vcd
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--clock-period-ps` | 40000000 | Clock period in picoseconds (40000000 = 25MHz) |
| `--baud-rate` | 115200 | UART baud rate |
| `--module-name` | testbench | VCD module name |
| `--max-cycles` | 1000000 | Maximum simulation cycles |
| `-v, --verbose` | false | Enable verbose output |

### Event Comparison

Compare simulation output events against a reference:

```bash
uv run chipflow-compare-events \
  --reference events_reference.json \
  --actual events.json
```

## Supported Peripherals

The harness supports the following ChipFlow peripherals:

- **UART**: TX/RX at configurable baud rate (default 115200)
- **GPIO**: Multi-bit set/change with support for high-Z patterns
- **SPI**: Controller mode with configurable width
- **I2C**: Standard mode (100kHz) with start/stop/ack support

## input.json Format

ChipFlow's test format uses action/wait commands:

```json
{
  "commands": [
    { "type": "wait", "peripheral": "gpio_1", "event": "change", "payload": "1010ZZZZ" },
    { "type": "action", "peripheral": "gpio_1", "event": "set", "payload": "00111100" },
    { "type": "wait", "peripheral": "uart_0", "event": "tx", "payload": 51 },
    { "type": "action", "peripheral": "uart_1", "event": "tx", "payload": 35 }
  ]
}
```

### Action Events

| Peripheral | Event | Payload | Description |
|------------|-------|---------|-------------|
| gpio_* | set | binary string or int | Set GPIO output value |
| uart_* | tx | int (byte) | Transmit byte to DUT RX |
| spi_* | set_data | int | Set data for SPI response |
| spi_* | set_width | int | Set SPI transfer width |
| i2c_* | ack | - | Send ACK on I2C bus |
| i2c_* | set_data | int (byte) | Set data for I2C read |

### Wait Events

| Peripheral | Event | Payload | Description |
|------------|-------|---------|-------------|
| gpio_* | change | pattern (with Z) | Wait for GPIO to match pattern |
| uart_* | tx | int (byte) | Wait for byte on DUT TX |
| spi_* | select | - | Wait for CS assertion |
| spi_* | deselect | - | Wait for CS deassertion |
| spi_* | data | int | Wait for SPI data transfer |
| i2c_* | start | - | Wait for I2C start condition |
| i2c_* | stop | - | Wait for I2C stop condition |
| i2c_* | address | int | Wait for I2C address byte |

## Running Tests

```bash
uv run --extra dev pytest tests/ -v
```

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ input.json  │───►│   chipflow  │───►│  test.vcd   │
│ pins.lock   │    │  vcd_gen.py │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ netlist.v   │───►│timing_sim_  │───►│ events.json │
│ liberty.lib │    │    cpu      │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Notes

### Clock Connection

The VCD generator assumes the clock is connected via the pin specified in
`pins.lock` (typically pin 2 for ChipFlow designs). For Caravel OpenFrame
designs, this maps to `gpio_in[2]`.

**Important**: Some post-P&R netlists may have dangling clock nets if the
clock pad cell was not included in the extraction. In such cases, you may
need to modify the netlist or use a different extraction method.

### Timing Considerations

- Clock period defaults to 40ns (25MHz) matching typical ChipFlow targets
- UART timing is calculated based on the clock frequency and baud rate
- Reset is held for 10 cycles before deassertion
