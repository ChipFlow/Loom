
from pathlib import Path

from amaranth import Module
from amaranth.lib import wiring
from amaranth.lib.wiring import Out, flipped, connect

from amaranth_soc import csr, wishbone
from amaranth_soc.csr.wishbone import WishboneCSRBridge

from chipflow_digital_ip.base import SoCID
from chipflow_digital_ip.memory import QSPIFlash
from amaranth_soc.wishbone.sram import WishboneSRAM
from chipflow_digital_ip.io import GPIOPeripheral, UARTPeripheral
from chipflow_digital_ip.processors import CV32E40P, OBIDebugModule
from chipflow.platform import (
        GPIOSignature, UARTSignature,
        QSPIFlashSignature, JTAGSignature,
        attach_data, SoftwareBuild
        )

__all__ = ["MySoC"]

class MySoC(wiring.Component):
    """Minimal mcu_soc for sky130 openframe (~21 pins).

    Removed vs full mcu_soc: motor PWMs, all SPIs, all I2C, second UART,
    second GPIO bank. Only keeps: flash, JTAG, 1 UART, 1 GPIO bank.
    CSR addresses are kept identical so firmware is compatible.
    """
    def __init__(self):
        interfaces = {
            "flash": Out(QSPIFlashSignature()),
            "cpu_jtag": Out(JTAGSignature())
        }

        self.user_spi_count = 0
        self.i2c_count = 0
        self.uart_count = 1

        self.gpio_banks = 1
        self.gpio_width = 8

        for i in range(self.uart_count):
            interfaces[f"uart_{i}"] = Out(UARTSignature())

        for i in range(self.gpio_banks):
            interfaces[f"gpio_{i}"] = Out(GPIOSignature(pin_count=self.gpio_width))

        super().__init__(interfaces)

        # Memory regions:
        self.mem_spiflash_base = 0x00000000
        self.mem_sram_base     = 0x10000000

        # Debug region
        self.debug_base        = 0xa0000000

        # CSR regions (same addresses as full mcu_soc):
        self.csr_base          = 0xb0000000
        self.csr_spiflash_base = 0xb0000000

        self.csr_gpio_base     = 0xb1000000
        self.csr_uart_base     = 0xb2000000
        self.csr_soc_id_base   = 0xb4000000

        self.periph_offset     = 0x00100000

        self.sram_size  = 0x800 # 2KiB
        self.bios_start = 0x100000 # 1MiB into spiflash to make room for a bitstream

    def elaborate(self, platform):
        m = Module()

        wb_arbiter  = wishbone.Arbiter(addr_width=30, data_width=32, granularity=8)
        wb_decoder  = wishbone.Decoder(addr_width=30, data_width=32, granularity=8)
        csr_decoder = csr.Decoder(addr_width=28, data_width=8)

        m.submodules.wb_arbiter  = wb_arbiter
        m.submodules.wb_decoder  = wb_decoder
        m.submodules.csr_decoder = csr_decoder

        connect(m, wb_arbiter.bus, wb_decoder.bus)

        # CPU

        cpu = CV32E40P(config="default", reset_vector=self.bios_start, dm_haltaddress=self.debug_base+0x800)
        wb_arbiter.add(cpu.ibus)
        wb_arbiter.add(cpu.dbus)

        m.submodules.cpu = cpu

        # Debug
        debug = OBIDebugModule()
        wb_arbiter.add(debug.initiator)
        wb_decoder.add(debug.target, name="debug", addr=self.debug_base)
        m.d.comb += cpu.debug_req.eq(debug.debug_req)

        m.d.comb += [
            debug.jtag_tck.eq(self.cpu_jtag.tck.i),
            debug.jtag_tms.eq(self.cpu_jtag.tms.i),
            debug.jtag_tdi.eq(self.cpu_jtag.tdi.i),
            debug.jtag_trst.eq(self.cpu_jtag.trst.i),
            self.cpu_jtag.tdo.o.eq(debug.jtag_tdo),
        ]

        m.submodules.debug = debug
        # SPI flash

        spiflash = QSPIFlash(addr_width=24, data_width=32)
        wb_decoder.add(spiflash.wb_bus, name="spiflash", addr=self.mem_spiflash_base)
        csr_decoder.add(spiflash.csr_bus, name="spiflash", addr=self.csr_spiflash_base - self.csr_base)
        m.submodules.spiflash = spiflash

        connect(m, flipped(self.flash), spiflash.pins)

        # SRAM

        sram = WishboneSRAM(size=self.sram_size, data_width=32, granularity=8)
        wb_decoder.add(sram.wb_bus, name="sram", addr=self.mem_sram_base)

        m.submodules.sram = sram

        # GPIOs
        for i in range(self.gpio_banks):
            gpio = GPIOPeripheral(pin_count=self.gpio_width)
            base_addr = self.csr_gpio_base + i * self.periph_offset
            csr_decoder.add(gpio.bus, name=f"gpio_{i}", addr=base_addr - self.csr_base)

            pins = getattr(self, f"gpio_{i}")
            connect(m, flipped(pins), gpio.pins)
            setattr(m.submodules, f"gpio_{i}", gpio)

        # UART
        for i in range(self.uart_count):
            uart = UARTPeripheral(init_divisor=int(25e6//115200), addr_width=5)
            base_addr = self.csr_uart_base + i * self.periph_offset
            csr_decoder.add(uart.bus, name=f"uart_{i}", addr=base_addr - self.csr_base)

            pins = getattr(self, f"uart_{i}")
            connect(m, flipped(pins), uart.pins)
            setattr(m.submodules, f"uart_{i}", uart)

        # SoC ID

        soc_id = SoCID(type_id=0xCA7F100F)
        csr_decoder.add(soc_id.bus, name="soc_id", addr=self.csr_soc_id_base - self.csr_base)

        m.submodules.soc_id = soc_id

        # Wishbone-CSR bridge

        wb_to_csr = WishboneCSRBridge(csr_decoder.bus, data_width=32)
        wb_decoder.add(wb_to_csr.wb_bus, name="csr", addr=self.csr_base, sparse=False)

        m.submodules.wb_to_csr = wb_to_csr

        sw = SoftwareBuild(sources=Path('design/software').glob('*.c'),
                           offset=self.bios_start)

        # you need to attach data to both the internal and external interfaces
        attach_data(self.flash, m.submodules.spiflash, sw)

        return m
