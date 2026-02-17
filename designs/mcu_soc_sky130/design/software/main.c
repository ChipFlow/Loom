#include <stdint.h>
#include "generated/soc.h"

void main() {
    uart_init(UART_0, 25000000/115200);

    puts("🐱: nyaa~!\r\n");

    puts("SoC type: ");
    puthex(SOC_ID->type);
    puts("\r\n");

    // SPI Flash config
    puts("Flash ID: ");
    puthex(spiflash_read_id(SPIFLASH));
    puts("\n");
    spiflash_set_qspi_flag(SPIFLASH);
    spiflash_set_quad_mode(SPIFLASH);
    puts("Quad mode\n");

    GPIO_0->mode = GPIO_PIN4_PUSH_PULL | GPIO_PIN5_PUSH_PULL \
                 | GPIO_PIN6_PUSH_PULL | GPIO_PIN7_PUSH_PULL;
    GPIO_0->output = 0x50;
    GPIO_0->setclr = GPIO_PIN4_CLEAR | GPIO_PIN5_SET \
                   | GPIO_PIN6_CLEAR | GPIO_PIN7_SET;
    GPIO_0->mode = GPIO_PIN4_INPUT_ONLY | GPIO_PIN5_INPUT_ONLY \
		 | GPIO_PIN6_INPUT_ONLY | GPIO_PIN7_INPUT_ONLY;

    puts("GPIO: ");
    puthex(GPIO_0->input);
    puts(" ");
    puthex(GPIO_0->input);
    puts("\n");

    while (1) {
    };
}
