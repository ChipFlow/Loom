/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef SPIFLASH_MODEL_H
#define SPIFLASH_MODEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpiFlashModel SpiFlashModel;

// Create a new flash model with given size in bytes
SpiFlashModel* spiflash_new(size_t size_bytes);

// Free the flash model
void spiflash_free(SpiFlashModel* flash);

// Load firmware into flash at given offset
// Returns number of bytes loaded, or -1 on error
int spiflash_load(SpiFlashModel* flash, const uint8_t* data, size_t len, size_t offset);

// Step the simulation
// clk: SPI clock state
// csn: chip select (active low)
// d_o: 4-bit data from controller to flash (MOSI on bit 0 in single mode)
// Returns: 4-bit data from flash to controller (MISO on bit 1 in single mode)
uint8_t spiflash_step(SpiFlashModel* flash, int clk, int csn, uint8_t d_o);

// Enable/disable verbose debug output
void spiflash_set_verbose(SpiFlashModel* flash, int verbose);

// Get the current command being processed
uint8_t spiflash_get_command(SpiFlashModel* flash);

// Get the byte count in current transaction
uint32_t spiflash_get_byte_count(SpiFlashModel* flash);

// Debug: get step count
uint32_t spiflash_get_step_count(SpiFlashModel* flash);

// Debug: get posedge count
uint32_t spiflash_get_posedge_count(SpiFlashModel* flash);

// Debug: get negedge count
uint32_t spiflash_get_negedge_count(SpiFlashModel* flash);

#ifdef __cplusplus
}
#endif

#endif // SPIFLASH_MODEL_H
