/* SPDX-License-Identifier: BSD-2-Clause */
// SPI Flash model for timing simulation
// Matches chipflow-lib/chipflow/models/spiflash.cc behavior exactly

#include "spiflash_model.h"
#include <vector>
#include <cstring>
#include <cstdio>
#include <array>

struct SpiFlashModel {
    // State pattern matching CXXRTL: s = current state, sn = next state
    struct State {
        int bit_count = 0;
        int byte_count = 0;
        unsigned data_width = 1;
        uint32_t addr = 0;
        uint8_t curr_byte = 0;
        uint8_t command = 0;
        uint8_t out_buffer = 0;
    } s, sn;

    // Signals (matching CXXRTL p_* naming)
    bool p_clk_o = false;      // Clock output from controller
    bool p_csn_o = true;       // Chip select (active low) output from controller
    uint8_t p_d_o = 0;         // Data from controller (4 bits)
    uint8_t p_d_i = 0;         // Data to controller (4 bits)

    // Previous signal values for edge detection
    bool prev_clk_o = false;
    bool prev_csn_o = true;

    // Flash data
    std::vector<uint8_t> data;

    // Debug counters
    uint32_t posedge_count = 0;
    uint32_t negedge_count = 0;
    uint32_t step_count = 0;
    bool verbose = false;

    SpiFlashModel(size_t size) : data(size, 0xFF) {}

    void process_byte() {
        sn.out_buffer = 0;
        if (sn.byte_count == 0) {
            sn.addr = 0;
            sn.data_width = 1;
            sn.command = sn.curr_byte;
            if (sn.command == 0xab) {
                // power up
            } else if (sn.command == 0x03 || sn.command == 0x9f || sn.command == 0xff
                || sn.command == 0x35 || sn.command == 0x31 || sn.command == 0x50
                || sn.command == 0x05 || sn.command == 0x01 || sn.command == 0x06) {
                // nothing to do
            } else if (sn.command == 0xeb) {
                sn.data_width = 4;
            } else {
                fprintf(stderr, "flash: unknown command %02x\n", sn.command);
            }
        } else {
            if (sn.command == 0x03) {
                // Single read
                if (sn.byte_count <= 3) {
                    sn.addr |= (uint32_t(sn.curr_byte) << ((3 - sn.byte_count) * 8));
                }
                if (sn.byte_count >= 3) {
                    size_t idx = sn.addr & 0x00FFFFFF;
                    if (idx < data.size()) {
                        sn.out_buffer = data[idx];
                    } else {
                        sn.out_buffer = 0xFF;
                    }
                    sn.addr = (sn.addr + 1) & 0x00FFFFFF;
                }
            } else if (sn.command == 0xeb) {
                // Quad read
                if (sn.byte_count <= 3) {
                    sn.addr |= (uint32_t(sn.curr_byte) << ((3 - sn.byte_count) * 8));
                }
                if (sn.byte_count >= 6) { // 1 mode, 2 dummy clocks
                    size_t idx = sn.addr & 0x00FFFFFF;
                    if (idx < data.size()) {
                        sn.out_buffer = data[idx];
                    } else {
                        sn.out_buffer = 0xFF;
                    }
                    sn.addr = (sn.addr + 1) & 0x00FFFFFF;
                }
            }
        }
        if (sn.command == 0x9f) {
            // Read ID
            static const std::array<uint8_t, 4> flash_id{0xCA, 0x7C, 0xA7, 0xFF};
            sn.out_buffer = flash_id[sn.byte_count % 4];
        }
    }

    // Edge detection helpers (matching CXXRTL)
    bool posedge_clk() const { return p_clk_o && !prev_clk_o; }
    bool negedge_clk() const { return !p_clk_o && prev_clk_o; }
    bool posedge_csn() const { return p_csn_o && !prev_csn_o; }

    // Evaluate combinational logic (like CXXRTL eval())
    void eval() {
        sn = s;  // Start with current state

        // Count edges for debugging
        if (posedge_clk()) posedge_count++;
        if (negedge_clk()) negedge_count++;

        if (posedge_csn()) {
            // Rising edge of CSN - deselect, reset state
            if (verbose) {
                fprintf(stderr, "flash: CSN HIGH (deselect), cmd=0x%02x, bytes=%d, addr=0x%06x, posedges=%u\n",
                        s.command, s.byte_count, s.addr, posedge_count);
            }
            sn.bit_count = 0;
            sn.byte_count = 0;
            sn.data_width = 1;
        } else if (posedge_clk() && !p_csn_o) {
            // Rising clock edge while selected - sample input
            // Sample current data - the Rust caller is responsible for passing
            // the appropriately delayed data value to model setup time.
            uint8_t sample_d = p_d_o;
            if (sn.data_width == 4) {
                sn.curr_byte = (sn.curr_byte << 4U) | (sample_d & 0xF);
            } else {
                sn.curr_byte = (sn.curr_byte << 1U) | (sample_d & 0x1);
            }
            sn.out_buffer = sn.out_buffer << sn.data_width;
            sn.bit_count += sn.data_width;

            // Debug: log bit sampling for first 50 posedges
            if (sn.bit_count >= 8) {
                process_byte();
                ++sn.byte_count;
                sn.bit_count = 0;
                // Log every byte completion for first 20 bytes, then every 100th
                if (verbose && (sn.byte_count <= 20 || sn.byte_count % 100 == 0)) {
                    fprintf(stderr, "  -> byte complete #%d: cmd=0x%02x, addr=0x%06x, out_buf=0x%02x\n",
                            sn.byte_count, sn.command, sn.addr, sn.out_buffer);
                }
            }
        } else if (negedge_clk() && !p_csn_o) {
            // Falling clock edge while selected - drive output
            if (sn.data_width == 4) {
                p_d_i = (sn.out_buffer >> 4U) & 0xFU;
            } else {
                // MISO on d[1] for single SPI mode
                p_d_i = ((sn.out_buffer >> 7U) & 0x1U) << 1U;
            }
        }
    }

    // Commit state changes (like CXXRTL commit())
    void commit() {
        s = sn;
        prev_clk_o = p_clk_o;
        prev_csn_o = p_csn_o;
    }

    // Step the simulation: set inputs, eval, commit, return output
    // This matches CXXRTL's agent.step() which calls eval() then commit()
    uint8_t step(bool clk, bool csn, uint8_t d_o) {
        step_count++;

        // Set input signals
        p_clk_o = clk;
        p_csn_o = csn;
        p_d_o = d_o;

        // Evaluate (computes next state and output)
        eval();

        // Commit (atomically update state)
        commit();

        // Return output value
        return p_d_i;
    }
};

extern "C" {

SpiFlashModel* spiflash_new(size_t size_bytes) {
    return new SpiFlashModel(size_bytes);
}

void spiflash_free(SpiFlashModel* flash) {
    delete flash;
}

int spiflash_load(SpiFlashModel* flash, const uint8_t* data, size_t len, size_t offset) {
    if (!flash || !data) return -1;
    if (offset >= flash->data.size()) return -1;

    size_t to_copy = len;
    if (offset + len > flash->data.size()) {
        to_copy = flash->data.size() - offset;
    }

    std::memcpy(flash->data.data() + offset, data, to_copy);
    return static_cast<int>(to_copy);
}

uint8_t spiflash_step(SpiFlashModel* flash, int clk, int csn, uint8_t d_o) {
    if (!flash) return 0;
    return flash->step(clk != 0, csn != 0, d_o);
}

void spiflash_set_verbose(SpiFlashModel* flash, int verbose) {
    if (flash) flash->verbose = (verbose != 0);
}

uint8_t spiflash_get_command(SpiFlashModel* flash) {
    if (!flash) return 0;
    return flash->s.command;
}

uint32_t spiflash_get_byte_count(SpiFlashModel* flash) {
    if (!flash) return 0;
    return flash->s.byte_count;
}

uint32_t spiflash_get_step_count(SpiFlashModel* flash) {
    if (!flash) return 0;
    return flash->step_count;
}

uint32_t spiflash_get_posedge_count(SpiFlashModel* flash) {
    if (!flash) return 0;
    return flash->posedge_count;
}

uint32_t spiflash_get_negedge_count(SpiFlashModel* flash) {
    if (!flash) return 0;
    return flash->negedge_count;
}

}
