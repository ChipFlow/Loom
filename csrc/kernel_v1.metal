// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Metal port for Apple Silicon

#include <metal_stdlib>
using namespace metal;

#include "event_buffer.h"

// Type aliases to match CUDA code (also defined in event_buffer.h for Metal)
#ifndef u32
typedef uint32_t u32;
#endif
typedef uint64_t usize;

// Vectorized read structures for efficient memory access
struct VectorRead2 {
    u32 c1, c2;
};

struct VectorRead4 {
    u32 c1, c2, c3, c4;
};

// Simulation parameters passed as a constant buffer
struct SimParams {
    usize num_blocks;
    usize num_major_stages;
    usize num_cycles;
    usize state_size;
    usize current_cycle;
    usize current_stage;
};

// Helper function to read VectorRead2 from device memory
inline VectorRead2 read_vec2(device const u32* base, uint idx) {
    device const VectorRead2* ptr = (device const VectorRead2*)(base) + idx;
    return *ptr;
}

// Helper function to read VectorRead4 from device memory
inline VectorRead4 read_vec4(device const u32* base, uint idx) {
    device const VectorRead4* ptr = (device const VectorRead4*)(base) + idx;
    return *ptr;
}

// Core simulation block function
// This is called by each threadgroup to simulate one block
inline void simulate_block_v1(
    uint tid,  // thread index within threadgroup
    device const u32* script,
    usize script_size,
    device const u32* input_state,
    device u32* output_state,
    device u32* sram_data,
    device u32* sram_xmask,  // X-mask shadow for SRAM (null if xprop disabled)
    threadgroup u32* shared_metadata,
    threadgroup u32* shared_writeouts,
    threadgroup u32* shared_state,
    threadgroup u32* shared_writeouts_x,  // X-mask sideband for writeouts
    threadgroup u32* shared_state_x,      // X-mask sideband for state
    // Arrival times in raw picoseconds (ushort), max representable 65,535ps.
    // Gate delays are stored as u16 in the script padding slots.
    // No quantization needed — 512 bytes of threadgroup memory per block is negligible.
    threadgroup ushort* shared_arrival,
    // Writeout arrival times captured during writeout hook
    threadgroup ushort* shared_writeout_arrival,
    // Timing constraint buffer: one u32 per state word, [setup_ps:16][hold_ps:16]
    device const u32* timing_constraints,
    u32 clock_period_ps,
    device struct EventBuffer* event_buffer,
    u32 cycle_i
) {
    int script_pi = 0;

    while (true) {
        VectorRead2 t2_1, t2_2;
        VectorRead4 t4_1, t4_2, t4_3, t4_4, t4_5;

        shared_metadata[tid] = script[script_pi + tid];
        script_pi += 256;
        t2_1 = read_vec2(script + script_pi, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int num_stages = shared_metadata[0];
        if (!num_stages) {
            break;
        }
        int is_last_part = shared_metadata[1];
        int num_ios = shared_metadata[2];
        int io_offset = shared_metadata[3];
        int num_srams = shared_metadata[4];
        int sram_offset = shared_metadata[5];
        int num_global_read_rounds = shared_metadata[6];
        int num_output_duplicates = shared_metadata[7];
        bool is_x_capable = (shared_metadata[8] != 0);
        int xmask_state_offset = (int)shared_metadata[9];

        u32 writeout_hook_i = shared_metadata[128 + tid / 2];
        if (tid % 2 == 0) {
            writeout_hook_i = writeout_hook_i & ((1 << 16) - 1);
        } else {
            writeout_hook_i = writeout_hook_i >> 16;
        }

        t4_1 = read_vec4(script + script_pi + 256 * 2 * num_global_read_rounds, tid);
        t4_2 = read_vec4(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4, tid);
        t4_3 = read_vec4(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 2, tid);
        t4_4 = read_vec4(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 3, tid);
        t4_5 = read_vec4(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 4, tid);

        u32 t_global_rd_state = 0;
        u32 t_global_rd_state_x = 0;
        for (int gr_i = 0; gr_i < num_global_read_rounds; gr_i += 2) {
            u32 idx = t2_1.c1;
            u32 mask = t2_1.c2;
            script_pi += 256 * 2;
            t2_2 = read_vec2(script + script_pi, tid);

            if (mask) {
                device const u32* real_input_array;
                device const u32* real_xmask_array;
                if (idx >> 31) {
                    real_input_array = output_state - (1u << 31);
                    real_xmask_array = output_state + xmask_state_offset - (1u << 31);
                } else {
                    real_input_array = input_state;
                    real_xmask_array = input_state + xmask_state_offset;
                }
                u32 value = real_input_array[idx];
                u32 xmval = is_x_capable ? real_xmask_array[idx] : 0;
                u32 m = mask;
                while (m) {
                    t_global_rd_state <<= 1;
                    t_global_rd_state_x <<= 1;
                    u32 lowbit = m & -m;
                    if (value & lowbit) t_global_rd_state |= 1;
                    if (xmval & lowbit) t_global_rd_state_x |= 1;
                    m ^= lowbit;
                }
            }

            if (gr_i + 1 >= num_global_read_rounds) break;
            idx = t2_2.c1;
            mask = t2_2.c2;
            script_pi += 256 * 2;
            t2_1 = read_vec2(script + script_pi, tid);

            if (mask) {
                device const u32* real_input_array;
                device const u32* real_xmask_array;
                if (idx >> 31) {
                    real_input_array = output_state - (1u << 31);
                    real_xmask_array = output_state + xmask_state_offset - (1u << 31);
                } else {
                    real_input_array = input_state;
                    real_xmask_array = input_state + xmask_state_offset;
                }
                u32 value = real_input_array[idx];
                u32 xmval = is_x_capable ? real_xmask_array[idx] : 0;
                u32 m = mask;
                while (m) {
                    t_global_rd_state <<= 1;
                    t_global_rd_state_x <<= 1;
                    u32 lowbit = m & -m;
                    if (value & lowbit) t_global_rd_state |= 1;
                    if (xmval & lowbit) t_global_rd_state_x |= 1;
                    m ^= lowbit;
                }
            }
        }
        shared_state[tid] = t_global_rd_state;
        shared_state_x[tid] = t_global_rd_state_x;
        // Initialize arrival times to 0 (inputs have zero arrival)
        shared_arrival[tid] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int bs_i = 0; bs_i < num_stages; ++bs_i) {
            u32 hier_input = 0, hier_input_x = 0;
            u32 hier_flag_xora = 0, hier_flag_xorb = 0, hier_flag_orb = 0;

            // Macro-like inline expansion for shuffle input (value + X-mask)
            #define SHUF_INPUT_K(k_outer, k_inner, t_shuffle) { \
                u32 k = k_outer * 4 + k_inner; \
                u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1); \
                u32 t_shuffle_2_idx = t_shuffle >> 16; \
                hier_input |= (shared_state[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
                hier_input |= (shared_state[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
                if (is_x_capable) { \
                    hier_input_x |= (shared_state_x[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
                    hier_input_x |= (shared_state_x[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
                } \
            }

            script_pi += 256 * 4 * 5;
            SHUF_INPUT_K(0, 0, t4_1.c1); SHUF_INPUT_K(0, 1, t4_1.c2);
            SHUF_INPUT_K(0, 2, t4_1.c3); SHUF_INPUT_K(0, 3, t4_1.c4);
            t4_1 = read_vec4(script + script_pi, tid);

            SHUF_INPUT_K(1, 0, t4_2.c1); SHUF_INPUT_K(1, 1, t4_2.c2);
            SHUF_INPUT_K(1, 2, t4_2.c3); SHUF_INPUT_K(1, 3, t4_2.c4);
            t4_2 = read_vec4(script + script_pi + 256 * 4, tid);

            SHUF_INPUT_K(2, 0, t4_3.c1); SHUF_INPUT_K(2, 1, t4_3.c2);
            SHUF_INPUT_K(2, 2, t4_3.c3); SHUF_INPUT_K(2, 3, t4_3.c4);
            t4_3 = read_vec4(script + script_pi + 256 * 4 * 2, tid);

            SHUF_INPUT_K(3, 0, t4_4.c1); SHUF_INPUT_K(3, 1, t4_4.c2);
            SHUF_INPUT_K(3, 2, t4_4.c3); SHUF_INPUT_K(3, 3, t4_4.c4);
            t4_4 = read_vec4(script + script_pi + 256 * 4 * 3, tid);

            #undef SHUF_INPUT_K

            hier_flag_xora = t4_5.c1;
            hier_flag_xorb = t4_5.c2;
            hier_flag_orb = t4_5.c3;
            t4_5 = read_vec4(script + script_pi + 256 * 4 * 4, tid);

            // Extract per-thread-position gate delay from padding slot (u16 raw picoseconds)
            u32 hier_flag_padding = t4_5.c4;
            ushort gate_delay = (ushort)(hier_flag_padding & 0xFFFFu);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            shared_state[tid] = hier_input;
            shared_state_x[tid] = hier_input_x;
            shared_arrival[tid] = 0;  // Reset arrival for shuffle inputs
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // hier[0]: threads 128-255 compute AND gates + track arrivals
            if (tid >= 128) {
                u32 hier_input_a = shared_state[tid - 128];
                u32 hier_input_b = hier_input;
                u32 a_eff = hier_input_a ^ hier_flag_xora;
                u32 b_eff = (hier_input_b ^ hier_flag_xorb) | hier_flag_orb;
                u32 ret = a_eff & b_eff;
                shared_state[tid] = ret;

                if (is_x_capable) {
                    u32 a_x = shared_state_x[tid - 128];
                    u32 b_x = hier_input_x;
                    u32 b_eff_x = b_x & ~hier_flag_orb;
                    u32 ret_x = (a_x | b_eff_x) & (a_eff | a_x) & (b_eff | b_eff_x);
                    shared_state_x[tid] = ret_x;
                }

                // Arrival tracking: max(input_a, input_b) + gate_delay
                // Pass-through (orb == 0xFFFFFFFF) means no gate, just wire
                ushort arr_a = (ushort)shared_arrival[tid - 128];
                ushort arr_b = (ushort)shared_arrival[tid];
                bool is_pass = (hier_flag_orb == 0xFFFFFFFF);
                ushort new_arr = is_pass ? arr_a : (ushort)(max(arr_a, arr_b) + (ushort)gate_delay);
                shared_arrival[tid] = new_arr;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // hier[1..3]: shared memory reduction + arrival tracking
            u32 tmp_cur_hi = 0;
            u32 tmp_cur_hi_x = 0;
            ushort tmp_cur_arr = 0;
            for (int hi = 1; hi <= 3; ++hi) {
                int hier_width = 1 << (7 - hi);
                if (tid >= (uint)hier_width && tid < (uint)(hier_width * 2)) {
                    u32 hier_input_a = shared_state[tid + hier_width];
                    u32 hier_input_b = shared_state[tid + hier_width * 2];
                    u32 a_eff = hier_input_a ^ hier_flag_xora;
                    u32 b_eff = (hier_input_b ^ hier_flag_xorb) | hier_flag_orb;
                    u32 ret = a_eff & b_eff;
                    tmp_cur_hi = ret;
                    shared_state[tid] = ret;

                    if (is_x_capable) {
                        u32 a_x = shared_state_x[tid + hier_width];
                        u32 b_x = shared_state_x[tid + hier_width * 2];
                        u32 b_eff_x = b_x & ~hier_flag_orb;
                        u32 ret_x = (a_x | b_eff_x) & (a_eff | a_x) & (b_eff | b_eff_x);
                        tmp_cur_hi_x = ret_x;
                        shared_state_x[tid] = ret_x;
                    }

                    // Arrival tracking
                    ushort arr_a = (ushort)shared_arrival[tid + hier_width];
                    ushort arr_b = (ushort)shared_arrival[tid + hier_width * 2];
                    bool is_pass = (hier_flag_orb == 0xFFFFFFFF);
                    ushort new_arr = is_pass ? arr_a : (ushort)(max(arr_a, arr_b) + (ushort)gate_delay);
                    tmp_cur_arr = new_arr;
                    shared_arrival[tid] = tmp_cur_arr;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // hier[4..7], within the first SIMD group (32 threads)
            // Pack arrival into u32 for SIMD shuffle (Metal simd_shuffle works on u32)
            u32 tmp_cur_arr_u32 = (u32)tmp_cur_arr;
            if (tid < 32) {
                for (int hi = 4; hi <= 7; ++hi) {
                    int hier_width = 1 << (7 - hi);
                    u32 hier_input_a = simd_shuffle_down(tmp_cur_hi, hier_width);
                    u32 hier_input_b = simd_shuffle_down(tmp_cur_hi, hier_width * 2);
                    u32 hier_a_x = is_x_capable ? simd_shuffle_down(tmp_cur_hi_x, hier_width) : 0;
                    u32 hier_b_x = is_x_capable ? simd_shuffle_down(tmp_cur_hi_x, hier_width * 2) : 0;
                    u32 arr_a_u32 = simd_shuffle_down(tmp_cur_arr_u32, hier_width);
                    u32 arr_b_u32 = simd_shuffle_down(tmp_cur_arr_u32, hier_width * 2);
                    if (tid >= (uint)hier_width && tid < (uint)(hier_width * 2)) {
                        u32 a_eff = hier_input_a ^ hier_flag_xora;
                        u32 b_eff = (hier_input_b ^ hier_flag_xorb) | hier_flag_orb;
                        tmp_cur_hi = a_eff & b_eff;
                        if (is_x_capable) {
                            u32 b_eff_x = hier_b_x & ~hier_flag_orb;
                            tmp_cur_hi_x = (hier_a_x | b_eff_x) & (a_eff | hier_a_x) & (b_eff | b_eff_x);
                        }
                        // Arrival tracking
                        bool is_pass = (hier_flag_orb == 0xFFFFFFFF);
                        ushort new_arr = is_pass ? (ushort)arr_a_u32 : (ushort)(max((ushort)arr_a_u32, (ushort)arr_b_u32) + (ushort)gate_delay);
                        tmp_cur_arr_u32 = (u32)new_arr;
                    }
                }
                u32 v1 = simd_shuffle_down(tmp_cur_hi, 1);
                u32 v1_x = is_x_capable ? simd_shuffle_down(tmp_cur_hi_x, 1) : 0;
                // hier[8..12]: bit-level operations within single u32
                // All 32 signals share one thread's arrival — arrival carries forward unchanged
                if (tid == 0) {
                    // Value lane
                    u32 r8 = ((v1 << 16) ^ hier_flag_xora) & ((v1 ^ hier_flag_xorb) | hier_flag_orb) & 0xffff0000;
                    u32 r9 = ((r8 >> 8) ^ hier_flag_xora) & (((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb) & 0xff00;
                    u32 r10 = ((r9 >> 4) ^ hier_flag_xora) & (((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb) & 0xf0;
                    u32 r11 = ((r10 >> 2) ^ hier_flag_xora) & (((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb) & 12;
                    u32 r12 = ((r11 >> 1) ^ hier_flag_xora) & (((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb) & 2;
                    tmp_cur_hi = r8 | r9 | r10 | r11 | r12;

                    if (is_x_capable) {
                        // X-mask lane: same structure but using X-prop AND formula
                        #define XPROP_HIER_BIT(a_v, a_x, b_v, b_x, msk) { \
                            u32 ae = a_v ^ hier_flag_xora; \
                            u32 be = (b_v ^ hier_flag_xorb) | hier_flag_orb; \
                            u32 bex = b_x & ~hier_flag_orb; \
                            u32 rv = (ae & be) & msk; \
                            u32 rx = ((a_x | bex) & (ae | a_x) & (be | bex)) & msk;
                        u32 r8_x, r9_x, r10_x, r11_x, r12_x;
                        {
                            u32 ae = (v1 << 16) ^ hier_flag_xora;
                            u32 be = (v1 ^ hier_flag_xorb) | hier_flag_orb;
                            u32 bex = v1_x & ~hier_flag_orb;
                            u32 ax = v1_x << 16;
                            r8_x = ((ax | bex) & (ae | ax) & (be | bex)) & 0xffff0000u;
                        }
                        {
                            u32 ae = (r8 >> 8) ^ hier_flag_xora;
                            u32 be = ((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb;
                            u32 ax = r8_x >> 8;
                            u32 bex = (r8_x >> 16) & ~hier_flag_orb;
                            r9_x = ((ax | bex) & (ae | ax) & (be | bex)) & 0xff00u;
                        }
                        {
                            u32 ae = (r9 >> 4) ^ hier_flag_xora;
                            u32 be = ((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb;
                            u32 ax = r9_x >> 4;
                            u32 bex = (r9_x >> 8) & ~hier_flag_orb;
                            r10_x = ((ax | bex) & (ae | ax) & (be | bex)) & 0xf0u;
                        }
                        {
                            u32 ae = (r10 >> 2) ^ hier_flag_xora;
                            u32 be = ((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb;
                            u32 ax = r10_x >> 2;
                            u32 bex = (r10_x >> 4) & ~hier_flag_orb;
                            r11_x = ((ax | bex) & (ae | ax) & (be | bex)) & 0xcu;
                        }
                        {
                            u32 ae = (r11 >> 1) ^ hier_flag_xora;
                            u32 be = ((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb;
                            u32 ax = r11_x >> 1;
                            u32 bex = (r11_x >> 2) & ~hier_flag_orb;
                            r12_x = ((ax | bex) & (ae | ax) & (be | bex)) & 0x2u;
                        }
                        tmp_cur_hi_x = r8_x | r9_x | r10_x | r11_x | r12_x;
                        #undef XPROP_HIER_BIT
                    }
                }
                shared_state[tid] = tmp_cur_hi;
                shared_state_x[tid] = tmp_cur_hi_x;
                shared_arrival[tid] = (ushort)tmp_cur_arr_u32;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // write out
            if ((writeout_hook_i >> 8) == (uint)bs_i) {
                shared_writeouts[tid] = shared_state[writeout_hook_i & 255];
                if (is_x_capable) {
                    shared_writeouts_x[tid] = shared_state_x[writeout_hook_i & 255];
                }
                shared_writeout_arrival[tid] = shared_arrival[writeout_hook_i & 255];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // sram & duplicate permutation
        u32 sram_duplicate_t = 0;
        u32 sram_duplicate_t_x = 0;

        #define SHUF_SRAM_DUPL_K(k_outer, k_inner, t_shuffle) { \
            u32 k = k_outer * 4 + k_inner; \
            u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1); \
            u32 t_shuffle_2_idx = t_shuffle >> 16; \
            sram_duplicate_t |= (shared_writeouts[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
            sram_duplicate_t |= (shared_writeouts[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
            if (is_x_capable) { \
                sram_duplicate_t_x |= (shared_writeouts_x[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
                sram_duplicate_t_x |= (shared_writeouts_x[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
            } \
        }

        script_pi += 256 * 4 * 5;
        SHUF_SRAM_DUPL_K(0, 0, t4_1.c1); SHUF_SRAM_DUPL_K(0, 1, t4_1.c2);
        SHUF_SRAM_DUPL_K(0, 2, t4_1.c3); SHUF_SRAM_DUPL_K(0, 3, t4_1.c4);
        t4_1 = read_vec4(script + script_pi, tid);

        SHUF_SRAM_DUPL_K(1, 0, t4_2.c1); SHUF_SRAM_DUPL_K(1, 1, t4_2.c2);
        SHUF_SRAM_DUPL_K(1, 2, t4_2.c3); SHUF_SRAM_DUPL_K(1, 3, t4_2.c4);
        t4_2 = read_vec4(script + script_pi + 256 * 4, tid);

        SHUF_SRAM_DUPL_K(2, 0, t4_3.c1); SHUF_SRAM_DUPL_K(2, 1, t4_3.c2);
        SHUF_SRAM_DUPL_K(2, 2, t4_3.c3); SHUF_SRAM_DUPL_K(2, 3, t4_3.c4);
        t4_3 = read_vec4(script + script_pi + 256 * 4 * 2, tid);

        SHUF_SRAM_DUPL_K(3, 0, t4_4.c1); SHUF_SRAM_DUPL_K(3, 1, t4_4.c2);
        SHUF_SRAM_DUPL_K(3, 2, t4_4.c3); SHUF_SRAM_DUPL_K(3, 3, t4_4.c4);
        t4_4 = read_vec4(script + script_pi + 256 * 4 * 3, tid);

        #undef SHUF_SRAM_DUPL_K

        sram_duplicate_t = (sram_duplicate_t & ~t4_5.c2) ^ t4_5.c1;
        if (is_x_capable) {
            sram_duplicate_t_x = sram_duplicate_t_x & ~t4_5.c2; // set0 clears X too
        }
        t4_5 = read_vec4(script + script_pi + 256 * 4 * 4, tid);

        // sram read
        device u32* ram = nullptr;
        device u32* ram_x = nullptr;
        u32 r = 0, w0 = 0, r_x = 0, w0_x = 0;
        u32 port_w_addr_iv = 0, port_w_wr_en = 0, port_w_wr_data_iv = 0;
        u32 port_w_wr_data_x = 0;

        if (tid < (uint)(num_srams * 4)) {
            u32 addrs = sram_duplicate_t;
            // SIMD shuffle for SRAM operations
            port_w_wr_en = simd_shuffle_down(sram_duplicate_t, 1);
            port_w_wr_data_iv = simd_shuffle_down(sram_duplicate_t, 2);
            if (is_x_capable) {
                port_w_wr_data_x = simd_shuffle_down(sram_duplicate_t_x, 2);
            }

            if (tid % 4 == 0) {
                u32 sram_i = tid / 4;
                u32 sram_st = sram_offset + sram_i * (1 << 13);
                u32 port_r_addr_iv = addrs & 0xffff;
                port_w_addr_iv = addrs >> 16;

                ram = sram_data + sram_st;
                r = ram[port_r_addr_iv];
                w0 = ram[port_w_addr_iv];
                if (is_x_capable && sram_xmask != nullptr) {
                    ram_x = sram_xmask + sram_st;
                    r_x = ram_x[port_r_addr_iv];
                    w0_x = ram_x[port_w_addr_iv];
                }
            }
        }

        // clock enable permutation
        u32 clken_perm = 0;
        u32 clken_perm_x = 0;

        #define SHUF_CLKEN_K(k_outer, k_inner, t_shuffle) { \
            u32 k = k_outer * 4 + k_inner; \
            u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1); \
            u32 t_shuffle_2_idx = t_shuffle >> 16; \
            clken_perm |= (shared_writeouts[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
            clken_perm |= (shared_writeouts[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
            if (is_x_capable) { \
                clken_perm_x |= (shared_writeouts_x[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
                clken_perm_x |= (shared_writeouts_x[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
            } \
        }

        script_pi += 256 * 4 * 5;
        SHUF_CLKEN_K(0, 0, t4_1.c1); SHUF_CLKEN_K(0, 1, t4_1.c2);
        SHUF_CLKEN_K(0, 2, t4_1.c3); SHUF_CLKEN_K(0, 3, t4_1.c4);
        SHUF_CLKEN_K(1, 0, t4_2.c1); SHUF_CLKEN_K(1, 1, t4_2.c2);
        SHUF_CLKEN_K(1, 2, t4_2.c3); SHUF_CLKEN_K(1, 3, t4_2.c4);
        SHUF_CLKEN_K(2, 0, t4_3.c1); SHUF_CLKEN_K(2, 1, t4_3.c2);
        SHUF_CLKEN_K(2, 2, t4_3.c3); SHUF_CLKEN_K(2, 3, t4_3.c4);
        SHUF_CLKEN_K(3, 0, t4_4.c1); SHUF_CLKEN_K(3, 1, t4_4.c2);
        SHUF_CLKEN_K(3, 2, t4_4.c3); SHUF_CLKEN_K(3, 3, t4_4.c4);

        #undef SHUF_CLKEN_K

        // sram commit
        if (tid < (uint)(num_srams * 4)) {
            if (tid % 4 == 0) {
                u32 sram_i = tid / 4;
                shared_writeouts[num_ios - num_srams + sram_i] = r;
                ram[port_w_addr_iv] = (w0 & ~port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
                if (is_x_capable) {
                    shared_writeouts_x[num_ios - num_srams + sram_i] = r_x;
                    if (ram_x != nullptr) {
                        ram_x[port_w_addr_iv] = (w0_x & ~port_w_wr_en) | (port_w_wr_data_x & port_w_wr_en);
                    }
                }
            }
        } else if (tid < (uint)(num_srams * 4 + num_output_duplicates)) {
            shared_writeouts[num_ios - num_srams - num_output_duplicates + (tid - num_srams * 4)] = sram_duplicate_t;
            if (is_x_capable) {
                shared_writeouts_x[num_ios - num_srams - num_output_duplicates + (tid - num_srams * 4)] = sram_duplicate_t_x;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        u32 writeout_inv = shared_writeouts[tid];
        u32 writeout_inv_x = is_x_capable ? shared_writeouts_x[tid] : 0;

        clken_perm = (clken_perm & ~t4_5.c2) ^ t4_5.c1;
        if (is_x_capable) {
            clken_perm_x = clken_perm_x & ~t4_5.c2; // set0 clears X
        }
        writeout_inv ^= t4_5.c3;
        // data_inv (t4_5.c3) doesn't affect X-mask (inversion preserves X)

        if (tid < (uint)num_ios) {
            u32 old_wo = input_state[io_offset + tid];
            u32 wo = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
            output_state[io_offset + tid] = wo;

            if (is_x_capable) {
                u32 old_x = input_state[xmask_state_offset + io_offset + tid];
                // X-mask DFF gating: X clken → output is X
                u32 wo_x = (old_x & ~clken_perm & ~clken_perm_x)
                         | (writeout_inv_x & clken_perm & ~clken_perm_x)
                         | clken_perm_x;
                output_state[xmask_state_offset + io_offset + tid] = wo_x;
            }
        }

        // DFF timing violation check (per writeout word)
        if (tid < (uint)num_ios && clken_perm != 0 && timing_constraints != nullptr) {
            u32 constraint = timing_constraints[io_offset + tid];
            if (constraint != 0) {
                ushort setup_ps = (ushort)(constraint >> 16);
                ushort hold_ps = (ushort)(constraint & 0xFFFF);
                ushort arrival = shared_writeout_arrival[tid];
                // Setup: arrival + setup must fit within clock period
                if (arrival > 0 && (u32)arrival + (u32)setup_ps > clock_period_ps) {
                    int slack = (int)clock_period_ps - (int)arrival - (int)setup_ps;
                    write_event(event_buffer, EVENT_TYPE_SETUP_VIOLATION,
                               io_offset + tid, cycle_i,
                               io_offset + tid, (u32)slack, (u32)arrival, (u32)setup_ps);
                }
                // Hold: arrival must exceed hold time
                if (arrival < hold_ps) {
                    int slack = (int)arrival - (int)hold_ps;
                    write_event(event_buffer, EVENT_TYPE_HOLD_VIOLATION,
                               io_offset + tid, cycle_i,
                               io_offset + tid, (u32)slack, (u32)arrival, (u32)hold_ps);
                }
            }
        }

        if (is_last_part) break;
    }
}

// Main compute kernel for one stage
// This kernel processes one stage for all blocks. The host will dispatch
// this kernel multiple times (once per stage per cycle) with explicit
// completion waits between dispatches, replacing CUDA's grid-wide sync.
kernel void simulate_v1_stage(
    device const usize* blocks_start [[buffer(0)]],
    device const u32* blocks_data [[buffer(1)]],
    device u32* sram_data [[buffer(2)]],
    device u32* states_noninteractive [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    device struct EventBuffer* event_buffer [[buffer(5)]],
    device const u32* timing_constraints [[buffer(6)]],
    device u32* sram_xmask [[buffer(7)]],  // X-mask shadow for SRAM (null if xprop disabled)
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Threadgroup shared memory (equivalent to CUDA __shared__)
    threadgroup u32 shared_metadata[256];
    threadgroup u32 shared_writeouts[256];
    threadgroup u32 shared_state[256];
    threadgroup u32 shared_writeouts_x[256];  // X-mask sideband for writeouts
    threadgroup u32 shared_state_x[256];      // X-mask sideband for state
    threadgroup ushort shared_arrival[256];  // Per-thread-position arrival times (raw picoseconds)
    threadgroup ushort shared_writeout_arrival[256];  // Arrival times captured at writeout
    shared_writeout_arrival[tid] = 0;  // Must initialize — only writeout threads update this
    shared_state_x[tid] = 0;
    shared_writeouts_x[tid] = 0;

    usize stage_i = params.current_stage;
    usize cycle_i = params.current_cycle;

    // Clock period is stored in timing_constraints[0] when constraints are present,
    // but we pass it via SimParams to avoid an extra read. Use state_size as a proxy
    // for whether constraints exist (non-null buffer check happens in simulate_block_v1).
    // Read clock_period from first element of timing_constraints buffer.
    u32 clock_period_ps = (timing_constraints != nullptr) ? timing_constraints[0] : 0;

    // Get script location for this block and stage
    usize script_start = blocks_start[stage_i * params.num_blocks + gid];
    usize script_end = blocks_start[stage_i * params.num_blocks + gid + 1];
    usize script_size = script_end - script_start;

    device const u32* script = blocks_data + script_start;
    device const u32* input_state = states_noninteractive + cycle_i * params.state_size;
    device u32* output_state = states_noninteractive + (cycle_i + 1) * params.state_size;

    // Constraints start at index 1 (index 0 is clock_period_ps)
    device const u32* constraints_data = (timing_constraints != nullptr) ? timing_constraints + 1 : nullptr;

    simulate_block_v1(
        tid,
        script,
        script_size,
        input_state,
        output_state,
        sram_data,
        sram_xmask,
        shared_metadata,
        shared_writeouts,
        shared_state,
        shared_writeouts_x,
        shared_state_x,
        shared_arrival,
        shared_writeout_arrival,
        constraints_data,
        clock_period_ps,
        event_buffer,
        (u32)cycle_i
    );
}

// ── State Prep Kernel ────────────────────────────────────────────────────
//
// Copies output state → input state and applies bit operations (set clock,
// posedge/negedge flags, flash D_IN, etc.) entirely on GPU.
//
// Used to batch many ticks in a single command buffer during idle periods
// (no CPU intervention needed between dispatches).
//
// Dispatched with a single threadgroup of 256 threads.

struct BitOp {
    u32 position;  // bit position in state buffer
    u32 value;     // 0 = clear, 1 = set
};

struct StatePrepParams {
    u32 state_size;    // number of u32 words per state slot
    u32 num_ops;       // number of bit set/clear operations
    u32 num_monitors;  // number of peripheral monitors to check (0 = skip)
    u32 tick_number;   // current tick number (written to control block on callback)
};

// ── state_prep: copy output→input and apply bit ops ──────────────────────

kernel void state_prep(
    device u32* states [[buffer(0)]],
    constant StatePrepParams& params [[buffer(1)]],
    constant BitOp* ops [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    u32 state_size = params.state_size;

    // Step 1: Copy output state → input state
    // output is at states[state_size..2*state_size], input at states[0..state_size]
    for (uint i = tid; i < state_size; i += 256) {
        states[i] = states[state_size + i];
    }

    // Ensure copy is complete before modifying bits
    threadgroup_barrier(mem_flags::mem_device);

    // Step 2: Apply bit operations to input state (only thread 0)
    if (tid == 0) {
        for (uint i = 0; i < params.num_ops; i++) {
            u32 pos = ops[i].position;
            u32 word_idx = pos >> 5;
            u32 bit_mask = 1u << (pos & 31u);
            if (ops[i].value != 0) {
                states[word_idx] |= bit_mask;
            } else {
                states[word_idx] &= ~bit_mask;
            }
        }
    }
}

// ── GPU-side Flash + UART IO Models ──────────────────────────────────────────
//
// These kernels move SPI flash and UART decoding entirely to the GPU,
// eliminating the per-tick CPU round-trip that was the simulation bottleneck.
//
// Per-tick pipeline (all GPU, no CPU interaction):
//   state_prep(fall) -> gpu_apply_flash_din -> simulate×N (fall)
//   -> gpu_flash_model_step (sees SPI CLK after falling edge)
//   -> state_prep(rise) -> gpu_apply_flash_din -> simulate×N (rise)
//   -> gpu_flash_model_step (sees SPI CLK after rising edge)
//   -> gpu_io_step (UART + bus trace)
//   [repeat × K ticks]
//   -> signal(batch_done)
//   CPU: drain UART channel + bus trace
//
// Flash runs TWICE per tick to match timing_sim_cpu behavior: the SPI CLK
// passes through clock gating (Q = sys_CLK & EN_latch), so the flash sees
// CLK=0 after the falling edge and CLK=EN_latch after the rising edge.

// ── Flash State (persistent across ticks) ────────────────────────────────────

struct FlashState {
    int bit_count;       // bits received in current byte
    int byte_count;      // bytes in current transaction
    uint data_width;     // 1 (SPI) or 4 (QSPI)
    uint addr;           // 24-bit read address
    uchar curr_byte;     // byte accumulator
    uchar command;       // current command (0x03, 0xEB, etc.)
    uchar out_buffer;    // MISO shift register
    uchar _pad1;
    uint prev_clk;       // model internal: last clk seen by flash_eval_commit
    uint prev_csn;       // delayed csn: output csn from previous tick
    uchar d_i;           // current MISO output nibble (4 bits)
    uchar _pad2[3];
    uchar prev_d_out;    // previous MOSI for setup delay
    uchar _pad3[3];
    uint in_reset;       // 1 = reset active, forces d_i=0x0F
    uint last_error_cmd; // nonzero if unknown command encountered
    uint model_prev_csn; // model internal: last csn seen by flash_eval_commit (for edge detection)
};

struct FlashDinParams {
    u32 d_in_pos[4];     // input state bit positions for d0..d3
    u32 has_flash;       // 0 = skip
};

struct FlashModelParams {
    u32 state_size;       // words per state slot
    u32 clk_out_pos;      // output state bit position: flash_clk
    u32 csn_out_pos;      // output state bit position: flash_csn
    u32 d_out_pos[4];     // output state bit positions: d0..d3
    u32 flash_data_size;  // firmware size in bytes
};

// ── Flash model inline helpers (ported from spiflash_model.cc) ───────────────

// Process a completed byte (command decode + address accumulation + data lookup)
inline void flash_process_byte(
    thread int& bit_count,
    thread int& byte_count,
    thread uint& data_width,
    thread uint& addr,
    thread uchar& curr_byte,
    thread uchar& command,
    thread uchar& out_buffer,
    thread uint& last_error_cmd,
    device const uchar* flash_data,
    u32 flash_data_size
) {
    out_buffer = 0;
    if (byte_count == 0) {
        addr = 0;
        data_width = 1;
        command = curr_byte;
        if (command == 0xab) {
            // power up - nothing to do
        } else if (command == 0x03 || command == 0x9f || command == 0xff
            || command == 0x35 || command == 0x31 || command == 0x50
            || command == 0x05 || command == 0x01 || command == 0x06) {
            // nothing to do
        } else if (command == 0xeb) {
            data_width = 4;
        } else {
            last_error_cmd = command;
        }
    } else {
        if (command == 0x03) {
            // Single read
            if (byte_count <= 3) {
                addr |= (uint(curr_byte) << ((3 - byte_count) * 8));
            }
            if (byte_count >= 3) {
                uint idx = addr & 0x00FFFFFFu;
                if (idx < flash_data_size) {
                    out_buffer = flash_data[idx];
                } else {
                    out_buffer = 0xFF;
                }
                addr = (addr + 1) & 0x00FFFFFFu;
            }
        } else if (command == 0xeb) {
            // Quad read
            if (byte_count <= 3) {
                addr |= (uint(curr_byte) << ((3 - byte_count) * 8));
            }
            if (byte_count >= 6) { // 1 mode, 2 dummy clocks
                uint idx = addr & 0x00FFFFFFu;
                if (idx < flash_data_size) {
                    out_buffer = flash_data[idx];
                } else {
                    out_buffer = 0xFF;
                }
                addr = (addr + 1) & 0x00FFFFFFu;
            }
        }
    }
    if (command == 0x9f) {
        // Read ID
        const uchar flash_id[4] = {0xCA, 0x7C, 0xA7, 0xFF};
        out_buffer = flash_id[byte_count % 4];
    }
}

// Single eval+commit step with persistent d_i (matching C++ p_d_i member behavior).
// d_i is only modified on negedge_clk; otherwise retains its previous value.
inline void flash_eval_commit_persistent(
    uint clk,
    uint csn,
    uchar d_out,
    // State (read/write):
    thread int& bit_count,
    thread int& byte_count,
    thread uint& data_width,
    thread uint& addr,
    thread uchar& curr_byte,
    thread uchar& command,
    thread uchar& out_buffer,
    thread uint& prev_clk,
    thread uint& prev_csn,
    thread uint& last_error_cmd,
    thread uchar& d_i,  // persistent: only updated on negedge
    // Flash data:
    device const uchar* flash_data,
    u32 flash_data_size
) {
    // Edge detection
    bool posedge_clk = (clk != 0) && (prev_clk == 0);
    bool negedge_clk = (clk == 0) && (prev_clk != 0);
    bool posedge_csn = (csn != 0) && (prev_csn == 0);

    if (posedge_csn) {
        bit_count = 0;
        byte_count = 0;
        data_width = 1;
    } else if (posedge_clk && csn == 0) {
        if (data_width == 4) {
            curr_byte = (curr_byte << 4U) | (d_out & 0xF);
        } else {
            curr_byte = (curr_byte << 1U) | (d_out & 0x1);
        }
        out_buffer = out_buffer << data_width;
        bit_count += data_width;
        if ((uint)bit_count >= 8) {
            flash_process_byte(bit_count, byte_count, data_width, addr,
                curr_byte, command, out_buffer, last_error_cmd,
                flash_data, flash_data_size);
            ++byte_count;
            bit_count = 0;
        }
    } else if (negedge_clk && csn == 0) {
        // Only update d_i on negedge (matching C++ p_d_i behavior)
        if (data_width == 4) {
            d_i = (out_buffer >> 4U) & 0xFU;
        } else {
            d_i = ((out_buffer >> 7U) & 0x1U) << 1U;
        }
    }
    prev_clk = clk;
    prev_csn = csn;
}

// ── gpu_apply_flash_din: write FlashState.d_i → input state ──────────────────
//
// Trivial single-thread kernel. Runs after each state_prep to apply flash
// data input bits before simulation.

kernel void gpu_apply_flash_din(
    device u32* states [[buffer(0)]],
    device const FlashState* flash_state [[buffer(1)]],
    constant FlashDinParams& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0 || params.has_flash == 0) return;

    uchar d_i = flash_state->d_i;

    for (uint i = 0; i < 4; i++) {
        u32 pos = params.d_in_pos[i];
        if (pos == 0xFFFFFFFFu) continue;
        u32 word_idx = pos >> 5;
        u32 bit_mask = 1u << (pos & 31u);
        if ((d_i >> i) & 1) {
            states[word_idx] |= bit_mask;
        } else {
            states[word_idx] &= ~bit_mask;
        }
    }
}

// ── gpu_flash_model_step: SPI flash FSM (dual-step with setup delay) ─────────
//
// Direct port of SpiFlashModel from spiflash_model.cc.
// Reads flash_clk, flash_csn, d_out from output state.
// Performs dual-step (setup delay): eval_commit(prev_d_out), eval_commit(d_out).
// Stores result d_i in FlashState for next gpu_apply_flash_din.
//
// Runs TWICE per tick: once after falling-edge simulate, once after rising-edge
// simulate. This matches timing_sim_cpu's dual flash stepping.

kernel void gpu_flash_model_step(
    device u32* states [[buffer(0)]],
    device FlashState* flash_state [[buffer(1)]],
    constant FlashModelParams& params [[buffer(2)]],
    device const uchar* flash_data [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;

    u32 state_size = params.state_size;

    // Read current output state signals
    u32 clk_word = params.clk_out_pos >> 5;
    u32 clk_bit = params.clk_out_pos & 31u;
    uint clk = (states[state_size + clk_word] >> clk_bit) & 1u;

    u32 csn_word = params.csn_out_pos >> 5;
    u32 csn_bit = params.csn_out_pos & 31u;
    uint csn = (states[state_size + csn_word] >> csn_bit) & 1u;

    // Read d_out nibble from output state
    uchar d_out = 0;
    for (uint i = 0; i < 4; i++) {
        u32 pos = params.d_out_pos[i];
        if (pos == 0xFFFFFFFFu) continue;
        u32 w = pos >> 5;
        u32 b = pos & 31u;
        d_out |= (uchar)(((states[state_size + w] >> b) & 1u) << i);
    }

    // If in reset, force d_i high and update prev values
    if (flash_state->in_reset) {
        flash_state->d_i = 0x0F;
        flash_state->prev_clk = clk;
        flash_state->prev_csn = csn;
        flash_state->model_prev_csn = csn;
        flash_state->prev_d_out = d_out;
        return;
    }

    // Load state into locals for eval_commit
    int bit_count = flash_state->bit_count;
    int byte_count = flash_state->byte_count;
    uint data_width = flash_state->data_width;
    uint addr = flash_state->addr;
    uchar curr_byte = flash_state->curr_byte;
    uchar command = flash_state->command;
    uchar out_buffer = flash_state->out_buffer;
    uint prev_clk = flash_state->prev_clk;
    uint prev_csn = flash_state->prev_csn;
    uint last_error_cmd = flash_state->last_error_cmd;
    uchar prev_d_out = flash_state->prev_d_out;

    // Dual-step for setup delay (matches old GPU sim CPU callback behavior):
    // The SPI clock, CSN, and data DFFs all advance simultaneously on the system
    // clock edge. We feed delayed CSN/data with current clock to model setup timing.
    //
    // model_prev_csn tracks the model's internal edge detection state (the CSN value
    // the model last saw). prev_csn holds the delayed CSN (previous tick's output).
    uint model_prev_csn = flash_state->model_prev_csn;

    // d_i persists across steps (matching C++ p_d_i member variable behavior).
    // It's only updated on negedge_clk inside flash_eval_commit.
    uchar d_i = flash_state->d_i;

    // Step 1: eval with prev_csn + prev_d_out (processes posedge, samples old data)
    flash_eval_commit_persistent(clk, prev_csn, prev_d_out,
        bit_count, byte_count, data_width, addr,
        curr_byte, command, out_buffer, prev_clk, model_prev_csn,
        last_error_cmd, d_i, flash_data, params.flash_data_size);

    // Step 2: eval with prev_csn + current d_out
    flash_eval_commit_persistent(clk, prev_csn, d_out,
        bit_count, byte_count, data_width, addr,
        curr_byte, command, out_buffer, prev_clk, model_prev_csn,
        last_error_cmd, d_i, flash_data, params.flash_data_size);

    // Write state back
    flash_state->bit_count = bit_count;
    flash_state->byte_count = byte_count;
    flash_state->data_width = data_width;
    flash_state->addr = addr;
    flash_state->curr_byte = curr_byte;
    flash_state->command = command;
    flash_state->out_buffer = out_buffer;
    flash_state->prev_clk = clk;
    flash_state->prev_csn = csn;  // store current output csn → becomes next tick's delayed csn
    flash_state->model_prev_csn = model_prev_csn;  // model's edge detection state
    flash_state->d_i = d_i;
    flash_state->prev_d_out = d_out;
    flash_state->last_error_cmd = last_error_cmd;
}

// ── UART Decoder State + Channel ─────────────────────────────────────────────

struct UartDecoderState {
    u32 state;          // 0=IDLE, 1=START, 2=DATA, 3=STOP
    u32 last_tx;
    u32 start_cycle;
    u32 bits_received;
    u32 value;
    u32 current_cycle;  // incremented each call
};

struct UartParams {
    u32 state_size;
    u32 tx_out_pos;       // output state bit position for UART TX
    u32 cycles_per_bit;   // clock_hz / baud_rate
    u32 has_uart;         // 0 = skip
};

struct UartChannel {
    u32 write_head;       // CPU reads this to know how many bytes are available
    u32 capacity;
    u32 _pad[2];
    uchar data[4096];     // decoded bytes ring buffer
};

// ── Wishbone Bus Trace Structs ──────────────────────────────────────────────

#define WB_TRACE_MAX_ADR_BITS 30
#define WB_TRACE_MAX_DAT_BITS 32
#define WB_TRACE_CHANNEL_CAP 16384

struct WbTraceParams {
    // ibus signal positions (in output state, 0xFFFFFFFF = unused)
    u32 ibus_cyc_pos;
    u32 ibus_stb_pos;
    u32 ibus_adr_pos[WB_TRACE_MAX_ADR_BITS];
    u32 ibus_rdata_pos[WB_TRACE_MAX_DAT_BITS];
    // dbus signal positions
    u32 dbus_cyc_pos;
    u32 dbus_stb_pos;
    u32 dbus_we_pos;
    u32 dbus_adr_pos[WB_TRACE_MAX_ADR_BITS];
    // ack positions
    u32 spiflash_ack_pos;
    u32 sram_ack_pos;
    u32 csr_ack_pos;
    // control
    u32 has_trace;  // 0 = skip
};

// Compact per-tick bus snapshot
struct WbTraceEntry {
    u32 tick;
    u32 flags;      // [0]=ibus_cyc [1]=ibus_stb [2]=dbus_cyc [3]=dbus_stb
                    // [4]=dbus_we [5]=spiflash_ack [6]=sram_ack [7]=csr_ack
    u32 ibus_adr;   // packed 30-bit address
    u32 ibus_rdata; // packed 32-bit instruction data
    u32 dbus_adr;   // packed 30-bit address
};

struct WbTraceChannel {
    u32 write_head;
    u32 capacity;
    u32 current_tick;
    u32 prev_flags;   // previous flags for edge detection
    // entries[capacity] follow immediately in memory (at byte offset 16)
};

// ── gpu_io_step: Combined UART decoder + Wishbone bus trace ─────────────────
//
// Runs once per tick. Decodes UART TX bytes, captures bus transactions.
// Single dispatch replaces both gpu_uart_step and gpu_wb_trace.

kernel void gpu_io_step(
    device u32* states [[buffer(0)]],
    device UartDecoderState* uart_state [[buffer(1)]],
    constant UartParams& uart_params [[buffer(2)]],
    device UartChannel* uart_channel [[buffer(3)]],
    device WbTraceChannel* wb_channel [[buffer(4)]],
    constant WbTraceParams& wb_params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;

    u32 state_size = uart_params.state_size;

    // Helper: read bit from output state
    #define READ_OUT_BIT(pos) \
        (((pos) != 0xFFFFFFFFu) ? ((states[state_size + ((pos) >> 5)] >> ((pos) & 31u)) & 1u) : 0u)

    // ── UART TX decoder ─────────────────────────────────────────────────
    if (uart_params.has_uart != 0) {
        u32 cycles_per_bit = uart_params.cycles_per_bit;
        u32 tx = READ_OUT_BIT(uart_params.tx_out_pos);

        u32 cycle = uart_state->current_cycle;
        u32 st = uart_state->state;
        u32 last_tx = uart_state->last_tx;
        u32 start_cycle = uart_state->start_cycle;
        u32 bits_received = uart_state->bits_received;
        u32 value = uart_state->value;

        if (st == 0) {
            if (last_tx == 1 && tx == 0) {
                st = 1;
                start_cycle = cycle;
            }
        } else if (st == 1) {
            if (cycle >= start_cycle + cycles_per_bit / 2) {
                if (tx == 0) {
                    st = 2;
                    start_cycle = start_cycle + cycles_per_bit;
                    bits_received = 0;
                    value = 0;
                } else {
                    st = 0;
                }
            }
        } else if (st == 2) {
            u32 bit_center = start_cycle + bits_received * cycles_per_bit + cycles_per_bit / 2;
            if (cycle >= bit_center) {
                value = value | (tx << bits_received);
                if (bits_received >= 7) {
                    st = 3;
                    start_cycle = start_cycle + 8 * cycles_per_bit;
                } else {
                    bits_received = bits_received + 1;
                }
            }
        } else if (st == 3) {
            if (cycle >= start_cycle + cycles_per_bit / 2) {
                if (tx == 1) {
                    u32 head = uart_channel->write_head;
                    u32 cap = uart_channel->capacity;
                    uart_channel->data[head % cap] = (uchar)(value & 0xFF);
                    uart_channel->write_head = head + 1;
                }
                st = 0;
            }
        }

        uart_state->state = st;
        uart_state->last_tx = tx;
        uart_state->start_cycle = start_cycle;
        uart_state->bits_received = bits_received;
        uart_state->value = value;
        uart_state->current_cycle = cycle + 1;
    }

    // ── Wishbone bus trace ──────────────────────────────────────────────
    if (wb_params.has_trace != 0) {
        u32 ibus_cyc = READ_OUT_BIT(wb_params.ibus_cyc_pos);
        u32 ibus_stb = READ_OUT_BIT(wb_params.ibus_stb_pos);
        u32 dbus_cyc = READ_OUT_BIT(wb_params.dbus_cyc_pos);
        u32 dbus_stb = READ_OUT_BIT(wb_params.dbus_stb_pos);
        u32 dbus_we  = READ_OUT_BIT(wb_params.dbus_we_pos);
        u32 spi_ack  = READ_OUT_BIT(wb_params.spiflash_ack_pos);
        u32 sram_ack = READ_OUT_BIT(wb_params.sram_ack_pos);
        u32 csr_ack  = READ_OUT_BIT(wb_params.csr_ack_pos);

        u32 flags = (ibus_cyc) | (ibus_stb << 1) | (dbus_cyc << 2) | (dbus_stb << 3)
                  | (dbus_we << 4) | (spi_ack << 5) | (sram_ack << 6) | (csr_ack << 7);

        u32 tick = wb_channel->current_tick;
        bool active = (ibus_cyc && ibus_stb) || (dbus_cyc && dbus_stb);
        bool changed = (flags != wb_channel->prev_flags);

        if (active || changed) {
            u32 ibus_adr = 0;
            for (u32 i = 0; i < WB_TRACE_MAX_ADR_BITS; i++) {
                ibus_adr |= READ_OUT_BIT(wb_params.ibus_adr_pos[i]) << i;
            }
            u32 ibus_rdata = 0;
            for (u32 i = 0; i < WB_TRACE_MAX_DAT_BITS; i++) {
                ibus_rdata |= READ_OUT_BIT(wb_params.ibus_rdata_pos[i]) << i;
            }
            u32 dbus_adr = 0;
            for (u32 i = 0; i < WB_TRACE_MAX_ADR_BITS; i++) {
                dbus_adr |= READ_OUT_BIT(wb_params.dbus_adr_pos[i]) << i;
            }

            u32 head = wb_channel->write_head;
            u32 cap = wb_channel->capacity;
            if (head < cap) {
                device WbTraceEntry* entries = (device WbTraceEntry*)((device uchar*)wb_channel + 16);
                device WbTraceEntry* e = &entries[head % cap];
                e->tick = tick;
                e->flags = flags;
                e->ibus_adr = ibus_adr;
                e->ibus_rdata = ibus_rdata;
                e->dbus_adr = dbus_adr;
                wb_channel->write_head = head + 1;
            }
        }

        wb_channel->prev_flags = flags;
        wb_channel->current_tick = tick + 1;
    }

    #undef READ_OUT_BIT
}
