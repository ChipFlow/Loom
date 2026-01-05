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
    threadgroup u32* shared_metadata,
    threadgroup u32* shared_writeouts,
    threadgroup u32* shared_state
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
        for (int gr_i = 0; gr_i < num_global_read_rounds; gr_i += 2) {
            u32 idx = t2_1.c1;
            u32 mask = t2_1.c2;
            script_pi += 256 * 2;
            t2_2 = read_vec2(script + script_pi, tid);

            if (mask) {
                device const u32* real_input_array;
                if (idx >> 31) real_input_array = output_state - (1u << 31);
                else real_input_array = input_state;
                u32 value = real_input_array[idx];
                while (mask) {
                    t_global_rd_state <<= 1;
                    u32 lowbit = mask & -mask;
                    if (value & lowbit) t_global_rd_state |= 1;
                    mask ^= lowbit;
                }
            }

            if (gr_i + 1 >= num_global_read_rounds) break;
            idx = t2_2.c1;
            mask = t2_2.c2;
            script_pi += 256 * 2;
            t2_1 = read_vec2(script + script_pi, tid);

            if (mask) {
                device const u32* real_input_array;
                if (idx >> 31) real_input_array = output_state - (1u << 31);
                else real_input_array = input_state;
                u32 value = real_input_array[idx];
                while (mask) {
                    t_global_rd_state <<= 1;
                    u32 lowbit = mask & -mask;
                    if (value & lowbit) t_global_rd_state |= 1;
                    mask ^= lowbit;
                }
            }
        }
        shared_state[tid] = t_global_rd_state;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int bs_i = 0; bs_i < num_stages; ++bs_i) {
            u32 hier_input = 0, hier_flag_xora = 0, hier_flag_xorb = 0, hier_flag_orb = 0;

            // Macro-like inline expansion for shuffle input
            #define SHUF_INPUT_K(k_outer, k_inner, t_shuffle) { \
                u32 k = k_outer * 4 + k_inner; \
                u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1); \
                u32 t_shuffle_2_idx = t_shuffle >> 16; \
                hier_input |= (shared_state[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
                hier_input |= (shared_state[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
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

            threadgroup_barrier(mem_flags::mem_threadgroup);
            shared_state[tid] = hier_input;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // hier[0]
            if (tid >= 128) {
                u32 hier_input_a = shared_state[tid - 128];
                u32 hier_input_b = hier_input;
                u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
                shared_state[tid] = ret;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // hier[1..3]
            u32 tmp_cur_hi = 0;
            for (int hi = 1; hi <= 3; ++hi) {
                int hier_width = 1 << (7 - hi);
                if (tid >= (uint)hier_width && tid < (uint)(hier_width * 2)) {
                    u32 hier_input_a = shared_state[tid + hier_width];
                    u32 hier_input_b = shared_state[tid + hier_width * 2];
                    u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
                    tmp_cur_hi = ret;
                    shared_state[tid] = ret;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // hier[4..7], within the first SIMD group (32 threads)
            if (tid < 32) {
                for (int hi = 4; hi <= 7; ++hi) {
                    int hier_width = 1 << (7 - hi);
                    u32 hier_input_a = simd_shuffle_down(tmp_cur_hi, hier_width);
                    u32 hier_input_b = simd_shuffle_down(tmp_cur_hi, hier_width * 2);
                    if (tid >= (uint)hier_width && tid < (uint)(hier_width * 2)) {
                        tmp_cur_hi = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
                    }
                }
                u32 v1 = simd_shuffle_down(tmp_cur_hi, 1);
                // hier[8..12]
                if (tid == 0) {
                    u32 r8 = ((v1 << 16) ^ hier_flag_xora) & ((v1 ^ hier_flag_xorb) | hier_flag_orb) & 0xffff0000;
                    u32 r9 = ((r8 >> 8) ^ hier_flag_xora) & (((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb) & 0xff00;
                    u32 r10 = ((r9 >> 4) ^ hier_flag_xora) & (((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb) & 0xf0;
                    u32 r11 = ((r10 >> 2) ^ hier_flag_xora) & (((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb) & 12;
                    u32 r12 = ((r11 >> 1) ^ hier_flag_xora) & (((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb) & 2;
                    tmp_cur_hi = r8 | r9 | r10 | r11 | r12;
                }
                shared_state[tid] = tmp_cur_hi;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // write out
            if ((writeout_hook_i >> 8) == (uint)bs_i) {
                shared_writeouts[tid] = shared_state[writeout_hook_i & 255];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // sram & duplicate permutation
        u32 sram_duplicate_t = 0;

        #define SHUF_SRAM_DUPL_K(k_outer, k_inner, t_shuffle) { \
            u32 k = k_outer * 4 + k_inner; \
            u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1); \
            u32 t_shuffle_2_idx = t_shuffle >> 16; \
            sram_duplicate_t |= (shared_writeouts[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
            sram_duplicate_t |= (shared_writeouts[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
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
        t4_5 = read_vec4(script + script_pi + 256 * 4 * 4, tid);

        // sram read
        device u32* ram = nullptr;
        u32 r = 0, w0 = 0;
        u32 port_w_addr_iv = 0, port_w_wr_en = 0, port_w_wr_data_iv = 0;

        if (tid < (uint)(num_srams * 4)) {
            u32 addrs = sram_duplicate_t;
            // SIMD shuffle for SRAM operations
            port_w_wr_en = simd_shuffle_down(sram_duplicate_t, 1);
            port_w_wr_data_iv = simd_shuffle_down(sram_duplicate_t, 2);

            if (tid % 4 == 0) {
                u32 sram_i = tid / 4;
                u32 sram_st = sram_offset + sram_i * (1 << 13);
                u32 port_r_addr_iv = addrs & 0xffff;
                port_w_addr_iv = addrs >> 16;

                ram = sram_data + sram_st;
                r = ram[port_r_addr_iv];
                w0 = ram[port_w_addr_iv];
            }
        }

        // clock enable permutation
        u32 clken_perm = 0;

        #define SHUF_CLKEN_K(k_outer, k_inner, t_shuffle) { \
            u32 k = k_outer * 4 + k_inner; \
            u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1); \
            u32 t_shuffle_2_idx = t_shuffle >> 16; \
            clken_perm |= (shared_writeouts[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) << (k * 2); \
            clken_perm |= (shared_writeouts[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
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
            }
        } else if (tid < (uint)(num_srams * 4 + num_output_duplicates)) {
            shared_writeouts[num_ios - num_srams - num_output_duplicates + (tid - num_srams * 4)] = sram_duplicate_t;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        u32 writeout_inv = shared_writeouts[tid];

        clken_perm = (clken_perm & ~t4_5.c2) ^ t4_5.c1;
        writeout_inv ^= t4_5.c3;

        if (tid < (uint)num_ios) {
            u32 old_wo = input_state[io_offset + tid];
            u32 wo = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
            output_state[io_offset + tid] = wo;
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
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Threadgroup shared memory (equivalent to CUDA __shared__)
    threadgroup u32 shared_metadata[256];
    threadgroup u32 shared_writeouts[256];
    threadgroup u32 shared_state[256];

    usize stage_i = params.current_stage;
    usize cycle_i = params.current_cycle;

    // Get script location for this block and stage
    usize script_start = blocks_start[stage_i * params.num_blocks + gid];
    usize script_end = blocks_start[stage_i * params.num_blocks + gid + 1];
    usize script_size = script_end - script_start;

    device const u32* script = blocks_data + script_start;
    device const u32* input_state = states_noninteractive + cycle_i * params.state_size;
    device u32* output_state = states_noninteractive + (cycle_i + 1) * params.state_size;

    simulate_block_v1(
        tid,
        script,
        script_size,
        input_state,
        output_state,
        sram_data,
        shared_metadata,
        shared_writeouts,
        shared_state
    );

    // TODO: Process simulation control nodes from the script
    // When the script includes SimControl data, event writing will happen here.
    // For now, the event_buffer parameter is available but unused.
    // Example usage when implemented:
    //   if (simcontrol_condition_met && tid == 0) {
    //       write_sim_control_event(event_buffer, EVENT_TYPE_STOP, (u32)cycle_i);
    //   }
}
