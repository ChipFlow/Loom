// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <crates/ulib/includes.hpp>
#include <cstdio>
#include <cooperative_groups.h>
#include "event_buffer.h"

struct alignas(8) VectorRead2 {
  u32 c1, c2;

  __device__ __forceinline__ void read(const VectorRead2 *t) {
    *this = *t;
  }
};

struct alignas(16) VectorRead4 {
  u32 c1, c2, c3, c4;

  __device__ __forceinline__ void read(const VectorRead4 *t) {
    *this = *t;
  }
};

__device__ void simulate_block_v1(
  const u32 *__restrict__ script,
  usize script_size,
  const u32 *__restrict__ input_state,
  u32 *__restrict__ output_state,
  u32 *__restrict__ sram_data,
  u32 *__restrict__ sram_xmask,  // X-mask shadow for SRAM (nullptr if xprop disabled)
  u32 *__restrict__ shared_metadata,
  u32 *__restrict__ shared_writeouts,
  u32 *__restrict__ shared_state,
  u32 *__restrict__ shared_writeouts_x,  // X-mask sideband for writeouts
  u32 *__restrict__ shared_state_x,      // X-mask sideband for state
  // Arrival times in raw picoseconds (u16), max representable 65,535ps.
  // Gate delays are stored as u16 in the script padding slots.
  // No quantization needed — 512 bytes of shared memory per block is negligible.
  u16 *__restrict__ shared_arrival,
  // Writeout arrival times captured during writeout hook
  u16 *__restrict__ shared_writeout_arrival,
  // Timing constraint buffer: one u32 per state word, [setup_ps:16][hold_ps:16]
  const u32 *__restrict__ timing_constraints,
  u32 clock_period_ps,
  EventBuffer *__restrict__ event_buffer,
  u32 cycle_i,
  int arrival_state_offset  // offset in output_state for arrival data (0 = disabled)
  )
{
  int script_pi = 0;
  while(true) {
    VectorRead2 t2_1, t2_2;
    VectorRead4 t4_1, t4_2, t4_3, t4_4, t4_5;
    shared_metadata[threadIdx.x] = script[script_pi + threadIdx.x];
    script_pi += 256;
    t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
    __syncthreads();
    int num_stages = shared_metadata[0];
    if(!num_stages) {
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
    u32 writeout_hook_i = shared_metadata[128 + threadIdx.x / 2];
    if(threadIdx.x % 2 == 0) {
      writeout_hook_i = writeout_hook_i & ((1 << 16) - 1);
    }
    else {
      writeout_hook_i = writeout_hook_i >> 16;
    }

    t4_1.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds) + threadIdx.x);
    t4_2.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4) + threadIdx.x);
    t4_3.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 2) + threadIdx.x);
    t4_4.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 3) + threadIdx.x);
    t4_5.read((const VectorRead4 *)(script + script_pi + 256 * 2 * num_global_read_rounds + 256 * 4 * 4) + threadIdx.x);
    u32 t_global_rd_state = 0;
    u32 t_global_rd_state_x = 0;
    for(int gr_i = 0; gr_i < num_global_read_rounds; gr_i += 2) {
      u32 idx = t2_1.c1;
      u32 mask = t2_1.c2;
      script_pi += 256 * 2;
      t2_2.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if(mask) {
        const u32 *real_input_array;
        const u32 *real_xmask_array;
        if(idx >> 31) {
          real_input_array = output_state - (1 << 31);
          real_xmask_array = output_state + xmask_state_offset - (1 << 31);
        } else {
          real_input_array = input_state;
          real_xmask_array = input_state + xmask_state_offset;
        }
        u32 value = real_input_array[idx];
        u32 xmval = is_x_capable ? real_xmask_array[idx] : 0;
        u32 m = mask;
        while(m) {
          t_global_rd_state <<= 1;
          t_global_rd_state_x <<= 1;
          u32 lowbit = m & -m;
          if(value & lowbit) t_global_rd_state |= 1;
          if(xmval & lowbit) t_global_rd_state_x |= 1;
          m ^= lowbit;
        }
      }

      if(gr_i + 1 >= num_global_read_rounds) break;
      idx = t2_2.c1;
      mask = t2_2.c2;
      script_pi += 256 * 2;
      t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if(mask) {
        const u32 *real_input_array;
        const u32 *real_xmask_array;
        if(idx >> 31) {
          real_input_array = output_state - (1 << 31);
          real_xmask_array = output_state + xmask_state_offset - (1 << 31);
        } else {
          real_input_array = input_state;
          real_xmask_array = input_state + xmask_state_offset;
        }
        u32 value = real_input_array[idx];
        u32 xmval = is_x_capable ? real_xmask_array[idx] : 0;
        u32 m = mask;
        while(m) {
          t_global_rd_state <<= 1;
          t_global_rd_state_x <<= 1;
          u32 lowbit = m & -m;
          if(value & lowbit) t_global_rd_state |= 1;
          if(xmval & lowbit) t_global_rd_state_x |= 1;
          m ^= lowbit;
        }
      }
    }
    shared_state[threadIdx.x] = t_global_rd_state;
    shared_state_x[threadIdx.x] = t_global_rd_state_x;
    // Initialize arrival times to 0 (inputs have zero arrival)
    shared_arrival[threadIdx.x] = 0;
    __syncthreads();

    for(int bs_i = 0; bs_i < num_stages; ++bs_i) {
      u32 hier_input = 0, hier_input_x = 0;
      u32 hier_flag_xora = 0, hier_flag_xorb = 0, hier_flag_orb = 0;
#define GEMV1_SHUF_INPUT_K(k_outer, k_inner, t_shuffle) {           \
        u32 k = k_outer * 4 + k_inner;                              \
        u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);          \
        u32 t_shuffle_2_idx = t_shuffle >> 16;                      \
                                                                    \
        hier_input |= (shared_state[t_shuffle_1_idx >> 5] >>        \
                       (t_shuffle_1_idx & 31) & 1) << (k * 2);      \
        hier_input |= (shared_state[t_shuffle_2_idx >> 5] >>        \
                       (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);  \
        if(is_x_capable) {                                          \
          hier_input_x |= (shared_state_x[t_shuffle_1_idx >> 5] >>  \
                           (t_shuffle_1_idx & 31) & 1) << (k * 2);  \
          hier_input_x |= (shared_state_x[t_shuffle_2_idx >> 5] >>  \
                           (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1); \
        }                                                           \
      }
#define GEMV1_SHUF_INPUT_K_4(k_outer, t_shuffle) {    \
        GEMV1_SHUF_INPUT_K(k_outer, 0, t_shuffle.c1); \
        GEMV1_SHUF_INPUT_K(k_outer, 1, t_shuffle.c2); \
        GEMV1_SHUF_INPUT_K(k_outer, 2, t_shuffle.c3); \
        GEMV1_SHUF_INPUT_K(k_outer, 3, t_shuffle.c4); \
      }
      script_pi += 256 * 4 * 5;
      GEMV1_SHUF_INPUT_K_4(0, t4_1);
      t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(1, t4_2);
      t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(2, t4_3);
      t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(3, t4_4);
      t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) + threadIdx.x);
#undef GEMV1_SHUF_INPUT_K
#undef GEMV1_SHUF_INPUT_K_4
      hier_flag_xora = t4_5.c1;
      hier_flag_xorb = t4_5.c2;
      hier_flag_orb = t4_5.c3;
      // Extract per-thread-position gate delay from padding slot (u16 raw picoseconds)
      u16 gate_delay = (u16)(t4_5.c4 & 0xFFFF);
      t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) + threadIdx.x);

      __syncthreads();
      shared_state[threadIdx.x] = hier_input;
      shared_state_x[threadIdx.x] = hier_input_x;
      shared_arrival[threadIdx.x] = 0;  // Reset arrival for shuffle inputs
      __syncthreads();

      // hier[0]: threads 128-255 compute AND gates + track arrivals
      if(threadIdx.x >= 128) {
        u32 hier_input_a = shared_state[threadIdx.x - 128];
        u32 hier_input_b = hier_input;
        u32 a_eff = hier_input_a ^ hier_flag_xora;
        u32 b_eff = (hier_input_b ^ hier_flag_xorb) | hier_flag_orb;
        u32 ret = a_eff & b_eff;
        shared_state[threadIdx.x] = ret;

        if(is_x_capable) {
          u32 a_x = shared_state_x[threadIdx.x - 128];
          u32 b_x = hier_input_x;
          u32 b_eff_x = b_x & ~hier_flag_orb;
          u32 ret_x = (a_x | b_eff_x) & (a_eff | a_x) & (b_eff | b_eff_x);
          shared_state_x[threadIdx.x] = ret_x;
        }

        // Arrival tracking: max(input_a, input_b) + gate_delay
        // Delay added even for pass-throughs (physical cells with accumulated delays)
        u16 arr_a = (u16)shared_arrival[threadIdx.x - 128];
        u16 arr_b = (u16)shared_arrival[threadIdx.x];
        bool is_pass = (hier_flag_orb == 0xFFFFFFFF);
        u16 base_arr = is_pass ? arr_a : (u16)max(arr_a, arr_b);
        u16 new_arr = (u16)(base_arr + (u16)gate_delay);
        shared_arrival[threadIdx.x] = new_arr;
      }
      __syncthreads();
      // hier[1..3]: shared memory reduction + arrival tracking
      u32 tmp_cur_hi;
      u32 tmp_cur_hi_x = 0;
      u16 tmp_cur_arr = 0;
      for(int hi = 1; hi <= 3; ++hi) {
        int hier_width = 1 << (7 - hi);
        if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
          u32 hier_input_a = shared_state[threadIdx.x + hier_width];
          u32 hier_input_b = shared_state[threadIdx.x + hier_width * 2];
          u32 a_eff = hier_input_a ^ hier_flag_xora;
          u32 b_eff = (hier_input_b ^ hier_flag_xorb) | hier_flag_orb;
          u32 ret = a_eff & b_eff;
          tmp_cur_hi = ret;
          shared_state[threadIdx.x] = ret;

          if(is_x_capable) {
            u32 a_x = shared_state_x[threadIdx.x + hier_width];
            u32 b_x = shared_state_x[threadIdx.x + hier_width * 2];
            u32 b_eff_x = b_x & ~hier_flag_orb;
            u32 ret_x = (a_x | b_eff_x) & (a_eff | a_x) & (b_eff | b_eff_x);
            tmp_cur_hi_x = ret_x;
            shared_state_x[threadIdx.x] = ret_x;
          }

          // Arrival tracking (delay added even for pass-throughs)
          u16 arr_a = (u16)shared_arrival[threadIdx.x + hier_width];
          u16 arr_b = (u16)shared_arrival[threadIdx.x + hier_width * 2];
          bool is_pass = (hier_flag_orb == 0xFFFFFFFF);
          u16 base_arr = is_pass ? arr_a : (u16)max(arr_a, arr_b);
          u16 new_arr = (u16)(base_arr + (u16)gate_delay);
          tmp_cur_arr = new_arr;
          shared_arrival[threadIdx.x] = tmp_cur_arr;
        }
        __syncthreads();
      }
      // hier[4..7], within the first warp.
      // Pack arrival into u32 for warp shuffle
      u32 tmp_cur_arr_u32 = (u32)tmp_cur_arr;
      if(threadIdx.x < 32) {
        for(int hi = 4; hi <= 7; ++hi) {
          int hier_width = 1 << (7 - hi);
          u32 hier_input_a = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width);
          u32 hier_input_b = __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width * 2);
          u32 hier_a_x = is_x_capable ? __shfl_down_sync(0xffffffff, tmp_cur_hi_x, hier_width) : 0;
          u32 hier_b_x = is_x_capable ? __shfl_down_sync(0xffffffff, tmp_cur_hi_x, hier_width * 2) : 0;
          u32 arr_a_u32 = __shfl_down_sync(0xffffffff, tmp_cur_arr_u32, hier_width);
          u32 arr_b_u32 = __shfl_down_sync(0xffffffff, tmp_cur_arr_u32, hier_width * 2);
          if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
            u32 a_eff = hier_input_a ^ hier_flag_xora;
            u32 b_eff = (hier_input_b ^ hier_flag_xorb) | hier_flag_orb;
            tmp_cur_hi = a_eff & b_eff;
            if(is_x_capable) {
              u32 b_eff_x = hier_b_x & ~hier_flag_orb;
              tmp_cur_hi_x = (hier_a_x | b_eff_x) & (a_eff | hier_a_x) & (b_eff | b_eff_x);
            }
            // Arrival tracking (delay added even for pass-throughs)
            bool is_pass = (hier_flag_orb == 0xFFFFFFFF);
            u16 base_arr = is_pass ? (u16)arr_a_u32 : (u16)max((u16)arr_a_u32, (u16)arr_b_u32);
            u16 new_arr = (u16)(base_arr + (u16)gate_delay);
            tmp_cur_arr_u32 = (u32)new_arr;
          }
        }
        u32 v1 = __shfl_down_sync(0xffffffff, tmp_cur_hi, 1);
        u32 v1_x = is_x_capable ? __shfl_down_sync(0xffffffff, tmp_cur_hi_x, 1) : 0;
        // hier[8..12]: bit-level operations within single u32
        // All 32 signals share one thread's arrival — arrival carries forward unchanged
        if(threadIdx.x == 0) {
          // Value lane
          u32 r8 = ((v1 << 16) ^ hier_flag_xora) & ((v1 ^ hier_flag_xorb) | hier_flag_orb) & 0xffff0000;
          u32 r9 = ((r8 >> 8) ^ hier_flag_xora) & (((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb) & 0xff00;
          u32 r10 = ((r9 >> 4) ^ hier_flag_xora) & (((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb) & 0xf0;
          u32 r11 = ((r10 >> 2) ^ hier_flag_xora) & (((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb) & 12;
          u32 r12 = ((r11 >> 1) ^ hier_flag_xora) & (((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb) & 2;
          tmp_cur_hi = r8 | r9 | r10 | r11 | r12;

          if(is_x_capable) {
            // X-mask lane: same structure with X-prop AND formula
            u32 r8_x, r9_x, r10_x, r11_x, r12_x;
            {
              u32 ae = (v1 << 16) ^ hier_flag_xora;
              u32 be = (v1 ^ hier_flag_xorb) | hier_flag_orb;
              u32 ax = v1_x << 16;
              u32 bex = v1_x & ~hier_flag_orb;
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
          }
        }
        shared_state[threadIdx.x] = tmp_cur_hi;
        shared_state_x[threadIdx.x] = tmp_cur_hi_x;
        shared_arrival[threadIdx.x] = (u16)tmp_cur_arr_u32;
      }
      __syncthreads();

      // write out
      if((writeout_hook_i >> 8) == bs_i) {
        shared_writeouts[threadIdx.x] = shared_state[writeout_hook_i & 255];
        if(is_x_capable) {
          shared_writeouts_x[threadIdx.x] = shared_state_x[writeout_hook_i & 255];
        }
        shared_writeout_arrival[threadIdx.x] = shared_arrival[writeout_hook_i & 255];
      }
    }
    __syncthreads();

    // sram & duplicate permutation
    u32 sram_duplicate_t = 0;
    u32 sram_duplicate_t_x = 0;
#define GEMV1_SHUF_SRAM_DUPL_K(k_outer, k_inner, t_shuffle) { \
      u32 k = k_outer * 4 + k_inner;                          \
      u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);      \
      u32 t_shuffle_2_idx = t_shuffle >> 16;                  \
                                                              \
      sram_duplicate_t |=                                     \
        (shared_writeouts[t_shuffle_1_idx >> 5] >>            \
         (t_shuffle_1_idx & 31) & 1) << (k * 2);              \
      sram_duplicate_t |=                                     \
        (shared_writeouts[t_shuffle_2_idx >> 5] >>            \
         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);          \
      if(is_x_capable) {                                      \
        sram_duplicate_t_x |=                                 \
          (shared_writeouts_x[t_shuffle_1_idx >> 5] >>        \
           (t_shuffle_1_idx & 31) & 1) << (k * 2);            \
        sram_duplicate_t_x |=                                 \
          (shared_writeouts_x[t_shuffle_2_idx >> 5] >>        \
           (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);        \
      }                                                       \
    }
#define GEMV1_SHUF_SRAM_DUPL_K_4(k_outer, t_shuffle) {  \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 0, t_shuffle.c1); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 1, t_shuffle.c2); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 2, t_shuffle.c3); \
      GEMV1_SHUF_SRAM_DUPL_K(k_outer, 3, t_shuffle.c4); \
    }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_SRAM_DUPL_K_4(0, t4_1);
    t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(1, t4_2);
    t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(2, t4_3);
    t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(3, t4_4);
    t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) + threadIdx.x);
#undef GEMV1_SHUF_SRAM_DUPL_K_4
#undef GEMV1_SHUF_SRAM_DUPL_K
    sram_duplicate_t = (sram_duplicate_t & ~t4_5.c2) ^ t4_5.c1;
    if(is_x_capable) {
      sram_duplicate_t_x = sram_duplicate_t_x & ~t4_5.c2; // set0 clears X too
    }
    t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) + threadIdx.x);

    // sram read fires here.
    u32 *ram = nullptr;
    u32 *ram_x = nullptr;
    u32 r = 0, w0 = 0, r_x = 0, w0_x = 0;
    u32 port_w_addr_iv, port_w_wr_en, port_w_wr_data_iv;
    u32 port_w_wr_data_x = 0;
    if(threadIdx.x < num_srams * 4) {
      u32 addrs = sram_duplicate_t;
      u32 last_tid = 32 + threadIdx.x / 32 * 32;
      u32 mask = (last_tid <= num_srams * 4)
        ? 0xffffffff : (0xffffffff >> (last_tid - num_srams * 4));
      port_w_wr_en = __shfl_down_sync(mask, sram_duplicate_t, 1);
      port_w_wr_data_iv = __shfl_down_sync(mask, sram_duplicate_t, 2);
      if(is_x_capable) {
        port_w_wr_data_x = __shfl_down_sync(mask, sram_duplicate_t_x, 2);
      }

      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        u32 sram_st = sram_offset + sram_i * (1 << 13);
        u32 port_r_addr_iv = addrs & 0xffff;
        port_w_addr_iv = addrs >> 16;

        ram = sram_data + sram_st;
        r = ram[port_r_addr_iv];
        w0 = ram[port_w_addr_iv];
        if(is_x_capable && sram_xmask != nullptr) {
          ram_x = sram_xmask + sram_st;
          r_x = ram_x[port_r_addr_iv];
          w0_x = ram_x[port_w_addr_iv];
        }
      }
    }
    // __syncthreads();

    // clock enable permutation
    u32 clken_perm = 0;
    u32 clken_perm_x = 0;
#define GEMV1_SHUF_CLKEN_K(k_outer, k_inner, t_shuffle) { \
      u32 k = k_outer * 4 + k_inner;                      \
      u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);  \
      u32 t_shuffle_2_idx = t_shuffle >> 16;              \
                                                          \
      clken_perm |=                                       \
        (shared_writeouts[t_shuffle_1_idx >> 5] >>        \
         (t_shuffle_1_idx & 31) & 1) << (k * 2);          \
      clken_perm |=                                       \
        (shared_writeouts[t_shuffle_2_idx >> 5] >>        \
         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);      \
      if(is_x_capable) {                                  \
        clken_perm_x |=                                   \
          (shared_writeouts_x[t_shuffle_1_idx >> 5] >>    \
           (t_shuffle_1_idx & 31) & 1) << (k * 2);        \
        clken_perm_x |=                                   \
          (shared_writeouts_x[t_shuffle_2_idx >> 5] >>    \
           (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);    \
      }                                                   \
    }
#define GEMV1_SHUF_CLKEN_K_4(k_outer, t_shuffle) {  \
      GEMV1_SHUF_CLKEN_K(k_outer, 0, t_shuffle.c1); \
      GEMV1_SHUF_CLKEN_K(k_outer, 1, t_shuffle.c2); \
      GEMV1_SHUF_CLKEN_K(k_outer, 2, t_shuffle.c3); \
      GEMV1_SHUF_CLKEN_K(k_outer, 3, t_shuffle.c4); \
    }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_CLKEN_K_4(0, t4_1);
    GEMV1_SHUF_CLKEN_K_4(1, t4_2);
    GEMV1_SHUF_CLKEN_K_4(2, t4_3);
    GEMV1_SHUF_CLKEN_K_4(3, t4_4);
#undef GEMV1_SHUF_CLKEN_K
#undef GEMV1_SHUF_CLKEN_K_4

    // sram commit
    if(threadIdx.x < num_srams * 4) {
      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        shared_writeouts[num_ios - num_srams + sram_i] = r;
        ram[port_w_addr_iv] = (w0 & ~port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
        if(is_x_capable) {
          shared_writeouts_x[num_ios - num_srams + sram_i] = r_x;
          if(ram_x != nullptr) {
            ram_x[port_w_addr_iv] = (w0_x & ~port_w_wr_en) | (port_w_wr_data_x & port_w_wr_en);
          }
        }
      }
    }
    else if(threadIdx.x < num_srams * 4 + num_output_duplicates) {
      shared_writeouts[num_ios - num_srams - num_output_duplicates + (threadIdx.x - num_srams * 4)] = sram_duplicate_t;
      if(is_x_capable) {
        shared_writeouts_x[num_ios - num_srams - num_output_duplicates + (threadIdx.x - num_srams * 4)] = sram_duplicate_t_x;
      }
    }

    __syncthreads();
    u32 writeout_inv = shared_writeouts[threadIdx.x];
    u32 writeout_inv_x = is_x_capable ? shared_writeouts_x[threadIdx.x] : 0;

    clken_perm = (clken_perm & ~t4_5.c2) ^ t4_5.c1;
    if(is_x_capable) {
      clken_perm_x = clken_perm_x & ~t4_5.c2; // set0 clears X
    }
    writeout_inv ^= t4_5.c3;
    // data_inv (t4_5.c3) doesn't affect X-mask (inversion preserves X)

    if(threadIdx.x < num_ios) {
      u32 old_wo = input_state[io_offset + threadIdx.x];
      u32 wo = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
      output_state[io_offset + threadIdx.x] = wo;

      if(is_x_capable) {
        u32 old_x = input_state[xmask_state_offset + io_offset + threadIdx.x];
        // X-mask DFF gating: X clken → output is X
        u32 wo_x = (old_x & ~clken_perm & ~clken_perm_x)
                 | (writeout_inv_x & clken_perm & ~clken_perm_x)
                 | clken_perm_x;
        output_state[xmask_state_offset + io_offset + threadIdx.x] = wo_x;
      }

      // Write arrival time to global memory for timed VCD output
      if(arrival_state_offset != 0) {
        output_state[arrival_state_offset + io_offset + threadIdx.x] =
            (u32)shared_writeout_arrival[threadIdx.x];
      }
    }

    // DFF timing violation check (per writeout word)
    if(threadIdx.x < num_ios && clken_perm != 0 && timing_constraints != nullptr) {
      u32 constraint = timing_constraints[io_offset + threadIdx.x];
      if(constraint != 0) {
        u16 setup_ps = (u16)(constraint >> 16);
        u16 hold_ps = (u16)(constraint & 0xFFFF);
        u16 arrival = shared_writeout_arrival[threadIdx.x];
        // Setup: arrival + setup must fit within clock period
        if(arrival > 0 && (u32)arrival + (u32)setup_ps > clock_period_ps) {
          int slack = (int)clock_period_ps - (int)arrival - (int)setup_ps;
          write_event(event_buffer, EVENT_TYPE_SETUP_VIOLATION,
                     io_offset + threadIdx.x, cycle_i,
                     io_offset + threadIdx.x, (u32)slack, (u32)arrival, (u32)setup_ps);
        }
        // Hold: arrival must exceed hold time
        if(arrival < hold_ps) {
          int slack = (int)arrival - (int)hold_ps;
          write_event(event_buffer, EVENT_TYPE_HOLD_VIOLATION,
                     io_offset + threadIdx.x, cycle_i,
                     io_offset + threadIdx.x, (u32)slack, (u32)arrival, (u32)hold_ps);
        }
      }
    }

    if(is_last_part) break;
  }
  assert(script_size == script_pi);
}

__global__ void simulate_v1_noninteractive_simple_scan(
  usize num_blocks,
  usize num_major_stages,
  const usize *__restrict__ blocks_start,
  const u32 *__restrict__ blocks_data,
  u32 *__restrict__ sram_data,
  u32 *__restrict__ sram_xmask,
  usize num_cycles,
  usize state_size,
  u32 *__restrict__ states_noninteractive,
  const u32 *__restrict__ timing_constraints,
  EventBuffer *__restrict__ event_buffer
  )
{
  assert(num_blocks == gridDim.x);
  assert(256 == blockDim.x);
  __shared__ u32 shared_metadata[256];
  __shared__ u32 shared_writeouts[256];
  __shared__ u32 shared_state[256];
  __shared__ u32 shared_writeouts_x[256];  // X-mask sideband for writeouts
  __shared__ u32 shared_state_x[256];      // X-mask sideband for state
  __shared__ u16 shared_arrival[256];  // Per-thread-position arrival times (raw picoseconds)
  __shared__ u16 shared_writeout_arrival[256];  // Arrival times captured at writeout
  shared_writeout_arrival[threadIdx.x] = 0;  // Must initialize — only writeout threads update this
  shared_writeouts_x[threadIdx.x] = 0;
  shared_state_x[threadIdx.x] = 0;
  __shared__ u32 script_starts[32], script_sizes[32];
  assert(num_major_stages <= 32);

  // Read clock period from first element of constraints buffer
  u32 clock_period_ps = (timing_constraints != nullptr) ? timing_constraints[0] : 0;
  const u32* constraints_data = (timing_constraints != nullptr) ? timing_constraints + 1 : nullptr;

  if(threadIdx.x < num_major_stages) {
    script_starts[threadIdx.x] = blocks_start[threadIdx.x * num_blocks + blockIdx.x];
    script_sizes[threadIdx.x] = blocks_start[threadIdx.x * num_blocks + blockIdx.x + 1] - script_starts[threadIdx.x];
  }
  __syncthreads();
  for(usize cycle_i = 0; cycle_i < num_cycles; ++cycle_i) {
    for(usize stage_i = 0; stage_i < num_major_stages; ++stage_i) {
      simulate_block_v1(
        blocks_data + script_starts[stage_i],
        script_sizes[stage_i],
        states_noninteractive + cycle_i * state_size,
        states_noninteractive + (cycle_i + 1) * state_size,
        sram_data,
        sram_xmask,
        shared_metadata, shared_writeouts, shared_state,
        shared_writeouts_x, shared_state_x,
        shared_arrival,
        shared_writeout_arrival,
        constraints_data,
        clock_period_ps,
        event_buffer,
        (u32)cycle_i,
        0  // arrival_state_offset: CUDA timing VCD not yet supported
        );
      cooperative_groups::this_grid().sync();
    }
  }
}
