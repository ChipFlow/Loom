// SPDX-License-Identifier: Apache-2.0
// HIP kernel launch wrapper for AMD GPUs.
// Mirrors kernel_v1.cu — uses hipLaunchCooperativeKernel for grid-wide sync.

#include <hip/hip_runtime.h>
#include "kernel_v1_impl.cuh"

#define checkHipErrors(call)                                    \
  do {                                                          \
    hipError_t err = call;                                      \
    if (err != hipSuccess) {                                    \
      printf("HIP error at %s %d: %s\n", __FILE__, __LINE__,   \
             hipGetErrorString(err));                            \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  } while (0)

// Original function without timing support (backward compatible).
extern "C"
void simulate_v1_noninteractive_simple_scan_hip(
  usize num_blocks,
  usize num_major_stages,
  const usize *blocks_start,
  const u32 *blocks_data,
  u32 *sram_data,
  u32 *sram_xmask,
  usize num_cycles,
  usize state_size,
  u32 *states_noninteractive
  )
{
  // Runtime warp size assertion — RDNA uses wave32, matching CUDA.
  int warp_size = 0;
  hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0);
  if (warp_size != 32) {
    printf("ERROR: Loom requires warpSize==32, but this GPU reports %d\n", warp_size);
    exit(EXIT_FAILURE);
  }

  const u32 *timing_constraints = nullptr;
  EventBuffer *event_buffer = nullptr;
  void *arg_ptrs[11] = {
    (void *)&num_blocks, (void *)&num_major_stages,
    (void *)&blocks_start, (void *)&blocks_data,
    (void *)&sram_data, (void *)&sram_xmask,
    (void *)&num_cycles, (void *)&state_size,
    (void *)&states_noninteractive,
    (void *)&timing_constraints, (void *)&event_buffer
  };
  checkHipErrors(hipLaunchCooperativeKernel(
    (void *)simulate_v1_noninteractive_simple_scan,
    dim3(num_blocks), dim3(256),
    arg_ptrs, 0, (hipStream_t)0
    ));
}

// Extended function with timing constraints and event buffer support.
extern "C"
void simulate_v1_noninteractive_timed_hip(
  usize num_blocks,
  usize num_major_stages,
  const usize *blocks_start,
  const u32 *blocks_data,
  u32 *sram_data,
  u32 *sram_xmask,
  usize num_cycles,
  usize state_size,
  u32 *states_noninteractive,
  const u32 *timing_constraints,
  u8 *event_buffer
  )
{
  // Runtime warp size assertion — RDNA uses wave32, matching CUDA.
  int warp_size = 0;
  hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0);
  if (warp_size != 32) {
    printf("ERROR: Loom requires warpSize==32, but this GPU reports %d\n", warp_size);
    exit(EXIT_FAILURE);
  }

  void *arg_ptrs[11] = {
    (void *)&num_blocks, (void *)&num_major_stages,
    (void *)&blocks_start, (void *)&blocks_data,
    (void *)&sram_data, (void *)&sram_xmask,
    (void *)&num_cycles, (void *)&state_size,
    (void *)&states_noninteractive,
    (void *)&timing_constraints, (void *)&event_buffer
  };
  checkHipErrors(hipLaunchCooperativeKernel(
    (void *)simulate_v1_noninteractive_simple_scan,
    dim3(num_blocks), dim3(256),
    arg_ptrs, 0, (hipStream_t)0
    ));
}
