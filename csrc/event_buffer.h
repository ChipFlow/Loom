// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Event buffer header for GPU->CPU communication of non-synthesizable constructs.
// This header is shared between CUDA and Metal implementations.

#ifndef EVENT_BUFFER_H
#define EVENT_BUFFER_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
typedef uint32_t u32;
typedef atomic_uint atomic_u32;
#else
#include <stdint.h>
typedef uint32_t u32;
#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)
typedef unsigned int atomic_u32;
#endif
#endif

// Maximum number of events that can be buffered per cycle
#define MAX_EVENTS 1024

// Event type constants (must match EventType enum in Rust)
#define EVENT_TYPE_STOP       0
#define EVENT_TYPE_FINISH     1
#define EVENT_TYPE_DISPLAY    2
#define EVENT_TYPE_ASSERT_FAIL 3
#define EVENT_TYPE_SETUP_VIOLATION 4
#define EVENT_TYPE_HOLD_VIOLATION  5

// A single event written by the GPU
struct Event {
    u32 event_type;     // Event type (see EVENT_TYPE_* constants) - uses u32 for alignment
    u32 message_id;     // Message ID for $display and assertions
    u32 cycle;          // Simulation cycle when event occurred
    u32 _reserved;      // Reserved for alignment
    u32 data[4];        // Data payload for $display format arguments
};

// The event buffer structure shared between GPU and CPU
// This must match the Rust EventBuffer layout exactly
struct EventBuffer {
    atomic_u32 count;      // Number of events (atomic for GPU writes)
    atomic_u32 overflow;   // Flag for overflow detection
    u32 _reserved[2];      // Padding for alignment
    struct Event events[MAX_EVENTS];
};

// Helper function to write an event to the buffer (GPU side)
#ifdef __METAL_VERSION__
inline void write_event(
    device struct EventBuffer* buffer,
    u32 event_type,
    u32 message_id,
    u32 cycle,
    u32 data0,
    u32 data1,
    u32 data2,
    u32 data3
) {
    u32 idx = atomic_fetch_add_explicit(&buffer->count, 1, memory_order_relaxed);
    if (idx < MAX_EVENTS) {
        buffer->events[idx].event_type = event_type;
        buffer->events[idx].message_id = message_id;
        buffer->events[idx].cycle = cycle;
        buffer->events[idx].data[0] = data0;
        buffer->events[idx].data[1] = data1;
        buffer->events[idx].data[2] = data2;
        buffer->events[idx].data[3] = data3;
    } else {
        atomic_store_explicit(&buffer->overflow, 1, memory_order_relaxed);
    }
}

// Simplified write for $stop/$finish (no message or data)
inline void write_sim_control_event(
    device struct EventBuffer* buffer,
    u32 event_type,
    u32 cycle
) {
    write_event(buffer, event_type, 0, cycle, 0, 0, 0, 0);
}
#endif

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)
__device__ inline void write_event(
    EventBuffer* buffer,
    u32 event_type,
    u32 message_id,
    u32 cycle,
    u32 data0,
    u32 data1,
    u32 data2,
    u32 data3
) {
    u32 idx = atomicAdd(&buffer->count, 1);
    if (idx < MAX_EVENTS) {
        buffer->events[idx].event_type = event_type;
        buffer->events[idx].message_id = message_id;
        buffer->events[idx].cycle = cycle;
        buffer->events[idx].data[0] = data0;
        buffer->events[idx].data[1] = data1;
        buffer->events[idx].data[2] = data2;
        buffer->events[idx].data[3] = data3;
    } else {
        atomicExch(&buffer->overflow, 1);
    }
}

__device__ inline void write_sim_control_event(
    EventBuffer* buffer,
    u32 event_type,
    u32 cycle
) {
    write_event(buffer, event_type, 0, cycle, 0, 0, 0, 0);
}
#endif

#endif // EVENT_BUFFER_H
