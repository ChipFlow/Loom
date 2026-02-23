// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Benchmarks for X-propagation CPU kernel overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use gem::sim::cpu_reference::{simulate_block_v1, simulate_block_v1_xprop};

/// Build a partition script with realistic IO count and multiple boomerang stages.
fn build_bench_script(is_x_capable: bool, num_ios: u32, num_stages: u32) -> Vec<u32> {
    let num_gr_rounds: u32 = 1;
    let num_srams: u32 = 0;
    let num_output_duplicates: u32 = 0;
    let io_offset: u32 = 0;
    let sram_offset: u32 = 0;
    let xmask_state_offset: u32 = num_ios;

    // Per stage: boomerang(5120)
    // Total: metadata(256) + global_read(512) + stages * boomerang(5120) + sram_dup(5120) + clken(5120) + dummy(256)
    let total_size = 256 + 512 + (num_stages as usize) * 5120 + 5120 + 5120 + 256;
    let mut script = vec![0u32; total_size];

    // Metadata
    script[0] = num_stages;
    script[1] = 0; // not last
    script[2] = num_ios;
    script[3] = io_offset;
    script[4] = num_srams;
    script[5] = sram_offset;
    script[6] = num_gr_rounds;
    script[7] = num_output_duplicates;
    script[8] = if is_x_capable { 1 } else { 0 };
    script[9] = xmask_state_offset;

    // Writeout hooks: initialize all to non-matching (stage 255)
    for i in 0..128 {
        script[128 + i] = 0xFF00FF00;
    }
    // Hook thread 0 to stage 0
    script[128] = (script[128] & 0xFFFF0000) | 0x0000;

    let mut pi = 256;

    // Global read (1 round)
    script[pi] = 0;
    script[pi + 1] = 1;
    pi += 512;

    // Boomerang stages
    for _stage in 0..num_stages {
        // Shuffle rounds (4096 words) — all zeros (identity)
        pi += 4096;
        // Flags (1024 words): passthrough for hier tree
        script[pi + 128 * 4 + 2] = 0xFFFFFFFF; // orb for thread 128
        for hi in 1..=7u32 {
            let hw = 1usize << (7 - hi);
            script[pi + hw * 4 + 2] = 0xFFFFFFFF;
        }
        script[pi + 0 * 4 + 2] = 0xFFFFFFFF;
        pi += 1024;
    }

    // SRAM + dup permutation (5120 words)
    pi += 5120;

    // Clock enable permutation
    pi += 4096;
    script[pi] = 0xFFFFFFFF; // inv for IO word 0
    pi += 1024;

    // Dummy end partition
    script[pi] = 0;
    script[pi + 1] = 1;

    script
}

fn bench_xprop_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("xprop_cpu_kernel");

    for &num_ios in &[32u32, 128, 256] {
        for &num_stages in &[1u32, 4, 8] {
            let script = build_bench_script(false, num_ios, num_stages);
            let script_xcap = build_bench_script(true, num_ios, num_stages);

            let state_size = num_ios as usize;
            let mut input_state = vec![0u32; state_size];
            let mut output_state = vec![0u32; state_size];
            let mut sram_data = vec![0u32; 0];

            let id = format!("ios={}_stages={}", num_ios, num_stages);

            // Baseline: two-state kernel
            group.bench_with_input(
                BenchmarkId::new("two_state", &id),
                &script,
                |b, script| {
                    b.iter(|| {
                        output_state.fill(0);
                        simulate_block_v1(
                            black_box(script),
                            black_box(&input_state),
                            black_box(&mut output_state),
                            black_box(&mut sram_data),
                            false,
                        );
                    })
                },
            );

            // X-prop: X-free partition (should match baseline)
            let script_xfree = build_bench_script(false, num_ios, num_stages);
            let mut input_xmask = vec![0u32; state_size];
            let mut output_xmask = vec![0u32; state_size];
            let mut sram_xmask = vec![0u32; 0];

            group.bench_with_input(
                BenchmarkId::new("xprop_xfree", &id),
                &script_xfree,
                |b, script| {
                    b.iter(|| {
                        output_state.fill(0);
                        output_xmask.fill(0);
                        simulate_block_v1_xprop(
                            black_box(script),
                            black_box(&input_state),
                            black_box(&mut output_state),
                            black_box(&input_xmask),
                            black_box(&mut output_xmask),
                            black_box(&mut sram_data),
                            black_box(&mut sram_xmask),
                            false,
                        );
                    })
                },
            );

            // X-prop: X-capable partition (full overhead)
            group.bench_with_input(
                BenchmarkId::new("xprop_xcapable", &id),
                &script_xcap,
                |b, script| {
                    b.iter(|| {
                        output_state.fill(0);
                        output_xmask.fill(0);
                        simulate_block_v1_xprop(
                            black_box(script),
                            black_box(&input_state),
                            black_box(&mut output_state),
                            black_box(&input_xmask),
                            black_box(&mut output_xmask),
                            black_box(&mut sram_data),
                            black_box(&mut sram_xmask),
                            false,
                        );
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_xprop_kernel);
criterion_main!(benches);
