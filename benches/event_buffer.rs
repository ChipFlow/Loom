// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Benchmarks for event buffer processing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use gem::event_buffer::{
    AssertAction, AssertConfig, Event, EventBuffer, EventType, SimStats, process_events,
};
use std::sync::atomic::Ordering;

fn setup_buffer_with_events(count: usize, event_type: EventType) -> EventBuffer {
    let buf = EventBuffer::new();
    for i in 0..count.min(1024) {
        let idx = buf.count.fetch_add(1, Ordering::AcqRel) as usize;
        // Safety: we own the buffer exclusively during setup
        unsafe {
            let events_ptr = &buf.events as *const _ as *mut [Event; 1024];
            (*events_ptr)[idx].event_type = event_type as u32;
            (*events_ptr)[idx].cycle = i as u32;
            (*events_ptr)[idx].message_id = 0;
        }
    }
    buf
}

fn bench_process_events(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_buffer");

    for count in [10, 100, 500, 1024] {
        group.bench_with_input(
            BenchmarkId::new("process_display_events", count),
            &count,
            |b, &count| {
                let buf = setup_buffer_with_events(count, EventType::Display);
                let config = AssertConfig::default();

                b.iter(|| {
                    let mut stats = SimStats::default();
                    let mut msg_count = 0u32;
                    let result = process_events(
                        black_box(&buf),
                        &config,
                        &mut stats,
                        |_msg_id, _cycle, _data| {
                            msg_count += 1;
                        },
                    );
                    black_box((result, msg_count))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("process_assert_events", count),
            &count,
            |b, &count| {
                let buf = setup_buffer_with_events(count, EventType::AssertFail);
                let config = AssertConfig {
                    on_failure: AssertAction::Log,
                    max_failures: None,
                };

                b.iter(|| {
                    let mut stats = SimStats::default();
                    let result =
                        process_events(black_box(&buf), &config, &mut stats, |_, _, _| {});
                    black_box((result, stats.assertion_failures))
                });
            },
        );
    }

    group.finish();
}

fn bench_buffer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_ops");

    group.bench_function("create_new", |b| {
        b.iter(|| black_box(EventBuffer::new()))
    });

    group.bench_function("reset", |b| {
        let buf = EventBuffer::new();
        buf.count.store(500, Ordering::Release);
        buf.overflow.store(1, Ordering::Release);

        b.iter(|| {
            buf.reset();
            black_box(&buf)
        })
    });

    group.bench_function("iter_1024_events", |b| {
        let buf = setup_buffer_with_events(1024, EventType::Display);

        b.iter(|| {
            let count: usize = buf.iter().count();
            black_box(count)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_process_events, bench_buffer_operations);
criterion_main!(benches);
