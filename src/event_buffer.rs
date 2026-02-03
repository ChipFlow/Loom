// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Event buffer for GPU→CPU communication of non-synthesizable constructs.
//!
//! The event buffer is a shared memory region where the GPU writes events
//! (e.g., $stop, $finish, $display, assertion failures) and the CPU processes
//! them between simulation stages.

use std::sync::atomic::{AtomicU32, Ordering};

/// Maximum number of events that can be buffered per cycle.
/// If exceeded, events are dropped (with a warning).
pub const MAX_EVENTS: usize = 1024;

/// Event types that can be reported by the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventType {
    /// $stop system task - pause simulation
    Stop = 0,
    /// $finish system task - terminate simulation
    Finish = 1,
    /// $display/$write output (Phase 2)
    Display = 2,
    /// Assertion failure (Phase 3)
    AssertFail = 3,
    /// Setup time violation (Experiment 4)
    SetupViolation = 4,
    /// Hold time violation (Experiment 4)
    HoldViolation = 5,
}

impl TryFrom<u32> for EventType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(EventType::Stop),
            1 => Ok(EventType::Finish),
            2 => Ok(EventType::Display),
            3 => Ok(EventType::AssertFail),
            4 => Ok(EventType::SetupViolation),
            5 => Ok(EventType::HoldViolation),
            _ => Err(()),
        }
    }
}

/// A single event written by the GPU.
/// Layout must match csrc/event_buffer.h exactly.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Event {
    /// Event type (see EventType enum) - u32 for GPU alignment
    pub event_type: u32,
    /// Message ID for $display and assertions (index into message table)
    pub message_id: u32,
    /// Simulation cycle when the event occurred
    pub cycle: u32,
    /// Reserved for alignment
    pub _reserved: u32,
    /// Data payload for $display format arguments
    pub data: [u32; 4],
}

/// The event buffer structure shared between GPU and CPU.
/// This is placed in shared/unified memory for zero-copy access.
#[repr(C)]
pub struct EventBuffer {
    /// Number of events currently in the buffer (atomic for GPU writes)
    pub count: AtomicU32,
    /// Flag indicating if events were dropped due to overflow
    pub overflow: AtomicU32,
    /// Reserved for padding/alignment
    pub _reserved: [u32; 2],
    /// The event array
    pub events: [Event; MAX_EVENTS],
}

impl Default for EventBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBuffer {
    /// Create a new empty event buffer.
    pub fn new() -> Self {
        Self {
            count: AtomicU32::new(0),
            overflow: AtomicU32::new(0),
            _reserved: [0; 2],
            events: [Event::default(); MAX_EVENTS],
        }
    }

    /// Reset the buffer for a new cycle.
    pub fn reset(&self) {
        self.count.store(0, Ordering::Release);
        self.overflow.store(0, Ordering::Release);
    }

    /// Get the current number of events in the buffer.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire) as usize
    }

    /// Check if any events were dropped due to overflow.
    pub fn had_overflow(&self) -> bool {
        self.overflow.load(Ordering::Acquire) != 0
    }

    /// Drain all events from the buffer, processing each with the given closure.
    /// The buffer is reset after draining.
    pub fn drain<F>(&self, mut f: F)
    where
        F: FnMut(&Event),
    {
        let count = self.count.load(Ordering::Acquire) as usize;
        let actual_count = count.min(MAX_EVENTS);

        for i in 0..actual_count {
            f(&self.events[i]);
        }

        if count > MAX_EVENTS {
            clilog::warn!(
                "Event buffer overflow: {} events dropped",
                count - MAX_EVENTS
            );
        }

        self.reset();
    }

    /// Iterate over events without consuming them.
    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        let count = self.count.load(Ordering::Acquire) as usize;
        let actual_count = count.min(MAX_EVENTS);
        self.events[..actual_count].iter()
    }
}

/// What action the simulation should take after processing events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimControl {
    /// Continue simulation normally
    Continue,
    /// Pause simulation (for $stop or debug)
    Pause,
    /// Terminate simulation (for $finish or fatal assertion)
    Terminate,
}

/// Configuration for how assertions should be handled.
#[derive(Debug, Clone)]
pub struct AssertConfig {
    /// Action to take on assertion failure
    pub on_failure: AssertAction,
    /// Maximum number of failures before stopping (None = unlimited)
    pub max_failures: Option<u32>,
}

impl Default for AssertConfig {
    fn default() -> Self {
        Self {
            on_failure: AssertAction::Log,
            max_failures: None,
        }
    }
}

/// Configuration for timing violation handling (Experiment 4).
#[derive(Debug, Clone)]
pub struct TimingConfig {
    /// Action to take on timing violation
    pub on_violation: TimingAction,
    /// Maximum number of violations before stopping (None = unlimited)
    pub max_violations: Option<u32>,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            on_violation: TimingAction::Log,
            max_violations: None,
        }
    }
}

/// Action to take when a timing violation occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingAction {
    /// Log the violation and continue
    Log,
    /// Warn and continue
    Warn,
    /// Terminate simulation
    Terminate,
}

/// Action to take when an assertion fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertAction {
    /// Log the failure and continue
    Log,
    /// Pause simulation (like $stop)
    Pause,
    /// Terminate simulation (like $finish)
    Terminate,
}

/// Statistics tracked during simulation.
#[derive(Debug, Default, Clone)]
pub struct SimStats {
    /// Number of assertion failures encountered
    pub assertion_failures: u32,
    /// Number of $stop calls
    pub stop_count: u32,
    /// Number of events dropped due to overflow
    pub events_dropped: u32,
    /// Number of setup timing violations (Experiment 4)
    pub setup_violations: u32,
    /// Number of hold timing violations (Experiment 4)
    pub hold_violations: u32,
}

/// Process events from the buffer and determine simulation control.
///
/// # Arguments
/// * `buffer` - The event buffer to process
/// * `assert_config` - Configuration for assertion handling
/// * `stats` - Statistics to update
/// * `message_handler` - Optional callback for $display messages
///
/// # Returns
/// The simulation control action to take
pub fn process_events<F>(
    buffer: &EventBuffer,
    assert_config: &AssertConfig,
    stats: &mut SimStats,
    mut message_handler: F,
) -> SimControl
where
    F: FnMut(u32, u32, &[u32]),
{
    let mut result = SimControl::Continue;

    if buffer.had_overflow() {
        stats.events_dropped += (buffer.count.load(Ordering::Acquire) as usize - MAX_EVENTS) as u32;
    }

    for event in buffer.iter() {
        let event_type = match EventType::try_from(event.event_type) {
            Ok(t) => t,
            Err(_) => {
                clilog::warn!("Unknown event type: {}", event.event_type);
                continue;
            }
        };

        match event_type {
            EventType::Stop => {
                clilog::info!("[cycle {}] $stop encountered", event.cycle);
                stats.stop_count += 1;
                result = SimControl::Pause;
            }
            EventType::Finish => {
                clilog::info!("[cycle {}] $finish encountered", event.cycle);
                return SimControl::Terminate;
            }
            EventType::Display => {
                message_handler(event.message_id, event.cycle, &event.data);
            }
            EventType::AssertFail => {
                clilog::warn!(
                    "[cycle {}] Assertion failed (id={})",
                    event.cycle,
                    event.message_id
                );
                stats.assertion_failures += 1;

                match assert_config.on_failure {
                    AssertAction::Log => {}
                    AssertAction::Pause => {
                        result = SimControl::Pause;
                    }
                    AssertAction::Terminate => {
                        return SimControl::Terminate;
                    }
                }

                if let Some(max) = assert_config.max_failures {
                    if stats.assertion_failures >= max {
                        clilog::error!("Maximum assertion failures ({}) reached, terminating", max);
                        return SimControl::Terminate;
                    }
                }
            }
            EventType::SetupViolation => {
                // data[0] = DFF cell ID, data[1] = slack (signed as u32)
                let cell_id = event.data[0];
                let slack = event.data[1] as i32;
                clilog::warn!(
                    "[cycle {}] SETUP VIOLATION: DFF {} slack={} ps",
                    event.cycle,
                    cell_id,
                    slack
                );
                stats.setup_violations += 1;
            }
            EventType::HoldViolation => {
                // data[0] = DFF cell ID, data[1] = slack (signed as u32)
                let cell_id = event.data[0];
                let slack = event.data[1] as i32;
                clilog::warn!(
                    "[cycle {}] HOLD VIOLATION: DFF {} slack={} ps",
                    event.cycle,
                    cell_id,
                    slack
                );
                stats.hold_violations += 1;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_buffer_new() {
        let buf = EventBuffer::new();
        assert_eq!(buf.len(), 0);
        assert!(!buf.had_overflow());
    }

    #[test]
    fn test_event_type_conversion() {
        assert_eq!(EventType::try_from(0u32), Ok(EventType::Stop));
        assert_eq!(EventType::try_from(1u32), Ok(EventType::Finish));
        assert_eq!(EventType::try_from(2u32), Ok(EventType::Display));
        assert_eq!(EventType::try_from(3u32), Ok(EventType::AssertFail));
        assert_eq!(EventType::try_from(4u32), Ok(EventType::SetupViolation));
        assert_eq!(EventType::try_from(5u32), Ok(EventType::HoldViolation));
        assert!(EventType::try_from(6u32).is_err());
    }

    #[test]
    fn test_event_buffer_reset() {
        let buf = EventBuffer::new();
        buf.count.store(5, Ordering::Release);
        buf.overflow.store(1, Ordering::Release);

        buf.reset();

        assert_eq!(buf.len(), 0);
        assert!(!buf.had_overflow());
    }

    // Helper to manually add an event to the buffer (simulates GPU write)
    fn add_event(buf: &mut EventBuffer, event_type: EventType, cycle: u32) {
        let idx = buf.count.fetch_add(1, Ordering::AcqRel) as usize;
        if idx < MAX_EVENTS {
            buf.events[idx].event_type = event_type as u32;
            buf.events[idx].cycle = cycle;
            buf.events[idx].message_id = 0;
        }
    }

    #[test]
    fn test_process_events_empty() {
        let buf = EventBuffer::new();
        let config = AssertConfig::default();
        let mut stats = SimStats::default();

        let control = process_events(&buf, &config, &mut stats, |_, _, _| {});

        assert_eq!(control, SimControl::Continue);
        assert_eq!(stats.stop_count, 0);
        assert_eq!(stats.assertion_failures, 0);
    }

    #[test]
    fn test_process_events_stop() {
        let mut buf = EventBuffer::new();
        add_event(&mut buf, EventType::Stop, 42);

        let config = AssertConfig::default();
        let mut stats = SimStats::default();

        let control = process_events(&buf, &config, &mut stats, |_, _, _| {});

        assert_eq!(control, SimControl::Pause);
        assert_eq!(stats.stop_count, 1);
    }

    #[test]
    fn test_process_events_finish() {
        let mut buf = EventBuffer::new();
        add_event(&mut buf, EventType::Finish, 100);

        let config = AssertConfig::default();
        let mut stats = SimStats::default();

        let control = process_events(&buf, &config, &mut stats, |_, _, _| {});

        assert_eq!(control, SimControl::Terminate);
    }

    #[test]
    fn test_process_events_assert_log() {
        let mut buf = EventBuffer::new();
        add_event(&mut buf, EventType::AssertFail, 50);
        add_event(&mut buf, EventType::AssertFail, 51);

        let config = AssertConfig {
            on_failure: AssertAction::Log,
            max_failures: None,
        };
        let mut stats = SimStats::default();

        let control = process_events(&buf, &config, &mut stats, |_, _, _| {});

        assert_eq!(control, SimControl::Continue);
        assert_eq!(stats.assertion_failures, 2);
    }

    #[test]
    fn test_process_events_assert_terminate() {
        let mut buf = EventBuffer::new();
        add_event(&mut buf, EventType::AssertFail, 50);

        let config = AssertConfig {
            on_failure: AssertAction::Terminate,
            max_failures: None,
        };
        let mut stats = SimStats::default();

        let control = process_events(&buf, &config, &mut stats, |_, _, _| {});

        assert_eq!(control, SimControl::Terminate);
        assert_eq!(stats.assertion_failures, 1);
    }

    #[test]
    fn test_process_events_max_failures() {
        let mut buf = EventBuffer::new();
        add_event(&mut buf, EventType::AssertFail, 1);
        add_event(&mut buf, EventType::AssertFail, 2);
        add_event(&mut buf, EventType::AssertFail, 3);

        let config = AssertConfig {
            on_failure: AssertAction::Log,
            max_failures: Some(2),
        };
        let mut stats = SimStats::default();

        let control = process_events(&buf, &config, &mut stats, |_, _, _| {});

        // Should terminate after 2 failures
        assert_eq!(control, SimControl::Terminate);
        assert_eq!(stats.assertion_failures, 2);
    }

    #[test]
    fn test_process_events_display_callback() {
        let mut buf = EventBuffer::new();
        // Manually set up a display event with message_id
        let idx = buf.count.fetch_add(1, Ordering::AcqRel) as usize;
        buf.events[idx].event_type = EventType::Display as u32;
        buf.events[idx].cycle = 25;
        buf.events[idx].message_id = 42;
        buf.events[idx].data = [1, 2, 3, 4];

        let config = AssertConfig::default();
        let mut stats = SimStats::default();
        let mut captured_msg_id = 0u32;
        let mut captured_cycle = 0u32;

        let control = process_events(&buf, &config, &mut stats, |msg_id, cycle, _data| {
            captured_msg_id = msg_id;
            captured_cycle = cycle;
        });

        assert_eq!(control, SimControl::Continue);
        assert_eq!(captured_msg_id, 42);
        assert_eq!(captured_cycle, 25);
    }

    #[test]
    fn test_finish_takes_priority() {
        let mut buf = EventBuffer::new();
        // Stop comes first, then Finish
        add_event(&mut buf, EventType::Stop, 10);
        add_event(&mut buf, EventType::Finish, 11);
        add_event(&mut buf, EventType::Stop, 12); // This shouldn't be processed

        let config = AssertConfig::default();
        let mut stats = SimStats::default();

        let control = process_events(&buf, &config, &mut stats, |_, _, _| {});

        // Finish should cause immediate termination
        assert_eq!(control, SimControl::Terminate);
        // Only one stop was processed before finish
        assert_eq!(stats.stop_count, 1);
    }

    // Helper to add a timing violation event with slack data
    fn add_timing_event(buf: &mut EventBuffer, event_type: EventType, cycle: u32, cell_id: u32, slack: i32) {
        let idx = buf.count.fetch_add(1, Ordering::AcqRel) as usize;
        if idx < MAX_EVENTS {
            buf.events[idx].event_type = event_type as u32;
            buf.events[idx].cycle = cycle;
            buf.events[idx].message_id = 0;
            buf.events[idx].data[0] = cell_id;
            buf.events[idx].data[1] = slack as u32;
        }
    }

    #[test]
    fn test_process_timing_violations() {
        let mut buf = EventBuffer::new();
        add_timing_event(&mut buf, EventType::SetupViolation, 10, 0, -50);
        add_timing_event(&mut buf, EventType::HoldViolation, 11, 1, -25);
        add_timing_event(&mut buf, EventType::SetupViolation, 12, 2, -30);

        let config = AssertConfig::default();
        let mut stats = SimStats::default();

        let control = process_events(&buf, &config, &mut stats, |_, _, _| {});

        // Timing violations don't stop simulation by default
        assert_eq!(control, SimControl::Continue);
        assert_eq!(stats.setup_violations, 2);
        assert_eq!(stats.hold_violations, 1);
    }
}
