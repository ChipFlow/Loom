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
}

impl TryFrom<u32> for EventType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(EventType::Stop),
            1 => Ok(EventType::Finish),
            2 => Ok(EventType::Display),
            3 => Ok(EventType::AssertFail),
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
                        clilog::error!(
                            "Maximum assertion failures ({}) reached, terminating",
                            max
                        );
                        return SimControl::Terminate;
                    }
                }
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
        assert!(EventType::try_from(4u32).is_err());
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
}
