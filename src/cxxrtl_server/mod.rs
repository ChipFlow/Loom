// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CXXRTL debug server implementation for Loom.
//!
//! Implements the CXXRTL debug protocol (<https://cxxrtl.org/protocol.html>)
//! to expose GPU-accelerated simulation results to interactive waveform
//! viewers like Surfer.
//!
//! # Architecture
//!
//! The server runs two threads:
//! - **Protocol thread**: handles TCP connection, parses commands, builds responses
//! - **Simulation thread**: runs the GPU dispatch loop, controlled by run/pause commands
//!
//! Communication between threads uses atomic state in [`sim_control::SimControl`].

pub mod design;
pub mod protocol;
pub mod signals;
pub mod sim_control;
pub mod transport;
pub mod waveform;

use std::collections::HashMap;
use std::sync::Arc;

use crate::aig::AIG;
use crate::flatten::FlattenedScriptV1;
use netlistdb::NetlistDB;

use design::{extract_items, extract_scopes, get_child_scopes, get_items_for_scope};
use protocol::*;
use signals::SignalRegistry;
use sim_control::SimControl;
use transport::{Connection, Server};
use waveform::query_interval;

/// Configuration for the CXXRTL server.
pub struct ServerConfig {
    /// TCP address to bind to (e.g. "127.0.0.1:9000").
    pub bind_addr: String,
    /// Femtoseconds per VCD timescale unit.
    pub timescale_fs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            bind_addr: "127.0.0.1:9000".to_string(),
            timescale_fs: 1_000_000_000_000, // 1ps default
        }
    }
}

/// The main CXXRTL protocol session, handling one client connection.
pub struct Session {
    conn: Connection,
    scopes: HashMap<String, protocol::ScopeInfo>,
    items_by_scope: HashMap<String, HashMap<String, protocol::ItemInfo>>,
    registry: SignalRegistry,
    sim_ctrl: Arc<SimControl>,
    /// Reference to the simulation state buffer.
    /// Updated as simulation progresses. The protocol thread reads this
    /// to service query_interval requests.
    states: Arc<std::sync::RwLock<Vec<u32>>>,
    /// State size per cycle in u32 words.
    state_size: u32,
    /// Cycle timestamps.
    offsets_timestamps: Arc<Vec<(usize, u64)>>,
    /// Timescale in femtoseconds.
    timescale_fs: u64,
    /// Events queued to be sent after the current command response.
    pending_events: Vec<serde_json::Value>,
}

impl Session {
    /// Create a new session from an accepted connection and design data.
    pub fn new(
        conn: Connection,
        netlistdb: &NetlistDB,
        aig: &AIG,
        script: &FlattenedScriptV1,
        sim_ctrl: Arc<SimControl>,
        states: Arc<std::sync::RwLock<Vec<u32>>>,
        offsets_timestamps: Arc<Vec<(usize, u64)>>,
        timescale_fs: u64,
    ) -> Self {
        let scopes = extract_scopes(netlistdb);
        let (items_by_scope, resolved_signals) = extract_items(netlistdb, aig, script);
        let registry = SignalRegistry::new(resolved_signals);

        Session {
            conn,
            scopes,
            items_by_scope,
            registry,
            sim_ctrl,
            states,
            state_size: script.reg_io_state_size,
            offsets_timestamps,
            timescale_fs,
            pending_events: Vec::new(),
        }
    }

    /// Run the protocol handler loop.
    ///
    /// Performs the greeting handshake, then dispatches commands until
    /// the client disconnects.
    pub fn run(&mut self) -> std::io::Result<()> {
        // Wait for client greeting
        let greeting_msg = self.conn.read_message()?;
        match greeting_msg {
            Some(msg) => {
                if msg.get("type").and_then(|v| v.as_str()) != Some("greeting") {
                    clilog::warn!("Expected greeting, got: {:?}", msg);
                    let err = response_error("protocol_error", "expected greeting message");
                    self.conn.send_message(&err)?;
                    return Ok(());
                }
                let version = msg.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
                if version != PROTOCOL_VERSION as u64 {
                    clilog::warn!(
                        "Client protocol version {} differs from server version {}",
                        version,
                        PROTOCOL_VERSION
                    );
                }
            }
            None => {
                clilog::info!("Client disconnected before greeting");
                return Ok(());
            }
        }

        // Send server greeting
        let server_greeting = ServerGreeting::new();
        let greeting_json = serde_json::to_value(&server_greeting)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        self.conn.send_message(&greeting_json)?;
        clilog::info!("CXXRTL greeting handshake complete");

        // Command loop
        loop {
            let msg = match self.conn.read_message()? {
                Some(msg) => msg,
                None => {
                    clilog::info!("CXXRTL client disconnected");
                    break;
                }
            };

            let msg_type = msg
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            if msg_type != "command" {
                let err = response_error("protocol_error", &format!("expected command, got '{}'", msg_type));
                self.conn.send_message(&err)?;
                continue;
            }

            match Command::from_json(&msg) {
                Ok(cmd) => {
                    let response = self.handle_command(cmd);
                    self.conn.send_message(&response)?;
                    // Send any queued async events after the response
                    let events: Vec<_> = self.pending_events.drain(..).collect();
                    for event in events {
                        self.conn.send_message(&event)?;
                    }
                }
                Err(e) => {
                    let err = response_error("invalid_command", &e);
                    self.conn.send_message(&err)?;
                }
            }
        }

        Ok(())
    }

    /// Dispatch a parsed command and return the JSON response.
    fn handle_command(&mut self, cmd: Command) -> serde_json::Value {
        match cmd {
            Command::ListScopes { scope } => {
                let child_scopes = get_child_scopes(&self.scopes, scope.as_deref());
                response_list_scopes(child_scopes)
            }

            Command::ListItems { scope } => {
                let items = get_items_for_scope(&self.items_by_scope, scope.as_deref());
                response_list_items(items)
            }

            Command::ReferenceItems { reference, items } => {
                let designations = items.as_deref();
                match self.registry.reference_items(&reference, designations) {
                    Ok(()) => response_reference_items(),
                    Err(e) => response_error("invalid_reference", &e),
                }
            }

            Command::QueryInterval {
                interval,
                collapse,
                items,
                item_values_encoding: _,
                diagnostics: _,
            } => {
                let begin = TimePoint(interval.0);
                let end = TimePoint(interval.1);
                let bound_ref = items.as_ref().and_then(|r| self.registry.get_reference(r));

                let states = self.states.read().unwrap();
                let samples = query_interval(
                    &states,
                    self.state_size,
                    &self.offsets_timestamps,
                    self.timescale_fs,
                    bound_ref,
                    &begin,
                    &end,
                    collapse,
                );
                response_query_interval(samples)
            }

            Command::GetSimulationStatus => {
                let status = self.sim_ctrl.status();
                let current_cycle = self.sim_ctrl.current_cycle();
                let latest_time = SimControl::cycle_to_time(
                    current_cycle,
                    &self.offsets_timestamps,
                    self.timescale_fs,
                );
                let next_time = if status == SimulationStatus::Paused {
                    let next_cycle = current_cycle + 1;
                    if (next_cycle as usize) <= self.offsets_timestamps.len() {
                        Some(SimControl::cycle_to_time(
                            next_cycle,
                            &self.offsets_timestamps,
                            self.timescale_fs,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };
                response_simulation_status(status, &latest_time, next_time.as_ref())
            }

            Command::RunSimulation {
                until_time: _,
                until_diagnostics: _,
                sample_item_values: _,
            } => {
                // In replay buffer mode (simulation already finished), we
                // send the response and then immediately send a finished event.
                // For future incremental simulation, this would dispatch to
                // the simulation thread instead.
                if self.sim_ctrl.status() == SimulationStatus::Finished {
                    // Send the response first, then the event
                    let resp = response_run_simulation();
                    // Queue the finished event to be sent after the response
                    let current_cycle = self.sim_ctrl.current_cycle();
                    let time = SimControl::cycle_to_time(
                        current_cycle,
                        &self.offsets_timestamps,
                        self.timescale_fs,
                    );
                    self.pending_events
                        .push(event_simulation_finished(&time));
                    resp
                } else {
                    let target_cycle = self.offsets_timestamps.len() as u64;
                    self.sim_ctrl
                        .request_run(target_cycle, true, Vec::new());
                    response_run_simulation()
                }
            }

            Command::PauseSimulation => {
                // In replay mode, simulation is already finished — just report
                // the final time.
                if self.sim_ctrl.status() == SimulationStatus::Finished {
                    let current_cycle = self.sim_ctrl.current_cycle();
                    let time = SimControl::cycle_to_time(
                        current_cycle,
                        &self.offsets_timestamps,
                        self.timescale_fs,
                    );
                    return response_pause_simulation(&time);
                }

                self.sim_ctrl.request_pause();
                self.sim_ctrl.wait_for_pause();
                let current_cycle = self.sim_ctrl.current_cycle();
                let time = SimControl::cycle_to_time(
                    current_cycle,
                    &self.offsets_timestamps,
                    self.timescale_fs,
                );
                response_pause_simulation(&time)
            }
        }
    }
}

/// Start the CXXRTL server, accepting one client and running the session.
///
/// This blocks the calling thread until the client disconnects.
pub fn serve(
    config: &ServerConfig,
    netlistdb: &NetlistDB,
    aig: &AIG,
    script: &FlattenedScriptV1,
    sim_ctrl: Arc<SimControl>,
    states: Arc<std::sync::RwLock<Vec<u32>>>,
    offsets_timestamps: Arc<Vec<(usize, u64)>>,
) -> std::io::Result<()> {
    let server = Server::bind(&config.bind_addr)?;
    let conn = server.accept()?;

    let mut session = Session::new(
        conn,
        netlistdb,
        aig,
        script,
        sim_ctrl,
        states,
        offsets_timestamps,
        config.timescale_fs,
    );

    session.run()
}
