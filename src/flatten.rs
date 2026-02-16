// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Partition scheduler and flattener

use crate::aig::{DriverType, EndpointGroup, AIG};
use crate::aigpdk::AIGPDK_SRAM_ADDR_WIDTH;
use crate::liberty_parser::TimingLibrary;
use crate::pe::{Partition, BOOMERANG_NUM_STAGES};
use crate::sdf_parser::SdfFile;
use crate::staging::StagedAIG;
use indexmap::IndexMap;
use netlistdb::NetlistDB;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use ulib::UVec;

pub const NUM_THREADS_V1: usize = 1 << (BOOMERANG_NUM_STAGES - 5);

// === Timing Data Structures for GPU ===

/// Packed delay representation for GPU consumption.
/// Uses 16-bit fixed-point values in picoseconds.
/// Range: 0-65535 ps (0-65.5 ns), resolution: 1 ps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PackedDelay {
    /// Cell rise delay in picoseconds
    pub rise_ps: u16,
    /// Cell fall delay in picoseconds
    pub fall_ps: u16,
}

impl PackedDelay {
    /// Create a new PackedDelay from rise/fall times.
    pub fn new(rise_ps: u16, fall_ps: u16) -> Self {
        Self { rise_ps, fall_ps }
    }

    /// Create a PackedDelay from u64 picosecond values, saturating at u16::MAX.
    pub fn from_u64(rise_ps: u64, fall_ps: u64) -> Self {
        Self {
            rise_ps: rise_ps.min(u16::MAX as u64) as u16,
            fall_ps: fall_ps.min(u16::MAX as u64) as u16,
        }
    }

    /// Get the maximum delay (for critical path analysis).
    pub fn max_delay(&self) -> u16 {
        self.rise_ps.max(self.fall_ps)
    }

    /// Pack into a single u32 for GPU transfer.
    /// Format: [rise_ps:16][fall_ps:16]
    pub fn to_u32(&self) -> u32 {
        ((self.rise_ps as u32) << 16) | (self.fall_ps as u32)
    }

    /// Unpack from a u32.
    pub fn from_u32(packed: u32) -> Self {
        Self {
            rise_ps: (packed >> 16) as u16,
            fall_ps: (packed & 0xFFFF) as u16,
        }
    }
}

/// DFF timing constraints for GPU checking.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DFFConstraint {
    /// Setup time in picoseconds
    pub setup_ps: u16,
    /// Hold time in picoseconds
    pub hold_ps: u16,
    /// Position of D input data arrival in state
    pub data_state_pos: u32,
    /// DFF cell ID (for error reporting)
    pub cell_id: u32,
}

impl DFFConstraint {
    /// Pack into two u32s for GPU transfer.
    /// Returns (timing_word, position_word).
    pub fn to_u32_pair(&self) -> (u32, u32) {
        let timing = ((self.setup_ps as u32) << 16) | (self.hold_ps as u32);
        let position = self.data_state_pos;
        (timing, position)
    }
}

/// A flattened script, for partition executor version 1.
/// See [FlattenedScriptV1::blocks_data] for the format details.
///
/// Generally, a script contains a number of major stages.
/// Each stage consists of the same number of blocks.
/// Each block contains a list of flattened partitions.
pub struct FlattenedScriptV1 {
    /// the number of blocks
    pub num_blocks: usize,
    /// the number of major stages
    pub num_major_stages: usize,
    /// the CSR start indices of stages and blocks.
    ///
    /// this length is num_blocks * num_major_stages + 1
    pub blocks_start: UVec<usize>,
    /// the partition instructions.
    ///
    /// the instructions follow a special format.
    /// it consists of zero or more partitions.
    /// 1. metadata section [1x256]:
    ///    the number of boomerang stages.
    ///      32-bit
    ///      if this is zero, the stage will not run. only happens
    ///      when the block has no partition mapped onto it.
    ///    is this the last boomerang stage?
    ///      32-bit, only 0 or 1
    ///    the number of valid write-outs.
    ///      32-bit
    ///    the write-out destination offset.
    ///      32-bit
    ///      (at this offset, we put all FFs and other outputs.)
    ///    the srams count and offsets.
    ///      32-bit count
    ///      32-bit memory offset for the first mem.
    ///    the number of global read rounds
    ///      32-bit count
    ///    the number of output-duplicate writeouts
    ///      32-bit count
    ///      this is used when one output pin is used by either
    ///      <both output and FFs>, or <multiple FFs with different
    ///      enabling conditions>.
    ///    padding towards 128
    ///    the location of early write-outs, excl. mem.
    ///      256 * [#boomstage & 0~256 id], compressed to 128
    /// 2. initial read-global permutation [2x]*rounds
    ///    32-bit indices*1 for each of the 256 threads.
    ///    32-bit valid mask for each of the 256 threads.
    ///    if the valid mask is zero, the memory is not read.
    ///    the result should be stored "like pext instruction",
    ///    but "reversed", and then appended to the low bits
    ///    in each round.
    ///    the index is encoded with a type bit at the highest bit:
    ///      if it is 0: it means it is offset from previous iteration.
    ///      if it is 1: it is offset from current iteration
    ///        which means it is an intermediate value coming from
    ///        the same cycle but a previous major stage.
    /// 3. boomerang sections, repeat below N*16
    ///    1. local shuffle permutation
    ///       32-bit indices * 16 for each of the 256 threads.
    ///    2. input (with inv) * 8192bits * (3+1padding)
    ///       32-bit * 256 threads * (3+1): xora, xorb, orb, [0 padding]
    ///       0xy: and gate, out = (a^x)&(b^y).
    ///       100: passthrough, out = a.
    ///       111: invalid, can be skipped.
    ///    -. write out, according to rest
    /// 4. global write-outs.
    ///    1. sram & additional endpoint copy permutations, inv. [16x].
    ///       only the inputs within sram and endpoint copy
    ///       range will be considered.
    ///       followed by [4x] invert, set0, and two 0 paddings.
    ///    2. permutation for the write-out enabler pins, inv. [16]
    ///       include itself inv and data inv.
    ///       followed by [3(+1padding)x]
    ///         clock invert, clock set0, data invert, and 0 padding
    ///    -. commit the write-out
    pub blocks_data: UVec<u32>,
    /// the state size including DFF and I/O states only.
    ///
    /// the inputs are always at front.
    pub reg_io_state_size: u32,
    /// the u32 array length for storing SRAMs.
    pub sram_storage_size: u32,
    /// expected input AIG pins layout
    pub input_layout: Vec<usize>,
    /// maps from primary outputs, FF:D and SRAM:PORT_R_RD_DATA AIG pins
    /// to state offset, index with invert.
    pub output_map: IndexMap<usize, u32>,
    /// maps from primary inputs, FF:Q/SRAM:* input AIG pins to state offset,
    /// index WITHOUT invert.
    pub input_map: IndexMap<usize, u32>,
    /// (for debug purpose) the relation between major stage, block and
    /// part indices as given in construction.
    pub stages_blocks_parts: Vec<Vec<Vec<usize>>>,
    /// Maps from assertion cell IDs to their condition positions in state.
    /// Each entry is (cell_id, state_bit_position, message_id, control_type).
    /// control_type: None = assertion, Some(Stop) = $stop, Some(Finish) = $finish
    pub assertion_positions: Vec<(usize, u32, u32, Option<crate::aig::SimControlType>)>,
    /// Maps from display cell IDs to their enable positions and format info.
    /// Each entry is (cell_id, enable_pos, format_string, arg_positions, arg_widths).
    pub display_positions: Vec<(usize, u32, String, Vec<u32>, Vec<u32>)>,

    // === Timing Analysis Fields ===

    /// Per-AIG-pin delays loaded from Liberty library.
    /// Index 0 is unused (Tie0). Indexed by aigpin.
    /// Empty if timing not loaded.
    pub gate_delays: Vec<PackedDelay>,

    /// DFF timing constraints for setup/hold checking.
    /// One entry per DFF in the design.
    pub dff_constraints: Vec<DFFConstraint>,

    /// Clock period in picoseconds for timing checks.
    pub clock_period_ps: u64,

    /// Whether timing data has been loaded.
    pub timing_enabled: bool,

    /// Delay injection info for GPU kernel.
    /// Each entry: (offset in blocks_data where the padding u32 lives,
    ///              list of AIG pin indices contributing to this thread position).
    /// Used by load_timing() / load_timing_from_sdf() to patch the script
    /// with per-thread-position max gate delays.
    pub delay_patch_map: Vec<(usize, Vec<usize>)>,

}

fn map_global_read_to_rounds(inputs_taken: &BTreeMap<u32, u32>) -> Vec<Vec<(u32, u32)>> {
    let inputs_taken = inputs_taken
        .iter()
        .map(|(&a, &b)| (a, b))
        .collect::<Vec<_>>();
    // the larger the sorting chunk size, the better the successful chance,
    // but the less efficient due to worse cache coherency.
    let mut chunk_size = inputs_taken.len();
    while chunk_size >= 1 {
        let mut slices = inputs_taken.chunks(chunk_size).collect::<Vec<_>>();
        slices.sort_by_cached_key(|&slice| {
            u32::MAX - slice.iter().map(|(_, mask)| mask.count_ones()).sum::<u32>()
        });
        let mut rounds_idx_masks: Vec<Vec<(u32, u32)>> = vec![vec![]; NUM_THREADS_V1];
        let mut round_map_j = 0;
        let mut fail = false;
        for slice in slices {
            for &(offset, mask) in slice {
                let wrap_fail_j = round_map_j;
                while rounds_idx_masks[round_map_j]
                    .iter()
                    .map(|(_, mask)| mask.count_ones())
                    .sum::<u32>()
                    + mask.count_ones()
                    > 32
                {
                    round_map_j += 1;
                    if round_map_j == NUM_THREADS_V1 {
                        round_map_j = 0;
                    }
                    if round_map_j == wrap_fail_j {
                        // panic!("failed to map at part {} mem offset {}", i, offset);
                        fail = true;
                        break;
                    }
                }
                if fail {
                    break;
                }
                rounds_idx_masks[round_map_j].push((offset, mask));
                round_map_j += 1;
                if round_map_j == NUM_THREADS_V1 {
                    round_map_j = 0;
                }
            }
            if fail {
                break;
            }
        }
        if !fail {
            // let max_rounds = rounds_idx_masks.iter().map(|v| v.len()).max().unwrap();
            // println!("max_rounds: {}, round_map_j: {}, inputs_taken len {}", max_rounds, round_map_j, inputs_taken.len());
            return rounds_idx_masks;
        }
        chunk_size /= 2;
    }
    panic!("cannot map global init to any multiples of rounds.");
}

/// temporaries for a part being flattened. will be discarded after built.
#[derive(Debug, Clone, Default)]
struct FlatteningPart {
    /// for each boomerang stage, the result bits layout.
    afters: Vec<Vec<usize>>,
    /// for each partition, the output bits layout not containing sram outputs yet.
    parts_after_writeouts: Vec<usize>,
    /// mapping from aig pin index to writeout position (0~8192)
    after_writeout_pin2pos: IndexMap<usize, u16>,
    /// the number of SRAMs to simulate in this part.
    num_srams: u32,
    /// number of normal writeouts
    num_normal_writeouts: u32,
    /// number of writeout slots for output duplication
    num_duplicate_writeouts: u32,
    /// number of total writeouts
    num_writeouts: u32,
    /// the outputs categorized into activations
    comb_outputs_activations: IndexMap<usize, IndexMap<usize, Option<u16>>>,
    /// the current (placed) count of duplicate permutes
    cnt_placed_duplicate_permute: u32,

    /// the starting offset for FFs, outputs, and SRAM read results.
    state_start: u32,
    /// the starting offset of SRAM storage.
    sram_start: u32,

    /// the partial permutation instructions for
    /// 1. sram inputs
    /// 2. duplicated output pins due to difference in polarity/clock en.
    ///
    /// len: 8192
    sram_duplicate_permute: Vec<u16>,
    /// invert bit for sram_duplicate.
    ///
    /// len: 256
    sram_duplicate_inv: Vec<u32>,
    /// set-0 bit for sram_duplicate.
    ///
    /// len: 256
    sram_duplicate_set0: Vec<u32>,
    /// the permutation for clock enable pins.
    ///
    /// len: 8192
    clken_permute: Vec<u16>,
    /// invert bit for clken
    ///
    /// len: 256
    clken_inv: Vec<u32>,
    /// set-0 bit for clken
    ///
    /// len: 256
    clken_set0: Vec<u32>,
    /// invert bit for data corresponding to clken
    ///
    /// len: 256
    data_inv: Vec<u32>,
}

fn set_bit_in_u32(v: &mut u32, pos: u32, bit: u8) {
    if bit != 0 {
        *v |= 1 << pos;
    } else {
        *v &= !(1 << pos);
    }
}

impl FlatteningPart {
    fn init_afters_writeouts(&mut self, aig: &AIG, staged: &StagedAIG, part: &Partition) {
        let afters = part
            .stages
            .iter()
            .map(|s| {
                let mut after = Vec::with_capacity(1 << BOOMERANG_NUM_STAGES);
                after.push(usize::MAX);
                for i in (1..=BOOMERANG_NUM_STAGES).rev() {
                    after.extend(s.hier[i].iter().copied());
                }
                after
            })
            .collect::<Vec<_>>();
        let wos = part
            .stages
            .iter()
            .zip(afters.iter())
            .map(|(s, after)| {
                s.write_outs
                    .iter()
                    .map(|&woi| after[woi * 32..(woi + 1) * 32].iter().copied())
                    .flatten()
            })
            .flatten()
            .collect::<Vec<_>>();

        // println!("test wos: {:?}", wos);

        self.afters = afters;
        self.parts_after_writeouts = wos;
        self.num_normal_writeouts = part
            .stages
            .iter()
            .map(|s| s.write_outs.len())
            .sum::<usize>() as u32;
        self.num_srams = 0;

        // map: output aig pin id -> ((clken, data iv) -> pos)
        let mut comb_outputs_activations = IndexMap::<usize, IndexMap<usize, Option<u16>>>::new();
        for &endpt_i in &part.endpoints {
            match staged.get_endpoint_group(aig, endpt_i) {
                EndpointGroup::RAMBlock(_) => {
                    self.num_srams += 1;
                }
                EndpointGroup::PrimaryOutput(idx) => {
                    comb_outputs_activations
                        .entry(idx >> 1)
                        .or_default()
                        .insert(2 | (idx & 1), None);
                }
                EndpointGroup::StagedIOPin(idx) => {
                    comb_outputs_activations
                        .entry(idx)
                        .or_default()
                        .insert(2, None);
                }
                EndpointGroup::DFF(dff) => {
                    comb_outputs_activations
                        .entry(dff.d_iv >> 1)
                        .or_default()
                        .insert(dff.en_iv << 1 | (dff.d_iv & 1), None);
                }
                EndpointGroup::SimControl(ctrl) => {
                    // SimControl condition is always active (no clock enable)
                    comb_outputs_activations
                        .entry(ctrl.condition_iv >> 1)
                        .or_default()
                        .insert(2 | (ctrl.condition_iv & 1), None);
                }
                EndpointGroup::Display(disp) => {
                    // Display enable is always active (no clock enable for now)
                    comb_outputs_activations
                        .entry(disp.enable_iv >> 1)
                        .or_default()
                        .insert(2 | (disp.enable_iv & 1), None);
                    // Also track argument signals
                    for &arg_iv in &disp.args_iv {
                        if arg_iv > 1 {
                            comb_outputs_activations
                                .entry(arg_iv >> 1)
                                .or_default()
                                .insert(2 | (arg_iv & 1), None);
                        }
                    }
                }
            }
        }
        self.num_duplicate_writeouts = ((comb_outputs_activations
            .values()
            .map(|v| v.len() - 1)
            .sum::<usize>()
            + 31)
            / 32) as u32;
        self.comb_outputs_activations = comb_outputs_activations;

        self.num_writeouts =
            self.num_normal_writeouts + self.num_srams + self.num_duplicate_writeouts;

        self.after_writeout_pin2pos = self
            .parts_after_writeouts
            .iter()
            .enumerate()
            .filter_map(|(i, &pin)| {
                if pin == usize::MAX {
                    None
                } else {
                    Some((pin, i as u16))
                }
            })
            .collect::<IndexMap<_, _>>();
    }

    /// returns permutation id, invert bit, and setzero bit
    fn query_permute_with_pin_iv(&self, pin_iv: usize) -> (u16, u8, u8) {
        if pin_iv <= 1 {
            return (0, pin_iv as u8, 1);
        }
        let pos = self.after_writeout_pin2pos.get(&(pin_iv >> 1)).unwrap();
        (*pos, (pin_iv & 1) as u8, 0)
    }

    /// places a sram_duplicate bit.
    fn place_sram_duplicate(&mut self, pos: usize, (perm, inv, set0): (u16, u8, u8)) {
        self.sram_duplicate_permute[pos] = perm;
        set_bit_in_u32(
            &mut self.sram_duplicate_inv[pos >> 5],
            (pos & 31) as u32,
            inv,
        );
        set_bit_in_u32(
            &mut self.sram_duplicate_set0[pos >> 5],
            (pos & 31) as u32,
            set0,
        );
    }

    /// places a writeout bit's clock enable and data invert.
    fn place_clken_datainv(
        &mut self,
        pos: usize,
        clken_iv_perm: u16,
        clken_iv_inv: u8,
        clken_iv_set0: u8,
        data_inv: u8,
    ) {
        self.clken_permute[pos] = clken_iv_perm;
        set_bit_in_u32(
            &mut self.clken_inv[pos >> 5],
            (pos & 31) as u32,
            clken_iv_inv,
        );
        set_bit_in_u32(
            &mut self.clken_set0[pos >> 5],
            (pos & 31) as u32,
            clken_iv_set0,
        );
        set_bit_in_u32(&mut self.data_inv[pos >> 5], (pos & 31) as u32, data_inv);
    }

    /// returns a final local position for a data output bit with given pin_iv and clken_iv.
    ///
    /// if is not already placed, we will place it as well as place
    /// the clock enable bit, duplication bit, and bitflags for clock and data.
    fn get_or_place_output_with_activation(&mut self, pin_iv: usize, clken_iv: usize) -> u16 {
        let (activ_idx, _, pos) = self
            .comb_outputs_activations
            .get(&(pin_iv >> 1))
            .unwrap()
            .get_full(&(clken_iv << 1 | (pin_iv & 1)))
            .unwrap();
        if let Some(pos) = *pos {
            return pos;
        }
        let (clken_iv_perm, clken_iv_inv, clken_iv_set0) = self.query_permute_with_pin_iv(clken_iv);
        let origpos = match self.after_writeout_pin2pos.get(&(pin_iv >> 1)) {
            Some(origpos) => *origpos,
            None => {
                panic!("position of pin_iv {} (clken_iv {}) not found.. buggy boomerang, check if netlist and gemparts mismatch.", pin_iv, clken_iv)
            }
        } as usize;
        let r_pos = if activ_idx == 0 {
            self.place_clken_datainv(
                origpos,
                clken_iv_perm,
                clken_iv_inv,
                clken_iv_set0,
                (pin_iv & 1) as u8,
            );
            origpos as u16
        } else {
            self.cnt_placed_duplicate_permute += 1;
            let dup_pos = ((self.num_writeouts - self.num_srams) * 32
                - self.cnt_placed_duplicate_permute) as usize;
            let dup_perm_pos = ((self.num_srams * 4 + self.num_duplicate_writeouts) * 32
                - self.cnt_placed_duplicate_permute) as usize;
            if dup_perm_pos >= 8192 {
                panic!("sram duplicate bit larger than expected..")
                // dup_perm_pos = 8191;
            }
            self.place_sram_duplicate(dup_perm_pos, (origpos as u16, 0, 0));
            self.place_clken_datainv(
                dup_pos,
                clken_iv_perm,
                clken_iv_inv,
                clken_iv_set0,
                (pin_iv & 1) as u8,
            );
            dup_pos as u16
        };
        *self
            .comb_outputs_activations
            .get_mut(&(pin_iv >> 1))
            .unwrap()
            .get_mut(&(clken_iv << 1 | (pin_iv & 1)))
            .unwrap() = Some(r_pos);
        r_pos
    }

    fn make_inputs_outputs(
        &mut self,
        aig: &AIG,
        staged: &StagedAIG,
        part: &Partition,
        input_map: &mut IndexMap<usize, u32>,
        staged_io_map: &mut IndexMap<usize, u32>,
        output_map: &mut IndexMap<usize, u32>,
    ) {
        self.sram_duplicate_permute = vec![0; 1 << BOOMERANG_NUM_STAGES];
        self.sram_duplicate_inv = vec![0u32; NUM_THREADS_V1];
        self.sram_duplicate_set0 = vec![u32::MAX; NUM_THREADS_V1];
        self.clken_permute = vec![0; 1 << BOOMERANG_NUM_STAGES];
        self.clken_inv = vec![0u32; NUM_THREADS_V1];
        self.clken_set0 = vec![u32::MAX; NUM_THREADS_V1];
        self.data_inv = vec![0u32; NUM_THREADS_V1];
        self.cnt_placed_duplicate_permute = 0;

        let mut cur_sram_id = 0;
        for &endpt_i in &part.endpoints {
            match staged.get_endpoint_group(aig, endpt_i) {
                EndpointGroup::RAMBlock(ram) => {
                    let sram_rd_data_local_offset = self.num_writeouts as usize
                        - self.num_srams as usize
                        + cur_sram_id as usize;
                    let sram_rd_data_global_start =
                        self.state_start + self.num_writeouts - self.num_srams + cur_sram_id;
                    let (perm_r_en_iv, perm_r_en_iv_inv, perm_r_en_iv_set0) =
                        self.query_permute_with_pin_iv(ram.port_r_en_iv);
                    for k in 0..32 {
                        let d = ram.port_r_rd_data[k];
                        if d == usize::MAX {
                            continue;
                        }
                        input_map.insert(d, sram_rd_data_global_start * 32 + k as u32);
                        output_map.insert(d << 1, sram_rd_data_global_start * 32 + k as u32);
                        self.place_clken_datainv(
                            sram_rd_data_local_offset * 32 + k,
                            perm_r_en_iv,
                            perm_r_en_iv_inv,
                            perm_r_en_iv_set0,
                            0,
                        );
                    }
                    let sram_input_perm_st = (cur_sram_id * 32 * 4) as usize;
                    for k in 0..13 {
                        self.place_sram_duplicate(
                            sram_input_perm_st + k,
                            self.query_permute_with_pin_iv(ram.port_r_addr_iv[k]),
                        );
                        self.place_sram_duplicate(
                            sram_input_perm_st + 16 + k,
                            self.query_permute_with_pin_iv(ram.port_w_addr_iv[k]),
                        );
                    }
                    for k in 0..32 {
                        self.place_sram_duplicate(
                            sram_input_perm_st + 32 + k,
                            self.query_permute_with_pin_iv(ram.port_w_wr_en_iv[k]),
                        );
                        self.place_sram_duplicate(
                            sram_input_perm_st + 64 + k,
                            self.query_permute_with_pin_iv(ram.port_w_wr_data_iv[k]),
                        );
                    }
                    cur_sram_id += 1;
                }
                EndpointGroup::PrimaryOutput(idx_iv) => {
                    if idx_iv <= 1 {
                        // Output tied to constant (0=false, 1=true) - map to position 0
                        // (state buffer bit 0 is always 0; for idx_iv=1, the sim won't use
                        // this since aigpin_iv <= 1 is skipped in GPIO mapping)
                        clilog::warn!(
                            PO_CONST_ERR,
                            "primary output idx_iv={} (tied to constant), skipping",
                            idx_iv
                        );
                        output_map.insert(idx_iv, 0);
                        continue;
                    }
                    let pos = self.state_start * 32
                        + self.get_or_place_output_with_activation(idx_iv, 1) as u32;
                    output_map.insert(idx_iv, pos);
                }
                EndpointGroup::StagedIOPin(idx) => {
                    if idx == 0 {
                        panic!("staged IO pin has zero..??")
                    }
                    let pos = self.state_start * 32
                        + self.get_or_place_output_with_activation(idx << 1, 1) as u32;
                    staged_io_map.insert(idx, pos);
                }
                EndpointGroup::DFF(dff) => {
                    if dff.d_iv == 0 {
                        clilog::warn!(
                            DFF_CONST_ERR,
                            "dff d_iv has zero, not fully optimized netlist. ignoring the error.."
                        );
                        input_map.insert(dff.q, 0);
                        continue;
                    }
                    let pos = self.state_start * 32
                        + self.get_or_place_output_with_activation(dff.d_iv, dff.en_iv) as u32;
                    output_map.insert(dff.d_iv, pos);
                    input_map.insert(dff.q, pos);
                }
                EndpointGroup::SimControl(ctrl) => {
                    // SimControl condition is like a primary output - always active
                    if ctrl.condition_iv == 0 {
                        continue;
                    }
                    let local_pos =
                        self.get_or_place_output_with_activation(ctrl.condition_iv, 1) as u32;
                    // Note: The position returned by get_or_place_output_with_activation is
                    // 1-indexed from how the combinational logic is scheduled. We need to
                    // convert to 0-indexed by subtracting 1.
                    // TODO: investigate root cause of off-by-one
                    let pos = self.state_start * 32 + local_pos.saturating_sub(1);
                    output_map.insert(ctrl.condition_iv, pos);
                }
                EndpointGroup::Display(disp) => {
                    // Display enable condition
                    if disp.enable_iv == 0 {
                        continue;
                    }
                    let local_pos =
                        self.get_or_place_output_with_activation(disp.enable_iv, 1) as u32;
                    let pos = self.state_start * 32 + local_pos.saturating_sub(1);
                    output_map.insert(disp.enable_iv, pos);

                    // Also place argument signals
                    for &arg_iv in &disp.args_iv {
                        if arg_iv > 1 {
                            let local_pos =
                                self.get_or_place_output_with_activation(arg_iv, 1) as u32;
                            let pos = self.state_start * 32 + local_pos.saturating_sub(1);
                            output_map.insert(arg_iv, pos);
                        }
                    }
                }
            }
        }
        assert_eq!(cur_sram_id, self.num_srams);
        assert_eq!(
            (self.cnt_placed_duplicate_permute + 31) / 32,
            self.num_duplicate_writeouts
        );

        // println!("test clken_permute: {:?}, wos (w/o sram or dup): {:?}", self.clken_permute, self.parts_after_writeouts);
    }

    /// Build the GPU script for this partition.
    /// Returns (script_data, delay_patch_entries) where each delay_patch_entry
    /// is (local_offset, vec_of_aigpins) for patching timing data later.
    fn build_script(
        &self,
        aig: &AIG,
        part: &Partition,
        input_map: &IndexMap<usize, u32>,
        staged_io_map: &IndexMap<usize, u32>,
    ) -> (Vec<u32>, Vec<(usize, Vec<usize>)>) {
        let mut script = Vec::<u32>::new();
        let mut delay_patches: Vec<(usize, Vec<usize>)> = Vec::new();

        // metadata
        script.push(part.stages.len() as u32);
        script.push(0);
        script.push(self.num_writeouts);
        script.push(self.state_start);
        script.push(self.num_srams);
        script.push(self.sram_start);
        script.push(0); // [6]=num global read rounds, assigned later
        script.push(self.num_duplicate_writeouts);
        // padding
        while script.len() < 128 {
            script.push(0);
        }
        // final 128: write-out locations
        // compressed 2-1
        let mut last_wo = u32::MAX;
        for (j, bs) in part.stages.iter().enumerate() {
            for &wo in &bs.write_outs {
                let cur_wo = (j as u32) << 8 | (wo as u32);
                if last_wo == u32::MAX {
                    last_wo = cur_wo;
                } else {
                    script.push(last_wo | (cur_wo << 16));
                    last_wo = u32::MAX;
                }
            }
        }
        if last_wo != u32::MAX {
            script.push(last_wo | (((1 << 16) - 1) << 16));
        }
        while script.len() < 256 {
            script.push(u32::MAX);
        }
        // read global (256x32)
        let mut inputs_taken = BTreeMap::<u32, u32>::new();
        for &inp in &part.stages[0].hier[0] {
            if inp == usize::MAX {
                continue;
            }
            match input_map.get(&inp) {
                Some(&pos) => {
                    *inputs_taken.entry(pos >> 5).or_default() |= 1 << (pos & 31);
                }
                None => match staged_io_map.get(&inp) {
                    Some(&pos) => {
                        *inputs_taken.entry((pos >> 5) | (1u32 << 31)).or_default() |=
                            1 << (pos & 31);
                    }
                    None => {
                        panic!("cannot find input pin {}, driver: {:?}, in either primary inputs or staged IOs", inp, aig.drivers[inp]);
                    }
                },
            }
        }
        // clilog::debug!(
        //     "part (?) inputs_taken len {}: {:?}",
        //     inputs_taken.len(),
        //     inputs_taken.iter().map(|(id, val)| format!("{}[{}]", id, val.count_ones())).collect::<Vec<_>>()
        // );
        let rounds_idx_masks = map_global_read_to_rounds(&inputs_taken);
        let num_global_stages = rounds_idx_masks.iter().map(|v| v.len()).max().unwrap() as u32;
        script[6] = num_global_stages;
        assert_eq!(script.len(), NUM_THREADS_V1);
        let global_perm_start = script.len();
        script.extend((0..(2 * num_global_stages as usize * NUM_THREADS_V1)).map(|_| 0));
        for (i, v) in rounds_idx_masks.iter().enumerate() {
            for (round, &(idx, mask)) in v.iter().enumerate() {
                script[global_perm_start + NUM_THREADS_V1 * 2 * round + (i * 2)] = idx;
                script[global_perm_start + NUM_THREADS_V1 * 2 * round + (i * 2 + 1)] = mask;
                // println!("test: round {} i {} idx {} mask {}",
                //          round, i, idx, mask);
            }
        }

        let outputpos2localpos = rounds_idx_masks
            .iter()
            .enumerate()
            .map(|(local_i, v)| {
                let mut local_op2lp = Vec::with_capacity(32);
                let mut bit_id = 0;
                for &(idx, mask) in v.iter().rev() {
                    let is_staged_io = (idx >> 31) != 0;
                    for k in (0..32).rev() {
                        if (mask >> k & 1) != 0 {
                            local_op2lp.push((
                                (is_staged_io, idx << 5 | k),
                                (local_i * 32 + bit_id) as u16,
                            ));
                            bit_id += 1;
                        }
                    }
                }
                assert!(bit_id <= 32);
                local_op2lp.into_iter()
            })
            .flatten()
            .collect::<IndexMap<_, _>>();
        // println!("output2localpos: {:?}", outputpos2localpos);

        let mut last_pin2localpos = IndexMap::new();
        for &inp in &part.stages[0].hier[0] {
            if inp == usize::MAX {
                continue;
            }
            let pos = match input_map.get(&inp) {
                Some(&pos) => (false, pos),
                None => (true, *staged_io_map.get(&inp).unwrap()),
            };
            last_pin2localpos.insert(inp, *outputpos2localpos.get(&pos).unwrap());
        }

        // boomerang sections start
        for (bs_i, bs) in part.stages.iter().enumerate() {
            let bs_perm = bs.hier[0]
                .iter()
                .map(|&pin| {
                    if pin == usize::MAX {
                        0
                    } else {
                        *last_pin2localpos.get(&pin).unwrap()
                    }
                })
                .collect::<Vec<_>>();

            let mut bs_xora = vec![0u32; NUM_THREADS_V1];
            let mut bs_xorb = vec![0u32; NUM_THREADS_V1];
            let mut bs_orb = vec![0u32; NUM_THREADS_V1];
            for hi in 1..bs.hier.len() {
                let hi_len = bs.hier[hi].len();
                for j in 0..hi_len {
                    let out = bs.hier[hi][j];
                    let a = bs.hier[hi - 1][j];
                    let b = bs.hier[hi - 1][j + hi_len];
                    if out == usize::MAX {
                        continue;
                    }
                    if out == a {
                        bs_orb[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                        continue;
                    }
                    let (a_iv, b_iv) = match aig.drivers[out] {
                        DriverType::AndGate(a_iv, b_iv) => (a_iv, b_iv),
                        _ => unreachable!(),
                    };
                    assert_eq!(a_iv >> 1, a);
                    assert_eq!(b_iv >> 1, b);
                    if (a_iv & 1) != 0 {
                        bs_xora[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                    }
                    if (b_iv & 1) != 0 {
                        bs_xorb[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                    }
                }
            }

            for k in 0..4 {
                for i in ((k * 8)..bs_perm.len()).step_by(32) {
                    script.push((bs_perm[i] as u32) | (bs_perm[i + 1] as u32) << 16);
                    script.push((bs_perm[i + 2] as u32) | (bs_perm[i + 3] as u32) << 16);
                    script.push((bs_perm[i + 4] as u32) | (bs_perm[i + 5] as u32) << 16);
                    script.push((bs_perm[i + 6] as u32) | (bs_perm[i + 7] as u32) << 16);
                }
            }
            // Compute per-thread-position delay (max gate delay across all
            // 32 signals evaluated in this thread position).
            // Used by GPU timing kernel for arrival time tracking.
            // Collect AIG pins per thread position for delay injection
            let mut thread_aigpins: Vec<Vec<usize>> = vec![Vec::new(); NUM_THREADS_V1];
            for hi in 1..bs.hier.len() {
                let hi_len = bs.hier[hi].len();
                for j in 0..hi_len {
                    let out = bs.hier[hi][j];
                    if out == usize::MAX {
                        continue;
                    }
                    let thread_pos = (hi_len + j) >> 5;
                    if thread_pos < NUM_THREADS_V1 {
                        thread_aigpins[thread_pos].push(out);
                    }
                }
            }

            for i in 0..NUM_THREADS_V1 {
                script.push(bs_xora[i]);
                script.push(bs_xorb[i]);
                script.push(bs_orb[i]);
                let padding_offset = script.len();
                script.push(0); // Padding slot — patched by inject_timing_to_script()
                if !thread_aigpins[i].is_empty() {
                    delay_patches.push((padding_offset, std::mem::take(&mut thread_aigpins[i])));
                }
            }

            last_pin2localpos = self.afters[bs_i]
                .iter()
                .enumerate()
                .filter_map(|(i, &pin)| {
                    if pin == usize::MAX {
                        None
                    } else {
                        Some((pin, i as u16))
                    }
                })
                .collect::<IndexMap<_, _>>();
        }

        // sram worker
        for k in 0..4 {
            for i in ((k * 8)..self.sram_duplicate_permute.len()).step_by(32) {
                script.push(
                    (self.sram_duplicate_permute[i] as u32)
                        | (self.sram_duplicate_permute[i + 1] as u32) << 16,
                );
                script.push(
                    (self.sram_duplicate_permute[i + 2] as u32)
                        | (self.sram_duplicate_permute[i + 3] as u32) << 16,
                );
                script.push(
                    (self.sram_duplicate_permute[i + 4] as u32)
                        | (self.sram_duplicate_permute[i + 5] as u32) << 16,
                );
                script.push(
                    (self.sram_duplicate_permute[i + 6] as u32)
                        | (self.sram_duplicate_permute[i + 7] as u32) << 16,
                );
            }
        }
        for i in 0..NUM_THREADS_V1 {
            script.push(self.sram_duplicate_inv[i]);
            script.push(self.sram_duplicate_set0[i]);
            script.push(0);
            script.push(0);
        }
        // clock enable signal
        for k in 0..4 {
            for i in ((k * 8)..self.clken_permute.len()).step_by(32) {
                script.push(
                    (self.clken_permute[i] as u32) | (self.clken_permute[i + 1] as u32) << 16,
                );
                script.push(
                    (self.clken_permute[i + 2] as u32) | (self.clken_permute[i + 3] as u32) << 16,
                );
                script.push(
                    (self.clken_permute[i + 4] as u32) | (self.clken_permute[i + 5] as u32) << 16,
                );
                script.push(
                    (self.clken_permute[i + 6] as u32) | (self.clken_permute[i + 7] as u32) << 16,
                );
            }
        }
        for i in 0..NUM_THREADS_V1 {
            script.push(self.clken_inv[i]);
            script.push(self.clken_set0[i]);
            script.push(self.data_inv[i]);
            script.push(0);
        }

        (script, delay_patches)
    }
}

fn build_flattened_script_v1(
    aig: &AIG,
    stageds: &[&StagedAIG],
    parts_in_stages: &[&[Partition]],
    num_blocks: usize,
    input_layout: Vec<usize>,
) -> FlattenedScriptV1 {
    // determine the output position.
    // this is the prerequisite for generating the read
    // permutations and more.
    // input map:
    // locate input pins and FF/SRAM Q's - for partition input
    // output map:
    // locate primary outputs - for circuit outs
    // staged io map:
    // store intermediate nodes between major stages
    let mut input_map = IndexMap::new();
    let mut output_map = IndexMap::new();
    let mut staged_io_map = IndexMap::new();
    let mut delay_patch_map: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, &input) in input_layout.iter().enumerate() {
        if input == usize::MAX {
            continue;
        }
        input_map.insert(input, i as u32);
    }

    let num_major_stages = parts_in_stages.len();

    let states_start = ((input_layout.len() + 31) / 32) as u32;
    let mut sum_state_start = states_start;
    let mut sum_srams_start = 0;

    // enumerate all major stages and build them one by one.

    // #[derive(Debug, Clone, Default)]
    // struct FlatteningStage {
    //     blocks_parts: Vec<Vec<usize>>,
    //     flattening_parts: Vec<FlatteningPart>,
    //     parts_data_split: Vec<Vec<u32>>,
    // }
    // let mut flattening_stages =
    //     Vec::<FlatteningStage>::with_capacity(num_major_stages);

    // assemble script per block.
    let mut blocks_data = Vec::new();
    let mut blocks_start = Vec::<usize>::with_capacity(num_blocks * num_major_stages + 1);
    let mut stages_blocks_parts = Vec::new();
    let mut stages_flattening_parts = Vec::new();

    for (i, (init_parts, &staged)) in parts_in_stages
        .into_iter()
        .copied()
        .zip(stageds.into_iter())
        .enumerate()
    {
        // first arrange parts onto blocks.
        let mut blocks_parts = vec![vec![]; num_blocks];
        let mut tot_nstages_blocks = vec![0; num_blocks];
        // below models the fixed pre&post-cost for each executor
        let executor_fixed_cost = 3;
        // masonry layout of blocks. assume parts are sorted with
        // decreasing order of #stages.
        for i in 0..init_parts.len().min(num_blocks) {
            blocks_parts[i].push(i);
            tot_nstages_blocks[i] = init_parts[i].stages.len() + executor_fixed_cost;
        }
        for i in num_blocks..init_parts.len() {
            let put = tot_nstages_blocks
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.cmp(b))
                .unwrap()
                .0;
            blocks_parts[put].push(i);
            tot_nstages_blocks[put] += init_parts[i].stages.len() + executor_fixed_cost;
        }
        // clilog::debug!("blocks_parts: {:?}", blocks_parts);
        clilog::debug!(
            "major stage {}: max total boomerang depth (w/ cost) {}",
            i,
            tot_nstages_blocks.iter().copied().max().unwrap()
        );

        // the intermediates for parts being flattened
        let mut flattening_parts: Vec<FlatteningPart> = vec![Default::default(); init_parts.len()];

        // basic index preprocessing for stages (parallel - each is independent)
        flattening_parts.par_iter_mut().enumerate().for_each(|(i, fp)| {
            fp.init_afters_writeouts(aig, staged, &init_parts[i]);
        });

        // allocate output state positions for all srams,
        // in the order of block affinity.
        for block in &blocks_parts {
            for &part_id in block {
                flattening_parts[part_id].state_start = sum_state_start;
                sum_state_start += flattening_parts[part_id].num_writeouts;
                flattening_parts[part_id].sram_start = sum_srams_start;
                sum_srams_start +=
                    flattening_parts[part_id].num_srams * (1 << AIGPDK_SRAM_ADDR_WIDTH);
            }
        }

        // besides input ports, we also have outputs from partitions.
        // they include original-placed comb output pins,
        // copied pins for different FF activation,
        // and SRAM read outputs.
        for part_id in 0..init_parts.len() {
            // clilog::debug!("initializing output for part {}", part_id);
            flattening_parts[part_id].make_inputs_outputs(
                aig,
                staged,
                &init_parts[part_id],
                &mut input_map,
                &mut staged_io_map,
                &mut output_map,
            );
        }
        stages_blocks_parts.push(blocks_parts);
        stages_flattening_parts.push(flattening_parts);
    }

    for ((blocks_parts, flattening_parts), init_parts) in stages_blocks_parts
        .iter()
        .zip(stages_flattening_parts.iter_mut())
        .zip(parts_in_stages.into_iter().copied())
    {
        // build script per part in parallel. we will later assemble them to blocks.
        let parts_results: Vec<(Vec<u32>, Vec<(usize, Vec<usize>)>)> = flattening_parts.par_iter()
            .enumerate()
            .map(|(part_id, fp)| {
                fp.build_script(
                    aig,
                    &init_parts[part_id],
                    &input_map,
                    &staged_io_map,
                )
            })
            .collect();

        for block_id in 0..num_blocks {
            blocks_start.push(blocks_data.len());
            if blocks_parts[block_id].is_empty() {
                let mut dummy = vec![0; NUM_THREADS_V1];
                dummy[1] = 1;
                blocks_data.extend(dummy.into_iter());
            } else {
                let num_parts = blocks_parts[block_id].len();
                let mut last_part_st = usize::MAX;
                for (i, &part_id) in blocks_parts[block_id].iter().enumerate() {
                    if i == num_parts - 1 {
                        last_part_st = blocks_data.len();
                    }
                    let base_offset = blocks_data.len();
                    let (ref script_data, ref patches) = parts_results[part_id];
                    blocks_data.extend(script_data.iter().copied());
                    // Adjust patch offsets from local to global blocks_data position
                    for (local_off, aigpins) in patches {
                        delay_patch_map.push((base_offset + local_off, aigpins.clone()));
                    }
                }
                assert_ne!(last_part_st, usize::MAX);
                blocks_data[last_part_st + 1] = 1;
            }
        }
    }
    blocks_start.push(blocks_data.len());
    blocks_data.extend((0..NUM_THREADS_V1 * 8).map(|_| 0)); // padding

    clilog::info!(
        "Built script for {} blocks, reg/io state size {}, sram size {}, script size {}",
        num_blocks,
        sum_state_start,
        sum_srams_start,
        blocks_data.len()
    );

    // Build assertion_positions from simcontrols
    let assertion_positions: Vec<_> = aig
        .simcontrols
        .iter()
        .filter_map(|(&cell_id, ctrl)| {
            // Get the position of the condition in the output state
            output_map
                .get(&ctrl.condition_iv)
                .map(|&pos| (cell_id, pos, ctrl.message_id, ctrl.control_type))
        })
        .collect();

    if !assertion_positions.is_empty() {
        clilog::info!(
            "Found {} assertion/simcontrol nodes",
            assertion_positions.len()
        );
        for &(cell_id, pos, msg_id, ctrl_type) in &assertion_positions {
            clilog::debug!(
                "  Assertion: cell={}, pos={} (word={}, bit={}), msg_id={}, type={:?}",
                cell_id,
                pos,
                pos >> 5,
                pos & 31,
                msg_id,
                ctrl_type
            );
        }
    }

    // Build display_positions from displays
    let display_positions: Vec<_> = aig
        .displays
        .iter()
        .filter_map(|(&cell_id, disp)| {
            output_map.get(&disp.enable_iv).map(|&enable_pos| {
                // Get positions for all argument signals
                let arg_positions: Vec<u32> = disp
                    .args_iv
                    .iter()
                    .filter_map(|&arg_iv| output_map.get(&arg_iv).copied())
                    .collect();
                (
                    cell_id,
                    enable_pos,
                    disp.format.clone(),
                    arg_positions,
                    disp.arg_widths.clone(),
                )
            })
        })
        .collect();

    if !display_positions.is_empty() {
        clilog::info!("Found {} display nodes", display_positions.len());
        for (cell_id, pos, format, arg_pos, _) in &display_positions {
            clilog::debug!(
                "  Display: cell={}, enable_pos={} (word={}, bit={}), format='{}', args={:?}",
                cell_id,
                pos,
                pos >> 5,
                pos & 31,
                format,
                arg_pos
            );
        }
    }

    FlattenedScriptV1 {
        num_blocks,
        num_major_stages,
        blocks_start: blocks_start.into(),
        blocks_data: blocks_data.into(),
        reg_io_state_size: sum_state_start,
        sram_storage_size: sum_srams_start,
        input_layout,
        input_map,
        output_map,
        stages_blocks_parts,
        assertion_positions,
        display_positions,
        // Timing fields - initialized empty, populated by load_timing()
        gate_delays: Vec::new(),
        dff_constraints: Vec::new(),
        clock_period_ps: 1000, // Default 1ns
        timing_enabled: false,
        delay_patch_map,
    }
}

impl FlattenedScriptV1 {
    /// build a flattened script.
    ///
    /// `init_parts` give the partitions to flatten.
    /// it is better sorted in advance in descending order
    /// of #layers for better duty cycling.
    ///
    /// `num_blocks` should be set to the hardware allowances,
    /// i.e. the number of SMs in your GPU.
    /// for example, A100 should set it to 108.
    ///
    /// `input_layout` should give the expected primary input
    /// memory layout, each one is an AIG bit index.
    /// padding bits should be set to usize::MAX.
    pub fn from(
        aig: &AIG,
        stageds: &[&StagedAIG],
        parts_in_stages: &[&[Partition]],
        num_blocks: usize,
        input_layout: Vec<usize>,
    ) -> FlattenedScriptV1 {
        build_flattened_script_v1(aig, stageds, parts_in_stages, num_blocks, input_layout)
    }

    /// Load timing data from a Liberty library and AIG.
    ///
    /// This populates gate_delays and dff_constraints for timing-aware simulation.
    /// Must be called after building the script if timing analysis is needed.
    pub fn load_timing(&mut self, aig: &AIG, lib: &TimingLibrary, clock_period_ps: u64) {
        // Get default delays from library
        let and_delay = lib.and_gate_delay("AND2_00_0").unwrap_or((1, 1));
        let dff_timing = lib.dff_timing();
        let sram_timing = lib.sram_timing();

        // Initialize gate delays for all AIG pins
        self.gate_delays = vec![PackedDelay::default(); aig.num_aigpins + 1];

        for i in 1..=aig.num_aigpins {
            let delay = match &aig.drivers[i] {
                DriverType::AndGate(_, _) => {
                    PackedDelay::from_u64(and_delay.0, and_delay.1)
                }
                DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) | DriverType::Tie0 => {
                    PackedDelay::default() // Zero delay for inputs
                }
                DriverType::DFF(_) => {
                    // Clock-to-Q delay
                    dff_timing
                        .as_ref()
                        .map(|t| PackedDelay::from_u64(t.clk_to_q_rise_ps, t.clk_to_q_fall_ps))
                        .unwrap_or_default()
                }
                DriverType::SRAM(_) => {
                    // SRAM read delay
                    sram_timing
                        .as_ref()
                        .map(|t| PackedDelay::from_u64(t.read_clk_to_data_rise_ps, t.read_clk_to_data_fall_ps))
                        .unwrap_or(PackedDelay::new(1, 1))
                }
            };
            self.gate_delays[i] = delay;
        }

        // Build DFF constraints
        self.dff_constraints = Vec::with_capacity(aig.dffs.len());
        let setup_time = dff_timing.as_ref().map(|t| t.max_setup()).unwrap_or(0) as u16;
        let hold_time = dff_timing.as_ref().map(|t| t.max_hold()).unwrap_or(0) as u16;

        for (&cell_id, dff) in &aig.dffs {
            // Find the state position for the D input
            let data_state_pos = self
                .output_map
                .get(&dff.d_iv)
                .copied()
                .unwrap_or(u32::MAX);

            self.dff_constraints.push(DFFConstraint {
                setup_ps: setup_time,
                hold_ps: hold_time,
                data_state_pos,
                cell_id: cell_id as u32,
            });
        }

        self.clock_period_ps = clock_period_ps;
        self.timing_enabled = true;

        clilog::info!(
            "Loaded timing: {} gate delays, {} DFF constraints, clock={}ps",
            self.gate_delays.len(),
            self.dff_constraints.len(),
            clock_period_ps
        );
    }

    /// Get the packed delay for an AIG pin (by index).
    pub fn get_delay(&self, aigpin: usize) -> PackedDelay {
        if aigpin < self.gate_delays.len() {
            self.gate_delays[aigpin]
        } else {
            PackedDelay::default()
        }
    }

    /// Check if timing data is available.
    pub fn has_timing(&self) -> bool {
        self.timing_enabled
    }

    /// Inject timing delay data into the GPU script's padding slots.
    /// Must be called after load_timing() or load_timing_from_sdf().
    /// Each padding u32 gets the raw max gate delay (in picoseconds, clamped
    /// to u16 range 0-65535) across all AIG pins mapped to that thread position.
    pub fn inject_timing_to_script(&mut self) {
        if self.gate_delays.is_empty() || self.delay_patch_map.is_empty() {
            return;
        }

        let mut patched = 0usize;
        for (offset, aigpins) in &self.delay_patch_map {
            let mut max_delay: u32 = 0;
            for &aigpin in aigpins {
                if aigpin < self.gate_delays.len() {
                    max_delay = max_delay.max(self.gate_delays[aigpin].max_delay() as u32);
                }
            }
            // Store raw picoseconds, clamped to u16 range
            let delay_ps = max_delay.min(65535);
            if *offset < self.blocks_data.len() {
                self.blocks_data[*offset] = delay_ps;
                if delay_ps > 0 {
                    patched += 1;
                }
            }
        }

        clilog::info!(
            "Injected timing to GPU script: {} padding slots patched ({} total)",
            patched,
            self.delay_patch_map.len()
        );
    }

    /// Build a per-word timing constraint buffer for GPU-side setup/hold checking.
    ///
    /// Returns `(clock_period_ps, constraints)` where `constraints` has one u32 per
    /// state word (`reg_io_state_size` words). Each u32 packs:
    ///   - bits [31:16] = min setup_ps across all DFFs in that word
    ///   - bits [15:0]  = min hold_ps across all DFFs in that word
    ///
    /// Words with no DFF constraints have value 0 (skipped by the kernel).
    /// This is conservative: the max arrival across the word is compared against the
    /// min constraint, which may over-report violations but never misses real ones.
    pub fn build_timing_constraint_buffer(&self) -> (u32, Vec<u32>) {
        let num_words = self.reg_io_state_size as usize;
        let mut constraints = vec![0u32; num_words];
        for c in &self.dff_constraints {
            if c.data_state_pos == u32::MAX {
                continue;
            }
            let word_idx = (c.data_state_pos / 32) as usize;
            if word_idx >= num_words {
                continue;
            }
            let existing = constraints[word_idx];
            let old_setup = (existing >> 16) as u16;
            let old_hold = (existing & 0xFFFF) as u16;
            // Most restrictive (min) constraint per word, treating 0 as "not yet set"
            let new_setup = if old_setup == 0 {
                c.setup_ps
            } else {
                old_setup.min(c.setup_ps)
            };
            let new_hold = if old_hold == 0 {
                c.hold_ps
            } else {
                old_hold.min(c.hold_ps)
            };
            constraints[word_idx] = ((new_setup as u32) << 16) | (new_hold as u32);
        }
        let clock_ps = self.clock_period_ps.min(u32::MAX as u64) as u32;
        (clock_ps, constraints)
    }

    /// Load timing from an SDF file with per-instance back-annotated delays.
    ///
    /// This replaces the uniform Liberty-based delays with per-instance IOPATH
    /// delays from post-layout SDF. Uses `aigpin_cell_origins` to map SDF instances
    /// to AIG pins. For pins with multiple origins (e.g., an inverter chain sharing
    /// an AIG pin via invert-bit reuse), delays are summed since they form a serial chain.
    ///
    /// For decomposed cells (e.g., SKY130 nand2 → 1 AND gate), the full IOPATH
    /// delay is applied to the output AIG pin. Internal AND gates get zero delay.
    ///
    /// Optionally falls back to Liberty for cells not found in SDF.
    pub fn load_timing_from_sdf(
        &mut self,
        aig: &AIG,
        netlistdb: &NetlistDB,
        sdf: &SdfFile,
        clock_period_ps: u64,
        liberty_fallback: Option<&TimingLibrary>,
        debug: bool,
    ) {
        // Build cell_id → SDF hierarchical instance path mapping.
        // SDF uses dot-separated paths: "u_cpu.alu.and_gate"
        // NetlistDB HierName iterates leaf-first, so we reverse and join with dots.
        let mut cellid_to_sdf_path: HashMap<usize, String> = HashMap::new();
        for cellid in 1..netlistdb.num_cells {
            let parts: Vec<&str> = netlistdb.cellnames[cellid].iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>();
            // parts is leaf-first, reverse to get root-first, then join with '.'
            let sdf_path: String = parts.iter().rev().cloned().collect::<Vec<_>>().join(".");
            cellid_to_sdf_path.insert(cellid, sdf_path);
        }

        // Build reverse map: SDF path → cellid for efficient lookup
        let mut sdf_path_to_cellid: HashMap<&str, usize> = HashMap::new();
        for (&cellid, path) in &cellid_to_sdf_path {
            sdf_path_to_cellid.insert(path.as_str(), cellid);
        }

        // Phase 2: Build wire delay map from INTERCONNECT entries.
        // For each dest instance, collect max wire delay across all input pins.
        // Key: (dest_cellid) → max SdfDelay across all input wires.
        let mut wire_delays_per_cell: HashMap<usize, crate::sdf_parser::SdfDelay> = HashMap::new();
        for cell in &sdf.cells {
            for ic in &cell.interconnects {
                // INTERCONNECT dest format: "instance.pin" or "instance.subinst.pin"
                // Extract the dest instance (everything before the last '.')
                if let Some(dot_pos) = ic.dest.rfind('.') {
                    let dest_inst = &ic.dest[..dot_pos];
                    if let Some(&dest_cellid) = sdf_path_to_cellid.get(dest_inst) {
                        let entry = wire_delays_per_cell
                            .entry(dest_cellid)
                            .or_insert(crate::sdf_parser::SdfDelay { rise_ps: 0, fall_ps: 0 });
                        entry.rise_ps = entry.rise_ps.max(ic.delay.rise_ps);
                        entry.fall_ps = entry.fall_ps.max(ic.delay.fall_ps);
                    }
                }
            }
        }
        let num_wire_delays = wire_delays_per_cell.len();

        // Initialize gate delays
        self.gate_delays = vec![PackedDelay::default(); aig.num_aigpins + 1];

        let mut matched = 0usize;
        let mut unmatched = 0usize;
        let mut fallback_count = 0usize;
        let mut unmatched_instances: Vec<String> = Vec::new();

        // Get Liberty fallback delays
        let lib_and_delay = liberty_fallback
            .and_then(|lib| lib.and_gate_delay("AND2_00_0"))
            .unwrap_or((0, 0));
        let lib_dff_timing = liberty_fallback.and_then(|lib| lib.dff_timing());
        let lib_sram_timing = liberty_fallback.and_then(|lib| lib.sram_timing());

        for aigpin in 1..=aig.num_aigpins {
            let origins = &aig.aigpin_cell_origins[aigpin];
            let delay = if !origins.is_empty() {
                // This AIG pin has one or more cell origins — sum delays from all.
                // Multiple origins arise when inverters share an AIG pin (invert-bit reuse).
                // They form a serial chain, so delays are additive.
                let mut total_rise: u64 = 0;
                let mut total_fall: u64 = 0;
                let mut any_matched = false;
                let mut any_unmatched = false;

                for (cellid, _cell_type, output_pin_name) in origins {
                    if let Some(sdf_path) = cellid_to_sdf_path.get(cellid) {
                        if let Some(sdf_cell) = sdf.get_cell(sdf_path) {
                            let iopath = sdf_cell.iopaths.iter()
                                .find(|p| p.output_pin == *output_pin_name)
                                .or_else(|| sdf_cell.iopaths.first());
                            if let Some(iopath) = iopath {
                                any_matched = true;
                                let wire = wire_delays_per_cell.get(cellid);
                                total_rise += iopath.delay.rise_ps + wire.map_or(0, |w| w.rise_ps);
                                total_fall += iopath.delay.fall_ps + wire.map_or(0, |w| w.fall_ps);
                            } else {
                                any_unmatched = true;
                                if debug && unmatched_instances.len() < 20 {
                                    unmatched_instances.push(format!("{} (no IOPATH)", sdf_path));
                                }
                            }
                        } else {
                            any_unmatched = true;
                            if debug && unmatched_instances.len() < 20 {
                                unmatched_instances.push(sdf_path.clone());
                            }
                        }
                    }
                }

                if any_matched {
                    matched += 1;
                    if any_unmatched { unmatched += 1; }
                    PackedDelay::from_u64(total_rise, total_fall)
                } else {
                    unmatched += 1;
                    self.get_liberty_fallback_delay(
                        &aig.drivers[aigpin],
                        lib_and_delay,
                        &lib_dff_timing,
                        &lib_sram_timing,
                        &mut fallback_count,
                    )
                }
            } else {
                // No cell origins — internal decomposition gate or input/tie
                match &aig.drivers[aigpin] {
                    DriverType::AndGate(_, _) => {
                        // Internal AND gate from cell decomposition — zero delay.
                        PackedDelay::default()
                    }
                    DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) | DriverType::Tie0 => {
                        PackedDelay::default()
                    }
                    DriverType::DFF(_) => {
                        self.get_liberty_fallback_delay(
                            &aig.drivers[aigpin],
                            lib_and_delay,
                            &lib_dff_timing,
                            &lib_sram_timing,
                            &mut fallback_count,
                        )
                    }
                    DriverType::SRAM(_) => {
                        self.get_liberty_fallback_delay(
                            &aig.drivers[aigpin],
                            lib_and_delay,
                            &lib_dff_timing,
                            &lib_sram_timing,
                            &mut fallback_count,
                        )
                    }
                }
            };
            self.gate_delays[aigpin] = delay;
        }

        // Build DFF constraints from SDF timing checks
        self.dff_constraints = Vec::with_capacity(aig.dffs.len());
        let lib_setup = lib_dff_timing.as_ref().map(|t| t.max_setup()).unwrap_or(0) as u16;
        let lib_hold = lib_dff_timing.as_ref().map(|t| t.max_hold()).unwrap_or(0) as u16;

        for (&cell_id, dff) in &aig.dffs {
            let data_state_pos = self
                .output_map
                .get(&dff.d_iv)
                .copied()
                .unwrap_or(u32::MAX);

            // Try to get setup/hold from SDF timing checks
            let (setup_ps, hold_ps) = if let Some(sdf_path) = cellid_to_sdf_path.get(&cell_id) {
                if let Some(sdf_cell) = sdf.get_cell(sdf_path) {
                    let setup = sdf_cell.timing_checks.iter()
                        .find(|c| c.check_type == crate::sdf_parser::TimingCheckType::Setup)
                        .map(|c| c.value_ps.max(0) as u16)
                        .unwrap_or(lib_setup);
                    let hold = sdf_cell.timing_checks.iter()
                        .find(|c| c.check_type == crate::sdf_parser::TimingCheckType::Hold)
                        .map(|c| c.value_ps.max(0) as u16)
                        .unwrap_or(lib_hold);
                    (setup, hold)
                } else {
                    (lib_setup, lib_hold)
                }
            } else {
                (lib_setup, lib_hold)
            };

            self.dff_constraints.push(DFFConstraint {
                setup_ps,
                hold_ps,
                data_state_pos,
                cell_id: cell_id as u32,
            });
        }

        self.clock_period_ps = clock_period_ps;
        self.timing_enabled = true;

        clilog::info!(
            "Loaded SDF timing: {} matched, {} unmatched ({} Liberty fallback), {} wire delays, {} DFF constraints, clock={}ps",
            matched, unmatched, fallback_count, num_wire_delays, self.dff_constraints.len(), clock_period_ps
        );

        if debug && !unmatched_instances.is_empty() {
            clilog::warn!("SDF unmatched instances (first {}):", unmatched_instances.len());
            for inst in &unmatched_instances {
                clilog::warn!("  {}", inst);
            }
        }
    }

    /// Get a fallback delay from Liberty for a given driver type.
    fn get_liberty_fallback_delay(
        &self,
        driver: &DriverType,
        and_delay: (u64, u64),
        dff_timing: &Option<crate::liberty_parser::DFFTiming>,
        sram_timing: &Option<crate::liberty_parser::SRAMTiming>,
        fallback_count: &mut usize,
    ) -> PackedDelay {
        *fallback_count += 1;
        match driver {
            DriverType::AndGate(_, _) => {
                PackedDelay::from_u64(and_delay.0, and_delay.1)
            }
            DriverType::DFF(_) => {
                dff_timing
                    .as_ref()
                    .map(|t| PackedDelay::from_u64(t.clk_to_q_rise_ps, t.clk_to_q_fall_ps))
                    .unwrap_or_default()
            }
            DriverType::SRAM(_) => {
                sram_timing
                    .as_ref()
                    .map(|t| PackedDelay::from_u64(t.read_clk_to_data_rise_ps, t.read_clk_to_data_fall_ps))
                    .unwrap_or(PackedDelay::new(1, 1))
            }
            _ => PackedDelay::default(),
        }
    }
}

#[cfg(test)]
mod sdf_delay_tests {
    use super::*;
    use crate::aig::AIG;
    use crate::sky130::SKY130LeafPins;
    use crate::sdf_parser::{SdfFile, SdfCorner};

    /// Helper: build a minimal FlattenedScriptV1 suitable for load_timing_from_sdf.
    fn make_minimal_script(_aig: &AIG) -> FlattenedScriptV1 {
        FlattenedScriptV1 {
            num_blocks: 0,
            num_major_stages: 0,
            blocks_start: Vec::<usize>::new().into(),
            blocks_data: Vec::<u32>::new().into(),
            reg_io_state_size: 0,
            sram_storage_size: 0,
            input_layout: Vec::new(),
            output_map: IndexMap::new(),
            input_map: IndexMap::new(),
            stages_blocks_parts: Vec::new(),
            assertion_positions: Vec::new(),
            display_positions: Vec::new(),
            gate_delays: Vec::new(),
            dff_constraints: Vec::new(),
            clock_period_ps: 1000,
            timing_enabled: false,
            delay_patch_map: Vec::new(),
        }
    }

    /// Helper: load inv_chain design + AIG.
    fn load_inv_chain() -> (netlistdb::NetlistDB, AIG) {
        let path = std::path::PathBuf::from("tests/timing_test/sky130_timing/inv_chain.v");
        assert!(path.exists(), "inv_chain.v not found");
        let netlistdb = netlistdb::NetlistDB::from_sverilog_file(
            &path, None, &SKY130LeafPins,
        ).expect("Failed to parse inv_chain.v");
        let aig = AIG::from_netlistdb(&netlistdb);
        (netlistdb, aig)
    }

    /// Helper: build cellid → SDF path map (same logic as load_timing_from_sdf).
    fn build_cellid_to_sdf_path(netlistdb: &netlistdb::NetlistDB) -> HashMap<usize, String> {
        let mut map = HashMap::new();
        for cellid in 1..netlistdb.num_cells {
            let parts: Vec<&str> = netlistdb.cellnames[cellid].iter()
                .map(|s| s.as_str()).collect::<Vec<_>>();
            let sdf_path: String = parts.iter().rev().cloned().collect::<Vec<_>>().join(".");
            map.insert(cellid, sdf_path);
        }
        map
    }

    // === Test 3: SDF delay application ===

    #[test]
    fn test_sdf_delay_application() {
        let (netlistdb, aig) = load_inv_chain();
        let sdf_content = include_str!("../tests/timing_test/inv_chain_pnr/inv_chain_test.sdf");
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ)
            .expect("Failed to parse SDF");

        let mut script = make_minimal_script(&aig);
        script.load_timing_from_sdf(&aig, &netlistdb, &sdf, 10000, None, true);

        assert_eq!(script.gate_delays.len(), aig.num_aigpins + 1,
            "gate_delays length should match num_aigpins + 1");

        // With accumulated origins, delays from all cells sharing an AIG pin are summed.
        // For inv_chain.v:
        //   - Shared chain pin (dff_in.Q): 1 DFF + 16 inverters → summed delays
        //   - dff_out.Q: standalone DFF with its own AIG pin

        let cellid_to_sdf_path = build_cellid_to_sdf_path(&netlistdb);
        let sdf_path_to_cellid: HashMap<&str, usize> = cellid_to_sdf_path.iter()
            .map(|(&cid, path)| (path.as_str(), cid))
            .collect();

        // Find the AIG pin that has the dff_out origin
        let dff_out_cellid = *sdf_path_to_cellid.get("dff_out").unwrap();
        let dff_out_aigpin = (1..=aig.num_aigpins)
            .find(|&ap| aig.aigpin_cell_origins[ap].iter().any(|(cid, _, _)| *cid == dff_out_cellid))
            .expect("dff_out should have a cell origin entry");

        let dff_out_delay = &script.gate_delays[dff_out_aigpin];
        // dff_out: IOPATH CLK Q rise=360ps, fall=340ps + wire (i15.Y→dff_out.D) rise=15ps, fall=12ps
        assert_eq!(dff_out_delay.rise_ps, 375,
            "dff_out rise: expected 375ps (360+15), got {}ps", dff_out_delay.rise_ps);
        assert_eq!(dff_out_delay.fall_ps, 352,
            "dff_out fall: expected 352ps (340+12), got {}ps", dff_out_delay.fall_ps);

        // The shared chain pin now accumulates all 17 origins (1 DFF + 16 inverters).
        // Total delay = sum of all IOPATH + wire delays in the chain.
        let chain_aigpin = (1..=aig.num_aigpins)
            .find(|&ap| aig.aigpin_cell_origins[ap].len() > 1)
            .expect("Should have a pin with multiple origins (the inverter chain)");
        let chain_delay = &script.gate_delays[chain_aigpin];

        // Accumulated: dff_in CLK→Q + 16 inverters + their wire delays
        // Expected: rise=1323ps, fall=1125ps (see test_accumulated_delay_analytical)
        assert_eq!(chain_delay.rise_ps, 1323,
            "chain pin rise: expected 1323ps (accumulated), got {}ps", chain_delay.rise_ps);
        assert_eq!(chain_delay.fall_ps, 1125,
            "chain pin fall: expected 1125ps (accumulated), got {}ps", chain_delay.fall_ps);

        // Verify timing is enabled after loading
        assert!(script.timing_enabled, "timing_enabled should be true after load_timing_from_sdf");
        assert_eq!(script.clock_period_ps, 10000);
    }

    #[test]
    fn test_internal_and_gates_zero_delay() {
        let (netlistdb, aig) = load_inv_chain();
        let sdf_content = include_str!("../tests/timing_test/inv_chain_pnr/inv_chain_test.sdf");
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ)
            .expect("Failed to parse SDF");

        let mut script = make_minimal_script(&aig);
        script.load_timing_from_sdf(&aig, &netlistdb, &sdf, 10000, None, false);

        for aigpin in 1..=aig.num_aigpins {
            if aig.aigpin_cell_origins[aigpin].is_empty() {
                let delay = &script.gate_delays[aigpin];
                assert_eq!(
                    delay.rise_ps, 0,
                    "AND gate aigpin {} without cell_origins should have zero rise delay, got {}",
                    aigpin, delay.rise_ps
                );
                assert_eq!(
                    delay.fall_ps, 0,
                    "AND gate aigpin {} without cell_origins should have zero fall delay, got {}",
                    aigpin, delay.fall_ps
                );
            }
        }
    }

    // === Test 3b: Analytical validation of accumulated delay ===

    #[test]
    fn test_accumulated_delay_analytical() {
        // Verify accumulated delay matches hand-computed SDF sum.
        // SDF typ corner values (middle of min:typ:max triples):
        //
        // Chain pin (dff_in.Q shared with inverters via invert-bit reuse):
        //   dff_in CLK→Q:  IOPATH (350, 330) + wire (0, 0)     = (350, 330)
        //   i0  A→Y:       IOPATH (50, 40)   + wire (15, 12)   = (65, 52)
        //   i1  A→Y:       IOPATH (52, 42)   + wire (8, 7)     = (60, 49)
        //   i2  A→Y:       IOPATH (51, 41)   + wire (9, 8)     = (60, 49)
        //   i3  A→Y:       IOPATH (53, 43)   + wire (8, 7)     = (61, 50)
        //   i4  A→Y:       IOPATH (50, 40)   + wire (10, 9)    = (60, 49)
        //   i5  A→Y:       IOPATH (54, 44)   + wire (8, 7)     = (62, 51)
        //   i6  A→Y:       IOPATH (51, 41)   + wire (9, 8)     = (60, 49)
        //   i7  A→Y:       IOPATH (52, 42)   + wire (8, 7)     = (60, 49)
        //   i8  A→Y:       IOPATH (50, 40)   + wire (11, 10)   = (61, 50)
        //   i9  A→Y:       IOPATH (53, 43)   + wire (8, 7)     = (61, 50)
        //   i10 A→Y:       IOPATH (51, 41)   + wire (9, 8)     = (60, 49)
        //   i11 A→Y:       IOPATH (54, 44)   + wire (8, 7)     = (62, 51)
        //   i12 A→Y:       IOPATH (52, 42)   + wire (10, 9)    = (62, 51)
        //   i13 A→Y:       IOPATH (51, 41)   + wire (8, 7)     = (59, 48)
        //   i14 A→Y:       IOPATH (50, 40)   + wire (9, 8)     = (59, 48)
        //   i15 A→Y:       IOPATH (53, 43)   + wire (8, 7)     = (61, 50)
        //   ─────────────────────────────────────────────────────────────
        //   Sum:                                                  (1323, 1125)
        //
        // dff_out (separate AIG pin):
        //   dff_out CLK→Q: IOPATH (360, 340) + wire (15, 12)   = (375, 352)
        //
        // This test validates that the SDF-to-delay pipeline produces exactly
        // these values, confirming the accumulation fix works correctly.

        let (netlistdb, aig) = load_inv_chain();
        let sdf_content = include_str!("../tests/timing_test/inv_chain_pnr/inv_chain_test.sdf");
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ)
            .expect("Failed to parse SDF");

        let mut script = make_minimal_script(&aig);
        script.load_timing_from_sdf(&aig, &netlistdb, &sdf, 10000, None, false);

        // Find the chain pin (has 17 origins: 1 DFF + 16 inverters)
        let chain_aigpin = (1..=aig.num_aigpins)
            .find(|&ap| aig.aigpin_cell_origins[ap].len() == 17)
            .expect("Should have a pin with 17 origins (1 DFF + 16 inverters)");

        let chain_delay = &script.gate_delays[chain_aigpin];
        assert_eq!(chain_delay.rise_ps, 1323,
            "Analytical chain rise: 350 + sum(16 inverter+wire) = 1323ps, got {}ps",
            chain_delay.rise_ps);
        assert_eq!(chain_delay.fall_ps, 1125,
            "Analytical chain fall: 330 + sum(16 inverter+wire) = 1125ps, got {}ps",
            chain_delay.fall_ps);

        // Verify each inverter's individual contribution by checking origins
        let origins = &aig.aigpin_cell_origins[chain_aigpin];
        assert_eq!(origins.len(), 17);

        // First origin should be the DFF
        let (_, ref ct, _) = origins[0];
        assert_eq!(ct, "dfxtp", "First origin should be the DFF");

        // Remaining 16 should be inverters
        for i in 1..=16 {
            let (_, ref ct, ref pin) = origins[i];
            assert_eq!(ct, "inv", "Origin {} should be an inverter", i);
            assert_eq!(pin, "Y", "Inverter output pin should be Y");
        }

        // dff_out verification
        let cellid_to_sdf_path = build_cellid_to_sdf_path(&netlistdb);
        let dff_out_cellid = cellid_to_sdf_path.iter()
            .find(|(_, path)| path.as_str() == "dff_out")
            .map(|(&cid, _)| cid)
            .expect("dff_out should exist");
        let dff_out_aigpin = (1..=aig.num_aigpins)
            .find(|&ap| aig.aigpin_cell_origins[ap].iter().any(|(cid, _, _)| *cid == dff_out_cellid))
            .expect("dff_out should have a cell origin");

        let dff_out_delay = &script.gate_delays[dff_out_aigpin];
        assert_eq!(dff_out_delay.rise_ps, 375,
            "Analytical dff_out rise: 360 + 15 = 375ps, got {}ps", dff_out_delay.rise_ps);
        assert_eq!(dff_out_delay.fall_ps, 352,
            "Analytical dff_out fall: 340 + 12 = 352ps, got {}ps", dff_out_delay.fall_ps);
    }

    // === Test 4: GPU delay injection quantization ===

    /// Helper: build a FlattenedScriptV1 for delay injection testing.
    /// `delays` are per-AIG-pin (index 0 is always default/Tie0).
    /// `patch_map` maps offsets in blocks_data to AIG pin indices.
    /// `blocks_data_size` is how many u32 slots to allocate.
    fn make_delay_injection_script(
        delays: Vec<PackedDelay>,
        patch_map: Vec<(usize, Vec<usize>)>,
        blocks_data_size: usize,
    ) -> FlattenedScriptV1 {
        FlattenedScriptV1 {
            num_blocks: 0,
            num_major_stages: 0,
            blocks_start: Vec::<usize>::new().into(),
            blocks_data: vec![0u32; blocks_data_size].into(),
            reg_io_state_size: 0,
            sram_storage_size: 0,
            input_layout: Vec::new(),
            output_map: IndexMap::new(),
            input_map: IndexMap::new(),
            stages_blocks_parts: Vec::new(),
            assertion_positions: Vec::new(),
            display_positions: Vec::new(),
            gate_delays: delays,
            dff_constraints: Vec::new(),
            clock_period_ps: 10000,
            timing_enabled: true,
            delay_patch_map: patch_map,
        }
    }

    #[test]
    fn test_basic_delay_injection() {
        // delay=100ps → stored as raw 100
        let mut script = make_delay_injection_script(
            vec![PackedDelay::default(), PackedDelay::new(100, 100)],
            vec![(0, vec![1])],
            1,
        );
        script.inject_timing_to_script();
        assert_eq!(script.blocks_data[0], 100, "100ps stored as raw picoseconds");
    }

    #[test]
    fn test_max_of_thread_position() {
        // Two aigpins [350ps, 0ps] → max=350, stored as raw 350
        let mut script = make_delay_injection_script(
            vec![PackedDelay::default(), PackedDelay::new(350, 350), PackedDelay::new(0, 0)],
            vec![(0, vec![1, 2])],
            1,
        );
        script.inject_timing_to_script();
        assert_eq!(script.blocks_data[0], 350, "max(350,0)=350 raw ps");
    }

    #[test]
    fn test_zero_stays_zero() {
        let mut script = make_delay_injection_script(
            vec![PackedDelay::default(), PackedDelay::new(0, 0)],
            vec![(0, vec![1])],
            1,
        );
        script.inject_timing_to_script();
        assert_eq!(script.blocks_data[0], 0, "Zero delay stays zero");
    }

    #[test]
    fn test_ordering_preserved() {
        // delays 5ps and 15ps → stored as raw 5 and 15
        let mut script = make_delay_injection_script(
            vec![PackedDelay::default(), PackedDelay::new(5, 5), PackedDelay::new(15, 15)],
            vec![(0, vec![1]), (1, vec![2])],
            2,
        );
        script.inject_timing_to_script();

        let d1 = script.blocks_data[0];
        let d2 = script.blocks_data[1];
        assert_eq!(d1, 5, "5ps stored as raw picoseconds");
        assert_eq!(d2, 15, "15ps stored as raw picoseconds");
        assert!(d1 < d2, "Ordering must be preserved: {} < {}", d1, d2);
    }

    // === Test 5: IOPATH pin name fallback ===

    #[test]
    fn test_iopath_exact_match() {
        // Cell with IOPATH output "Y" matching cell_origin "Y" → exact match
        use crate::sdf_parser::*;

        let sdf_content = r#"(DELAYFILE
  (SDFVERSION "3.0")
  (DESIGN "test")
  (TIMESCALE 1ns)
  (CELL
    (CELLTYPE "inv")
    (INSTANCE u_inv)
    (DELAY
      (ABSOLUTE
        (IOPATH A Y (0.050:0.050:0.050) (0.040:0.040:0.040))
      )
    )
  )
)"#;
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ)
            .expect("parse SDF");
        let cell = sdf.get_cell("u_inv").expect("u_inv not found");
        let iopath = cell.iopaths.iter().find(|p| p.output_pin == "Y");
        assert!(iopath.is_some(), "Should find exact IOPATH match for Y");
        assert_eq!(iopath.unwrap().delay.rise_ps, 50);
        assert_eq!(iopath.unwrap().delay.fall_ps, 40);
    }

    #[test]
    fn test_iopath_fallback_first() {
        // Cell with IOPATH output "QN" but cell_origin says "Q" → no match → fallback to first
        use crate::sdf_parser::*;

        let sdf_content = r#"(DELAYFILE
  (SDFVERSION "3.0")
  (DESIGN "test")
  (TIMESCALE 1ns)
  (CELL
    (CELLTYPE "dff")
    (INSTANCE u_dff)
    (DELAY
      (ABSOLUTE
        (IOPATH CLK QN (0.200:0.200:0.200) (0.180:0.180:0.180))
      )
    )
  )
)"#;
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ)
            .expect("parse SDF");
        let cell = sdf.get_cell("u_dff").expect("u_dff not found");

        // Try to match "Q" — won't find it
        let exact_match = cell.iopaths.iter().find(|p| p.output_pin == "Q");
        assert!(exact_match.is_none(), "Should NOT find exact match for Q");

        // Fallback: use first IOPATH
        assert!(!cell.iopaths.is_empty(), "Should have at least one IOPATH");
        let fallback = &cell.iopaths[0];
        assert_eq!(fallback.output_pin, "QN");
        assert_eq!(fallback.delay.rise_ps, 200);
        assert_eq!(fallback.delay.fall_ps, 180);
    }

    #[test]
    fn test_sdf_dff_constraints_loaded() {
        let (netlistdb, aig) = load_inv_chain();
        let sdf_content = include_str!("../tests/timing_test/inv_chain_pnr/inv_chain_test.sdf");
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ)
            .expect("Failed to parse SDF");

        let mut script = make_minimal_script(&aig);
        script.load_timing_from_sdf(&aig, &netlistdb, &sdf, 10000, None, false);

        // inv_chain has 2 DFFs: dff_in and dff_out
        assert_eq!(script.dff_constraints.len(), 2,
            "Expected 2 DFF constraints, got {}", script.dff_constraints.len());

        // Build a map from cell_id to constraint for easier lookup
        let cellid_to_sdf_path = build_cellid_to_sdf_path(&netlistdb);
        let constraint_by_name: Vec<(&str, &DFFConstraint)> = script.dff_constraints.iter()
            .map(|c| {
                let name = cellid_to_sdf_path.get(&(c.cell_id as usize))
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                (name, c)
            })
            .collect();

        // dff_in: SETUP 0.080ns=80ps, HOLD -0.030ns → clamped to 0
        let dff_in = constraint_by_name.iter().find(|(name, _)| *name == "dff_in")
            .expect("dff_in constraint not found");
        assert_eq!(dff_in.1.setup_ps, 80,
            "dff_in setup: expected 80ps, got {}ps", dff_in.1.setup_ps);
        assert_eq!(dff_in.1.hold_ps, 0,
            "dff_in hold: expected 0ps (negative clamped), got {}ps", dff_in.1.hold_ps);

        // dff_out: SETUP 0.085ns=85ps, HOLD -0.028ns → clamped to 0
        let dff_out = constraint_by_name.iter().find(|(name, _)| *name == "dff_out")
            .expect("dff_out constraint not found");
        assert_eq!(dff_out.1.setup_ps, 85,
            "dff_out setup: expected 85ps, got {}ps", dff_out.1.setup_ps);
        assert_eq!(dff_out.1.hold_ps, 0,
            "dff_out hold: expected 0ps (negative clamped), got {}ps", dff_out.1.hold_ps);

        // data_state_pos should be u32::MAX because make_minimal_script has empty output_map
        assert_eq!(dff_in.1.data_state_pos, u32::MAX,
            "dff_in data_state_pos should be u32::MAX with empty output_map");
        assert_eq!(dff_out.1.data_state_pos, u32::MAX,
            "dff_out data_state_pos should be u32::MAX with empty output_map");
    }

    #[test]
    fn test_iopath_no_paths_zero_delay() {
        // Cell with no IOPATHs → zero delay (with liberty fallback = None → default)
        use crate::sdf_parser::*;

        let sdf_content = r#"(DELAYFILE
  (SDFVERSION "3.0")
  (DESIGN "test")
  (TIMESCALE 1ns)
  (CELL
    (CELLTYPE "buf")
    (INSTANCE u_buf)
  )
)"#;
        let sdf = SdfFile::parse_str(sdf_content, SdfCorner::Typ)
            .expect("parse SDF");
        let cell = sdf.get_cell("u_buf").expect("u_buf not found");
        assert!(cell.iopaths.is_empty(), "Should have no IOPATHs");
        // In load_timing_from_sdf, this case falls through to liberty fallback
        // With no liberty fallback, it returns PackedDelay::default() = (0, 0)
    }
}

#[cfg(test)]
mod constraint_buffer_tests {
    use super::*;

    /// Helper: build a minimal FlattenedScriptV1 for constraint buffer testing.
    fn make_script_with_constraints(
        reg_io_state_size: u32,
        clock_period_ps: u64,
        constraints: Vec<DFFConstraint>,
    ) -> FlattenedScriptV1 {
        FlattenedScriptV1 {
            num_blocks: 0,
            num_major_stages: 0,
            blocks_start: Vec::<usize>::new().into(),
            blocks_data: Vec::<u32>::new().into(),
            reg_io_state_size,
            sram_storage_size: 0,
            input_layout: Vec::new(),
            output_map: IndexMap::new(),
            input_map: IndexMap::new(),
            stages_blocks_parts: Vec::new(),
            assertion_positions: Vec::new(),
            display_positions: Vec::new(),
            gate_delays: Vec::new(),
            dff_constraints: constraints,
            clock_period_ps,
            timing_enabled: true,
            delay_patch_map: Vec::new(),
        }
    }

    #[test]
    fn test_empty_constraints() {
        let script = make_script_with_constraints(4, 10000, Vec::new());
        let (clock_ps, buf) = script.build_timing_constraint_buffer();
        assert_eq!(clock_ps, 10000);
        assert_eq!(buf.len(), 4);
        assert!(buf.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_single_dff_constraint() {
        let script = make_script_with_constraints(4, 25000, vec![
            DFFConstraint { setup_ps: 200, hold_ps: 50, data_state_pos: 35, cell_id: 1 },
        ]);
        let (clock_ps, buf) = script.build_timing_constraint_buffer();
        assert_eq!(clock_ps, 25000);
        // data_state_pos 35 → word_idx 1
        assert_eq!(buf[0], 0);
        assert_eq!(buf[1], (200u32 << 16) | 50);
        assert_eq!(buf[2], 0);
        assert_eq!(buf[3], 0);
    }

    #[test]
    fn test_multiple_dffs_same_word_takes_min() {
        let script = make_script_with_constraints(2, 10000, vec![
            DFFConstraint { setup_ps: 300, hold_ps: 100, data_state_pos: 5, cell_id: 1 },
            DFFConstraint { setup_ps: 150, hold_ps: 200, data_state_pos: 10, cell_id: 2 },
        ]);
        let (_clock_ps, buf) = script.build_timing_constraint_buffer();
        // Both in word 0 → min(300,150)=150 setup, min(100,200)=100 hold
        assert_eq!(buf[0], (150u32 << 16) | 100);
    }

    #[test]
    fn test_skip_invalid_data_state_pos() {
        let script = make_script_with_constraints(2, 10000, vec![
            DFFConstraint { setup_ps: 200, hold_ps: 50, data_state_pos: u32::MAX, cell_id: 1 },
        ]);
        let (_clock_ps, buf) = script.build_timing_constraint_buffer();
        assert!(buf.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_skip_out_of_range_word() {
        let script = make_script_with_constraints(2, 10000, vec![
            DFFConstraint { setup_ps: 200, hold_ps: 50, data_state_pos: 100, cell_id: 1 },
        ]);
        let (_clock_ps, buf) = script.build_timing_constraint_buffer();
        // word_idx = 100/32 = 3, but num_words = 2 → skipped
        assert!(buf.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_clock_period_saturation() {
        let script = make_script_with_constraints(1, u64::MAX, Vec::new());
        let (clock_ps, _buf) = script.build_timing_constraint_buffer();
        assert_eq!(clock_ps, u32::MAX);
    }

    // === Violation detection logic tests ===
    // These reproduce the GPU kernel's setup/hold check arithmetic in pure Rust.
    // The GPU kernel checks:
    //   Setup: arrival > 0 && arrival + setup > clock_period → violation
    //   Hold:  arrival < hold → violation

    /// Simulate the GPU kernel's setup violation check.
    /// Returns Some(slack) if violation, None otherwise.
    fn check_setup_violation(arrival: u16, setup_ps: u16, clock_period: u32) -> Option<i32> {
        if arrival > 0 && (arrival as u32 + setup_ps as u32) > clock_period {
            let slack = clock_period as i32 - arrival as i32 - setup_ps as i32;
            Some(slack)
        } else {
            None
        }
    }

    /// Simulate the GPU kernel's hold violation check.
    /// Returns Some(slack) if violation, None otherwise.
    fn check_hold_violation(arrival: u16, hold_ps: u16) -> Option<i32> {
        if (arrival as u32) < (hold_ps as u32) {
            let slack = arrival as i32 - hold_ps as i32;
            Some(slack)
        } else {
            None
        }
    }

    #[test]
    fn test_setup_violation_detection() {
        // Clock period too short for arrival + setup
        // arrival=900, setup=200 → 900+200=1100 > 1000 → violation
        let result = check_setup_violation(900, 200, 1000);
        assert!(result.is_some(), "Should detect setup violation");
        assert_eq!(result.unwrap(), -100, "Slack should be -100ps");

        // Borderline: arrival=800, setup=200 → 800+200=1000, NOT > 1000 → no violation
        let result = check_setup_violation(800, 200, 1000);
        assert!(result.is_none(), "No violation when arrival+setup == clock_period");
    }

    #[test]
    fn test_setup_violation_with_realistic_inv_chain() {
        // Use real inv_chain accumulated delay: 1323ps rise, setup=85ps (dff_out)
        let arrival: u16 = 1323;
        let setup: u16 = 85;

        // clock_period=1200ps → 1323+85=1408 > 1200 → violation, slack=-208
        let result = check_setup_violation(arrival, setup, 1200);
        assert!(result.is_some(), "Should violate at 1200ps clock");
        assert_eq!(result.unwrap(), -208, "Slack should be -208ps");

        // clock_period=1500ps → 1323+85=1408 ≤ 1500 → no violation
        let result = check_setup_violation(arrival, setup, 1500);
        assert!(result.is_none(), "Should not violate at 1500ps clock");

        // clock_period=1400ps → 1323+85=1408 > 1400 → borderline violation, slack=-8
        let result = check_setup_violation(arrival, setup, 1400);
        assert!(result.is_some(), "Should violate at 1400ps clock");
        assert_eq!(result.unwrap(), -8, "Slack should be -8ps");
    }

    #[test]
    fn test_hold_violation_detection() {
        // arrival=10, hold=50 → 10 < 50 → violation, slack=-40
        let result = check_hold_violation(10, 50);
        assert!(result.is_some(), "Should detect hold violation");
        assert_eq!(result.unwrap(), -40, "Slack should be -40ps");

        // arrival=50, hold=50 → 50 is NOT < 50 → no violation
        let result = check_hold_violation(50, 50);
        assert!(result.is_none(), "No violation when arrival == hold");

        // arrival=0, hold=50 → 0 < 50 → violation (hold check has no arrival>0 guard)
        let result = check_hold_violation(0, 50);
        assert!(result.is_some(), "arrival=0 should still trigger hold violation");
        assert_eq!(result.unwrap(), -50, "Slack should be -50ps");
    }

    #[test]
    fn test_no_violation_with_zero_constraint() {
        // Constraint word = 0 means no DFF at this word → kernel skips entirely
        // Simulate: extract setup/hold from packed word
        let constraint_word: u32 = 0;
        let setup_ps = (constraint_word >> 16) as u16;
        let hold_ps = (constraint_word & 0xFFFF) as u16;

        assert_eq!(setup_ps, 0);
        assert_eq!(hold_ps, 0);

        // With zero constraints, no violation is possible:
        // Setup: any arrival + 0 > clock_period only if arrival > clock_period (unlikely in u16)
        // Hold: arrival < 0 is impossible for u16
        let result = check_hold_violation(0, hold_ps);
        assert!(result.is_none(), "Zero hold constraint should never trigger");

        let result = check_hold_violation(500, hold_ps);
        assert!(result.is_none(), "Zero hold constraint should never trigger");
    }

    #[test]
    fn test_constraint_buffer_with_tight_clock() {
        // Build constraints with setup=200ps DFF at word 5 (data_state_pos=160..191)
        let script = make_script_with_constraints(8, 1000, vec![
            DFFConstraint { setup_ps: 200, hold_ps: 50, data_state_pos: 160, cell_id: 1 },
        ]);
        let (clock_ps, buf) = script.build_timing_constraint_buffer();
        assert_eq!(clock_ps, 1000);

        // Verify constraint is at word 5 (160/32 = 5)
        assert_eq!(buf[5], (200u32 << 16) | 50, "Word 5 should have packed constraint");
        // Other words should be zero
        for i in [0, 1, 2, 3, 4, 6, 7] {
            assert_eq!(buf[i], 0, "Word {} should have no constraint", i);
        }

        // Now simulate arrival=850ps at this word → setup violation
        let setup_ps = (buf[5] >> 16) as u16;
        let result = check_setup_violation(850, setup_ps, clock_ps);
        assert!(result.is_some(), "arrival=850 + setup=200 = 1050 > 1000 → violation");
        assert_eq!(result.unwrap(), -50, "Slack should be -50ps");
    }

    #[test]
    fn test_setup_skips_zero_arrival() {
        // The GPU kernel has an `arrival > 0` guard for setup checks.
        // arrival=0 means "no data propagated through combinational logic"
        // (e.g., first cycle, or a DFF whose inputs are constant).
        // Even with a tight clock where arrival + setup > clock_period,
        // the check must NOT fire when arrival == 0.
        let result = check_setup_violation(0, 200, 100);
        assert!(result.is_none(),
            "arrival=0 must skip setup check even when 0+200=200 > 100");

        // Confirm a non-zero arrival with same parameters DOES fire
        let result = check_setup_violation(1, 200, 100);
        assert!(result.is_some(),
            "arrival=1 should trigger: 1+200=201 > 100");

        // Also verify arrival=0 with a very large setup still doesn't fire
        let result = check_setup_violation(0, u16::MAX, 1);
        assert!(result.is_none(),
            "arrival=0 must always skip, regardless of setup or clock_period");
    }

    #[test]
    fn test_setup_u16_max_arrival() {
        // Test arrival near u16::MAX (65535ps) with setup that pushes sum past
        // u16 range. The kernel uses u32 arithmetic:
        //   (u32)arrival + (u32)setup_ps > clock_period_ps
        // so it must not overflow at 16-bit boundary.

        // arrival=65000, setup=1000 → (u32)66000 > 60000 → violation
        let result = check_setup_violation(65000, 1000, 60000);
        assert!(result.is_some(),
            "65000+1000=66000 > 60000 → violation");
        assert_eq!(result.unwrap(), 60000 - 65000 - 1000,
            "Slack should be -6000ps");

        // arrival=65000, setup=1000 → (u32)66000 > 70000 → no violation
        let result = check_setup_violation(65000, 1000, 70000);
        assert!(result.is_none(),
            "65000+1000=66000 <= 70000 → no violation");

        // Extreme: both at u16::MAX
        // arrival=65535, setup=65535 → (u32)131070 > 131069 → violation
        let result = check_setup_violation(u16::MAX, u16::MAX, 131069);
        assert!(result.is_some(),
            "u16::MAX + u16::MAX = 131070 > 131069 → violation");
        assert_eq!(result.unwrap(), 131069 - 65535 - 65535,
            "Slack should be -1ps");

        // Same extreme but clock_period accommodates it
        let result = check_setup_violation(u16::MAX, u16::MAX, 131070);
        assert!(result.is_none(),
            "u16::MAX + u16::MAX = 131070 <= 131070 → no violation");
    }
}
