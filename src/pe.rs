// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Partition executor

use crate::aig::{
    bitset_or_inplace, bitset_union_popcount, DriverType, EndpointGroup, TopoTraverser, AIG,
};
use crate::staging::StagedAIG;
use indexmap::{IndexMap, IndexSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

/// The number of boomerang stages.
///
/// This determines the shuffle width, i.e., kernel width.
/// `kernel width = (1 << BOOMERANG_NUM_STAGES)`.
pub const BOOMERANG_NUM_STAGES: usize = 13;

const BOOMERANG_MAX_WRITEOUTS: usize = 1 << (BOOMERANG_NUM_STAGES - 5);

/// One Boomerang stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoomerangStage {
    /// the boomerang hierarchy, 8192 -> 4096 -> ... -> 1.
    ///
    /// each element is an aigpin index (without iv).
    /// its parent indices should either be a passthrough or an
    /// and gate mapping.
    pub hier: Vec<Vec<usize>>,
    /// the 32-packed elements in the hierarchy where there should be
    /// a pass-through.
    pub write_outs: Vec<usize>,
}

/// One partitioned block: a basic execution unit on GPU.
///
/// A block is mapped to a GPU block with the following resource
/// constraints:
/// 1. the number of unique inputs should not exceed 8191.
/// 2. the number of unique outputs should not exceed 8191.
///    for srams and dffs, outputs include all enable pins and bus pins.
///    there might be unusable holes but the effective capacity is at least
///    4095.
/// 3. the number of intermediate pins alive at each stage should not
///    exceed 4095.
/// 4. the number of SRAM output groups should not exceed 64.
///    64 = 8192 / (32 * 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// the endpoints that are realized by this partition.
    pub endpoints: Vec<usize>,
    /// the boomerang stages.
    ///
    /// between stages there will automatically be shuffles.
    pub stages: Vec<BoomerangStage>,
}

/// build a single boomerang stage given the current inputs and
/// outputs.
fn build_one_boomerang_stage(
    aig: &AIG,
    unrealized_comb_outputs: &mut IndexSet<usize>,
    unrealized_comb_outputs_dense: &mut Vec<bool>,
    realized_inputs: &mut IndexSet<usize>,
    realized_inputs_dense: &mut Vec<bool>,
    total_write_outs: &mut usize,
    num_reserved_writeouts: usize,
    traverser: &mut TopoTraverser,
) -> Option<BoomerangStage> {
    let mut hier = Vec::new();
    for i in 0..=BOOMERANG_NUM_STAGES {
        hier.push(vec![usize::MAX; 1 << (BOOMERANG_NUM_STAGES - i)]);
    }

    // first discover the (remaining) subgraph to implement.
    let endpoints_vec: Vec<usize> = unrealized_comb_outputs.iter().copied().collect();
    let order = traverser.topo_traverse(aig, Some(&endpoints_vec), Some(&realized_inputs));
    // Dense id2order: aigpin -> topo order index. usize::MAX = not present.
    let mut id2order = vec![usize::MAX; aig.num_aigpins + 1];
    for (order_i, &i) in order.iter().enumerate() {
        id2order[i] = order_i;
    }
    let mut level = vec![0; order.len()];
    for (order_i, i) in order.iter().copied().enumerate() {
        if realized_inputs_dense[i] {
            continue;
        }
        let mut lvli: usize = 0;
        if let DriverType::AndGate(a, b) = aig.drivers[i] {
            if a >= 2 {
                lvli = lvli.max(level[id2order[a >> 1]] + 1);
            }
            if b >= 2 {
                lvli = lvli.max(level[id2order[b >> 1]] + 1);
            }
        }
        level[order_i] = lvli;
    }
    let max_level = level.iter().copied().max().unwrap();
    clilog::trace!("boomerang current max level: {}", max_level);

    fn place_bit(
        aig: &AIG,
        hier: &mut Vec<Vec<usize>>,
        hier_visited_nodes_count: &mut Vec<usize>,
        hier_visited_len: &mut usize,
        hier_active_nodes: &mut Vec<usize>,
        level: &Vec<usize>,
        id2order: &Vec<usize>,
        hi: usize,
        j: usize,
        nd: usize,
    ) {
        hier[hi][j] = nd;
        if hi == 0 {
            return;
        }
        if hier_visited_nodes_count[nd] == 0 {
            *hier_visited_len += 1;
            hier_active_nodes.push(nd);
        }
        hier_visited_nodes_count[nd] += 1;
        let lvlnd = level[id2order[nd]];
        assert!(lvlnd <= hi);
        if lvlnd != hi {
            place_bit(
                aig,
                hier,
                hier_visited_nodes_count,
                hier_visited_len,
                hier_active_nodes,
                level,
                id2order,
                hi - 1,
                j,
                nd,
            );
        } else {
            let (a, b) = match aig.drivers[nd] {
                DriverType::AndGate(a, b) => (a, b),
                _ => panic!(),
            };
            let hier_hi_len = hier[hi].len();
            place_bit(
                aig,
                hier,
                hier_visited_nodes_count,
                hier_visited_len,
                hier_active_nodes,
                level,
                id2order,
                hi - 1,
                j,
                a >> 1,
            );
            place_bit(
                aig,
                hier,
                hier_visited_nodes_count,
                hier_visited_len,
                hier_active_nodes,
                level,
                id2order,
                hi - 1,
                j + hier_hi_len,
                b >> 1,
            );
        }
    }

    fn purge_bit(
        aig: &AIG,
        hier: &mut Vec<Vec<usize>>,
        hier_visited_nodes_count: &mut Vec<usize>,
        hier_visited_len: &mut usize,
        hier_active_nodes: &mut Vec<usize>,
        level: &Vec<usize>,
        id2order: &Vec<usize>,
        hi: usize,
        j: usize,
    ) {
        if hier[hi][j] == usize::MAX {
            return;
        }
        let nd = hier[hi][j];
        hier[hi][j] = usize::MAX;
        if hi == 0 {
            return;
        }
        hier_visited_nodes_count[nd] -= 1;
        if hier_visited_nodes_count[nd] == 0 {
            *hier_visited_len -= 1;
            if let Some(pos) = hier_active_nodes.iter().position(|&x| x == nd) {
                hier_active_nodes.swap_remove(pos);
            }
        }
        let hier_hi_len = hier[hi].len();
        purge_bit(
            aig,
            hier,
            hier_visited_nodes_count,
            hier_visited_len,
            hier_active_nodes,
            level,
            id2order,
            hi - 1,
            j,
        );
        purge_bit(
            aig,
            hier,
            hier_visited_nodes_count,
            hier_visited_len,
            hier_active_nodes,
            level,
            id2order,
            hi - 1,
            j + hier_hi_len,
        );
    }

    // the nodes that are implemented in the hierarchy.
    // we only count for hierarchy[1 and more], [0] is not counted.
    // Dense Vec: index by aigpin, value = reference count (0 = not present).
    let mut hier_visited_nodes_count: Vec<usize> = vec![0; aig.num_aigpins + 1];
    let mut hier_visited_len: usize = 0;
    let mut hier_active_nodes: Vec<usize> = Vec::new();
    let mut selected_level = max_level.min(BOOMERANG_NUM_STAGES);

    /// compute the maximum number of steps needed from this node
    /// to reach an endpoint node.
    ///
    /// during this path, except the starting point, no node should
    /// already be inside the boomerang hierarchy.
    fn compute_reverse_level(
        order: &Vec<usize>,
        id2order: &Vec<usize>,
        unrealized_comb_outputs: &IndexSet<usize>,
        realized_inputs_dense: &Vec<bool>,
        hier_visited_nodes_count: &Vec<usize>,
        aig: &AIG,
    ) -> Vec<usize> {
        let mut reverse_level = vec![usize::MAX; order.len()];
        for &i in unrealized_comb_outputs.iter() {
            reverse_level[id2order[i]] = 0;
        }
        for (order_i, i) in order.iter().copied().enumerate().rev() {
            if realized_inputs_dense[i] || hier_visited_nodes_count[i] > 0 {
                continue;
            }
            let rlvli = reverse_level[order_i];
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if a >= 2 {
                    let a = id2order[a >> 1];
                    let rlvla = &mut reverse_level[a];
                    if *rlvla == usize::MAX || *rlvla < rlvli + 1 {
                        *rlvla = rlvli + 1;
                    }
                }
                if b >= 2 {
                    let b = id2order[b >> 1];
                    let rlvlb = &mut reverse_level[b];
                    if *rlvlb == usize::MAX || *rlvlb < rlvli + 1 {
                        *rlvlb = rlvli + 1;
                    }
                }
            }
        }
        reverse_level
    }

    /// compute the set of nodes that must be implemented in level 1
    /// in addition to the current hierarchy.
    ///
    /// the necessary_level1 nodes can only come from level 0 or
    /// level 1.
    /// a level 1 node is necessary if it is not already
    /// implemented, and it still drives a downstream endpoint.
    /// a level 0 node is necessary if it is not already implemented,
    /// and it either (1) is needed by a level>=2 node, or (2) is
    /// itself an unrealized endpoint.
    fn compute_lvl1_necessary_nodes(
        order: &Vec<usize>,
        id2order: &Vec<usize>,
        level: &Vec<usize>,
        reverse_level: &Vec<usize>,
        aig: &AIG,
        unrealized_comb_outputs_dense: &Vec<bool>,
        hier_visited_nodes_count: &Vec<usize>,
    ) -> IndexSet<usize> {
        let mut lvl1_necessary_nodes = IndexSet::new();
        for order_i in 0..order.len() {
            if hier_visited_nodes_count[order[order_i]] > 0 {
                continue;
            }
            if reverse_level[order_i] == usize::MAX {
                continue;
            }
            if level[order_i] == 0 {
                if unrealized_comb_outputs_dense[order[order_i]] {
                    lvl1_necessary_nodes.insert(order[order_i]);
                }
                continue;
            }
            if level[order_i] == 1 {
                lvl1_necessary_nodes.insert(order[order_i]);
            } else {
                let (a, b) = match aig.drivers[order[order_i]] {
                    DriverType::AndGate(a, b) => (a, b),
                    _ => panic!(),
                };
                if a >= 2 && level[id2order[a >> 1]] == 0 && hier_visited_nodes_count[a >> 1] == 0 {
                    lvl1_necessary_nodes.insert(a >> 1);
                }
                if b >= 2 && level[id2order[b >> 1]] == 0 && hier_visited_nodes_count[b >> 1] == 0 {
                    lvl1_necessary_nodes.insert(b >> 1);
                }
            }
        }
        lvl1_necessary_nodes
    }

    let mut reverse_level = compute_reverse_level(
        &order,
        &id2order,
        unrealized_comb_outputs,
        realized_inputs_dense,
        &hier_visited_nodes_count,
        aig,
    );

    let mut last_lvl1_necessary_nodes = IndexSet::new();

    while selected_level >= 2 {
        // Collect all available slots at this level
        let available_slots: Vec<usize> = (0..hier[selected_level].len())
            .filter(|&i| hier[selected_level][i] == usize::MAX)
            .collect();

        if available_slots.is_empty() {
            clilog::trace!("no space at level {}", selected_level);
            selected_level -= 1;
            continue;
        }

        // Collect all candidate nodes at this level, sorted by topological order (timing proxy)
        // Nodes earlier in topological order have shorter paths from inputs = earlier arrival
        let mut candidates: Vec<(usize, usize)> = (0..order.len())
            .filter(|&order_i| {
                level[order_i] == selected_level
                    && hier_visited_nodes_count[order[order_i]] == 0
                    && reverse_level[order_i] != usize::MAX
            })
            .map(|order_i| (order_i, reverse_level[order_i]))
            .collect();

        if candidates.is_empty() {
            clilog::trace!("no node at level {}", selected_level);
            selected_level -= 1;
            continue;
        }

        // Sort candidates: primary by reverse_level (output proximity), secondary by order_i (timing)
        // This groups timing-similar nodes together while still prioritizing output-critical paths
        candidates.sort_by(|a, b| {
            // First compare reverse_level (higher = closer to output = higher priority)
            b.1.cmp(&a.1)
                // Then by topological order (earlier = similar timing grouped together)
                .then_with(|| a.0.cmp(&b.0))
        });

        // Pick the highest priority candidate
        let selected_node_ord = candidates[0].0;
        let slot_at_level = available_slots[0];
        let selected_node = order[selected_node_ord];

        place_bit(
            aig,
            &mut hier,
            &mut hier_visited_nodes_count,
            &mut hier_visited_len,
            &mut hier_active_nodes,
            &level,
            &id2order,
            selected_level,
            slot_at_level,
            selected_node,
        );

        let reverse_level_upd = compute_reverse_level(
            &order,
            &id2order,
            unrealized_comb_outputs,
            realized_inputs_dense,
            &hier_visited_nodes_count,
            aig,
        );

        // store the nodes that need to be put on the 1-level
        // (simple ands).
        // they are periodically checked to ensure they have space.
        let lvl1_necessary_nodes = compute_lvl1_necessary_nodes(
            &order,
            &id2order,
            &level,
            &reverse_level_upd,
            aig,
            unrealized_comb_outputs_dense,
            &hier_visited_nodes_count,
        );

        let num_lvl1_hier_taken = hier[1].iter().filter(|i| **i != usize::MAX).count();

        clilog::trace!(
            "taken one node at level {}, used 1-level space {}, hier visited unique {}, num nodes necessary in lvl1 {}",
            selected_level, num_lvl1_hier_taken,
            hier_visited_len, lvl1_necessary_nodes.len()
        );

        if lvl1_necessary_nodes.len() + num_lvl1_hier_taken.max(hier_visited_len)
            >= (1 << (BOOMERANG_NUM_STAGES - 1))
        {
            clilog::trace!("REVERSED the plan due to overflow");
            purge_bit(
                aig,
                &mut hier,
                &mut hier_visited_nodes_count,
                &mut hier_visited_len,
                &mut hier_active_nodes,
                &level,
                &id2order,
                selected_level,
                slot_at_level,
            );
            selected_level -= 1;
            continue;
        }

        reverse_level = reverse_level_upd;
        last_lvl1_necessary_nodes = lvl1_necessary_nodes;
    }

    if last_lvl1_necessary_nodes.is_empty() {
        last_lvl1_necessary_nodes = compute_lvl1_necessary_nodes(
            &order,
            &id2order,
            &level,
            &reverse_level,
            aig,
            unrealized_comb_outputs_dense,
            &hier_visited_nodes_count,
        );
    }

    // the hierarchy is now constructed except all 1-level nodes.
    // it's time to place them. during this process, we heuristically collect
    // endpoint nodes into consecutive space for early write-out.
    //
    // we first try to finalize all endpoints that have to appear in
    // level 1.
    // after that, we will try if we can write out all others scattered.
    let mut endpoints_lvl1 = Vec::new();
    let mut endpoints_untouched = Vec::new();
    let mut endpoints_hier = IndexSet::new();
    for &endpt in unrealized_comb_outputs.iter() {
        if hier_visited_nodes_count[endpt] > 0 {
            endpoints_hier.insert(endpt);
        } else if last_lvl1_necessary_nodes.contains(&endpt) {
            endpoints_lvl1.push(endpt);
        } else {
            endpoints_untouched.push(endpt);
        }
    }

    // sort level-1 endpoints by logic level so signals with similar timing
    // land in the same 32-slot groups, tightening the conservative timing estimate.
    let timer_sort = clilog::stimer!("timing-aware lvl1 sort");
    endpoints_lvl1.sort_by_key(|&nd| level[id2order[nd]]);
    clilog::finish!(timer_sort);

    // collect all 32-consecutive level 1 spaces.
    // (num occupied, i), will be sorted later.
    let mut spaces = Vec::new();
    for i in 0..hier[1].len() / 32 {
        let mut num_occupied = 0u8;
        for j in i * 32..(i + 1) * 32 {
            if hier[1][j] != usize::MAX {
                num_occupied += 1;
            }
        }
        if num_occupied < 10 {
            spaces.push((num_occupied, i * 32))
        }
    }
    spaces.sort();
    let mut spaces_j = 0;
    let mut endpt_lvl1_i = 0;
    let mut realized_endpoints = IndexSet::new();
    let mut write_outs = Vec::new();
    // heuristically push level 1 endpoints.
    while spaces_j < spaces.len()
        && (endpoints_untouched.is_empty() || // if we can try all
         endpoints_lvl1.len() - endpt_lvl1_i >= (32 - spaces[spaces_j].0) as usize)
    {
        let i = spaces[spaces_j].1;
        for j in i..i + 32 {
            if endpt_lvl1_i >= endpoints_lvl1.len() {
                break;
            }
            if hier[1][j] == usize::MAX {
                let endpt_i = endpoints_lvl1[endpt_lvl1_i];
                place_bit(
                    aig,
                    &mut hier,
                    &mut hier_visited_nodes_count,
                    &mut hier_visited_len,
                    &mut hier_active_nodes,
                    &level,
                    &id2order,
                    1,
                    j,
                    endpt_i,
                );
                realized_endpoints.insert(endpt_i);
                endpt_lvl1_i += 1;
            } else if unrealized_comb_outputs_dense[hier[1][j]] {
                realized_endpoints.insert(hier[1][j]);
            }
        }
        *total_write_outs += 1;
        write_outs.push((i + hier[1].len()) / 32);
        spaces_j += 1;
    }

    if *total_write_outs > BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts {
        clilog::trace!("boomerang: write out overflowed");
        return None;
    }

    // place all remaining lvl1 nodes, sorted by logic level for timing-aware packing.
    let mut sorted_remaining_lvl1: Vec<usize> = last_lvl1_necessary_nodes
        .iter()
        .copied()
        .filter(|&nd| hier_visited_nodes_count[nd] == 0 && !realized_endpoints.contains(&nd))
        .collect();
    sorted_remaining_lvl1.sort_by_key(|&nd| level[id2order[nd]]);

    let mut hier1_j = 0;
    for &nd in &sorted_remaining_lvl1 {
        while hier[1][hier1_j] != usize::MAX {
            hier1_j += 1;
            if hier1_j >= hier[1].len() {
                clilog::trace!("boomerang: overflow putting lvl1");
                return None;
            }
        }
        place_bit(
            aig,
            &mut hier,
            &mut hier_visited_nodes_count,
            &mut hier_visited_len,
            &mut hier_active_nodes,
            &level,
            &id2order,
            1,
            hier1_j,
            nd,
        );
    }
    while hier[1][hier1_j] != usize::MAX {
        hier1_j += 1;
        if hier1_j >= hier[1].len() {
            clilog::trace!("boomerang: overflow putting lvl1 (just a zero pin..)");
            return None;
        }
    }

    // measure timing spread per 32-signal group at hier[1].
    {
        let mut max_spread: usize = 0;
        let mut total_spread: usize = 0;
        let mut num_groups: usize = 0;
        for group in 0..hier[1].len() / 32 {
            let levels: Vec<usize> = (group * 32..(group + 1) * 32)
                .filter_map(|j| {
                    let nd = hier[1][j];
                    if nd != usize::MAX {
                        Some(level[id2order[nd]])
                    } else {
                        None
                    }
                })
                .collect();
            if levels.len() > 1 {
                let spread = levels.iter().max().unwrap() - levels.iter().min().unwrap();
                max_spread = max_spread.max(spread);
                total_spread += spread;
                num_groups += 1;
            }
        }
        if num_groups > 0 {
            clilog::info!(
                "Timing packing: {} groups, avg level spread {:.1}, max spread {}",
                num_groups,
                total_spread as f64 / num_groups as f64,
                max_spread
            );
        }
    }

    // check if we can make this the last stage.
    if endpoints_untouched.is_empty() {
        let mut add_write_outs = IndexSet::new();
        for hi in 1..=BOOMERANG_NUM_STAGES {
            for j in 0..hier[hi].len() {
                let nd = hier[hi][j];
                if endpoints_hier.contains(&nd) && !realized_endpoints.contains(&nd) {
                    add_write_outs.insert((j + hier[hi].len()) / 32);
                    if add_write_outs.len() + *total_write_outs
                        > BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts
                    {
                        break;
                    }
                }
            }
        }
        if add_write_outs.len() + *total_write_outs
            <= BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts
        {
            for wo in add_write_outs {
                write_outs.push(wo);
                *total_write_outs += 1;
            }
            for endpt in endpoints_hier {
                realized_endpoints.insert(endpt);
            }
        }
    }

    for &i in &hier_active_nodes {
        realized_inputs.insert(i);
        realized_inputs_dense[i] = true;
    }
    for &i in &realized_endpoints {
        assert!(unrealized_comb_outputs.swap_remove(&i));
        unrealized_comb_outputs_dense[i] = false;
    }

    Some(BoomerangStage { hier, write_outs })
}

impl Partition {
    /// Cheap pre-check that rejects merges that would obviously fail `build_one`.
    ///
    /// Checks SRAM count and writeout overflow without building the full hierarchy.
    /// Returns `true` if the merge should be rejected.
    fn quick_reject(aig: &AIG, staged: &StagedAIG, endpoints: &[usize]) -> bool {
        let mut num_srams = 0usize;
        let mut comb_outputs_activations = IndexMap::<usize, IndexSet<usize>>::new();
        for &endpt_i in endpoints {
            let edg = staged.get_endpoint_group(aig, endpt_i);
            match edg {
                EndpointGroup::DFF(dff) => {
                    comb_outputs_activations
                        .entry(dff.d_iv >> 1)
                        .or_default()
                        .insert(dff.en_iv << 1 | (dff.d_iv & 1));
                }
                EndpointGroup::PrimaryOutput(pin) => {
                    comb_outputs_activations
                        .entry(pin >> 1)
                        .or_default()
                        .insert(2 | (pin & 1));
                }
                EndpointGroup::RAMBlock(_) => {
                    num_srams += 1;
                }
                EndpointGroup::SimControl(ctrl) => {
                    comb_outputs_activations
                        .entry(ctrl.condition_iv >> 1)
                        .or_default()
                        .insert(2 | (ctrl.condition_iv & 1));
                }
                EndpointGroup::Display(disp) => {
                    comb_outputs_activations
                        .entry(disp.enable_iv >> 1)
                        .or_default()
                        .insert(2 | (disp.enable_iv & 1));
                    for &arg_iv in &disp.args_iv {
                        if arg_iv > 1 {
                            comb_outputs_activations
                                .entry(arg_iv >> 1)
                                .or_default()
                                .insert(2 | (arg_iv & 1));
                        }
                    }
                }
                EndpointGroup::StagedIOPin(pin) => {
                    comb_outputs_activations.entry(pin).or_default().insert(2);
                }
            }
        }
        let num_output_dups: usize = comb_outputs_activations
            .iter()
            .map(|(_, ckens)| ckens.len() - 1)
            .sum();
        let num_reserved_writeouts = num_srams + (num_output_dups + 31) / 32;
        num_reserved_writeouts >= BOOMERANG_MAX_WRITEOUTS
            || num_srams * 4 + num_output_dups > BOOMERANG_MAX_WRITEOUTS
    }

    /// build one partition given a set of endpoints to realize.
    ///
    /// if the resource is overflowed, None will be returned.
    /// see [Partition] for resource constraints.
    pub fn build_one(aig: &AIG, staged: &StagedAIG, endpoints: &Vec<usize>) -> Option<Partition> {
        Self::build_one_cancellable(aig, staged, endpoints, None)
    }

    /// Like `build_one`, but checks `cancel` flag between boomerang stages.
    ///
    /// Returns `None` if cancelled or if resource constraints are exceeded.
    fn build_one_cancellable(
        aig: &AIG,
        staged: &StagedAIG,
        endpoints: &Vec<usize>,
        cancel: Option<&AtomicBool>,
    ) -> Option<Partition> {
        let mut unrealized_comb_outputs = IndexSet::new();
        let mut realized_inputs = staged.primary_inputs.as_ref().cloned().unwrap_or_default();
        let mut num_srams = 0;
        let mut comb_outputs_activations = IndexMap::<usize, IndexSet<usize>>::new();
        for &endpt_i in endpoints {
            let edg = staged.get_endpoint_group(aig, endpt_i);
            edg.for_each_input(|i| {
                unrealized_comb_outputs.insert(i);
            });
            match edg {
                EndpointGroup::DFF(dff) => {
                    comb_outputs_activations
                        .entry(dff.d_iv >> 1)
                        .or_default()
                        .insert(dff.en_iv << 1 | (dff.d_iv & 1));
                }
                EndpointGroup::PrimaryOutput(pin) => {
                    comb_outputs_activations
                        .entry(pin >> 1)
                        .or_default()
                        .insert(2 | (pin & 1));
                }
                EndpointGroup::RAMBlock(_) => {
                    num_srams += 1;
                }
                EndpointGroup::SimControl(ctrl) => {
                    comb_outputs_activations
                        .entry(ctrl.condition_iv >> 1)
                        .or_default()
                        .insert(2 | (ctrl.condition_iv & 1));
                }
                EndpointGroup::Display(disp) => {
                    comb_outputs_activations
                        .entry(disp.enable_iv >> 1)
                        .or_default()
                        .insert(2 | (disp.enable_iv & 1));
                    for &arg_iv in &disp.args_iv {
                        if arg_iv > 1 {
                            comb_outputs_activations
                                .entry(arg_iv >> 1)
                                .or_default()
                                .insert(2 | (arg_iv & 1));
                        }
                    }
                }
                EndpointGroup::StagedIOPin(pin) => {
                    comb_outputs_activations.entry(pin).or_default().insert(2);
                }
            }
        }
        let num_output_dups = comb_outputs_activations
            .iter()
            .map(|(_, ckens)| ckens.len() - 1)
            .sum::<usize>();
        let num_reserved_writeouts = num_srams + (num_output_dups + 31) / 32;
        if num_reserved_writeouts >= BOOMERANG_MAX_WRITEOUTS
            || num_srams * 4 + num_output_dups > BOOMERANG_MAX_WRITEOUTS
        {
            // overflowed writeout
            return None;
        }
        // Build dense boolean arrays for fast contains() checks in inner loops.
        let mut unrealized_comb_outputs_dense = vec![false; aig.num_aigpins + 1];
        for &i in unrealized_comb_outputs.iter() {
            unrealized_comb_outputs_dense[i] = true;
        }
        let mut realized_inputs_dense = vec![false; aig.num_aigpins + 1];
        for &i in realized_inputs.iter() {
            realized_inputs_dense[i] = true;
        }

        let mut stages = Vec::<BoomerangStage>::new();
        let mut total_write_outs = 0;
        let mut traverser = TopoTraverser::new(aig.num_aigpins);
        while !unrealized_comb_outputs.is_empty() {
            // Check cancel flag between stages
            if let Some(flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    return None;
                }
            }
            let stage = build_one_boomerang_stage(
                aig,
                &mut unrealized_comb_outputs,
                &mut unrealized_comb_outputs_dense,
                &mut realized_inputs,
                &mut realized_inputs_dense,
                &mut total_write_outs,
                num_reserved_writeouts,
                &mut traverser,
            )?;
            stages.push(stage);
        }
        Some(Partition {
            endpoints: endpoints.clone(),
            stages,
        })
    }
}

/// Collect combinational output pins for a set of endpoints.
fn collect_comb_outputs(aig: &AIG, staged: &StagedAIG, endpoints: &[usize]) -> Vec<usize> {
    let mut comb_outputs = Vec::new();
    for &endpt_i in endpoints {
        staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
            comb_outputs.push(i);
        });
    }
    comb_outputs
}

/// Given an initial clustering solution of endpoints, generate and map a
/// refined solution.
///
/// The refined solution will have smaller number of partitions
/// as we aggressively merge the partitions when possible.
pub fn process_partitions(
    aig: &AIG,
    staged: &StagedAIG,
    mut parts: Vec<Vec<usize>>,
    prebuilt: Option<Vec<Partition>>,
    max_stage_degrad: usize,
) -> Option<Vec<Partition>> {
    // Phase 1: Compute node counts and bitsets for each partition using
    // dense visited buffers (TopoTraverser) instead of IndexSet-based DFS.
    let (cnt_nodes, mut node_bitsets): (Vec<usize>, Vec<Vec<u64>>) = parts
        .par_iter()
        .map(|v| {
            let comb_outputs = collect_comb_outputs(aig, staged, v);
            let mut traverser = TopoTraverser::new(aig.num_aigpins);
            let (order, bitset) = traverser.topo_traverse_with_bitset(
                aig,
                Some(&comb_outputs),
                staged.primary_inputs.as_ref(),
            );
            (order.len(), bitset)
        })
        .unzip();

    let all_original_parts = if let Some(prebuilt) = prebuilt {
        assert_eq!(
            prebuilt.len(),
            parts.len(),
            "prebuilt partitions count mismatch"
        );
        prebuilt
    } else {
        let built = parts
            .par_iter()
            .enumerate()
            .map(|(i, v)| {
                let part = Partition::build_one(aig, staged, v);
                if part.is_none() {
                    clilog::error!("Partition {} exceeds resource constraint.", i);
                }
                part
            })
            .collect::<Vec<_>>();
        built.into_iter().collect::<Option<Vec<_>>>()?
    };
    let max_original_nstages = all_original_parts
        .iter()
        .map(|p| p.stages.len())
        .max()
        .unwrap();

    let mut effective_parts = Vec::<Partition>::new();
    let max_trials = (all_original_parts.len() / 8).max(20);
    for (i, mut partition_self) in all_original_parts.into_iter().enumerate() {
        if parts[i].is_empty() {
            continue;
        }
        let mut merge_blacklist = HashSet::<usize>::new();
        let mut cnt_node_i = cnt_nodes[i];

        loop {
            // Score merge candidates using bitset union + popcount
            // instead of full DFS per candidate. This is exact because both
            // traversals share the same primary_inputs boundary, so the merged
            // traversal's node set equals the union of the two individual sets.
            let node_bitset_i = &node_bitsets[i];
            let mut merge_choices = parts[i + 1..parts.len()]
                .par_iter()
                .enumerate()
                .filter_map(|(j, v)| {
                    if v.is_empty() {
                        return None;
                    }
                    let abs_j = i + j + 1;
                    if merge_blacklist.contains(&abs_j) {
                        return None;
                    }
                    // O(num_aigpins/64) bitset union instead of O(subgraph) DFS
                    let merged_count = bitset_union_popcount(node_bitset_i, &node_bitsets[abs_j]);
                    Some((
                        merged_count - cnt_nodes[abs_j].max(cnt_node_i),
                        merged_count,
                        abs_j,
                    ))
                })
                .collect::<Vec<_>>();
            merge_choices.sort();
            let mut merged = false;

            /// Result of a speculative merge trial.
            #[derive(Clone)]
            enum TrialResult {
                /// Trial completed: partition_ij is None if build failed, Some if succeeded.
                Completed(Option<Partition>),
                /// Trial was cancelled because another trial succeeded first.
                /// Do not blacklist -- the merge may still be valid.
                Cancelled,
            }
            #[derive(Clone)]
            struct PartsPartitions {
                parts_ij: Vec<usize>,
                result: TrialResult,
            }
            let mut merge_trials: Vec<Option<PartsPartitions>> = vec![None; merge_choices.len()];
            let mut parallel_trial_stride = 4;

            for (merge_i, &(_cnt_diff, cnt_new, j)) in merge_choices.iter().enumerate() {
                if merge_trials[merge_i].is_none() {
                    if merge_i > max_trials {
                        break; // do not try too many
                    }
                    // Cancel-on-success for speculative parallel trials
                    let cancel_flag = AtomicBool::new(false);
                    let rhs = merge_trials.len().min(merge_i + parallel_trial_stride);
                    merge_trials[merge_i..rhs]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(merge_j, trial)| {
                            let j = merge_choices[merge_i + merge_j].2;
                            let parts_ij: Vec<usize> =
                                parts[i].iter().chain(parts[j].iter()).copied().collect();

                            // Cheap pre-check before expensive build_one
                            if Partition::quick_reject(aig, staged, &parts_ij) {
                                *trial = Some(PartsPartitions {
                                    parts_ij,
                                    result: TrialResult::Completed(None),
                                });
                                return;
                            }

                            // Check cancel before starting expensive work
                            if cancel_flag.load(Ordering::Relaxed) {
                                *trial = Some(PartsPartitions {
                                    parts_ij,
                                    result: TrialResult::Cancelled,
                                });
                                return;
                            }

                            let partition_ij = Partition::build_one_cancellable(
                                aig,
                                staged,
                                &parts_ij,
                                Some(&cancel_flag),
                            );
                            if partition_ij.is_some() {
                                cancel_flag.store(true, Ordering::Relaxed);
                                *trial = Some(PartsPartitions {
                                    parts_ij,
                                    result: TrialResult::Completed(partition_ij),
                                });
                            } else {
                                // build_one returned None -- could be genuine failure
                                // or cancellation mid-build. If cancelled, don't blacklist.
                                let was_cancelled = cancel_flag.load(Ordering::Relaxed);
                                *trial = Some(PartsPartitions {
                                    parts_ij,
                                    result: if was_cancelled {
                                        TrialResult::Cancelled
                                    } else {
                                        TrialResult::Completed(None)
                                    },
                                });
                            }
                        });
                    parallel_trial_stride *= 2;
                }

                let PartsPartitions { parts_ij, result } = merge_trials[merge_i].take().unwrap();

                match result {
                    TrialResult::Completed(None) => {
                        merge_blacklist.insert(j);
                    }
                    TrialResult::Cancelled => {
                        // Don't blacklist -- this trial was interrupted, not proven infeasible.
                        // It will be retried if needed in a future iteration.
                    }
                    TrialResult::Completed(Some(partition))
                        if partition.stages.len() > max_original_nstages + max_stage_degrad =>
                    {
                        clilog::debug!(
                            "skipped merging {} with {} due to nstage degradation: \
                                        {} > {}",
                            i,
                            j,
                            partition.stages.len(),
                            max_original_nstages + max_stage_degrad
                        );
                        merge_blacklist.insert(j);
                    }
                    TrialResult::Completed(Some(partition)) => {
                        clilog::info!("merged partition {} with {}", i, j);
                        // Update bitset: OR in j's bitset
                        let j_bitset = std::mem::take(&mut node_bitsets[j]);
                        bitset_or_inplace(&mut node_bitsets[i], &j_bitset);
                        parts[i] = parts_ij;
                        parts[j] = vec![];
                        partition_self = partition;
                        merged = true;
                        cnt_node_i = cnt_new;
                        break;
                    }
                }
            }
            if !merged {
                break;
            }
        }

        clilog::info!("part {}: #stages {}", i, partition_self.stages.len());
        effective_parts.push(partition_self);
    }
    effective_parts.sort_by_key(|p| usize::MAX - p.stages.len());
    Some(effective_parts)
}

/// Read a cluster solution from hgr.part.xx file.
/// Then call [process_partitions].
pub fn process_partitions_from_hgr_parts_file(
    aig: &AIG,
    staged: &StagedAIG,
    hgr_parts_file: &PathBuf,
    max_stage_degrad: usize,
) -> Option<Vec<Partition>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let mut parts = Vec::<Vec<usize>>::new();
    let f_parts = File::open(&hgr_parts_file).unwrap();
    let f_parts = BufReader::new(f_parts);
    for (i, line) in f_parts.lines().enumerate() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        let part_id = line.parse::<usize>().unwrap();
        while parts.len() <= part_id {
            parts.push(vec![]);
        }
        parts[part_id].push(i);
    }
    clilog::info!(
        "read parts file {} with {} parts",
        hgr_parts_file.display(),
        parts.len()
    );

    process_partitions(aig, staged, parts, None, max_stage_degrad)
}
