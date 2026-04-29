use std::{cell::RefCell, collections::VecDeque, ffi::{CStr, CString}, num::NonZeroU32, rc::{Rc, Weak}, str::FromStr, time::Instant};
use crate::utils::*;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use crate::{buffer::{self, Buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};

use super::{sparse_tree::*, recursive::*};

pub const BOTTOM_NODE: u32 = u32::MAX;
pub const FULL_NODE: u32 = u32::MAX-1; 
pub const SVO_DEPTH: u32 = 5;
pub const TOTAL_SIZE: u32 = 1 << (SVO_DEPTH * 2);

fn surface_area_bitmask(bitmask: u64) -> u32 {
    let mut surface_area = 0u32;
    for x in  0..4i32 {
        for y in 0..4i32 {
            for z in 0..4i32 {
                let vec = vek::Vec3::new(x,y,z);
                let i = (z + y * 4 + x * 4 * 4) as u32;

                if !is_set(bitmask, i) {
                    continue;
                } 


                for offset in OFFSETS {
                    let vec_offsetted = vec + offset;
                            
                    if vec_offsetted.cmpge(&vek::Vec3::zero()).reduce_bitand() && vec_offsetted.cmplt(&vek::Vec3::broadcast(4)).reduce_bitand() {
                        let i_offsetted = (vec_offsetted.z + vec_offsetted.y * 4 + vec_offsetted.x * 4 * 4) as u32;
                        if !is_set(bitmask, i_offsetted) {
                            surface_area += 1;
                        } 
                    } else {
                        surface_area += 1;
                    }
                }
            }
        }
    }
    surface_area
}

struct TraversalNode<'a> {
    node: &'a FlatNode,
    depth: usize,
    parent_base_child_index: Option<usize>,
    self_packed_child_offset: usize,
}

pub fn convert_to_buffers(root_index: usize, nodes: &[FlatNode]) -> (Vec<u64>, Vec<u32>) {
    let start = Instant::now();
    let mut queue = VecDeque::<TraversalNode>::new();
    queue.push_back(TraversalNode { node: &nodes[root_index], depth: 0, parent_base_child_index: None, self_packed_child_offset: 0 });

    let mut bitmask_vec = Vec::<u64>::new();
    let mut index_vec = Vec::<u32>::new();
    let mut nodes_visited = 0;
    let mut test_count = 0u32;
    let mut base_indices_for_depth = Vec::<u32>::new();

    while let Some(TraversalNode { node, depth, parent_base_child_index: parent_index, self_packed_child_offset  }) = queue.pop_front() {
        // keeps track of the "base" indinces where we start a specific AS hierarchy level
        if base_indices_for_depth.len() < (depth+1) {
            base_indices_for_depth.push(nodes_visited);
        }

        let self_index = index_vec.len();
        
        // VERIFY: makes sure that the packed child index matches up
        if let Some(parent) = parent_index {
            debug_assert_eq!(self_index, parent + self_packed_child_offset);
        }

        // creates a 64 bit mask that contains which children are enabled
        let mut bitmask = node.children.as_ref().map(|children| children.iter()
            .enumerate()
            .filter_map(|(i, x)| x.as_ref().map(|_| i))
            .fold(0u64, |prev, i| ((1u64 << i) as u64) | prev)
        ).unwrap_or_default();
        
        let mut base_child_index = test_count + 1;

        // check if we are handling the base case
        if node.bottom {
            base_child_index = BOTTOM_NODE;
            if node.children.is_none() {
                bitmask = u64::MAX;
            }
        } else {
            if node.full {
                // node is full, discard children nodes, and store a specialized magic value that indicates that this node is full
                base_child_index = FULL_NODE;
            } else {
                // node is not full, we must compute bitmask of children and stuff
                if let Some(children) = node.children.as_ref() {
                    for (pci, (ci, child)) in children.iter().enumerate().filter_map(|(ci, x)| x.as_ref().map(|x| (ci, x))).enumerate() {
                        queue.push_back(TraversalNode { node: &nodes[*child], depth: depth + 1, parent_base_child_index: Some(base_child_index as usize), self_packed_child_offset: pci });
                        test_count += 1;

                        // VERIFY: makes sure that the packed child index matches up
                        let mask = (1u64 << ci) - 1;
                        let masked = bitmask & mask;
                        let test = masked.count_ones();
                        debug_assert_eq!(test, pci as u32);
                    }
                }
            }
        }

        bitmask_vec.push(bitmask);
        index_vec.push(base_child_index);
        nodes_visited += 1;
    }

    const CALCULATE_SAH: bool = false;

    log::debug!("base indices for depths:");
    for (i, base_index) in base_indices_for_depth.iter().enumerate() {
        log::debug!(" - depth {i}: {base_index}");
    }

    let sah_total = if CALCULATE_SAH {
        log::debug!("calculating total SAH...");
        bitmask_vec.par_iter().map(|bitmask| {
            let surface_area_4x4x4 = 4*4 * 6;
            surface_area_bitmask(*bitmask) as f64 / (surface_area_4x4x4 as f64)
        }).sum::<f64>()
    } else {
        -1f64
    };


    log::debug!("calculating total fullness...");
    let fullness_total = bitmask_vec.par_iter().map(|bitmask| {
        bitmask.count_ones() as f32
    }).sum::<f32>();


    let fullness_normalized_total_nodes = (fullness_total / nodes_visited as f32) * 100f32;
    let sah_normalized_total_nodes = (sah_total as f32 / nodes_visited as f32) * 100f32;

    log::debug!("converted svo, nodes visited: {nodes_visited}, length: {}", bitmask_vec.len());
    log::debug!("metrics:");
    log::debug!(" - fullness total: {fullness_total}");
    log::debug!(" - fullness normalized: {fullness_normalized_total_nodes:.2}%");

    if CALCULATE_SAH {
        log::debug!(" - sah total: {sah_total}");
        log::debug!(" - sah normalized: {sah_normalized_total_nodes:.2}%");
    }
    
    let end = Instant::now();
    log::debug!(" - time taken: {}ms", (end-start).as_millis());
    (bitmask_vec, index_vec)
}

struct ConvertTraversalNode<'a> {
    node: &'a RecursiveNode,
    parent_index: usize,
    child_index_relative: u32,
}

pub fn convert_recursive_to_flat_map(node: RecursiveNode) -> Vec<FlatNode> {
    let mut map = Vec::<FlatNode>::new();
    let mut queue = VecDeque::<ConvertTraversalNode>::new();
    queue.push_back(ConvertTraversalNode { node: &node, parent_index: usize::MAX, child_index_relative: 0 });

    while let Some(ConvertTraversalNode { node, parent_index, child_index_relative }) = queue.pop_front() {
        map.push(FlatNode { children: None, bottom: node.bottom, full: node.full });
        let index = map.len() - 1;

        if parent_index != usize::MAX {
            let children = map[parent_index].children.get_or_insert_with(|| Box::new([const {None}; 64]));
            children[child_index_relative as usize] = Some(index);
        }

        if let Some(children) = node.children.as_ref() {
            for (child_index_relative, child) in children.iter().enumerate().filter_map(|(i, opt)| opt.as_ref().map(|c| (i, c))) {
                queue.push_back(ConvertTraversalNode {
                    node: &child,
                    parent_index: index,
                    child_index_relative: child_index_relative as u32,
                });
            }
        }
    }

    map
}
