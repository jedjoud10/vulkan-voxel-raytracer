use std::{cell::RefCell, collections::VecDeque, ffi::{CStr, CString}, num::NonZeroU32, ops::ControlFlow, rc::{Rc, Weak}, str::FromStr, time::Instant};
use crate::utils::*;
use ash::vk;
use bit_set::BitSet;
use bit_vec::BitVec;
use gpu_allocator::vulkan::{Allocation, Allocator};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use crate::{buffer::{self, Buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};

use super::{SVO_DEPTH, TOTAL_SIZE};

pub struct SparseVoxelOctree {
    pub bitmask_buffer: Buffer,
    pub index_buffer: Buffer,
    pub nodes: Vec<FlatNode>,
}

impl SparseVoxelOctree {
    // TODO: impl
    pub fn register_chunk(&mut self, chunk_position: vek::Vec3<u32>, chunk: BitVec) {
        let full = chunk.all();
        let empty = chunk.any();

        /*
        const CHUNK_64_HEIGHT: u32 = 3;

        let mut tmp_nodes = Vec::<FlatNode>::new();

        // bottom up approach: do multiple passes, starting from the bottom
        let mut any_mip = chunk.clone();
        let mut all_mip = chunk;
        for pass in 0..3 {
            let mip_size = 64 / (1 << ((1+pass)*2));
            let next_any_mip = BitSet::with_capacity((mip_size as usize).pow(3));
            let next_all_mip = BitSet::with_capacity((mip_size as usize).pow(3));

            for x in 0..mip_size {
                for y in 0..mip_size {
                    for z in 0..mip_size {
                        let mut any = false;
                        let mut all = true;

                        for local_x in 0..4 {
                            for local_y in 0..4 {
                                for local_z in 0..4 {
                                    let offset = vek::Vec3::new(x,y,z);
                                    let local_offset = vek::Vec3::new(local_x, local_y, local_z);
                                    let position: vek::Vec3<i32> = local_offset * 4 + offset;
                                    let i = (position.x + position.y * 4 + position.z * 4 * 4) as usize; 
                                    any |= any_mip.get(i).unwrap();
                                    all &= all_mip.get(i).unwrap();
                                }
                            }
                        }

                        let i = (x + y * 4 + z * 4 * 4) as usize; 
                        next_any_mip.get_mut().set();
                    }
                }
            }
        }
        

        
        let prefire_callback = |nodes: &mut Vec<FlatNode>, node: &TopDownTraversalNode, child_index_relative: u32| -> ControlFlow<(), ()> {
            ControlFlow::Continue(())
        };

        let postfire_callback = |nodes: &mut Vec<FlatNode>, node: &TopDownTraversalNode, child_index_absolute: usize, child_index_relative: u32| -> ControlFlow<(), ()> {
            if node.height == CHUNK_64_HEIGHT {
            
                let chunk_node = &mut nodes[child_index_absolute as usize];
                // do the funny moment stuff here...
            
            }

            ControlFlow::Continue(())
        };

        self.traverse(chunk_position, CHUNK_64_HEIGHT-1, prefire_callback, postfire_callback);
        */
    }

    fn traverse<A: FnMut(&mut Vec<FlatNode>, &TopDownTraversalNode, u32) -> ControlFlow<(), ()>, B: FnMut(&mut Vec<FlatNode>, &TopDownTraversalNode, usize, u32) -> ControlFlow<(), ()>>(&mut self, pos: vek::Vec3<u32>, target_height: u32, mut prefire_callback: A, mut postfire_callback: B) {
        assert!(pos.cmpge(&vek::Vec3::broadcast(0u32)).reduce_and());
        assert!(pos.cmplt(&vek::Vec3::broadcast(TOTAL_SIZE)).reduce_and());
        
        let mut current = Some(TopDownTraversalNode { index: 0, height: SVO_DEPTH-1, origin: vek::Vec3::zero() });

        while let Some(node) = current.take() {
            let size = 1 << (node.height*2);
            
            log::trace!("depth: {}, origin: {}, size: {}, bottom: {}", node.height, node.origin, size, self.nodes[node.index].bottom);
            let child_offset = (pos - node.origin) / vek::Vec3::broadcast(size);
            log::trace!("child offset: {}", child_offset);

            assert!(child_offset.cmpge(&vek::Vec3::broadcast(0u32)).reduce_and());
            assert!(child_offset.cmplt(&vek::Vec3::broadcast(4u32)).reduce_and());
            let (x, y, z) = child_offset.into_tuple();
            let child_index_relative = z + y * 4 + x * 4 * 4;

            let r = prefire_callback(&mut self.nodes, &node, child_index_relative);
            if r.is_break() {
                break;
            }

            let child_index_absolute = if let Some(idx) = self.nodes[node.index].children.as_mut().unwrap()[child_index_relative as usize] {
                idx
            } else {
                self.nodes.push(FlatNode {
                    children: None,
                    bottom: node.height == 1,
                    full: false,
                });
                let idx = self.nodes.len()-1;
                self.nodes[node.index].children.as_mut().unwrap()[child_index_relative as usize] = Some(idx);
                idx
            };
            
            let r = postfire_callback(&mut self.nodes, &node, child_index_absolute, child_index_relative);
            if r.is_break() {
                break;
            }

            if node.height > target_height {
                current = Some(TopDownTraversalNode {
                    index: child_index_absolute as usize,
                    height: node.height - 1,
                    origin: child_offset * size + node.origin,
                });
            }
        }
    } 
    
    // TODO: unshittify
    pub fn set(&mut self, pos: vek::Vec3<u32>, voxel: bool) {
        if pos.cmplt(&vek::Vec3::broadcast(0u32)).reduce_or() || pos.cmpge(&vek::Vec3::broadcast(TOTAL_SIZE)).reduce_or() {
            log::trace!("voxel at {pos} is OOB, ignoring");
            return;
        }

        log::trace!("setting voxel at {pos} to {voxel}");

        let mut path = Vec::<BottomUpPath>::new();

        let prefire_callback = |nodes: &mut Vec<FlatNode>, node: &TopDownTraversalNode, child_index_relative: u32| -> ControlFlow<(), ()> {
            // no need to do anything if we are trying to add a voxel to an already full sub-tree
            if voxel && nodes[node.index].full {
                return ControlFlow::Break(());
            }

            nodes[node.index].children.get_or_insert_with(|| Box::new([const { None }; 64]));

            // if we are removing a voxel from a full node, we must add all of its OTHER children that must be full, except the one we are modifying
            if !voxel && nodes[node.index].full {
                for i in 0..64 {
                    if i != child_index_relative as usize /* && self.nodes[node.index].children.as_ref().unwrap()[i].is_none() */ {
                        nodes.push(FlatNode {
                            children: None,
                            bottom: node.height == 1,
                            full: true,
                        });
                        let idx = nodes.len()-1;
                        nodes[node.index].children.as_mut().unwrap()[i] = Some(idx);
                    }
                }
            }

            ControlFlow::Continue(())
        };


        let postfire_callback = |nodes: &mut Vec<FlatNode>, node: &TopDownTraversalNode, child_index_absolute: usize, child_index_relative: u32| -> ControlFlow<(), ()> {
            path.push(BottomUpPath { parent_index: node.index, child_index_relative, child_index_absolute });

            // set bottom most child as FULL
            if node.height == 0 && voxel {
                nodes[child_index_absolute as usize].full = true;
            }
            
            // remove bottom most child
            if node.height == 0 && !voxel {            
                // TODO: also remove the node in the array, but that requires shifting all indices and recalculating them... :(
                // need to use a slotmap....
                nodes[node.index].children.as_mut().unwrap()[child_index_relative as usize].take().unwrap();
            }

            // if the parent (node.index) was full, then the just added node must ALSO be full so that we can recursively set its children as full until we reach the bottom
            if !voxel && nodes[node.index].full {
                nodes[node.index].full = false;
                nodes[child_index_absolute as usize].full = true;
            }

            ControlFlow::Continue(())
        };

        self.traverse(pos, 0, prefire_callback, postfire_callback);

        // BOTTOM UP
        for (depth, node) in path.iter().enumerate().rev() {
            if voxel {
                // recalculate node fullness based on children fullness
                if let Some(children) = self.nodes[node.child_index_absolute].children.as_ref() {
                    let full = children.iter().all(|child| child.as_ref().map(|c| self.nodes[*c as usize].full).unwrap_or_default());
                    self.nodes[node.child_index_absolute].full = full;
                }
            } else {
                // get rid of node if all children are missing
                let borrowed = &self.nodes[node.child_index_absolute];
                if (borrowed.children.is_none() && !borrowed.full) || borrowed.children.as_ref().map(|children| children.iter().all(|opt_child| opt_child.is_none())).unwrap_or_default() {
                    self.nodes[node.parent_index].children.as_mut().unwrap()[node.child_index_relative as usize] = None;
                }
            }
        }
    }

    // TODO: optimize by only re-building what has changed, though that will require a way to figure out new indices and bitmasks without having to rewrite the entirety of the buffers
    // FIXME: very very bad! does a full rebuild of the AS even though we might have only modified a few nodes here and there
    // unfortunately, as the AS is using packed buffer and packed nodes, *adding* a new node will require you to shift all nodes after that
    // depending on the child indexing order, this could be very cheap (i.e inserting near the end) or very expensive (i.e inserting near the front and having to shift all elements after that)
    pub unsafe fn rebuild(&mut self, device: &ash::Device, pool: vk::CommandPool, queue: vk::Queue, mut allocator: &mut Allocator) {
        let (bitmasks, indices) = super::convert_to_buffers(0, &self.nodes);
        let bitmasks_buffer_bytes = bytemuck::cast_slice::<_, u8>(bitmasks.as_slice());
        let indices_buffer_bytes = bytemuck::cast_slice::<_, u8>(indices.as_slice());

        // TODO: use the dedicated per-frame command buffer and a scratch buffer and avoid doing the device.wait_idle() inside these calls
        buffer::write_to_buffer(device, pool, queue, self.bitmask_buffer.buffer, &mut allocator, bitmasks_buffer_bytes);
        buffer::write_to_buffer(device, pool, queue, self.index_buffer.buffer, &mut allocator, indices_buffer_bytes);
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.bitmask_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
    }
}

struct TopDownTraversalNode {
    index: usize,
    height: u32,
    origin: vek::Vec3<u32>,
}

// TODO: optimize space using NonZero and bitmasks for bottom&full
pub struct FlatNode {
    pub children: Option<Box<[Option<usize>; 64]>>,
    pub bottom: bool,
    pub full: bool,
}

struct BottomUpPath {
    pub parent_index: usize,
    pub child_index_relative: u32,
    pub child_index_absolute: usize,
}