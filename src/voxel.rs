use std::{cell::RefCell, collections::VecDeque, ffi::{CStr, CString}, num::NonZeroU32, rc::{Rc, Weak}, str::FromStr, time::Instant};
use crate::utils::*;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use crate::{buffer::{self, Buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};

const BOTTOM_NODE: u32 = u32::MAX;
const FULL_NODE: u32 = u32::MAX-1; 
const SVO_DEPTH: u32 = 5;
const TOTAL_SIZE: u32 = 1 << (SVO_DEPTH * 2);

struct TopDownConstructorTraversalNode {
    index: usize,
    height: u32,
    origin: vek::Vec3<u32>,
}

pub struct SparseVoxelOctree {
    pub bitmask_buffer: Buffer,
    pub index_buffer: Buffer,
    pub nodes: Vec<FlatNode>,
}

impl SparseVoxelOctree {
    pub fn register_chunk(&mut self, chunk_position: vek::Vec3<u32>, chunk: [bool; 64*64*64]) {
        todo!();
    }
    
    pub fn set(&mut self, pos: vek::Vec3<u32>, voxel: bool) {
        log::trace!("setting voxel at {pos} to {voxel}");

        let mut current = Some(TopDownConstructorTraversalNode { index: 0, height: SVO_DEPTH-1, origin: vek::Vec3::zero() });
        let mut path = Vec::<BottomUpPath>::new();

        // TOP DOWN
        while let Some(node) = current.take() {
            let size = 1 << (node.height*2);
            
            log::trace!("depth: {}, origin: {}, size: {}, bottom: {}", node.height, node.origin, size, self.nodes[node.index].bottom);
            let child_offset = (pos - node.origin) / vek::Vec3::broadcast(size);
            log::trace!("child offset: {}", child_offset);

            debug_assert!(child_offset.cmpge(&vek::Vec3::broadcast(0u32)).reduce_and());
            debug_assert!(child_offset.cmplt(&vek::Vec3::broadcast(4u32)).reduce_and());
            let (x, y, z) = child_offset.into_tuple();
            let child_index_relative = z + y * 4 + x * 4 * 4;

            // no need to do anything if we are trying to add a voxel to an already full sub-tree
            if voxel && self.nodes[node.index].full {
                break;
            }

            self.nodes[node.index].children.get_or_insert_with(|| Box::new([const { None }; 64]));

            // if we are removing a voxel from a full node, we must add all of its OTHER children that must be full, except the one we are modifying
            let mut was_full = false;
            if !voxel && self.nodes[node.index].full {
                self.nodes[node.index].full = false;
                was_full = true;
                for i in 0..64 {
                    if i != child_index_relative as usize /* && self.nodes[node.index].children.as_ref().unwrap()[i].is_none() */ {
                        self.nodes.push(FlatNode {
                            children: None,
                            bottom: node.height == 1,
                            full: true,
                        });
                        let idx = self.nodes.len()-1;
                        self.nodes[node.index].children.as_mut().unwrap()[i] = Some(idx);
                    }
                }
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
            
            path.push(BottomUpPath { parent_index: node.index, child_index_relative, child_index_absolute });

            // set bottom most child as FULL
            if node.height == 0 && voxel {
                self.nodes[child_index_absolute as usize].full = true;
            }
            
            // remove bottom most child
            if node.height == 0 && !voxel {            
                // TODO: also remove the node in the array, but that requires shifting all indices and recalculating them... :(
                // need to use a slotmap....
                self.nodes[node.index].children.as_mut().unwrap()[child_index_relative as usize].take().unwrap();
            }

            // if the parent (node.index) was full, then the just added node must ALSO be full so that we can recursively set its children as full until we reach the bottom
            if was_full {
                self.nodes[child_index_absolute as usize].full = true;
            }

            if node.height > 0 {
                current = Some(TopDownConstructorTraversalNode {
                    index: child_index_absolute as usize,
                    height: node.height - 1,
                    origin: child_offset * size + node.origin,
                });
            }
        }

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
        let (bitmasks, indices) = convert_to_buffers(0, &self.nodes);
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

// TODO: optimize space using NonZero and bitmasks for bottom&full
pub struct FlatNode {
    children: Option<Box<[Option<usize>; 64]>>,
    bottom: bool,
    full: bool,
}

struct BottomUpPath {
    parent_index: usize,
    child_index_relative: u32,
    child_index_absolute: usize,
}

pub struct RecursiveNode {
    children: Option<Box<[Option<Box<RecursiveNode>>; 64]>>,
    bottom: bool, // set to true when height=0 or height=1
    full: bool, // set to true when all underlying nodes are full. This avoids having to store them in memory. If this is set then we will discard the result of `children` for any computations if the ray hits this node
}

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

fn full_node() -> RecursiveNode {
    RecursiveNode {
        children: None,
        bottom: false,
        full: true,
    }
}

fn test_sparse_voxel_octree_recurse(base: u32, seed: u32) -> RecursiveNode {
    if base == 0 {
        return RecursiveNode {
            children: None,
            bottom: true,
            full: false,
        };
    }

    let mut children = [const { None }; 64];

    // small chance to exit early
    if (pseudo_random(0xA23 ^ base ^ seed) % 10) < 2 {
        return RecursiveNode {
            children: None,
            bottom: false,
            full: false,
        }
    }

    let mut recursive= SmallVec::<[u8; 8]>::new();

    for x in  0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let vec = vek::Vec3::new(x,y,z);
                let i = x + y * 4 + z * 4 * 4;

                // base plate
                if y == 0 {
                    children[i] = Some(Box::new(full_node()));
                    //children[i] = Some(Box::new(base_plate(base - 1)));
                }

                if y == 1 {
                    if (x == 0 || x == 3) || (z == 0 || z == 3) {
                        // inductive case
                        recursive.push(i as u8);
                    } else {
                        // mid plate
                        //recursive.push(i as u8);
                        children[i] = Some(Box::new(full_node()));
                        //children[i] = Some(Box::new(base_plate(base - 1)));
                    }
                }

                if y == 2 {
                    if (x == 1 || x == 2) && (z == 1 || z == 2) {
                        // inductive case
                        recursive.push(i as u8);
                    }
                }
            }
        }
    }

    let generated_children = recursive.into_par_iter().map(|i| {
        let dst_index = (pseudo_random((*i as u32) ^ 0x03f23 ^ base ^ seed) % 64);
        let child = Box::new(test_sparse_voxel_octree_recurse(base - 1, dst_index));
        (i, child)
    }).collect::<Vec<_>>();

    for (dst_index, child) in generated_children {
        children[*dst_index as usize] = Some(child);
    }
    let full = children.iter().all(|child| child.as_ref().map(|c| c.full).unwrap_or_default());
    
    RecursiveNode {
        children: Some(Box::new(children)),
        bottom: base == 1,
        full: full, 
    }
}

fn test_sparse_voxel_octree_root() -> RecursiveNode {
    test_sparse_voxel_octree_recurse(SVO_DEPTH, 0x0323f)
}

struct ConvertTraversalNode<'a> {
    node: &'a RecursiveNode,
    parent_index: usize,
    child_index_relative: u32,
}

fn convert_recursive_to_flat_map(node: RecursiveNode) -> Vec<FlatNode> {
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

// each node contains a u64 bitmask that checks if any of its children are leaf nodes
// another buffer stores the "children base" index references as u16s
// it contains a bitmask of its children
pub unsafe fn create_sparse_voxel_octree(
    device: &ash::Device,
    mut allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    queue_family_index: u32,
) -> SparseVoxelOctree {
    log::debug!("creating svo nodes...");

    let max_svo_element_size = 4096 * 64 * 64;
    let bitmask_buffer = buffer::create_buffer(&device, &mut allocator, max_svo_element_size * size_of::<u64>(), &binder, "sparse voxel octree brick bitmasks", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
    let index_buffer = buffer::create_buffer(&device, &mut allocator, max_svo_element_size * size_of::<u32>(), &binder, "sparse voxel octree child indices", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
    //let root = Rc::new(RefCell::new(Node { children: None, bottom: false, full: false, parent: None  }));
    let mut svo = SparseVoxelOctree { bitmask_buffer, index_buffer, nodes: convert_recursive_to_flat_map(test_sparse_voxel_octree_root()) };
    
    for i in 0..50000 {
        let x = pseudo_random(i) % TOTAL_SIZE;
        let y = pseudo_random(i + x * 3231) % 32 + 256;
        let z = pseudo_random(i + y * 1212) % TOTAL_SIZE;
        
        svo.set(vek::Vec3::new(x,y,z), true);
    }

    svo.rebuild(device, pool, queue, allocator);
    svo    
}