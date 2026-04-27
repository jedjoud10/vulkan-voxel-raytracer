use std::{cell::RefCell, collections::VecDeque, ffi::{CStr, CString}, rc::{Rc, Weak}, str::FromStr, time::Instant};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;

use crate::{buffer::{self, Buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};

pub struct SparseVoxelOctree {
    pub bitmask_buffer: Buffer,
    pub index_buffer: Buffer,
}

impl SparseVoxelOctree {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.bitmask_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
    }
}

pub struct Node {
    children: Option<Box<[Option<Rc<RefCell<Node>>>; 64]>>,
    bottom: bool, // set to true when height=0 or height=1
    full: bool, // set to true when all underlying nodes are full. This avoids having to store them in memory. If this is set then we will discard the result of `children` for any computations if the ray hits this node
    parent: Option<Weak<RefCell<Node>>>,
}

fn pseudo_random(seed: u32) -> u32 {
    let mut value = seed;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    value = value.wrapping_mul(0x322adf);
    value ^= value >> 11;
    value = value.wrapping_add(0x9e3779b9);
    value
}

fn is_set(bitmask: u64, index: u32) -> bool {
    ((bitmask >> index) & 1) == 1
}

const OFFSETS: [vek::Vec3::<i32>; 6] = [
    vek::Vec3::new(-1, 0, 0),
    vek::Vec3::new(1, 0, 0),
    vek::Vec3::new(0, -1, 0),
    vek::Vec3::new(0, 1, 0),
    vek::Vec3::new(0, 0, -1),
    vek::Vec3::new(0, 0, 1),
];

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

/*
fn full_node() -> Node {
    Node {
        children: None,
        bottom: false,
        full: true,
    }
}
*/

/*
fn test_sparse_voxel_octree_recurse(base: u32, seed: u32) -> Node {
    if base == 0 {
        return Node {
            children: None,
            bottom: true,
            full: false,
        };
    }

    let mut children = [const { None }; 64];

    // small chance to exit early
    if (pseudo_random(0xA23 ^ base ^ seed) % 10) < 2 {
        return Node {
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
    
    Node {
        children: Some(Box::new(children)),
        bottom: base == 1,
        full: false, // TODO: implement thing that figures this out by looking at `children`
    }
}

fn test_sparse_voxel_octree_root() -> Node {
    test_sparse_voxel_octree_recurse(SVO_DEPTH, 0x0323f)
}
*/

const BOTTOM_NODE: u32 = u32::MAX;
const FULL_NODE: u32 = u32::MAX-1; 
const SVO_DEPTH: u32 = 5;
const TOTAL_SIZE: u32 = 1 << (SVO_DEPTH * 2);

struct TopDownConstructorTraversalNode {
    node: Rc<RefCell<Node>>,
    depth: u32,
    origin: vek::Vec3<u32>,
}

struct BottomUpPath {
    node: Rc<RefCell<Node>>,
    node_child_index_in_parent: u32,
}


pub struct TopDownASConstructor {
    pub root: Rc<RefCell<Node>>,
}

impl TopDownASConstructor {
    pub fn register_chunk(&mut self, chunk_position: vek::Vec3<u32>, chunk: [bool; 64*64*64]) {

    }
    
    pub fn set(&mut self, pos: vek::Vec3<u32>, voxel: bool) {
        log::trace!("setting voxel at {pos} to {voxel}");

        let mut current = Some(TopDownConstructorTraversalNode { node: self.root.clone(), depth: SVO_DEPTH-1, origin: vek::Vec3::zero() });
        let mut path = Vec::<BottomUpPath>::new();

        // TOP DOWN
        while let Some(node) = current.take() {
            let size = 1 << (node.depth*2);
            
            log::trace!("depth: {}, origin: {}, size: {}, bottom: {}", node.depth, node.origin, size, node.node.borrow().bottom);
            let child_offset = (pos - node.origin) / vek::Vec3::broadcast(size);
            log::trace!("child offset: {}", child_offset);

            debug_assert!(child_offset.cmpge(&vek::Vec3::broadcast(0u32)).reduce_and());
            debug_assert!(child_offset.cmplt(&vek::Vec3::broadcast(4u32)).reduce_and());
            let (x, y, z) = child_offset.into_tuple();
            let child_index = z + y * 4 + x * 4 * 4;

            let mut node_node = node.node.borrow_mut();
            let children = node_node.children.get_or_insert_with(|| Box::new([const { None }; 64])).as_mut_slice();
            

            let child = children[child_index as usize].get_or_insert_with(|| {
                Rc::new(RefCell::new(Node {
                    children: None,
                    bottom: node.depth == 1,
                    full: false,
                    parent: Some(Rc::downgrade(&node.node)),
                }))
            }).clone();

            
            path.push(BottomUpPath { node: child.clone(), node_child_index_in_parent: child_index });

            // set bottom most child as FULL
            if node.depth == 0 && voxel {
                let mut borrowed = child.borrow_mut();
                borrowed.full = true;
            }
            
            // remove bottom most child
            if node.depth == 0 && !voxel {            
                if !voxel {
                    children[child_index as usize].take().unwrap();
                }
            }

            if node.depth > 0 {
                current = Some(TopDownConstructorTraversalNode {
                    node: child,
                    depth: node.depth - 1,
                    origin: child_offset * size + node.origin,
                });
            }
        }

        // BOTTOM UP
        for node in path.into_iter().rev() {
            let mut borrowed = node.node.borrow_mut();
            if voxel {
                // recalculate node fullness based on children fullness
                borrowed.full = borrowed.children.as_ref().map(|children| children.iter().all(|child| child.as_ref().map(|c| c.borrow().full).unwrap_or_default())).unwrap_or_default();
            } else {
                // get rid of node if all children are missing
                if borrowed.children.is_none() || borrowed.children.as_ref().map(|children| children.iter().all(|opt_child| opt_child.is_none())).unwrap_or_default() {
                    let index_in_parent_child_buffer = node.node_child_index_in_parent;
                    let parent = borrowed.parent.clone().expect("no parent").upgrade().expect("upgrade failed");
                    let mut parent_borrowed = parent.borrow_mut();
                    parent_borrowed.children.as_mut().unwrap()[index_in_parent_child_buffer as usize] = None;
                }
            }
        }
    }
}

struct TraversalNode {
    node: Rc<RefCell<Node>>,
    depth: usize,
    parent_base_child_index: Option<usize>,
    self_packed_child_offset: usize,
}

pub fn convert_to_buffers(node: Rc<RefCell<Node>>) -> (Vec<u64>, Vec<u32>) {
    let start = Instant::now();
    let mut queue = VecDeque::<TraversalNode>::new();
    queue.push_back(TraversalNode { node: node, depth: 0, parent_base_child_index: None, self_packed_child_offset: 0 });

    let mut bitmask_vec = Vec::<u64>::new();
    let mut index_vec = Vec::<u32>::new();
    let mut nodes_visited = 0;
    let mut test_count = 0u32;
    let mut base_indices_for_depth = Vec::<u32>::new();

    while let Some(TraversalNode { node, depth, parent_base_child_index: parent_index, self_packed_child_offset  }) = queue.pop_front() {
        let node = node.borrow();

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
                        queue.push_back(TraversalNode { node: child.clone(), depth: depth + 1, parent_base_child_index: Some(base_child_index as usize), self_packed_child_offset: pci });
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

    log::debug!("base indices for depths:");
    for (i, base_index) in base_indices_for_depth.iter().enumerate() {
        log::debug!(" - depth {i}: {base_index}");
    }

    let sah_total = if false {
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
    log::debug!(" - sah total: {sah_total}");
    log::debug!(" - sah normalized: {sah_normalized_total_nodes:.2}%");
    
    let end = Instant::now();
    log::debug!(" - time taken: {}ms", (end-start).as_millis());

    /*
    for i in 0..50 {
        let index = (i + base_indices_for_depth.last().unwrap()) as usize;
        let node_bitmask = &bitmask_vec[index];
        let node_index = &index_vec[index];
        log::debug!("node index: {node_index}, bitmask: {node_bitmask:#b}");
    }
    */


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
) -> (SparseVoxelOctree, TopDownASConstructor) {
    log::debug!("creating svo nodes...");
    //let root_node = test_sparse_voxel_octree_root();
    //let mut constructor = TopDownASConstructor { root: root_node };
    let mut constructor = TopDownASConstructor { root: Rc::new(RefCell::new(Node { children: None, bottom: false, full: false, parent: None  })) };

    for i in 0..50000 {
        let x = pseudo_random(i) % TOTAL_SIZE;
        let y = pseudo_random(i + x * 3231) % 4;
        let z = pseudo_random(i + y * 1212) % TOTAL_SIZE;
        
        constructor.set(vek::Vec3::new(x,y,z), true);
    }

    constructor.set(vek::Vec3::new(0,0,0), true);
    constructor.set(vek::Vec3::new(0,1,0), true);

    constructor.set(vek::Vec3::new(10,20,30), true);
    constructor.set(vek::Vec3::new(40,50,60), true);
    
    let root_node = constructor.root.clone();
    
    let (bitmasks, indices) = convert_to_buffers(root_node);

    // TODO: does this make sense? no... but... it works...
    let max_svo_element_size = 4096 * 64 * 64;

    let bitmask_buffer = buffer::create_buffer(&device, &mut allocator, max_svo_element_size * size_of::<u64>(), &binder, "sparse voxel octree brick bitmasks", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
    let index_buffer = buffer::create_buffer(&device, &mut allocator, max_svo_element_size * size_of::<u32>(), &binder, "sparse voxel octree child indices", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);

    let bitmasks_buffer_bytes = bytemuck::cast_slice::<_, u8>(bitmasks.as_slice());
    let indices_buffer_bytes = bytemuck::cast_slice::<_, u8>(indices.as_slice());
    buffer::write_to_buffer(device, pool, queue, bitmask_buffer.buffer, &mut allocator, bitmasks_buffer_bytes);
    buffer::write_to_buffer(device, pool, queue, index_buffer.buffer, &mut allocator, indices_buffer_bytes);

    (SparseVoxelOctree { bitmask_buffer, index_buffer }, constructor)
}

#[derive(Clone, Copy)]
pub struct Voxel {
    pub active: bool,
    pub reflective: bool,
    pub refractive: bool,
    pub placed: bool,
}

impl Voxel {
    pub fn into_raw(self) -> u8 {
        self.active as u8 | (self.reflective as u8) << 1 | (self.refractive as u8) << 2 | (self.placed as u8) << 3
    }
}