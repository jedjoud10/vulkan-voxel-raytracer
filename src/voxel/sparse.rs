use std::{collections::VecDeque, time::Instant};
use crate::utils::*;
use crate::voxel::chunk::CHUNK_SIZE;
use crate::voxel::util::{PartialSparseImageChunk, SparseImageChunk};
use ash::vk;
use gpu_allocator::vulkan::Allocator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use vek::Clamp;
use crate::buffer::{self, Buffer};
use super::{SVO_DEPTH, TOTAL_SIZE, FULL_NODE};
use super::chunk::Chunk;

pub struct SparseVoxelOctree {
    pub bitmask_buffer: Buffer,
    pub index_buffer: Buffer,
    pub aabb_buffer: Buffer,
    pub root: TopLevelAccelerationStructureNode,
    pub chunks: Vec<Chunk>,
}

impl SparseVoxelOctree {
    pub unsafe fn new_with_root_node(
        device: &ash::Device,
        allocator: &mut Allocator,
        binder: &Option<ash::ext::debug_utils::Device>,
    ) -> Self {
        // each node contains a u64 bitmask that checks if any of its children are leaf nodes
        // another buffer stores the "children base" index references as u16s
        // it contains a bitmask of its children
        log::debug!("creating SVO...");
        let max_svo_element_size = 4096 * 64 * 16;
        let aabb_buffer = buffer::create_buffer(device, allocator, max_svo_element_size * size_of::<u64>(), binder, "sparse voxel octree aabb bounds", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        let bitmask_buffer = buffer::create_buffer(device, allocator, max_svo_element_size * size_of::<u64>(), binder, "sparse voxel octree brick bitmasks", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        let index_buffer = buffer::create_buffer(device, allocator, max_svo_element_size * size_of::<u32>(), binder, "sparse voxel octree child indices", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        
        Self {
            bitmask_buffer,
            index_buffer,
            aabb_buffer,
            chunks: Vec::new(),
            root: TopLevelAccelerationStructureNode::root(),
        }
    }

    pub fn register_chunk(&mut self, chunk: Chunk) {
        log::debug!("registering chunk {}", chunk.position);

        if chunk.is_empty() {
            log::debug!("chunk was empty, ignoring");
            return;
        }

        let pos = chunk.position * CHUNK_SIZE as u32;
        
        log::trace!("boogaloo time for chunk with origin world pos : {pos} and chunk pos : {}", chunk.position);
        let mut current = Some(TopDownTraversalNode2 { node: &mut self.root, height: SVO_DEPTH-1, origin: vek::Vec3::zero() });

        while let Some(node) = current.take() {
            let size = 1 << (node.height*2);
            
            log::trace!("height: {}, origin: {}, size: {}", node.height, node.origin, size);
            let child_offset = (pos - node.origin) / vek::Vec3::broadcast(size);
            log::trace!("child offset: {}", child_offset);

            assert!(child_offset.cmpge(&vek::Vec3::broadcast(0u32)).reduce_and());
            assert!(child_offset.cmplt(&vek::Vec3::broadcast(4u32)).reduce_and());
            let (x, y, z) = child_offset.into_tuple();

            let child_index_relative = x + y * 4 + z * 4 * 4;

            node.node.bounds.expand_to_contain(chunk.bounds);


            // no need to do anything if we are trying to add a voxel to an already full sub-tree
            if chunk.is_full() && node.node.full {
                log::trace!("node was already full and chunk was already full too. exiting early");
                break;
            }

            node.node.children.get_or_insert_with(|| {
                if node.height == 3 {
                    log::trace!("adding chunk node children");
                    TopLevelAccelerationStructureNodeChildren::ChunkNodeChildren { children: Box::new([const { None }; 64]) }
                } else {
                    log::trace!("adding recursive node children");
                    TopLevelAccelerationStructureNodeChildren::RecursiveNodeChildren { children: Box::new([const { None }; 64]) }
                }
            });

            // if we are removing a voxel from a full node, we must add all of its OTHER children that must be full, except the one we are modifying
            if !chunk.is_full() && node.node.full {
                log::trace!("replacing siblings with full recursive nodes");
                for i in 0..64 {
                    if i != child_index_relative as usize /* && self.nodes[node.index].children.as_ref().unwrap()[i].is_none() */ {
                        let new_full_child = TopLevelAccelerationStructureNode {
                            children: None,
                            full: true,
                            bounds: vek::Aabb::new_empty(vek::Vec3::zero())
                        };
                        let k = node.node.children.as_mut().unwrap();
                        match k {
                            TopLevelAccelerationStructureNodeChildren::RecursiveNodeChildren { children } => {
                                children[i] = Some(Box::new(new_full_child));
                            },
                            TopLevelAccelerationStructureNodeChildren::ChunkNodeChildren { .. } => unreachable!(),
                        } 
                    }
                }
            }

            if node.height == 3 {
                // this node is the parent of the chunk level acceleration structure nodes
                let children = match node.node.children.as_mut().unwrap() {
                    TopLevelAccelerationStructureNodeChildren::RecursiveNodeChildren { .. } => unreachable!(),
                    TopLevelAccelerationStructureNodeChildren::ChunkNodeChildren { children } => children,
                };

                let child_mut_ref = &mut children[child_index_relative as usize];

                if let Some(valid_child) = child_mut_ref {
                    // overwrite?
                    log::trace!("overwrite chunk child node sparse representation");
                    *valid_child = chunk.sparse_representation.clone(); 
                    break;
                } else {
                    log::trace!("first write chunk child node sparse representation");
                    *child_mut_ref = Some(chunk.sparse_representation.clone());
                    break;
                }
            } else {
                // top level acceleration structure node
                let children = match node.node.children.as_mut().unwrap() {
                    TopLevelAccelerationStructureNodeChildren::RecursiveNodeChildren { children } => children,
                    TopLevelAccelerationStructureNodeChildren::ChunkNodeChildren { .. } => unreachable!(),
                };
            
                let child_mut_ref = &mut children[child_index_relative as usize];
                
                if let Some(valid_child) = child_mut_ref {
                    valid_child.bounds.expand_to_contain(chunk.bounds);
                } else {
                    let new_child = TopLevelAccelerationStructureNode {
                        children: None,
                        full: false,
                        bounds: chunk.bounds,
                    };
                    *child_mut_ref = Some(Box::new(new_child));
                }

                // if the parent (node.index) was full, then the just added node must ALSO be full so that we can recursively set its children as full until we reach the bottom
                if chunk.is_full() && node.node.full {
                    node.node.full = false;
                    child_mut_ref.as_mut().unwrap().full = true;
                }
            
                current = Some(TopDownTraversalNode2 {
                    node: child_mut_ref.as_mut().unwrap(),
                    height: node.height - 1,
                    origin: child_offset * size + node.origin,
                });
            }
        }
    
        // TODO: only thing missing here is getting rid of walks in the tree that solely consist of empty nodes


        self.chunks.push(chunk);
    }

    pub unsafe fn rebuild(&mut self, device: &ash::Device, pool: vk::CommandPool, queue: vk::Queue, allocator: &mut Allocator) {
        let res = convert_to_buffers(self);
        self.apply_update_gpu_buffers(device, pool, queue, allocator, &res);
    }

    pub unsafe fn apply_update_gpu_buffers(&mut self, device: &ash::Device, pool: vk::CommandPool, queue: vk::Queue, allocator: &mut Allocator, built: &SparseVoxelTreeBuildResultGpuBuffers) {
        log::info!("writing new data to sparse voxel tree buffers...");
        let bitmasks_buffer_bytes = bytemuck::cast_slice::<_, u8>(built.bitmasks.as_slice());
        let indices_buffer_bytes = bytemuck::cast_slice::<_, u8>(built.indices.as_slice());
        let aabb_buffer_bytes = bytemuck::cast_slice::<_, u8>(built.aabbs.as_slice());
    
        // TODO: use the dedicated per-frame command buffer and a scratch buffer and avoid doing the device.wait_idle() inside these calls
        buffer::write_to_buffer(device, pool, queue, self.bitmask_buffer.buffer, allocator, bitmasks_buffer_bytes);
        buffer::write_to_buffer(device, pool, queue, self.index_buffer.buffer, allocator, indices_buffer_bytes);
        buffer::write_to_buffer(device, pool, queue, self.aabb_buffer.buffer, allocator, aabb_buffer_bytes);
    }
    
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.bitmask_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
        self.aabb_buffer.destroy(device, allocator);
    }
}

struct TopDownTraversalNode2<'a> {
    node: &'a mut TopLevelAccelerationStructureNode,
    height: u32,
    origin: vek::Vec3<u32>,
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub struct SparseVoxelTreeBuildResultGpuBuffers {
    #[n(0)] pub indices: Vec<u32>,
    #[n(1)] pub bitmasks: Vec<u64>,
    #[n(2)] pub aabbs: Vec<u64>,
}

#[derive(minicbor::Encode, minicbor::Decode)]
pub struct SparseVoxelTreeSaveState {
    #[n(0)]
    pub chunks: Vec<Chunk>,
}

impl TopLevelAccelerationStructureNode {
    pub fn root() -> Self {
        Self { children: None, full: false, bounds: vek::Aabb::new_empty(vek::Vec3::zero()) }
    }
}

pub enum TopLevelAccelerationStructureNodeChildren {
    RecursiveNodeChildren {
        children: Box<[Option<Box<TopLevelAccelerationStructureNode>>; 64]>,
    },

    ChunkNodeChildren {
        children: Box<[Option<Vec<ChunkLevelAccelerationStructureNode>>; 64]>,
    },
}

pub struct TopLevelAccelerationStructureNode {
    pub bounds: vek::Aabb<u32>,
    pub children: Option<TopLevelAccelerationStructureNodeChildren>,
    pub full: bool,
}

#[derive(Clone)]
pub struct ChunkLevelAccelerationStructureNode {
    pub bounds: vek::Aabb<u32>,
    pub children: Option<Box<[Option<usize>; 64]>>,
    pub full: bool,
}

enum TraversalNodeType<'a> {
    TopLevel {
        node: &'a TopLevelAccelerationStructureNode,
    },

    ChunkLevel {
        chunk_flat_array: &'a [ChunkLevelAccelerationStructureNode],
        node: &'a ChunkLevelAccelerationStructureNode,
    }
}

fn array_of_options_to_bitmask<T>(arr: &[Option<T>; 64]) -> u64 {
    arr.iter()
        .enumerate()
        .filter_map(|(i, x)| x.as_ref().map(|_| i))
        .fold(0u64, |prev, i| (1u64 << i) | prev)
}

impl<'a> TraversalNodeType<'a> {
    fn get_children_bitmask(&self) -> u64 {
        match self {
            TraversalNodeType::TopLevel { node } => {
                node.children.as_ref().map(|children| {
                    match children {
                        TopLevelAccelerationStructureNodeChildren::RecursiveNodeChildren { children } => array_of_options_to_bitmask(children),
                        TopLevelAccelerationStructureNodeChildren::ChunkNodeChildren { children } => array_of_options_to_bitmask(children),
                    }
                }).unwrap_or_default()
            },
            TraversalNodeType::ChunkLevel { node, .. } => {
                node.children.as_ref().map(|children| array_of_options_to_bitmask(children)).unwrap_or_default()
            },
        }
    }

    fn is_full(&self) -> bool {
        match self {
            TraversalNodeType::TopLevel { node } => node.full,
            TraversalNodeType::ChunkLevel { node, .. } => node.full,
        }
    }
    
    fn bounds(&self) -> vek::Aabb<u32> {
        match self {
            TraversalNodeType::TopLevel { node } => node.bounds,
            TraversalNodeType::ChunkLevel { node, .. } => node.bounds,
        }
    }
}

struct TraversalNode<'a> {
    node_type: TraversalNodeType<'a>,
    height: u32,
    parent_base_child_index: Option<usize>,
    self_packed_child_offset: usize,
}

fn pack_aabb_bounds(vek::Aabb { min, max }: vek::Aabb<u32>, represents_cuboid: bool) -> u64 {
    let min = min.clamped(0, TOTAL_SIZE-1);
    let max = max.clamped(0, TOTAL_SIZE-1);

    let min = (min.x | min.y << 10 | min.z << 20) as u64;
    let max = (max.x | max.y << 10 | max.z << 20) as u64;
    let mut flags = 0u64;
    
    crate::utils::set_bit(&mut flags, 0, represents_cuboid);
    
    debug_assert_eq!(flags & 0b1111, flags);
    min | max << 30 | flags << 60
}


// TODO: optimize by only re-building what has changed, though that will require a way to figure out new indices and bitmasks without having to rewrite the entirety of the buffers
// FIXME: very very bad! does a full rebuild of the AS even though we might have only modified a few nodes here and there
// unfortunately, as the AS is using packed buffer and packed nodes, *adding* a new node will require you to shift all nodes after that
// depending on the child indexing order, this could be very cheap (i.e inserting near the end) or very expensive (i.e inserting near the front and having to shift all elements after that)
pub fn convert_to_buffers(svo: &SparseVoxelOctree) -> SparseVoxelTreeBuildResultGpuBuffers {
    let start = Instant::now();
    let mut queue = VecDeque::<TraversalNode>::new();
    let root = &svo.root;
    queue.push_back(TraversalNode { node_type: TraversalNodeType::TopLevel { node: root }, height: SVO_DEPTH, parent_base_child_index: None, self_packed_child_offset: 0 });

    let mut bitmask_vec = Vec::<u64>::new();
    let mut index_vec = Vec::<u32>::new();
    let mut aabb_vec = Vec::<u64>::new();
    
    // metrics stuff
    let mut total_tight_to_coarse_volume_ratio = 0f64;
    let mut total_num_bits_set_total = 0u128;
    let mut num_bits_set_percentage_histogram = [0u128; 4]; // 25%, 50%, 75%, 100%
    let mut total_num_cuboid_nodes = 0u128;
    let mut total_num_full_nodes = 0u128;
    let mut total_num_full_bitmask_nodes = 0u128;

    let mut nodes_visited = 0;
    let mut test_count = 0u32;
    let mut base_indices_for_height = vec![0u32; SVO_DEPTH as usize + 1];

    // first pass that will create the index vec and bitmask vec
    while let Some(TraversalNode { node_type, height, parent_base_child_index, self_packed_child_offset  }) = queue.pop_front() {
        let self_index = index_vec.len();
        base_indices_for_height[height as usize] += 1;
        
        // VERIFY: makes sure that the packed child index matches up
        if let Some(parent_base_child_index) = parent_base_child_index {
            debug_assert_eq!(self_index, parent_base_child_index + self_packed_child_offset);
        }

        // creates a 64 bit mask that contains which children are enabled
        let bitmask = node_type.get_children_bitmask();

        // index of the FIRST child of this node
        // top bits also store some flag information ig
        // entire thing can be set to FULL_NODE to specify that it doesn't have any children but that it IS full
        let mut base_child_index = test_count + 1;

        // metrics stuff...
        let num_bits_set = bitmask.count_ones();
        let num_bits_set_percentage = num_bits_set as f32 / 64f32;
        total_num_bits_set_total += num_bits_set as u128;
        num_bits_set_percentage_histogram[(num_bits_set_percentage * 3f32).ceil() as usize] += 1;
        
        let coarse_volume = 2u32.pow(SVO_DEPTH * 2) as f64;
        let size = node_type.bounds().max - node_type.bounds().min;
        let tight_volume = size.x * size.y * size.z;
        total_tight_to_coarse_volume_ratio += tight_volume as f64 / coarse_volume;

        if bitmask == u64::MAX {
            total_num_full_bitmask_nodes += 1;
        }
        
        // we can avoid doing DDA for the bottom-most nodes if we KNOW that they represents cuboids
        // since we already do a ray-AABB test to check for collision, we can use the result of that to get the hit output stuff and avoid doing DDA completely
        let represents_cuboid = bitmask.count_ones() == tight_volume && height == 1; // if number of set voxels is equal to AABB volume, then AABB volume is tight around a cuboid (this only holds correctly for the leaf nodes)
        if represents_cuboid {
            total_num_cuboid_nodes += 1;
        }

        // check if we are handling the base case
        if height == 0 {
            // should never reach this point, because the nodes at height 0 are handled by their parent anyways
            unreachable!();
        } else {
            if node_type.is_full() {
                // node is full, discard children nodes, and store a specialized magic value that indicates that this node is full
                base_child_index = FULL_NODE;
                total_num_full_nodes += 1;
            } else if height > 1 {
                // if bitmask is full, then there's no need to do the bitmask buffer fetch in the shader
                
                // commented it out as we have removed that conditional fetching logic from shader anyways :p
                /*
                if bitmask == u64::MAX {
                    base_child_index |= 1 << 30;
                }
                */

                // node is not full, we must compute bitmask of children and stuff
                match node_type {
                    TraversalNodeType::TopLevel { node } => {
                        if let Some(children) = node.children.as_ref() {
                            match children {
                                TopLevelAccelerationStructureNodeChildren::RecursiveNodeChildren { children } => {
                                    for (pci, (ci, child)) in children.iter().enumerate().filter_map(|(ci, x)| x.as_ref().map(|x| (ci, x))).enumerate() {
                                        // `self_packed_child_offset = pci` assumes that we are doing this in BFS order. with DFS order, that is no longer the case
                                        queue.push_back(TraversalNode { node_type: TraversalNodeType::TopLevel { node: &child }, height: height - 1, parent_base_child_index: Some(base_child_index as usize), self_packed_child_offset: pci });
                                        test_count += 1;

                                        // VERIFY: makes sure that the packed child index matches up
                                        let mask = (1u64 << ci) - 1;
                                        let masked = bitmask & mask;
                                        let test = masked.count_ones();
                                        debug_assert_eq!(test, pci as u32);
                                    }
                                },
                                TopLevelAccelerationStructureNodeChildren::ChunkNodeChildren { children } => {
                                    for (pci, (ci, chunk_flat_array)) in children.iter().enumerate().filter_map(|(ci, x)| x.as_ref().map(|x| (ci, x))).enumerate() {
                                        // `self_packed_child_offset = pci` assumes that we are doing this in BFS order. with DFS order, that is no longer the case
                                        queue.push_back(TraversalNode { node_type: TraversalNodeType::ChunkLevel { chunk_flat_array, node: &chunk_flat_array[0] }, height: height - 1, parent_base_child_index: Some(base_child_index as usize), self_packed_child_offset: pci });
                                        test_count += 1;
                                    
                                        // VERIFY: makes sure that the packed child index matches up
                                        let mask = (1u64 << ci) - 1;
                                        let masked = bitmask & mask;
                                        let test = masked.count_ones();
                                        debug_assert_eq!(test, pci as u32);
                                    }
                                },
                            }
                            
                        }
                    },
                    TraversalNodeType::ChunkLevel { chunk_flat_array, node } => {
                        if let Some(children) = node.children.as_ref() {
                            for (pci, (ci, child)) in children.iter().enumerate().filter_map(|(ci, x)| x.as_ref().map(|x| (ci, x))).enumerate() {
                                // `self_packed_child_offset = pci` assumes that we are doing this in BFS order. with DFS order, that is no longer the case
                                queue.push_back(TraversalNode { node_type: TraversalNodeType::ChunkLevel { chunk_flat_array, node: &chunk_flat_array[*child] }, height: height - 1, parent_base_child_index: Some(base_child_index as usize), self_packed_child_offset: pci });
                                test_count += 1;
                            
                                // VERIFY: makes sure that the packed child index matches up
                                let mask = (1u64 << ci) - 1;
                                let masked = bitmask & mask;
                                let test = masked.count_ones();
                                debug_assert_eq!(test, pci as u32);
                            }
                        }
                    },
                }
                
            }
        }

        bitmask_vec.push(bitmask);
        index_vec.push(base_child_index);
        aabb_vec.push(pack_aabb_bounds(node_type.bounds(), false));
        nodes_visited += 1;
    }

    const CALCULATE_SAH: bool = false;

    log::debug!("base indices for height:");
    for (height, base_index) in base_indices_for_height.iter().enumerate().rev() {
        log::debug!(" - height {height}: {base_index}");
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

    let sah_normalized_total_nodes = (sah_total as f32 / nodes_visited as f32) * 100f32;

    log::debug!("converted svo, nodes visited: {nodes_visited}, length: {}", bitmask_vec.len());
    log::debug!("metrics:");
    if CALCULATE_SAH {
        log::debug!(" - sah total: {sah_total}");
        log::debug!(" - sah normalized: {sah_normalized_total_nodes:.2}%");
    }
    
    let end = Instant::now();
    log::debug!(" - time taken: {}ms", (end-start).as_millis());
    log::debug!(" - num bits set per node: {}bits/node on avg", total_num_bits_set_total as f64 / nodes_visited as f64);
    log::debug!(" - tight volume / coarse volume: {}", total_tight_to_coarse_volume_ratio / nodes_visited as f64);
    log::debug!(" - number of classified full nodes: {}", total_num_full_nodes);
    log::debug!(" - number of cuboids nodes: {}", total_num_cuboid_nodes);
    log::debug!(" - number of nodes with full bitmask: {}", total_num_full_bitmask_nodes);
    log::debug!(" - bitmask fullness histogram: ");
    
    for (i, val) in num_bits_set_percentage_histogram.iter().enumerate() {
        log::debug!("   - fullness in range ({}%  -  {}%): {}", i*25, (i+1)*25, val);    
    }

    /*
    for k in aabb_vec.iter().take(32) {
        log::debug!("min:{}, max:{}", k.min, k.max);
    }
    */

    SparseVoxelTreeBuildResultGpuBuffers {
        indices: index_vec,
        bitmasks: bitmask_vec,
        aabbs: aabb_vec,
    }
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
