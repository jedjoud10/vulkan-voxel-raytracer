use std::{cell::RefCell, collections::VecDeque, ffi::{CStr, CString}, num::NonZeroU32, rc::{Rc, Weak}, str::FromStr, time::Instant};
use crate::utils::*;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use crate::{buffer::{self, Buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};

mod recursive;
mod sparse_tree;
mod util;

use recursive::*;
use sparse_tree::*;
use util::*;

pub use sparse_tree::SparseVoxelOctree;

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
    log::debug!("creating SVO...");
    let max_svo_element_size = 4096 * 64 * 64;
    let bitmask_buffer = buffer::create_buffer(&device, &mut allocator, max_svo_element_size * size_of::<u64>(), &binder, "sparse voxel octree brick bitmasks", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
    let index_buffer = buffer::create_buffer(&device, &mut allocator, max_svo_element_size * size_of::<u32>(), &binder, "sparse voxel octree child indices", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
    let mut svo = SparseVoxelOctree { bitmask_buffer, index_buffer, nodes: convert_recursive_to_flat_map(create_recursive_structure()) };
    
    /*
    for i in 0..50000 {
        let x = pseudo_random(i) % TOTAL_SIZE;
        let y = pseudo_random(i + x * 3231) % 32 + 256;
        let z = pseudo_random(i + y * 1212) % TOTAL_SIZE;
        
        svo.set(vek::Vec3::new(x,y,z), true);
    }
    */

    svo.rebuild(device, pool, queue, allocator);
    svo    
}