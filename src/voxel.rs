use std::{cell::RefCell, collections::VecDeque, ffi::{CStr, CString}, num::NonZeroU32, rc::{Rc, Weak}, str::FromStr, time::Instant};
use crate::utils::*;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use crate::{buffer::{self, Buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};

mod recursive;
mod sparse;
mod util;

use recursive::*;
use sparse::*;
use util::*;

pub use sparse::SparseVoxelOctree;

pub unsafe fn create_sparse_structures(
    device: &ash::Device,
    mut allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    queue_family_index: u32,
) -> (SparseVoxelOctree, SparseVoxelTexture) {
    // each node contains a u64 bitmask that checks if any of its children are leaf nodes
    // another buffer stores the "children base" index references as u16s
    // it contains a bitmask of its children
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
    log::info!("created & updated sparse voxel tree buffers");

    let chunks = convert_to_sparse_image_chunks(0, &svo.nodes);
    let svt = create_sparse_voxel_texture(&device, &mut allocator, binder, queue, pool, queue_family_index, chunks);
    log::info!("created sparse voxel texture");

    (svo, svt)
}

pub struct SparseVoxelTexture {
    pub image: vk::Image,
    pub allocations: Vec<Allocation>,
}

impl SparseVoxelTexture {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image(self.image, None);
        
        for allocation in self.allocations {
            allocator.free(allocation).unwrap();
        }
    }
}

unsafe fn create_sparse_voxel_texture(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    queue_family_index: u32,
    chunks: Vec<SparseImageChunk>,
) -> SparseVoxelTexture {
    let queue_family_indices = [queue_family_index];
    let image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: TOTAL_SIZE,
            height: TOTAL_SIZE,
            depth: TOTAL_SIZE,
        })
        .format(vk::Format::R8_UNORM)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::SPARSE_RESIDENCY | vk::ImageCreateFlags::SPARSE_BINDING)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let image = device.create_image(&image_create_info, None).unwrap();
    let sparse_requirements = device.get_image_sparse_memory_requirements(image);

    log::debug!("{sparse_requirements:?}");
    if sparse_requirements.is_empty() {
        log::error!("no sparse memory requirements available");
        panic!();
    }

    let requirements = device.get_image_memory_requirements(image);
    log::debug!("{requirements:?}");

    let picked_sparse_requirement = sparse_requirements.iter().filter(|req| {
        let granularity = req.format_properties.image_granularity;
        granularity.depth <= 64 && granularity.height <= 64 && granularity.width <= 64 && granularity.depth.is_power_of_two() && granularity.height.is_power_of_two() && granularity.width.is_power_of_two()
    }).next();
    log::debug!("pick sparse requirement: {picked_sparse_requirement:?}");

    let sparse_requirement = if let Some(x) = picked_sparse_requirement {
        x
    } else {
        log::error!("could not find a sparse memory requirement with required granularity");
        panic!();
    };
    let extent_granularity = sparse_requirement.format_properties.image_granularity;
    
    // fuck it
    //let axis_granularity = extent_granularity.width.max(extent_granularity.height).max(extent_granularity.depth);
    let axis_granularity = 64;

    let axis_num_bind_chunks = TOTAL_SIZE / axis_granularity;
    let bind_chunk_volume = axis_granularity.pow(3);
    
    let mut sparse_image_memory_binds = Vec::<vk::SparseImageMemoryBind>::new();

    let image_subresource= vk::ImageSubresource::default()
        .mip_level(0)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .array_layer(0);


    let mut allocations = Vec::default();
    /*
    for x in 0..axis_num_bind_chunks {
        for y in 0..axis_num_bind_chunks {
            for z in 0..axis_num_bind_chunks {
                let bind_chunk_requirement = vk::MemoryRequirements::default()
                    .alignment(requirements.alignment)
                    .memory_type_bits(requirements.memory_type_bits)
                    .size(bind_chunk_volume as u64);

                let allocation = allocator
                    .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                        name: &format!("Sparse Voxel Texture Binding Chunk {x}-{y}-{z}"),
                        requirements: bind_chunk_requirement,
                        linear: false,
                        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                        location: gpu_allocator::MemoryLocation::GpuOnly,
                    })
                    .unwrap();
                
                log::debug!("{}", allocation.size());

                if let Some(binder) = binder {
                    let name = format!("Sparse Voxel Texture Memory Binding Chunk {x}-{y}-{z}");
                    let cstring = CString::from_str(&name).unwrap();
                    let marker = vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(allocation.memory())
                        .object_name(cstring.as_c_str());
                    binder.set_debug_utils_object_name(&marker).unwrap();
                }

                let memory_bind = vk::SparseImageMemoryBind::default()
                    .extent(vk::Extent3D::default()
                        .depth(axis_granularity)
                        .width(axis_granularity)
                        .height(axis_granularity))
                    .memory(allocation.memory())
                    .memory_offset(allocation.offset())
                    .subresource(image_subresource);

                sparse_image_memory_binds.push(memory_bind);
                allocations.push(allocation);
            }
        }   
    }
    */

    for SparseImageChunk { origin, data } in chunks {
        assert!(data.len() == 64*64*64);
        log::debug!("{origin:?}");

        let (x,y,z) = origin.into_tuple();
        let bind_chunk_requirement = vk::MemoryRequirements::default()
            .alignment(requirements.alignment)
            .memory_type_bits(requirements.memory_type_bits)
            .size(bind_chunk_volume as u64);

        let mut allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: &format!("Sparse Voxel Texture Binding Chunk {x}-{y}-{z}"),
                requirements: bind_chunk_requirement,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
            })
            .unwrap();
        
        log::debug!("{}", allocation.size());
        let slice = allocation.mapped_slice_mut().unwrap();
        slice[..(bind_chunk_volume as usize)].copy_from_slice(&data);

        if let Some(binder) = binder {
            let name = format!("Sparse Voxel Texture Memory Binding Chunk {x}-{y}-{z}");
            let cstring = CString::from_str(&name).unwrap();
            let marker = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(allocation.memory())
                .object_name(cstring.as_c_str());
            binder.set_debug_utils_object_name(&marker).unwrap();
        }

        let memory_bind = vk::SparseImageMemoryBind::default()
            .extent(vk::Extent3D::default()
                .depth(axis_granularity)
                .width(axis_granularity)
                .height(axis_granularity))
            .offset(vk::Offset3D::default().x(x as i32).y(y as i32).z(z as i32))
            .memory(allocation.memory())
            .memory_offset(allocation.offset())
            .subresource(image_subresource);

        sparse_image_memory_binds.push(memory_bind);
        allocations.push(allocation);
    }

    let sparse_image_memory_bind_info = [vk::SparseImageMemoryBindInfo::default()
        .binds(&sparse_image_memory_binds)
        .image(image)];

    let binds = [vk::BindSparseInfo::default().image_binds(&sparse_image_memory_bind_info)];
    device.queue_bind_sparse(queue, &binds, vk::Fence::null()).unwrap();

    if let Some(binder) = binder {
        let marker = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(image)
            .object_name(c"Sparse Voxel Texture");
        binder.set_debug_utils_object_name(&marker).unwrap();
    }

    SparseVoxelTexture {
        image,
        allocations,
    }
}