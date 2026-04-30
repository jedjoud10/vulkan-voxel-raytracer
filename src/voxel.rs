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

    let chunks = convert_to_sparse_image_chunks(&svo.nodes);
    let svt = create_sparse_voxel_texture(&device, &mut allocator, binder, queue, pool, queue_family_index, chunks);
    log::info!("created sparse voxel texture");

    (svo, svt)
}

pub struct SparseVoxelTexture {
    pub sparse_image: vk::Image,
    pub sparse_image_view: vk::ImageView,
    pub sparse_image_allocations: Vec<Allocation>,

    pub metadata_image: vk::Image,
    pub metadata_image_view: vk::ImageView,
    pub metadata_image_allocation: Allocation,
}

impl SparseVoxelTexture {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.sparse_image_view, None);
        device.destroy_image(self.sparse_image, None);
        
        for allocation in self.sparse_image_allocations {
            allocator.free(allocation).unwrap();
        }

        device.destroy_image_view(self.metadata_image_view, None);
        device.destroy_image(self.metadata_image, None);
        allocator.free(self.metadata_image_allocation).unwrap();
    }
}

unsafe fn create_sparse_voxel_texture(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    queue_family_index: u32,
    mut chunks: Vec<SparseImageChunk>,
) -> SparseVoxelTexture {
    let queue_family_indices = [queue_family_index];

    // create sparse image that will span entire volume
    let sparse_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: TOTAL_SIZE,
            height: TOTAL_SIZE,
            depth: TOTAL_SIZE,
        })
        .format(vk::Format::R8_UINT)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::GENERAL)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::SPARSE_RESIDENCY | vk::ImageCreateFlags::SPARSE_BINDING)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let sparse_image = device.create_image(&sparse_image_create_info, None).unwrap();
    let sparse_requirements = device.get_image_sparse_memory_requirements(sparse_image);

    // make sure there are sparse memory requirements
    log::debug!("{sparse_requirements:?}");
    if sparse_requirements.is_empty() {
        log::error!("no sparse memory requirements available");
        panic!();
    }

    // pick one the of the requirements    
    let requirements = device.get_image_memory_requirements(sparse_image);
    log::debug!("{requirements:?}");
    let picked_sparse_requirement = sparse_requirements.iter().filter(|req| {
        let granularity = req.format_properties.image_granularity;
        granularity.depth <= 64 && granularity.height <= 64 && granularity.width <= 64 && granularity.depth.is_power_of_two() && granularity.height.is_power_of_two() && granularity.width.is_power_of_two()
    }).next();
    let sparse_requirement = if let Some(x) = picked_sparse_requirement {
        log::debug!("picked sparse requirement: {x:?}");
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


    // create metadata image that will contain metadata info about sparse binding chunks
    // this one is NOT sparse
    let metadata_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: axis_num_bind_chunks,
            height: axis_num_bind_chunks,
            depth: axis_num_bind_chunks,
        })
        .format(vk::Format::R8_UINT)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::GENERAL)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let metadata_image = device.create_image(&sparse_image_create_info, None).unwrap();
    let metadata_image_requirements = device.get_image_memory_requirements(metadata_image);
    
    // stores the origin of the chunk and offset in the `total_data_to_copy` buffer
    let mut binding_chunks = Vec::<(vek::Vec3<u32>, usize)>::new();
    let mut sparse_image_allocations = Vec::default();
    let mut total_data_to_copy = Vec::<u8>::new();
    let mut metadata_image_bytes = vec![0u8; (axis_num_bind_chunks*axis_num_bind_chunks*axis_num_bind_chunks) as usize];
    let mut offset_in_staging_buffer = 0;

    // create backing allocations for the binding chunks  
    for SparseImageChunk { origin, data, full } in chunks.iter() {
        if *full {
            // just set metadata, don't allocate anything
            let metadata_element_index = util::offset_to_index((origin / axis_granularity).as_::<usize>(), axis_num_bind_chunks as usize);
            metadata_image_bytes[metadata_element_index] = 255;
        } else {
            let (x,y,z) = origin.into_tuple();
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
            sparse_image_allocations.push(allocation);

            total_data_to_copy.extend_from_slice(data);
            binding_chunks.push((*origin, offset_in_staging_buffer));
            offset_in_staging_buffer += data.len();
        }
        
    }

    
    // create allocation for metadata image and upload stuff to it
    let metadata_image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: &format!("wtf to name this"),
            requirements: metadata_image_requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedImage(metadata_image),
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();


    // create staging buffer to copy metadata  texel data 
    let (metadata_staging_buffer, metadata_staging_buffer_alloc) = buffer::create_staging_buffer(device, allocator, &metadata_image_bytes);

    // create staging buffer to copy sparse texel data 
    let (sparse_staging_buffer, sparse_staging_buffer_alloc) = buffer::create_staging_buffer(device, allocator, &total_data_to_copy);

    // create the actual memory copies depending on binding chunks
    let mut buffer_image_copies = Vec::<vk::BufferImageCopy>::new();
    
    let image_subresource_layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
        .mip_level(0)
        .base_array_layer(0);
    for (origin, offset) in binding_chunks {
        buffer_image_copies.push(vk::BufferImageCopy::default()
            .buffer_offset(offset as u64)
            .buffer_image_height(0)
            .buffer_row_length(0)
            .image_extent(vk::Extent3D::default().depth(axis_granularity).width(axis_granularity).height(axis_granularity))
            .image_offset(vk::Offset3D::default().x(origin.x as i32).y(origin.y as i32).z(origin.z as i32))
            .image_subresource(image_subresource_layers)
        );
    }

    // bind sparse chunks to image
    let sparse_image_memory_bind_info = [vk::SparseImageMemoryBindInfo::default()
        .binds(&sparse_image_memory_binds)
        .image(sparse_image)];
    let binds = [vk::BindSparseInfo::default().image_binds(&sparse_image_memory_bind_info)];
    device.queue_bind_sparse(queue, &binds, vk::Fence::null()).unwrap();



    // create command buffer
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    // do the sparse staging buffer to sparse image copy
    device.cmd_copy_buffer_to_image(cmd, sparse_staging_buffer, sparse_image, vk::ImageLayout::GENERAL, &buffer_image_copies);

    // do the metadata staging buffer to metadata image copy
    device.cmd_copy_buffer_to_image(cmd, metadata_staging_buffer, metadata_image, vk::ImageLayout::GENERAL, &[vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_image_height(0)
        .buffer_row_length(0)
        .image_extent(vk::Extent3D::default().depth(axis_num_bind_chunks).width(axis_num_bind_chunks).height(axis_num_bind_chunks))
        .image_offset(vk::Offset3D::default())
        .image_subresource(image_subresource_layers)
    ]);

    // end command buffer and submit
    device.end_command_buffer(cmd).unwrap();
    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);
    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();

    // free staging buffers
    allocator.free(sparse_staging_buffer_alloc).unwrap();
    device.destroy_buffer(sparse_staging_buffer, None);
    allocator.free(metadata_staging_buffer_alloc).unwrap();
    device.destroy_buffer(metadata_staging_buffer, None);

    if let Some(binder) = binder {
        let marker = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(sparse_image)
            .object_name(c"Sparse Voxel Texture");
        binder.set_debug_utils_object_name(&marker).unwrap();
    }

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let sparse_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(vk::Format::R8_UINT)
        .image(sparse_image)
        .subresource_range(subresource_range)
        .view_type(vk::ImageViewType::TYPE_3D);
    let sparse_image_view = device
        .create_image_view(&sparse_image_view_create_info, None)
        .unwrap();

    let metadata_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(vk::Format::R8_UINT)
        .image(metadata_image)
        .subresource_range(subresource_range)
        .view_type(vk::ImageViewType::TYPE_3D);
    let metadata_image_view = device
        .create_image_view(&metadata_image_view_create_info, None)
        .unwrap();

    SparseVoxelTexture {
        sparse_image,
        sparse_image_view,
        sparse_image_allocations,
        metadata_image,
        metadata_image_view,
        metadata_image_allocation,
    }
}