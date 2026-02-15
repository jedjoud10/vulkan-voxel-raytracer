use std::{collections::VecDeque, ffi::{CStr, CString}, str::FromStr};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::{buffer::{self, Buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};

pub const MIP_LEVELS: usize = 9;
pub const SIZE: u32 = 1 << ((MIP_LEVELS-1) as u32);
pub const _SIZE: usize = SIZE as usize;

pub unsafe fn create_voxel_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    binder: &Option<ash::ext::debug_utils::Device>,
    name: &CStr,
) -> (vk::Image, Allocation, vk::ImageView) {
    let voxel_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: SIZE,
            height: SIZE,
            depth: SIZE,
        })
        .format(format)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(usage)
        .samples(vk::SampleCountFlags::TYPE_1)
        .array_layers(1);
    let voxel_image = device.create_image(&voxel_image_create_info, None).unwrap();
    let requirements = device.get_image_memory_requirements(voxel_image);

    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: &name.to_string_lossy(),
            requirements: requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedImage(voxel_image),
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let device_memory = allocation.memory();

    device
        .bind_image_memory(voxel_image, device_memory, 0)
        .unwrap();

    let subresource_range = vk::ImageSubresourceRange::default()
        .base_mip_level(0)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);
    let voxel_image_view_create_info = vk::ImageViewCreateInfo::default()
        .image(voxel_image)
        .format(format)
        .view_type(vk::ImageViewType::TYPE_3D)
        .subresource_range(subresource_range);
    let voxel_image_view = device
        .create_image_view(&voxel_image_view_create_info, None)
        .unwrap();

    if let Some(binder) = binder {
        let marker = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(voxel_image)
            .object_name(name);
        binder.set_debug_utils_object_name(&marker).unwrap();
    }
    
    (voxel_image, allocation, voxel_image_view)
}

pub struct VoxelImage {
    pub image: vk::Image,
    pub allocation: Allocation,
    pub per_mip_views: [vk::ImageView; MIP_LEVELS as usize],
    pub image_view_whole: vk::ImageView,
}

impl VoxelImage {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        for x in self.per_mip_views {
            device.destroy_image_view(x, None);    
        }
        
        device.destroy_image_view(self.image_view_whole, None);
        device.destroy_image(self.image, None);
        allocator.free(self.allocation).unwrap();
    }
}

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

pub unsafe fn create_sparse_voxel_octree(
    device: &ash::Device,
    mut allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    name: &str,
) -> SparseVoxelOctree {
    // TODO: does this make sense? no... but... it works...
    let max_svo_element_size = 4096 * 64 * 64;

    let bitmask_buffer = buffer::create_buffer(&device, &mut allocator, max_svo_element_size * size_of::<u64>(), &binder, "sparse voxel octree brick bitmasks");
    
    
    let index_buffer = buffer::create_buffer(&device, &mut allocator, max_svo_element_size * size_of::<u32>(), &binder, "sparse voxel octree child indices");

    SparseVoxelOctree { bitmask_buffer, index_buffer }
}

pub unsafe fn create_voxel_octree_mip_map_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    binder: &Option<ash::ext::debug_utils::Device>,
    name: &str,
) -> VoxelImage {
    let voxel_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: SIZE,
            height: SIZE,
            depth: SIZE,
        })
        .format(format)
        .image_type(vk::ImageType::TYPE_3D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(MIP_LEVELS as u32)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(usage)
        .samples(vk::SampleCountFlags::TYPE_1)
        .array_layers(1);
    let voxel_image = device.create_image(&voxel_image_create_info, None).unwrap();
    let requirements = device.get_image_memory_requirements(voxel_image);

    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: &name,
            requirements: requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedImage(voxel_image),
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let device_memory = allocation.memory();

    device
        .bind_image_memory(voxel_image, device_memory, 0)
        .unwrap();


    let mut per_mip_image_views: [vk::ImageView; MIP_LEVELS] = [Default::default(); MIP_LEVELS];
    for i in 0..MIP_LEVELS {
        let subresource_range = vk::ImageSubresourceRange::default()
            .base_mip_level(i as u32)
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .level_count(1);
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(voxel_image)
            .format(format)
            .view_type(vk::ImageViewType::TYPE_3D)
            .subresource_range(subresource_range);
        let image_view = device
            .create_image_view(&image_view_create_info, None)
            .unwrap();

        
        crate::debug::set_object_name(image_view, binder, format!("{name} image view mip[{i}]"));
        per_mip_image_views[i] = image_view;
    }

    crate::debug::set_object_name(voxel_image, binder, name);

    let subresource_range = vk::ImageSubresourceRange::default()
        .base_mip_level(0)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(MIP_LEVELS as u32);
    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .image(voxel_image)
        .format(format)
        .view_type(vk::ImageViewType::TYPE_3D)
        .subresource_range(subresource_range);
    let whole_image_view = device
        .create_image_view(&image_view_create_info, None)
        .unwrap();

        
    crate::debug::set_object_name(whole_image_view, binder, format!("{name} image view for whole"));
    
    VoxelImage {
        image: voxel_image,
        allocation,
        per_mip_views: per_mip_image_views,
        image_view_whole: whole_image_view,
    }
}

pub unsafe fn generate_voxel_image(
    device: &ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    queue_family_index: u32,
    voxel_image_wrapper: &VoxelImage,
    voxel_indices_image: vk::Image,
    voxel_indices_image_view: vk::ImageView,
    voxel_generate_pipeline: &VoxelGeneratePipeline,
) {
    log::info!("generating voxel image...");
    let VoxelImage { image: voxel_image, allocation: _, per_mip_views: voxel_mips, image_view_whole: voxel_image_view } = *voxel_image_wrapper;
    
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];

    let cmd_buffer_begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device
        .begin_command_buffer(cmd, &cmd_buffer_begin_info)
        .unwrap();

    let all_mips_subresource_range = vk::ImageSubresourceRange::default()
        .base_mip_level(0)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(MIP_LEVELS as u32);

    let single_mip_subresource_range = vk::ImageSubresourceRange::default()
        .base_mip_level(0)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let first_transition_voxel_image = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_image)
        .subresource_range(all_mips_subresource_range);
    let first_transition_voxel_indices = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_indices_image)
        .subresource_range(single_mip_subresource_range);
    let image_memory_barriers = [first_transition_voxel_image, first_transition_voxel_indices];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    // we will need multiple descriptor sets
    // the first one will be for the initial voxel generation
    // the rest are needed for the propagation of voxel values upwards in the mip-chain. each descriptor set will have the "previous-image" binding and "current-image" bindings
    let layouts = Vec::from_iter(std::iter::repeat_n(voxel_generate_pipeline.descriptor_set_layout, MIP_LEVELS));

    // updates the descriptors once and for all
    // this way, subsequent dispatch calls can use the updated descriptors directly
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_sets = device
        .allocate_descriptor_sets(&descriptor_set_allocate_info)
        .unwrap();
    {    
        let descriptor_base_layer_voxel_image_info = vk::DescriptorImageInfo::default()
            .image_view(voxel_mips[0])
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_image_infos = [descriptor_base_layer_voxel_image_info];
    
        // the first descriptor that we allocated will be used for the src voxel image that we will write to originally
        let base_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(descriptor_sets[0])
            .image_info(&descriptor_image_infos);
        device.update_descriptor_sets(&[base_descriptor_write], &[]);
        log::info!("updated bottom mip-level descriptor");

        // we must also write the descriptor sets for the propagated voxel images
        for i in 1..MIP_LEVELS {
            let previous_image_view = voxel_mips[i-1];
            let next_image_view = voxel_mips[i];

            let descriptor_previous_voxel_image_info = vk::DescriptorImageInfo::default()
                .image_view(previous_image_view)
                .image_layout(vk::ImageLayout::GENERAL)
                .sampler(vk::Sampler::null());
            let descriptor_next_voxel_image_info = vk::DescriptorImageInfo::default()
                .image_view(next_image_view)
                .image_layout(vk::ImageLayout::GENERAL)
                .sampler(vk::Sampler::null());
            let descriptor_image_infos = [descriptor_previous_voxel_image_info, descriptor_next_voxel_image_info];

            let propagate_descriptor_write = vk::WriteDescriptorSet::default()
                .descriptor_count(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .dst_binding(1)
                .dst_set(descriptor_sets[i])
                .image_info(&descriptor_image_infos);

            device.update_descriptor_sets(&[propagate_descriptor_write], &[]);
            log::info!("updated mip-level {i} descriptor");
        }
    }

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        voxel_generate_pipeline.entry_points[0].pipeline_layout,
        0,
        &descriptor_sets[0..1],
        &[],
    );

    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        voxel_generate_pipeline.entry_points[0].pipeline,
    );

    device.cmd_dispatch(cmd, SIZE / 8, SIZE / 8, SIZE / 8);

    let second_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_image)
        .subresource_range(single_mip_subresource_range);
    let image_memory_barriers = [second_transition];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        voxel_generate_pipeline.entry_points[1].pipeline,
    );

    for i in 1..MIP_LEVELS {
        let previous_image_view = voxel_mips[i-1];
        let next_image_view = voxel_mips[i];

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            voxel_generate_pipeline.entry_points[1].pipeline_layout,
            0,
            &descriptor_sets[(i)..(i+1)],
            &[],
        );
        
        device.cmd_dispatch(cmd, SIZE / 8, SIZE / 8, SIZE / 8);

        let next_voxel_image_subresource_range = vk::ImageSubresourceRange::default()
            .base_mip_level(i as u32)
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .level_count(1);


        let next_voxel_image_memory_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(voxel_image)
            .subresource_range(next_voxel_image_subresource_range);
        let image_memory_barriers: [vk::ImageMemoryBarrier2<'_>; 1] = [next_voxel_image_memory_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);
    }

    device.end_command_buffer(cmd).unwrap();

    let cmds = [cmd];
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&cmds)
        .signal_semaphores(&[])
        .wait_dst_stage_mask(&[])
        .wait_semaphores(&[]);

    let fence = device.create_fence(&Default::default(), None).unwrap();

    device.queue_submit(queue, &[submit_info], fence).unwrap();
    device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
    device.free_command_buffers(pool, &[cmd]);
    device.free_descriptor_sets(descriptor_pool, &descriptor_sets).unwrap();
    device.destroy_fence(fence, None);
}


struct Node {
    children: Option<Box<[Option<Box<Node>>; 64]>>,
    bottom: bool,
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

fn test_sparse_voxel_octree_recurse(base: u32, seed: u32) -> Node {
    if base == 0 {
        return Node {
            children: None,
            bottom: true,
        };
    }

    let mut children = [const { None }; 64];

    for x in  0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let i = x + y * 4 + z * 4 * 4;

                if (y == 0) {
                    children[i] = Some(Box::new(test_sparse_voxel_octree_recurse(base - 1, i as u32)));
                }
            }
        }
    }

    for x in 0..8 {
        let index = pseudo_random(x ^ 0x03f23 ^ base ^ seed) % 64;
        children[index as usize] = Some(Box::new(test_sparse_voxel_octree_recurse(base - 1, index ^ x)));
    }

    Node {
        children: Some(Box::new(children)),
        bottom: base == 1,
    }
}

fn test_sparse_voxel_octree_root() -> Node {
    return test_sparse_voxel_octree_recurse(5, 0x0323f);

    let mut children = [const { None }; 64];

    for x in 0..8 {
        let index = pseudo_random(x ^ 0x03f23) % 64;

        // these nodes will be discarded when we convert to buffers
        children[index as usize] = Some(Box::new(Node {
            children: None,
            bottom: true,
        }));
    }

    Node {
        children: Some(Box::new(children)),
        bottom: true,
    }
}

struct TraversalNode<'a> {
    node: &'a Node,
    depth: usize,
    parent_base_child_index: Option<usize>,
    self_packed_child_offset: usize,
}

// root 0: bitmask: u64, base child index: 0


fn convert_to_buffers(node: Node) -> (Vec<u64>, Vec<u32>) {
    let mut queue = VecDeque::<TraversalNode>::new();
    queue.push_back(TraversalNode { node: &node, depth: 0, parent_base_child_index: None, self_packed_child_offset: 0 });

    let mut bitmask_vec = Vec::<u64>::new();
    let mut index_vec = Vec::<u32>::new();
    let mut nodes_visited = 0;
    let mut test_count = 0u32;

    while let Some(TraversalNode { node, depth, parent_base_child_index: parent_index, self_packed_child_offset  }) = queue.pop_front() {
        /*
        // if we are the first child and have a valid parent reference, update the "base child" index of the parent
        if let Some(parent_node_index) = parent_node_index && first_child {
            index_vec[parent_node_index as usize] = index_vec.len() as u16;
        }
        */

        let self_index = index_vec.len();
        if let Some(parent) = parent_index {
            assert_eq!(self_index, parent + self_packed_child_offset);
        }

        let bitmask = node.children.as_ref().map(|children| children.iter()
            .enumerate()
            .filter_map(|(i, x)| x.as_ref().map(|_| i))
            .fold(0u64, |prev, i| ((1u64 << i) as u64) | prev)
        ).unwrap_or_default();
        
        let mut base_child_index = test_count + 1;

        if node.bottom || bitmask == 0 {
            base_child_index = u32::MAX;
        } else {
            if let Some(children) = node.children.as_ref() {
                for (pci, (ci, child)) in children.iter().enumerate().filter_map(|(ci, x)| x.as_ref().map(|x| (ci, x))).enumerate() {
                    queue.push_back(TraversalNode { node: child, depth: depth + 1, parent_base_child_index: Some(base_child_index as usize), self_packed_child_offset: pci });
                    //log::debug!("creating child shi depth: {} ci: {}, pci: {}", depth+1, ci, pci);
                    test_count += 1;


                    let mask = (1u64 << ci) - 1;
                    let masked = bitmask & mask;
                    let test = masked.count_ones();
                    assert_eq!(test, pci as u32);
                }
            }
        }

        if (depth == 0) {
            log::debug!("{:b}, depth: {}, base child index: {}", bitmask, depth, base_child_index);
        }

        bitmask_vec.push(bitmask);
        index_vec.push(base_child_index);





        nodes_visited += 1;
    }

    log::debug!("converted svo, nodes visited: {nodes_visited}, length: {}", bitmask_vec.len());

    (bitmask_vec, index_vec)
}

// each node contains a u64 bitmask that checks if any of its children are leaf nodes
// another buffer stores the "children base" index references as u16s
// it contains a bitmask of its children

pub unsafe fn convert_mips_svo(
    device: &ash::Device,
    allocator: &mut Allocator,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    queue_family_index: u32,
    voxel_image: &VoxelImage,
    svo: &SparseVoxelOctree,
) {
    log::debug!("creating svo nodes...");
    let test = test_sparse_voxel_octree_root();

    let (bitset_vec, index_vec) = convert_to_buffers(test);

    let bitmask_data = bitset_vec.as_slice();
    let bitmask_data_bytes: &[u8] = bytemuck::cast_slice(bitmask_data); 

    buffer::write_to_buffer(device, pool, queue, svo.bitmask_buffer.buffer, allocator, bitmask_data_bytes);

    // init: fill the index buffer with invalid indices (0xFFFF)
    buffer::fill_buffer(&device, pool, queue, svo.index_buffer.buffer, u32::MAX);

    let index_data = index_vec.as_slice();
    let index_data_bytes: &[u8] = bytemuck::cast_slice(&index_data); 
    buffer::write_to_buffer(device, pool, queue, svo.index_buffer.buffer, allocator, index_data_bytes);
}

pub unsafe fn execute_voxel_tick_compute(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    queue_family_index: u32,
    surface_buffer: vk::Buffer,
    counter_buffer: vk::Buffer,
    visible_surface_buffer: vk::Buffer,
    visible_surface_counter_buffer: vk::Buffer,
    indirect_dispatch_buffer: vk::Buffer,
    voxel_image: &VoxelImage,
    voxel_indices_image: vk::Image,
    voxel_indices_image_view: vk::ImageView,
    tick_voxel_compute_pipeline: &VoxelTickPipeline,
    push_constants: PushConstants2,
) -> vk::DescriptorSet {

    let subresource_range = vk::ImageSubresourceRange::default()
        .base_mip_level(0)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let voxel_image_read_to_write = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_image.image)
        .subresource_range(subresource_range);
    let voxel_indices_image_read_to_write = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_indices_image)
        .subresource_range(subresource_range);
    let voxel_surface_buffer_read_to_write = vk::BufferMemoryBarrier2::default()
        .buffer(surface_buffer)
        .size(u64::MAX)
        .offset(0)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ);
    let voxel_counter_buffer_read_to_write = vk::BufferMemoryBarrier2::default()
        .buffer(counter_buffer)
        .size(u64::MAX)
        .offset(0)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE);
    let voxel_visible_surfaces_buffer_read_to_write = vk::BufferMemoryBarrier2::default()
        .buffer(visible_surface_buffer)
        .size(u64::MAX)
        .offset(0)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ);
    let voxel_visible_surfaces_counter_buffer_read_to_write = vk::BufferMemoryBarrier2::default()
        .buffer(visible_surface_counter_buffer)
        .size(u64::MAX)
        .offset(0)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE);

    let image_memory_barriers = [voxel_image_read_to_write, voxel_indices_image_read_to_write];
    let buffer_memory_barriers = [voxel_surface_buffer_read_to_write, voxel_counter_buffer_read_to_write, voxel_visible_surfaces_buffer_read_to_write, voxel_visible_surfaces_counter_buffer_read_to_write];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers).buffer_memory_barriers(&buffer_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let layouts = [tick_voxel_compute_pipeline.descriptor_set_layout];
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_sets = device
        .allocate_descriptor_sets(&descriptor_set_allocate_info)
        .unwrap();
    let descriptor_set = descriptor_sets[0];

    let descriptor_voxel_image_view_info = vk::DescriptorImageInfo::default()
        .image_view(voxel_image.image_view_whole)
        .image_layout(vk::ImageLayout::GENERAL)
        .sampler(vk::Sampler::null());
    let descriptor_voxel_surface_index_image_view_info = vk::DescriptorImageInfo::default()
        .image_view(voxel_indices_image_view)
        .image_layout(vk::ImageLayout::GENERAL)
        .sampler(vk::Sampler::null());

    let descriptor_voxel_surface_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(surface_buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_voxel_surface_counter_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(counter_buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_visible_surface_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(visible_surface_buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_visible_surface_counter_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(visible_surface_counter_buffer)
        .offset(0)
        .range(u64::MAX);
    let indirect_dispatch_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(indirect_dispatch_buffer)
        .offset(0)
        .range(u64::MAX);

    let descriptor_image_infos = [descriptor_voxel_image_view_info, descriptor_voxel_surface_index_image_view_info];
    let descriptor_buffer_infos = [descriptor_voxel_surface_buffer_info, descriptor_voxel_surface_counter_buffer_info, descriptor_visible_surface_buffer_info, descriptor_visible_surface_counter_buffer_info, indirect_dispatch_buffer_info];

    let image_descriptor_write = vk::WriteDescriptorSet::default()
        .descriptor_count(descriptor_image_infos.len() as u32)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .dst_binding(0)
        .dst_set(descriptor_set)
        .image_info(&descriptor_image_infos);

    let buffer_descriptor_write = vk::WriteDescriptorSet::default()
        .descriptor_count(descriptor_buffer_infos.len() as u32)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .dst_binding(2)
        .dst_set(descriptor_set)
        .buffer_info(&descriptor_buffer_infos);

    device
        .update_descriptor_sets(&[image_descriptor_write, buffer_descriptor_write], &[]);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        tick_voxel_compute_pipeline.entry_points[1].pipeline_layout,
        0,
        &descriptor_sets,
        &[],
    );

    let data = [0u32];
    let raw = bytemuck::cast_slice::<u32, u8>(&data);
    device.cmd_update_buffer(cmd, counter_buffer, 0, raw);
    device.cmd_update_buffer(cmd, visible_surface_counter_buffer, 0, raw);

    let data = [0u32, 0u32, 0u32];
    let raw = bytemuck::cast_slice::<u32, u8>(&data);
    device.cmd_update_buffer(cmd, indirect_dispatch_buffer, 0, raw);

    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        tick_voxel_compute_pipeline.entry_points[0].pipeline,
    );

    let barrier = vk::MemoryBarrier2::default()
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
    let barriers = [barrier];
    let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let raw = bytemuck::bytes_of(&push_constants);
    device.cmd_push_constants(cmd, tick_voxel_compute_pipeline.entry_points[0].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, raw);
    
    device.cmd_dispatch(cmd, SIZE / 8, SIZE / 8, SIZE / 8);




    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        tick_voxel_compute_pipeline.entry_points[1].pipeline,
    );

    let barrier = vk::MemoryBarrier2::default()
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let barriers = [barrier];
    let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let raw = bytemuck::bytes_of(&push_constants);
    device.cmd_push_constants(cmd, tick_voxel_compute_pipeline.entry_points[1].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, raw);
    
    device.cmd_dispatch(cmd, SIZE / 8, SIZE / 8, SIZE / 8);

    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        tick_voxel_compute_pipeline.entry_points[3].pipeline,
    );

    let barrier = vk::MemoryBarrier2::default()
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let barriers = [barrier];
    let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let raw = bytemuck::bytes_of(&push_constants);
    device.cmd_push_constants(cmd, tick_voxel_compute_pipeline.entry_points[3].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, raw);
    
    device.cmd_dispatch(cmd, 1, 1, 1);






    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        tick_voxel_compute_pipeline.entry_points[2].pipeline,
    );

    let barrier = vk::MemoryBarrier2::default()
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
    let barriers = [barrier];
    let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let raw = bytemuck::bytes_of(&push_constants);
    device.cmd_push_constants(cmd, tick_voxel_compute_pipeline.entry_points[2].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, raw);
    
    device.cmd_dispatch_indirect(cmd, indirect_dispatch_buffer, 0);





    let voxel_image_write_to_read = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::TRANSFER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_image.image)
        .subresource_range(subresource_range);
    let voxel_indices_image_write_to_read = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_indices_image)
        .subresource_range(subresource_range);
    let voxel_surface_buffer_write_to_read = vk::BufferMemoryBarrier2::default()
        .buffer(surface_buffer)
        .size(u64::MAX)
        .offset(0)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ);
    let voxel_counter_buffer_write_to_read = vk::BufferMemoryBarrier2::default()
        .buffer(counter_buffer)
        .size(u64::MAX)
        .offset(0)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ);
    let image_memory_barriers = [voxel_image_write_to_read, voxel_indices_image_write_to_read];
    let buffer_memory_barriers = [voxel_surface_buffer_write_to_read, voxel_counter_buffer_write_to_read];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers).buffer_memory_barriers(&buffer_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);
    return descriptor_set;
}

pub unsafe fn update_voxel(
    device: &ash::Device,
    allocator: &mut Allocator,
    queue: vk::Queue,
    pool: vk::CommandPool,
    voxel_image: &VoxelImage,
    toggle: bool,
    position: vek::Vec3<u32>,
) {
    let src_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(size_of::<u64>() as u64);
    let src = device.create_buffer(&src_create_info, None).unwrap();

    let requirements = device.get_buffer_memory_requirements(src);
    let mut allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Staging Buffer...",
            requirements: requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(src),
            location: gpu_allocator::MemoryLocation::CpuToGpu,
        })
        .unwrap();
    let device_memory = allocation.memory();
    device.bind_buffer_memory(src, device_memory, 0).unwrap();
    let raw = allocation.mapped_slice_mut().unwrap();
    raw[0] = 0b01010101;
    raw[2] = 0b01010101;
    raw[4] = 0b01010101;
    raw[6] = 0b01010101;


    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let subresource_layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(0);

    let region = vk::BufferImageCopy2::default()
        .buffer_offset(0)
        .buffer_image_height(0)
        .buffer_row_length(0)
        .image_offset(vk::Offset3D {
            x: position.x as i32,
            y: position.y as i32,
            z: position.z as i32,
        })
        .image_extent(vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        }).image_subresource(subresource_layers);

    let regions = [region];
    let copy_buffer_to_image_info = vk::CopyBufferToImageInfo2::default()
        .dst_image(voxel_image.image)
        .dst_image_layout(vk::ImageLayout::GENERAL)
        .regions(&regions)
        .src_buffer(src);

    device.cmd_copy_buffer_to_image2(cmd, &copy_buffer_to_image_info);
    
    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);

    device.end_command_buffer(cmd).unwrap();

    let fence = device.create_fence(&Default::default(), None).unwrap();
    device.queue_submit(queue, & [submit], fence).unwrap();
    device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
    device.destroy_fence(fence, None);
    allocator.free(allocation).unwrap();
    device.destroy_buffer(src, None);
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