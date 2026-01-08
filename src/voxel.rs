use std::ffi::CStr;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::pipeline::PushConstants2;

pub const SIZE: u32 = 64;
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

pub unsafe fn create_voxel_surface_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> (vk::Buffer, Allocation) {
    // store semi-worst case scenario?
    const SOME_ARBITRARY_SIZE_FOR_MAX_NUMBER_OF_CUBES_IDK: usize = _SIZE*_SIZE*_SIZE / 64;
    let size = size_of::<vek::Vec4<u8>>() * 6 * 16 * SOME_ARBITRARY_SIZE_FOR_MAX_NUMBER_OF_CUBES_IDK;


    let voxel_buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(size as u64);
    let buffer = device.create_buffer(&voxel_buffer_create_info, None).unwrap();

    let requirements = device.get_buffer_memory_requirements(buffer);

    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Voxel Surface Buffer Allocation",
            requirements: requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(buffer),
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();

    if let Some(binder) = binder {
        let marker = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(buffer)
            .object_name(c"voxel surface buffer");
        binder.set_debug_utils_object_name(&marker).unwrap();
    }

    
    let device_memory = allocation.memory();
    device.bind_buffer_memory(buffer, device_memory, 0).unwrap();
    (buffer, allocation)
}


pub unsafe fn create_voxel_counter_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> (vk::Buffer, Allocation) {
    let voxel_buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(size_of::<u32>() as u64);
    let buffer = device.create_buffer(&voxel_buffer_create_info, None).unwrap();

    let requirements = device.get_buffer_memory_requirements(buffer);

    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Voxel Index Counter Allocation",
            requirements: requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(buffer),
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();

    if let Some(binder) = binder {
        let marker = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(buffer)
            .object_name(c"voxel index counter buffer");
        binder.set_debug_utils_object_name(&marker).unwrap();
    }
    
    let device_memory = allocation.memory();
    device.bind_buffer_memory(buffer, device_memory, 0).unwrap();
    (buffer, allocation)
}

pub unsafe fn generate_voxel_image(
    device: &ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    queue_family_index: u32,
    voxel_image: vk::Image,
    voxel_image_view: vk::ImageView,
    voxel_indices_image: vk::Image,
    voxel_indices_image_view: vk::ImageView,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
) {
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
    let subresource_range = vk::ImageSubresourceRange::default()
        .base_mip_level(0)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let first_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_image)
        .subresource_range(subresource_range);
    let first_transition_amogus = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_indices_image)
        .subresource_range(subresource_range);
    let image_memory_barriers = [first_transition, first_transition_amogus];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let layouts = [descriptor_set_layout];
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_sets = device
        .allocate_descriptor_sets(&descriptor_set_allocate_info)
        .unwrap();
    let descriptor_set = descriptor_sets[0];

    let descriptor_image_info = vk::DescriptorImageInfo::default()
        .image_view(voxel_image_view)
        .image_layout(vk::ImageLayout::GENERAL)
        .sampler(vk::Sampler::null());
    let descriptor_image_infos = [descriptor_image_info];

    let descriptor_write = vk::WriteDescriptorSet::default()
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .dst_binding(0)
        .dst_set(descriptor_set)
        .image_info(&descriptor_image_infos);

    device
        .update_descriptor_sets(&[descriptor_write], &[]);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &descriptor_sets,
        &[],
    );

    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        pipeline,
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
        .subresource_range(subresource_range);
    let image_memory_barriers = [second_transition];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

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


pub unsafe fn update_voxel_thingies(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    queue_family_index: u32,
    surface_buffer: vk::Buffer,
    counter_buffer: vk::Buffer,
    voxel_image: vk::Image,
    voxel_image_view: vk::ImageView,
    voxel_indices_image: vk::Image,
    voxel_indices_image_view: vk::ImageView,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
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
        .image(voxel_image)
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

    let image_memory_barriers = [voxel_image_read_to_write, voxel_indices_image_read_to_write];
    let buffer_memory_barriers = [voxel_surface_buffer_read_to_write, voxel_counter_buffer_read_to_write];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers).buffer_memory_barriers(&buffer_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let layouts = [descriptor_set_layout];
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_sets = device
        .allocate_descriptor_sets(&descriptor_set_allocate_info)
        .unwrap();
    let descriptor_set = descriptor_sets[0];

    let descriptor_image_info = vk::DescriptorImageInfo::default()
        .image_view(voxel_image_view)
        .image_layout(vk::ImageLayout::GENERAL)
        .sampler(vk::Sampler::null());
    let descriptor_image_infos = [descriptor_image_info];

    let descriptor_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(surface_buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_buffer_infos = [descriptor_buffer_info];

    let descriptor_index_image_info = vk::DescriptorImageInfo::default()
        .image_view(voxel_indices_image_view)
        .image_layout(vk::ImageLayout::GENERAL)
        .sampler(vk::Sampler::null());
    let descriptor_index_image_infos = [descriptor_index_image_info];

    let descriptor_buffer_counter_info = vk::DescriptorBufferInfo::default()
        .buffer(counter_buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_buffer_counter_infos = [descriptor_buffer_counter_info];

    let descriptor_write_1 = vk::WriteDescriptorSet::default()
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .dst_binding(0)
        .dst_set(descriptor_set)
        .image_info(&descriptor_image_infos);

    let descriptor_write_2 = vk::WriteDescriptorSet::default()
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .dst_binding(1)
        .dst_set(descriptor_set)
        .buffer_info(&descriptor_buffer_infos);

    let descriptor_write_3 = vk::WriteDescriptorSet::default()
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .dst_binding(2)
        .dst_set(descriptor_set)
        .image_info(&descriptor_index_image_infos);

    let descriptor_write_4 = vk::WriteDescriptorSet::default()
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .dst_binding(3)
        .dst_set(descriptor_set)
        .buffer_info(&descriptor_buffer_counter_infos);

    device
        .update_descriptor_sets(&[descriptor_write_1, descriptor_write_2, descriptor_write_3, descriptor_write_4], &[]);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &descriptor_sets,
        &[],
    );

    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        pipeline,
    );

    let data = [0u32];
    let raw = bytemuck::cast_slice::<u32, u8>(&data);

    device.cmd_update_buffer(cmd, counter_buffer, 0, raw);

    let barrier = vk::MemoryBarrier2::default()
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ)
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
    let barriers = [barrier];
    let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let raw = bytemuck::bytes_of(&push_constants);
    device.cmd_push_constants(cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, raw);
    
    device.cmd_dispatch(cmd, SIZE / 8, SIZE / 8, SIZE / 8);

    let voxel_image_write_to_read = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::TRANSFER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(voxel_image)
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
    voxel_image: vk::Image,
    voxel: u8,
    position: vek::Vec3<u32>,
) {
    let src_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(1);
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
    raw.copy_from_slice(&[voxel]);

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
        .dst_image(voxel_image)
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