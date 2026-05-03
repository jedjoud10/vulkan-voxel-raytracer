use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
pub struct Skybox {
    pub skybox_image: vk::Image,
    pub skybox_image_view: vk::ImageView,
    pub skybox_array_image_view: vk::ImageView,
    
    pub clouds_image: vk::Image,
    pub clouds_image_view: vk::ImageView,
    
    pub skybox_image_allocation: Allocation,
    pub clouds_image_allocation: Allocation,

    pub sampler: vk::Sampler,
}

impl Skybox {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.skybox_image_view, None);
        device.destroy_image_view(self.skybox_array_image_view, None);
        device.destroy_image(self.skybox_image, None);
        
        device.destroy_sampler(self.sampler, None);
        
        device.destroy_image(self.clouds_image, None);
        device.destroy_image_view(self.clouds_image_view, None);
        
        
        allocator.free(self.skybox_image_allocation).unwrap();
        allocator.free(self.clouds_image_allocation).unwrap();
        
    }
}

pub const SKYBOX_RESOLUTION: u32 = 256;
pub const CLOUDS_RESOLUTION: u32 = 512;



pub unsafe fn create_skybox(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    queue_family_index: u32,
) -> Skybox {
    let filter = vk::Filter::NEAREST;
    let resolution = SKYBOX_RESOLUTION;
    let queue_family_indices = [queue_family_index];
    let format = vk::Format::R32G32B32A32_SFLOAT;

    let skybox_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: resolution,
            height: resolution,
            depth: 1,
        })
        .format(format)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(6);
    let skybox_image = device.create_image(&skybox_image_create_info, None).unwrap();
    crate::debug::set_object_name(skybox_image, binder, "Skybox Texture");

    let requirements = device.get_image_memory_requirements(skybox_image);
    let skybox_image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            requirements: requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(skybox_image, skybox_image_allocation.memory(), skybox_image_allocation.offset()).unwrap();
    
    let clouds_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: resolution,
            height: resolution,
            depth: 1,
        })
        .format(format)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let clouds_image = device.create_image(&clouds_image_create_info, None).unwrap();
    crate::debug::set_object_name(clouds_image, binder, "Clouds Texture");

    let requirements = device.get_image_memory_requirements(clouds_image);
    let clouds_image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            requirements: requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(clouds_image, clouds_image_allocation.memory(), clouds_image_allocation.offset()).unwrap();

    // create command buffer
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let skybox_image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(6)
        .base_mip_level(0)
        .level_count(1);
    let clouds_image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .base_mip_level(0)
        .level_count(1);

    let skybox_image_layout_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::empty())
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(skybox_image)
        .subresource_range(skybox_image_subresource_range);
    let clouds_image_layout_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::empty())
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(clouds_image)
        .subresource_range(clouds_image_subresource_range);
    let image_memory_barriers = [skybox_image_layout_transition, clouds_image_layout_transition];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    // end command buffer and submit
    device.end_command_buffer(cmd).unwrap();
    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);
    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();

    let skybox_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .base_array_layer(0)
        .layer_count(6)
        .level_count(1);

    let clouds_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(format)
        .image(skybox_image)
        .subresource_range(skybox_subresource_range)
        .view_type(vk::ImageViewType::CUBE);
    let image_view = device
        .create_image_view(&image_view_create_info, None)
        .unwrap();

    let array_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(format)
        .image(skybox_image)
        .subresource_range(skybox_subresource_range)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY);
    let skybox_array_image_view = device
        .create_image_view(&array_image_view_create_info, None)
        .unwrap();


    let clouds_image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(format)
        .image(clouds_image)
        .subresource_range(clouds_subresource_range)
        .view_type(vk::ImageViewType::TYPE_2D);
    let clouds_image_view = device
        .create_image_view(&clouds_image_view_create_info, None)
        .unwrap();

    let sampler_create_info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);

    let sampler = device.create_sampler(&sampler_create_info, None).unwrap();

    Skybox {
        skybox_image,
        skybox_image_view: image_view,
        skybox_image_allocation,
        sampler,
        skybox_array_image_view,
        clouds_image,
        clouds_image_view,
        clouds_image_allocation,
    }
}