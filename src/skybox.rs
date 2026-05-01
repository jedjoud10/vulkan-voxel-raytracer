use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::buffer;

pub struct Skybox {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub allocation: Allocation,
}

impl Skybox {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        device.destroy_sampler(self.sampler, None);
        allocator.free(self.allocation).unwrap();
    }
}

pub unsafe fn create_skybox(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    queue_family_index: u32,
) -> Skybox {
    let filter = vk::Filter::NEAREST;
    let resolution = 32;
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
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(6);
    let image = device.create_image(&skybox_image_create_info, None).unwrap();
    crate::debug::set_object_name(image, binder, "Skybox Texture");

    let requirements = device.get_image_memory_requirements(image);
    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            requirements: requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();
    device.bind_image_memory(image, allocation.memory(), allocation.offset()).unwrap();

    // create command buffer
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let image_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(6)
        .base_mip_level(0)
        .level_count(1);

    let image_layout_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags2::empty())
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(image)
        .subresource_range(image_subresource_range);
    let image_memory_barriers = [image_layout_transition];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    let mut texels = Vec::<vek::Vec4<f32>>::new();
    for layer in 0..6 {
        for x in 0..resolution {
            for y in 0..resolution {
                texels.push(vek::Vec4::new(x as f32 / resolution as f32, y as f32 / resolution as f32, layer as f32 / 6.0, 1.0));
            }
        }
    }

    let staging_buffer = buffer::create_staging_buffer2(device, allocator, bytemuck::cast_slice(texels.as_slice()));
    
    let image_subresource_layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(6)
        .mip_level(0)
        .base_array_layer(0);
    device.cmd_copy_buffer_to_image(cmd, staging_buffer.buffer, image, vk::ImageLayout::GENERAL, &[vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_image_height(0)
        .buffer_row_length(0)
        .image_extent(vk::Extent3D::default().depth(1).width(resolution).height(resolution))
        .image_offset(vk::Offset3D::default())
        .image_subresource(image_subresource_layers)
    ]);

    /*
    let clear_color_value = vk::ClearColorValue {
        float32: [0.5f32; 4]
    };
    device.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &clear_color_value, &[image_subresource_range]);
    */
    // end command buffer and submit
    device.end_command_buffer(cmd).unwrap();
    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);
    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();

    drop(staging_buffer);

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .base_array_layer(0)
        .layer_count(6)
        .level_count(1);

    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .components(vk::ComponentMapping::default())
        .flags(vk::ImageViewCreateFlags::empty())
        .format(format)
        .image(image)
        .subresource_range(subresource_range)
        .view_type(vk::ImageViewType::CUBE);
    let image_view = device
        .create_image_view(&image_view_create_info, None)
        .unwrap();

    let sampler_create_info = vk::SamplerCreateInfo::default()
        .mag_filter(filter)
        .min_filter(filter)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);

    let sampler = device.create_sampler(&sampler_create_info, None).unwrap();

    Skybox {
        image,
        image_view,
        allocation,
        sampler,
    }
}