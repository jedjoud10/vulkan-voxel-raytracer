use std::ffi::{CStr, CString};

use ash::vk;

pub unsafe fn create_swapchain(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    extent: vk::Extent2D,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> (
    ash::khr::swapchain::Device,
    vk::SwapchainKHR,
    Vec<vk::Image>,
    vk::Format,
) {
    let surface_capabilities = surface_loader
        .get_physical_device_surface_capabilities(physical_device, surface_khr)
        .unwrap();
    let present_modes: Vec<vk::PresentModeKHR> = surface_loader
        .get_physical_device_surface_present_modes(physical_device, surface_khr)
        .unwrap();
    let surface_formats: Vec<vk::SurfaceFormatKHR> = surface_loader
        .get_physical_device_surface_formats(physical_device, surface_khr)
        .unwrap();
    let present = present_modes
        .iter()
        .copied()
        .find(|&x| x == vk::PresentModeKHR::IMMEDIATE || x == vk::PresentModeKHR::MAILBOX)
        .unwrap();
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface_khr)
        .min_image_count(surface_capabilities.min_image_count)
        .image_format(surface_formats[0].format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(extent)
        .image_array_layers(1)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .image_usage(
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE,
        )
        .clipped(true)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .old_swapchain(vk::SwapchainKHR::null())
        .present_mode(present);

    let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);
    let swapchain = swapchain_loader
        .create_swapchain(&swapchain_create_info, None)
        .unwrap();
    let images = swapchain_loader.get_swapchain_images(swapchain).unwrap();

    if let Some(binder) = binder {
        for (i, image) in images.iter().enumerate() {
            let name = CString::new(format!("swapchain image {i}")).unwrap();
            let marker = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(*image)
                .object_name(&name);
            binder.set_debug_utils_object_name(&marker).unwrap();
        }
    }

    (swapchain_loader, swapchain, images, surface_formats[0].format)
}

pub const SCALING_FACTOR: u32 = 1;

pub unsafe fn create_temporary_target_render_image(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    queue_family_index: u32,
    extent: vk::Extent2D,
    binder: &Option<ash::ext::debug_utils::Device>,
    name: &CStr
) -> (vk::Image, gpu_allocator::vulkan::Allocation) {
    let queue_family_indices = [queue_family_index];
    let surface_formats: Vec<vk::SurfaceFormatKHR> = surface_loader
        .get_physical_device_surface_formats(physical_device, surface_khr)
        .unwrap();
    let rt_image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: extent.width / SCALING_FACTOR,
            height: extent.height / SCALING_FACTOR,
            depth: 1,
        })
        .format(surface_formats[0].format)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        //.tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let rt_image = device.create_image(&rt_image_create_info, None).unwrap();
    let requirements = device.get_image_memory_requirements(rt_image);

    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Render Texture Image Allocation",
            requirements: requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedImage(rt_image),
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let device_memory = allocation.memory();
    device
        .bind_image_memory(rt_image, device_memory, 0)
        .unwrap();

    if let Some(binder) = binder {
        let marker = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(rt_image)
            .object_name(name);
        binder.set_debug_utils_object_name(&marker).unwrap();
    }

    (rt_image, allocation)
}

pub unsafe fn transfer_rt_images(
    device: &ash::Device,
    queue_family_index: u32,
    images: &[(vk::Image, gpu_allocator::vulkan::Allocation)],
    pool: vk::CommandPool,
    queue: vk::Queue,
) {
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];

    let cmd_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device
        .begin_command_buffer(cmd, &cmd_buffer_begin_info)
        .unwrap();

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let barriers = images.iter().map(|(image, _)| {
        vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(*image)
            .subresource_range(subresource_range)
    });

    let barriers = barriers.collect::<Vec<_>>();
    let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    device.end_command_buffer(cmd).unwrap();

    let cmds = [cmd];
    //let wait_masks = [vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::ALL_GRAPHICS];
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&cmds)
        .wait_semaphores(&[])
        .signal_semaphores(&[])
        .wait_dst_stage_mask(&[]);
    let fence = device.create_fence(&Default::default(), None).unwrap();
    device.queue_submit(queue, &[submit_info], fence).unwrap();
    device.wait_for_fences(&[fence], false, u64::MAX).unwrap();
    device.destroy_fence(fence, None);
    device.free_command_buffers(pool, &[cmd]);
}
