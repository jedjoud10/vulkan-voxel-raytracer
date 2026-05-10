use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use ash::vk;

pub unsafe fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
    let images = vk::DescriptorPoolSize::default()
        .descriptor_count(160)
        .ty(vk::DescriptorType::STORAGE_IMAGE);
    let buffers = vk::DescriptorPoolSize::default()
        .descriptor_count(60)
        .ty(vk::DescriptorType::STORAGE_BUFFER);
    let combined_image_samplers = vk::DescriptorPoolSize::default()
        .descriptor_count(40)
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
    let descriptor_pool_sizes = [images, buffers, combined_image_samplers];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(320)
        .pool_sizes(&descriptor_pool_sizes);

    
    device
        .create_descriptor_pool(&descriptor_pool_create_info, None)
        .unwrap()
}

pub unsafe fn create_query_pool(
    device: &ash::Device
) -> vk::QueryPool {
    let create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2);
    let query = device.create_query_pool(&create_info, None).unwrap();

    query
}

pub unsafe fn find_appropriate_queue_family_index(
    physical_device: vk::PhysicalDevice,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
) -> usize {
    queue_family_properties
        .iter()
        .enumerate()
        .position(|(i, props)| {
            let present = surface_loader
                .get_physical_device_surface_support(physical_device, i as u32, surface_khr)
                .unwrap();
            let graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let compute = props.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let has_timestamps = props.timestamp_valid_bits > 0;
            present && graphics && compute && has_timestamps
        })
        .unwrap()
}

pub unsafe fn find_async_compute_queue(
    _physical_device: vk::PhysicalDevice,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
) -> usize {
    queue_family_properties
        .iter()
        .enumerate()
        .position(|(_i, props)| {
            let graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let compute = props.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let transfer = props.queue_flags.contains(vk::QueueFlags::TRANSFER);

            !graphics & compute & !transfer
        })
        .unwrap()
}

pub unsafe fn create_surface(
    instance: &ash::Instance,
    entry: &ash::Entry,
    window: &winit::window::Window,
) -> (ash::khr::surface::Instance, vk::SurfaceKHR) {
    let surface = ash_window::create_surface(
        entry,
        instance,
        window.display_handle().unwrap().as_raw(),
        window.window_handle().unwrap().as_raw(),
        None,
    )
    .unwrap();
    let surface_loader = ash::khr::surface::Instance::new(entry, instance);
    (surface_loader, surface)
}