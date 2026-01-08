use ash::vk;

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
            present && graphics && compute
        })
        .unwrap()
}

pub unsafe fn find_async_compute_queue(
    physical_device: vk::PhysicalDevice,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
) -> usize {
    queue_family_properties
        .iter()
        .enumerate()
        .position(|(i, props)| {
            let graphics = props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let compute = props.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let transfer = props.queue_flags.contains(vk::QueueFlags::TRANSFER);

            return !graphics & compute & !transfer;
        })
        .unwrap()
}