use ash::vk;

use crate::queue;

pub unsafe fn create_device_and_queue(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
) -> (ash::Device, u32, vk::Queue) {
    let queue_family_properties =
        instance.get_physical_device_queue_family_properties(physical_device);
    let queue_family_index = queue::find_appropriate_queue_family_index(
        physical_device,
        queue_family_properties,
        &surface_loader,
        surface_khr,
    ) as u32;

    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_priorities(&[1.0])
        .queue_family_index(queue_family_index);
    let queue_create_infos = [queue_create_info];

    let device_features = vk::PhysicalDeviceFeatures::default();
    let mut device_features_13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(true);
    let mut device_features_12 = vk::PhysicalDeviceVulkan12Features::default()
        .storage_buffer8_bit_access(true)
        .shader_int8(true);

    let device_extension_names = [
        ash::khr::swapchain::NAME,
        //ash::ext::debug_marker::NAME,
    ];

    let device_extension_names_ptrs = device_extension_names
        .iter()
        .map(|cstr| cstr.as_ptr())
        .collect::<Vec<_>>();

    let device_create_info = vk::DeviceCreateInfo::default()
        .enabled_extension_names(&device_extension_names_ptrs)
        .enabled_features(&device_features)
        .queue_create_infos(&queue_create_infos)
        .push_next(&mut device_features_13)
        .push_next(&mut device_features_12);

    let device = instance
        .create_device(physical_device, &device_create_info, None)
        .unwrap();
    log::info!("created device");

    let queue = device.get_device_queue(queue_family_index, 0);
    log::info!("fetched queue");

    (device, queue_family_index, queue)
}
