use std::ffi::CStr;

use ash::vk;

// Check wether or not a physical device is suitable for rendering
// This checks the minimum requirements that we need to achieve to be able to render
pub(super) unsafe fn get_physical_device_score(
    physical_device: vk::PhysicalDevice,
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
) -> Option<u32> {
    let properties = instance.get_physical_device_properties(physical_device);
    let surface_capabilities: vk::SurfaceCapabilitiesKHR = surface_loader
        .get_physical_device_surface_capabilities(physical_device, surface_khr)
        .unwrap();
    let present_modes: Vec<vk::PresentModeKHR> = surface_loader
        .get_physical_device_surface_present_modes(physical_device, surface_khr)
        .unwrap();
    let surface_formats: Vec<vk::SurfaceFormatKHR> = surface_loader
        .get_physical_device_surface_formats(physical_device, surface_khr)
        .unwrap();

    log::info!(
        "checking physical device {}...",
        CStr::from_ptr(properties.device_name.as_ptr())
            .to_str()
            .unwrap()
    );

    let mut score = 0;

    let double_buffering_supported = surface_capabilities.min_image_count >= 2;
    log::info!("double buffering: {double_buffering_supported}");

    let present_modes_supported = present_modes
        .iter()
        .find(|&&present| {
            matches!(present, vk::PresentModeKHR::FIFO_RELAXED)
                || matches!(present, vk::PresentModeKHR::IMMEDIATE)
        })
        .is_some();

    log::info!("present modes supported: {present_modes_supported}");
    let surface_compatible = surface_formats
        .iter()
        .find(|format| {
            let format_ = format.format == vk::Format::B8G8R8A8_SRGB;
            let color_space_ = format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
            format_ && color_space_
        })
        .is_some();
    log::info!("compatible surface: {surface_compatible}");

    if !double_buffering_supported || !present_modes_supported || !surface_compatible {
        return None;
    }

    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        score += 100;
    }

    Some(score)
}
