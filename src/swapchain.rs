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
    Vec<vk::ImageView>,
    vk::Format,
) {
    let num_swapchain_images = super::per_frame_data::FRAMES_IN_FLIGHT as u32;
    
    let present_modes: Vec<vk::PresentModeKHR> = surface_loader
        .get_physical_device_surface_present_modes(physical_device, surface_khr)
        .unwrap();
    log::debug!("present modes {:?}", present_modes);
    

    let surface_formats: Vec<vk::SurfaceFormatKHR> = surface_loader
        .get_physical_device_surface_formats(physical_device, surface_khr)
        .unwrap();
    log::debug!("surface formats {:?}", surface_formats);
    
    let swapchain_format = surface_formats[0].format;
    let _present = present_modes
        .iter()
        .copied()
        .find(|&x| x == vk::PresentModeKHR::IMMEDIATE || x == vk::PresentModeKHR::MAILBOX)
        .unwrap();
    
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface_khr)
        .min_image_count(num_swapchain_images)
        .image_format(swapchain_format)
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
        .present_mode(vk::PresentModeKHR::IMMEDIATE);

    let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);
    let swapchain = swapchain_loader
        .create_swapchain(&swapchain_create_info, None)
        .unwrap();
    let images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
    let image_views = images.iter().map(|swapchain_image| {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(swapchain_format)
            .image(*swapchain_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        device.create_image_view(&image_view_create_info, None).unwrap()
    }).collect::<Vec<vk::ImageView>>();
    
    for (i, image) in images.iter().enumerate() {
        crate::debug::set_object_name(*image, binder, format!("swapchain image {i}"));
    }

    for (i, image_view) in image_views.iter().enumerate() {
        crate::debug::set_object_name(*image_view, binder, format!("swapchain image view {i}"));
    }
    
    (swapchain_loader, swapchain, images, image_views, swapchain_format)
}