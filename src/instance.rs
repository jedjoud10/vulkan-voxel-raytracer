use std::ffi::CStr;

use ash::vk;
use raw_window_handle::RawDisplayHandle;

const REQUIRED_INSTANCE_EXTENSIONS: &'static [&'static CStr] =
    &[ash::ext::debug_utils::NAME, ash::khr::surface::NAME, ash::ext::validation_features::NAME];

const REQUIRED_INSTANCE_VALIDATION_LAYERS: &'static [&'static CStr] = &[
    #[cfg(debug_assertions)]
    c"VK_LAYER_KHRONOS_validation",
];

pub unsafe fn create_instance(
    entry: &ash::Entry,
    raw_display_handle: RawDisplayHandle,
) -> ash::Instance {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Test Vulkan App")
        .api_version(vk::API_VERSION_1_3)
        .application_version(0)
        .engine_version(0)
        .engine_name(c"Unnamed");

    let mut extension_names_ptrs = ash_window::enumerate_required_extensions(raw_display_handle)
        .unwrap()
        .to_vec();

    extension_names_ptrs.extend(REQUIRED_INSTANCE_EXTENSIONS.iter().map(|s| s.as_ptr()));

    let validation_ptrs = REQUIRED_INSTANCE_VALIDATION_LAYERS
        .iter()
        .map(|cstr| cstr.as_ptr())
        .collect::<Vec<_>>();

    let enabled_validation_features = [
        #[cfg(debug_assertions)]
        vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
    ];
    let mut validation_features = ash::vk::ValidationFeaturesEXT::default()
        .enabled_validation_features(&enabled_validation_features);

    let instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&validation_ptrs)
        .enabled_extension_names(&extension_names_ptrs)
        .push_next(&mut validation_features);
    entry.create_instance(&instance_create_info, None).unwrap()
}
