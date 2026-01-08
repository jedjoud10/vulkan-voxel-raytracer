use ash::vk::{self};
use std::ffi::{c_void, CStr};

#[cfg(debug_assertions)]
pub unsafe fn create_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static>
{
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(debug_callback))
}

pub unsafe fn create_debug_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Option<(
    ash::ext::debug_utils::Instance,
    ash::vk::DebugUtilsMessengerEXT,
)> {
    let debug_messenger;
    #[cfg(debug_assertions)]
    {
        let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
        let messenger = debug_utils
            .create_debug_utils_messenger(&create_debug_messenger_create_info(), None)
            .unwrap();
        debug_messenger = Some((debug_utils, messenger));
    }
    #[cfg(not(debug_assertions))]
    {
        debug_messenger = None;
    }

    debug_messenger
}

pub unsafe fn create_debug_marker(
    instance: &ash::Instance,
    device: &ash::Device,
) -> ash::ext::debug_utils::Device {
    ash::ext::debug_utils::Device::new(instance, device)
}

#[cfg(debug_assertions)]
pub unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _cvoid: *mut c_void,
) -> u32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        c""
    } else {
        CStr::from_ptr(callback_data.p_message_id_name)
    };

    let message = if callback_data.p_message.is_null() {
        c""
    } else {
        CStr::from_ptr(callback_data.p_message)
    };

    pub const VERBOSE: u32 = 0b1;
    pub const INFO: u32 = 0b1_0000;
    pub const WARNING: u32 = 0b1_0000_0000;
    pub const ERROR: u32 = 0b1_0000_0000_0000;

    match message_severity.as_raw() {
        VERBOSE => log::debug!(
            "{:?} [{} ({})] : {}\n",
            message_type,
            message_id_name.to_str().unwrap(),
            &message_id_number.to_string(),
            message.to_str().unwrap(),
        ),
        INFO => {
            if (message_id_number == 0x4fe1fef9) {
                log::info!("{}", message.to_str().unwrap());
                /*
                let bruh = message.to_str().unwrap().split('|').collect::<Vec<&str>>();
                let concat = bruh[2..].join("|");
                */
            } else {
                log::info!(
                    "{:?} [{} ({})] : {}\n",
                    message_type,
                    message_id_name.to_str().unwrap(),
                    &message_id_number.to_string(),
                    message.to_str().unwrap(),
                )
            }
        },
        WARNING => log::warn!(
            "{:?} [{} ({})] : {}\n",
            message_type,
            message_id_name.to_str().unwrap(),
            &message_id_number.to_string(),
            message.to_str().unwrap(),
        ),
        ERROR => log::error!(
            "{:?} [{} ({})] : {}\n",
            message_type,
            message_id_name.to_str().unwrap(),
            &message_id_number.to_string(),
            message.to_str().unwrap(),
        ),
        _ => {}
    }

    vk::FALSE
}
