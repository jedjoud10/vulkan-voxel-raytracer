use std::{ffi::CString, str::FromStr};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};


pub struct Buffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
}

impl Buffer {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        device.destroy_buffer(self.buffer, None);
        allocator.free(self.allocation).unwrap();
    }
}

pub unsafe fn create_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    size: usize,
    binder: &Option<ash::ext::debug_utils::Device>,
    name: &str,
    usage: vk::BufferUsageFlags,
) -> Buffer {
    let buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(size as u64);
    let buffer = device.create_buffer(&buffer_create_info, None).unwrap();

    let requirements = device.get_buffer_memory_requirements(buffer);

    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: &format!("{name} allocation"),
            requirements: requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(buffer),
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();

    let tmp = CString::from_str(name).unwrap();

    if let Some(binder) = binder {
        let marker = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(buffer)
            .object_name(tmp.as_c_str());
        binder.set_debug_utils_object_name(&marker).unwrap();
    }

    
    let device_memory = allocation.memory();
    device.bind_buffer_memory(buffer, device_memory, 0).unwrap();

    Buffer {
        buffer, allocation
    }
}

pub unsafe fn create_counter_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    name: &str,
) -> Buffer {
    create_buffer(device, allocator, size_of::<u32>(), binder, name, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
}