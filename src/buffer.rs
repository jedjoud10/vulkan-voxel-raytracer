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
) -> Buffer {
    log::debug!("creating buffer {name} ({}kb)", size / 1024);
    let buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())

        // TODO: fix this hack
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER)
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

    log::debug!("created buffer {name} of size {size}");

    Buffer {
        buffer, allocation
    }
}

pub unsafe fn fill_buffer(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    dst_buffer: vk::Buffer,
    data: u32,
) {
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    device.cmd_fill_buffer(cmd, dst_buffer, 0, vk::WHOLE_SIZE, data);

    device.end_command_buffer(cmd).unwrap();

    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);


    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();
}

pub unsafe fn write_to_buffer(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    dst_buffer: vk::Buffer,
    allocator: &mut Allocator,
    data: &[u8]
) {
    let staging_buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(data.len() as u64);

    let staging_buffer = device.create_buffer(&staging_buffer_create_info, None).unwrap();

    let requirements = device.get_buffer_memory_requirements(staging_buffer);
    let mut allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Staging Buffer",
            requirements: requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(staging_buffer),
            location: gpu_allocator::MemoryLocation::CpuToGpu,
        })
        .unwrap();

    let device_memory = allocation.memory();
    device.bind_buffer_memory(staging_buffer, device_memory, 0).unwrap();
    allocation.mapped_slice_mut().unwrap().copy_from_slice(data);


    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let region = vk::BufferCopy2::default()
        .dst_offset(0)
        .size(data.len() as u64)
        .src_offset(0);

    let regions = [region];
    let copy_staging_buffer_to_buffer = vk::CopyBufferInfo2::default()
        .dst_buffer(dst_buffer)
        .regions(&regions)
        .src_buffer(staging_buffer);

    device.cmd_copy_buffer2(cmd, &copy_staging_buffer_to_buffer);
    device.end_command_buffer(cmd).unwrap();

    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);


    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();
    allocator.free(allocation).unwrap();
    device.destroy_buffer(staging_buffer, None);
}



pub unsafe fn create_counter_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    name: &str,
) -> Buffer {
    create_buffer(device, allocator, size_of::<u32>(), binder, name)
}