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
    flags: vk::BufferUsageFlags,
) -> Buffer {
    let bytes_formatted = bytesize::ByteSize::b(size as u64);
    log::debug!("creating buffer {} ({})", name, bytes_formatted.display().si());
    let buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(flags)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(size as u64);
    let buffer = device.create_buffer(&buffer_create_info, None).unwrap();

    let requirements = device.get_buffer_memory_requirements(buffer);

    let allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: &format!("{name} allocation"),
            requirements: requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
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
    device.bind_buffer_memory(buffer, device_memory, allocation.offset()).unwrap();
    
    log::debug!("created buffer {} of size {}", name, bytes_formatted.display().si());

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


// `write_to_buffer` calls that update more than these amount of bytes will revert to using the staging buffer implementation
const BUFFER_WRITE_INLINE_MAX_BYTES_THRESHOLD: usize = 65536; // vulkan spec states that data size must be less than this 

// this either creates a staging buffer write or writes to the buffer through cmd_update_buffer
// switches between both impls depending on the amount of data to write
pub unsafe fn write_to_buffer(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    dst_buffer: vk::Buffer,
    allocator: &mut Allocator,
    bytes: &[u8]
) {
    let start = std::time::Instant::now();
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];
    device.begin_command_buffer(cmd, &Default::default()).unwrap();

    let bytes_formatted = bytesize::ByteSize::b(bytes.len() as u64);
    let staging_buffer_opt = if bytes.len() < BUFFER_WRITE_INLINE_MAX_BYTES_THRESHOLD {
        // inline (command buffer write) impl
        log::info!("writing {} to buffer, using inline path", bytes_formatted.display().si());
        device.cmd_update_buffer(cmd, dst_buffer, 0, bytes);
        None
    } else {
        log::info!("writing {} to buffer, using staging buffer path", bytes_formatted.display().si());
        let (staging_buffer, allocation) = create_staging_buffer(device, allocator, bytes);

        let region = vk::BufferCopy2::default()
            .dst_offset(0)
            .size(bytes.len() as u64)
            .src_offset(0);

        let regions = [region];
        let copy_staging_buffer_to_buffer = vk::CopyBufferInfo2::default()
            .dst_buffer(dst_buffer)
            .regions(&regions)
            .src_buffer(staging_buffer);

        device.cmd_copy_buffer2(cmd, &copy_staging_buffer_to_buffer);
        Some((staging_buffer, allocation))
    };

    
    device.end_command_buffer(cmd).unwrap();

    let buffers = [cmd];
    let submit = vk::SubmitInfo::default()
        .command_buffers(&buffers);

    device.queue_submit(queue, & [submit], vk::Fence::null()).unwrap();

    // TODO: definitely don't do this if we want to optimize, but for now it's ok
    device.device_wait_idle().unwrap();

    // destroy staging buffer if we used it
    if let Some((staging_buffer, allocation)) = staging_buffer_opt {
        allocator.free(allocation).unwrap();
        device.destroy_buffer(staging_buffer, None);
    }

    let end = std::time::Instant::now();
    log::debug!("buffer write took {}μs", (end-start).as_micros());
}

pub unsafe fn create_staging_buffer(device: &ash::Device, allocator: &mut Allocator, bytes: &[u8]) -> (vk::Buffer, Allocation) {
    // staging buffer impl
    let staging_buffer_create_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .size(bytes.len() as u64);

    let staging_buffer = device.create_buffer(&staging_buffer_create_info, None).unwrap();

    let requirements = device.get_buffer_memory_requirements(staging_buffer);
    let mut allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Staging Buffer",
            requirements: requirements,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
        })
        .unwrap();

    device.bind_buffer_memory(staging_buffer, allocation.memory(), allocation.offset()).unwrap();
        
    let dst_slice = allocation.mapped_slice_mut().unwrap();

    // FIXME: for some reason on nvidia the slice has different size? shouldn't gpu_allocator handle this type of stuff...
    dst_slice[..(bytes.len())].copy_from_slice(bytes);
    (staging_buffer, allocation)
}



pub unsafe fn create_counter_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
    name: &str,
) -> Buffer {
    create_buffer(device, allocator, size_of::<u32>(), binder, name, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
}