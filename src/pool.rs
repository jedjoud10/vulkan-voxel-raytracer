use ash::vk;

pub unsafe fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
    let images = vk::DescriptorPoolSize::default()
        .descriptor_count(5)
        .ty(vk::DescriptorType::STORAGE_IMAGE);
    let buffers = vk::DescriptorPoolSize::default()
        .descriptor_count(3)
        .ty(vk::DescriptorType::STORAGE_BUFFER);
    let descriptor_pool_sizes = [images, buffers];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(2)
        .pool_sizes(&descriptor_pool_sizes);

    let descriptor_pool = device
        .create_descriptor_pool(&descriptor_pool_create_info, None)
        .unwrap();
    descriptor_pool
}
