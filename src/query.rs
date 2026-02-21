use ash::vk;

pub unsafe fn create_query_pool(
    device: &ash::Device,
) -> vk::QueryPool {
    let create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2);
    device.create_query_pool(&create_info, None).unwrap()
}