use ash::vk;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstants {
    pub screen_resolution: vek::Vec2<f32>,
    pub _padding: vek::Vec2<f32>,
    pub matrix: vek::Mat4<f32>,
    pub position: vek::Vec4<f32>,
    pub sun: vek::Vec4<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstants2 {
    pub forward: vek::Vec4<f32>,
    pub position: vek::Vec4<f32>,
    pub sun: vek::Vec4<f32>,
    pub tick: u32,
    pub delta: f32,
}

pub unsafe fn create_render_compute_pipeline(
    raw: &[u32],
    device: &ash::Device,
) -> (
    vk::ShaderModule,
    vk::DescriptorSetLayout,
    vk::PipelineLayout,
    vk::Pipeline,
) {
    let render_compute_shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());
    let render_compute_shader_module = device
        .create_shader_module(&render_compute_shader_module_create_info, None)
        .unwrap();

    let render_compute_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"main")
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(render_compute_shader_module);

    let render_descriptor_set_layout_binding_rt_image = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let render_descriptor_set_layout_binding_voxel_image = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let render_descriptor_set_layout_binding_voxel_surface_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let render_descriptor_set_layout_binding_voxel_surface_index_image = vk::DescriptorSetLayoutBinding::default()
        .binding(3)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let render_descriptor_set_layout_bindings = [
        render_descriptor_set_layout_binding_rt_image,
        render_descriptor_set_layout_binding_voxel_image,
        render_descriptor_set_layout_binding_voxel_surface_buffer,
        render_descriptor_set_layout_binding_voxel_surface_index_image
    ];

    let render_descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&render_descriptor_set_layout_bindings);

    let render_compute_descriptor_set_layout = device
        .create_descriptor_set_layout(&render_descriptor_set_layout_create_info, None)
        .unwrap();
    let render_compute_descriptor_set_layouts = [render_compute_descriptor_set_layout];

    let render_push_constant_range = vk::PushConstantRange::default()
        .offset(0)
        .size(size_of::<PushConstants>() as u32)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);
    let render_push_constants = [render_push_constant_range];

    let render_compute_pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&render_push_constants)
        .flags(vk::PipelineLayoutCreateFlags::empty())
        .set_layouts(&render_compute_descriptor_set_layouts);

    let render_compute_pipeline_layout = device
        .create_pipeline_layout(&render_compute_pipeline_layout_create_info, None)
        .unwrap();

    let render_compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
        .layout(render_compute_pipeline_layout)
        .stage(render_compute_stage_create_info);
    let render_compute_pipelines = device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[render_compute_pipeline_create_info],
            None,
        )
        .unwrap();
    let render_compute_pipeline = render_compute_pipelines[0];
    
    (
        render_compute_shader_module,
        render_compute_descriptor_set_layout,
        render_compute_pipeline_layout,
        render_compute_pipeline,
    )
}

pub unsafe fn create_compute_voxel_pipelines(
    raw: &[u32],
    device: &ash::Device,
) -> (vk::ShaderModule, [(
    vk::DescriptorSetLayout,
    vk::PipelineLayout,
    vk::Pipeline,
); 2]) {
    let compute_shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());
    let compute_shader_module = device
        .create_shader_module(&compute_shader_module_create_info, None)
        .unwrap();

    let compute_init_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"main")
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_shader_module);

    let compute_test_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"update")
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_shader_module);

    let descriptor_set_layout_binding_voxel_image = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let descriptor_set_layout_bindings = [
        descriptor_set_layout_binding_voxel_image,
    ];

    let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&descriptor_set_layout_bindings);

    let compute_descriptor_set_layout = device
        .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        .unwrap();
    let compute_descriptor_set_layouts = [compute_descriptor_set_layout];

    let compute_pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&[])
        .flags(vk::PipelineLayoutCreateFlags::empty())
        .set_layouts(&compute_descriptor_set_layouts);

    let compute_pipeline_layout = device
        .create_pipeline_layout(&compute_pipeline_layout_create_info, None)
        .unwrap();

    let compute_pipeline_init_create_info = vk::ComputePipelineCreateInfo::default()
        .layout(compute_pipeline_layout)
        .stage(compute_init_stage_create_info);

    let descriptor_set_layout_binding_surface_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let descriptor_set_layout_binding_voxel_surface_index_image = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let descriptor_set_layout_binding_counter_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(3)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);

    let descriptor_set_layout_bindings = [
        descriptor_set_layout_binding_voxel_image,
        descriptor_set_layout_binding_surface_buffer,
        descriptor_set_layout_binding_voxel_surface_index_image,
        descriptor_set_layout_binding_counter_buffer
    ];
    
    let descriptor_set_test_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&descriptor_set_layout_bindings);

    let compute_descriptor_test_set_layout = device
        .create_descriptor_set_layout(&descriptor_set_test_layout_create_info, None)
        .unwrap();
    let compute_descriptor_test_set_layouts = [compute_descriptor_test_set_layout];

    let compute_pipeline_test_layout_push_constant_range = vk::PushConstantRange::default()
        .offset(0)
        .size(size_of::<PushConstants2>() as u32)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);
    let compute_pipeline_test_layout_push_constant_ranges = [compute_pipeline_test_layout_push_constant_range];

    let compute_pipeline_test_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&compute_pipeline_test_layout_push_constant_ranges)
        .flags(vk::PipelineLayoutCreateFlags::empty())
        .set_layouts(&compute_descriptor_test_set_layouts);

    let compute_pipeline_test_layout = device
        .create_pipeline_layout(&compute_pipeline_test_layout_create_info, None)
        .unwrap();
    
    let compute_pipeline_test_create_info = vk::ComputePipelineCreateInfo::default()
        .layout(compute_pipeline_test_layout)
        .stage(compute_test_stage_create_info);

    let compute_pipelines = device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[compute_pipeline_init_create_info, compute_pipeline_test_create_info],
            None,
        )
        .unwrap();

    let first = (compute_descriptor_set_layout, compute_pipeline_layout, compute_pipelines[0]);
    let second = (compute_descriptor_test_set_layout, compute_pipeline_test_layout, compute_pipelines[1]);
    
    let compute_pipeline = compute_pipelines[0];
    (compute_shader_module,[first, second])
}