use std::{ffi::{CStr, CString}, ops::Mul, str::FromStr};

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
    pub debug_type: u32,
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

pub struct ComputePipeline {
    pub module: vk::ShaderModule,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl ComputePipeline {
    pub unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        device.destroy_shader_module(self.module, None);
    }
}

pub struct SingleEntryPointWrapper {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

pub struct MultiComputePipeline<const N: usize> {
    pub module: vk::ShaderModule,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub entry_points: [SingleEntryPointWrapper; N],
}

impl<const N: usize> MultiComputePipeline<N> {
    pub unsafe fn destroy(self, device: &ash::Device) {
        for x in self.entry_points {
            device.destroy_pipeline(x.pipeline, None);
            device.destroy_pipeline_layout(x.pipeline_layout, None);
        }
        
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        device.destroy_shader_module(self.module, None);
    }
}

pub type VoxelGeneratePipeline = MultiComputePipeline<2>;
pub type VoxelTickPipeline = MultiComputePipeline<4>;
pub type RenderPipeline = MultiComputePipeline<1>;

pub unsafe fn create_render_compute_pipeline(
    raw: &[u32],
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> RenderPipeline {
    let render_compute_shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());

    let render_compute_shader_module = device
        .create_shader_module(&render_compute_shader_module_create_info, None)
        .unwrap();

    crate::debug::set_object_name(render_compute_shader_module, binder, "render compute shader module");

    let output = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let svo_bitmasks = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let svo_indices = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let bindings = [
        output,
        svo_bitmasks,
        svo_indices,
    ];

    let render_descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&bindings);

    let render_compute_descriptor_set_layout = device
        .create_descriptor_set_layout(&render_descriptor_set_layout_create_info, None)
        .unwrap();

    crate::debug::set_object_name(render_compute_descriptor_set_layout, binder, "render compute descriptor set layout");
    let render_compute_descriptor_set_layouts = [render_compute_descriptor_set_layout];

    let push_constant_size = Some(size_of::<PushConstants>());
    let main_entry_point = create_single_entry_point_pipeline(device, &binder, render_compute_shader_module, "main", render_compute_descriptor_set_layout, push_constant_size);
    
    return MultiComputePipeline {
        module: render_compute_shader_module,
        descriptor_set_layout: render_compute_descriptor_set_layout,
        entry_points: [main_entry_point]
    }
}

pub unsafe fn create_tick_voxel_compute_pipeline(
    raw: &[u32],
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> VoxelTickPipeline {
    let compute_shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());
    let compute_shader_module = device
        .create_shader_module(&compute_shader_module_create_info, None)
        .unwrap();
    crate::debug::set_object_name(compute_shader_module, binder, "tick voxel compute shader module");


    let descriptor_set_layout_binding_voxel_image = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let descriptor_set_layout_binding_voxel_surface_index_image = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let descriptor_set_layout_binding_surface_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let descriptor_set_layout_binding_counter_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(3)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let descriptor_set_layout_binding_visible_surfaces_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(4)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let descriptor_set_layout_binding_visible_surfaces_counter_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(5)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let descriptor_set_layout_binding_indirect_dispatch_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(6)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);

    let bindings = [
        descriptor_set_layout_binding_voxel_image,
        descriptor_set_layout_binding_surface_buffer,
        descriptor_set_layout_binding_voxel_surface_index_image,
        descriptor_set_layout_binding_counter_buffer,
        descriptor_set_layout_binding_visible_surfaces_buffer,
        descriptor_set_layout_binding_visible_surfaces_counter_buffer,
        descriptor_set_layout_binding_indirect_dispatch_buffer
    ];

    let descriptor_set_test_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&bindings);
    
    let compute_descriptor_test_set_layout = device
        .create_descriptor_set_layout(&descriptor_set_test_layout_create_info, None)
        .unwrap();

    crate::debug::set_object_name(compute_descriptor_test_set_layout, binder, "tick voxel compute descriptor set layout");

    let push_constant_size = Some(size_of::<PushConstants2>());
    let unwrap_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "unwrap", compute_descriptor_test_set_layout, push_constant_size);
    let unpack_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "unpack", compute_descriptor_test_set_layout, push_constant_size);
    let tick_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "main", compute_descriptor_test_set_layout, push_constant_size);
    let copy_dispatch_params_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "copyDispatchSize", compute_descriptor_test_set_layout, push_constant_size);
    
    return MultiComputePipeline {
        module: compute_shader_module,
        descriptor_set_layout: compute_descriptor_test_set_layout,
        entry_points: [unwrap_entry_point, unpack_entry_point, tick_entry_point, copy_dispatch_params_entry_point]
    }
}

pub unsafe fn create_single_entry_point_pipeline(device: &ash::Device, binder: &Option<ash::ext::debug_utils::Device>, compute_shader_module: vk::ShaderModule, entry_point_name: &str, descriptor_set_layout: vk::DescriptorSetLayout, push_constant_size: Option<usize>) -> SingleEntryPointWrapper {
    let string = CString::from_str(entry_point_name).unwrap();

    log::info!("creating single entry point pipline for {entry_point_name}. pc range: {push_constant_size:?}");
    let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(string.as_c_str())
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_shader_module);
    
    let descriptor_set_layouts: [vk::DescriptorSetLayout; 1] = [descriptor_set_layout];
    
    let compute_pipeline_layout = if let Some(push_constant_size) = push_constant_size {
        let push_constant_range = vk::PushConstantRange::default()
            .offset(0)
            .size(push_constant_size as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let push_constant_ranges = [push_constant_range];
        
        let compute_pipeline_test_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constant_ranges)
            .flags(vk::PipelineLayoutCreateFlags::empty())
            .set_layouts(&descriptor_set_layouts);
        
        let compute_pipeline_layout = device
            .create_pipeline_layout(&compute_pipeline_test_layout_create_info, None)
            .unwrap();
        compute_pipeline_layout
    } else {
        let compute_pipeline_test_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&[])
            .flags(vk::PipelineLayoutCreateFlags::empty())
            .set_layouts(&descriptor_set_layouts);
        
        let compute_pipeline_layout = device
            .create_pipeline_layout(&compute_pipeline_test_layout_create_info, None)
            .unwrap();
        compute_pipeline_layout
    };
    
    crate::debug::set_object_name(compute_pipeline_layout, binder, format!("entry point '{entry_point_name}' compute pipeline layout"));
    
    let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
        .layout(compute_pipeline_layout)
        .stage(shader_stage_create_info);

    let compute_pipelines = device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[compute_pipeline_create_info],
            None,
        )
        .unwrap();
    
    crate::debug::set_object_name(compute_pipelines[0], binder, format!("entry point '{entry_point_name}' compute pipeline"));

    return SingleEntryPointWrapper { pipeline_layout: compute_pipeline_layout, pipeline: compute_pipelines[0] }
}

pub unsafe fn create_generate_voxel_compute_pipeline(
    raw: &[u32],
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> VoxelGeneratePipeline {
    let compute_shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());
    let compute_shader_module = device
        .create_shader_module(&compute_shader_module_create_info, None)
        .unwrap();
    crate::debug::set_object_name(compute_shader_module, binder, "generate voxel compute shader module");

    let descriptor_set_layout_binding_voxel_image_base_level = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let descriptor_set_layout_binding_prev_mipmapped_voxels = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let descriptor_set_layout_binding_next_mipmapped_voxels = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);

    let bindings = [
        descriptor_set_layout_binding_voxel_image_base_level,
        descriptor_set_layout_binding_prev_mipmapped_voxels,
        descriptor_set_layout_binding_next_mipmapped_voxels
    ];

    let descriptor_set_test_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&bindings);
    
    let compute_descriptor_test_set_layout = device
        .create_descriptor_set_layout(&descriptor_set_test_layout_create_info, None)
        .unwrap();

    crate::debug::set_object_name(compute_descriptor_test_set_layout, binder, "generate voxel compute descriptor set layout");

    let generate_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "main", compute_descriptor_test_set_layout, None);
    let propagate_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "propagateMipMaps", compute_descriptor_test_set_layout, None);
    
    return MultiComputePipeline {
        module: compute_shader_module,
        descriptor_set_layout: compute_descriptor_test_set_layout,
        entry_points: [generate_entry_point, propagate_entry_point]
    }
    
    /*
    let compute_shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());
    let compute_shader_module = device
        .create_shader_module(&compute_shader_module_create_info, None)
        .unwrap();
    let compute_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(c"main")
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

    let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
        .layout(compute_pipeline_layout)
        .stage(compute_stage_create_info);

    let compute_pipelines = device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[compute_pipeline_create_info],
            None,
        )
        .unwrap();

    return ComputePipeline {
        module: compute_shader_module,
        descriptor_set_layout: compute_descriptor_set_layout,
        pipeline_layout: compute_pipeline_layout,
        pipeline: compute_pipelines[0]
    }
    */
}