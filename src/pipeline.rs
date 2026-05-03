use std::{ffi::CString, str::FromStr};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use smallvec::SmallVec;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstants {
    pub screen_resolution: vek::Vec2<f32>,
    pub _padding: vek::Vec2<f32>,
    pub matrix: vek::Mat4<f32>,
    pub position: vek::Vec4<f32>,
    pub sun: vek::Vec4<f32>,
    pub debug_type: u32,
    pub time: f32,
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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SkyComputePushConstants {
    pub sun: vek::Vec4<f32>,
}

pub struct SingleEntryPointWrapper {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

pub struct SpecConstant<'a> {
    pub bytes: &'a [u8],
}

pub struct MultiComputePipeline<const ENTRY_POINTS: usize, const DESCRIPTOR_SETS: usize> {
    pub module: vk::ShaderModule,
    pub entry_points: [SingleEntryPointWrapper; ENTRY_POINTS],
    pub descriptor_set_layout: [vk::DescriptorSetLayout; DESCRIPTOR_SETS],
}

impl<const ENTRY_POINTS: usize, const DESCRIPTOR_SETS: usize> MultiComputePipeline<ENTRY_POINTS, DESCRIPTOR_SETS> {
    pub unsafe fn destroy(self, device: &ash::Device) {
        for single_entry_point_wrapper in self.entry_points {
            device.destroy_pipeline(single_entry_point_wrapper.pipeline, None);
            device.destroy_pipeline_layout(single_entry_point_wrapper.pipeline_layout, None);
        }
        
        for descriptor_set_layout in self.descriptor_set_layout {
            device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        }
        device.destroy_shader_module(self.module, None);
    }
}

pub type RenderPipeline = MultiComputePipeline<1, 2>;
pub type SkyPipeline = MultiComputePipeline<2, 1>;

#[derive(Pod, Zeroable, Copy, Clone)]
#[repr(C)]
pub struct RenderPipelineSpecConstants {
    pub shadow_samples: u32,
    pub max_ray_iterations: u32,
    pub round_normals: u32,
    pub ambient_occlusion: u32,
    pub wavy_reflections: u32,
    pub pixelated_shadows: u32,
    pub group_size: u32,
}

pub unsafe fn create_render_compute_pipeline(
    raw: &[u32],
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
    constants: RenderPipelineSpecConstants,
) -> RenderPipeline {
    let render_compute_shader_module = create_shader_module(raw, device, binder, "render compute shader module");

    let output = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let bindings = [output];
    let first_descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&bindings);
    let first_descriptor_set_layout = device
        .create_descriptor_set_layout(&first_descriptor_set_layout_create_info, None)
        .unwrap();
    crate::debug::set_object_name(first_descriptor_set_layout, binder, "render compute descriptor set layout 1 (per frame dst rt)");

    let svt_image = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let svt_meta_image = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let svo_bitmasks = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let svo_indices = vk::DescriptorSetLayoutBinding::default()
        .binding(3)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let svo_aabbs = vk::DescriptorSetLayoutBinding::default()
        .binding(4)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let lights_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(5)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let skybox_sampler = vk::DescriptorSetLayoutBinding::default()
        .binding(6)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1);
    let clouds_sampler = vk::DescriptorSetLayoutBinding::default()
        .binding(7)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1);
    let bindings = [
        svt_image,
        svt_meta_image,
        svo_bitmasks,
        svo_indices,
        svo_aabbs,
        lights_buffer,
        skybox_sampler,
        clouds_sampler,
    ];
    let second_descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&bindings);
    let second_descriptor_set_layout = device
        .create_descriptor_set_layout(&second_descriptor_set_layout_create_info, None)
        .unwrap();
    crate::debug::set_object_name(second_descriptor_set_layout, binder, "render compute descriptor set layout 2 (constant stuff)");

    let render_compute_descriptor_set_layouts = [first_descriptor_set_layout, second_descriptor_set_layout];

    let push_constant_size = Some(size_of::<PushConstants>());

    // FIXME: this assumes that spec constant fields are ALL u32s
    let spec_constant_bytes = bytemuck::cast_slice::<u8, u32>(bytemuck::bytes_of(&constants));
    let spec_constants = spec_constant_bytes.into_iter().map(|x| SpecConstant { bytes: bytemuck::bytes_of(x) }).collect::<Vec<_>>();

    let main_entry_point = create_single_entry_point_pipeline(device, &binder, render_compute_shader_module, "main", &render_compute_descriptor_set_layouts, push_constant_size, Some(&spec_constants));
    
    return MultiComputePipeline {
        module: render_compute_shader_module,
        descriptor_set_layout: render_compute_descriptor_set_layouts,
        entry_points: [main_entry_point]
    }
}

pub unsafe fn create_sky_pipeline(
    raw: &[u32],
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> SkyPipeline {
    let shader_module = create_shader_module(raw, device, binder, "sky compute shader module");

    let skybox = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let clouds = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1);
    let clouds_sampler = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1);
    let bindings = [
        skybox,
        clouds,
        clouds_sampler
    ];

    let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&bindings);
    let descriptor_set_layout = device
        .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        .unwrap();
    crate::debug::set_object_name(descriptor_set_layout, binder, "sky compute descriptor set layout");

    let render_compute_descriptor_set_layouts = [descriptor_set_layout];

    let size = size_of::<SkyComputePushConstants>();


    let spec_constant_fields = [crate::skybox::SKYBOX_RESOLUTION, crate::skybox::CLOUDS_RESOLUTION];
    let spec_constants = spec_constant_fields.iter().map(|x| SpecConstant { bytes: bytemuck::bytes_of(x) }).collect::<Vec<_>>();


    let clouds_entry_point = create_single_entry_point_pipeline(device, &binder, shader_module, "write_clouds", &render_compute_descriptor_set_layouts, Some(size), Some(&spec_constants));
    let skybox_entry_point = create_single_entry_point_pipeline(device, &binder, shader_module, "write_skybox", &render_compute_descriptor_set_layouts, Some(size), Some(&spec_constants));
    
    return MultiComputePipeline {
        module: shader_module,
        descriptor_set_layout: render_compute_descriptor_set_layouts,
        entry_points: [clouds_entry_point, skybox_entry_point],
    }
}

unsafe fn create_shader_module(raw: &[u32], device: &ash::Device, binder: &Option<ash::ext::debug_utils::Device>, name: &str) -> vk::ShaderModule {
    log::debug!("creating shader shader module '{name}'");
    let shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());

    let shader_module = device
        .create_shader_module(&shader_module_create_info, None)
        .unwrap();
    crate::debug::set_object_name(shader_module, binder, name);
    log::debug!("created shader shader module '{name}'");
    shader_module
}

/*
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
    let descriptor_set_layouts = [compute_descriptor_test_set_layout];


    let push_constant_size = Some(size_of::<PushConstants2>());
    let unwrap_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "unwrap", &descriptor_set_layouts, push_constant_size, None);
    let unpack_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "unpack", &descriptor_set_layouts, push_constant_size, None);
    let tick_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "main", &descriptor_set_layouts, push_constant_size, None);
    let copy_dispatch_params_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "copyDispatchSize", &descriptor_set_layouts, push_constant_size, None);
    
    return MultiComputePipeline {
        module: compute_shader_module,
        descriptor_set_layout: compute_descriptor_test_set_layout,
        entry_points: [unwrap_entry_point, unpack_entry_point, tick_entry_point, copy_dispatch_params_entry_point]
    }
}
*/

pub unsafe fn create_single_entry_point_pipeline(
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
    compute_shader_module: vk::ShaderModule,
    entry_point_name: &str,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    push_constant_size: Option<usize>,
    spec_constants: Option<&[SpecConstant]>
) -> SingleEntryPointWrapper {
    let string = CString::from_str(entry_point_name).unwrap();

    let mut specialization_entries = Vec::<vk::SpecializationMapEntry>::new();

    let mut data = Vec::<u8>::new();
    if let Some(spec_constants) = spec_constants {
        let mut last_offset = 0u32;
        for (i, spec) in spec_constants.iter().enumerate() {
            specialization_entries.push(vk::SpecializationMapEntry::default()
                .constant_id(i as u32)
                .offset(last_offset)
                .size(spec.bytes.len())
            );
            data.extend_from_slice(spec.bytes);
            last_offset += spec.bytes.len() as u32;
        }
    }

    let specialization_info = vk::SpecializationInfo::default()
        .map_entries(&specialization_entries)
        .data(&data);

    log::info!("creating single entry point pipline for {entry_point_name}. pc range: {push_constant_size:?}");
    let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
        .flags(vk::PipelineShaderStageCreateFlags::empty())
        .name(string.as_c_str())
        .stage(vk::ShaderStageFlags::COMPUTE)
        .specialization_info(&specialization_info)
        .module(compute_shader_module);
        
    let mut push_constant_ranges = SmallVec::<[vk::PushConstantRange;1]>::new();
    if let Some(push_constant_size) = push_constant_size {
        let push_constant_range = vk::PushConstantRange::default()
            .offset(0)
            .size(push_constant_size as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        push_constant_ranges.push(push_constant_range);
    };

    let compute_pipeline_test_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(&push_constant_ranges.as_slice())
        .flags(vk::PipelineLayoutCreateFlags::empty())
        .set_layouts(&descriptor_set_layouts);
    
    let compute_pipeline_layout = device
        .create_pipeline_layout(&compute_pipeline_test_layout_create_info, None)
        .unwrap();

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
/*
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

    let generate_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "main", compute_descriptor_test_set_layout, None, None);
    let propagate_entry_point = create_single_entry_point_pipeline(device, &binder, compute_shader_module, "propagateMipMaps", compute_descriptor_test_set_layout, None, None);
    
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
*/