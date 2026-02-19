use std::{collections::VecDeque, ffi::{CStr, CString}, str::FromStr};
use vek::*;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::{buffer::{self, Buffer, create_buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}, voxel::SparseVoxelOctree};
use crate::pipeline::*;

pub struct RayTraceBuffers {
    pub ray_inputs: Buffer,
    pub ray_outputs: Buffer,
    pub ray_counts: Buffer,
    pub dispatch_size_command: Buffer,
}

impl RayTraceBuffers {
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.ray_inputs.destroy(device, allocator);
        self.ray_outputs.destroy(device, allocator);
        self.ray_counts.destroy(device, allocator);
        self.dispatch_size_command.destroy(device, allocator);
    }
}

pub unsafe fn create_ray_trace_buffers(
    device: &ash::Device,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> RayTraceBuffers {
    /*
    struct RayInput {
        float3 position;
        float3 direction;
    }

    struct RayOutput {
        bool hit;
    }
    */

    const MAX_RAYS: usize = 4096 * 64 * 32 * 2;
    
    let ray_input_struct_size = size_of::<Vec3::<f32>>() * 3;
    let ray_output_struct_size = size_of::<u32>();

    log::info!("create ray trace buffers");
    RayTraceBuffers {
        ray_inputs: create_buffer(device, allocator, ray_input_struct_size * MAX_RAYS, binder, "ray inputs buffer", vk::BufferUsageFlags::STORAGE_BUFFER),
        ray_outputs: create_buffer(device, allocator, ray_output_struct_size * MAX_RAYS, binder, "ray output buffer", vk::BufferUsageFlags::STORAGE_BUFFER),
        ray_counts: create_buffer(device, allocator, size_of::<u32>(), binder, "ray count buffer", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
        dispatch_size_command: create_buffer(device, allocator, size_of::<Vec3::<u32>>(), binder, "ray dispatch size buffer", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER),
    }
}

/*
pub type RayTracePipeline = MultiComputePipeline<2>;

pub unsafe fn create_ray_trace_compute_pipeline(
    raw: &[u32],
    device: &ash::Device,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> RayTracePipeline {
    let ray_tace_compute_shader_module_create_info = vk::ShaderModuleCreateInfo::default()
        .code(raw)
        .flags(vk::ShaderModuleCreateFlags::empty());

    let ray_trace_compute_shader_module = device
        .create_shader_module(&ray_tace_compute_shader_module_create_info, None)
        .unwrap();

    crate::debug::set_object_name(ray_trace_compute_shader_module, binder, "ray trace compute shader module");

    let svo_bitmasks_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let svo_indices_buffer = vk::DescriptorSetLayoutBinding::default()
        .binding(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let ray_inputs = vk::DescriptorSetLayoutBinding::default()
        .binding(2)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let ray_outputs = vk::DescriptorSetLayoutBinding::default()
        .binding(3)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let ray_counts = vk::DescriptorSetLayoutBinding::default()
        .binding(4)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let indirect_dispatch_command = vk::DescriptorSetLayoutBinding::default()
        .binding(5)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);
    let bindings = [
        svo_bitmasks_buffer, svo_indices_buffer, ray_inputs, ray_outputs, ray_counts, indirect_dispatch_command
    ];

    let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty())
        .bindings(&bindings);

    let descriptor_set_layout = device
        .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        .unwrap();

    crate::debug::set_object_name(descriptor_set_layout, binder, "ray trace descriptor set layout");

    let copy_dispatch_size = create_single_entry_point_pipeline(device, &binder, ray_trace_compute_shader_module, "copyDispatchSize", descriptor_set_layout, None);
    let trace_rays_recursive = create_single_entry_point_pipeline(device, &binder, ray_trace_compute_shader_module, "traceRaysRecursive", descriptor_set_layout, None);
    
    return MultiComputePipeline { module: ray_trace_compute_shader_module, descriptor_set_layout, entry_points: [copy_dispatch_size, trace_rays_recursive] };
}

pub unsafe fn trace_rays(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    allocated_descriptor_set: vk::DescriptorSet,
    ray_trace_buffers: &RayTraceBuffers,
    ray_trace_pipeline: &RayTracePipeline,
    svo: &SparseVoxelOctree,
    
    // TODO: remove, just a testing hack for now
    ray_count: u32,
    camera_position: Vec3::<f32>
) {
    // previously allocated in the frame
    let descriptor_set = allocated_descriptor_set;

    let descriptor_svo_bitmask_info = vk::DescriptorBufferInfo::default()
        .buffer(svo.bitmask_buffer.buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_svo_indices_info = vk::DescriptorBufferInfo::default()
        .buffer(svo.index_buffer.buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_ray_inputs_info = vk::DescriptorBufferInfo::default()
        .buffer(ray_trace_buffers.ray_inputs.buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_ray_outputs_info = vk::DescriptorBufferInfo::default()
        .buffer(ray_trace_buffers.ray_outputs.buffer)
        .offset(0)
        .range(u64::MAX);
    let descriptor_ray_count_info = vk::DescriptorBufferInfo::default()
        .buffer(ray_trace_buffers.ray_counts.buffer)
        .offset(0)
        .range(u64::MAX);

    let buffer_bindings = [descriptor_svo_bitmask_info, descriptor_svo_indices_info, descriptor_ray_inputs_info, descriptor_ray_outputs_info, descriptor_ray_count_info];
    
    let descriptor_write = vk::WriteDescriptorSet::default()
        .descriptor_count(buffer_bindings.len() as u32)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .dst_binding(0)
        .dst_set(descriptor_set)
        .buffer_info(&buffer_bindings);
    device.update_descriptor_sets(&[descriptor_write], &[]);

    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        ray_trace_pipeline.entry_points[1].pipeline_layout,
        0,
        &[descriptor_set],
        &[],
    );
    device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        ray_trace_pipeline.entry_points[1].pipeline,
    );
    
    /*
    // FIXME: for some fucking reason push constants make it shit itself
    // literally just calling vkCmdPushConstants changes behaviour. wtf
    // HACK!! This is just to speed up the tracing of primary rays since the rays always start at the camera position
    let mut push_constants = [vek::Vec4::<f32>::from_point(camera_position)];
    let raw = bytemuck::bytes_of_mut(&mut push_constants);

    device.cmd_push_constants(
        cmd,
        ray_trace_pipeline.entry_points[1].pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &raw,
    );
    */

    // write number of rays created, which is the total number of pixels on the screen
    let data = [ray_count as u32];
    let raw = bytemuck::cast_slice::<u32, u8>(&data);
    device.cmd_update_buffer(cmd, ray_trace_buffers.ray_counts.buffer, 0, raw);

    // dispatch the required number of rays
    let dispatch_size = ray_count.div_ceil(64) as u32;
    device.cmd_dispatch(cmd, dispatch_size, 1, 1);
}
*/