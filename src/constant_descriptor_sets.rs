use ash;
use ash::vk;
use clap::Parser;
use crate::input::Button;
use crate::input::Input;
use crate::movement::Movement;
use winit::event::MouseButton;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::KeyCode;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{Window, WindowId};
use crate::statistics::Statistics;

use crate::swapchain;
use crate::statistics;
use crate::ticker;
use crate::input;
use crate::pipeline;
use crate::skybox;
use crate::voxel;
use crate::buffer;
use crate::instance;
use crate::physical_device;
use crate::device;
use crate::movement;
use crate::debug;
use crate::others;

pub struct ConstantDescriptorSets {
    pub render_compute_pipeline_descriptor_set: vk::DescriptorSet,
    pub sky_compute_pipeline_descriptor_set: vk::DescriptorSet,
}

impl ConstantDescriptorSets {
    pub unsafe fn create_constant_descriptor_sets(
        device: &ash::Device,
        pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        render_compute_pipeline: &pipeline::RenderPipeline,
        sky_compute_pipeline: &pipeline::SkyPipeline,
        skybox: &skybox::Skybox,
        svt: &voxel::SparseVoxelTexture,
        svo: &voxel::SparseVoxelOctree,
        lights_buffer: &buffer::Buffer,
    ) -> Self {
        let constant_descriptor_set_layouts = [render_compute_pipeline.descriptor_set_layout[1], sky_compute_pipeline.descriptor_set_layout[0]];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&constant_descriptor_set_layouts);
        let all_descriptor_sets = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();
        let render_compute_pipeline_descriptor_set= all_descriptor_sets[0];
        let sky_compute_pipeline_descriptor_set = all_descriptor_sets[1];


        let descriptor_skybox_image_info = vk::DescriptorImageInfo::default()
            .image_view(skybox.skybox_array_image_view)
            .sampler(vk::Sampler::null())
            .image_layout(vk::ImageLayout::GENERAL);
        let descriptor_clouds_image_info = vk::DescriptorImageInfo::default()
            .image_view(skybox.clouds_image_view)
            .sampler(vk::Sampler::null())
            .image_layout(vk::ImageLayout::GENERAL);

        let descriptor_image_infos_1 = [descriptor_skybox_image_info, descriptor_clouds_image_info];

        let sky_compute_descriptor_write_1 = vk::WriteDescriptorSet::default()
            .descriptor_count(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(sky_compute_pipeline_descriptor_set)
            .image_info(&descriptor_image_infos_1);

        let descriptor_svt_image_info = vk::DescriptorImageInfo::default()
            .image_view(svt.sparse_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_svt_metadata_image_info = vk::DescriptorImageInfo::default()
            .image_view(svt.metadata_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_svo_bitmasks_info = vk::DescriptorBufferInfo::default()
            .buffer(svo.bitmask_buffer.buffer)
            .offset(0)
            .range(u64::MAX);
        let descriptor_svo_indices_info = vk::DescriptorBufferInfo::default()
            .buffer(svo.index_buffer.buffer)
            .offset(0)
            .range(u64::MAX);
        let descriptor_svo_aabbs_info = vk::DescriptorBufferInfo::default()
            .buffer(svo.aabb_buffer.buffer)
            .offset(0)
            .range(u64::MAX);
        let descriptor_light_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(lights_buffer.buffer)
            .offset(0)
            .range(u64::MAX);
        let descriptor_skybox_sampler_info = vk::DescriptorImageInfo::default()
            .image_view(skybox.skybox_image_view)
            .sampler(skybox.sampler)
            .image_layout(vk::ImageLayout::GENERAL);
        let descriptor_clouds_sampler_info = vk::DescriptorImageInfo::default()
            .image_view(skybox.clouds_image_view)
            .sampler(skybox.sampler)
            .image_layout(vk::ImageLayout::GENERAL);

 
        let descriptor_svt_image_infos = [descriptor_svt_image_info, descriptor_svt_metadata_image_info];
        let descriptor_svo_buffers_infos = [descriptor_svo_bitmasks_info, descriptor_svo_indices_info, descriptor_svo_aabbs_info, descriptor_light_buffer_info];
        let descriptor_skybox_combined_image_sampler_infos = [descriptor_skybox_sampler_info, descriptor_clouds_sampler_info];

        let render_compute_svt_images_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(descriptor_svt_image_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(render_compute_pipeline_descriptor_set)
            .image_info(&descriptor_svt_image_infos);
        let render_compute_svo_buffers_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(descriptor_svo_buffers_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_binding(2)
            .dst_set(render_compute_pipeline_descriptor_set)
            .buffer_info(&descriptor_svo_buffers_infos);
        let render_compute_skybox_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(descriptor_skybox_combined_image_sampler_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_binding(6)
            .dst_set(render_compute_pipeline_descriptor_set)
            .image_info(&descriptor_skybox_combined_image_sampler_infos);
        
        device.update_descriptor_sets(&[sky_compute_descriptor_write_1, render_compute_svt_images_descriptor_write, render_compute_svo_buffers_descriptor_write, render_compute_skybox_descriptor_write], &[]);

        Self {
            render_compute_pipeline_descriptor_set,
            sky_compute_pipeline_descriptor_set,
        }
    }
}
