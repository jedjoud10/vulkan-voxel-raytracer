use ash;
use ash::vk;
use clap::Parser;
use gpu_allocator::vulkan::Allocation;
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

pub struct PerFrameDescriptorSets {
    pub main_render_per_frame: vk::DescriptorSet,
}

pub struct PerFrameData {
    pub swapchain_image: vk::Image,
    pub rt_image: vk::Image,
    pub rt_image_allocation: Option<Allocation>,
    pub present_complete_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub end_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,
    pub per_frame_descriptor_sets: PerFrameDescriptorSets,
    pub src_image_view: vk::ImageView,
    pub dst_image_view: vk::ImageView,
}

impl PerFrameData {
    pub unsafe fn create_per_frame_data(
        device: &ash::Device,
        pool: vk::CommandPool,
        swapchain_format: vk::Format,
        descriptor_pool: vk::DescriptorPool,
        render_compute_pipeline: &pipeline::RenderPipeline,
        rt_image: vk::Image,
        rt_image_allocation: Allocation,
        swapchain_image: vk::Image
    ) -> Self {
        let present_complete_semaphore = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let render_finished_semaphore = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let end_fence = device.create_fence(&Default::default(), None).unwrap();
        log::info!("created semaphores and fence");

        let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(pool);
        let cmd = device
            .allocate_command_buffers(&cmd_buffer_create_info)
            .unwrap()[0];

        let per_frame_descriptor_set_layouts = [render_compute_pipeline.descriptor_set_layout[0]];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&per_frame_descriptor_set_layouts);
        let all_descriptor_sets_for_frame = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();

        let mut tmp = Self {
            swapchain_image,
            rt_image,
            rt_image_allocation: None,
            present_complete_semaphore,
            render_finished_semaphore,
            end_fence,
            cmd,
            per_frame_descriptor_sets: PerFrameDescriptorSets {
                main_render_per_frame: all_descriptor_sets_for_frame[0],
            },
            src_image_view: vk::ImageView::null(),
            dst_image_view: vk::ImageView::null(),
        };

        tmp.recreate_image_views_and_update_descriptor_sets(device, swapchain_format, rt_image, rt_image_allocation, swapchain_image);

        tmp
    }

    pub unsafe fn recreate_image_views_and_update_descriptor_sets(
        &mut self,
        device: &ash::Device,
        swapchain_format: vk::Format,
        rt_image: vk::Image,
        rt_image_allocation: Allocation,
        swapchain_image: vk::Image,
    ) {
        self.rt_image = rt_image;
        self.rt_image_allocation = Some(rt_image_allocation);
        self.swapchain_image = swapchain_image;

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        let src_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(swapchain_format)
            .image(rt_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let dst_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(swapchain_format)
            .image(self.swapchain_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        self.src_image_view = device
            .create_image_view(&src_image_view_create_info, None)
            .unwrap();
        self.dst_image_view = device
            .create_image_view(&dst_image_view_create_info, None)
            .unwrap();

        let descriptor_rt_image_info = vk::DescriptorImageInfo::default()
            .image_view(self.src_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_image_infos = [descriptor_rt_image_info];

        let image_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(descriptor_image_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(self.per_frame_descriptor_sets.main_render_per_frame)
            .image_info(&descriptor_image_infos);
        
        device.update_descriptor_sets(&[image_descriptor_write], &[]);

    }
}