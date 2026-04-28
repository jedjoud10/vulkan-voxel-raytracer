#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_imports)]

mod assets;
mod debug;
mod query;
use assets::damn;
mod device;
mod input;
mod instance;
mod movement;
mod physical_device;
mod pipeline;
mod pool;
mod queue;
mod surface;
mod swapchain;
mod voxel;
mod ticker;
mod buffer;
mod rays;
mod statistics;
mod utils;

use ash;
use ash::vk;
use clap::Parser;
use gpu_allocator::vulkan::Allocation;
use input::Button;
use input::Input;
use movement::Movement;
use pipeline::PushConstants2;
use winit::event::MouseButton;
use std::collections::HashMap;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::KeyCode;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{Window, WindowId};
use statistics::Statistics;

use crate::buffer::*;
use crate::pipeline::*;
use crate::rays::*;
use crate::voxel::*;


#[derive(clap::Parser, Debug)]
#[command(about = "Vulkan DDA Voxel Raytracer", long_about = None)]
struct Args {
    /// Factor to use to decrease the screen resolution
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=4))]
    resolution_scaling_factor: u32,

    /// Number of shadow samples to use. Set to 0 to disable shadows completely. Set to 1 to use hard-shadows.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=16))]
    shadow_samples: u32,

    /// Maximum number of rays to trace iteratively for reflections / refractions
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=8))]
    max_ray_iterations: u32,

    /// Whether or not to use round spherical normals
    #[arg(long, default_value_t = false)]
    round_normals: bool,

    /// Whether or not to use ray traced ambient occlusion
    #[arg(long, default_value_t = false)]
    ambient_occlusion: bool,

    /// Fun setting to make all mirror reflections wavey lolol
    #[arg(long, default_value_t = false)]
    wavy_reflections: bool,

    /// Setting to make all shadows pixelated
    #[arg(long, default_value_t = false)]
    pixelated_shadows: bool,

    /// Setting to start in fullscreen from the start. This can be toggled in-game using F5
    #[arg(long, default_value_t = false)]
    fullscreen: bool,
}

struct PerFrameData {
    swapchain_image: vk::Image,
    rt_image: vk::Image,
    rt_image_allocation: Option<Allocation>,
    present_complete_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    end_fence: vk::Fence,
    cmd: vk::CommandBuffer,
    all_descriptor_sets_for_frame: Vec<vk::DescriptorSet>,
    src_image_view: vk::ImageView,
    dst_image_view: vk::ImageView,
}

struct InternalApp {
    frame_count: u64,

    input: Input,
    movement: Movement,

    window: Window,
    entry: ash::Entry,
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    
    debug: Option<(
        ash::ext::debug_utils::Instance,
        vk::DebugUtilsMessengerEXT
    )>,
    debug_marker: Option<ash::ext::debug_utils::Device>,
    surface_loader: ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    swapchain_format: vk::Format,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,

    frames_in_flight: Vec<PerFrameData>,

    queue: vk::Queue,
    queue_family_index: u32,
    pool: vk::CommandPool,

    render_compute_pipeline: RenderPipeline,

    query_pool: vk::QueryPool,
    timestamp_period: f32,

    descriptor_pool: vk::DescriptorPool,
    allocator: gpu_allocator::vulkan::Allocator,
    
    svo: SparseVoxelOctree,

    ticker: ticker::Ticker,
    sun: vek::Vec3<f32>,
    debug_type: u32,
    args: Args,
    stats: Statistics,
}

impl InternalApp {
    pub unsafe fn new(event_loop: &ActiveEventLoop, args: Args) -> Self {
        let mut assets = HashMap::<&str, Vec<u8>>::new();
        asset!("raymarcher.spv", assets);
        asset!("voxel_tick.spv", assets);
        asset!("voxel_generate.spv", assets);

        // FIXME: ugly but works fuck it
        let assets: HashMap<&str, &[u32]> = HashMap::from_iter(assets.iter().map(|(a, b)| (*a, bytemuck::cast_slice::<u8, u32>(&b))));

        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        if args.fullscreen {
            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }

        window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .unwrap();
        window.set_cursor_visible(false);
        let raw_display_handle = window.display_handle().unwrap().as_raw();
        let entry = ash::Entry::load().unwrap();

        let instance = instance::create_instance(&entry, raw_display_handle);
        log::info!("created instance");
        let debug_messenger = debug::create_debug_messenger(&entry, &instance).map(|x| {
            log::info!("created debug utils messenger");
            x
        });

        let (surface_loader, surface_khr) = surface::create_surface(&instance, &entry, &window);
        log::info!("created surface");

        let mut physical_device_candidates = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .map(|physical_device| {
                let score = physical_device::get_physical_device_score(
                    physical_device,
                    &instance,
                    &surface_loader,
                    surface_khr,
                );
                (physical_device, score)
            })
            .filter_map(|(a, b)| b.map(|val| (a, val)))
            .collect::<Vec<(vk::PhysicalDevice, u32)>>();
        physical_device_candidates.sort_by_key(|(_, score)| *score);

        if physical_device_candidates.is_empty() {
            log::error!("no physical device was chosen!");
            panic!();
        }

        let physical_device = physical_device_candidates.last().unwrap().0;
        let mut physical_device_properties = vk::PhysicalDeviceProperties2::default();
        instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);
        let physical_device_name = physical_device_properties.properties.device_name_as_c_str().unwrap().to_str().unwrap();

        log::info!("selected physical device \"{}\"", physical_device_name);

        let (device, queue_family_index, queue) = device::create_device_and_queue(
            &instance,
            physical_device,
            &surface_loader,
            surface_khr,
        );
        let queue_family_indices = [queue_family_index];
        log::info!("created device and fetched main queue");

        let debug_marker = debug_messenger.is_some().then(|| {
            let device = debug::create_debug_marker(&instance, &device);
            log::info!("created debug marker object names binder");
            device
        });

        let mut allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: physical_device.clone(),
                debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                buffer_device_address: false,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            })
            .unwrap();
        log::info!("created gpu allocator");

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = device.create_command_pool(&pool_create_info, None).unwrap();
        log::info!("create cmd pool");

        let mut extent = vk::Extent2D {
            width: 800,
            height: 600,
        };

        if args.fullscreen {
            extent = vk::Extent2D {
                width: window.inner_size().width,
                height: window.inner_size().height,
            }
        }

        let (swapchain_loader, swapchain, images, swapchain_format) = swapchain::create_swapchain(
            &instance,
            &surface_loader,
            surface_khr,
            physical_device,
            &device,
            extent,
            &debug_marker,
        );
        log::info!("created swapchain with {} in-flight images", images.len());

        let rt_images: Vec<(vk::Image, Allocation)> = (0..images.len())
            .into_iter()
            .map(|_| {
                swapchain::create_temporary_target_render_image(
                    &instance,
                    &surface_loader,
                    surface_khr,
                    physical_device,
                    &device,
                    &mut allocator,
                    queue_family_index,
                    extent,
                    &debug_marker,
                    args.resolution_scaling_factor,
                    c"temporary render target image"
                )
            })
            .collect();
        log::info!("created {} in-flight render texture images", images.len());

        swapchain::transfer_rt_images(&device, queue_family_index, &rt_images, pool, queue);
        log::info!("transferred layout of render texture images");

        let descriptor_pool = pool::create_descriptor_pool(&device);
        log::info!("created descriptor pool");

        let spec_constants = RenderPipelineSpecConstants {
            shadow_samples: args.shadow_samples,
            max_ray_iterations: args.max_ray_iterations,
            round_normals: if args.round_normals { 1 } else { 0 },
            ambient_occlusion: if args.ambient_occlusion { 1 } else { 0 }, 
            wavy_reflections: if args.wavy_reflections { 1 } else { 0 }, 
            pixelated_shadows: if args.pixelated_shadows { 1 } else { 0 }, 
        };
        let render_compute_pipeline = pipeline::create_render_compute_pipeline(&*assets["raymarcher.spv"], &device, &debug_marker, spec_constants);
        log::info!("created render compute pipeline");

        let svo = voxel::create_sparse_voxel_octree(
            &device,
            &mut allocator,
            &debug_marker,
            queue,
            pool,
            descriptor_pool,
            queue_family_index
        );
        log::info!("created sparse voxel octree buffers");

        log::info!("creating frames in flight structures...");
        let mut frames_in_flight = Vec::<PerFrameData>::new();
        for ((rt_image, rt_image_allocation), swapchain_image) in rt_images.into_iter().zip(images.into_iter()) {
            let begin_semaphore = device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap();
            let end_semaphore = device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap();
            let end_fence = device.create_fence(&Default::default(), None).unwrap();
            log::info!("created semaphores and fence");

            
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
                .image(swapchain_image)
                .subresource_range(subresource_range)
                .view_type(vk::ImageViewType::TYPE_2D);

            let src_image_view = device
                .create_image_view(&src_image_view_create_info, None)
                .unwrap();
            let dst_image_view = device
                .create_image_view(&dst_image_view_create_info, None)
                .unwrap();

            let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(pool);
            let cmd = device
                .allocate_command_buffers(&cmd_buffer_create_info)
                .unwrap()[0];            

            let layouts = [render_compute_pipeline.descriptor_set_layout];
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);
            let all_descriptor_sets_for_frame = device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .unwrap();

            frames_in_flight.push(PerFrameData {
                swapchain_image,
                rt_image,
                rt_image_allocation: Some(rt_image_allocation),
                present_complete_semaphore: begin_semaphore,
                render_finished_semaphore: end_semaphore,
                end_fence,
                cmd,
                all_descriptor_sets_for_frame,
                src_image_view,
                dst_image_view,
                
            });
        }

        let query_pool = query::create_query_pool(&device);
        let timestamp_period = physical_device_properties.properties.limits.timestamp_period;

        Self {
            frame_count: 0,
            input: Default::default(),
            movement: Movement::new(),
            window,
            instance,
            entry,
            device,
            physical_device,
            surface_loader,
            surface_khr,
            debug: debug_messenger,
            debug_marker,
            swapchain_loader,
            swapchain_format,
            swapchain,
            queue_family_index,
            queue,
            pool,
            render_compute_pipeline,
            descriptor_pool,
            query_pool,
            timestamp_period,
            allocator,
            svo,
            frames_in_flight,
            ticker: ticker::Ticker { accumulator: 0f32, count: 0 },
            sun: vek::Vec3::new(1f32, 0.3f32,0.5f32).normalized(),
            debug_type: 0,
            stats: Default::default(),
            args,
        }
    }

    pub unsafe fn click(&mut self, add: bool) {
        let position = (self.movement.forward() * 5.0f32 + self.movement.position).floor().as_::<u32>();

        self.svo.set(position, add);
        self.svo.rebuild(&self.device, self.pool, self.queue, &mut self.allocator);
    }

    pub unsafe fn resize(&mut self, width: u32, height: u32) {
        log::warn!("resizing! width: {width}, height: {height}");
        self.device.device_wait_idle().unwrap();

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);

        for frame in self.frames_in_flight.iter_mut() {
            self.device.destroy_image_view(frame.dst_image_view, None);
            self.device.destroy_image_view(frame.src_image_view, None);

            self.device.destroy_image(frame.rt_image, None);
            self.allocator.free(frame.rt_image_allocation.take().unwrap()).unwrap();
        }

        let extent = vk::Extent2D { width, height };

        let (swapchain_loader, swapchain, images, swapchain_format) = swapchain::create_swapchain(
            &self.instance,
            &self.surface_loader,
            self.surface_khr,
            self.physical_device,
            &self.device,
            extent,
            &self.debug_marker,
        );

        for (frame, swapchain_image) in self.frames_in_flight.iter_mut().zip(images) {
            frame.swapchain_image = swapchain_image;
        }

        self.swapchain_loader = swapchain_loader;
        self.swapchain_format = swapchain_format;
        self.swapchain = swapchain;

        let rt_images: Vec<(vk::Image, Allocation)> = (0..self.frames_in_flight.len())
            .into_iter()
            .map(|_| {
                swapchain::create_temporary_target_render_image(
                    &self.instance,
                    &self.surface_loader,
                    self.surface_khr,
                    self.physical_device,
                    &self.device,
                    &mut self.allocator,
                    self.queue_family_index,
                    extent,
                    &self.debug_marker,
                    self.args.resolution_scaling_factor,
                    c"temporary render target image"
                )
            })
            .collect();
        swapchain::transfer_rt_images(
            &self.device,
            self.queue_family_index,
            &rt_images,
            self.pool,
            self.queue,
        );

        for (frame, (rt_image, rt_image_allocation)) in self.frames_in_flight.iter_mut().zip(rt_images) {
            frame.rt_image = rt_image;
            frame.rt_image_allocation = Some(rt_image_allocation);

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
                .image(frame.swapchain_image)
                .subresource_range(subresource_range)
                .view_type(vk::ImageViewType::TYPE_2D);

            frame.src_image_view = self.device
                .create_image_view(&src_image_view_create_info, None)
                .unwrap();
            frame.dst_image_view = self.device
                .create_image_view(&dst_image_view_create_info, None)
                .unwrap();

        }

        self.device.device_wait_idle().unwrap();
    }

    pub unsafe fn render(&mut self, delta: f32, elapsed: f32) {
        let frame_index = self.frame_count % (self.frames_in_flight.len() as u64);
        let PerFrameData {
            swapchain_image,
            rt_image,
            rt_image_allocation,
            present_complete_semaphore,
            end_fence,
            cmd,
            all_descriptor_sets_for_frame,
            src_image_view,
            dst_image_view,
            ..
        } = &self.frames_in_flight[frame_index as usize];

        let cmd = *cmd;
        let swapchain_image = *swapchain_image;
        let rt_image = *rt_image;
        let present_complete_semaphores = [*present_complete_semaphore];
        let end_fence = *end_fence;
        let src_image_view = *src_image_view;
        let dst_image_view = *dst_image_view;
        let all_descriptor_sets_for_frame = all_descriptor_sets_for_frame;
        let render_descriptor_set = all_descriptor_sets_for_frame[0];

        
        if let Err(err) = self.device.wait_for_fences(&[end_fence], true, u64::MAX) {
            log::error!("wait on fence err: {:?}", err);
        }

        self.device.reset_fences(&[end_fence]).unwrap();

        let (acquired_swapchain_image_index, _) = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain,
                u64::MAX,
                *present_complete_semaphore,
                vk::Fence::null(),
            )
            .unwrap();

        let render_finished_semaphore = [self.frames_in_flight[acquired_swapchain_image_index as usize].render_finished_semaphore];
        
        let dst_image = swapchain_image;
        let src_image= rt_image;

        let cmd_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::empty());
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device
            .begin_command_buffer(cmd, &cmd_buffer_begin_info)
            .unwrap();
        self.device.cmd_reset_query_pool(cmd, self.query_pool, 0, 2);

        //self.sun = vek::Vec3::new((elapsed * 0.1f32).sin(), (elapsed * 0.05).sin(), (elapsed * 0.1f32).cos()).normalized();

        let push_constants = PushConstants2 {
            forward: vek::Mat4::from(self.movement.rotation).mul_direction(-vek::Vec3::unit_z()).with_w(0.0f32),
            position: self.movement.position.with_w(0.0f32),
            sun: self.sun.normalized().with_w(0f32),
            tick: self.ticker.count,
            delta: delta.max(1f32 / ticker::TICKS_PER_SECOND),
        };


        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        let dst_undefined_to_blit_dst_layout_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(dst_image)
            .subresource_range(subresource_range);
        let image_memory_barriers = [dst_undefined_to_blit_dst_layout_transition];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        /*
        self.device.cmd_clear_color_image(cmd, dst_image, vk::ImageLayout::GENERAL, &vk::ClearColorValue {
            float32: [elapsed.sin() * 0.5 + 0.5; 4]
            }, &[subresource_range]);
        */



        let descriptor_rt_image_info = vk::DescriptorImageInfo::default()
            .image_view(src_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_svo_bitmasks_info = vk::DescriptorBufferInfo::default()
            .buffer(self.svo.bitmask_buffer.buffer)
            .offset(0)
            .range(u64::MAX);
        let descriptor_svo_indices_info = vk::DescriptorBufferInfo::default()
            .buffer(self.svo.index_buffer.buffer)
            .offset(0)
            .range(u64::MAX);

        let descriptor_rt_image_infos = [descriptor_rt_image_info];
        let descriptor_svo_infos = [descriptor_svo_bitmasks_info, descriptor_svo_indices_info];

        let image_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(render_descriptor_set)
            .image_info(&descriptor_rt_image_infos);
        let buffer_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(descriptor_svo_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_binding(1)
            .dst_set(render_descriptor_set)
            .buffer_info(&descriptor_svo_infos);
        

        self.device.update_descriptor_sets(&[image_descriptor_write, buffer_descriptor_write], &[]);
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.render_compute_pipeline.entry_points[0].pipeline_layout,
            0,
            &[render_descriptor_set],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.render_compute_pipeline.entry_points[0].pipeline,
        );

        let size = self.window.inner_size();
        let size = vek::Vec2::<u32>::new(size.width, size.height)
            .map(|val| val / self.args.resolution_scaling_factor);

        let group_size = 8 as f32;
        let width_group_size = (size.x as f32 / group_size).ceil() as u32;
        let height_group_size = (size.y as f32 / group_size).ceil() as u32;
        let size_f32 = size.map(|x| x as f32);

        let push_constants = pipeline::PushConstants {
            screen_resolution: size_f32,
            _padding: Default::default(),
            matrix: self.movement.proj_matrix * self.movement.view_matrix,
            position: self.movement.position.with_w(0f32),
            sun: self.sun.normalized().with_w(0f32),
            debug_type: self.debug_type as u32,
            time: elapsed,
        };

        let raw = bytemuck::bytes_of(&push_constants);

        self.device.cmd_push_constants(
            cmd,
            self.render_compute_pipeline.entry_points[0].pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            raw,
        );

        self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, self.query_pool, 0);

        self.device.cmd_dispatch(cmd, width_group_size, height_group_size, 1);

        self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, self.query_pool, 1);

        let src_shader_write_to_transfer_src = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(src_image)
            .subresource_range(subresource_range);
        let image_memory_barriers = [src_shader_write_to_transfer_src];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        let origin_offset = vk::Offset3D::default();
        let src_extent_offset = vk::Offset3D::default()
            .x(self.window.inner_size().width as i32 / self.args.resolution_scaling_factor as i32)
            .y(self.window.inner_size().height as i32 / self.args.resolution_scaling_factor as i32)
            .z(1);
        let dst_extent_offset = vk::Offset3D::default()
            .x(self.window.inner_size().width as i32)
            .y(self.window.inner_size().height as i32)
            .z(1);
        let src_offsets = [origin_offset, src_extent_offset];
        let dst_offsets = [origin_offset, dst_extent_offset];

        let subresource_layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(0);

        let image_blit = vk::ImageBlit::default()
            .src_offsets(src_offsets)
            .src_subresource(subresource_layers)
            .dst_offsets(dst_offsets)
            .dst_subresource(subresource_layers);

        // TODO: implement fast path that does not even allocate RT and just renders to the swapchain image directly
        // replacing this blit by a copy does not improve performance much, but removing it completely definitely should
        let regions = [image_blit];
        self.device.cmd_blit_image(
            cmd,
            src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
            vk::Filter::NEAREST,
        );

        let src_transfer_src_to_shader_read = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(src_image)
            .subresource_range(subresource_range);

        let blit_dst_to_present_layout_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(dst_image)
            .subresource_range(subresource_range);

        let image_memory_barriers = [src_transfer_src_to_shader_read, blit_dst_to_present_layout_transition];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        self.device.end_command_buffer(cmd).unwrap();

        let cmds = [cmd];
        let wait_masks =
            [vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::COMPUTE_SHADER];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(&render_finished_semaphore)
            .wait_dst_stage_mask(&wait_masks)
            .wait_semaphores(&present_complete_semaphores);
        self.device
            .queue_submit(self.queue, &[submit_info], end_fence)
            .unwrap();

        let swapchains = [self.swapchain];
        let indices = [acquired_swapchain_image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(&render_finished_semaphore);

        self.swapchain_loader
            .queue_present(self.queue, &present_info)
            .unwrap();

        // FIXME: there's still something wrong with frames in flight presenting. lots of stuttering and weird shit happening wtf
        // remove this when shit is fixed pls thx
        self.device.wait_for_fences(&[end_fence], true, u64::MAX).unwrap(); 


        let mut timestamps = [0u64; 2];
        let okay = self.device.get_query_pool_results(self.query_pool, 0, &mut timestamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).is_ok();
        if okay {
            let delta_in_ms = ((timestamps[1].saturating_sub(timestamps[0])) as f64 * self.timestamp_period as f64) / 1000000.0f64;
            self.stats.push_query_timings(delta_in_ms);
        }
        
        self.stats.end_of_frame(self.frame_count);
        self.frame_count += 1;
    }

    pub unsafe fn destroy(mut self) {
        self.device.device_wait_idle().unwrap();

        self.render_compute_pipeline.destroy(&self.device);
        log::info!("destroyed render compute pipeline");

        /*
        self.tick_voxel_compute_pipeline.destroy(&self.device);
        log::info!("destroyed tick voxel compute pipeline");

        self.generate_voxel_compute_pipeline.destroy(&self.device);
        log::info!("destroyed generate voxel compute pipeline");
        */


        self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        log::info!("destroyed descriptor pool");

        /*
        self.voxel_image.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed voxel image");

        self.device.destroy_image_view(self.voxel_surface_index_image.2, None);
        self.device.destroy_image(self.voxel_surface_index_image.0, None);
        self.allocator.free(self.voxel_surface_index_image.1).unwrap();
        log::info!("destroyed voxel surface index image");

        self.voxel_surface_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed voxel surface buffer");

        self.voxel_surface_counter_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed voxel counter buffer");

        self.visible_surface_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed visible surfaces buffer");

        self.visible_surface_counter_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed visible surfaces counter buffer");

        self.visible_surface_indirect_dispatch_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed indirect dispatch buffer");
        */

        self.svo.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed SVO buffer");

        self.device.destroy_query_pool(self.query_pool, None);
        log::info!("destroyed query pool");

        log::info!("waiting for all frame in flight fences...");
        let fences = self.frames_in_flight.iter().map(|x| x.end_fence).collect::<Vec<_>>();
        self.device
            .wait_for_fences(&fences, true, u64::MAX)
            .unwrap();
        for frame in self.frames_in_flight.into_iter() {
            self.device.destroy_image_view(frame.dst_image_view, None);
            self.device.destroy_image_view(frame.src_image_view, None);
            log::info!("destroyed image views for frame data");

            self.device.destroy_image(frame.rt_image, None);
            self.allocator.free(frame.rt_image_allocation.unwrap()).unwrap();
            log::info!("destroyed render target image frame data");

            self.device.destroy_semaphore(frame.present_complete_semaphore, None);
            self.device.destroy_semaphore(frame.render_finished_semaphore, None);
            self.device.destroy_fence(frame.end_fence, None);
            log::info!("destroyed semaphores and fences frame data");            

            self.device.free_command_buffers(self.pool, &[frame.cmd]);
            log::info!("destroyed cmd buffer frame data");            
        }


        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        log::info!("destroyed swapchain");

        self.surface_loader.destroy_surface(self.surface_khr, None);
        log::info!("destroyed surface");


        self.device.destroy_command_pool(self.pool, None);
        log::info!("destroyed cmd pool");

        self.device.destroy_device(None);
        log::info!("destroyed device");

        if let Some((inst, debug_messenger)) = self.debug {
            inst.destroy_debug_utils_messenger(debug_messenger, None);
            log::info!("destroyed debug utils messenger");
        }

        self.instance.destroy_instance(None);
        log::info!("destroyed instance");
    }
}

struct App {
    internal: Option<InternalApp>,
    args: Option<Args>,
    start: Instant,
    last: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        unsafe {
            self.internal = Some(InternalApp::new(event_loop, self.args.take().unwrap()));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => unsafe {
                event_loop.exit();
                self.internal.take().unwrap().destroy();
            },
            WindowEvent::RedrawRequested => unsafe {
                let inner = self.internal.as_mut().unwrap();
                let new = Instant::now();
                let elapsed = (new - self.start).as_secs_f32();
                let delta = (new - self.last).as_secs_f32();

                let size = inner.window.inner_size().cast::<f32>();
                inner
                    .movement
                    .update(&inner.input, size.width / size.height, delta);

                if inner.input.get_button(KeyCode::F5).pressed() {
                    if inner.window.fullscreen().is_none() {
                        inner
                            .window
                            .set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                    } else {
                        inner.window.set_fullscreen(None);
                    }
                }

                let left = inner.input.get_button(Button::Mouse(MouseButton::Left)).pressed();
                let right = inner.input.get_button(Button::Mouse(MouseButton::Right)).pressed();

                if left || right {
                    inner.click(left);
                }

                if inner.input.get_button(Button::Keyboard(KeyCode::KeyP)).pressed() {
                    let render_time_avg = inner.stats.get_average_in_ms();
                    let delta_ms = delta * 1000f32;
                    log::info!("CPU delta: {delta_ms:.3}, Main Compute Render Time Average: {render_time_avg:.3}");
                }

                if inner.input.get_button(Button::Keyboard(KeyCode::KeyL)).pressed() {
                    inner.stats.start_benchmarking(inner.frame_count);
                }

                if inner.input.get_button(Button::Keyboard(KeyCode::KeyH)).pressed() {
                    inner.debug_type = (inner.debug_type + 1) % 6;
                }

                if inner.input.get_button(Button::Mouse(MouseButton::Middle)).held() {
                    inner.sun = inner.movement.forward();
                }

                if inner.input.get_button(Button::Keyboard(KeyCode::KeyQ)).pressed() {
                    event_loop.exit();
                    self.internal.take().unwrap().destroy();
                    return;
                }

                inner.window.request_redraw();
                inner.render(delta, elapsed);
                self.last = new;
                input::update(&mut inner.input);
            },
            WindowEvent::Resized(new) => unsafe {
                let inner = self.internal.as_mut().unwrap();
                inner.resize(new.width, new.height);
            },

            // This is horrid...
            _ => {
                let inner = self.internal.as_mut().unwrap();
                input::window_event(&mut inner.input, &event);
            }
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let inner = self.internal.as_mut().unwrap();
        input::device_event(&mut inner.input, &event);
    }
}


pub fn main() {
    let args = Args::parse();
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        start: Instant::now(),
        last: Instant::now(),
        internal: None,
        args: Some(args),
    };
    event_loop.run_app(&mut app).unwrap();
}
