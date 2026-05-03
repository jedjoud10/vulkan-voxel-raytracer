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
use crate::asset;

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
use crate::per_frame_data::PerFrameData;
use crate::constant_descriptor_sets::ConstantDescriptorSets;

pub struct InternalApp {
    frame_count: u64,

    pub input: Input,
    movement: Movement,
    entry: ash::Entry,
    pub window: Window,
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

    render_compute_pipeline: pipeline::RenderPipeline,
    sky_compute_pipeline: pipeline::SkyPipeline,

    query_pool: vk::QueryPool,
    timestamp_period: f32,

    descriptor_pool: vk::DescriptorPool,
    allocator: gpu_allocator::vulkan::Allocator,

    const_descriptor_sets: ConstantDescriptorSets,
    
    svo: voxel::SparseVoxelOctree,
    svt: voxel::SparseVoxelTexture,

    skybox: skybox::Skybox,
    lights_buffer: buffer::Buffer,
    lights: Vec<vek::Vec4<f32>>,

    ticker: ticker::Ticker,
    sun: vek::Vec3<f32>,
    debug_type: u32,
    args: crate::Args,
    stats: Statistics,
}

impl InternalApp {
    pub unsafe fn new(event_loop: &ActiveEventLoop, args: crate::Args) -> Self {
        let mut assets = HashMap::<&str, &[u32]>::new();
        asset!("raytracer.spv", assets);
        asset!("sky_compute.spv", assets);

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

        let (surface_loader, surface_khr) = others::create_surface(&instance, &entry, &window);
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
                debug_settings: gpu_allocator::AllocatorDebugSettings {
                    log_leaks_on_shutdown: false,
                    log_frees: false,
                    ..Default::default()
                },
                buffer_device_address: false,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            })
            .unwrap();
        log::info!("created gpu allocator");

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = device.create_command_pool(&pool_create_info, None).unwrap();
        log::info!("created cmd pool");

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
                    args.downscale_factor,
                    c"temporary render target image"
                )
            })
            .collect();
        log::info!("created {} in-flight render texture images", images.len());

        swapchain::transfer_rt_images(&device, queue_family_index, &rt_images, pool, queue);
        log::info!("transferred layout of render texture images");

        let descriptor_pool = others::create_descriptor_pool(&device);
        log::info!("created descriptor pool");

        let spec_constants = pipeline::RenderPipelineSpecConstants {
            shadow_samples: args.shadow_samples,
            max_ray_iterations: args.max_ray_iterations,
            round_normals: if args.round_normals { 1 } else { 0 },
            ambient_occlusion: if args.ambient_occlusion { 1 } else { 0 }, 
            wavy_reflections: if args.wavy_reflections { 1 } else { 0 }, 
            pixelated_shadows: if args.pixelated_shadows { 1 } else { 0 }, 
            group_size: 2u32.pow(args.group_size_exp),
        };
        let render_compute_pipeline = pipeline::create_render_compute_pipeline(&*assets["raytracer.spv"], &device, &debug_marker, spec_constants);
        log::info!("created render compute pipeline");

        let sky_compute_pipeline = pipeline::create_sky_pipeline(&*assets["sky_compute.spv"], &device, &debug_marker);
        log::info!("created sky compute pipeline");

        let (svo, svt) = voxel::create_sparse_structures(
            &device,
            &mut allocator,
            &debug_marker,
            queue,
            pool,
            queue_family_index
        );
        log::info!("created sparse voxel structures");

        let skybox = skybox::create_skybox(
            &device,
            &mut allocator,
            &debug_marker,
            queue,
            pool,
            queue_family_index
        );
        log::info!("created skybox");

        let mut frames_in_flight = Vec::<PerFrameData>::new();
        for ((rt_image, rt_image_allocation), swapchain_image) in rt_images.into_iter().zip(images.into_iter()) {
            frames_in_flight.push(PerFrameData::create_per_frame_data(&device, pool, swapchain_format, descriptor_pool, &render_compute_pipeline, rt_image, rt_image_allocation, swapchain_image));
        }
        log::info!("created frames in flight structures");

        let lights_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<vek::Vec4<f32>>() * 10, &debug_marker, "lights buffer", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        let mut lights = Vec::<vek::Vec4<f32>>::new();

        for i in 0..10 {
            let x = rand::random_range((voxel::TOTAL_SIZE as f32 / 2.0f32 - 10f32)..(voxel::TOTAL_SIZE as f32 / 2.0f32 + 10f32));
            let y = rand::random_range(500f32..(voxel::TOTAL_SIZE as f32));
            let z = rand::random_range((voxel::TOTAL_SIZE as f32 / 2.0f32 - 10f32)..(voxel::TOTAL_SIZE as f32 / 2.0f32 + 10f32));
            lights.push(vek::Vec4::new(x,y,z, 1.0));
        }

        buffer::write_to_buffer(&device, pool, queue, lights_buffer.buffer, &mut allocator, bytemuck::cast_slice(lights.as_slice()));
        log::info!("created lights buffer");

        let const_descriptor_sets = ConstantDescriptorSets::create_constant_descriptor_sets(&device, pool, descriptor_pool, &render_compute_pipeline, &sky_compute_pipeline, &skybox, &svt, &svo, &lights_buffer);
        log::info!("created constant descriptor sets");

        let query_pool = others::create_query_pool(&device);
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
            lights_buffer,
            surface_loader,
            surface_khr,
            debug: debug_messenger,
            debug_marker,
            swapchain_loader,
            swapchain_format,
            swapchain,
            queue_family_index,
            queue,
            const_descriptor_sets,
            pool,
            render_compute_pipeline,
            sky_compute_pipeline,
            descriptor_pool,
            query_pool,
            timestamp_period,
            allocator,
            svo,
            svt,
            skybox,
            frames_in_flight,
            ticker: ticker::Ticker { accumulator: 0f32, count: 0 },
            sun: vek::Vec3::new(1f32, 0.3f32,0.5f32).normalized(),
            debug_type: 0,
            stats: Default::default(),
            args,
            lights,
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

        for frame in self.frames_in_flight.iter_mut() {
            self.device.destroy_image_view(frame.dst_image_view, None);
            self.device.destroy_image_view(frame.src_image_view, None);

            self.device.destroy_image(frame.rt_image, None);
            self.allocator.free(frame.rt_image_allocation.take().unwrap()).unwrap();
        }

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);

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
                    self.args.downscale_factor,
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

        for ((frame, (rt_image, rt_image_allocation)), swapchain_image) in self.frames_in_flight.iter_mut().zip(rt_images).zip(images) {
            frame.recreate_image_views_and_update_descriptor_sets(&self.device, swapchain_format, rt_image, rt_image_allocation, swapchain_image);
        }

        self.device.device_wait_idle().unwrap();
    }

    pub unsafe fn pre_render(&mut self, delta: f32) -> ControlFlow<()> {
        let size = self.window.inner_size().cast::<f32>();
        self
            .movement
            .update(&self.input, size.width / size.height, delta);
        if self.input.get_button(KeyCode::F5).pressed() {
            if self.window.fullscreen().is_none() {
                self
                    .window
                    .set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            } else {
                self.window.set_fullscreen(None);
            }
        }
        let left = self.input.get_button(Button::Mouse(MouseButton::Left)).pressed();
        let right = self.input.get_button(Button::Mouse(MouseButton::Right)).pressed();
        if left || right {
            self.click(left);
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyP)).pressed() {
            let render_time_avg = self.stats.get_average_in_ms();
            let delta_ms = delta * 1000f32;
            log::info!("CPU delta: {delta_ms:.3}, Main Compute Render Time Average: {render_time_avg:.3}");
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyL)).pressed() {
            self.stats.start_benchmarking(self.frame_count);
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyH)).pressed() {
            self.debug_type = (self.debug_type + 1) % 6;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyJ)).pressed() {
            let report = self.allocator.generate_report();
            log::debug!("{:?}", report);
        }
        if self.input.get_button(Button::Mouse(MouseButton::Middle)).held() {
            self.sun = self.movement.forward();
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyQ)).pressed() {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }

    pub unsafe fn render(&mut self, delta: f32, elapsed: f32) {
        let frame_index = self.frame_count % (self.frames_in_flight.len() as u64);
        let constant_descriptor_sets = &self.const_descriptor_sets;
        let PerFrameData {
            swapchain_image,
            rt_image,
            present_complete_semaphore,
            end_fence,
            cmd,
            per_frame_descriptor_sets,
            ..
        } = &self.frames_in_flight[frame_index as usize];

        let cmd = *cmd;
        let swapchain_image = *swapchain_image;
        let rt_image = *rt_image;
        let present_complete_semaphores = [*present_complete_semaphore];
        let end_fence = *end_fence;

        if let Err(err) = self.device.wait_for_fences(&[end_fence], true, u64::MAX) {
            log::error!("wait on fence err: {:?}", err);
        }

        let mut timestamps = [0u64; 2];
        let okay = self.device.get_query_pool_results(self.query_pool, 0, &mut timestamps, vk::QueryResultFlags::TYPE_64).is_ok();
        if okay {
            let delta_in_ms = ((timestamps[1].saturating_sub(timestamps[0])) as f64 * self.timestamp_period as f64) / 1000000.0f64;
            self.stats.push_query_timings(delta_in_ms);
        }

        self.device.reset_fences(&[end_fence]).unwrap();


        for (i, light) in self.lights.iter_mut().enumerate() { 
            let tmp = vek::Lerp::lerp(light.xyz(), self.movement.position + vek::Vec3::new((i as f32).sin(), 0.0, (i as f32).cos()) * 0.05f32, 3.5 * delta);
            *light = vek::Vec4::from_point(tmp);
        }

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
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device
            .begin_command_buffer(cmd, &cmd_buffer_begin_info)
            .unwrap();
        self.device.cmd_reset_query_pool(cmd, self.query_pool, 0, 2);

        self.device.cmd_update_buffer(cmd, self.lights_buffer.buffer, 0, bytemuck::cast_slice(&self.lights));

        //self.sun = vek::Vec3::new((elapsed * 0.1f32).sin(), (elapsed * 0.05).sin(), (elapsed * 0.1f32).cos()).normalized();

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

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sky_compute_pipeline.entry_points[0].pipeline_layout,
            0,
            &[constant_descriptor_sets.sky_compute_pipeline_descriptor_set],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sky_compute_pipeline.entry_points[0].pipeline,
        );

        let push_constants = pipeline::SkyComputePushConstants {
            sun: self.sun.normalized().with_w(elapsed),
        };
        self.device.cmd_push_constants(cmd, self.sky_compute_pipeline.entry_points[0].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push_constants));
        self.device.cmd_dispatch(cmd, skybox::CLOUDS_RESOLUTION.div_ceil(32), skybox::CLOUDS_RESOLUTION.div_ceil(32), 1);

        // No need for barrier as the two compute shaders don't read / write shared resources!!! yay!!!
        /*
        let clouds_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let clouds_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.clouds_image)
            .subresource_range(clouds_subresource_range);
        let image_memory_barriers = [clouds_image_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);
        */
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sky_compute_pipeline.entry_points[1].pipeline_layout,
            0,
            &[constant_descriptor_sets.sky_compute_pipeline_descriptor_set],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sky_compute_pipeline.entry_points[1].pipeline,
        );

        self.device.cmd_push_constants(cmd, self.sky_compute_pipeline.entry_points[1].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push_constants));
        self.device.cmd_dispatch(cmd, skybox::SKYBOX_RESOLUTION.div_ceil(32), skybox::SKYBOX_RESOLUTION.div_ceil(32), 6);

        

        let skybox_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(6);
        let skybox_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.skybox_image)
            .subresource_range(skybox_subresource_range);
        let image_memory_barriers = [skybox_image_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.render_compute_pipeline.entry_points[0].pipeline_layout,
            0,
            &[per_frame_descriptor_sets.main_render_per_frame, constant_descriptor_sets.render_compute_pipeline_descriptor_set],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.render_compute_pipeline.entry_points[0].pipeline,
        );

        let size = self.window.inner_size();
        let size = vek::Vec2::<u32>::new(size.width, size.height)
            .map(|val| val / self.args.downscale_factor);

        let group_size = 2u32.pow(self.args.group_size_exp) as f32;
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
            .x(self.window.inner_size().width as i32 / self.args.downscale_factor as i32)
            .y(self.window.inner_size().height as i32 / self.args.downscale_factor as i32)
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
            .dst_access_mask(vk::AccessFlags2::NONE)
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
        //self.device.wait_for_fences(&[end_fence], true, u64::MAX).unwrap(); 
       
        self.stats.end_of_frame(self.frame_count);
        self.frame_count += 1;
    }

    pub unsafe fn destroy(mut self) {
        self.device.device_wait_idle().unwrap();

        self.render_compute_pipeline.destroy(&self.device);
        log::info!("destroyed render compute pipeline");

        self.sky_compute_pipeline.destroy(&self.device);
        log::info!("destroyed sky compute pipeline");

        self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        log::info!("destroyed descriptor pool");

        self.svo.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed SVO buffers & stuff");

        self.svt.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed SVT");

        self.skybox.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed skybox");

        self.lights_buffer.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed lights buffer");

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

        drop(self.allocator);
        self.device.destroy_device(None);
        log::info!("destroyed device");

        if let Some((inst, debug_messenger)) = self.debug {
            inst.destroy_debug_utils_messenger(debug_messenger, None);
            log::info!("destroyed debug utils messenger");
        }

        self.instance.destroy_instance(None);
        log::info!("destroyed instance");

        drop(self.entry); // DO NOT REMOVE ENTRY FROM STRUCT. NEEDED!!!
        log::info!("everything is done!");
    }
}
