use ash::vk;
use gpu_allocator::vulkan::Allocation;
use rand::RngExt;
use rand::SeedableRng;
use crate::input::Button;
use crate::input::Input;
use crate::movement::Movement;
use crate::samplers;
use winit::event::MouseButton;
use std::collections::HashMap;
use std::ops::ControlFlow;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::KeyCode;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::Window;
use crate::statistics::Statistics;
use crate::asset;

use crate::swapchain;
use crate::ticker;
use crate::pipeline;
use crate::skybox;
use crate::voxel;
use crate::buffer;
use crate::instance;
use crate::physical_device;
use crate::device;
use crate::debug;
use crate::others;
use crate::per_frame_data::PerFrameData;
use crate::constant_data::ConstantData;

pub struct InternalApp {
    // entry, physical device, logical device
    entry: ash::Entry,
    device: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    
    // debug stuff
    debug: Option<(
        ash::ext::debug_utils::Instance,
        vk::DebugUtilsMessengerEXT
    )>,
    debug_marker: Option<ash::ext::debug_utils::Device>,
    
    // surface & swapchain
    surface_loader: ash::khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    swapchain_format: vk::Format,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    
    // queue
    queue: vk::Queue,
    queue_family_index: u32,
    
    // cmd buffs
    pool: vk::CommandPool,
    
    // pipelines
    render_compute_pipeline: pipeline::RenderPipeline,
    sky_compute_pipeline: pipeline::SkyPipeline,
    post_process_compute_pipeline: pipeline::PostProcessPipeline,
    voxel_compute_pipeline: pipeline::VoxelPipeline,
    
    // descriptors & frames in flight
    frames_in_flight: Vec<PerFrameData>,
    descriptor_pool: vk::DescriptorPool,
    const_descriptor_sets: ConstantData,
    
    // sparse stuff
    svo: voxel::SparseVoxelOctree,
    svt: voxel::SparseVoxelTexture,
        
    // important too
    allocator: gpu_allocator::vulkan::Allocator,
    
    // other GPU stuff
    query_pool: vk::QueryPool,
    timestamp_period: f32,
    skybox: skybox::Skybox,
    lights_buffer: buffer::Buffer,
    lights: Vec<vek::Vec4<f32>>,
    samplers: samplers::Samplers,
    
    // other CPU stuff
    pub was_resized: bool,
    pub window: Window,
    pub input: Input,    
    movement: Movement,
    frame_count: u64,
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
        asset!("post_process_compute.spv", assets);
        asset!("voxel_interesting_compute.spv", assets);

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
        let debug_messenger = debug::create_debug_messenger(&entry, &instance).inspect(|_x| {
            log::info!("created debug utils messenger");
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
                physical_device,
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

        let (swapchain_loader, swapchain, swapchain_images, swapchain_image_views, swapchain_format) = swapchain::create_swapchain(
            &instance,
            &surface_loader,
            surface_khr,
            physical_device,
            &device,
            extent,
            &debug_marker,
        );
        log::info!("created swapchain with {} images", swapchain_images.len());

        // swapchain::transfer_rt_images(&device, queue_family_index, &rt_images, pool, queue);
        // log::info!("transferred layout of render texture images");

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
        let render_compute_pipeline = pipeline::create_render_compute_pipeline(assets["raytracer.spv"], &device, &debug_marker, spec_constants);
        log::info!("created render compute pipeline");

        let sky_compute_pipeline = pipeline::create_sky_pipeline(assets["sky_compute.spv"], &device, &debug_marker);
        log::info!("created sky compute pipeline");

        let post_process_compute_pipeline = pipeline::create_post_process_pipeline(assets["post_process_compute.spv"], &device, &debug_marker, &args);
        log::info!("created post process compute pipeline");

        let voxel_compute_pipeline = pipeline::create_voxel_pipeline(assets["voxel_interesting_compute.spv"], &device, &debug_marker);
        log::info!("created voxel compute pipeline");

        let (svo, svt) = voxel::create_sparse_structures(
            &device,
            &mut allocator,
            &debug_marker,
            queue,
            pool,
            queue_family_index,
            args.force_regenerate,
        );
        log::info!("created sparse voxel structures");

        let samplers = samplers::Samplers::create_samplers(&device);
        log::info!("created samplers");        

        let skybox = skybox::create_skybox(
            &device,
            &mut allocator,
            &debug_marker,
            queue,
            pool,
            queue_family_index
        );
        log::info!("created skybox");

        let frames_in_flight = (0..crate::per_frame_data::FRAMES_IN_FLIGHT).into_iter().map(|_| {
            PerFrameData::create_per_frame_data(&device, pool, descriptor_pool, &post_process_compute_pipeline)
        }).collect::<Vec<_>>();
        log::info!("created frames in flight structures");

        const NUM_LIGHTS: usize = 100;

        let lights_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<vek::Vec4<f32>>() * NUM_LIGHTS, &debug_marker, "lights buffer", vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        let mut lights = Vec::<vek::Vec4<f32>>::new();

        for _i in 0..NUM_LIGHTS {
            let x = rand::random_range((voxel::TOTAL_SIZE as f32 / 2.0f32 - 10f32)..(voxel::TOTAL_SIZE as f32 / 2.0f32 + 10f32));
            let y = rand::random_range(0f32..(voxel::TOTAL_SIZE as f32));
            let z = rand::random_range((voxel::TOTAL_SIZE as f32 / 2.0f32 - 10f32)..(voxel::TOTAL_SIZE as f32 / 2.0f32 + 10f32));
            lights.push(vek::Vec4::new(x,y,z, 1.0));
        }

        buffer::write_to_buffer(&device, pool, queue, lights_buffer.buffer, &mut allocator, bytemuck::cast_slice(lights.as_slice()));
        log::info!("created lights buffer");

        let mut const_descriptor_sets = ConstantData::create_constant_descriptor_sets(&device, descriptor_pool, &render_compute_pipeline, &sky_compute_pipeline, &post_process_compute_pipeline, &voxel_compute_pipeline, &samplers, &skybox, &svt, &svo, &lights_buffer);
        const_descriptor_sets.recreate_rt_images_and_image_views_and_update_descriptor_sets(&device, swapchain_format, &mut allocator, queue_family_index, extent, &debug_marker, descriptor_pool, &samplers, &post_process_compute_pipeline, args.downscale_factor);
        crate::constant_data::transfer_layout_for_images(&device, queue_family_index, &const_descriptor_sets, pool, queue);
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
            was_resized: false,
            frames_in_flight,
            ticker: ticker::Ticker { accumulator: 0f32, count: 0 },
            sun: vek::Vec3::new(1f32, 0.3f32,0.5f32).normalized(),
            debug_type: 0,
            stats: Default::default(),
            args,
            lights,
            post_process_compute_pipeline,
            voxel_compute_pipeline,
            samplers,
            swapchain_images,
            swapchain_image_views,
        }
    }

    pub unsafe fn click(&mut self, add: bool) {
        let position = (self.movement.forward() * 5.0f32 + self.movement.position).floor().as_::<u32>();

        /*
        self.svo.set(position, add);
        self.svo.rebuild(&self.device, self.pool, self.queue, &mut self.allocator);
        */
    }

    pub unsafe fn recreate_swapchain(&mut self) {
        log::warn!("recreating swapchain");
        self.was_resized = false;
        self.device.device_wait_idle().unwrap();

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        for swapchain_image_view in self.swapchain_image_views.iter() {
            self.device.destroy_image_view(*swapchain_image_view, None);
        }

        let width = self.window.inner_size().width;
        let height = self.window.inner_size().height;
        
        let extent = vk::Extent2D { width, height };

        let (swapchain_loader, swapchain, swapchain_images, swapchain_image_views, swapchain_format) = swapchain::create_swapchain(
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
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.swapchain = swapchain;

        self.const_descriptor_sets.destroy_rt_images_and_image_views(&self.device, self.descriptor_pool, &mut self.allocator);
        self.const_descriptor_sets.recreate_rt_images_and_image_views_and_update_descriptor_sets(&self.device, swapchain_format, &mut self.allocator, self.queue_family_index, extent, &self.debug_marker, self.descriptor_pool, &self.samplers, &self.post_process_compute_pipeline, self.args.downscale_factor);
        crate::constant_data::transfer_layout_for_images(&self.device, self.queue_family_index, &self.const_descriptor_sets, self.pool, self.queue);
                
        for frame in self.frames_in_flight.iter_mut() {
            self.device.destroy_semaphore(frame.present_complete_semaphore, None);
            self.device.destroy_semaphore(frame.render_finished_semaphore, None);

            let create_info = vk::SemaphoreCreateInfo::default();
            frame.render_finished_semaphore = self.device.create_semaphore(&create_info, None).unwrap();
            frame.present_complete_semaphore = self.device.create_semaphore(&create_info, None).unwrap();
        }


        self.device.device_wait_idle().unwrap();
    }

    pub unsafe fn pre_render(&mut self, delta: f32) -> ControlFlow<()> {
        let size = self.window.inner_size().cast::<f32>();
        self.movement.update(&self.input, size.width / size.height, delta);
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
            self.debug_type = (self.debug_type as i32 + 1).rem_euclid(6) as u32;
        }
        if self.input.get_button(Button::Keyboard(KeyCode::KeyG)).pressed() {
            self.debug_type = (self.debug_type as i32 - 1).rem_euclid(6) as u32;
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
        //let frame_in_flight_index = 0;
        let frame_in_flight_index = self.frame_count % (self.frames_in_flight.len() as u64);
        let constant_descriptor_sets = &self.const_descriptor_sets;
        let PerFrameData {
            present_complete_semaphore,
            end_fence,
            cmd,
            per_frame_descriptor_sets,
            ..
        } = &self.frames_in_flight[frame_in_flight_index as usize];

        let cmd = *cmd;
        let present_complete_semaphores = [*present_complete_semaphore];
        let end_fence = *end_fence;

        if let Err(err) = self.device.wait_for_fences(&[end_fence], true, u64::MAX) {
            log::error!("wait on fence err: {:?}", err);
            //return;
        } else {
            let mut timestamps = [0u64; 2];
            let okay = self.device.get_query_pool_results(self.query_pool, 0, &mut timestamps, vk::QueryResultFlags::TYPE_64).is_ok();
            if okay {
                let delta_in_ms = ((timestamps[1].saturating_sub(timestamps[0])) as f64 * self.timestamp_period as f64) / 1000000.0f64;
                self.stats.push_query_timings(delta_in_ms);
            }
        }

        let mut rng = rand::rngs::SmallRng::seed_from_u64(421);

        for (i, light) in self.lights.iter_mut().enumerate() { 
            let axis = vek::Vec3::new(rng.random_range(-1f32..1f32), rng.random_range(-1f32..1f32), rng.random_range(-1f32..1f32));
            
            let disk_matrix = vek::Mat4::rotation_3d(elapsed, axis);

            let target_position = self.movement.position + disk_matrix.mul_point(vek::Vec3::unit_x()) * 5.0f32;

            let tmp = vek::Lerp::lerp(light.xyz(), target_position, 3.5 * delta);
            *light = vek::Vec4::from_point(tmp);
        }

        let (acquired_swapchain_image_index, suboptimal) = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain,
                u64::MAX,
                *present_complete_semaphore,
                vk::Fence::null(),
            )
            .unwrap();

        let swapchain_image = self.swapchain_images[acquired_swapchain_image_index as usize]; // then compose onto this...
        let swapchain_image_view = self.swapchain_image_views[acquired_swapchain_image_index as usize];

        //log::debug!("frame in flight index: {frame_in_flight_index}, acquire swapchain image index: {acquired_swapchain_image_index}");

        let descriptor_rt_image_view_info = vk::DescriptorImageInfo::default()
            .image_view(swapchain_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());

        // rt image for compositor (write only)
        let composition_compute_descriptor_image_infos_1 = [descriptor_rt_image_view_info];
        let composition_compute_image_descriptor_write_1 = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(per_frame_descriptor_sets.compositor_per_frame)
            .image_info(&composition_compute_descriptor_image_infos_1);

        self.device.update_descriptor_sets(&[composition_compute_image_descriptor_write_1], &[]);

        if suboptimal || self.was_resized {
            log::debug!("suboptimal: {suboptimal}");
            log::debug!("was resized: {}", self.was_resized);
            
            self.recreate_swapchain();
            self.was_resized = false;
            return;
        }

        self.device.reset_fences(&[end_fence]).unwrap();

        let render_finished_semaphore = [self.frames_in_flight[acquired_swapchain_image_index as usize].render_finished_semaphore];

        let cmd_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device
            .begin_command_buffer(cmd, &cmd_buffer_begin_info)
            .unwrap();
        self.device.cmd_reset_query_pool(cmd, self.query_pool, 0, 2);

        if !self.svt.sparse_partial_image_chunks.is_empty() {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.voxel_compute_pipeline.entry_points[0].pipeline_layout,
                0,
                &[constant_descriptor_sets.voxel_compute_pipeline_descriptor_set],
                &[],
            );
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.voxel_compute_pipeline.entry_points[0].pipeline,
            );
        
            let target_chunk_idfk = &self.svt.sparse_partial_image_chunks[self.frame_count as usize % self.svt.sparse_partial_image_chunks.len()];
            let push_constants = vek::Vec4::<u32>::from_point(target_chunk_idfk.origin);
            let raw = bytemuck::bytes_of(&push_constants);
        
            self.device.cmd_push_constants(
                cmd,
                self.voxel_compute_pipeline.entry_points[0].pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                raw,
            );
        
            self.device.cmd_dispatch(cmd, 8, 8, 8);
        }

        //self.sun = vek::Vec3::new((elapsed * 0.1f32).sin(), (elapsed * 0.05).sin(), (elapsed * 0.1f32).cos()).normalized();

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        let lights_buffer_barrier = vk::BufferMemoryBarrier2::default()
            .buffer(self.lights_buffer.buffer)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .size(vk::WHOLE_SIZE);
        let svt_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.svt.sparse_image)
            .subresource_range(subresource_range);
        let image_memory_barriers = [svt_image_barrier];
        let buffer_memory_barriers = [lights_buffer_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers).buffer_memory_barriers(&buffer_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);
        
        self.device.cmd_update_buffer(cmd, self.lights_buffer.buffer, 0, bytemuck::cast_slice(&self.lights));

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.render_compute_pipeline.entry_points[0].pipeline_layout,
            0,
            &[constant_descriptor_sets.main_render, constant_descriptor_sets.render_compute_pipeline_descriptor_set],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.render_compute_pipeline.entry_points[0].pipeline,
        );

        let size = self.window.inner_size();
        let window_size_no_downscale = vek::Vec2::<u32>::new(size.width, size.height);
        let size = vek::Vec2::<u32>::new(size.width, size.height) / self.args.downscale_factor;

        let group_size = 2u32.pow(self.args.group_size_exp);
        let size_f32 = size.map(|x| x as f32);

        let matrix = self.movement.proj_matrix.inverted() * self.movement.view_matrix;
        let push_constants = pipeline::PushConstants {
            screen_resolution: size_f32,
            _padding: Default::default(),
            matrix,
            position: self.movement.position.with_w(0f32),
            sun: self.sun.normalized().with_w(0f32),
            debug_type: self.debug_type,
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
        self.device.cmd_dispatch(cmd, size.x.div_ceil(group_size), size.y.div_ceil(group_size), 1);
        self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, self.query_pool, 1);

        
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
        self.device.cmd_dispatch(cmd, skybox::CLOUDS_RESOLUTION.div_ceil(8), skybox::CLOUDS_RESOLUTION.div_ceil(8), 1);

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
        self.device.cmd_dispatch(cmd, skybox::SKYBOX_RESOLUTION.div_ceil(8), skybox::SKYBOX_RESOLUTION.div_ceil(8), 6);
        
        let src_shader_write_to_shader_read = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(constant_descriptor_sets.rendered_image)
            .subresource_range(subresource_range);
        let skybox_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(6);
        let clouds_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let skybox_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.skybox_image)
            .subresource_range(skybox_subresource_range);
        let clouds_image_barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(self.skybox.clouds_image)
            .subresource_range(clouds_subresource_range);
        let image_memory_barriers = [src_shader_write_to_shader_read, skybox_image_barrier, clouds_image_barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        let full_passes_bloom = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(constant_descriptor_sets.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let image_memory_barriers = [full_passes_bloom];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        // execute bloom downsample passes
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.post_process_compute_pipeline.entry_points[1].pipeline,
        );
        for mip in 0..(constant_descriptor_sets.bloom_mip_image_views.len() as u32-1) {
            // no need to pipeline barrier for the first pass, as we just waited for the render texture image to finish right before this
            if mip > 0 {
                // wait on previous mip level to be done
                let previous_mip_level_subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .base_mip_level(mip)
                .level_count(1);
                let previous_mip_image_memory_barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(constant_descriptor_sets.bloom_image)
                    .subresource_range(previous_mip_level_subresource_range);
                let barriers = [previous_mip_image_memory_barrier];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                self.device.cmd_pipeline_barrier2(cmd, &dep);
            }
            
            
            let previous_mip_size = size / (1 << (mip)); // larger mip
            let next_mip_size = size / (1 << (mip+1)); // smaller mip

            
            
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post_process_compute_pipeline.entry_points[1].pipeline_layout,
                0,
                &[constant_descriptor_sets.compositor_downsample_bloom[mip as usize]],
                &[],
            );
            
            let downsample_dispatch_push_constants = previous_mip_size.as_::<f32>();

            self.device.cmd_push_constants(cmd, self.post_process_compute_pipeline.entry_points[1].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&downsample_dispatch_push_constants));
            self.device.cmd_dispatch(cmd, next_mip_size.x.div_ceil(8), next_mip_size.y.div_ceil(8), 1);
        }

        let full_passes_bloom = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(constant_descriptor_sets.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let image_memory_barriers = [full_passes_bloom];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        // execute bloom upsample passes
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.post_process_compute_pipeline.entry_points[2].pipeline,
        );

        // there is no need to go down to the largest mip since we will be sampling from a smaller mip anyways
        let minimum_upsampling_mip = 2;

        for mip in (minimum_upsampling_mip..(constant_descriptor_sets.bloom_mip_image_views.len() as u32 - 1)).rev() {
            // no need to pipeline barrier for the very first pass (we did a full pipeline barrier for the entire bloom image right before this)
            if mip != constant_descriptor_sets.bloom_mip_image_views.len() as u32 - 2 {
                // wait on previous mip level to be done
                let previous_mip_level_subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(mip+1)
                    .level_count(1);
                let previous_mip_image_memory_barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(constant_descriptor_sets.bloom_image)
                    .subresource_range(previous_mip_level_subresource_range);
                let barriers = [previous_mip_image_memory_barrier];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                self.device.cmd_pipeline_barrier2(cmd, &dep);
            }
            
            
            let previous_mip_size = size / (1 << (mip+1)); // smaller mip
            let next_mip_size = size / (1 << (mip)); // larger mip

            
            
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.post_process_compute_pipeline.entry_points[2].pipeline_layout,
                0,
                &[constant_descriptor_sets.compositor_upsample_bloom[mip as usize]],
                &[],
            );
            
            let upsample_dispatch_push_constants = previous_mip_size.as_::<f32>();

            self.device.cmd_push_constants(cmd, self.post_process_compute_pipeline.entry_points[2].pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&upsample_dispatch_push_constants));
            self.device.cmd_dispatch(cmd, next_mip_size.x.div_ceil(8), next_mip_size.y.div_ceil(8), 1);
        }

        let last_pass_bloom = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(constant_descriptor_sets.bloom_image)
            .subresource_range(vk::ImageSubresourceRange::default().level_count(vk::REMAINING_MIP_LEVELS).layer_count(1).aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).base_array_layer(0));
        let swapchain_image_undefined_to_blit_dst_layout_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(swapchain_image)
            .subresource_range(subresource_range);
        let image_memory_barriers = [last_pass_bloom, swapchain_image_undefined_to_blit_dst_layout_transition];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers);
        self.device.cmd_pipeline_barrier2(cmd, &dep);

        // execute composition pass
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.post_process_compute_pipeline.entry_points[0].pipeline_layout,
            0,
            //&[per_frame_descriptor_sets.compositor_per_frame, constant_descriptor_sets.compositor_compute_pipeline_descriptor_set],
            &[constant_descriptor_sets.compositor, per_frame_descriptor_sets.compositor_per_frame],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.post_process_compute_pipeline.entry_points[0].pipeline,
        );

        // TODO: remove duplicate push constants by saving to uniform buffer type shift
        self.device.cmd_push_constants(
            cmd,
            self.post_process_compute_pipeline.entry_points[0].pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            raw,
        );

        self.device.cmd_dispatch(cmd, window_size_no_downscale.x.div_ceil(8), window_size_no_downscale.y.div_ceil(8), 1);

        let blit_dst_to_present_layout_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .image(swapchain_image)
            .subresource_range(subresource_range);

        let image_memory_barriers = [blit_dst_to_present_layout_transition];
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

        let _start = std::time::Instant::now();
        let suboptimal = self.swapchain_loader
            .queue_present(self.queue, &present_info)
            .unwrap();
        let _end = std::time::Instant::now();


        self.stats.end_of_frame(self.frame_count);
        self.frame_count += 1;


        //log::debug!("CPU thread took: {}us", (_end-_start).as_micros());

        if suboptimal {
            self.recreate_swapchain();
        }
    }

    pub unsafe fn destroy(mut self) {
        self.device.device_wait_idle().unwrap();

        self.render_compute_pipeline.destroy(&self.device);
        log::info!("destroyed render compute pipeline");

        self.sky_compute_pipeline.destroy(&self.device);
        log::info!("destroyed sky compute pipeline");

        self.post_process_compute_pipeline.destroy(&self.device);
        log::info!("destroyed post process compute pipeline");

        self.voxel_compute_pipeline.destroy(&self.device);
        log::info!("destroyed voxel compute pipeline");

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
            frame.destroy_everything(&self.device, self.pool);
        }

        self.const_descriptor_sets.destroy_rt_images_and_image_views(&self.device, self.descriptor_pool, &mut self.allocator);
        log::info!("destroyed const descriptor sets");

        for swapchain_image_view in self.swapchain_image_views {
            self.device.destroy_image_view(swapchain_image_view, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        log::info!("destroyed swapchain");


        self.surface_loader.destroy_surface(self.surface_khr, None);
        log::info!("destroyed surface");

        self.samplers.destroy_samplers(&self.device);
        log::info!("destroyed samplers");

        self.device.destroy_command_pool(self.pool, None);
        log::info!("destroyed cmd pool");
        
        self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        log::info!("destroyed descriptor pool");

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
