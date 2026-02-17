#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_imports)]

mod assets;
mod debug;
use assets::convert;
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

use ash;
use ash::vk;
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

use crate::buffer::*;
use crate::pipeline::*;
use crate::rays::*;
use crate::voxel::*;

struct InternalApp {
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
    images: Vec<vk::Image>,
    rt_images: Vec<(vk::Image, Allocation)>,
    begin_semaphore: vk::Semaphore,
    end_semaphore: vk::Semaphore,
    end_fence: vk::Fence,
    queue: vk::Queue,
    queue_family_index: u32,
    pool: vk::CommandPool,

    render_compute_pipeline: RenderPipeline,
    generate_voxel_compute_pipeline: VoxelGeneratePipeline,
    tick_voxel_compute_pipeline: VoxelTickPipeline,
    ray_trace_pipeline: RayTracePipeline,

    ray_trace_buffers: RayTraceBuffers,

    descriptor_pool: vk::DescriptorPool,
    allocator: gpu_allocator::vulkan::Allocator,
    voxel_image: VoxelImage,
    voxel_surface_index_image: (vk::Image, Allocation, vk::ImageView),
    voxel_surface_buffer: Buffer,
    voxel_surface_counter_buffer: Buffer,
    svo: SparseVoxelOctree,
    visible_surface_buffer: Buffer,
    visible_surface_counter_buffer: Buffer,
    visible_surface_indirect_dispatch_buffer: Buffer,
    ticker: ticker::Ticker,
    sun: vek::Vec3<f32>,
    debug_type: u32,
}

impl InternalApp {
    pub unsafe fn new(event_loop: &ActiveEventLoop) -> Self {
        let mut assets = HashMap::<&str, Vec<u32>>::new();
        asset!("raymarcher.spv", assets);
        asset!("voxel_tick.spv", assets);
        asset!("voxel_generate.spv", assets);
        asset!("ray_stuff.spv", assets);

        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();
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
        physical_device_candidates.sort_by(|(_, a), (_, b)| a.cmp(b));

        if physical_device_candidates.is_empty() {
            log::error!("no physical device was chosen!");
            panic!();
        }

        let physical_device = physical_device_candidates[0].0;
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

        let extent = vk::Extent2D {
            width: 800,
            height: 600,
        };

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
                    c"temporary render target image"
                )
            })
            .collect();
        log::info!("created {} in-flight render texture images", images.len());

        swapchain::transfer_rt_images(&device, queue_family_index, &rt_images, pool, queue);
        log::info!("transferred layout of render texture images");

        let begin_semaphore = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let end_semaphore = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let end_fence = device.create_fence(&Default::default(), None).unwrap();
        log::info!("created semaphores and fence");

        let descriptor_pool = pool::create_descriptor_pool(&device);
        log::info!("created descriptor pool");

        let render_compute_pipeline = pipeline::create_render_compute_pipeline(&*assets["raymarcher.spv"], &device, &debug_marker);
        log::info!("created render compute pipeline");

        let generate_voxel_compute_pipeline = pipeline::create_generate_voxel_compute_pipeline(&*assets["voxel_generate.spv"], &device, &debug_marker);
        log::info!("created voxel generate compute pipeline");

        let tick_voxel_compute_pipeline = pipeline::create_tick_voxel_compute_pipeline(&*assets["voxel_tick.spv"], &device, &debug_marker);
        log::info!("created voxel tick compute pipeline");

        let voxel_image = voxel::create_voxel_octree_mip_map_image(&device, &mut allocator, vk::Format::R64_UINT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST, &debug_marker, "voxel image");
        log::info!("created voxel image");

        let svo = voxel::create_sparse_voxel_octree(&device, &mut allocator, &debug_marker, "sparse voxel octree");
        log::info!("created sparse voxel octree buffers");

        let voxel_surface_index_image = voxel::create_voxel_image(&device, &mut allocator, vk::Format::R32_UINT, vk::ImageUsageFlags::STORAGE, &debug_marker, c"voxel image indices");
        log::info!("created voxel surface indices image");

        const SOME_ARBITRARY_SIZE_FOR_MAX_NUMBER_OF_CUBES_IDK: usize = _SIZE*_SIZE*_SIZE / 64;
        const VOXEL_SURFACE_BUFFER_SIZE: usize = size_of::<vek::Vec4<u8>>() * 6 * 16 * SOME_ARBITRARY_SIZE_FOR_MAX_NUMBER_OF_CUBES_IDK;

        let voxel_surface_buffer = buffer::create_buffer(&device, &mut allocator, VOXEL_SURFACE_BUFFER_SIZE, &debug_marker, "surface buffer");
        log::info!("created voxel surface buffer");
        
        let voxel_surface_counter_buffer = buffer::create_counter_buffer(&device, &mut allocator, &debug_marker, "surface counter buffer");
        log::info!("created voxel surface counter");

        let visible_surface_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<u32>() * 1024 * 1024, &debug_marker, "visible surface buffer");
        log::info!("created visible surfaces buffer");

        let visible_surface_counter_buffer = buffer::create_counter_buffer(&device, &mut allocator, &debug_marker, "surface counter buffer");
        log::info!("created visible surfaces counter");

        let visible_surface_indirect_dispatch_buffer = buffer::create_buffer(&device, &mut allocator, size_of::<u32>() * 3, &debug_marker, "indirect dispatch buffer");
        log::info!("created visible surfaces indirect buffer");

        voxel::generate_voxel_image(
            &device,
            queue,
            pool,
            descriptor_pool,
            queue_family_index,
            &voxel_image,
            voxel_surface_index_image.0,
            voxel_surface_index_image.2,
            &generate_voxel_compute_pipeline,            
        );

        voxel::convert_mips_svo(
            &device,
            &mut allocator,
            queue,
            pool,
            descriptor_pool,
            queue_family_index,
            &voxel_image,
            &svo
        );

        let ray_trace_pipeline = rays::create_ray_trace_compute_pipeline(&*assets["ray_stuff.spv"], &device, &debug_marker);
        log::info!("created ray trace pipeline");
        let ray_trace_buffers = rays::create_ray_trace_buffers(&device, &mut allocator, &debug_marker);
        log::info!("created ray trace buffers");

        Self {
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
            images,
            begin_semaphore,
            end_semaphore,
            queue_family_index,
            end_fence,
            queue,
            pool,
            render_compute_pipeline,
            generate_voxel_compute_pipeline,
            tick_voxel_compute_pipeline,
            descriptor_pool,
            allocator,
            voxel_image,
            svo,
            rt_images,
            ticker: ticker::Ticker { accumulator: 0f32, count: 0 },
            voxel_surface_buffer,
            voxel_surface_index_image,
            voxel_surface_counter_buffer,
            visible_surface_buffer,
            visible_surface_counter_buffer,
            visible_surface_indirect_dispatch_buffer,
            sun: vek::Vec3::unit_y() + vek::Vec3::unit_x(),
            debug_type: 0,
            ray_trace_pipeline,
            ray_trace_buffers,
        }
    }

    pub unsafe fn click(&mut self, add: bool) {
        let forward = vek::Mat4::from(self.movement.rotation).mul_direction(-vek::Vec3::unit_z()).with_w(0.0f32);
        let position = (self.movement.position + forward * 2.0).map(|x| x as u32);


        voxel::update_voxel(
            &self.device,
            &mut self.allocator,
            self.queue,
            self.pool,
            &self.voxel_image,
            true,
            /*voxel::Voxel {
                active: true,
                reflective: self.ticker.count % 4 == 0,
                refractive: self.ticker.count % 4 == 1,
                placed: true,
            }.into_raw(),
            */
            position,
        )
    }

    pub unsafe fn resize(&mut self, width: u32, height: u32) {
        self.device.device_wait_idle().unwrap();

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);

        for (image, allocation) in self.rt_images.drain(..) {
            self.device.destroy_image(image, None);
            self.allocator.free(allocation).unwrap();
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
        self.images = images;
        self.swapchain_loader = swapchain_loader;
        self.swapchain_format = swapchain_format;
        self.swapchain = swapchain;

        let rt_images: Vec<(vk::Image, Allocation)> = (0..self.images.len())
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
        self.rt_images = rt_images;
    }

    pub unsafe fn render(&mut self, delta: f32, elapsed: f32) {
        self.device.reset_fences(&[self.end_fence]).unwrap();

        let (index, _) = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.begin_semaphore,
                vk::Fence::null(),
            )
            .unwrap();
        let dst_image = self.images[index as usize];
        let (src_image, _) = self.rt_images[index as usize];

        let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.pool);
        let cmd = self
            .device
            .allocate_command_buffers(&cmd_buffer_create_info)
            .unwrap()[0];

        let cmd_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device
            .begin_command_buffer(cmd, &cmd_buffer_begin_info)
            .unwrap();

        //self.sun = vek::Vec3::new(1f32, 0.3f32,0.5f32).normalized();
        self.sun = vek::Vec3::new((elapsed * 0.1f32).sin(), (elapsed * 0.05).sin(), (elapsed * 0.1f32).cos()).normalized();

        let push_constants = PushConstants2 {
            forward: vek::Mat4::from(self.movement.rotation).mul_direction(-vek::Vec3::unit_z()).with_w(0.0f32),
            position: self.movement.position.with_w(0.0f32),
            sun: self.sun.normalized().with_w(0f32),
            tick: self.ticker.count,

            // FIXME: assumes we are running the shadow calc for every frame...
            delta: delta.max(1f32 / ticker::TICKS_PER_SECOND),
        };

        /*
        let desc_temp = self.ticker.update(delta).then(|| voxel::execute_voxel_tick_compute(
            &self.device,
            cmd,
            self.descriptor_pool,
            self.queue_family_index,
            self.voxel_surface_buffer.buffer,
            self.voxel_surface_counter_buffer.buffer,
            self.visible_surface_buffer.buffer,
            self.visible_surface_counter_buffer.buffer,
            self.visible_surface_indirect_dispatch_buffer.buffer,
            &self.voxel_image,
            self.voxel_surface_index_image.0,
            self.voxel_surface_index_image.2,
            &self.tick_voxel_compute_pipeline,
            push_constants
        ));
        */

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        let src_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(self.swapchain_format)
            .image(src_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let dst_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(self.swapchain_format)
            .image(dst_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let src_image_view = self
            .device
            .create_image_view(&src_image_view_create_info, None)
            .unwrap();
        let dst_image_view = self
            .device
            .create_image_view(&dst_image_view_create_info, None)
            .unwrap();

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


        // we need: render compute + pipeline + ray compute pipeline
        let layouts = [self.render_compute_pipeline.descriptor_set_layout, self.ray_trace_pipeline.descriptor_set_layout];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let all_descriptor_sets_for_frame = self
            .device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();
        let render_descriptor_set = all_descriptor_sets_for_frame[0];
        let ray_trace_descriptor_set = all_descriptor_sets_for_frame[1];


        let descriptor_rt_image_info = vk::DescriptorImageInfo::default()
            .image_view(src_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_ray_inputs_info = vk::DescriptorBufferInfo::default()
            .buffer(self.ray_trace_buffers.ray_inputs.buffer)
            .offset(0)
            .range(u64::MAX);
        let descriptor_ray_outputs_info = vk::DescriptorBufferInfo::default()
            .buffer(self.ray_trace_buffers.ray_outputs.buffer)
            .offset(0)
            .range(u64::MAX);
        let descriptor_svo_bitmasks_info = vk::DescriptorBufferInfo::default()
            .buffer(self.svo.bitmask_buffer.buffer)
            .offset(0)
            .range(u64::MAX);
        let descriptor_svo_indices_info = vk::DescriptorBufferInfo::default()
            .buffer(self.svo.index_buffer.buffer)
            .offset(0)
            .range(u64::MAX);

        let descriptor_rt_image_infos = [descriptor_rt_image_info];
        let descriptor_ray_inputs_infos = [descriptor_ray_inputs_info, descriptor_ray_outputs_info, descriptor_svo_bitmasks_info, descriptor_svo_indices_info];

        let image_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(render_descriptor_set)
            .image_info(&descriptor_rt_image_infos);
        let buffer_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(descriptor_ray_inputs_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_binding(1)
            .dst_set(render_descriptor_set)
            .buffer_info(&descriptor_ray_inputs_infos);
        

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
            .map(|val| val / swapchain::SCALING_FACTOR);

        let width_group_size = (size.x as f32 / 8f32).ceil() as u32;
        let height_group_size = (size.y as f32 / 8f32).ceil() as u32;
        let size_f32 = size.map(|x| x as f32);

        let push_constants = pipeline::PushConstants {
            screen_resolution: size_f32,
            _padding: Default::default(),
            matrix: self.movement.proj_matrix * self.movement.view_matrix,
            position: self.movement.position.with_w(0f32),
            sun: self.sun.normalized().with_w(0f32),
            debug_type: self.debug_type as u32,
        };

        let raw = bytemuck::bytes_of(&push_constants);

        self.device.cmd_push_constants(
            cmd,
            self.render_compute_pipeline.entry_points[0].pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            raw,
        );

        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let barriers = [barrier];
        let dep_wait_all: vk::DependencyInfo<'_> = vk::DependencyInfo::default().memory_barriers(&barriers);

        // generate rays, store data in `ray_inputs` buffer
        self.device.cmd_dispatch(cmd, width_group_size, height_group_size, 1);

        // execute the ray trace pipeline now
        self.device.cmd_pipeline_barrier2(cmd, &dep_wait_all);
        rays::trace_rays(&self.device, cmd, all_descriptor_sets_for_frame[1], &self.ray_trace_buffers, &self.ray_trace_pipeline, &self.svo, size.x * size.y, self.movement.position);
        self.device.cmd_pipeline_barrier2(cmd, &dep_wait_all);

        // execute the apply pipeline that fetches result from the ray trace pipeline
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.render_compute_pipeline.entry_points[1].pipeline_layout,
            0,
            &[render_descriptor_set],
            &[],
        );
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.render_compute_pipeline.entry_points[1].pipeline,
        );
        self.device.cmd_dispatch(cmd, width_group_size, height_group_size, 1);

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
            .x(self.window.inner_size().width as i32 / swapchain::SCALING_FACTOR as i32)
            .y(self.window.inner_size().height as i32 / swapchain::SCALING_FACTOR as i32)
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
        let rendered_semaphores = [self.end_semaphore];
        let acquire_sempahores = [self.begin_semaphore];
        let wait_masks =
            [vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::ALL_GRAPHICS];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(&rendered_semaphores)
            .wait_dst_stage_mask(&wait_masks)
            .wait_semaphores(&acquire_sempahores);
        self.device
            .queue_submit(self.queue, &[submit_info], self.end_fence)
            .unwrap();

        let swapchains = [self.swapchain];
        let indices = [index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(&rendered_semaphores);

        self.device
            .wait_for_fences(&[self.end_fence], true, u64::MAX)
            .unwrap();
        self.swapchain_loader
            .queue_present(self.queue, &present_info)
            .unwrap();
        self.device.free_command_buffers(self.pool, &[cmd]);

        self.device.destroy_image_view(src_image_view, None);
        self.device.destroy_image_view(dst_image_view, None);
        self.device
            .free_descriptor_sets(self.descriptor_pool, &all_descriptor_sets_for_frame)
            .unwrap();
        
        /*
        if let Some(desc_temp) = desc_temp{
            self.device.free_descriptor_sets(self.descriptor_pool, &[desc_temp]).unwrap();
        }
        */
    }

    pub unsafe fn destroy(mut self) {
        self.render_compute_pipeline.destroy(&self.device);
        log::info!("destroyed render compute pipeline");

        self.tick_voxel_compute_pipeline.destroy(&self.device);
        log::info!("destroyed tick voxel compute pipeline");

        self.generate_voxel_compute_pipeline.destroy(&self.device);
        log::info!("destroyed generate voxel compute pipeline");

        self.ray_trace_pipeline.destroy(&self.device);
        log::info!("destroyed ray trace compute pipeline");

        self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        log::info!("destroyed descriptor pool");

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

        self.svo.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed SVO buffer");

        self.ray_trace_buffers.destroy(&self.device, &mut self.allocator);
        log::info!("destroyed ray trace buffers");

        // TODO: Just cope with the error messages vro
        self.device
            .wait_for_fences(&[self.end_fence], true, u64::MAX)
            .unwrap();
        self.device.destroy_semaphore(self.begin_semaphore, None);
        self.device.destroy_semaphore(self.end_semaphore, None);
        self.device.destroy_fence(self.end_fence, None);
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        log::info!("destroyed swapchain");

        self.surface_loader.destroy_surface(self.surface_khr, None);
        log::info!("destroyed surface");

        for (image, allocation) in self.rt_images {
            self.device.destroy_image(image, None);
            self.allocator.free(allocation).unwrap();
        }
        log::info!("destroyed render target images");

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
    start: Instant,
    last: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        unsafe {
            self.internal = Some(InternalApp::new(event_loop));
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
                    dbg!(delta);
                }

                if inner.input.get_button(Button::Keyboard(KeyCode::KeyH)).pressed() {
                    inner.debug_type = (inner.debug_type + 1) % 6;
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
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        start: Instant::now(),
        last: Instant::now(),
        internal: None,
    };
    event_loop.run_app(&mut app).unwrap();
}
