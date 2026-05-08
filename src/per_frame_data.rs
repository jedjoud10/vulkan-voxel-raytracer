use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::pipeline;

pub struct PerFrameDescriptorSets {
    pub main_render_per_frame: vk::DescriptorSet,
    pub compositor_per_frame: vk::DescriptorSet,
    pub compositor_downsample_bloom_per_frame: Vec<vk::DescriptorSet>,
    pub compositor_upsample_bloom_per_frame: Vec<vk::DescriptorSet>,
}

pub struct PerFrameData {
    pub swapchain_image: vk::Image,
    pub rt_image: vk::Image,
    pub rt_image_allocation: Option<Allocation>,
    pub rendered_image: vk::Image,
    pub rendered_image_allocation: Option<Allocation>,
    
    pub bloom_image: vk::Image,
    pub bloom_image_allocation: Option<Allocation>,
    pub bloom_sampler: vk::Sampler,
    pub bloom_mip_image_views: Vec<vk::ImageView>,

    pub present_complete_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub end_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,
    pub per_frame_descriptor_sets: PerFrameDescriptorSets,
    
    pub rendered_image_view: vk::ImageView,
    pub rt_image_view: vk::ImageView,
    pub swapchain_image_view: vk::ImageView,
    pub entire_bloom_image_view: vk::ImageView,
}

impl PerFrameData {
    pub unsafe fn create_per_frame_data(
        device: &ash::Device,
        pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        render_compute_pipeline: &pipeline::RenderPipeline,
        post_process_compute_pipeline: &pipeline::PostProcessPipeline,
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

        let per_frame_descriptor_set_layouts = [render_compute_pipeline.descriptor_set_layout[0], post_process_compute_pipeline.descriptor_set_layout[0]];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&per_frame_descriptor_set_layouts);
        let all_descriptor_sets_for_frame = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();

        // TODO: don't fucking create this per frame you absolute fucking DUMBASS
        let bloom_sampler_create_info = vk::SamplerCreateInfo::default()
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .max_lod(100f32)
            .min_lod(0f32);
        let bloom_sampler = device.create_sampler(&bloom_sampler_create_info, None).unwrap();

        Self {
            swapchain_image,
            rt_image: vk::Image::null(),
            rt_image_allocation: None,
            rendered_image: vk::Image::null(),
            rendered_image_allocation: None,
            present_complete_semaphore,
            render_finished_semaphore,
            end_fence,
            cmd,
            per_frame_descriptor_sets: PerFrameDescriptorSets {
                main_render_per_frame: all_descriptor_sets_for_frame[0],
                compositor_per_frame: all_descriptor_sets_for_frame[1],
                compositor_downsample_bloom_per_frame: Vec::default(),
                compositor_upsample_bloom_per_frame: Vec::default(),
            },
            rt_image_view: vk::ImageView::null(),
            swapchain_image_view: vk::ImageView::null(),
            rendered_image_view: vk::ImageView::null(),
            bloom_image: vk::Image::null(),
            bloom_image_allocation: None,
            entire_bloom_image_view: vk::ImageView::null(),
            bloom_sampler,
            bloom_mip_image_views: Default::default(),
        }
    }

    pub unsafe fn recreate_rt_images_and_image_views_and_update_descriptor_sets(
        &mut self,
        device: &ash::Device,
        swapchain_format: vk::Format,
        swapchain_image: vk::Image,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        queue_family_index: u32,
        extent: vk::Extent2D,
        binder: &Option<ash::ext::debug_utils::Device>,
        descriptor_pool: vk::DescriptorPool,
        post_process_compute_pipeline: &pipeline::PostProcessPipeline,
        scaling_factor: u32,
    ) {
        log::debug!("recreate images & descriptor set stuff for per-frame-data...");
        self.swapchain_image = swapchain_image;


        let (rt_image, rt_image_allocation) = create_image(device, swapchain_format, allocator, queue_family_index, extent, binder, scaling_factor, "Render Texture (post-process)", None);
        self.rt_image = rt_image;
        self.rt_image_allocation = Some(rt_image_allocation);
        

        let rendered_image_format = vk::Format::R16G16B16A16_SFLOAT;
        let (rendered_image, rendered_image_allocation) = create_image(device, rendered_image_format, allocator, queue_family_index, extent, binder, scaling_factor, "Tmp Rendered Texture (pre-process)", None);

        let scaled = vek::Vec2::new(extent.width, extent.height) / scaling_factor;
        let bloom_mip_levels = scaled.map(|x| u32::ilog2(x)).reduce_min() - 2;

        let (bloom_image, bloom_image_allocation) = create_image(device, rendered_image_format, allocator, queue_family_index, extent, binder, scaling_factor, "Bloom Texture", Some(bloom_mip_levels));
        
        self.rendered_image = rendered_image;
        self.rendered_image_allocation = Some(rendered_image_allocation);

        self.bloom_image = bloom_image;
        self.bloom_image_allocation = Some(bloom_image_allocation);


        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let entire_bloom_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(1);

        let rt_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(swapchain_format)
            .image(rt_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let swapchain_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(swapchain_format)
            .image(self.swapchain_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let rendered_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(rendered_image_format)
            .image(self.rendered_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let entire_bloom_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(rendered_image_format)
            .image(self.bloom_image)
            .subresource_range(entire_bloom_subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        self.rt_image_view = device
            .create_image_view(&rt_image_view_create_info, None)
            .unwrap();
        self.swapchain_image_view = device
            .create_image_view(&swapchain_image_view_create_info, None)
            .unwrap();
        self.rendered_image_view = device
            .create_image_view(&rendered_image_view_create_info, None)
            .unwrap();
        self.entire_bloom_image_view = device
            .create_image_view(&entire_bloom_image_view_create_info, None)
            .unwrap();

        // allocate descriptor sets for downsample
        let bloom_pass_descriptor_set_layouts_repeated_n_times = std::iter::repeat(post_process_compute_pipeline.descriptor_set_layout[1]).take(bloom_mip_levels as usize).collect::<Vec<_>>();
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&bloom_pass_descriptor_set_layouts_repeated_n_times);
        let bloom_descriptor_sets = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();
        self.per_frame_descriptor_sets.compositor_downsample_bloom_per_frame = bloom_descriptor_sets;


        // allocate descriptor sets for upsample
        let bloom_descriptor_sets = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();
        self.per_frame_descriptor_sets.compositor_upsample_bloom_per_frame = bloom_descriptor_sets;


        self.bloom_mip_image_views.clear();

        // create bloom image views
        log::debug!("creating bloom image views...");
        for mip_level in 0..bloom_mip_levels {
            let bloom_image_view_create_info = vk::ImageViewCreateInfo::default()
                .components(vk::ComponentMapping::default())
                .flags(vk::ImageViewCreateFlags::empty())
                .format(rendered_image_format)
                .image(self.bloom_image)
                .subresource_range(vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .base_mip_level(mip_level)
                    .level_count(1)
                )
                .view_type(vk::ImageViewType::TYPE_2D);


            let image_view = device
                .create_image_view(&bloom_image_view_create_info, None)
                .unwrap();
            self.bloom_mip_image_views.push(image_view);
        }

        // update downsample bloom descriptor sets
        log::debug!("updating downsample bloom descriptor sets...");
        for mip_level in 0..(bloom_mip_levels-1) {
            let dst_bloom_descriptor_set = self.per_frame_descriptor_sets.compositor_downsample_bloom_per_frame[mip_level as usize];

            let descriptor_image_write = if mip_level == 0 {
                // first mip will read immediately from the rendered texture
                vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.rendered_image_view).sampler(self.bloom_sampler)
            } else {
                vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.bloom_mip_image_views[mip_level as usize]).sampler(self.bloom_sampler)
            };
            
            // write previous mip to descriptor set
            let descriptor_image_writes = [descriptor_image_write]; 
            let previous_mip_descriptor_write = vk::WriteDescriptorSet::default()
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(0)
                .dst_set(dst_bloom_descriptor_set)
                .image_info(&descriptor_image_writes);
            
            // write next mip to descriptor set
            let descriptor_image_write = vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.bloom_mip_image_views[mip_level as usize + 1]);
            let descriptor_image_writes = [descriptor_image_write];
            let next_mip_descriptor_write = vk::WriteDescriptorSet::default()
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .dst_binding(1)
                .dst_set(dst_bloom_descriptor_set)
                .image_info(&descriptor_image_writes);

            device.update_descriptor_sets(&[previous_mip_descriptor_write, next_mip_descriptor_write], &[]);
        }

        
        // update upsample bloom descriptor sets
        log::debug!("updating upsample bloom descriptor sets...");
        for mip_level in 0..(bloom_mip_levels-1) {
            let dst_bloom_descriptor_set = self.per_frame_descriptor_sets.compositor_upsample_bloom_per_frame[mip_level as usize];

            // write previous mip to descriptor set
            let descriptor_image_write = vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.bloom_mip_image_views[mip_level as usize + 1]).sampler(self.bloom_sampler);
            let descriptor_image_writes = [descriptor_image_write]; 
            let previous_mip_descriptor_write = vk::WriteDescriptorSet::default()
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(0)
                .dst_set(dst_bloom_descriptor_set)
                .image_info(&descriptor_image_writes);
            
            // write next mip to descriptor set
            let descriptor_image_write = vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.bloom_mip_image_views[mip_level as usize]);
            let descriptor_image_writes = [descriptor_image_write];
            let next_mip_descriptor_write = vk::WriteDescriptorSet::default()
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .dst_binding(1)
                .dst_set(dst_bloom_descriptor_set)
                .image_info(&descriptor_image_writes);

            device.update_descriptor_sets(&[previous_mip_descriptor_write, next_mip_descriptor_write], &[]);
        }

        let descriptor_rt_image_view_info = vk::DescriptorImageInfo::default()
            .image_view(self.rt_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_rendered_image_view_info = vk::DescriptorImageInfo::default()
            .image_view(self.rendered_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_entire_bloom_image_view_info = vk::DescriptorImageInfo::default()
            .image_view(self.entire_bloom_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(self.bloom_sampler);

        // rendered image for raytracer (write only)
        let render_compute_descriptor_image_infos = [descriptor_rendered_image_view_info];
        let render_compute_image_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(self.per_frame_descriptor_sets.main_render_per_frame)
            .image_info(&render_compute_descriptor_image_infos);

        // rendered image for compositor (read only)
        let composition_compute_descriptor_image_infos_2 = [descriptor_rendered_image_view_info];
        let composition_compute_image_descriptor_write_2 = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(self.per_frame_descriptor_sets.compositor_per_frame)
            .image_info(&composition_compute_descriptor_image_infos_2);

        // rt image for compositor (write only)
        let composition_compute_descriptor_image_infos_1 = [descriptor_rt_image_view_info];
        let composition_compute_image_descriptor_write_1 = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(1)
            .dst_set(self.per_frame_descriptor_sets.compositor_per_frame)
            .image_info(&composition_compute_descriptor_image_infos_1);
        
        // entire bloom image for compositor (read only)
        let composition_compute_descriptor_image_infos_3 = [descriptor_entire_bloom_image_view_info];
        let composition_compute_image_descriptor_write_3 = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_binding(2)
            .dst_set(self.per_frame_descriptor_sets.compositor_per_frame)
            .image_info(&composition_compute_descriptor_image_infos_3);
        

        log::debug!("updating the other types of descriptor sets now...");
        device.update_descriptor_sets(&[render_compute_image_descriptor_write, composition_compute_image_descriptor_write_1, composition_compute_image_descriptor_write_2, composition_compute_image_descriptor_write_3], &[]);
    }
    
    pub unsafe fn destroy_rt_images_and_image_views(&mut self, device: &ash::Device, descriptor_pool: vk::DescriptorPool, allocator: &mut gpu_allocator::vulkan::Allocator) {
        device.destroy_image_view(self.swapchain_image_view, None);
        device.destroy_image_view(self.rt_image_view, None);
        device.destroy_image_view(self.rendered_image_view, None);
        device.destroy_image_view(self.entire_bloom_image_view, None);
        log::info!("destroyed image views for frame data");

        for image_view in self.bloom_mip_image_views.iter() {
            device.destroy_image_view(*image_view, None);
            log::info!("destroyed bloom image view");
        }
        self.bloom_mip_image_views.clear();

        device.destroy_image(self.rt_image, None);
        allocator.free(self.rt_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed render target image frame data");
    
        device.destroy_image(self.rendered_image, None);
        allocator.free(self.rendered_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed rendered image frame data");

        device.destroy_image(self.bloom_image, None);
        allocator.free(self.bloom_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed bloom image frame data");

        // the ONLY descriptor sets that need to be freed are the ones used by the bloom pass (since their number changes with the log2 of the screen resolution!)
        // the other per-frame descriptor sets don't need to be freed, we ARE in the per-frame data structure! we pre-allocated them for a reason!
        let mut desc_sets_to_free = Vec::<vk::DescriptorSet>::new();
        desc_sets_to_free.extend_from_slice(&self.per_frame_descriptor_sets.compositor_downsample_bloom_per_frame);
        desc_sets_to_free.extend_from_slice(&self.per_frame_descriptor_sets.compositor_upsample_bloom_per_frame);
        
        device.free_descriptor_sets(descriptor_pool, &desc_sets_to_free).unwrap();
        log::info!("freed descriptor sets");

        self.per_frame_descriptor_sets.compositor_downsample_bloom_per_frame.clear();
        self.per_frame_descriptor_sets.compositor_upsample_bloom_per_frame.clear();
    }

    pub unsafe fn destroy_everything(mut self, device: &ash::Device, cmd_pool: vk::CommandPool, descriptor_pool: vk::DescriptorPool, allocator: &mut gpu_allocator::vulkan::Allocator) {
        self.destroy_rt_images_and_image_views(device, descriptor_pool, allocator);

        device.destroy_semaphore(self.present_complete_semaphore, None);
        device.destroy_semaphore(self.render_finished_semaphore, None);
        device.destroy_fence(self.end_fence, None);
        log::info!("destroyed semaphores and fences frame data");            

        device.free_command_buffers(cmd_pool, &[self.cmd]);
        log::info!("destroyed cmd buffer frame data");      

        device.destroy_sampler(self.bloom_sampler, None);
        log::info!("destroyed bloom sampler");            
    }
}

unsafe fn create_image(
    device: &ash::Device,
    format: vk::Format,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    queue_family_index: u32,
    extent: vk::Extent2D,
    binder: &Option<ash::ext::debug_utils::Device>,
    scaling_factor: u32,
    name: &str,
    mip_levels: Option<u32>,
) -> (vk::Image, Allocation) {
    let queue_family_indices = [queue_family_index];
    let image_create_info = vk::ImageCreateInfo::default()
        .extent(vk::Extent3D {
            width: extent.width / scaling_factor,
            height: extent.height / scaling_factor,
            depth: 1,
        })
        .format(format)
        .image_type(vk::ImageType::TYPE_2D)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .mip_levels(mip_levels.unwrap_or(1))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .samples(vk::SampleCountFlags::TYPE_1)
        .queue_family_indices(&queue_family_indices)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1);
    let image = device.create_image(&image_create_info, None).unwrap();
    let requirements = device.get_image_memory_requirements(image);

    let image_allocation = allocator
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: &format!("{name} Image Allocation"),
            requirements,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            location: gpu_allocator::MemoryLocation::GpuOnly,
        })
        .unwrap();

    device
        .bind_image_memory(image, image_allocation.memory(), image_allocation.offset())
        .unwrap();

    crate::debug::set_object_name(image, binder, name);
    (image, image_allocation)
}

pub unsafe fn transfer_layout_for_images(
    device: &ash::Device,
    queue_family_index: u32,
    per_frame_data: &[PerFrameData],
    pool: vk::CommandPool,
    queue: vk::Queue,
) {
    let cmd_buffer_create_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(pool);
    let cmd = device
        .allocate_command_buffers(&cmd_buffer_create_info)
        .unwrap()[0];

    let cmd_buffer_begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device
        .begin_command_buffer(cmd, &cmd_buffer_begin_info)
        .unwrap();

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let mut barriers = Vec::<vk::ImageMemoryBarrier2>::new();

    for frame_in_flight in per_frame_data.iter() {
        let rt_image_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(frame_in_flight.rt_image)
            .subresource_range(subresource_range);

        let rendered_image_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(frame_in_flight.rendered_image)
            .subresource_range(subresource_range);

        let bloom_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(1);
        let bloom_image_transition = vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_queue_family_index(queue_family_index)
            .dst_queue_family_index(queue_family_index)
            .image(frame_in_flight.bloom_image)
            .subresource_range(bloom_subresource_range);

        barriers.extend_from_slice(&[rt_image_transition, rendered_image_transition, bloom_image_transition]);
    }

    let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    device.cmd_pipeline_barrier2(cmd, &dep);

    
    device.end_command_buffer(cmd).unwrap();
    let cmds = [cmd];
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&cmds);
    device.queue_submit(queue, &[submit_info], vk::Fence::null()).unwrap();
    device.device_wait_idle().unwrap();
    device.free_command_buffers(pool, &[cmd]);
}
