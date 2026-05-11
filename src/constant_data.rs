use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::pipeline;
use crate::samplers;
use crate::skybox;
use crate::voxel;
use crate::buffer;

pub struct ConstantData {
    pub render_compute_pipeline_descriptor_set: vk::DescriptorSet,
    pub sky_compute_pipeline_descriptor_set: vk::DescriptorSet,
    pub voxel_compute_pipeline_descriptor_set: vk::DescriptorSet,

    pub rendered_image: vk::Image,
    pub rendered_image_allocation: Option<Allocation>,
    pub rendered_depth_image: vk::Image,
    pub rendered_depth_image_allocation: Option<Allocation>,
    
    pub bloom_image: vk::Image,
    pub bloom_image_allocation: Option<Allocation>,
    pub bloom_mip_image_views: Vec<vk::ImageView>,

    pub rendered_image_view: vk::ImageView,
    pub rendered_depth_image_image_view: vk::ImageView,
    pub entire_bloom_image_view: vk::ImageView,

    pub main_render: vk::DescriptorSet,
    pub compositor: vk::DescriptorSet,
    pub compositor_downsample_bloom: Vec<vk::DescriptorSet>,
    pub compositor_upsample_bloom: Vec<vk::DescriptorSet>,
}

impl ConstantData {
    pub unsafe fn create_constant_descriptor_sets(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        render_compute_pipeline: &pipeline::RenderPipeline,
        sky_compute_pipeline: &pipeline::SkyPipeline,
        post_process_compute_pipeline: &pipeline::PostProcessPipeline,
        voxel_compute_pipeline: &pipeline::VoxelPipeline,
        samplers: &samplers::Samplers,
        skybox: &skybox::Skybox,
        svt: &voxel::SparseVoxelTexture,
        svo: &voxel::SparseVoxelOctree,
        lights_buffer: &buffer::Buffer,
    ) -> Self {
        let constant_descriptor_set_layouts = [render_compute_pipeline.descriptor_set_layout[1], sky_compute_pipeline.descriptor_set_layout[0], voxel_compute_pipeline.descriptor_set_layout[0]];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&constant_descriptor_set_layouts);
        let all_descriptor_sets = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();
        let render_compute_pipeline_descriptor_set= all_descriptor_sets[0];
        let sky_compute_pipeline_descriptor_set = all_descriptor_sets[1];
        let voxel_compute_pipeline_descriptor_set = all_descriptor_sets[2];


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
            .sampler(samplers.skybox_sampler)
            .image_layout(vk::ImageLayout::GENERAL);
        let descriptor_clouds_sampler_info = vk::DescriptorImageInfo::default()
            .image_view(skybox.clouds_image_view)
            .sampler(samplers.skybox_sampler)
            .image_layout(vk::ImageLayout::GENERAL);
        let descriptor_svt_sampler_info = vk::DescriptorImageInfo::default()
            .image_view(svt.sampled_sparse_image_view)
            .sampler(svt.sampled_sparse_image_sampler)
            .image_layout(vk::ImageLayout::GENERAL);

 
        let descriptor_svt_image_infos = [descriptor_svt_image_info, descriptor_svt_metadata_image_info];
        let descriptor_svo_buffers_infos = [descriptor_svo_bitmasks_info, descriptor_svo_indices_info, descriptor_svo_aabbs_info, descriptor_light_buffer_info];
        let descriptor_combined_image_sampler_infos = [descriptor_skybox_sampler_info, descriptor_clouds_sampler_info, descriptor_svt_sampler_info];

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
            .descriptor_count(descriptor_combined_image_sampler_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_binding(6)
            .dst_set(render_compute_pipeline_descriptor_set)
            .image_info(&descriptor_combined_image_sampler_infos);

        let voxel_compute_descriptor_images_infos = [descriptor_svt_image_info];
        let voxel_compute_descriptor_write_1 = vk::WriteDescriptorSet::default()
            .descriptor_count(voxel_compute_descriptor_images_infos.len() as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(voxel_compute_pipeline_descriptor_set)
            .image_info(&voxel_compute_descriptor_images_infos);


        device.update_descriptor_sets(&[render_compute_skybox_descriptor_write, sky_compute_descriptor_write_1, render_compute_svt_images_descriptor_write, render_compute_svo_buffers_descriptor_write, voxel_compute_descriptor_write_1], &[]);

        let per_frame_descriptor_set_layouts = [render_compute_pipeline.descriptor_set_layout[0], post_process_compute_pipeline.descriptor_set_layout[0]];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&per_frame_descriptor_set_layouts);
        let all_descriptor_sets_for_frame = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();

        Self {
            render_compute_pipeline_descriptor_set,
            sky_compute_pipeline_descriptor_set,
            voxel_compute_pipeline_descriptor_set,
            rendered_image_view: vk::ImageView::null(),
            bloom_image: vk::Image::null(),
            bloom_image_allocation: None,
            entire_bloom_image_view: vk::ImageView::null(),
            bloom_mip_image_views: Default::default(),
            rendered_image: vk::Image::null(),
            rendered_image_allocation: None,
            main_render: all_descriptor_sets_for_frame[0],
            compositor: all_descriptor_sets_for_frame[1],
            compositor_downsample_bloom: Vec::default(),
            compositor_upsample_bloom: Vec::default(),
            rendered_depth_image: vk::Image::null(),
            rendered_depth_image_allocation: None,
            rendered_depth_image_image_view: vk::ImageView::null(),
        }
    }
    
    pub unsafe fn recreate_rt_images_and_image_views_and_update_descriptor_sets(
        &mut self,
        device: &ash::Device,
        swapchain_format: vk::Format,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        queue_family_index: u32,
        extent: vk::Extent2D,
        binder: &Option<ash::ext::debug_utils::Device>,
        descriptor_pool: vk::DescriptorPool,
        samplers: &crate::samplers::Samplers,
        post_process_compute_pipeline: &pipeline::PostProcessPipeline,
        scaling_factor: u32,
    ) {
        log::debug!("recreate images & descriptor set stuff for per-frame-data...");

        let rendered_image_format = vk::Format::R16G16B16A16_SFLOAT;
        let (rendered_image, rendered_image_allocation) = create_image(device, rendered_image_format, allocator, queue_family_index, extent, binder, scaling_factor, "Tmp Rendered Texture (pre-process)", None, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED);

        let depth_image_format = vk::Format::D32_SFLOAT;
        let (depth_image, depth_image_allocation) = create_image(device, depth_image_format, allocator, queue_family_index, extent, binder, scaling_factor, "Depth Texture", None, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);

        let scaled = vek::Vec2::new(extent.width, extent.height) / scaling_factor;
        let bloom_mip_levels = scaled.map(|x| u32::ilog2(x)).reduce_min() - 2;

        let (bloom_image, bloom_image_allocation) = create_image(device, rendered_image_format, allocator, queue_family_index, extent, binder, scaling_factor, "Bloom Texture", Some(bloom_mip_levels), vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);
        
        self.rendered_image = rendered_image;
        self.rendered_image_allocation = Some(rendered_image_allocation);

        self.bloom_image = bloom_image;
        self.bloom_image_allocation = Some(bloom_image_allocation);

        self.rendered_depth_image = depth_image;
        self.rendered_depth_image_allocation = Some(depth_image_allocation);


        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let detph_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .level_count(1)
            .layer_count(1);
        let entire_bloom_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(1);

        let rendered_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(rendered_image_format)
            .image(self.rendered_image)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let rendered_depth_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(depth_image_format)
            .image(self.rendered_depth_image)
            .subresource_range(detph_subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        let entire_bloom_image_view_create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .flags(vk::ImageViewCreateFlags::empty())
            .format(rendered_image_format)
            .image(self.bloom_image)
            .subresource_range(entire_bloom_subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);

        self.rendered_image_view = device
            .create_image_view(&rendered_image_view_create_info, None)
            .unwrap();
        self.entire_bloom_image_view = device
            .create_image_view(&entire_bloom_image_view_create_info, None)
            .unwrap();
        self.rendered_depth_image_image_view = device
            .create_image_view(&rendered_depth_image_view_create_info, None)
            .unwrap();

        // allocate descriptor sets for downsample
        let bloom_pass_descriptor_set_layouts_repeated_n_times = std::iter::repeat(post_process_compute_pipeline.descriptor_set_layout[2]).take(bloom_mip_levels as usize).collect::<Vec<_>>();
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&bloom_pass_descriptor_set_layouts_repeated_n_times);
        let bloom_descriptor_sets = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();
        self.compositor_downsample_bloom = bloom_descriptor_sets;


        // allocate descriptor sets for upsample
        let bloom_descriptor_sets = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();
        self.compositor_upsample_bloom = bloom_descriptor_sets;


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
            let dst_bloom_descriptor_set = self.compositor_downsample_bloom[mip_level as usize];

            let descriptor_image_write = if mip_level == 0 {
                // first mip will read immediately from the rendered texture
                vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.rendered_image_view).sampler(samplers.bloom_sampler)
            } else {
                vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.bloom_mip_image_views[mip_level as usize]).sampler(samplers.bloom_sampler)
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
            let dst_bloom_descriptor_set = self.compositor_upsample_bloom[mip_level as usize];

            // write previous mip to descriptor set
            let descriptor_image_write = vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.bloom_mip_image_views[mip_level as usize + 1]).sampler(samplers.bloom_sampler);
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

        let descriptor_rendered_image_view_info = vk::DescriptorImageInfo::default()
            .image_view(self.rendered_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::null());
        let descriptor_entire_bloom_image_view_info = vk::DescriptorImageInfo::default()
            .image_view(self.entire_bloom_image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(samplers.bloom_sampler);

        // rendered image for raytracer (write only)
        let render_compute_descriptor_image_infos = [descriptor_rendered_image_view_info];
        let render_compute_image_descriptor_write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(self.main_render)
            .image_info(&render_compute_descriptor_image_infos);

        // rendered image for compositor (read only)
        let composition_compute_descriptor_image_infos_2 = [descriptor_rendered_image_view_info];
        let composition_compute_image_descriptor_write_2 = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .dst_set(self.compositor)
            .image_info(&composition_compute_descriptor_image_infos_2);
        
        // entire bloom image for compositor (read only)
        let composition_compute_descriptor_image_infos_3 = [descriptor_entire_bloom_image_view_info];
        let composition_compute_image_descriptor_write_3 = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_binding(1)
            .dst_set(self.compositor)
            .image_info(&composition_compute_descriptor_image_infos_3);
        

        log::debug!("updating the other types of descriptor sets now...");
        device.update_descriptor_sets(&[render_compute_image_descriptor_write, composition_compute_image_descriptor_write_2, composition_compute_image_descriptor_write_3], &[]);
    }
    
    pub unsafe fn destroy_rt_images_and_image_views(&mut self, device: &ash::Device, descriptor_pool: vk::DescriptorPool, allocator: &mut gpu_allocator::vulkan::Allocator) {
        device.destroy_image_view(self.rendered_image_view, None);
        device.destroy_image_view(self.entire_bloom_image_view, None);
        device.destroy_image_view(self.rendered_depth_image_image_view, None);
        log::info!("destroyed image views");

        for image_view in self.bloom_mip_image_views.iter() {
            device.destroy_image_view(*image_view, None);
            log::info!("destroyed bloom image view");
        }
        self.bloom_mip_image_views.clear();
    
        device.destroy_image(self.rendered_image, None);
        allocator.free(self.rendered_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed rendered image");

        device.destroy_image(self.bloom_image, None);
        allocator.free(self.bloom_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed bloom image");

        device.destroy_image(self.rendered_depth_image, None);
        allocator.free(self.rendered_depth_image_allocation.take().unwrap()).unwrap();
        log::info!("destroyed depth image");

        // the ONLY descriptor sets that need to be freed are the ones used by the bloom pass (since their number changes with the log2 of the screen resolution!)
        // the other per-frame descriptor sets don't need to be freed, we ARE in the per-frame data structure! we pre-allocated them for a reason!
        let mut desc_sets_to_free = Vec::<vk::DescriptorSet>::new();
        desc_sets_to_free.extend_from_slice(&self.compositor_downsample_bloom);
        desc_sets_to_free.extend_from_slice(&self.compositor_upsample_bloom);
        
        device.free_descriptor_sets(descriptor_pool, &desc_sets_to_free).unwrap();
        log::info!("freed bloom descriptor sets");

        self.compositor_downsample_bloom.clear();
        self.compositor_upsample_bloom.clear();
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
    usage: vk::ImageUsageFlags,
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
        .usage(usage)
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
    const_data: &ConstantData,
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
    let depth_subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::DEPTH)
        .level_count(1)
        .layer_count(1);

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
        .image(const_data.rendered_image)
        .subresource_range(subresource_range);

    let depth_image_transition = vk::ImageMemoryBarrier2::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
        .src_queue_family_index(queue_family_index)
        .dst_queue_family_index(queue_family_index)
        .image(const_data.rendered_depth_image)
        .subresource_range(depth_subresource_range);

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
        .image(const_data.bloom_image)
        .subresource_range(bloom_subresource_range);

    let barriers = [rendered_image_transition, bloom_image_transition, depth_image_transition];
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
