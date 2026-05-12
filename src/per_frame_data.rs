use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use crate::pipeline::{self, PerFrameUniformData};

pub const FRAMES_IN_FLIGHT: usize = 3;

pub struct PerFrameDescriptorSets {
    pub compositor_per_frame: vk::DescriptorSet,
    pub rasterizer_per_frame: vk::DescriptorSet,
    pub background_rasterizer_per_frame: vk::DescriptorSet,
}

pub struct PerFrameData {
    pub present_complete_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub end_fence: vk::Fence,
    pub cmd: vk::CommandBuffer,
    pub per_frame_descriptor_sets: PerFrameDescriptorSets,
    
    pub uniform_buffer: crate::buffer::Buffer,
}

impl PerFrameData {
    pub unsafe fn create_per_frame_data(
        device: &ash::Device,
        pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        post_process_compute_pipeline: &pipeline::PostProcessPipeline,
        rasterized_pipeline: &pipeline::RasterizationRenderPipeline,
        background_rasterized_pipeline: &pipeline::RasterizationBackgroundPipeline,
        allocator: &mut Allocator,
        binder: &Option<ash::ext::debug_utils::Device>,
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

        let per_frame_descriptor_set_layouts = [post_process_compute_pipeline.descriptor_set_layout[1], rasterized_pipeline.descriptor_set_layout[0], background_rasterized_pipeline.descriptor_set_layout[0]];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&per_frame_descriptor_set_layouts);
        let all_descriptor_sets_for_frame = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .unwrap();


        let uniform_buffer = crate::buffer::create_buffer(device, allocator, size_of::<PerFrameUniformData>(), binder, "per frame uniform buffer", vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);

        Self {
            present_complete_semaphore,
            render_finished_semaphore,
            end_fence,
            cmd,
            per_frame_descriptor_sets: PerFrameDescriptorSets {
                compositor_per_frame: all_descriptor_sets_for_frame[0],
                rasterizer_per_frame: all_descriptor_sets_for_frame[1],
                background_rasterizer_per_frame: all_descriptor_sets_for_frame[2],
            },
            uniform_buffer,
        }
    }
    
    pub unsafe fn destroy_everything(self, device: &ash::Device, cmd_pool: vk::CommandPool, allocator: &mut Allocator) {
        device.destroy_semaphore(self.present_complete_semaphore, None);
        device.destroy_semaphore(self.render_finished_semaphore, None);
        device.destroy_fence(self.end_fence, None);
        log::info!("destroyed semaphores and fences frame data");            

        device.free_command_buffers(cmd_pool, &[self.cmd]);
        log::info!("destroyed cmd buffer frame data");       

        self.uniform_buffer.destroy(device, allocator);
        log::info!("destroyed per frame uniform buffer");       
    }
}