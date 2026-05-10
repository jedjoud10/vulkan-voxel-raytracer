use ash::vk;

pub struct Samplers {
    pub bloom_sampler: vk::Sampler,
    pub skybox_sampler: vk::Sampler,
}

impl Samplers {
    pub unsafe fn create_samplers(device: &ash::Device) -> Self {
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

        let skybox_sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT);

        let skybox_sampler = device.create_sampler(&skybox_sampler_create_info, None).unwrap();

        Self {
            bloom_sampler,
            skybox_sampler,
        }
    }

    pub unsafe fn destroy_samplers(self, device: &ash::Device) {
        device.destroy_sampler(self.bloom_sampler, None);
        device.destroy_sampler(self.skybox_sampler, None);
    }
}