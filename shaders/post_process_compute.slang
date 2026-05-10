#language slang 2026
#include <ray_stuff_other.slang>
#include <lighting.slang>
#include <pbr.slang>
#include <noises.slang>

[[vk::binding(0, 0)]]
readonly RWTexture2D<float4> pre_compose;

[[vk::binding(1, 0)]]
Sampler2D<float4> entire_bloom_image;

[[vk::binding(0, 1)]]
RWTexture2D<float4> swapchain_image;

[SpecializationConstant] const uint DOWNSCALE_FACTOR = 1;

//#define SHORT_CIRCUIT_NO_TONEMAPPING

[shader("compute")]
[numthreads(8, 8, 1)]
void write_swapchain_image(uint3 id: SV_DispatchThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform uint debug_type, uniform float time) {
#ifdef SHORT_CIRCUIT_NO_TONEMAPPING
    swapchain_image[id.xy] = pre_compose[id.xy / DOWNSCALE_FACTOR];
    return;
#endif

    uint2 src_coords = id.xy / DOWNSCALE_FACTOR;
    uint2 dst_coords = id.xy;
    
    
    float2 screen_uvs_negative_one_to_one = ((float2)src_coords) / screen;
    screen_uvs_negative_one_to_one *= 2.0;
    screen_uvs_negative_one_to_one -= 1.0;
    screen_uvs_negative_one_to_one.y = -screen_uvs_negative_one_to_one.y;
    screen_uvs_negative_one_to_one.x = -screen_uvs_negative_one_to_one.x;

    float2 screen_uvs = -screen_uvs_negative_one_to_one * 0.5 + 0.5;

    float3 final_colour = pre_compose[src_coords].xyz;

    /*
    float total = 1;
    if (pre_post_process.w > 0.3) {
        float blur_size = pre_post_process.w;

        for (int x = -4; x <= 4; x++) {
            for (int y = -4; y <= 4; y++) {
                if (x != 0 && y != 0) {
                    float strength = 1/length(int2(x,y));
                    total += strength;

                    float4 sample = pre_compose[id.xy + int2(x,y) * blur_size];

                    if (sample.w > 0.3) {
                        final_colour += sample.xyz * strength;
                    }
                }
            }    
        }
        
        final_colour /= total;
    }
    */
    
    float3 bloom = entire_bloom_image.SampleLevel(screen_uvs, 2.0).xyz;

    float3 col = pow(aces(final_colour + bloom * 0.05), 1/2.2);
    
    //float vignette = saturate(length(abs(screen_uvs_negative_one_to_one)) * 0.8 - 0.8);
    //col = lerp(col, 0.0, vignette);


    swapchain_image[dst_coords] = float4(col, 0);
}


[[vk::binding(0, 0)]]
Sampler2D<float4> previous_bloom_mip;

[[vk::binding(1, 0)]]
RWTexture2D<float4> next_bloom_mip;

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_downsample(uint3 id: SV_DispatchThreadID, uniform float2 previous_bloom_mip_resolution) {
    float4 sum = 0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 uvs = ((float2)(id.xy * 2) + float2(x,y) + 1) / previous_bloom_mip_resolution;
            float4 loaded = previous_bloom_mip.Sample(uvs);

            if (length(loaded) > 0.6) {
                sum += clamp(loaded, 0, 1000);
            }
        }
    }

    next_bloom_mip[id.xy] = sum / 9;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_upsample(uint3 id: SV_DispatchThreadID, uniform float2 previous_bloom_mip_resolution) {
    float4 sum = 0;
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            float2 uvs = ((float2)(id.xy + int2(x,y))) * 0.5 / previous_bloom_mip_resolution;
            sum += previous_bloom_mip.Sample(uvs);
        }
    }
    
    if (!any(isnan(sum))) {
        next_bloom_mip[id.xy] = sum / 4.0;
    }
}