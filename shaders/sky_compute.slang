#language slang 2026

#include "noises.slang"
#include "sky.slang"

[[vk::binding(0, 0)]]
RWTexture2DArray<float4> skybox_image;

[[vk::binding(1, 0)]]
RWTexture2D<float4> clouds_image;

[SpecializationConstant] const uint SKYBOX_RESOLUTION = 32;
[SpecializationConstant] const uint CLOUDS_RESOLUTION = 32;

[shader("compute")]
[numthreads(8, 8, 1)]
void write_clouds(uint3 id: SV_DispatchThreadID, uniform float4 sun_and_time) {
    float2 uv = id.xy / (float)CLOUDS_RESOLUTION;
    float3 sun = sun_and_time.xyz;
    float time = sun_and_time.w;

    float2 pos = (uv-0.5) * 8000;

    float amplitude = 0.7;
    float scale = 2.3;
    float3 value = 0;

    for (int i = 0; i < 4; i++) {
        let noised = sdnoise(pos * pow(scale, i) * 0.0015 + time * 0.03) * pow(amplitude, i);
        value += noised;
    }

    // scale opacity by another noisy function
    float opacity = value.x * smoothstep(0, 1, snoise(pos * 0.0005) * 1.5 - 0.2) * 2.0;
    float2 derivatives = value.yz;

    //opacity *= saturate(3000-length(pos));

    float3 ray_dir = normalize(float3(pos.x, 800, pos.y));

    float3 bottom_normals = normalize(float3(derivatives.x, -1, derivatives.y));
    float3 top_normals = normalize(float3(-derivatives.x, 1, -derivatives.y));

    // factor to use to "grey out" the clouds when the sun is down
    float sun_strength = smoothstep(0, 0.2, sun.y);

    // backlight from the sun that passes through the clouds
    float3 scattered = saturate(pow(saturate(dot(ray_dir, sun)) + 0.3, 4)) * sun_strength;

    // some specular "silver lining"
    float3 reflected = reflect(sun, bottom_normals);
    float3 silver_lining = pow(saturate(dot(ray_dir, reflected)), 0.5) * sun_strength;

    float3 ambient = sky(sun, top_normals, 0, false, false);

    // final composition
    float3 col = (lerp(silver_lining * 0.3, 1.4, scattered) + 0.4) * (ambient * 1.0 + 0.3);
    clouds_image[id.xy] = float4(col, saturate(opacity));
}

float3 uv_coords_to_ray_dir(float2 uv, int cubemap_face) {
    float3 coords = 0;

    // -X
    if (cubemap_face == 0) {
        coords = float3(-1, -uv.y, uv.x); 
    }

    // +X
    if (cubemap_face == 1) {
        coords = float3(1, -uv.y, -uv.x);
    }

    // -Y
    if (cubemap_face == 3) {
        coords = float3(-uv.x, -1, uv.y);
    }

    // +Y
    if (cubemap_face == 2) {
        coords = float3(-uv.x, 1, -uv.y);
    }

    // -Z
    if (cubemap_face == 4) {
        coords = float3(-uv.x, -uv.y, -1);
    }

    // +Z
    if (cubemap_face == 5) {
        coords = float3(uv.x, -uv.y, 1);
    }

    float3 ray_dir = normalize(coords);
    ray_dir.xz = -ray_dir.xz;
    return ray_dir;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void write_skybox(uint3 id: SV_DispatchThreadID, uniform float4 sun_and_time) {
    float2 uv = ((id.xy / (float)SKYBOX_RESOLUTION) * 2.0 - 1.0);
    float3 sun = sun_and_time.xyz;
    float time = sun_and_time.w;
    
    float3 ray_dir = uv_coords_to_ray_dir(uv, id.z);
    float3 sky_colour = sky(sun, ray_dir, 0.0, true, true);
    skybox_image.Store(id, float4(sky_colour, 0));
}


[shader("compute")]
[numthreads(8, 8, 1)]
void convolute_skybox(uint3 id: SV_DispatchThreadID, uniform float4 sun_and_time) {
    float2 uv = ((id.xy / (float)SKYBOX_RESOLUTION) * 2.0 - 1.0);
    
    //float3 ray_dir = uv_coords_to_ray_dir(uv, id.z);
}