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
[numthreads(32, 32, 1)]
void write_clouds(uint3 id: SV_DispatchThreadID, uniform float4 sun_and_time) {
    float2 uv = id.xy / (float)CLOUDS_RESOLUTION;
    float3 sun = sun_and_time.xyz;
    float time = sun_and_time.w;

    float2 pos = (uv - 0.5) * 8000;

    float amplitude = 0.7;
    float scale = 2.3;
    float3 value = 0;

    for (int i = 0; i < 4; i++) {
        value += sdnoise(pos * pow(scale, i) * 0.0015 + time * 0.03) * pow(amplitude, i);
    }

    //value *= smoothstep(0, 1, snoise(pos * 0.0005) * 1.5 - 0.2);

    // scale opacity by distance
    float opacity = value.x;
    float2 derivatives = value.yz;

    //opacity *= saturate(3000-length(pos));

    float3 ray_dir = normalize(float3(pos.x, 800, pos.y));

    float3 bottom_normals = normalize(float3(derivatives.x, -1, derivatives.y));
    float3 top_normals = normalize(float3(-derivatives.x, 1, -derivatives.y));

    float3 col = lerp(saturate(dot(sun, top_normals)), 0.05, 1-smoothstep(0, 0.2, sun.y));

    int2 offset = (int2)((hash22(pos * 0.554684) * 2.0 - 1.0) * 1);
    float4 old = clouds_image[(uint2)(id.xy+offset)];
    AllMemoryBarrier();
    clouds_image[id.xy] = lerp(float4(col, clamp(opacity, 0, 1)), old, 0.95);
}

[shader("compute")]
[numthreads(32, 32, 1)]
void write_skybox(uint3 id: SV_DispatchThreadID, uniform float4 sun_and_time) {
    float2 uv = ((id.xy / (float)SKYBOX_RESOLUTION) * 2.0 - 1.0);
    float3 sun = sun_and_time.xyz;
    float time = sun_and_time.w;
    
    float3 coords = 0;

    // -X
    if (id.z == 0) {
        coords = float3(-1, -uv.y, uv.x); 
    }

    // +X
    if (id.z == 1) {
        coords = float3(1, -uv.y, -uv.x);
    }

    // -Y
    if (id.z == 3) {
        coords = float3(-uv.x, -1, uv.y);
    }

    // +Y
    if (id.z == 2) {
        coords = float3(-uv.x, 1, -uv.y);
    }

    // -Z
    if (id.z == 4) {
        coords = float3(-uv.x, -uv.y, -1);
    }

    // +Z
    if (id.z == 5) {
        coords = float3(uv.x, -uv.y, 1);
    }

    float3 ray_dir = normalize(coords);
    ray_dir.xz = -ray_dir.xz;
    
    float3 sky_colour = sky(sun, ray_dir, 0.0, true, true);
    skybox_image.Store(id, float4(sky_colour, 0));
}