#language slang 2026
#include "sky.slang"

[[vk::binding(0, 0)]]
RWTexture2DArray<float4> skybox;

[SpecializationConstant] const uint RESOLUTION = 32;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform float4 sun) {
    float2 uv = ((id.xy / (float)RESOLUTION) * 2.0 - 1.0);
    
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
    
    float3 col = sky(sun.xyz, ray_dir, 0, true, true);
    skybox.Store(id, float4(col, 0));
}