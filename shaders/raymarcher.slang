#language slang 2026
#include <lighting.slang>
#include <pbr.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(1, 0)]]
RWStructuredBuffer<RayInput> ray_inputs;

[[vk::binding(2, 0)]]
RWStructuredBuffer<RayOutput> ray_outputs;

enum DebugType: uint {
    Combined,
    Iterations,
    Recursions,
    FullHalt,
    World,
    Timer,
}

[shader("compute")]
[numthreads(8, 8, 1)]
void generateRays(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform DebugType debug_type) {
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;
    
    float3 ray_dir = normalize((mul(mat, float4(screen_uvs, 1, 0))).xyz);
    float3 ray_pos = position.xyz;

    uint index = encode_ray_index(id.xy, lid.xy, gid.xy, (uint2)screen);
    ray_inputs[index] = RayInput(float4(ray_pos, 0), float4(ray_dir, 0));
    output[id.xy] = float4(0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void applyRays(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform DebugType debug_type) {
    uint index = encode_ray_index(id.xy, lid.xy, gid.xy, (uint2)screen);
    RayOutput result = ray_outputs[index];
    //output[id.xy] = float4(result.hit ? 1 : 0, result.hit2 ? 1 : 0, 0, 0);
    output[id.xy] = float4((result.data % 100) / 100.0);
}