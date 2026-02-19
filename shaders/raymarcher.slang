#language slang 2026
#include <ray_stuff_other.slang>
#include <lighting.slang>
#include <pbr.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(1, 0)]]
StructuredBuffer<uint64_t> svo_bitmasks_buffer;

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> svo_indices_buffer;

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
void main(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform DebugType debug_type) {
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;

    float3 ray_dir = normalize((mul(mat, float4(screen_uvs, 1, 0))).xyz);
    float3 ray_pos = position.xyz;

    SparseVoxelOctree svo = SparseVoxelOctree(svo_indices_buffer, svo_bitmasks_buffer);

    /*
    bool hit = any_hit_trace_fine(ray_pos, ray_dir, svo);
    */
    int iter = 0;
    DdaTraversalOutput ray = trace_shi(ray_pos, ray_dir, svo, iter);
    //output[id.xy] = float4(ray.hit ? 1 : 0);
    output[id.xy] = float4(iter / 32.0);
    //output[id.xy] = float4(ray.hit ? 1 : 0);
    /*
    GroupMemoryBarrierWithWaveSync();
    AllMemoryBarrierWithGroupSync();

    float3 color = 0.0;
    if (ray.hit) {
        float3 normal = get_face_normal(ray.face, sign(ray_dir));
        float3 diffuse = 1.0;
        float ao = 0.0;

        float3 d = reflect(ray_dir, normal);
        float shadow = 1.0;
        float sky_visibility = 1.0;
        PbrSurfaceData pbr_surface_data = PbrSurfaceData(diffuse, normal, 0.95, 0.0, ao, shadow, sky_visibility);
        CameraData camera_data = CameraData(-ray_dir);
        SunData sun_data = SunData(sun.xyz, sun.y > 0 ? 0.8 : 0);
        //color = normal;
        color = brdf(pbr_surface_data, camera_data, sun_data);
        //color = block_pos;
    } else {
        color = sky(sun.xyz, ray_dir);
    }

    //output[id.xy] = float4(((float3)tst.block_pos % 10) / 10.0, 0);
    output[id.xy] = float4(aces(color), 0);
    */
}