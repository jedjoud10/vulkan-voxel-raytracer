#language slang 2026
#include <ray_stuff_other.slang>
#include <lighting.slang>
#include <pbr.slang>
#include <noises.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(1, 0)]]
StructuredBuffer<uint64_t> svo_bitmasks_buffer;

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> svo_indices_buffer;

enum DebugType: uint {
    Combined,
    Iterations,
    BufferFetches,
    Normals,
    World,
    ExitType,
}

// GPTed. could not find this fuckass function online. Im so fucking stupid
float3 heatmap(float t) {
    return clamp(float3(
        1.5 - abs(4.0 * t - 3.0),
        1.5 - abs(4.0 * t - 2.0),
        1.5 - abs(4.0 * t - 1.0)
    ), 0.0, 1.0);
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

    
    
    
    Statistics stats = Statistics(0, 0);
    //bool hit = any_hit_trace_fine(ray_pos, ray_dir, svo);
    //uint clock = getRealtimeClock().x;
    var ray = trace_shi(ray_pos, ray_dir, svo, stats);

    //
    //
    //
    //var b2 = trace_shi(b.world + sun.xyz * 1, sun.xyz, svo, stats);
    /*
    //bool b = any_hit_trace_fine(ray_pos, ray_dir, svo, stats);
    uint clock2 = getRealtimeClock().x;
    float elapsed = (clock2 - clock) / 4096.0;

    // if this is a high value, it means that, in the wave, a bunch of threads exited / are doing jack shit whilst other are doing work
    // if this is a lower value, it means that most threads are doing the same amount of work
    float some_number = (WaveActiveMax(stats.iterations) - stats.iterations) / 256.0;
    */

    //output[id.xy] = float4(heatmap(stats.buffer_fetches / 32.0), 0);
    //output[id.xy] = float4(heatmap(stats.iterations / 256.0), 0);
    //output[id.xy] = float4((ray.world % 32.0) / 32.0, 0);
    //return;
    //output[id.xy] = float4(ray.face / 6.0);
    
    //ray.world = get_accurate_world_position(ray.world - ray_dir * 0.05, ray_dir);

    //return;
    //output[id.xy] = float4(b.face / 6.0);
    //output[id.xy] = float4((b.world % 4) / 4.0, 0);
    //output[id.xy] = float4(iter / 64.0);
    //output[id.xy] = float4(b2.hit ? 1 : 0);

    float3 color = 0.0;
    if (ray.hit) {
        // improve the accuracy of the world position by utilizing the face (assumption we can make since we are dealing with pure cubic voxels)
        float3 world = get_accurate_world_position_with_face(ray.world, ray.block_pos, ray.face, sign(ray_dir));
        float3 normal = normalize(get_face_normal(ray.face, sign(ray_dir)));
        normal = normalize(world - ray.block_pos - 0.5);

        float3 diffuse = 1.0;
        float ao = 1.0;

        float3 d = reflect(ray_dir, normal);
        float shadow = 1.0;
        float sky_visibility = 1.0;

        shadow = 0.0;
        for (int k = 0; k < 4; k++) {
            float3 rng = (hash33(floor(world) * 23.12123545 + k * 30.43324) * 2.0 - 1.0);
            float3 dir = normalize(sun.xyz + rng * 0.05);
            float d = 0;
            bool t = trace_shi(floor(world) - ray_dir * 0.001 + sun.xyz * 0.004 + dir * 0.05, dir, svo, stats).hit;
            shadow += (t ? 0.0 : 1) / 4.0;
        }

        /*
        sky_visibility = 0.0;
        for (int k = 0; k < 1; k++) {
            float3 rng = (hash33(world * 23.12123545 + k * 30.43324) * 2.0 - 1.0);
            float3 dir = normalize(d + rng * 0.05 * 0.0);
            float d = 0;
            bool t = trace_shi(world - ray_dir * 0.05 + dir * 0.05, dir, svo, stats).hit;
            sky_visibility += (t ? 0.0 : 1) / 1.0;
        }
        */

        PbrSurfaceData pbr_surface_data = PbrSurfaceData(diffuse, normal, 0.85, 0.0, ao, shadow, sky_visibility);
        CameraData camera_data = CameraData(-ray_dir);
        SunData sun_data = SunData(sun.xyz, sun.y > 0 ? 0.8 : 0);

        
        //color = normal;
        color = brdf(pbr_surface_data, camera_data, sun_data);
        //color = block_pos;
    } else {
        color = sky(sun.xyz, ray_dir);
    }

    //output[id.xy] = float4(((float3)tst.block_pos % 10) / 10.0, 0);
    switch (debug_type) {
        case DebugType::Combined:
            output[id.xy] = float4(aces(color), 0);
            break;
        case DebugType::Iterations:
            output[id.xy] = float4(heatmap(stats.iterations / 256.0), 0);
            break;
        case DebugType::BufferFetches:
            output[id.xy] = float4(heatmap(stats.buffer_fetches / 256.0), 0);
            break;
        case DebugType::ExitType:
            output[id.xy] = float4(stats.exit_type / 3.0);
            break;
        case DebugType::World:
            //output[id.xy] = float4((ray.block_pos % 10) / 10.0, 0);
            //output[id.xy] = float4(sin(ray.world) * 0.5 + 0.5, 0);
            output[id.xy] = float4((ray.world % 10) / 10.0, 0);
            break;
        case DebugType::Normals:
            output[id.xy] = float4((get_accurate_world_position_with_face(ray.world, ray.block_pos, ray.face, sign(ray_dir)) % 10) / 10.0, 0);
            //output[id.xy] = float4(get_face_normal(ray.face, sign(ray_dir)) * 0.5 + 0.5, 0);
            break;
        default:
            break;
    }
    //output[id.xy] = float4(heatmap(stats.iterations / 256.0), 0);
}