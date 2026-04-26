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


[SpecializationConstant] const uint SHADOW_SAMPLES = 0;
//[SpecializationConstant] const uint MAX_ITERATIONS = 1;
[SpecializationConstant] const bool ROUND_NORMALS = false;
[SpecializationConstant] const bool AMBIENT_OCCLUSION = false;

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
    
    float3 iterative_ray_dir = ray_dir;
    float3 iterative_ray_pos = ray_pos;
    float3 first_normals = 0.0;
    float3 first_world = 0.0;
    
    var ray = trace_shi<4096>(iterative_ray_pos, iterative_ray_dir, svo, stats);
    
    // improve the accuracy of the world position by utilizing the face (assumption we can make since we are dealing with pure cubic voxels)
    ray.world = get_accurate_world_position_with_face(ray.world, ray.block_pos, ray.face, sign(iterative_ray_dir));
    
    float3 color = 1.0;
    if (ray.hit) {
        float3 world = ray.world;
        uint3 block_pos = ray.block_pos;
        float3 normal = get_face_normal(ray.face, sign(iterative_ray_dir));

        // first ray has some debug stuff going on
        first_normals = normal;
        first_world = world;
        
        if (ROUND_NORMALS) {
            normal = normalize(world - ray.block_pos - 0.5);
        }

        float3 diffuse = 1.0;
        //float3 diffuse = per_block_unique_colour(block_pos);
        float roughness = hash13(block_pos * float3(3.23123, 5.23423, -5.234234));
        float metallic = hash13(block_pos * float3(132.12312, -2321.32132, 3.2132));
        
        // AMBIENT OCCLUSION
        float ao = 1.0;
        if (AMBIENT_OCCLUSION) {
            for (int k = 0; k < 4; k++) {
                float3 rng = (hash33(world * 23.12123545 + k * 30.43324) * 2.0 - 1.0);
                
                // don't affect the ray in the major axis direction
                rng[ray.face] = 0;
                
                float3 dir = normalize(normal + rng);
                float3 start = world - iterative_ray_dir * 0.05 + dir * 0.05;
                var ao_ray = trace_shi<4>(start, dir, svo, stats);

                if (ao_ray.hit) {
                    // AO strength decreases as the hit object is further
                    // closer objects have more influence
                    float strength = clamp(8.0 / distance(ao_ray.world, start), 0, 1);
                    ao -= strength / 4.0;
                }
            }
            ao = clamp(ao, 0, 1);
        }

        //float3 rng = (hash33((world) * 23.12123545 + i * 30.43324) * 2.0 - 1.0);
        //float3 dir = reflect(iterative_ray_dir, normalize(normal + rng * 0.05));
        float3 dir = reflect(iterative_ray_dir, normal);
        float shadow = 1.0;
        float sky_visibility = 1.0;

        // SHADOW STUFF!!!!
        if (SHADOW_SAMPLES > 1) {
            shadow = 0.0;
            for (int k = 0; k < SHADOW_SAMPLES; k++) {
                float3 rng = (hash33((world) * 23.12123545 + k * 30.43324) * 2.0 - 1.0);
                float3 dir = normalize(sun.xyz + rng * 0.05);
                float d = 0;
                bool t = trace_shi<4096>((world) - ray_dir * 0.001 + sun.xyz * 0.004 + dir * 0.05, dir, svo, stats).hit;
                shadow += (t ? 0.0 : 1) / SHADOW_SAMPLES;
            }
        } else {
            bool t = trace_shi<4096>((world) - iterative_ray_dir * 0.001 + sun.xyz * 0.004, sun.xyz, svo, stats).hit;
            shadow = (t ? 0.0 : 1);
        }

        // sky visibility is an extra term that is used for ambient lighting
        /*
        sky_visibility = 0.0;
        for (int k = 0; k < 4; k++) {
            float3 rng = (hash33(world * 23.12123545 + k * 30.43324) * 2.0 - 1.0);
            float3 dir = normalize(dir + rng * 0.25 * pow(roughness, 2));
            float d = 0;
            bool t = trace_shi<64>(world - iterative_ray_dir * 0.05 + dir * 0.05, dir, svo, stats).hit;
            sky_visibility += (t ? 0.0 : 1) / 4.0;
        }
        sky_visibility = lerp(sky_visibility, 1, roughness);  
        */

        PbrSurfaceData pbr_surface_data = PbrSurfaceData(diffuse, normal, roughness, metallic, ao, shadow, sky_visibility);
        CameraData camera_data = CameraData(-iterative_ray_dir);

        // approximate sun color based on height
        float3 sun_colour = sun_colour(sun.xyz);
        SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (1.2 * sun_colour) : 0);

        color = brdf(pbr_surface_data, camera_data, sun_data);
    } else {
        color = sky(sun.xyz, iterative_ray_dir);
    }

    /*
    Combined,
    Iterations,
    BufferFetches,
    Normals,
    World,
    ExitType,
    */

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
        case DebugType::Normals:
            output[id.xy] = float4(first_normals, 0);
            break;
        case DebugType::World:
            output[id.xy] = float4((first_world % 10) / 10.0, 0);
            break;
        case DebugType::ExitType:
            output[id.xy] = float4(stats.exit_type / 3.0);
            break;
        default:
            break;
    }
}