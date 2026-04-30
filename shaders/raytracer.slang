#language slang 2026
#include <ray_stuff_other.slang>
#include <lighting.slang>
#include <pbr.slang>
#include <noises.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(1, 0)]]
Texture3D<uint8_t> svt;

[[vk::binding(2, 0)]]
Texture3D<uint8_t> svt_metadata;

[[vk::binding(3, 0)]]
StructuredBuffer<uint64_t> svo_bitmasks_buffer;

[[vk::binding(4, 0)]]
StructuredBuffer<uint32_t> svo_indices_buffer;

[SpecializationConstant] const uint SHADOW_SAMPLES = 1;
[SpecializationConstant] const uint MAX_ITERATIONS = 2;
[SpecializationConstant] const bool ROUND_NORMALS = false;
[SpecializationConstant] const bool AMBIENT_OCCLUSION = false;
[SpecializationConstant] const bool WAVY_REFLECTIONS = false;
[SpecializationConstant] const bool PIXELATED_SHADOWS = false;
[SpecializationConstant] const uint GROUP_SIZE = 8;

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
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform DebugType debug_type, uniform float time) {
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

    float3 nth_normals = 0.0;
    float3 nth_world = 0.0;
    
    float3 color = 1.0;
        
    [loop]
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        var ray = trace_shi<int.maxValue>(iterative_ray_pos, iterative_ray_dir, svo, stats);
        
        // improve the accuracy of the world position by utilizing the face (assumption we can make since we are dealing with pure cubic voxels)
        ray.world = get_accurate_world_position_with_face(ray.world, ray.block_pos, ray.face, sign(iterative_ray_dir));
        
        float3 temp_color = 0.0;
        if (ray.hit) {
            float3 world = ray.world;
            uint3 block_pos = ray.block_pos;
            float3 normal = get_face_normal(ray.face, sign(iterative_ray_dir));

            // fetch the nth normal and world position
            if ((((uint)round(time)) % MAX_ITERATIONS) == i) {
                nth_normals = normal;
                nth_world = world;
            }

            if (ROUND_NORMALS) {
                normal = normalize(world - ray.block_pos - 0.5);
            }


            float3 diffuse = 1.0;

            float roughness = 0.50;
            float special = 0;
            float metallic = 0;
            
            // pick random colour
            if (block_pos.x > TOTAL_SIZE / 2) {
                diffuse = lerp(per_block_unique_colour(block_pos), 1, 0.5);
            }

            // pick random roughness + metallic
            if (block_pos.z > TOTAL_SIZE / 2) {
                roughness = hash13(block_pos * float3(3.23123, 5.23423, -5.234234));
                metallic = hash13(block_pos * float3(132.12312, -2321.32132, 3.2132));
            }

            // pick random special mirror block
            if (block_pos.y > TOTAL_SIZE / 2) {
                special = hash13(floor(block_pos / 8));
            }

            if (special > 0.8 && metallic > 0.0) {
                iterative_ray_pos = world + iterative_ray_dir * 0.001;
                
                const int ITERNUM = 256;

                float dt = 0.015;
                float3 center = block_pos + 0.5; 
                float3 ray_velocity = iterative_ray_dir; 

                [loop]
                al: for (int k = 0; k < ITERNUM; k++) {
                    stats.iterations += 1;
                    
                    float3 force_torwards_center = -(iterative_ray_pos - center);
                    float3 normalized_force_torwards_center = force_torwards_center / pow(length(force_torwards_center), 2);
                    normalized_force_torwards_center *= (sin(time * 0.8) * 1.0);

                    iterative_ray_dir += normalized_force_torwards_center * dt * 0.3;
                    iterative_ray_pos += normalize(iterative_ray_dir) * dt;

                    if (any(floor(iterative_ray_pos) != block_pos)) {
                        break al;
                    }
                }

                iterative_ray_dir = normalize(iterative_ray_dir);
                
                /*
                // pure mirror
                // TODO: I wanted to get refractions working, but that requires more coordination between the ray-tracer and the lighting system
                // the ray-tracer needs to keep track of the current "medium" in which the ray is traveling in, and discard that medium when a block of that medium comes up
                // for example, if the ray entered some glass voxels, the raytracer must discard glass blocks in the next iteration as to avoid colliding with them.
                // this will need us to implement an actual block type system so that will probably be for later. maybe we will need to use a 3D texture + the AS for fast lookups but lower memory usage. idk 
                float3 new_normal = normal;

                if (WAVY_REFLECTIONS) {
                    new_normal = normalize(snoise(world * 0.85 + time) * 0.05 + normal);
                }

                float3 dir = reflect(iterative_ray_dir, new_normal + (hash33(world * float3(3.2312, -80.4098361, 94.309483)) * 2.0 - 1.0) * 0.03 * roughness);
                iterative_ray_dir = dir;
                iterative_ray_pos = world + dir * 0.001;
                color *= diffuse * metallic;
                */
            } else {
                // AMBIENT OCCLUSION
                float ao = 1.0;
                if (AMBIENT_OCCLUSION) {
                    ao = 0;

                    ao_solver solver = ao_solver(Fetcher(svt, svt_metadata), block_pos, world-block_pos, ray.face, sign(iterative_ray_dir));
                    ao = lerp(solver.ao(), 1, 0.8);

                    /*
                    [loop]
                    for (int j = 0; j < 5; j++) {
                        //ao += cube_overlap_density_svo(world + normal * (j+0.5), 0.8 + j * 0.8, svo, stats);
                        //ao += cube_overlap_density_svt(world + normal * (j+0.5), 0.8 + j * 0.8, svt, svt_metadata, stats);
                        ao += planar_overlap_density_svt(world + normal * (j*0.5+0.5), 0.8 + j * 0.8, ray.face, Fetcher(svt, svt_metadata), stats);
                        if (ao > 2) {
                            break;
                        }
                    }
                
                    ao = clamp(ao, 0, 2);
                    ao *= 0.2;
                    ao = clamp(1-ao, 0, 1);
                    */
            
                    /*
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
                    */
                }

                //output[id.xy] = ao;
                //return;

                float shadow = 1.0;
                float sky_visibility = 1.0;

                // SHADOW STUFF!!!!
                float3 shadow_ray_start = world;
                if (PIXELATED_SHADOWS) {
                    shadow_ray_start = floor(world * 4) / 4.0;
                }
                if (SHADOW_SAMPLES > 1) {
                    shadow = 0.0;
                    [loop]
                    for (int k = 0; k < SHADOW_SAMPLES; k++) {
                        float3 rng = (hash33((shadow_ray_start) * 23.12123545 + k * 30.43324) * 2.0 - 1.0);
                        float3 dir = normalize(sun.xyz + rng * 0.025);
                        float d = 0;
                        float strength = pow(dot(dir, sun.xyz), 40);
                        bool t = trace_shi<4096>((shadow_ray_start) + sun.xyz * 0.004 + dir * 0.05, dir, svo, stats).hit;
                        shadow += (t ? 0.0 : strength) / SHADOW_SAMPLES;
                    }
                } else if (SHADOW_SAMPLES == 1) {
                    bool t = trace_shi<4096>((shadow_ray_start) + sun.xyz * 0.004, sun.xyz, svo, stats).hit;
                    shadow = (t ? 0.0 : 1);
                } else {
                    shadow = 1.0;
                }

                /*
                // sky visibility is an extra term that is used for ambient lighting
                if (roughness > 0.3) {
                    float3 reflected = reflect(iterative_ray_dir, normal);
                    int sample_count = floor(roughness * 4);

                    if (sample_count >= 1) {
                        sky_visibility = 0.0;
                        for (int k = 0; k < sample_count; k++) {
                            float3 rng = (hash33(world * 23.12123545 + k * 30.43324) * 2.0 - 1.0);
                            float3 dir = normalize(reflected + rng * 0.25 * pow(roughness, 2));
                            float d = 0;
                            bool t = trace_shi<8>(world - iterative_ray_dir * 0.05 + dir * 0.05, dir, svo, stats).hit;
                            sky_visibility += (t ? 0.0 : 1) / sample_count;
                        }
                        sky_visibility = lerp(sky_visibility, 1, roughness);  
                    }
                }
                */

                PbrSurfaceData pbr_surface_data = PbrSurfaceData(diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow, sky_visibility);
                CameraData camera_data = CameraData(-iterative_ray_dir);

                // approximate sun color based on height
                float3 sun_colour = sun_colour(sun.xyz);
                SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (1.2 * sun_colour) : 0);

                temp_color = brdf(pbr_surface_data, camera_data, sun_data);
                color *= temp_color;

                /*
                {
                    float r = 0;
                    float3 d = reflect(ray_dir, normal);
                    [loop]
                    for (int j = 0; j < 64; j++) {
                        r += cube_trace_shi(world + d * (j+0.5) * 0.2, 0.1 + pow(j * 0.2, 1.3), svo, stats);
                        if (r > 0.8) {
                            break;
                        }
                    }
                
                    color *= clamp(1-r*0.05, 0, 1);
                }
                */

                break;
            }
        } else {
            color *= sky(sun.xyz, iterative_ray_dir);
            break;
        }

        // if we reach this point, this means that all previous rays have been reflected. return 0
        if (i == (MAX_ITERATIONS)-1) {
            color *= 0.0;
        }
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
            output[id.xy] = float4(nth_normals, 0);
            break;
        case DebugType::World:
            output[id.xy] = float4((nth_world % 10) / 10.0, 0);
            break;
        case DebugType::ExitType:
            output[id.xy] = float4(heatmap(1 - (stats.active_threads_per_call / stats.calls) / 64.0), 0);
            //output[id.xy] = float4(stats.exit_type / 3.0);
            break;
        default:
            break;
    }
}