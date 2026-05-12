#language slang 2026
#include <ray_stuff_other.slang>
#include <lighting.slang>
#include <pbr.slang>
#include <noises.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(0, 1)]]
RWTexture3D<uint8_t4> svt;

[[vk::binding(1, 1)]]
RWTexture3D<uint8_t> svt_metadata;

[[vk::binding(2, 1)]]
StructuredBuffer<uint64_t> svo_bitmasks_buffer;

[[vk::binding(3, 1)]]
StructuredBuffer<uint32_t> svo_indices_buffer;

[[vk::binding(4, 1)]]
StructuredBuffer<uint64_t> svo_aabbs_buffer;

[[vk::binding(5, 1)]]
StructuredBuffer<float4> lights_buffer;

[[vk::binding(6, 1)]]
SamplerCube<float4> skybox_sampler;

[[vk::binding(7, 1)]]
Sampler2D<float4> clouds_sampler;

[[vk::binding(8, 1)]]
Sampler3D<float4> svt_sampler;


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

float4 dda<let N : int>(
    float3 ray_dir,
    float3 ray_pos,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);
    float4 summed = 0;

    for (int i = 0; i < N; i++) {
        float4 voxel = (float4)svt.Load((uint3)(floored_pos)) / 255.0;
        
        summed += voxel.w / N;
        
        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    return summed;
}


float4 raymarch<let N : int>(
    float3 ray_dir,
    float3 ray_pos,
) {
    float4 summed = 0;

    for (int i = 0; i < N; i++) {
        float4 voxel = (float4)svt_sampler.Sample(ray_pos / (float)TOTAL_SIZE);
        
        summed += voxel.w / N;

        ray_pos += ray_dir;
    }

    return summed;
}

#define DEBUG

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform DebugType debug_type, uniform float time) {
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;

    let ndc = float4(screen_uvs, -1, 1);
    float3 ray_dir = normalize(mul(mat, ndc).xyz);
    float3 ray_pos = position.xyz;
    
    SparseVoxelOctree svo = SparseVoxelOctree(svo_indices_buffer, svo_bitmasks_buffer, svo_aabbs_buffer);
    Statistics stats = Statistics(0, 0);
    
    float3 iterative_ray_dir = ray_dir;
    float3 iterative_ray_pos = ray_pos;

#ifdef DEBUG
    float3 nth_normals = 0.0;
    float3 nth_world = 0.0;
#endif
    
    float3 color = 1.0;
    float alpha = 0.0;

    //let start = clock2x32ARB();

    //var ray = trace_shi<int.maxValue>(iterative_ray_pos, iterative_ray_dir, svo, stats);
    
    //let end = clock2x32ARB();
    //output[id.xy] = float4(ray.world - ray.block_pos, 0);
    //return;
    //output[id.xy] = float4((end.x - start.x) / (4096*16) + length(ray.world - ray.block_pos) * 0.001);
    
    bool inwater = false;
    
    [loop]
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        var ray = trace_shi<int.maxValue>(iterative_ray_pos, iterative_ray_dir, svo, stats);

        /*
        let water = ray_plane_intersect_generic(iterative_ray_dir, iterative_ray_pos, float3(0, 1, 0), float3(0, 80, 0));
        if (water.hasValue && !inwater) {
            if (water.value < ray.distance || !ray.hit) {
                float3 testst = iterative_ray_pos+iterative_ray_dir*water.value;
                float3 water_normal = normalize(float3(snoise(testst.xz * 0.08)*.2, 8, snoise(testst.xz * 0.08 + 32.31212)*.2));
                iterative_ray_dir = reflect(iterative_ray_dir, water_normal);
                iterative_ray_pos = testst + float3(0, 0.1, 0);
                color *= 0.30;
                color += pow(saturate(dot(reflect(sun.xyz, water_normal), iterative_ray_dir)), 13);
                inwater = true;
                alpha += 0.3;
                continue;
            }
        }
        */

        float3 temp_color = 0.0;
        if (ray.hit) {
            float3 normal = get_face_normal(ray.face, sign(iterative_ray_dir));

            // we can solve for the missing UV coordinates without needing the distance!!!
            float3 k = 0;
            k[ray.face] = ray.specific_block_axis_thing;
            float32_t distance = ray_plane_intersect_generic(iterative_ray_dir, iterative_ray_pos, normal, k+max(normal, 0)).value;
            float3 world = iterative_ray_pos + iterative_ray_dir * distance;
            world = get_accurate_world_position_with_face2(world, ray.specific_block_axis_thing, ray.face, sign(iterative_ray_dir));
            uint3 block_pos = floor(world);

            if (inwater) {
                alpha = distance / 200;
            }

#ifdef DEBUG
            // fetch the nth normal and world position
            if (i == 0) {
                nth_normals = normal;
                nth_world = (block_pos % 16) / 16.0;
                //nth_world = world;
            }
#endif

            if (ROUND_NORMALS) {
                //normal = normalize(world - ray.block_pos - 0.5);
            }


            //normal = normalize(normal + (hash33(floor(world * 16)) - 0.5) * 0.3);
            
            float3 diffuse = 1.0;
            //float3 diffuse = (float3)svt.Load(block_pos).xyz / 255.0;

            float roughness = 0.80;
            float special = 0;
            float metallic = 0;

            // pick random colour
            if (block_pos.x > TOTAL_SIZE / 2) {
                diffuse = lerp(per_block_unique_colour(block_pos), 1, 0.5);
            }
                
                
            /*
            // pick random roughness + metallic
            if (block_pos.z > TOTAL_SIZE / 2) {
                roughness = hash13(block_pos * float3(3.23123, 5.23423, -5.234234));
                metallic = hash13(block_pos * float3(132.12312, -2321.32132, 3.2132));
            }

            // pick random special mirror block
            if (block_pos.y > 4) {
                special = hash13(floor(block_pos / 8));
            }
            */

            if (special > 0.8 && metallic > 0.0) {
                
                /*
                iterative_ray_pos = world + iterative_ray_dir * 0.001;
                
                const int ITERNUM = 32;

                float dt = 0.045;
                float3 center = block_pos + 0.5 + float3(sin(time), cos(time), sin(time * 0.4 + 4.1223)) * 0.2; 
                float3 ray_velocity = iterative_ray_dir; 

                [loop]
                raymarch: for (int k = 0; k < ITERNUM; k++) {
                    stats.iterations += 1;
                    
                    float3 force_torwards_center = -(iterative_ray_pos - center);
                    float3 normalized_force_torwards_center = force_torwards_center / (pow(length(force_torwards_center) + 0.8, 8));
                    normalized_force_torwards_center *= (sin(time * 0.8) * 2.0);

                    iterative_ray_dir += normalized_force_torwards_center * dt * 12.3;
                    iterative_ray_pos += normalize(iterative_ray_dir) * dt;
                    //color *= lerp(snoise(iterative_ray_pos * 4.8) * 0.5 + 0.5, hash33(floor(iterative_ray_pos * 32) * 4.2132) + 0.5, 0.98);

                    if (any(floor(iterative_ray_pos) != block_pos)) {
                        break raymarch;
                    }
                }

                iterative_ray_dir = normalize(iterative_ray_dir);
                */
                // mirror
                // TODO: I wanted to get refractions working, but that requires more coordination between the ray-tracer and the lighting system
                // the ray-tracer needs to keep track of the current "medium" in which the ray is traveling in, and discard that medium when a block of that medium comes up
                // for example, if the ray entered some glass voxels, the raytracer must discard glass blocks in the next iteration as to avoid colliding with them.
                // this will need us to implement an actual block type system so that will probably be for later. maybe we will need to use a 3D texture + the AS for fast lookups but lower memory usage. idk 
                float3 new_normal = normal;

                if (WAVY_REFLECTIONS) {
                    new_normal = normalize(snoise(world * 0.85 + time) * 0.05 + normal);
                }

                float3 dir = reflect(iterative_ray_dir, new_normal);
                //float3 dir = reflect(iterative_ray_dir, normalize(new_normal + (hash33(world * float3(3.2312, -80.4098361, 94.309483)) * 2.0 - 1.0) * 0.03 * roughness));
                iterative_ray_dir = dir;
                iterative_ray_pos = world + dir * 0.001;
                color *= diffuse * metallic;
            } else {
                // AMBIENT OCCLUSION
                float ao = 1.0;
                if (AMBIENT_OCCLUSION) {
                    float3 wtf = select(normal < 0, -2, normal);
                    ao = 1 - saturate(svt_sampler.Sample((world + wtf) / (float)TOTAL_SIZE).w * 0.5);

                    
                    //ao -= Fetcher(svt, svt_metadata).is_solid((uint3)floor(block_pos + wtf)) * 0.5;

                    //ao_solver solver = ao_solver(Fetcher(svt, svt_metadata), block_pos, world-block_pos, ray.face, sign(iterative_ray_dir));
                    //ao = lerp(solver.ao(), 1, 0.0);
                    /*
                    [loop]
                    for (int j = 0; j < 2; j++) {
                        //ao += cube_overlap_density_svo(world + normal * (j+0.5), 0.8 + j * 0.8, svo, stats);
                        //ao += cube_overlap_density_svt(world + normal * (j+0.5), 0.8 + j * 0.8, svt, svt_metadata, stats);
                        ao += planar_overlap_density_svt(world + normal * (j*2+0.5), 0.8 + 2 * j * 0.8, ray.face, Fetcher(svt, svt_metadata), stats);
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
                        float3 dir = normalize(sun.xyz + rng * 0.005);
                        float d = 0;
                        float strength = pow(dot(dir, sun.xyz), 40);
                        bool t = trace_shi<4096>((shadow_ray_start) + sun.xyz * 0.004 + dir * 0.05, dir, svo, stats).hit;
                        shadow += (t ? 0.0 : (strength * (1-clouds(dir, shadow_ray_start, clouds_sampler).a))) / SHADOW_SAMPLES;
                    }
                } else if (SHADOW_SAMPLES == 1) {
                    bool t = trace_shi<4096>((shadow_ray_start) + sun.xyz * 0.05, sun.xyz, svo, stats).hit;
                    //  - raymarch<256>(sun.xyz, shadow_ray_start).x * 20.0
                    shadow = (t ? 0.0 : 1 * (1-clouds(sun.xyz, shadow_ray_start, clouds_sampler).a));
                    //shadow = saturate(shadow);

                } else {
                    shadow = 1.0;
                }

                PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, pow(ao, 1.4), shadow);
                CameraData camera_data = CameraData(-iterative_ray_dir);

                // approximate sun color based on height
                float3 sun_colour = sun_colour(sun.xyz);
                SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);
                SkyData sky_data = SkyData(skybox_sampler, clouds_sampler);

                temp_color = lighting(pbr_surface_data, camera_data, sun_data, sky_data);

                /*
                for (int l = 0; l < 10; l++) {
                    float3 light_pos = lights_buffer[l].xyz;
                    float3 light_towards_position = shadow_ray_start - light_pos;
                    float d = length(light_towards_position);
                    if (d < 10) {
                        
                        float unshadowed_dist = length(light_towards_position);
                        light_towards_position = normalize(light_towards_position);
                        float shadowed_dist = trace_shi<4096>(light_pos, light_towards_position, svo, stats).distance;
                        float shadow = (((shadowed_dist + 0.1) > unshadowed_dist) ? 1.0 : 0);
                        float n_dot_l = saturate(dot(normal, -normalize(light_towards_position)));
                        float strength = (1 / (d*d)) * smoothstep(0, 1, 10-d);
                        temp_color += shadow * n_dot_l * strength * hash33(l * 3.21312) * 0.3; 
                        
                    }
                }
                */

                color *= temp_color;
                output[id.xy] = float4(color, alpha);

                break;
            }
        } else {
            alpha = 10;

            // ray miss stuff
            color *= sample_skybox_and_clouds(iterative_ray_pos, iterative_ray_dir, clouds_sampler, skybox_sampler);
            break;
        }

        // if we reach this point, this means that all previous rays have been reflected. return 0
        if (i == (MAX_ITERATIONS)-1) {
            color *= 0.0;
        }
    }

#ifdef DEBUG
    switch (debug_type) {
        case DebugType::Combined:
            //output[id.xy] = float4(color, alpha);
            break;
        case DebugType::Iterations:
            output[id.xy] = raymarch<64>(ray_dir, ray_pos);    
            //output[id.xy] = dda<64>(ray_dir, ray_pos);
            //return;
            //output[id.xy] = float4(heatmap(stats.iterations / 256.0), 0);
            break;
        case DebugType::BufferFetches:
            output[id.xy] = float4(heatmap(stats.buffer_fetches / 32.0), 0);
            break;
        case DebugType::Normals:
            output[id.xy] = float4(saturate(nth_normals), 0);
            break;
        case DebugType::World:
            output[id.xy] = float4((nth_world % 16) / 16, 0);
            break;
        case DebugType::ExitType:
            output[id.xy] = float4(heatmap(stats.num_entered_but_missed / 64.0), 0);
            //output[id.xy] = float4(heatmap(1 - (stats.active_threads_per_call / stats.calls) / 64.0), 0);
            //output[id.xy] = float4(stats.exit_type / 4.0);
            break;
        default:
            break;
    }
#else
    output[id.xy] = float4(color, alpha);
#endif

}