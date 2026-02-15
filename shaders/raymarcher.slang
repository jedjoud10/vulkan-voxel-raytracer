#language slang 2026
#include <lighting.slang>
#include <pbr.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(1, 0)]]
Texture3D<uint64_t> voxels;

[[vk::binding(2, 0)]]
StructuredBuffer<SurfaceData> surface_data_buffer;

[[vk::binding(3, 0)]]
Texture3D<uint> voxels_indices;

[[vk::binding(4, 0)]]
StructuredBuffer<uint64_t> svo_bitmasks_buffer;

[[vk::binding(5, 0)]]
StructuredBuffer<uint16_t> svo_indices_buffer;

[SpecializationConstant] const bool USE_PIXELATED_SHADOWS = false;
[SpecializationConstant] const bool USE_ROUND_NORMALS = false;
[SpecializationConstant] const bool USE_NOISY_REFRACTIONS_REFLECTIONS = false;

enum DebugType: uint {
    Combined,
    Iterations,
    Recursions,
    FullHalt,
    World,
    Timer,
}

groupshared uint min_clock;
groupshared uint max_clock;


bool dda_multi_level<let K : int>(
    uint16_t base_index,
    uint64_t raw,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    float distance,
) {
    int voxel_size = 1 << max(K, 0);
    float3 modified_ray_pos = (ray_pos + ray_dir * (distance - 0.01)) / voxel_size;

    /*
    uint3 sub_cell_start_pos = (uint3)floor((ray_pos + ray_dir * ((float)popped.distance + 1.5)) / voxel_size);
    uint3 sub_cell_end_pos = sub_cell_start_pos + 1;
    */

    float3 floored_pos = floor(modified_ray_pos);
    float3 side_dist = (floored_pos - modified_ray_pos + 0.5 + 0.5 * dir_sign);

    int face_local = 0;
    for (int i = 0; i < DDA_STEPS; i++) {
        // skip if we are outside the bounds of the map
        if (any((int3)(floored_pos) < 0) || any((int3)(floored_pos) >= (SIZE >> max(K, 0)))) {
            break;
        }

        uint8_t3 local = (uint16_t3)floored_pos % (uint16_t3)4;
        uint8_t bit_index = local.z + local.y * 4 + local.x * 4 * 4;
        
        /*
        // handle partially full nodes / leaf voxel nodes
        if (raw != 0) {
            float3 test = (floored_pos * voxel_size - ray_pos + voxel_size * 0.5 - 0.5 * voxel_size * dir_sign) * inv_dir;
            float dist = max3(test.x, test.y, test.z);

            // completely full nodes are well, completely full. we can do the leaf-level logic since we know we can skip them
            if ((raw & 4) == 4) {
                stats.full_halts++;
            
                // set output parameters
                float3 w = ray_pos + ray_dir * dist;
                output.world = w;
                output.face = face_local;
                output.block_pos = (uint3)floor(w + ray_dir * 0.02);
                
                return true;
            }


            if (K > 0) {
                stats.recursions++;

                int3 next_min_bounds_world_space = (int3)floored_pos * voxel_size;
                int3 next_max_bounds_world_space = (int3)floored_pos * voxel_size + voxel_size;


                if (dda_multi_level<K-LEVEL_DIFF, hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, next_min_bounds_world_space, next_max_bounds_world_space, inv_dir, dir_sign, max(dist, 0), stats, output)) {
                    return true;
                }
            } else {
                // set output parameters
                output.world = ray_pos + ray_dir * dist;
                output.face = face_local;
                output.block_pos = (uint3)floored_pos;
                
                return true;
            }
        }



        stats.total_iterations++;
        */

        float3 reconst = side_dist * inv_dir;
        bool3 t = min3(reconst.x, reconst.y, reconst.z) == reconst;
        int3 eqs = select(t, 1, 0);
        face_local = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        floored_pos += select(t, dir_sign, 0);
        side_dist += select(t, dir_sign, 0);
    }
    

    return false;
}

#define MAX(a, b)  (((a) > (b)) ? (a) : (b)) 

bool test2<let K : int>(
    uint16_t base_index,
    uint64_t raw,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    float distance,
    int3 bounds_min,
    int3 bounds_max,
    out float3 col,
) {
    if (raw == 0) {
        return false;
    }

    int voxel_size = 1 << max(K*2, 0);
    int3 min_bounds_local_space = bounds_min / voxel_size;
    int3 max_bounds_local_space = bounds_max / voxel_size;

    float3 modified_ray_pos = (ray_pos + ray_dir * (distance + 0.01)) / voxel_size;
    uint16_t3 floored_pos = (int3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = 0;

    col.x += 0.1;

    [loop]
    for (uint8_t i = 0; i < 12; i++) {
        col.y += 0.01;
        if (any(floored_pos < 0 | floored_pos >= (SIZE / voxel_size))) {
            return false;
        }

        bool oob = any((int3)(floored_pos) < (min_bounds_local_space) | (int3)(floored_pos) > (max_bounds_local_space - 1));
        bool oob_loose = any((int3)(floored_pos) < (min_bounds_local_space-1) | (int3)(floored_pos) > (max_bounds_local_space));
        
        
        if (oob_loose) {
            return false;
        }
        
        uint8_t3 local = (uint16_t3)floored_pos % (uint16_t3)4;
        uint8_t bit_index = local.z + local.y * 4 + local.x * 4 * 4;

        if (bit_index < 64 && ((raw >> bit_index) & 1) == 1 && !oob) {            
            if (K > 0) {
                uint16_t new_base_index = svo_indices_buffer[base_index + bit_index];
                
                if (new_base_index != 0xFFFF) {
                    uint64_t new_raw = svo_bitmasks_buffer[base_index + bit_index];

                    int3 next_min_bounds_world_space = (int3)floored_pos * voxel_size;
                    int3 next_max_bounds_world_space = (int3)floored_pos * voxel_size + voxel_size;

                    if (new_raw == -1 && new_base_index == 0xFFFF) {
                        //col = 1.0;
                        return true;
                    }

                    float distance_offset = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * voxel_size) + 0.01;

                    if (test2<K - 1>(new_base_index, raw, ray_dir, ray_pos, inv_dir, dir_sign, distance + distance_offset, next_min_bounds_world_space, next_max_bounds_world_space, col)) {
                        return true;
                    }
                } else {
                    return true;
                }
            } else {
                //col = 1.0;
                //col = (float)i / 16.0;
                return true;
            }
        }


        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));
    }

    return false;
}

float3 test(float3 ray_dir, float3 ray_pos) {
    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    float3 col = 0.0;
    bool hit = test2<4>(0, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0.0, 0, 9999999, col);
    return col;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform DebugType debug_type) {

    if (all(lid.xy == 0)) {
        max_clock = 0;
        min_clock = 0xFFFFFFFF;
    }
    GroupMemoryBarrierWithGroupSync();
    
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;
    
    float3 ray_dir = normalize((mul(mat, float4(screen_uvs, 1, 0))).xyz);
    float3 ray_pos = position.xyz;
    
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    
    //hit = dda_traverse2<true, true, true>(fetcher, ray_dir, lid, ray_pos, stats);
    uint per_clock = 0;
    
    if (debug_type == DebugType.Timer) {
        per_clock = getRealtimeClock().x;
    }
    
    /*
    //output[id.xy] = float4(stats.full_halts / 8.0);
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(data, stats.total_iterations);
    GroupMemoryBarrierWithGroupSync();

    int diff = data;
    //int diff = WaveActiveSum(stats.total_iterations) / 64.0;
    output[id.xy] = float4(diff / 64.0);
    */
    
    float3 col = test(ray_dir, ray_pos);
    //col = (float3)hit2.block_pos % 4 / 4.0;
    //col = ((float3)thingy.block_pos.get().value % 4) / 4.0;
    switch (debug_type) {
        case DebugType.Combined:
            //col = ((float3)hit2.block_pos % 4) / 4.0;
            //col = stats.recursions / 64.0;    
            //col = ((ray_pos + ray_dir * stats.data - 0.01) % 1.0);    
            //col = stats.pause_resumes / 8.0;
            //col = float3(stats.total_iterations / 256.0, stats.recursions / 32.0, stats.bottom_level_iterations / 64.0);
            break;
        case DebugType.FullHalt:
            //col = abs(get_face_normal(hit2.face, sign(ray_dir)));
            //col = thingy.block_pos.get().value % 10 / 10.0;
            //col = stats.full_halts / 8.0;
            //col = hit ? 1 : 0;
            break;
        case DebugType.Iterations:
            /*
            if (stats.total_iterations < 256.0) {
                col = stats.total_iterations / 256.0;
            } else {
                col = float3(1, 0, 1);
            }
            */
            break;
        case DebugType.Recursions:
            //col = distance(ray_pos, thingy.world.get().value) / 10.0;
            //col = stats.data;
            //col = (hit ? 1 : 0) * (thingy.world.get().value % 10.0 / 10.0);
            break;
        case DebugType.Timer:
            uint clock = getRealtimeClock().x;
            uint elapsed = clock - per_clock;

            GroupMemoryBarrierWithGroupSync();
            AllMemoryBarrier();
            InterlockedMax(max_clock, elapsed);
            InterlockedMin(min_clock, elapsed);
            AllMemoryBarrier();
            GroupMemoryBarrierWithGroupSync();
            //col = (max_clock - min_clock) / 1024.0;
            col = (max_clock - min_clock) / 1024.0;
            break;
        case DebugType.World:
            //col = hit2.world % 10.0 / 10.0;
            //col = hit ? 1 : 0;
            break;
    }
    
    output[id.xy] = float4(col, 0);
    return;
}