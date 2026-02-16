#language slang 2026
#include <other.slang>

[[vk::binding(0, 0)]]
StructuredBuffer<uint64_t> svo_bitmasks_buffer;

[[vk::binding(1, 0)]]
StructuredBuffer<uint32_t> svo_indices_buffer;

[[vk::binding(2, 0)]]
StructuredBuffer<RayInput> ray_inputs;

[[vk::binding(3, 0)]]
RWStructuredBuffer<RayOutput> ray_outputs;

[[vk::binding(4, 0)]]
StructuredBuffer<Atomic<uint>> ray_counts;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint32_t3> indirect_dispatch_command;

static const int DEPTH = 5;
static const int TOTAL_SIZE = 1 << (DEPTH * 2);
DdaTraversalOutput trace_recurse<let K : int>(
    uint32_t base_index,
    uint64_t raw,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    float distance,
    uint16_t3 bounds_min,
    uint16_t3 bounds_max,
) {
    if (raw == 0) {
        return DdaTraversalOutput(false);
    }

    int voxel_size = 1 << max(K*2, 0);
    uint16_t3 min_bounds_local_space = bounds_min / voxel_size;
    uint16_t3 max_bounds_local_space = bounds_max / voxel_size;

    float3 modified_ray_pos = (ray_pos + ray_dir * (distance + 0.01)) / voxel_size;
    int16_t3 floored_pos = (int16_t3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = 0;


    [loop]
    for (uint8_t i = 0; i < 12; i++) {
        if (any(floored_pos < 0 | floored_pos >= (TOTAL_SIZE / voxel_size))) {
            return DdaTraversalOutput(false);
        }

        bool oob = any((floored_pos) < (min_bounds_local_space) | (int3)(floored_pos) > ((int3)max_bounds_local_space - 1));
        bool oob_loose = any((int3)(floored_pos) < ((int3)min_bounds_local_space-1) | floored_pos > (max_bounds_local_space));
        
        
        if (oob_loose) {
            return DdaTraversalOutput(false);
        }
        
        uint8_t3 local = (uint16_t3)floored_pos % (uint16_t3)4;
        uint8_t bit_index = local.z + local.y * 4 + local.x * 4 * 4;

        if (bit_index < 64 && ((raw >> bit_index) & 1) == 1 && !oob) {  
            float distance_offset = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * voxel_size) + 0.001;
            uint8_t face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
            if (K > 0) {
                // calculate child offset index, which is NOT the same as the bit index
                // we need to count the number of ones that come before (excluding) `bit_index` in `raw` 
                // create a mask for the first `n` bits
                uint64_t mask = (1 << bit_index) - 1;
                uint32_t child_offset_index = (uint32_t)countbits(mask & raw);
                uint32_t actual_child_index = base_index + child_offset_index;

                uint32_t new_base_index = svo_indices_buffer[actual_child_index];
                
                if (new_base_index != 0xFFFF) {
                    uint64_t new_raw = svo_bitmasks_buffer[actual_child_index];

                    uint16_t3 next_min_bounds_world_space = (uint16_t3)floored_pos * voxel_size;
                    uint16_t3 next_max_bounds_world_space = (uint16_t3)floored_pos * voxel_size + voxel_size;


                    var result = trace_recurse<K - 1>(new_base_index, new_raw, ray_dir, ray_pos, inv_dir, dir_sign, distance + distance_offset, next_min_bounds_world_space, next_max_bounds_world_space);
                    if (result.hit) {
                        return result;
                    }
                } else {
                    return DdaTraversalOutput(true, ray_pos + ray_dir * (distance + distance_offset), (uint3)floored_pos, face);
                }
            } else {
                return DdaTraversalOutput(true, ray_pos + ray_dir * (distance + distance_offset), (uint3)floored_pos, face);
            }
        }


        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));
    }

    return DdaTraversalOutput(false);
}

/*
struct IterativeRayShit {
    bool miss;
    uint32_t base_index;
    uint64_t raw;
    uint16_t3 bounds_min;
    uint16_t3 bounds_max;
    float distance;
}

IterativeRayShit trace_iterate_advance<let K : int>(
    IterativeRayShit previous,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
) {
    if (previous.raw == 0) {
        return IterativeRayShit(true);
    }

    int voxel_size = 1 << max(K*2, 0);
    uint16_t3 min_bounds_local_space = previous.bounds_min / voxel_size;
    uint16_t3 max_bounds_local_space = previous.bounds_max / voxel_size;

    float3 modified_ray_pos = (ray_pos + ray_dir * (previous.distance + 0.01)) / voxel_size;
    int16_t3 floored_pos = (int16_t3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = 0;


    [loop]
    for (uint8_t i = 0; i < 12; i++) {
        if (any(floored_pos < 0 | floored_pos >= (TOTAL_SIZE / voxel_size))) {
            return IterativeRayShit(true);
        }

        bool oob = any((floored_pos) < (min_bounds_local_space) | (int3)(floored_pos) > ((int3)max_bounds_local_space - 1));
        bool oob_loose = any((int3)(floored_pos) < ((int3)min_bounds_local_space-1) | floored_pos > (max_bounds_local_space));
        
        
        if (oob_loose) {
            return IterativeRayShit(true);
        }
        
        uint8_t3 local = (uint16_t3)floored_pos % (uint16_t3)4;
        uint8_t bit_index = local.z + local.y * 4 + local.x * 4 * 4;

        if (bit_index < 64 && ((previous.raw >> bit_index) & 1) == 1 && !oob) {  
            float distance_offset = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * voxel_size) + 0.001;
            uint8_t face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
            
            // calculate child offset index, which is NOT the same as the bit index
            // we need to count the number of ones that come before (excluding) `bit_index` in `raw` 
            // create a mask for the first `n` bits
            uint64_t mask = (1 << bit_index) - 1;
            uint32_t child_offset_index = (uint32_t)countbits(mask & raw);
            uint32_t actual_child_index = base_index + child_offset_index;

            uint32_t new_base_index = svo_indices_buffer[actual_child_index];
            
            if (new_base_index != 0xFFFF) {
                uint64_t new_raw = svo_bitmasks_buffer[actual_child_index];

                uint16_t3 next_min_bounds_world_space = (uint16_t3)floored_pos * voxel_size;
                uint16_t3 next_max_bounds_world_space = (uint16_t3)floored_pos * voxel_size + voxel_size;


                var result = trace_recurse<K - 1>(new_base_index, new_raw, ray_dir, ray_pos, inv_dir, dir_sign, distance + distance_offset, next_min_bounds_world_space, next_max_bounds_world_space);
                if (result.hit) {
                    return result;
                }
            } else {
                return DdaTraversalOutput(true, ray_pos + ray_dir * (distance + distance_offset), (uint3)floored_pos, face);
            }
    }


        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));
    }

    return DdaTraversalOutput(false);
}
*/

[shader("compute")]
[numthreads(1, 1, 1)]
void copyDispatchSize(uint3 id: SV_DispatchThreadID) {
    int raw = ray_counts[0].load(MemoryOrder.Relaxed);
    int size = (int)ceil((float)raw / 64);
    indirect_dispatch_command[0] = uint32_t3(size, 1, 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void traceRaysRecursive(uint3 id: SV_DispatchThreadID) {
    if (id.x >= ray_counts[0].load(MemoryOrder.Relaxed)) {
        return;
    }

    RayInput input = ray_inputs[id.x];
    float3 ray_dir = input.direction.xyz;
    float3 ray_pos = WaveReadLaneFirst(input.position.xyz);

    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    DdaTraversalOutput output = trace_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0.0, 0, 9999999);
    ray_outputs[id.x] = RayOutput(output.hit);
}