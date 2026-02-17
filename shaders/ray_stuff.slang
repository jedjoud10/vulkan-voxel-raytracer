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
DdaTraversalOutput dda_trace_recurse<let K : int>(
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


                    var result = dda_trace_recurse<K - 1>(new_base_index, new_raw, ray_dir, ray_pos, inv_dir, dir_sign, distance + distance_offset, next_min_bounds_world_space, next_max_bounds_world_space);
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

// https://tavianator.com/2022/ray_box_boundary.html
bool ray_box_intersection(
    half3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    int16_t3 aabb_min,
    int16_t3 aabb_max,
    out float _tmin,
) {
    float tmin = 0.0;
    float tmax = 999999;

    /*
    bool3 sign = dir_sign > 0;
    float3 bmin = select(sign, aabb_min, aabb_max);
    float3 bmax = select(!sign, aabb_min, aabb_max);

    float3 dmin = (bmin - ray_pos) * inv_dir * dir_sign;
    float3 dmax = (bmax - ray_pos) * inv_dir * dir_sign;

    tmin = max3(dmin.x, dmin.y, dmin.z);
    tmax = min3(dmax.x, dmax.y, dmax.z);
    */

    for (int i = 0; i < 3; i++) {
        bool sign = dir_sign[i] > 0;
        float bmin = select(sign, aabb_min[i], aabb_max[i]);
        float bmax = select(!sign, aabb_min[i], aabb_max[i]);

        float dmin = (bmin - ray_pos[i]) * inv_dir[i] * dir_sign[i];
        float dmax = (bmax - ray_pos[i]) * inv_dir[i] * dir_sign[i];

        tmin = max(dmin, tmin);
        tmax = min(dmax, tmax);
    }

    _tmin = tmin;
    return tmin < tmax;
}

bool cone_box_intersection(
    half3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    int16_t3 aabb_min,
    int16_t3 aabb_max,
) {
    float angle = 3.1415 / 100.0;
    float3 ray_dir = copysign(1.0 / inv_dir, dir_sign);
    float3 center = ((float3)aabb_max + (float3)aabb_min) * 0.5;
    float3 k = (float3)aabb_max - (float3)aabb_min;
    float aabb_radius = max3(k.x, k.y, k.z);

    // https://en.wikipedia.org/wiki/Vector_projection
    float3 a = center - ray_pos;
    float3 a_1 = (length(a) * dot(normalize(a), normalize(ray_dir))) * ray_dir;
    float3 a_2 = a - a_1;
    float box_to_line_distance = length(a_2);
    float distance_along_ray = length(a_1);
    
    // y = x * tan(theta)
    float circle_slice_radius = tan(angle) * distance_along_ray;
    
    // check if sphere contains center
    float d = circle_slice_radius - (box_to_line_distance);

    return d < 30000;
}

// FIXME: for some reason firstbitlow does not support uint64_t... even though it does support them... idk
uint8_t fixedfirstbitlow(uint64_t value) {
    uint64_t val = __intCast<uint64_t>(value);
    uint2 halves = __asuint2(val);
    uint lowestBitHalf = halves.x != 0 ? halves.x : halves.y;
    uint count = spirv_asm {
        OpExtInst $$uint result glsl450 FindILsb $lowestBitHalf
    };

    if (count == ~0u)
        return uint8_t(-1);

    if (halves.x == 0)
        count += 32;

    return uint8_t(count);
}

bool aabb_cone_test_recurse<let K : int>(
    uint32_t base_index,
    uint64_t raw,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    uint16_t3 block_origin,
) {
    if (raw == 0) {
        return false;
    }

    int voxel_size = 1 << max(K*2, 0);

    uint64_t remaining = raw;
    for (uint8_t i = 0; i < countbits(raw); i++) {
        uint8_t bit_index = fixedfirstbitlow(remaining);
        remaining &= remaining - 1;

        uint8_t z = bit_index % 4;
        uint8_t y = (bit_index / 4) % 4;
        uint8_t x = (bit_index / 16) % 4; 

        uint16_t3 world_space_bounds_min = block_origin + (uint16_t3)(uint16_t3(x, y, z) * (uint16_t)voxel_size / 2);
        uint16_t3 world_space_bounds_max = world_space_bounds_min + (uint16_t3)((uint16_t)voxel_size / 2);

        uint32_t actual_child_index = base_index + i;
        uint32_t new_base_index = svo_indices_buffer[actual_child_index];
        
        if (new_base_index != 0xFFFF) {
            uint64_t new_raw = svo_bitmasks_buffer[actual_child_index];
            bool hit = cone_box_intersection(inv_dir, dir_sign, ray_pos, world_space_bounds_min, world_space_bounds_max);

            if (hit) {
                if (K > 3) {
                    if (aabb_cone_test_recurse<K - 1>(new_base_index, new_raw, ray_dir, ray_pos, inv_dir, dir_sign, world_space_bounds_min)) {
                        return true;
                    }
                } else {
                    return true;
                }
            }
        } else {
            return true;
        }
    }

    return false;
}


bool aabb_test_recurse<let K : int>(
    uint32_t base_index,
    uint64_t raw,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    uint16_t3 block_origin,
    inout float min_distance_hit,
) {
    if (raw == 0) {
        return false;
    }

    int voxel_size = 1 << max(K*2, 0);

    uint64_t remaining = raw;
    for (uint8_t i = 0; i < countbits(raw); i++) {
        uint8_t bit_index = fixedfirstbitlow(remaining);
        remaining &= remaining - 1;

        uint8_t z = bit_index % 4;
        uint8_t y = (bit_index / 4) % 4;
        uint8_t x = (bit_index / 16) % 4; 

        uint16_t3 world_space_bounds_min = block_origin + (uint16_t3)(uint16_t3(x, y, z) * (uint16_t)voxel_size / 2);
        uint16_t3 world_space_bounds_max = world_space_bounds_min + (uint16_t3)((uint16_t)voxel_size / 2);

        uint32_t actual_child_index = base_index + i;
        uint32_t new_base_index = svo_indices_buffer[actual_child_index];

        uint64_t new_raw = svo_bitmasks_buffer[actual_child_index];
        float tmin = 0.0;
        bool hit = ray_box_intersection(inv_dir, dir_sign, ray_pos, world_space_bounds_min, world_space_bounds_max, tmin);
        
        if (new_base_index != 0xFFFF) {
            if (hit) {
                if (K > 3) {
                    aabb_test_recurse<K - 1>(new_base_index, new_raw, ray_dir, ray_pos, inv_dir, dir_sign, world_space_bounds_min, min_distance_hit);
                } else {
                    min_distance_hit = min(min_distance_hit, tmin);
                }
            }
        } else {
            min_distance_hit = min(min_distance_hit, tmin);
        }
    }

    return false;
}

// 00100011100110
// 

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
    /*
    DdaTraversalOutput output = dda_trace_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0.0, 0, 9999999);
    ray_outputs[id.x] = RayOutput(output.hit);
    */
    bool output = aabb_cone_test_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0);
    float min_distance = 999999;
    //bool output2 = aabb_test_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0, min_distance);
    //bool output = aabb_test_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0);
    //ray_outputs[id.x] = RayOutput(output, output2);
    ray_outputs[id.x] = RayOutput(output ? 1 : 0);
}