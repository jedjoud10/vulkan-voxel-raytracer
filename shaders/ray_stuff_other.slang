#include <other.slang>

struct RayIntermediateOutput {
    bool hit;
}

struct RayInput {
    float4 position;
    float4 direction;
    uint index;
}

struct RayOutput {
    float data;
}

struct SparseVoxelOctree {
    StructuredBuffer<uint32_t> indices_buffer;
    StructuredBuffer<uint64_t> bitmasks_buffer;
}

struct Statistics {
    int iterations;
    float poop;
    int buffer_fetches;
}

static const int DEPTH = 7;
static const int TOTAL_SIZE = 1 << (DEPTH * 2);

// TODO: OPTIMIZE OPTIMIZE OPTIMIZE OPTIMIZE OPTIMIZE
// 1. optimize occupancy, we still have buffer latency left to hide
// 2. optimize raw speed, it is still relatively slow on my HW
DdaTraversalOutput dda_trace_recurse<let K : int, let END : int>(
    uint32_t base_index,
    uint64_t raw,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    float distance,
    SparseVoxelOctree svo,
    int16_t3 parent_world_space_origin,
    inout Statistics stats,
) {
    // quick check
    if (any(ray_pos < 0 | ray_pos >= TOTAL_SIZE)) {
        return DdaTraversalOutput(false);
    }

    int voxel_size = 1 << max(K*2, 0);

    /*
    // quick check if we have few children; avoids doing DDA
    int inactive_lanes = (WaveGetLaneCount() - WaveActiveCountBits(true));
    uint4 mask_ballot = WaveActiveBallot(true);
    uint64_t mask = mask_ballot.x | (uint64_t)mask_ballot.y << 32;
    if (inactive_lanes > countbits(raw) && K == 3) {
        uint lane_index = WaveGetLaneIndex();
        bool is_active = (mask >> lane_index) & 1 == 1;
        

        AllMemoryBarrierWithWaveSync();
        uint first_lane_wave_index = WaveReadLaneFirst(lane_index);
        int broadcasted_voxel_size = WaveReadLaneFirst(voxel_size);
        uint64_t broadcasted_raw = WaveReadLaneFirst(raw);
        int16_t3 broadcasted_parent_world_space_origin = WaveReadLaneFirst(parent_world_space_origin);
        float3 broadcasted_ray_pos = WaveReadLaneFirst(ray_pos);
        half3 broadcasted_ray_inv_dir = WaveReadLaneFirst(inv_dir);
        int8_t3 broadcasted_ray_dir_sign = WaveReadLaneFirst(dir_sign);
        AllMemoryBarrierWithWaveSync();

        if (!is_active) {
            stats.poop=broadcasted_ray_inv_dir.x;
        }

        // this needs to be a number 0..countbits(raw) unique to each "helper" lane
        AllMemoryBarrierWithWaveSync();
        uint child_offset_index = WavePrefixCountBits(is_active);
        AllMemoryBarrierWithWaveSync();

        // only need the required lanes...
        bool hit = false;

        // must NOT be the main lane, must do the work on the helper lanes...
        // this does mean that only ONE lane can be accelerated by other helper lanes...
        AllMemoryBarrierWithWaveSync();
        if (lane_index != first_lane_wave_index && is_active) {

            //printf("%i", child_offset_index);
            if (child_offset_index < countbits(broadcasted_raw)) {
                // need to get the index of the `child_offset_index`-th set bit in `raw`
                uint64_t remaining = broadcasted_raw;
                uint8_t bit_index = fixedfirstbitlow(remaining);
                for (int i = 0; i < child_offset_index; i++) {
                    bit_index = fixedfirstbitlow(remaining);
                    remaining &= (remaining - 1);
                }

                // we store nodes in a simple linear order
                uint8_t z = bit_index % 4;
                uint8_t y = (bit_index / 4) % 4;
                uint8_t x = (bit_index / 16) % 4; 

                // get bounds of node
                uint16_t3 world_space_bounds_min = broadcasted_parent_world_space_origin + (uint16_t3)(uint16_t3(x, y, z) * (uint16_t)broadcasted_voxel_size);
                uint16_t3 world_space_bounds_max = world_space_bounds_min + (uint16_t3)((uint16_t)broadcasted_voxel_size);

                // check node vs ray
                float t = 99999;
                if (ray_box_intersection(broadcasted_ray_inv_dir, broadcasted_ray_dir_sign, broadcasted_ray_pos, (float3)world_space_bounds_min-0.1, (float3)world_space_bounds_max+0.1, t)) {
                    hit = true;
                }
            }
        } else {
        }
        
        // if all of them missed, means that no need to traverse this one
        AllMemoryBarrierWithWaveSync();
        if (WaveActiveAllTrue(!hit)) {
            return DdaTraversalOutput(false);
        }
    }
    */

    // do everything in world space first, and then we convert to local space
    float3 modified_ray_pos = ((ray_pos + ray_dir * distance) / (float)voxel_size);
    int16_t3 floored_pos = (int16_t3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = 0;
    
    // FLOORED POS IS ALWAYS IN THE RANGE [0..4)
    // idea from https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
    // it works very nicely!
    floored_pos -= parent_world_space_origin / voxel_size;
    //floored_pos = floored_pos % 4;

    /*
    if (countbits(raw) < 4) {
        stats.poop += 1;
        // flip substrings inside the 64-bit mask according to sign direction
        // to allow us to iterate through the bits from NEAR to FAR along the ray
        // this will still iterate over chunks that we don't need to iterate over, but distances are exact at least!
        uint64_t remaining = flip_according_dir_sign(raw, dir_sign);    
        bool hit = false;
        while (remaining != 0) {
            // must undo our bit shit, we need the actual bit index inside the original `raw` value
            uint8_t bit_index = fetch_bit_index_unflipped(fixedfirstbitlow(remaining), dir_sign);
            remaining &= (remaining - 1);

            // we store nodes in a simple linear order
            uint8_t z = bit_index % 4;
            uint8_t y = (bit_index / 4) % 4;
            uint8_t x = (bit_index / 16) % 4; 

            // get bounds of node
            uint16_t3 world_space_bounds_min = parent_world_space_origin + (uint16_t3)(uint16_t3(x, y, z) * (uint16_t)voxel_size);
            uint16_t3 world_space_bounds_max = world_space_bounds_min + (uint16_t3)((uint16_t)voxel_size);

            // calculate the packed child offset. it is the sum of the previous high bits before `bit_index` in `raw` 
            uint64_t mask = (1 << bit_index) - 1;
            uint32_t child_offset_index = (uint32_t)countbits(mask & raw);
            uint32_t actual_child_index = base_index + child_offset_index;

            // check node vs ray
            float t = 99999;
            if (raw != 0 && ray_box_intersection(inv_dir, dir_sign, ray_pos, (float3)world_space_bounds_min, (float3)world_space_bounds_max, t)) {
                hit = true;
                break;
            }
        }

        if (!hit) {
            return DdaTraversalOutput(false);
        }
    }
    */
    
    [loop]
    for (uint8_t i = 0; i < 12; i++) {
        // exit early if we leave the space
        // HACK: this is a looser bound to accomodate for floating point shit fard
        if (any(floored_pos < -1 | floored_pos > 4)) {
            return DdaTraversalOutput(false);
        }

        // we know that floored pos is the local coordinate directly anyways
        uint8_t3 local = (uint8_t3)floored_pos % 4;
        uint8_t bit_index = local.z + local.y * 4 + local.x * 4 * 4;
        
        bool oob = any((floored_pos) < 0 | floored_pos > 3);

        [branch]
        if (bit_index < 64 && ((raw >> bit_index) & 1) == 1 && !oob) {  
            // still need an offset though, for the normals :(((
            float distance_offset = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * voxel_size) + 0.01;

            // world space position of the node origin
            uint3 origin = (uint3)parent_world_space_origin + (uint3)floored_pos * (uint)voxel_size;

            // LOD kinda?
            // I dislike this because it's technically "cheating" but I could not find any other way to decrease bandwidth pressure (we have too many buffer reads! this is killing perf!)
            float distance_to_surface = distance + distance_offset;
            float distance_to_cube = length(ray_pos - (origin + voxel_size / 2));
            bool lod_check = pow(distance_to_cube, 0.40) < (pow(max(voxel_size, 8), 0.7)) || (K > 5);

            [branch]
            if (K > END && base_index != 0xFFFFFFFF && lod_check) {
                // calculate child offset index, which is NOT the same as the bit index
                // we need to count the number of ones that come before (excluding) `bit_index` in `raw` 
                // create a mask for the first `n` bits
                uint64_t mask = ((uint64_t)1 << bit_index) - 1;
                uint32_t child_offset_index = (uint32_t)countbits(mask & raw);

                uint32_t actual_child_index = base_index + child_offset_index;
                uint32_t new_base_index = svo.indices_buffer[actual_child_index];
                uint64_t new_raw = svo.bitmasks_buffer[actual_child_index];
                stats.buffer_fetches += 1;

                var result = dda_trace_recurse<K - 1, END>(new_base_index, new_raw, ray_dir, ray_pos, inv_dir, dir_sign, distance_to_surface, svo, origin, stats);
                if (result.hit) {
                    return result;
                }
            } else {
                uint8_t face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
                return DdaTraversalOutput(true, ray_pos + ray_dir * distance_to_surface, origin, face);
            }
        }


        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));
        stats.iterations += 1;
    }

    return DdaTraversalOutput(false);
}

// https://tavianator.com/2022/ray_box_boundary.html
bool ray_box_intersection(
    half3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    float3 aabb_min,
    float3 aabb_max,
    out float _tmin,
) {
    float tmin = 0.0;
    float tmax = 999999;
    
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    tmin = max3(k1.x, k1.y, k1.z);
    tmax = min3(k2.x, k2.y, k2.z);


    _tmin = tmin;

    return max(tmin, 0) <= tmax;
}
bool ray_box_intersection(
    half3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    float3 aabb_min,
    float3 aabb_max,
) {
    float tmin = 0.0;
    float tmax = 999999;
    
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    tmin = max3(k1.x, k1.y, k1.z);
    tmax = min3(k2.x, k2.y, k2.z);

    return max(tmin, 0) <= tmax;
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

// flip axii of 64-bit brick mask according to dir sign
// make sure to use the `fetch_bit_index_unflipped` to undo this when you get the firstbitlow index, since it will not be in the original position
// https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
// the purpose of this is to iterate through the children FRONT to BACK along the ray. (should) avoid incurring a performance to hidden voxels 
uint64_t flip_according_dir_sign(uint64_t raw, int8_t3 dir_sign) {
    // reverse the bits inside 4 bit segments
    if (dir_sign.z < 0) {
        // reverse 4 bit segments
        // swap odd and even bits
        raw = ((raw >> 1) & 0x5555555555555555) | ((raw & 0x5555555555555555) << 1);
        // swap consecutive pairs
        raw = ((raw >> 2) & 0x3333333333333333) | ((raw & 0x3333333333333333) << 2);
    }

    // swap 16-bit segments (2 bytes)
    if (dir_sign.y < 0) {
        // swap nibbles ... 
        raw = ((raw >> 4) & 0x0F0F0F0F0F0F0F0F) | ((raw & 0x0F0F0F0F0F0F0F0F) << 4);
        // swap bytes
        raw = ((raw >> 8) & 0x00FF00FF00FF00FF) | ((raw & 0x00FF00FF00FF00FF) << 8);
    }

    // reverse the 4-bit segments inside 16-bit segment
    if (dir_sign.x < 0) {
        // swap 2-byte long pairs
        raw = ((raw >> 16) & 0x0000FFFF0000FFFF) | ((raw & 0x0000FFFF0000FFFF) << 16);
        // swap 4-byte long pairs
        raw = ((raw >> 32) & 0x00000000FFFFFFFF) | ((raw & 0x00000000FFFFFFFF) << 32);
    }

    return raw;
}

// undoes the flipping that we did originally
uint8_t fetch_bit_index_unflipped(uint8_t bit_index, int8_t3 dir_sign) {
    uint8_t z = bit_index % 4;
    uint8_t y = (bit_index / 4) % 4;
    uint8_t x = (bit_index / 16) % 4; 

    if (dir_sign.z < 0) {
        z = 3 - z;
    }

    if (dir_sign.y < 0) {
        y = 3 - y;
    }

    if (dir_sign.x < 0) {
        x = 3 - x;
    }
    
    return z + y * 4 + x * 4 * 4;
}

// this is faster than the "distance" based checks since this does not need to sort children to avoid "ghost see through", since we only care about "any hit"
// surprisingly runs at full occupancy!!
// still slower than the DDA one though... which is weird because the DDA one has lower occupancy and much higher buffer latency...
bool any_hit_aabb_test_recurse<let K : int, let END : int>(
    uint32_t base_index,
    uint64_t raw,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    uint16_t3 block_origin,
    SparseVoxelOctree svo,
    out float tmin,
    inout bool finished,
) {
    int voxel_size = 1 << max(K*2, 0);

    // flip substrings inside the 64-bit mask according to sign direction
    // to allow us to iterate through the bits from NEAR to FAR along the ray
    // this will still iterate over chunks that we don't need to iterate over, but distances are exact at least!
    uint64_t remaining = flip_according_dir_sign(raw, dir_sign);    
    while (remaining != 0) {
        // must undo our bit shit, we need the actual bit index inside the original `raw` value
        uint8_t bit_index = fetch_bit_index_unflipped(fixedfirstbitlow(remaining), dir_sign);
        remaining &= (remaining - 1);

        // we store nodes in a simple linear order
        uint8_t z = bit_index % 4;
        uint8_t y = (bit_index / 4) % 4;
        uint8_t x = (bit_index / 16) % 4; 

        // get bounds of node
        uint16_t3 world_space_bounds_min = block_origin + (uint16_t3)(uint16_t3(x, y, z) * (uint16_t)voxel_size);
        uint16_t3 world_space_bounds_max = world_space_bounds_min + (uint16_t3)((uint16_t)voxel_size);

        // calculate the packed child offset. it is the sum of the previous high bits before `bit_index` in `raw` 
        uint64_t mask = (1 << bit_index) - 1;
        uint32_t child_offset_index = (uint32_t)countbits(mask & raw);
        uint32_t actual_child_index = base_index + child_offset_index;

        // check node vs ray
        float t = 99999;
        if (raw != 0 && ray_box_intersection(inv_dir, dir_sign, ray_pos, (float3)world_space_bounds_min, (float3)world_space_bounds_max, t)) {
            tmin = 10;
            uint32_t new_base_index = svo.indices_buffer[actual_child_index];
            if (new_base_index != 0xFFFF) {
                if (K > END) {
                    uint64_t new_raw = svo.bitmasks_buffer[actual_child_index];
                    if (any_hit_aabb_test_recurse<K - 1, END>(new_base_index, new_raw, ray_pos, inv_dir, dir_sign, world_space_bounds_min, svo, tmin, finished)) {
                        finished = true;
                        return true;
                    }     
                } else {
                    finished = true;
                    return true;
                }
            } else {
                finished = true;
                return true;
            }
        }
    }

    if (K == 0) {
        finished = false;
    }

    return false;
}


/*
bool any_hit_cooperative_matrix(
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    uint16_t3 block_origin,
    SparseVoxelOctree svo
) {
    int stack_size = 0;
    uint32_t[DEPTH] base_indices;
    uint64_t[DEPTH] bitmasks;
    uint64_t[DEPTH] remainings;

    // add root node
    base_indices[0] = 1;
    bitmasks[0] = svo.bitmasks_buffer[0];
    remainings[0] = flip_according_dir_sign(bitmasks[0], dir_sign);
    stack_size++;

    while (stack_size > 0) {
        // depth is gonna be the index inside the stack!
        let bottom_up_depth = stack_size - 1;
        let top_down_depth = DEPTH - bottom_up_depth; // from root=DEPTH, going to leaf nodes DEPTH=0
        let i = stack_size-1;
        let voxel_size = 1 << (top_down_depth * 2);

        // fetch bitmask and base index
        uint64_t raw = bitmasks[i];
        uint32_t base_index = base_indices[i];

        // hit something!
        if (base_index == 0xFFFFFFFF) {
            return true;
        }

        uint64_t remaining = remainings[i];
        while (remaining != 0) {
            // must undo our bit shit, we need the actual bit index inside the original `raw` value
            uint8_t bit_index = fetch_bit_index_unflipped(fixedfirstbitlow(remaining), dir_sign);
            remaining &= (remaining - 1);
        
            // we store nodes in a simple linear order
            uint8_t z = bit_index % 4;
            uint8_t y = (bit_index / 4) % 4;
            uint8_t x = (bit_index / 16) % 4; 
        
            // get bounds of node
            uint16_t3 world_space_bounds_min = block_origin + (uint16_t3)(uint16_t3(x, y, z) * (uint16_t)voxel_size);
            uint16_t3 world_space_bounds_max = world_space_bounds_min + (uint16_t3)((uint16_t)voxel_size);
        
            // calculate the packed child offset. it is the sum of the previous high bits before `bit_index` in `raw` 
            uint64_t mask = (1 << bit_index) - 1;
            uint32_t child_offset_index = (uint32_t)countbits(mask & raw);
            uint32_t actual_child_index = base_index + child_offset_index;
        
            // check node vs ray
            float t = 99999;
            if (raw != 0 && ray_box_intersection(inv_dir, dir_sign, ray_pos, (float3)world_space_bounds_min, (float3)world_space_bounds_max, t)) {
                uint32_t new_base_index = svo.indices_buffer[actual_child_index];
                uint64_t new_raw = svo.bitmasks_buffer[actual_child_index];
                
                if (top_down_depth > 0) {
                    // save current node status to stack and go deeper...
                    remainings[0]
                } else {
                    return true;
                }
            }
        }
    }


    return false;
}
*/

DdaTraversalOutput trace_shi(
    float3 ray_pos,
    half3 ray_dir,
    SparseVoxelOctree svo,
    out Statistics iter,
) {
    half3 inv_dir = 1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    return dda_trace_recurse<DEPTH - 1, 0>(1, svo.bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0.0, svo, 0, iter);
}

bool any_hit_trace_fine(
    float3 ray_pos,
    half3 ray_dir,
    SparseVoxelOctree svo,
    out Statistics stats,
) {
    stats = Statistics();
    half3 inv_dir = 1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    float tmin = 0;
    bool finished = false;
    return any_hit_aabb_test_recurse<DEPTH-1, 0>(1, svo.bitmasks_buffer[0], ray_pos, inv_dir, dir_sign, 0, svo, tmin, finished);
}

/*
bool any_hit_trace_coarse(
    float3 ray_pos,
    half3 ray_dir,
    SparseVoxelOctree svo,
    out float tmin,
) {
    float3 avg_ray_dir = WaveActiveSum(ray_dir.xyz) / 64.0;
    half3 avg_inv_dir = (half3)1.0 / abs(avg_ray_dir);
    int8_t3 avg_dir_sign = (int8_t3)sign(avg_ray_dir);

    return any_hit_aabb_test_recurse<DEPTH-1, 2>(1, svo.bitmasks_buffer[0], ray_pos, avg_inv_dir, avg_dir_sign, 0, svo, tmin);
    //return dda_trace_recurse<DEPTH-1, 2>(1, svo.bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0.0, 0, 99999, svo).hit;
}
*/