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
    int buffer_fetches;
    int exit_type;
}

static const int DEPTH = 5;
static const float EPS = 0.00001;
static const uint32_t BOTTOM_NODE = 0xFFFFFFFF;
static const uint32_t FULL_NODE = 0xFFFFFFFF-1; 
static const int TOTAL_SIZE = 1 << (DEPTH * 2);

// TODO: OPTIMIZE OPTIMIZE OPTIMIZE OPTIMIZE OPTIMIZE
// 1. optimize occupancy, we still have buffer latency left to hide
// 2. optimize raw speed, it is still relatively slow on my HW
DdaTraversalOutput dda_trace_recurse<let K : int, let END : int, let MAX_DISTANCE : int>(
    uint32_t base_index,
    uint64_t raw,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    float distance,
    SparseVoxelOctree svo,
    uint16_t3 parent_world_space_origin,
    bool full_node,
    uint8_t previous_face,
    inout Statistics stats,
) {
    // quick check
    if (any(ray_pos < 0 | ray_pos >= TOTAL_SIZE)) {
        return DdaTraversalOutput(false);
    }

    uint16_t voxel_size = (uint16_t)1 << max(K*2, 0);

    // do everything in world space first, and then we convert to local space
    float3 modified_ray_pos = ((ray_pos + ray_dir * distance) / (float)voxel_size);
    
    // ray will always be in the parent's working space, so we offset by the parent origin and clamp coords to [0..4)
    modified_ray_pos -= parent_world_space_origin / voxel_size;
    modified_ray_pos = clamp(modified_ray_pos, EPS, 4 - EPS);
    uint16_t3 floored_pos = (uint16_t3)floor(modified_ray_pos);
    floored_pos = (uint16_t3)clamp(floored_pos, 0, 4);

    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    uint8_t face = previous_face;
    bool3 eqs = false;    
    
    [loop]
    for (uint8_t i = 0; i < 12; i++) {
        // exit early if we leave the space
        if (any(floored_pos < 0 | floored_pos > 3)) {
            return DdaTraversalOutput(false);
        }

        // we know that floored pos is the local coordinate directly anyways
        uint8_t3 local = (uint8_t3)floored_pos;
        uint8_t bit_index = local.z | local.y << 2 | local.x << 4;
        
        [branch]
        if ((bit_index < 64 && ((raw >> bit_index) & 1) == 1) || full_node) {  
            // still need an offset though, for the normals :(((
            float distance_offset = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * voxel_size);
            if (distance_offset > MAX_DISTANCE) {
                return DdaTraversalOutput(false); 
            }

            // world space position of the node origin
            uint16_t3 origin = parent_world_space_origin + floored_pos * voxel_size;

            // LOD kinda?
            // I dislike this because it's technically "cheating" but I could not find any other way to decrease bandwidth pressure (we have too many buffer reads! this is killing perf!)
            float distance_to_surface = distance + distance_offset;
            float distance_to_cube = length(ray_pos - (origin + voxel_size / 2));
            bool lod_check = pow(distance_to_cube * 0.3, 0.9) < (max(voxel_size, 64)) || (K > 5);
            lod_check = true; // disables it...

            [branch]
            if (full_node) {
                if (K > 0) {
                    stats.exit_type = 2;
                    return dda_trace_recurse<K - 1, END, MAX_DISTANCE>(0, uint64_t.maxValue, ray_dir, ray_pos, inv_dir, dir_sign, distance_to_surface, svo, origin, true, face, stats);
                } else {
                    float3 world = ray_pos + ray_dir * (distance_to_surface - EPS);
                    float3 whatever_this_is = modified_ray_pos + ray_dir * distance_offset + parent_world_space_origin / voxel_size;
                    stats.exit_type = 2;
                    return DdaTraversalOutput(true, clamp(world, parent_world_space_origin, parent_world_space_origin+voxel_size*4), origin, face);
                }
            }

            // check if we have reached the bottom most node and exit if so
            [branch]
            if (K > 0 && base_index != BOTTOM_NODE && lod_check) {
                // calculate child offset index, which is NOT the same as the bit index
                // we need to count the number of ones that come before (excluding) `bit_index` in `raw` 
                // create a mask for the first `n` bits
                uint64_t mask = ((uint64_t)1 << bit_index) - 1;
                uint32_t child_offset_index = (uint32_t)countbits(mask & raw);


                // POSSIBLE OPTIMIZATION: store a "delta offset" compared to the base node, which could allow us to encode this as a u16 instead...
                //    - we can store the starting index for an entire level (depth=0,1,2,3,...) of the tree in a wayyy smaller buffer and use that as an offsetter as well, since we know that child indices are greater than parent indices!
                // this should allow us to use u16, even though we have an extra buffer fetch, but that one should be cheap because it is depth based (uses `K`, compiler should store that in scalar mem)
                uint32_t actual_child_index = base_index + child_offset_index;
                uint32_t new_base_index = svo.indices_buffer[actual_child_index];

                // check if we reached a full node and if we did, we must somehow calculate block position and world position accurately 
                if (new_base_index == FULL_NODE || K <= END) {
                    stats.exit_type = 2;
                    return dda_trace_recurse<K - 1, END, MAX_DISTANCE>(new_base_index, uint64_t.maxValue, ray_dir, ray_pos, inv_dir, dir_sign, distance_to_surface, svo, origin, true, face, stats);
                }

                // TODO:
                // ANOTHER POSSIBLE OPTIMIZATION: have a separate "5-bvh" mode for nodes if we can construct proper bvhs for them during traversal
                // for super flat / planar nodes, this should work in our favour because it could represent the entire brick node (u64) in wayyy less memory
                // assuming AABB bounds are quantized (2 bits per axii, since we use the range 0..4), a single AABB would take 2*3 * 2 = 12 bits
                // we can store 5 BVHs in one u64, but we can do better than; we can store them separately in their own buffer
                //    this way, if we have < 5 BVHs for the node, we don't need to fetch 64 whole bits!
                // each BVH would need to store data to its children nodes though... fuck
                // ok solution: each BVH node also contains a COUNT and CHILD OFFSET. since we can have up to 64 children in a single node, both these values must be represented by 6 bits
                // we can order the children in `svo.indices_buffer` and `svo.bitmasks_buffer` to our own liking afterwards, they don't need to follow the z-y-x bitmask DDA order
                // each BVH thing would then need 2*3 * 2 + 6 + 6 = 24 bits fuck.... WAIT IM FUCKING STUPID CHILD COUNT IS THE VOLUME OF THE BVH ITSELF
                // bruh
                // ok 6 bits removed, down to 18... we just need... to get rid of 2 more bits........ argh.....
                // easiest solution is to use less bits for the offset. that would limit the number of bhv nodes in total if we have a bunch of ones that only use a few nodes but should be fine
                // 2*3*2 + 4 = 16 bits YAY!!!!!!
                // we can then do a different intersection test using those BVH nodes instead, should be cheaper than traversing a nearly empty DDA cell 
                uint64_t new_raw = svo.bitmasks_buffer[actual_child_index];
                stats.buffer_fetches += 1;

                var result = dda_trace_recurse<K - 1, END, MAX_DISTANCE>(new_base_index, new_raw, ray_dir, ray_pos, inv_dir, dir_sign, distance_to_surface, svo, origin, false, face, stats);
                if (result.hit) {
                    return result;
                }
            } else {
                float3 world = ray_pos + ray_dir * (distance_to_surface - EPS);
                stats.exit_type = 3;
                return DdaTraversalOutput(true, clamp(world, parent_world_space_origin, parent_world_space_origin+voxel_size*4), origin, face);
            }
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));
        stats.iterations += 1;
    }

    return DdaTraversalOutput(false);
}


DdaTraversalOutput trace_shi<let MAX_DISTANCE: uint>(
    float3 ray_pos,
    half3 ray_dir,
    SparseVoxelOctree svo,
    out Statistics iter,
) {
    half3 inv_dir = 1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    return dda_trace_recurse<DEPTH - 1, 0, MAX_DISTANCE>(1, svo.bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0.0, svo, 0, false, 0, iter);
}