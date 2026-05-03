#include <other.slang>
#include <noises.slang>

struct SparseVoxelOctree {
    StructuredBuffer<uint32_t> indices_buffer;
    StructuredBuffer<uint64_t> bitmasks_buffer;
    StructuredBuffer<uint64_t> aabbs_buffer;
}

struct Statistics {
    int iterations;
    int buffer_fetches;
    int exit_type;
    int active_threads_per_call;
    int calls;
}

static const int DEPTH = 5;
static const float EPS = 0.00001;
static const uint32_t BOTTOM_NODE = 0xFFFFFFFF;
static const uint32_t FULL_NODE = 0xFFFFFFFF-1; 
static const int TOTAL_SIZE = 1 << (DEPTH * 2);


// https://tavianator.com/2022/ray_box_boundary.html
bool ray_box_intersection(
    float3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    int16_t3 aabb_min,
    int16_t3 aabb_max,
    out float _tmin,
) {
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);


    _tmin = tmin;

    return max(tmin, 0) <= tmax;
}

bool ray_box_intersection3(
    float3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    int16_t3 aabb_min,
    int16_t3 aabb_max,
    out float _tmin,
    out int face,
) {
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);


    _tmin = tmin;

    bool3 eqs = max3(k1.x, k1.y, k1.z) == k1;
    face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);

    return max(tmin, 0) <= tmax;
}

bool ray_box_intersection2(
    float3 inv_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    float3 aabb_min,
    float3 aabb_max,
) {
    float3 t1 = (aabb_min - ray_pos) * inv_dir * dir_sign;
    float3 t2 = (aabb_max - ray_pos) * inv_dir * dir_sign;

    float3 k1 = min(t1, t2);
    float3 k2 = max(t1, t2);

    float tmin = max3(k1.x, k1.y, k1.z);
    float tmax = min3(k2.x, k2.y, k2.z);

    return max(tmin, 0) <= tmax;
}


struct DdaTraversalOutput {
    bool hit;
    float3 world; // this is not entirely accurate as there can be floating point issues here and there. DO NOT TRUST THIS FOR EXACT RESULTS!!
    uint3 block_pos; // this is always 100% accurate when we reach the bottom most node of the acceleration structure
    float distance;
    uint8_t face;
}

// TODO: OPTIMIZE OPTIMIZE OPTIMIZE OPTIMIZE OPTIMIZE
// 1. optimize occupancy, we still have buffer latency left to hide
// 2. optimize raw speed, it is still relatively slow on my HW
DdaTraversalOutput dda_trace_recurse<let K : int, let END : int, let MAX_DISTANCE : int>(
    uint32_t base_index,
    uint32_t actual_child_index,
    uint64_t bounds,
    float3 ray_dir,
    float3 ray_pos,
    float3 inv_dir,
    int8_t3 dir_sign,
    float distance,
    SparseVoxelOctree svo,
    uint16_t3 parent_world_space_origin,
    uint8_t face,
    inout Statistics stats,
) {
    uint16_t3 aabb_min = (uint16_t3)uint64_t3(bounds & 0b1111111111, (bounds >> 10) & 0b1111111111, (bounds >> 20) & 0b1111111111);
    uint16_t3 aabb_max = (uint16_t3)uint64_t3((bounds >> 30) & 0b1111111111, (bounds >> 40) & 0b1111111111, (bounds >> 50) & 0b1111111111);                     

    float tmin = 0;
    int face2 = 0;
    if (ray_box_intersection3(inv_dir, dir_sign, ray_pos, aabb_min, aabb_max, tmin, face2)) {
        // make sure we are dealing with leaf nodes and that they represent volume cuboids
        if (K == 0 && ((bounds >> 60) & 1) == 1) {
            float3 world = ray_pos + ray_dir * (tmin - EPS);
            uint3 block_pos = floor(world);
               
            // apply "accurate" block pos since we know the bounds of the full node and input face
            int3 tmp_sign = max(0, -dir_sign);
            block_pos[face2] = aabb_max[face2] - 1;
            stats.exit_type = 4;
            return DdaTraversalOutput(true, world, block_pos, tmin, face2);
        }
    } else {
        return DdaTraversalOutput(false);
    }

    uint64_t raw = svo.bitmasks_buffer[actual_child_index];
    stats.buffer_fetches += 1;

    /*
    if (WaveActiveCountBits(true) < 16) {
        return DdaTraversalOutput(false);
    }
    */

    uint16_t voxel_size = (uint16_t)1 << max(K*2, 0);
    
    // do everything in world space first, and then we convert to local space
    // ray will always be in the parent's working space, so we offset by the parent origin and clamp coords to [0..4)
    float3 modified_ray_pos = ((ray_pos + ray_dir * distance - parent_world_space_origin) / (float)voxel_size);
    modified_ray_pos = clamp(modified_ray_pos, EPS, 4 - EPS);
    
    // yes technically you can repr this by a single byte and do packing/unpacking ops
    // I tried that, it did not help perf.
    int8_t3 floored_pos = (int8_t3)floor(modified_ray_pos);
    floored_pos = (int8_t3)clamp(floored_pos, 0, 4);

    float3 side_dist = ((dir_sign * ((floored_pos - modified_ray_pos) + 0.5) + 0.5) * inv_dir);
    bool3 eqs = false;    
    
    
    for (uint8_t i = 0; i < 12; i++) {
        stats.iterations += 1;
        /*
        if ((i*voxel_size) > MAX_DISTANCE) {
            return DdaTraversalOutput(false); 
        }
        */

        // we know that floored pos is the local coordinate directly anyways
        uint8_t3 local = (uint8_t3)floored_pos;
        uint8_t bit_index = local.x | local.y << 2 | local.z << 4; 

        // world space position of the node origin
        uint16_t3 origin = parent_world_space_origin + floored_pos * voxel_size;

        stats.calls += 1;
        stats.active_threads_per_call += WaveActiveCountBits((((raw >> bit_index) & 1) == 1));
        


        [branch]
        if ((((raw >> bit_index) & 1) == 1)) {  
            // still need an offset though, for the normals :(((
            float distance_offset = length(select(eqs, (side_dist - inv_dir) * voxel_size, 0));
            float distance_to_surface = distance + distance_offset;

            // check if we have reached the bottom most node and exit if so
            [branch]
            if (K > 0) {
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
                stats.buffer_fetches += 1;

                // check if we reached a full node and if we did, we must somehow calculate block position and world position accurately 
                if (new_base_index == FULL_NODE || K <= END) {
                    float tmin = 0;
                    ray_box_intersection(inv_dir, dir_sign, ray_pos, origin, origin+voxel_size, tmin);
                    float3 world = ray_pos + ray_dir * (tmin - EPS);
                    uint3 block_pos = floor(world);

                    // apply "accurate" block pos since we know the bounds of the full node and input face
                    int3 tmp_sign = max(0, -dir_sign);
                    block_pos[face] = origin[face] + tmp_sign[face] * (voxel_size-1);

                    return DdaTraversalOutput(true, world, block_pos, tmin, face);
                }
                
                uint64_t bounds = svo.aabbs_buffer[actual_child_index];  
                stats.buffer_fetches += 1;

                var result = dda_trace_recurse<K - 1, END, MAX_DISTANCE>(new_base_index, actual_child_index, bounds, ray_dir, ray_pos, inv_dir, dir_sign, distance_to_surface, svo, origin, face, stats);
                if (result.hit) {
                    return result;
                }
                
            } else {
                float3 world = ray_pos + ray_dir * (distance_to_surface - EPS);
                stats.exit_type = 3;

                /*
                for (int i = 0; i < 8; i++) {
                    stats.iterations += 1;

                    if (ray_box_intersection2(inv_dir, dir_sign, ray_pos, clamp(origin + hash33(i+origin), origin, origin+1), clamp(origin + hash33(i+origin * 3.1232), origin, origin+1))) {
                        return DdaTraversalOutput(true, world, origin, face);
                    }
                } 
                */
                
                /*
                if (sin(world.x * 3) > 0) {
                    return DdaTraversalOutput(true, world, origin, face);
                }
                */
                    
                /*
                float3 center = origin + 0.5;
                for (int i = 0; i < 64; i++) {
                    stats.iterations += 1;
                    float d = length(center - world) - 0.3 + snoise(world * 3.2) * 0.1;
                    if (d < 0.02) {
                        return DdaTraversalOutput(true, world, origin, 0, face);
                    }    
                    world += ray_dir * d;
                } 
                */
                // TODO: i have removed this as it does nothing: clamp(world, parent_world_space_origin, parent_world_space_origin+voxel_size*4)
                // if something goes wrong add it again 
                return DdaTraversalOutput(true, world, origin, distance_to_surface - EPS, face);
            }
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));

        // exit early if we leave the space
        if (any(floored_pos < 0 | floored_pos > 3)) {
            return DdaTraversalOutput(false);
        }
    }

    return DdaTraversalOutput(false);
}

float volume_aabb_aabb_overlap(int3 aabb_min, int3 aabb_max, float3 aabb2_center, float aabb2_half_length) {
    float3 aabb_min_2 = aabb2_center - aabb2_half_length;
    float3 aabb_max_2 = aabb2_center + aabb2_half_length;

    bool3 a = aabb_min <= aabb_max_2;
    bool3 b = aabb_min_2 <= aabb_max;

    float3 maxxino = max(aabb_min, aabb_min_2);
    float3 minnimo = min(aabb_max, aabb_max_2);

    float3 v = (maxxino - minnimo);

    let vol = max(-v.x*v.y*v.z, 0);
    return vol;
}

bool aabb_aabb_intersection(int3 aabb_min, int3 aabb_max, float3 aabb2_center, float aabb2_half_length) {
    float3 aabb_min_2 = aabb2_center - aabb2_half_length;
    float3 aabb_max_2 = aabb2_center + aabb2_half_length;

    bool3 a = aabb_min <= aabb_max_2;
    bool3 b = aabb_min_2 <= aabb_max;

    return all(a && b);
}

// Source - https://stackoverflow.com/a/41457896
// Posted by josch, modified by community. See post 'Timeline' for change history
// Retrieved 2026-04-26, License - CC BY-SA 3.0
bool sphere_aabb_intersection(int3 Bmin, int3 Bmax, float3 C, float r) {
    float dmin = 0;
    float r2 = r*r;

    /*
    for(int i = 0; i < 3; i++ ) {
        if( C[i] < Bmin[i] ) {
            dmin += pow( C[i] - Bmin[i], 2 ); 
        } else if ( C[i] > Bmax[i] ) {
            dmin += pow( C[i] - Bmax[i], 2 );
        }
    }
    return dmin <= r2;
    */ 

    float3 a = select(C < Bmin, pow( C - Bmin, 2 ), 0);
    dmin += a.x + a.y + a.z;
    float3 b = select((C > Bmax) & !(C < Bmin), pow( C - Bmax, 2 ), 0);
    dmin += b.x + b.y + b.z;    
    return dmin <= r2;
}

float calculate_aabb_overlap<let K : int>(
    uint32_t base_index,
    uint64_t raw,
    float3 cube_pos,
    float cube_half_length,
    SparseVoxelOctree svo,
    uint16_t3 parent_world_space_origin,
    inout Statistics statistics,
) {
    uint16_t voxel_size = (uint16_t)1 << max(K*2, 0);

    int3 min_bounds = max(floor((cube_pos - cube_half_length - parent_world_space_origin) / voxel_size), 0);
    int3 max_bounds = min(floor((cube_pos + cube_half_length - parent_world_space_origin) / voxel_size), 3);

    float sum = 0.0;
    float total_volume = pow(cube_half_length*2, 3);

    [loop]
    for (int x = min_bounds.x; x <= max_bounds.x; x++) {
        [loop]
        for (int y = min_bounds.y; y <= max_bounds.y; y++) {
            [loop]
            for (int z = min_bounds.z; z <= max_bounds.z; z++) {
                int3 localt = int3(x,y,z);               
                                
                uint16_t3 origin = parent_world_space_origin + localt * voxel_size;

                uint8_t3 local = (uint8_t3)localt;
                uint8_t bit_index = local.x | local.y << 2 | local.z << 4;

                statistics.iterations += 1;

                [branch]
                if (base_index == FULL_NODE || (((raw >> bit_index) & 1) == 1)) {  
                    [branch]
                    if (K > 0 && (base_index != BOTTOM_NODE || base_index == FULL_NODE)) {
                        uint64_t mask = ((uint64_t)1 << bit_index) - 1;
                        uint32_t child_offset_index = (uint32_t)countbits(mask & raw);
                    
                        uint32_t actual_child_index = base_index + child_offset_index;
                        uint32_t new_base_index = svo.indices_buffer[actual_child_index];
                    
                        if (base_index == FULL_NODE || new_base_index == FULL_NODE) {
                            float overlap_volume = calculate_aabb_overlap<K - 1>(FULL_NODE, uint64_t.maxValue, cube_pos, cube_half_length, svo, origin, statistics);
                            sum += overlap_volume;
                            continue;
                        }
                    
                        uint64_t new_raw = svo.bitmasks_buffer[actual_child_index];
                        statistics.buffer_fetches += 1;
                        sum += calculate_aabb_overlap<K - 1>(new_base_index, new_raw, cube_pos, cube_half_length, svo, origin, statistics);
                    } else {
                        float overlap_volume = volume_aabb_aabb_overlap(origin, origin + voxel_size, cube_pos, cube_half_length);
                        sum += overlap_volume;
                        continue;
                    }
                }
            }
        }    
    }

    return sum;
}
DdaTraversalOutput trace_shi<let MAX_DISTANCE: uint>(
    float3 ray_pos,
    float3 ray_dir,
    SparseVoxelOctree svo,
    inout Statistics iter,
) {
    if (any(ray_pos < 0 | ray_pos >= TOTAL_SIZE)) {
        return DdaTraversalOutput(false);
    }

    float3 inv_dir = 1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    return dda_trace_recurse<DEPTH - 1, 0, MAX_DISTANCE>(1, 0, svo.aabbs_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0.0, svo, 0, 0, iter);
}


float cube_overlap_density_svo(
    float3 cube_pos,
    float cube_half_length,
    SparseVoxelOctree svo,
    inout Statistics iter,
) {
    float total_volume = pow(cube_half_length*2, 3);
    return calculate_aabb_overlap<DEPTH - 1>(1, svo.bitmasks_buffer[0], cube_pos, cube_half_length, svo, 0, iter) / total_volume;
}

float cube_overlap_density_svt(
    float3 cube_pos,
    float cube_half_length,
    Fetcher fetcher,
    inout Statistics statistics,
) {
    float total_volume = pow(cube_half_length*2, 3);
    int k = ceil(cube_half_length);

    float sum = 0.0;

    [loop]
    for (int x = -k; x <= k; x++) {
        [loop]
        for (int y = -k; y <= k; y++) {
            [loop]
            for (int z = -k; z <= k; z++) {
                int3 localt = int3(x,y,z);               
                                
                statistics.iterations += 1;

                uint16_t3 pos = cube_pos + int16_t3(x,y,z);
                if (fetcher.is_solid(pos)) {
                    sum += volume_aabb_aabb_overlap(pos, pos + 1, cube_pos, cube_half_length);
                }
            }
        }    
    }

    return sum / total_volume;
}

float planar_overlap_density_svt(
    float3 plane_pos,
    float plane_half_length,
    int axis, // 1,2,3
    Fetcher fetcher,
    inout Statistics statistics,
) {
    float total_volume = pow(plane_half_length*2, 2);
    int k = ceil(plane_half_length);

    float sum = 0.0;

    [loop]
    for (int x = -k; x <= k; x++) {
        [loop]
        for (int y = -k; y <= k; y++) {
            statistics.iterations += 1;

            int16_t3 offset = 0;
            if (axis == 0) {
                offset.y = x;
                offset.z = y;
            } else if (axis == 1) {
                offset.x = x;
                offset.z = y;
            } else if (axis == 2) {
                offset.x = x;
                offset.y = y;
            }

            uint16_t3 pos = plane_pos + offset;
            if (fetcher.is_solid(pos)) {
                sum += volume_aabb_aabb_overlap(pos, pos + 1, plane_pos, plane_half_length);
            }
        }    
    }

    return sum / total_volume;
}