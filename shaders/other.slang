#ifndef OTHER
#define OTHER
#include <hash.slang>
#include <queue.slang>
#include <stack.slang>
#include <octree.slang>
#include <voxel.slang>
#include <debug_util.slang>

static const int MIP_LEVELS = 9;
static const int SIZE = 1 << (MIP_LEVELS-1);
static const uint INVALID = 0x3FFFFFF;

struct Fetcher {
    Texture3D<uint64_t> voxels;

    Voxel fetch(int3 position) {
        uint64_t raw = 0;

        if (all(position >= 0) && all(position < SIZE)) {
            raw = voxels.Load(int4(position, 0));
        }
        
        return Voxel.from_raw(raw);
    }

    static const uint8_t1 zero = (uint8_t)0;
    uint64_t fetch_raw(uint8_t3 position, uint8_t1 level = zero) {
        return voxels.Load(uint8_t4(position, level));
    }
}

static const uint8_t STARTING_LEVEL = MIP_LEVELS - 3;
static const uint8_t LEVEL_DIFF = 2;
static const uint8_t DDA_STEPS = 12;

public struct MultiLevelThingyOutput<bool hasFace, bool hasWorld, bool hasBlockPos>  {
    public Conditional<int, hasFace> face;
    public Conditional<float3, hasWorld> world;
    public Conditional<uint3, hasBlockPos> block_pos;
}

public struct TraceStats {
    public uint8_t1 total_iterations;
    public uint8_t1 recursions;
    public uint8_t1 bottom_level_iterations;
    public float3 data;
}

// cube: 6 quads
// each quad: 4x4 segments, total of 16

// ray intersection:
// input quad: id in [0..6], represented by 3 bits
// output quad: id in [0..6], represented by 3 bits
// input quad segment: 4 bits
// output quad segment: 4 bits

#define MAX(a, b)  (((a) > (b)) ? (a) : (b)) 


void dda_bitset_final<let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    float base_distance,
    float3 inv_dir,
    int8_t3 dir_sign,
    out MultiLevelThingyOutput<hasFace, hasWorld, hasBlockPos> output,
) {
    float3 modified_ray_pos = (ray_pos + ray_dir * base_distance) * 4.0;
    int3 floored_pos = (int3)floor(modified_ray_pos);
    uint8_t3 block_pos = (uint8_t3)(floored_pos / 4);
    uint64_t raw = fetcher.fetch_raw(block_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = false;

    [loop]
    for (uint8_t i = 0; i < 14; i++) {
        if (any((floored_pos / 4) != block_pos)) {
            block_pos = (uint8_t3)(floored_pos / 4);
            raw = fetcher.fetch_raw(block_pos);
        }

        int3 local = floored_pos % 4;
        int bit_index = local.z + local.y * 4 + local.x * 4 * 4;
        if (bit_index >= 0 && bit_index < 64 && ((raw >> bit_index) & 1) == 1) {
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) / 4.0);
            float distance = base_distance + local_dist;
            output.world.set(ray_pos + ray_dir * distance);
            output.block_pos.set((uint3)floored_pos);
            bool3 test_eqs = eqs;
            output.face.set(firstbithigh(test_eqs.x | test_eqs.y << 1 | test_eqs.z << 2));
            
            /*

            float tmin = 0.0;
            float tmax = 0.0;
            int entrance_face = 0;
            int exit_face = 0;
            bool hit = ray_box_intersection_2(inv_dir, dir_sign, ray_pos, block_pos, block_pos + 1, tmin, tmax, entrance_face, exit_face);

            float3 entrance = ray_pos + ray_dir * (tmin + 0.01);
            float3 exit = ray_pos + ray_dir * (tmax - 0.01);
            
            // convert coordinates to "cube" space (0..1)
            entrance = entrance % 1;
            exit = exit % 1;

            // get the uvs according to the face id...
            uint face_pair_index = face_pair_index(entrance_face, exit_face);
            float2 entrance_uvs = get_face_uvs(entrance_face, entrance);
            float2 exit_uvs = get_face_uvs(exit_face, exit);

            // convert uvs to segment ids
            uint2 entrance_segment = (uint2)floor(entrance_uvs * 4);
            uint2 exit_segment = (uint2)floor(exit_uvs * 4);

            uint face_1_segment_index = entrance_segment.x + entrance_segment.y * 4;
            uint face_2_segment_index = exit_segment.x + exit_segment.y * 4;

            output.world.set(float3(face_1_segment_index / 16.0));
            */

            /*
            float3 modified_ray_pos = (ray_pos + ray_dir * base_distance);
            uint16_t3 floored_pos = (int3)floor(modified_ray_pos);
            float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);

            bool3 eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
            int entrance_face = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
            side_dist += select(eqs, inv_dir, 0.0);

            float local_dist2 = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir)) + base_distance;
            float3 entrance = ray_pos + ray_dir * (local_dist2 + 0.1);
            entrance = entrance % 1.0;
            float2 entrance_uvs = get_face_uvs(entrance_face, entrance);
            uint2 entrance_segment = (uint2)floor(entrance_uvs * 4);     
            
            output.world.set(entrance);
            */

            return;
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, 1, 0) * inv_dir;
        floored_pos += select(eqs, 1, 0) * dir_sign;
    }
}

uint8_t face_pair_index(uint8_t face_1, uint8_t face_2) {
    // i love gpt (this works)
    return face_1 * (11 - face_1) / 2 + (face_2 - face_1 - 1);
}

void just_check_one_pls(
    half3 inv_dir,
    half3 ray_dir,
    int8_t3 dir_sign,
    float3 ray_pos,
    int16_t3 block_pos,
    float base_distance,
    out float tmin,
    out float tmax,
    out uint8_t entrance_face_axis,
    out uint8_t exit_face_axis, 
) {
    float3 modified_ray_pos = (ray_pos + ray_dir * base_distance);
    uint16_t3 floored_pos = (int3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    
    bool3 eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
    entrance_face_axis = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
    side_dist += select(eqs, inv_dir, 0.0);
    tmin = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir)) + base_distance;
    
    eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
    exit_face_axis = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
    side_dist += select(eqs, inv_dir, 0.0);    
    tmax = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir)) + base_distance;
}

uint get_precomputed_index(float3 ray_pos, half3 ray_dir, half3 inv_dir, int8_t3 dir_sign, float base_distance) {
    float3 modified_ray_pos = (ray_pos + ray_dir * base_distance);
    uint16_t3 floored_pos = (int3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    
    bool3 eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
    int entrance_face = global_face_2(firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2), dir_sign);
    side_dist += select(eqs, inv_dir, 0.0);

    float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir));
    float3 entrance = ray_pos + ray_dir * (local_dist + base_distance + 0.1);
    entrance = entrance % 1.0;
    float2 entrance_uvs = get_face_uvs(entrance_face, entrance);
    uint2 entrance_segment = (uint2)floor(entrance_uvs * 16);        
    uint face_1_segment_index = entrance_segment.x + entrance_segment.y * 16;
    
    eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
    int exit_face = global_face_2(firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2), -dir_sign);
    side_dist += select(eqs, inv_dir, 0.0);
    
    local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir));
    float3 exit = ray_pos + ray_dir * (local_dist + base_distance - 0.1);
    exit = exit % 1.0;
    float2 exit_uvs = get_face_uvs(exit_face, exit);
    uint2 exit_segment = (uint2)floor(exit_uvs * 16);
    uint face_2_segment_index = exit_segment.x + exit_segment.y * 16;

    if (entrance_face > exit_face) {
        int tmp = exit_face;
        exit_face = entrance_face;
        entrance_face = tmp;

        float tmp2 = face_2_segment_index;
        face_2_segment_index = face_1_segment_index;
        face_1_segment_index = tmp2;
    }

    // convert to indices
    uint face_pair_index = face_pair_index(entrance_face, exit_face);

    // convert to lookup index
    return face_pair_index << 16 | face_1_segment_index << 8 | face_2_segment_index;
}

static const uint8_t SUBDIVISIONS = 16; 
static const int BITS_PER_FACE_SEGMENT = 8; 
static const bool FUNKY = false; 

bool dda_bitset<let tracing: bool>(
    uint64_t raw,
    half3 ray_dir,
    float3 ray_pos,
    float base_distance,
    int16_t3 block_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    StructuredBuffer<uint64_t> dda_precomputed,
    inout Conditional<TraceStats, tracing> stats,
) {

    if (raw == 0) {
        return false;
    }

    if (raw == uint64_t.maxValue) {
        return true;
    }

    if (FUNKY) {
        float tmin = 0.0;
        float tmax = 0.0;
        uint8_t entrance_face = 0;
        uint8_t exit_face = 0;
        ray_box_intersection_2(inv_dir, dir_sign, ray_pos, block_pos, block_pos + 1, tmin, tmax, entrance_face, exit_face);
        //just_check_one_pls(inv_dir, ray_dir, dir_sign, ray_pos, block_pos, base_distance, tmin, tmax, entrance_face, exit_face);

        uint8_t global_entrance_face = global_face_2(entrance_face, dir_sign);
        uint8_t fst = global_entrance_face;
        float fst_dst = tmin + 0.001;
        float3 fst_world = (ray_pos + ray_dir * fst_dst) - block_pos;
        float2 fst_uvs = get_face_uvs(fst, fst_world);
        uint8_t2 fst_seg = (uint8_t2)clamp((uint8_t2)floor(fst_uvs * SUBDIVISIONS), 0, SUBDIVISIONS);
        uint8_t fst_index = fst_seg.x + fst_seg.y * SUBDIVISIONS;

        uint8_t global_exit_face = global_face_2(exit_face, -dir_sign);
        uint8_t snd = global_exit_face;
        float snd_dst = tmax - 0.001;
        float3 snd_world = (ray_pos + ray_dir * snd_dst) - block_pos;
        float2 snd_uvs = get_face_uvs(snd, snd_world);
        uint8_t2 snd_seg = (uint8_t2)clamp((uint8_t2)floor(snd_uvs * SUBDIVISIONS), 0, SUBDIVISIONS);
        uint8_t snd_index = snd_seg.x + snd_seg.y * SUBDIVISIONS;


        // swap them around to uphold the fst < snd
        if (fst > snd) {
            uint8_t tmp = snd;
            snd = fst;
            fst = tmp;

            uint8_t tmp2 = snd_index;
            snd_index = fst_index;
            fst_index = tmp2;
        }



        uint face_pair_index = face_pair_index(fst, snd);



        
        
        uint index = face_pair_index << (BITS_PER_FACE_SEGMENT + BITS_PER_FACE_SEGMENT) | (uint)fst_index << BITS_PER_FACE_SEGMENT | snd_index;

        /*

        */

        // check mask
        //index = get_precomputed_index(ray_pos, ray_dir, inv_dir, dir_sign, base_distance);
        /*
        if ((dda_precomputed[index] & raw) == 0) {
            return false;
        }
        */

        bool hit = (dda_precomputed.Load(index) & raw) != 0;
        return hit;

        /*
        if (tracing && hit) {
            TraceStats _stats = stats.get().value;
            _stats.data = fst_index / 256.0;
            //_stats.data = countbits(dda_precomputed[index] & raw) / 12.0;
            //printf("entry: %i exit: %i s1: %i s2: %i", entrance_face, exit_face, face_1_segment_index, face_2_segment_index);
            //_stats.data = float3(snd_seg.x / 16.0, fst_seg.x / 16.0, 0);
            //_stats.data = float3(fst_index / 256.0, snd_index / 256.0, face_pair_index / 15.0);
            //_stats.data = global_face_2(entrance_face, dir_sign) / 6.0;
            stats.set(_stats);
            return hit;
        }

        return (dda_precomputed[index] & raw) != 0;
        */
    } 
    
    float3 modified_ray_pos = (ray_pos + ray_dir * base_distance) * 4.0;
    uint16_t3 floored_pos = (int3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool now_inside = all((floored_pos / 4) == block_pos);

    [loop]
    for (uint8_t i = 0; i < 14; i++) {
        if (all((floored_pos / 4) == block_pos)) {
            now_inside = true;
        }

        if (any((floored_pos / 4) != block_pos) && now_inside) {
            break;
        }

        uint8_t3 local = (uint16_t3)floored_pos % (uint16_t3)4;
        uint8_t bit_index = local.z + local.y * 4 + local.x * 4 * 4;
        if (now_inside && bit_index >= 0 && bit_index < 64 && ((raw >> bit_index) & 1) == 1) {
            return true;
        }

        bool3 eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));

        if (tracing) {
            TraceStats _stats = stats.get().value;
            _stats.total_iterations++;
            _stats.bottom_level_iterations++;
            stats.set(_stats);
        }
    }


    return false;
}



bool dda_bitset_simple(
    uint64_t raw,
    float3 ray_pos,
    float3 ray_dir,
    float distance,
    int16_t3 block_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    out uint3 umokayumhello,
) {
    if (raw == 0) {
        return false;
    }

    if (raw == uint64_t.maxValue) {
        return true;
    }

    float3 modified_ray_pos = (ray_pos + ray_dir * distance) * 4.0;
    uint16_t3 floored_pos = (int3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool now_inside = all((floored_pos / 4) == block_pos);
    bool3 eqs = 0;

    [loop]
    for (uint8_t i = 0; i < 14; i++) {
        if (all((floored_pos / 4) == block_pos)) {
            now_inside = true;
        }

        if (any((floored_pos / 4) != block_pos) && now_inside) {
            break;
        }

        uint8_t3 local = (uint16_t3)floored_pos % (uint16_t3)4;
        uint8_t bit_index = local.z + local.y * 4 + local.x * 4 * 4;
        if (now_inside && bit_index >= 0 && bit_index < 64 && ((raw >> bit_index) & 1) == 1) {
            umokayumhello = floored_pos;
            return true;
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));
    }


    return false;
}


// MAX(K, 0) is the mip level
bool dda_multi_level<let K : int16_t, let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool, let tracing: bool>(
    Fetcher fetcher,
    half3 ray_dir,
    float3 ray_pos,
    uint lid_index,
    uint16_t3 bounds_local_space,
    half3 inv_dir,
    int8_t3 dir_sign,
    float base_distance,
    out float out_distance,
    uint64_t parent,
    StructuredBuffer<uint64_t> dda_precomputed,
    inout Conditional<TraceStats, tracing> stats,
) {
    float3 modified_ray_pos = (ray_pos + ray_dir * (base_distance - 0.01)) / (1 << MAX(K, 0));
    int16_t3 floored_pos = (int16_t3)floor(modified_ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = false;

    //int face_local = 0;
    for (uint8_t i = 0; i < DDA_STEPS; i++) {
        // skip if we are outside the bounds of the map
        if (any(floored_pos < 0 | floored_pos >= (SIZE >> MAX(K, 0)))) {
            break;
        }
        
        // skip if we are outside the bounds of the parent
        if (((any((floored_pos+1) < (bounds_local_space & uint16_t3(0xFF)) | (floored_pos-1) >= ((bounds_local_space >> 8) & uint16_t3(0xFF))))) && K < STARTING_LEVEL) {
            break;
        }
        
        
        uint8_t3 local = (uint16_t3)floored_pos % (uint16_t3)4;
        uint8_t bit_index = local.z + local.y * 4 + local.x * 4 * 4;

        if (bit_index >= 0 && bit_index < 64 && ((parent >> bit_index) & 1) == 1) {
            // handle partially full nodes / leaf voxel nodes
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * (1 << MAX(K, 0)));
            float distance = base_distance + local_dist;

            
            // completely full nodes are well, completely full. we can do the leaf-level logic since we know we can skip them
            if (K > 0) {
                uint64_t raw = fetcher.fetch_raw((uint8_t3)(floored_pos), MAX(K - LEVEL_DIFF, 0));
                if (tracing) {
                    TraceStats _stats = stats.get().value;
                    _stats.recursions++;
                    stats.set(_stats);
                }

                const uint8_t SOME_THING = (uint8_t)(K - (MAX(K-LEVEL_DIFF, 0)));
                uint8_t3 next_min_bounds_local_space = ((uint8_t3)floored_pos << SOME_THING);
                uint8_t3 next_max_bounds_local_space = (uint8_t3)(floored_pos+1) << SOME_THING;
            
                uint16_t3 next_bounds_local_space = (uint16_t3)next_min_bounds_local_space | ((uint16_t3)next_max_bounds_local_space << 8);

                if (dda_multi_level<K-LEVEL_DIFF, hasFace, hasWorld, hasBlockPos, tracing>(fetcher, ray_dir, ray_pos, lid_index, next_bounds_local_space, inv_dir, dir_sign, max(distance - 0.01, 0), out_distance, raw, dda_precomputed, stats)) {
                    return true;
                }
            } else {
                uint64_t raw = fetcher.fetch_raw((uint8_t3)(floored_pos), MAX(K, 0));
                if (dda_bitset(raw, ray_dir, ray_pos, (max(distance - 0.08, 0)), floored_pos, inv_dir, dir_sign, dda_precomputed, stats)) {
                    out_distance = (max(distance - 0.08, 0));
                    return true;
                }
            }
        }

        if (tracing) {
            TraceStats _stats = stats.get().value;
            _stats.total_iterations++;
            stats.set(_stats);
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, (half3)0);
        floored_pos += select(eqs, dir_sign, (int16_t3)0);
        //int3 eqs = select(t, 1, 0);
        //int face_local = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
    }
    

    return false;
}

bool dda_traverse<let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool, let tracing: bool>(   
    Fetcher fetcher,
    half3 ray_dir,
    float3 ray_pos,
    uint3 lid,
    StructuredBuffer<uint64_t> dda_precomputed,
    out MultiLevelThingyOutput<hasFace, hasWorld, hasBlockPos> output,
    out Conditional<TraceStats, tracing> stats,
) {
    //return dda<128, hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, output);
    stats.set(TraceStats(0, 0, 0));

    half3 inv_dir = (half3)1.0 / max(abs(ray_dir), 0.001);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    uint lid_index = lid.x + lid.y * 16;
    float distance = 0.0;
    uint3 block_pos = 0;

    /*    
    if (!dda<4, 128>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return false;
    }
    
    if (!dda<2, 128>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return false;
    }

    if (dda<0, 128>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        output.world.set(ray_pos + ray_dir * distance);
        output.block_pos.set(block_pos);
        return true;
    }
    */
    return false;


    /*
    bool hit = dda_multi_level<STARTING_LEVEL, hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, lid_index, (uint16_t3(SIZE) << 8), inv_dir, dir_sign, 0, distance, uint64_t.maxValue, dda_precomputed, stats);
    
    if (hit) {
        //dda_bitset_final(fetcher, ray_dir, ray_pos, distance - 0.01, inv_dir, dir_sign, output);
    }

    return hit;
    */
}

struct DdaTraversalOutput {
    bool hit;
    float3 world;
    uint3 block_pos;
}

DdaTraversalOutput dda_traverse_near_high_quality(   
    Fetcher fetcher,
    half3 ray_dir,
    float3 ray_pos,
) {
    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    float distance = 0.0;
    uint3 block_pos = 0;


    
    return DdaTraversalOutput(false);
}

DdaTraversalOutput dda_traverse_far_low_quality(   
    Fetcher fetcher,
    half3 ray_dir,
    float3 ray_pos,
) {
    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    float distance = 0.0;
    uint3 block_pos = 0;

    if (dda_near_high_quality<64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return DdaTraversalOutput(true, ray_pos + ray_dir * distance, block_pos);
    }
    
    if (dda_far_low_quality<1, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return DdaTraversalOutput(true, ray_pos + ray_dir * distance, block_pos);
    }    


    /*
    if (!dda_far_low_quality<3, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return DdaTraversalOutput(false);
    }   

    if (!dda_far_low_quality<2, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return DdaTraversalOutput(false);
    }
    
    if (!dda_far_low_quality<1, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return DdaTraversalOutput(false);
    }

    if (dda_far_low_quality<0, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return DdaTraversalOutput(true, ray_pos + ray_dir * distance, block_pos);
    }
    */
    
    return DdaTraversalOutput(false);
}

bool dda_far_low_quality<let K : int16_t, let N : int>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    out float distance,
    out uint3 umokayumhello,
) {
    umokayumhello = 0;
    float3 modified_pos = (ray_pos + ray_dir * distance) / (1 << MAX(K, 0));
    uint16_t3 floored_pos = (uint16_t3)floor(modified_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = false;

    [loop]
    for (int i = 0; i < N; i++) {
        if (any(floored_pos < 0 | floored_pos >= (SIZE >> MAX(K, 0)))) {
            return false;
        }

        uint64_t raw = fetcher.fetch_raw((int3)(floored_pos), MAX(K, 0));

        if (raw != 0) {
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * (1 << MAX(K, 0)));
            
            umokayumhello = floored_pos;
            distance += local_dist;
            return true;
            
            /*
            if (K == 0) {
                umokayumhello = floored_pos;
                if (dda_bitset_simple(raw, ray_pos, ray_dir, distance + local_dist, floored_pos, inv_dir, dir_sign, umokayumhello)) {
                    return true;
                }
            } else {
                distance += local_dist;
                return true;
            }
            */
            

        }

        // creates an int3 for the smallest value in side_dist
        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, (half3)0);
        floored_pos += select(eqs, dir_sign, (int16_t3)0);

        //distance += min3(reconst.x, reconst.y, reconst.z);
        //face_local = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);

    }


    return false;
}


bool dda_near_high_quality<let N : int>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    out float distance,
    out uint3 umokayumhello,
) {
    uint16_t3 floored_pos = (uint16_t3)floor(ray_pos);
    float3 side_dist = ((dir_sign * (floored_pos - ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = false;

    [loop]
    for (int i = 0; i < N; i++) {
        if (any(floored_pos < 0 | floored_pos >= (SIZE))) {
            return false;
        }

        uint64_t raw = fetcher.fetch_raw((int3)(floored_pos), 0);

        if (raw != 0) {
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir));
            
            if (dda_bitset_simple(raw, ray_pos, ray_dir, local_dist, floored_pos, inv_dir, dir_sign, umokayumhello)) {
                distance = local_dist;
                return true;
            }

        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, (half3)0);
        floored_pos += select(eqs, dir_sign, (int16_t3)0);
    }


    return false;
}


float3 per_block_unique_colour(uint3 block_pos) {
    float3 col = normalize(hash33(block_pos * float3(23.231, -435.4354, 9412.1)));
    return lerp(col, 1.0, 0.8f);
}

// TODO: need to calculate how "close" we get to the surface...
float3 dda_shadownate(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);
    float3 color = 1.0;

    for (int i = 0; i < 512; i++) {
        Voxel voxel = fetcher.fetch((int3)(floored_pos));

        if (voxel.active) {
            return 1.0;
        }

        /*
        if (voxel.active && voxel.refractive) {
            color *= per_block_unique_colour((uint3)floor(floored_pos));
        }

        if (voxel.active && !voxel.refractive) {
            return 0.0;
        }
        */

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    return 0.0;
}

public static const int3[] FACE_NORMALS = {
    int3(1, 0, 0),
    int3(-1, 0, 0),
    int3(0, 1, 0),
    int3(0, -1, 0),
    int3(0, 0, 1),
    int3(0, 0, -1),
};

float3 get_face_normal(int face, float3 sign) {
    return -(float3)(face == int3(0, 1, 2)) * sign;
}

int global_face(int face, float3 sign) {
    if (face == 0) {
        return sign.x > 0 ? 1 : 0;
    } else if (face == 1) {
        return sign.y > 0 ? 3 : 2;
    } else if (face == 2) {
        return sign.z > 0 ? 5 : 4;
    }

    return -1;
}

static const uint8_t4[4 * 4] null_data = { uint8_t4(0, 0, 0, 0) };
static const uint8_t1[16 * 16] null_data2 = { 0 };

struct SurfaceData {
    uint8_t4[4 * 4] colors;
    //uint8_t1[16 * 16] sky_visibility;
    //uint[4 * 4] colors2;

    __init() {
        colors = null_data;
        //sky_visibility = null_data2;
        // colors2 = null_data2;
    }
}

float3 unpack_gi_color(uint packed) {
    uint8_t4 unpacked_current = uint8_t4(0);
    unpacked_current.x = (uint8_t)(packed & 0xFF);
    unpacked_current.y = (uint8_t)((packed >> 8) & 0xFF);
    unpacked_current.z = (uint8_t)((packed >> 16) & 0xFF);
    return unpacked_current.xyz / 255.0;
}

uint pack_gi_color(float3 color) {
    uint8_t4 unpacked = uint8_t4(clamp(color, 0, 1) * 255, 0);
    uint packed = 0;
    packed |= unpacked.x;
    packed |= ((uint)unpacked.y) << 8;
    packed |= ((uint)unpacked.z) << 16;
    return packed;
}

#endif