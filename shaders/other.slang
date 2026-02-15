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

    uint64_t fetch_raw_2(uint8_t3 position, uint8_t1 level = zero) {
        var pos = uint8_t4(position, level);

        if (WaveActiveAllEqual(pos)) {            
            // only one thread should fetch            
            return WaveReadLaneFirst(voxels.Load(pos));
        } else {
            //sorry poopy
            return voxels.Load(pos);
        }

        return false;
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

bool dda_bitset_simple(
    uint64_t raw,
    float3 ray_pos,
    float3 ray_dir,
    inout float distance,
    int16_t3 block_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    out uint3 brick_pos,
    out uint8_t face,
) {
    if (raw == 0) {
        return false;
    }

    float tmp = distance;
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
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir)) / 4.0;
            distance += local_dist - 0.1;

            brick_pos = floored_pos;
            face = (uint8_t)firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
            return true;
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, 0.0);
        floored_pos += select(eqs, (int16_t3)dir_sign, int16_t3(0));
    }

    distance = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir)) / 4.0; + tmp;
    return false;
}


struct DdaTraversalOutput {
    bool hit;
    float3 world;
    uint3 block_pos;
    uint8_t face;
}

DdaTraversalOutput dda_traverse_near_high_quality(   
    Fetcher fetcher,
    half3 ray_dir,
    float3 ray_pos,
) {
    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    uint3 block_pos = 0;
    float distance = 0;

    return dda_near_high_quality<128>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance);
}

DdaTraversalOutput dda_traverse_far_low_quality(   
    Fetcher fetcher,
    half3 ray_dir,
    float3 ray_pos,
) {
    float distance = 0.0;
    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);

    /*
    if (dda_near_high_quality<64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance, block_pos)) {
        return DdaTraversalOutput(true, ray_pos + ray_dir * distance, block_pos);
    }
    */

        /*
        if (!dda_far_low_quality<4, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance)) {
            return DdaTraversalOutput(false);
        }    

        if (!dda_far_low_quality<2, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance)) {
            return DdaTraversalOutput(false);
        }  
        
        return dda_near_high_quality<32>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance);
        */


    
    for (int i = 0; i < 1; i++) {
        if (!dda_far_low_quality<4, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance)) {
            return DdaTraversalOutput(false);
        }    

        if (!dda_far_low_quality<1, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance)) {
            return DdaTraversalOutput(false);
        }  
        
        var v = dda_near_high_quality<8>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance);
        if (v.hit) {
            return v;
        }  
    }
    
    
    //return dda_near_high_quality<64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance);

    /*
    if (!dda_far_low_quality<4, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance)) {
        return DdaTraversalOutput(false);
    }   

    if (!dda_far_low_quality<2, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance)) {
        return DdaTraversalOutput(false);
    }
    */
        
    /*
    if (!dda_far_low_quality<1, 64>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, distance)) {
        return DdaTraversalOutput(false);
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
    inout float distance,
) {
    float3 modified_pos = (ray_pos + ray_dir * distance) / (1 << MAX(K, 0));
    uint16_t3 floored_pos = (uint16_t3)floor(modified_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = false;

    [loop]
    for (int i = 0; i < N; i++) {
        if (any(floored_pos < 0 | floored_pos >= (SIZE / (1 << MAX(K, 0))))) {
            return false;
        }

        bool test = fetcher.fetch_raw((int3)(floored_pos), MAX(K, 0)) != 0;

        if (test) {
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * (1 << MAX(K, 0)));
            distance += local_dist  - 0.1;
            return true;
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, (half3)0);
        floored_pos += select(eqs, dir_sign, (int16_t3)0);
    }

    float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * (1 << MAX(K, 0)));
    distance += local_dist - 0.1;


    return true;
}


DdaTraversalOutput dda_near_high_quality<let N : int>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    inout float base_distance,
) {
    float tmp = base_distance;
    float3 modified_pos = (ray_pos + ray_dir * base_distance);
    uint16_t3 floored_pos = (uint16_t3)floor(modified_pos);
    float3 side_dist = ((dir_sign * (floored_pos - modified_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = false;

    [loop]
    for (int i = 0; i < N; i++) {
        if (any(floored_pos < 0 | floored_pos >= (SIZE))) {
            return DdaTraversalOutput(false);
        }

        uint64_t raw = fetcher.fetch_raw((int3)(floored_pos), 0);

        if (raw != 0) {
            base_distance = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir)) + tmp  - 0.1;
            
            uint3 brick_pos = 0;
            uint8_t face = 0;
            if (dda_bitset_simple(raw, ray_pos, ray_dir, base_distance, floored_pos, inv_dir, dir_sign, brick_pos, face)) {
                return DdaTraversalOutput(true, ray_pos + ray_dir * base_distance, brick_pos, face);
            }
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, inv_dir, (half3)0);
        floored_pos += select(eqs, dir_sign, (int16_t3)0);
    }

    base_distance = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir)) + tmp  - 0.1;

    return DdaTraversalOutput(false);
}

bool dda<let N : int>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    out float dist
) {
    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
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
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir)) - 0.05;
            
            uint3 brick_pos = 0;
            uint8_t face = 0;
            if (dda_bitset_simple(raw, ray_pos, ray_dir, local_dist, floored_pos, inv_dir, dir_sign, brick_pos, face)) {
                dist = local_dist;
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

int8_t3 get_face_normal(int face, int8_t3 sign) {
    return (int8_t3)copysign(-(int8_t3)(face == int3(0, 1, 2)), sign);
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