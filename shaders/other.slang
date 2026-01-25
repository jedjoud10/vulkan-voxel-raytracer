#ifndef OTHER
#define OTHER
#include <hash.slang>
#include <queue.slang>
#include <stack.slang>
#include <octree.slang>
#include <voxel.slang>
#include <debug_util.slang>

static const int MIP_LEVELS = 7;
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
static const uint8_t LEVEL_DIFF = 3;
static const uint8_t DDA_STEPS = 22;

public struct MultiLevelThingyOutput<bool hasFace, bool hasWorld, bool hasBlockPos>  {
    public Conditional<int, hasFace> face;
    public Conditional<float3, hasWorld> world;
    public Conditional<uint3, hasBlockPos> block_pos;
}

public struct TraceStats {
    public uint8_t1 total_iterations;
    public uint8_t1 recursions;
    public uint8_t1 bottom_level_iterations;
    public float data;
}

// cube: 6 quads
// each quad: 4x4 segments, total of 16

// ray intersection:
// input quad: id in [0..6], represented by 3 bits
// output quad: id in [0..6], represented by 3 bits
// input quad segment: 4 bits
// output quad segment: 4 bits

#define MAX(a, b)  (((a) > (b)) ? (a) : (b)) 

bool dda_bitset<let tracing: bool>(
    uint64_t raw,
    half3 ray_dir,
    float3 ray_pos,
    float base_distance,
    int16_t3 block_pos,
    half3 inv_dir,
    int8_t3 dir_sign,
    out float distance,
    inout Conditional<TraceStats, tracing> stats,
) {
    distance = 0.0;

    if (raw == 0) {
        return false;
    }

    float3 modified_ray_pos = (ray_pos + ray_dir * base_distance) * 4.0;
    int3 floored_pos = (int3)floor(modified_ray_pos);
    half3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = false;

    [loop]
    for (uint8_t i = 0; i < 12; i++) {
        if (any((floored_pos / 4) != block_pos)) {
            break;
        }
        
        int3 local = floored_pos % 4;
        int bit_index = local.z + local.y * 4 + local.x * 4 * 4;
        if (bit_index >= 0 && bit_index < 64 && ((raw >> bit_index) & 1) == 1) {
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) / 4.0);
            distance = base_distance + local_dist;
            /*
            output.world.set(ray_pos + ray_dir * distance);
            output.block_pos.set((uint3)floored_pos);

            bool3 test_eqs = select(i == 0, parent_eqs, eqs);
            output.face = firstbithigh(test_eqs.x | test_eqs.y << 1 | test_eqs.z << 2);
            */
            return true;
        }

        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, (half3)1, (half3)0) * inv_dir;
        floored_pos += select(eqs, 1, 0) * dir_sign;

        if (tracing) {
            TraceStats _stats = stats.get().value;
            _stats.total_iterations++;
            _stats.bottom_level_iterations++;
            stats.set(_stats);
        }
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
    out float output,
    inout Conditional<TraceStats, tracing> stats,
) {
    float3 modified_ray_pos = (ray_pos + ray_dir * (base_distance - 0.01)) / (1 << MAX(K, 0));
    int16_t3 floored_pos = (int16_t3)floor(modified_ray_pos);
    half3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
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
        
        uint64_t raw = fetcher.fetch_raw((uint8_t3)(floored_pos), MAX(K, 0));
        if (raw != 0) {
            // handle partially full nodes / leaf voxel nodes
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) * (1 << MAX(K, 0)));
            float distance = base_distance + local_dist;

            // completely full nodes are well, completely full. we can do the leaf-level logic since we know we can skip them
            if (K > 0) {
                if (tracing) {
                    TraceStats _stats = stats.get().value;
                    _stats.recursions++;
                    stats.set(_stats);
                }

                const uint8_t SOME_THING = (uint8_t)(K - (MAX(K-LEVEL_DIFF, 0)));
                uint8_t3 next_min_bounds_local_space = ((uint8_t3)floored_pos << SOME_THING);
                uint8_t3 next_max_bounds_local_space = (uint8_t3)(floored_pos+1) << SOME_THING;
            
                uint16_t3 next_bounds_local_space = (uint16_t3)next_min_bounds_local_space | ((uint16_t3)next_max_bounds_local_space << 8);

                if (dda_multi_level<K-LEVEL_DIFF, hasFace, hasWorld, hasBlockPos, tracing>(fetcher, ray_dir, ray_pos, lid_index, next_bounds_local_space, inv_dir, dir_sign, max(distance - 0.001, 0), output, stats)) {
                    return true;
                }
            } else {
                if (dda_bitset(raw, ray_dir, ray_pos, (max(distance - 0.001, 0)), floored_pos, inv_dir, dir_sign, output, stats)) {
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
    out MultiLevelThingyOutput<hasFace, hasWorld, hasBlockPos> output,
    out Conditional<TraceStats, tracing> stats,
) {
    stats.set(TraceStats(0, 0, 0));

    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    uint lid_index = lid.x + lid.y * 16;
    float distance = 0.0;
    bool hit = dda_multi_level<STARTING_LEVEL, hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, lid_index, (uint16_t3(SIZE) << 8), inv_dir, dir_sign, 0, distance, stats);

    float3 point_pos = (ray_pos + ray_dir * (distance - 0.05)) * 4.0;
    int16_t3 floored_pos = (int16_t3)floor(point_pos);
    float3 side_dist = ((dir_sign * (floored_pos - point_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    bool3 eqs = false;

    [loop]
    for (int i = 0; i < 4; i++) {
        uint64_t raw = fetcher.fetch_raw(floored_pos / 4, 0);
        int3 local = floored_pos % 4;
        int bit_index = local.z + local.y * 4 + local.x * 4 * 4;
        if (bit_index >= 0 && bit_index < 64 && ((raw >> bit_index) & 1) == 1) {
            float local_dist = length(select(eqs, 1.0, 0.0) * (side_dist - inv_dir) / 4.0);
            distance = local_dist;

            output.world.set(point_pos + ray_dir * distance);
            output.block_pos.set((uint3)floored_pos);    
            output.face = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
            return true;
        }


        eqs = min3(side_dist.x, side_dist.y, side_dist.z) == side_dist;
        side_dist += select(eqs, (half3)1, (half3)0) * inv_dir;
        floored_pos += select(eqs, 1, 0) * dir_sign;
    }
    return hit;
}

bool dda<let N : int, let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    out MultiLevelThingyOutput<hasFace, hasWorld, hasBlockPos> output,
    inout TraceStats stats,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / abs(ray_dir);
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = ((dir_sign * (floored_pos - ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    //int face_local = 0;
    int3 eqs = 0;
    //float distance = 0;

    [loop]
    for (int i = 0; i < N; i++) {
        Voxel voxel = fetcher.fetch((int3)(floored_pos));

        if (voxel.active) {
            break;
        }

        // creates an int3 for the smallest value in side_dist
        eqs = select(min3(side_dist.x, side_dist.y, side_dist.z) == side_dist, 1, 0);
        side_dist += eqs * inv_dir;
        floored_pos += eqs * dir_sign;

        //distance += min3(reconst.x, reconst.y, reconst.z);
        //face_local = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        stats.total_iterations += 1;

    }

    float d = length((eqs) * (side_dist - inv_dir));
    stats.data = d;
    output.world = ray_pos + ray_dir * d;
    output.block_pos = (uint3)floored_pos;
    
    //output.face = face_local;


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