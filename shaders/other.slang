#ifndef OTHER
#define OTHER
#include <hash.slang>
#include <queue.slang>
#include <stack.slang>
#include <octree.slang>
#include <voxel.slang>

static const int MIP_LEVELS = 8;
static const int SIZE = 1 << (MIP_LEVELS-1);
static const uint INVALID = 0x3FFFFFF;


struct Fetcher {
    Texture3D<uint8_t> voxels;

    Voxel fetch(int3 position) {
        uint8_t raw = 0;

        if (all(position >= 0) && all(position < SIZE)) {
            raw = voxels.Load(int4(position, 0));
        }
        
        return Voxel.from_raw(raw);
    }

    static const uint8_t1 zero = (uint8_t)0;
    uint8_t fetch_raw(uint8_t3 position, uint8_t1 level = zero) {
        return voxels.Load(uint8_t4(position, level));
        /*
        uint8_t raw = 0;s

        if (all(position >= 0) && all(position < (SIZE >> level))) {
            raw = voxels.Load(int4(position, level));
        }
        
        return raw;
        */
    }
}


/*
static const int LEVEL_DIFF = 1;
static const int DDA_STEPS = 8;
*/

struct Leaf {
    int parent_id;
    int num_bodies;
    int child_start;
    int bounds_stuff;
}



// root
//   |-  leaf 1 
//   |-  leaf 2
//   |-  leaf 3 
//   |-  leaf 4 
//   |-  leaf 5
//   |-  ... 

static const uint8_t LEVEL_DIFF = 4;
static const uint8_t DDA_STEPS = 24;

public struct MultiLevelThingyOutput<bool hasFace, bool hasWorld, bool hasBlockPos>  {
    public Conditional<int, hasFace> face;
    public Conditional<float3, hasWorld> world;
    public Conditional<uint3, hasBlockPos> block_pos;
}

public struct TraceStats {
    public uint8_t1 total_iterations;
    public uint8_t1 recursions;
    public uint8_t1 full_halts;
    public uint8_t1 total_iterations_at_full_halt;
    public uint8_t1 last_level_raw_dd;
}

// this function calls itself recursively
// for some reason this performs decently compared to a queue 
// probably cause of scratch buffer stuff, even though the asm code for the queue impl was much shorter than this recursive version
// it makes sense though, octree are inherently recursive structures. you must traverse them recursively. traversing them using a queue is not a good idea 
bool dda_multi_level<let K : int, let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    int3 box_min_bounds_world_space,
    int3 box_max_bounds_world_space,
    float3 inv_dir,
    float3 dir_sign,
    float distance,
    inout TraceStats stats,
    out MultiLevelThingyOutput<hasFace, hasWorld, hasBlockPos> output,
) {
    int voxel_size = 1 << max(K, 0);

    int3 min_bounds_local_space = box_min_bounds_world_space / voxel_size;
    int3 max_bounds_local_space = box_max_bounds_world_space / voxel_size;


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

        // skip if we are outside the bounds of the parent
        if (any((int3)(floored_pos) < (min_bounds_local_space - 1)) || any((int3)(floored_pos) >= (max_bounds_local_space + 1))) {
            break;
        }
        
        uint8_t raw = fetcher.fetch_raw((int3)(floored_pos), max(K, 0));

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
        float3 reconst = side_dist * inv_dir;
        bool3 t = min3(reconst.x, reconst.y, reconst.z) == reconst;
        int3 eqs = select(t, 1, 0);
        face_local = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        floored_pos += select(t, dir_sign, 0);
        side_dist += select(t, dir_sign, 0);
    }
    

    return false;
}

const static int STACK_SIZE = 32;

groupshared PackedNode[16*16*STACK_SIZE] stacks;


//groupshared Stack<DDA_STEPS, PackedNode>[16*16] reverse_stacks;

// this will traverse using DDA through a specific node
bool dda_stackful<let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool>(
    Fetcher fetcher,
    half3 ray_dir,
    float3 ray_pos,
    half3 inv_dir,
    half3 dir_sign,
    uint8_t stack_id,
    Node popped,
    inout uint8_t stack_length,
    inout TraceStats stats,
    out half closest_distance,
) {
    uint8_t depth_level = popped.depth_level;
    uint8_t voxel_size = (uint8_t)SIZE >> depth_level;

    uint8_t global_distance = popped.distance * voxel_size;
    float3 modified_ray_pos = (ray_pos + ray_dir * max(global_distance + 0.04, 0)) / voxel_size;

    float3 floored_pos = floor((float3)modified_ray_pos);
    float3 side_dist = (floored_pos - modified_ray_pos + 0.5 + copysign(0.5, dir_sign));

    // stack used to reverse the ordering of the visited nodes, since we need to visit them from closest to furthest
    Stack<12, uint8_t> local_reverse_stack = Stack<12, uint8_t>();
    uint8_t local_stack_length = 0;
    half dist = 0;

    [loop]
    for (uint8_t i = 0; i < DDA_STEPS; i++) {
        // skip if we are outside the bounds of the map
        if (any(floored_pos < 0) || any(floored_pos >= (SIZE / (voxel_size)))) {
            break;
        }
        
        /*
        // skip if we are outside the bounds of the parent
        if (any((int3)(floored_pos) < (min_bounds_local_space - 1)) || any((int3)(floored_pos) >= (max_bounds_local_space + 1))) {
            break;
        }
        */
        uint8_t raw = fetcher.fetch_raw((uint8_t3)(floored_pos), (uint8_t1)(MIP_LEVELS - 1) - depth_level);

        // handle partially full nodes / leaf voxel nodes
        if (raw != 0) {

            // check if we reached a leaf node
            if (depth_level >= MIP_LEVELS - 1) {
                // set output parameter
                closest_distance = (half)dist - 1;
                return true;
            } else {
                local_reverse_stack.append((uint8_t1)floor(max(dist - 1 - 0.5, 0)));
            }

            // we know we can't trace *after* completely full nodes, so we can stop here
            if ((raw & 4) == 4) {
                stats.total_iterations_at_full_halt = stats.total_iterations;
                stats.full_halts++;
                break;
            }
        }

        stats.total_iterations++;
        half3 reconst = side_dist * inv_dir;
        bool3 t = min3(reconst.x, reconst.y, reconst.z) == reconst;
        floored_pos += select(t, dir_sign, half3.dzero());
        side_dist += select(t, dir_sign, half3.dzero());
    }

    uint8_t t;
    
    [loop]
    while (local_reverse_stack.tryDequeue(t)) {
        PackedNode n = Node(min(depth_level + LEVEL_DIFF, (uint8_t)MIP_LEVELS - 1), 0).into_raw();
        n.y = t;

        if (stack_length >= STACK_SIZE) {
            printf("stack too small!");
            return false;
        }

        stacks[stack_id * STACK_SIZE + stack_length] = n;
        stack_length += 1;
    }
    
    return false;
}

bool dda_traverse<let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool>(   
    Fetcher fetcher,
    float3 ray_dir,
    uint3 lid,
    float3 ray_pos,
    out MultiLevelThingyOutput<hasFace, hasWorld, hasBlockPos> output,
    out TraceStats stats,
) {
    stats = TraceStats(0, 0, 0);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    return dda_multi_level<MIP_LEVELS - 1, hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, 0, SIZE, inv_dir, dir_sign, 0, stats, output);
}

bool dda_traverse2<let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool>(   
    Fetcher fetcher,
    half3 ray_dir,
    uint3 lid,
    float3 ray_pos,
    //out MultiLevelThingyOutput<hasFace, hasWorld, hasBlockPos> output,
    out TraceStats stats,
) {
    stats = TraceStats(0, 0, 0);
    half3 inv_dir = 1 / ray_dir;
    half3 dir_sign = (half3)sign(ray_dir);

    uint8_t stack_id = (uint8_t)lid.x + (uint8_t)lid.y * 16;

    stacks[stack_id * STACK_SIZE] = Node(3,0).into_raw();
    uint8_t stack_length = 1;


    //return dda_stackful<hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, stats, output);
    //Stack<STACK_SIZE, PackedNode> stack = Stack<STACK_SIZE, PackedNode>();
    //Stack<STACK_SIZE, PackedNode> stack = wtf[stack_id];
    //Stack<STACK_SIZE, PackedNode> stack = Stack<STACK_SIZE, PackedNode>();
    //stack.length = 0;
    //stack.append(Node(0,0).into_raw());

    half closest_distance = 0;

    PackedNode raw = Node(2,0).into_raw();
    [loop]
    while (stack_length > 0) {
        stack_length -= 1;
        PackedNode raw = stacks[stack_id * STACK_SIZE + stack_length];


        
        stats.recursions++;

        /*
        Node popped = Node.from_raw(raw);
        if (popped.depth_level <= MIP_LEVELS - 2) {
            if (dda_stackful<hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, stack_id, popped, stack, stats, output)) {
                return true;
            }
        } else if (dda<18, hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos + ray_dir * popped.distance, output, stats)) {
            return true;
        }
        */

        Node popped = Node.from_raw(raw);

        if (dda_stackful<hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, inv_dir, dir_sign, stack_id, popped, stack_length, stats, closest_distance)) {
            //output.world = ray_pos + ray_dir * closest_distance;
            return true;
        }
    }

    return false;
    //return dda_multi_level<MIP_LEVELS - 1, hasFace, hasWorld, hasBlockPos>(fetcher, ray_dir, ray_pos, 0, SIZE, inv_dir, dir_sign, 0, stats, output);
    
}

/*
bool dda_with_face<let N : int>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    out float3 world,
    out uint3 block_pos,
    out float3 block_uvs,
    out int face,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);


    for (int i = 0; i < N; i++) {
        Voxel voxel = fetcher.fetch((int3)(floored_pos));

        if (voxel.active) {
            float3 test = (floored_pos - ray_pos + 0.5 - 0.5 * dir_sign) * inv_dir;
            float max = max3(test.x, test.y, test.z);
            world = ray_pos + ray_dir * max;
            face = face_local;
            block_uvs = world - floored_pos;
            block_pos = (uint3)floored_pos;
            return true;
        }

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        face_local = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    return false;
}
*/

bool dda<let N : int, let hasFace : bool, let hasWorld : bool, let hasBlockPos: bool>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    out MultiLevelThingyOutput<hasFace, hasWorld, hasBlockPos> output,
    inout TraceStats stats,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);
    int face_local = 0;

    [loop]
    for (int i = 0; i < N; i++) {
        Voxel voxel = fetcher.fetch((int3)(floored_pos));

        if (voxel.active) {
            float3 test = (floored_pos - ray_pos + 0.5 - 0.5 * dir_sign) * inv_dir;
            float max = max3(test.x, test.y, test.z);
            output.world = ray_pos + ray_dir * max;
            output.face = face_local;
            //output.block_uvs = output.world - floored_pos;
            //output.block_pos = (uint3)floored_pos;
            return true;
        }

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        face_local = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
        stats.total_iterations += 1;
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