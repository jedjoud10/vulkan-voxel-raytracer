#ifndef OTHER
#define OTHER
#include <hash.slang>

static const int MIP_LEVELS = 9;
static const int SIZE = 1 << (MIP_LEVELS-1);
static const uint INVALID = 0x3FFFFFF;


public static const int3[] CHILDREN_OFFSETS = {
    int3(0, 0, 0),
    int3(1, 0, 0),
    int3(0, 0, 1),
    int3(1, 0, 1),
    int3(0, 1, 0),
    int3(1, 1, 0),
    int3(0, 1, 1),
    int3(1, 1, 1),
};

enum Type: uint8_t {
    Grass,
    Rock,
}

struct Voxel {
    bool active;
    bool reflective;
    bool refractive;
    bool placed;
    Type type;

    uint8_t into_raw() {
        uint8_t raw = 0;
        raw |= active ? 1 : 0;
        raw |= reflective ? 2 : 0;
        raw |= refractive ? 4 : 0;
        raw |= placed ? 8 : 0;
        //raw |= (uint8_t)type << 4;
        return raw;
    }

    static Voxel from_raw(uint8_t raw) {
        Voxel voxel;
        voxel.active = (raw & 1) == 1;
        voxel.reflective = ((raw >> 1) & 1) == 1;
        voxel.refractive = ((raw >> 2) & 1) == 1;
        voxel.placed = ((raw >> 3) & 1) == 1;
        //voxel.type = (Type)((raw >> 8) & 0xF);
        return voxel;
    }
}

struct Fetcher {
    Texture3D<uint8_t> voxels;

    Voxel fetch(int3 position) {
        uint8_t raw = 0;

        if (all(position >= 0) && all(position < SIZE)) {
            raw = voxels.Load(int4(position, 0));
        }
        
        return Voxel.from_raw(raw);
    }

    uint8_t fetch_raw(int3 position, int level = 0) {
        uint8_t raw = 0;

        if (all(position >= 0) && all(position < (SIZE >> level))) {
            raw = voxels.Load(int4(position, level));
        }
        
        return raw;
    }
}

struct Stack<let N: int, T> {
    T[N] elements;
    half[N] distances;
    uint length;

    __init() {
        length = 0;
    }

    [mutating]
    void swap(int a, int b) {
        half tmp_dist = distances[a];
        T tmp_packed = elements[a];

        distances[a] = distances[b];
        elements[a] = elements[b]; 
        
        distances[b] = tmp_dist;
        elements[b] = tmp_packed; 
    }

    [mutating]
    void append(T element, float tmin) {
        if (length == N) {
            //printf("cannot append to stack. stack full. length=%i", length);
            return;
        }

        distances[length] = tmin;
        elements[length] = element;

        // do swaps until it is in the right spot...
        int cur_index = length;
        while (cur_index >= 1 && distances[cur_index - 1] < distances[cur_index]) {
            swap(cur_index - 1, cur_index);
            cur_index = cur_index - 1;
        }

        length++;
    }

    [mutating]
    bool tryDequeue(out T element, out float tmin) {
        if (length == 0) {
            return false;
        }

        element = elements[length-1];
        tmin = distances[length-1];
        length--;
        return true;
    }
}

// https://tavianator.com/2022/ray_box_boundary.html
bool ray_box_intersection(
    float3 inv_dir,
    float3 ray_pos,
    float3 aabb_min,
    float3 aabb_max,
    out float _tmin,
) {
    float tmin = 0.0, tmax = 999999;

    float3 t1 = (aabb_min - ray_pos) * inv_dir;
    float3 t2 = (aabb_max - ray_pos) * inv_dir;

    float3 _min = min(t1, t2);
    float3 _max = max(t1, t2);

    tmin = max(max3(_min.x, _min.y, _min.z), tmin);
    tmax = min(min3(_max.x, _max.y, _max.z), tmax);

    _tmin = tmin;

    return tmin < tmax;
}

struct Node {
    uint3 position;
    uint depth_level; // we only need a few bits for this, no more than 3-4
    uint8_t1 value; // we only use 1 bit of this

    uint into_raw() {
        // pack the position in the first 3 bytes
        // which means that max map/chunk size IS limited to 256. oh well
        // needs chunking!
        uint packed = position.x & 0xFF | ((position.y & 0xFF) << 8) | ((position.z & 0xFF) << 16);

        // pack the depth level as well. 6 bits
        packed |= (depth_level & 0b111111) << 24;

        // two bits that tells us if the node is a node contains matter 
        // first bit is used for leaf voxels (ones at level 0) to see if they are "active" or not
        // second bit is the "propagated" bit. gets set whenever a parent node has valid children that we must traverse through as well.
        packed |= ((uint)value & 0b11) << 30;

        return packed;
    }

    static Node from_raw(uint raw) {
        Node node;
        node.position = uint3(raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF);
        node.depth_level = (raw >> 24) & 0b111111;
        node.value = (uint8_t1)((raw >> 30) & 0b11);
        return node;
    }
}

bool traverse(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    out float3 position,
    out uint3 block_pos,
    out float3 block_uvs,
    out int face,
    out int iterations,
) {
    iterations = 0;
    if (dda_with_face<8>(fetcher, ray_dir, ray_pos, position, block_pos, block_uvs, face)) {
        return true;
    } else {
        ray_pos += ray_dir * sqrt(8);
    }

    //return ray_box_intersection(ray_dir, ray_pos, -3, 3) ? 1 : 0;
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    Stack<12, uint> stack;
    stack.append(Node(0, 0, 2).into_raw(), 0); // add the root node 

    static const int HYBRID_LVL = 6; 
    uint raw_node;
    Node node;
    float node_tmin;
    while (stack.tryDequeue(raw_node, node_tmin)) {
        node = Node.from_raw(raw_node);
        uint8_t1 fetched = node.value;
        uint depth = node.depth_level;
        uint3 pos = node.position;
        
        // check if this node is a parent and it has children... 
        if ((fetched & 2) == 2 && depth < HYBRID_LVL) {
            // add the children to the stack
            for (int c = 0; c < 8; c++) {
                uint3 child_position = pos * 2 + CHILDREN_OFFSETS[c];
                int child_level = depth + 1;
                uint4 packed = uint4(child_position, child_level);

                int mip_scale = (1 << packed.w);
                float3 aabb_size = SIZE / mip_scale;
                float3 aabb_min = packed.xyz * aabb_size;
                float3 aabb_max = packed.xyz * aabb_size + aabb_size; 

                // if empty, skip...
                uint8_t1 child_fetched = fetcher.fetch_raw(packed.xyz, MIP_LEVELS - packed.w - 1);
                if (child_fetched != 0) {
                    // child must intersect ray...
                    float local_tmin = 0;
                    if (ray_box_intersection(inv_dir, ray_pos, aabb_min, aabb_max, local_tmin)) {
                        stack.append(Node(child_position, child_level, child_fetched).into_raw(), local_tmin);
                    }
                }
            }
        }

        // check if this node is a voxel leaf node...
        if (depth >= HYBRID_LVL) {
            position = ray_pos + (node_tmin - 0.1) * ray_dir;
            if (dda_with_face<1 << (MIP_LEVELS - HYBRID_LVL + 1)>(fetcher, ray_dir, ray_pos + (node_tmin - 0.1) * ray_dir, position, block_pos, block_uvs, face)) {
                return true;
            }
        }

        /*
        // check if this node is a voxel leaf node...
        if ((fetched & 1) == 1) {
            float tmp = 0;
            ray_box_intersection(inv_dir, ray_pos, pos, pos + 1, tmp);
            position = ray_pos + (tmp + 0.001) * ray_dir;
            dda_with_face<1>(fetcher, ray_dir, ray_pos + (tmp - 0.01) * ray_dir, face);
            return true;
        }
        */

        iterations++;
    }

    return false;
}

#include <queue.slang>

static const int LEVEL_DIFF = 3;
static const int DDA_STEPS = 22;

public struct MultiLevelThingyOutput {
    public int face;
    public float3 world;
    public float3 block_uvs;
    public uint3 block_pos;
}

// this function calls itself recursively
// for some reason this performs decently compared to a queue 
// probably cause of scratch buffer stuff, even though the asm code for the queue impl was much shorter than this recursive version
// it makes sense though, octree are inherently recursive structures. you must traverse them recursively. traversing them using a queue is not a good idea 
bool dda_multi_level<let N : int, let K : int>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    int3 box_min_bounds_world_space,
    int3 box_max_bounds_world_space,
    float distance,
    inout int iterations,
    out MultiLevelThingyOutput output,
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
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - modified_ray_pos + 0.5 + 0.5 * dir_sign);

    int face_local = 0;
    for (int i = 0; i < N; i++) {
        if (any((int3)(floored_pos) < 0) || any((int3)(floored_pos) >= (SIZE >> max(K, 0)))) {
            break;
        }

        if (any((int3)(floored_pos) < (min_bounds_local_space - 1)) || any((int3)(floored_pos) > (max_bounds_local_space + 1))) {
            break;
        }
        
        uint8_t raw = fetcher.fetch_raw((int3)(floored_pos), max(K, 0));

        // handle partially full nodes / leaf voxel nodes
        if (raw != 0) {
            float3 test = (floored_pos * voxel_size - ray_pos + voxel_size * 0.5 - 0.5 * voxel_size * dir_sign) * inv_dir;
            float dist = max3(test.x, test.y, test.z);
            if (K > 0) {
                iterations++;

                int3 next_min_bounds_world_space = (int3)floored_pos * voxel_size;
                int3 next_max_bounds_world_space = (int3)floored_pos * voxel_size + voxel_size;

                if (dda_multi_level<N, K-LEVEL_DIFF>(fetcher, ray_dir, ray_pos, next_min_bounds_world_space, next_max_bounds_world_space, max(dist, 0), iterations, output)) {
                    return true;
                }
            } else {
                // set output parameters
                output.world = ray_pos + ray_dir * dist;
                output.block_uvs = output.world - floored_pos;
                output.face = face_local;
                output.block_pos = (uint3)floored_pos;
                return true;
            }
        }

        // skip completely full nodes
        if ((raw & 4) == 4) {
            break;
        }

        iterations++;
        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        face_local = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }
    

    return false;
}

bool dda_traverse(   
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    out MultiLevelThingyOutput output,
) {
    /*
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    */
    int iterations = 0;
    return dda_multi_level<DDA_STEPS, MIP_LEVELS - 1>(fetcher, ray_dir, ray_pos, 0, SIZE, 0, iterations, output);
}

int traverse2(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
) {
    //return ray_box_intersection(ray_dir, ray_pos, -3, 3) ? 1 : 0;
    float3 inv_dir = 1 / ray_dir;
    Stack<2, uint> stack;
    stack.append(Node(0, 0, 2).into_raw(), 0); // add the root node 

    uint raw_node;
    Node node;
    float node_tmin;
    int iterations = 0;
    while (stack.tryDequeue(raw_node, node_tmin)) {
        node = Node.from_raw(raw_node);
        uint8_t1 fetched = node.value;
        uint depth = node.depth_level;
        uint3 pos = node.position;
        
        // check if this node is a parent and it has children... 
        if ((fetched & 2) == 2 && depth != (MIP_LEVELS - 1)) {
            int child_level = depth + 1;
            int mip_scale = (1 << child_level);
            int aabb_size = SIZE >> child_level;

            // add the children to the stack
            for (int c = 0; c < 8; c++) {
                uint3 child_position = pos * 2 + CHILDREN_OFFSETS[c];
                uint4 packed = uint4(child_position, child_level);

                float3 aabb_min = packed.xyz * aabb_size;
                float3 aabb_max = packed.xyz * aabb_size + aabb_size; 

                // if empty, skip...
                uint8_t1 child_fetched = fetcher.fetch_raw(packed.xyz, MIP_LEVELS - packed.w - 1);
                if (child_fetched != 0) {
                    // child must intersect ray...
                    float local_tmin = 0;
                    if (ray_box_intersection(inv_dir, ray_pos, aabb_min, aabb_max, local_tmin)) {
                        stack.append(Node(child_position, child_level, child_fetched).into_raw(), local_tmin);
                    }
                }
            }
        }

        // check if this node is a voxel leaf node...
        if ((fetched & 1) == 1) {
            return iterations;
        }

        iterations++;
    }

    return iterations;
}

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

    int face_local = 0;
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


bool dda<let N : int>(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);

    for (int i = 0; i < N; i++) {
        Voxel voxel = fetcher.fetch((int3)(floored_pos));

        if (voxel.active) {
            return true;
        }

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
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

    for (int i = 0; i < 64; i++) {
        Voxel voxel = fetcher.fetch((int3)(floored_pos));

        if (voxel.active && voxel.refractive) {
            color *= per_block_unique_colour((uint3)floor(floored_pos));
        }

        if (voxel.active && !voxel.refractive) {
            return 0.0;
        }

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    return color;
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