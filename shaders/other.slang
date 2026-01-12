#ifndef OTHER
#define OTHER
#include <hash.slang>

static const int MIP_LEVELS = 7;
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

struct Voxel {
    bool active;
    bool reflective;
    bool refractive;
    bool placed;

    uint8_t into_raw() {
        uint8_t raw = 0;
        raw |= active ? 1 : 0;
        raw |= reflective ? 2 : 0;
        raw |= refractive ? 4 : 0;
        raw |= placed ? 8 : 0;
        return raw;
    }

    static Voxel from_raw(uint8_t raw) {
        Voxel voxel;
        voxel.active = (raw & 1) == 1;
        voxel.reflective = ((raw >> 1) & 1) == 1;
        voxel.refractive = ((raw >> 2) & 1) == 1;
        voxel.placed = ((raw >> 3) & 1) == 1;
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

        if (all(position >= 0) && all(position < SIZE)) {
            raw = voxels.Load(int4(position, level));
        }
        
        return raw;
    }
}

struct Stack<let N: int, T> {
    T[N] elements;
    float[N] distances;
    uint length;

    __init() {
        length = 0;
    }

    // https://en.wikipedia.org/wiki/Insertion_sort
    [mutating]
    void sort() {
        int i = 1;
        while (i < length) {
            int j = i;

            while (j > 0 && distances[j-1] < distances[j]) {
                swap(j, j-1);
                j--;
            }

            i++;
        }
    }

    [mutating]
    void swap(int a, int b) {
        float tmp_dist = distances[a];
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

        /*
        // do swaps until it is in the right spot...
        int cur_index = length;
        while (cur_index > 1 && distances[cur_index - 1] < distances[cur_index]) {

        }
        */

        length++;
    }

    [mutating]
    bool tryDequeue(out T element) {
        if (length == 0) {
            return false;
        }

        element = elements[length-1];
        length--;
        return true;
    }
}

// https://tavianator.com/2022/ray_box_boundary.html
bool ray_box_intersection(
    float3 ray_dir,
    float3 ray_pos,
    float3 aabb_min,
    float3 aabb_max,
    out float _tmin,
) {
    float3 inv_dir = 1 / ray_dir;
    float tmin = 0.0, tmax = 999999;

    for (int d = 0; d < 3; ++d) {
        float t1 = (aabb_min[d] - ray_pos[d]) * inv_dir[d];
        float t2 = (aabb_max[d] - ray_pos[d]) * inv_dir[d];

        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }

    _tmin = tmin;

    return tmin < tmax;
}

struct Node {
    uint3 position;
    uint depth_level;
    float tmin;
    uint8_t1 value;
}

float4 traverse(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
) {
    //return ray_box_intersection(ray_dir, ray_pos, -3, 3) ? 1 : 0;
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    Stack<16, Node> stack;
    stack.append(Node(0, 0, 0, 64), 0); // add the root node 

    Node node;
    int iterations = 0;
    bool hit = false;
    while (stack.tryDequeue(node)) {
        uint8_t1 fetched = node.value;
        uint depth = node.depth_level;
        uint3 pos = node.position;
        
        // check if this node is a parent and it has children... 
        if ((fetched & 64) == 64 && depth != (MIP_LEVELS - 1)) {
            // add the children to the stack
            for (int c = 0; c < 8; c++) {
                uint3 child_position = pos * 2 + CHILDREN_OFFSETS[c];
                int child_level = depth + 1;
                uint4 packed = uint4(child_position, child_level);

                int mip_scale = (1 << packed.w);
                float3 aabb_size = SIZE / mip_scale;
                float3 aabb_min = packed.xyz * aabb_size;
                float3 aabb_max = packed.xyz * aabb_size + aabb_size; 


                // child must intersect ray...
                float local_tmin = 0;
                if (ray_box_intersection(ray_dir, ray_pos, aabb_min, aabb_max, local_tmin)) {
                    // if empty, skip...
                    uint8_t1 child_fetched = fetcher.fetch_raw(packed.xyz, MIP_LEVELS - packed.w - 1);
                    if (child_fetched != 0) {
                        stack.append(Node(child_position, child_level, local_tmin, child_fetched), local_tmin);
                    }
                }
            }

            stack.sort();
        }

        // check if this node is a voxel leaf node...
        if ((fetched & 1) == 1) {
            hit = true;
            break;
        }

        iterations++;
    }

    return iterations / 32.0;
    //return hit ? 1.0 : 0.0;
    //return iterations / 4.0;
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