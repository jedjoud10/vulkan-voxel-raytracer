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

struct Queue<let N: int, T> {
    T[N] elements;
    uint readHead;
    uint writeHead;
    uint length;

    __init() {
        length = 0;
        writeHead = 0;
        readHead = 0;
    }

    [mutating]
    void append(T element) {
        if (length == N) {
            printf("cannot append to queue. queue full. length=%i, wh=%i, rh=%i", length, writeHead, readHead);
            return;
        }

        elements[writeHead] = element;
        writeHead++;
        length++;
        writeHead %= N;
    }

    [mutating]
    bool tryDequeue(out T element) {
        if (readHead == writeHead) {
            return false;
        }

        element = elements[readHead];
        readHead++; 
        readHead %= N; 
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
) {
    float3 inv_dir = 1 / ray_dir;
    float tmin = 0.0, tmax = 999999;

    for (int d = 0; d < 3; ++d) {
        float t1 = (aabb_min[d] - ray_pos[d]) * inv_dir[d];
        float t2 = (aabb_max[d] - ray_pos[d]) * inv_dir[d];

        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }

    return tmin < tmax;
}

float4 traverse(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
) {
    //return ray_box_intersection(ray_dir, ray_pos, -3, 3) ? 1 : 0;
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    Queue<16, uint4> queue;
    queue.append(uint4(0, 0, 0, 0)); // add the root node 

    uint4 pos;
    int iterations = 0;
    bool hit = false;
    while (queue.tryDequeue(pos)) {
        uint8_t1 fetched = fetcher.fetch_raw(pos.xyz, MIP_LEVELS - pos.w - 1);
        int mip_scale = (1 << pos.w);
        float3 aabb_size = SIZE / mip_scale;
        float3 aabb_min = pos.xyz * aabb_size;
        float3 aabb_max = pos.xyz * aabb_size + aabb_size; 
        
        if (iterations == 2) {
            //printf("%i %i %i %f %f", pos.x, pos.y, pos.z, aabb_min.x, aabb_max.x);
        }

        // check if we intersect with this node
        if (ray_box_intersection(ray_dir, ray_pos, aabb_min, aabb_max)) {
            // check if this node is a parent and it has children... 
            if ((fetched & 64) == 64 && pos.w != (MIP_LEVELS - 1)) {
                // add the children to the queue
                for (int c = 0; c < 8; c++) {
                    int3 child_position = pos.xyz * 2 + CHILDREN_OFFSETS[c];
                    int child_level = pos.w + 1;
                    uint4 packed = uint4(child_position, child_level);

                    
                    // if empty, skip...
                    if (fetcher.fetch_raw(packed.xyz, MIP_LEVELS - packed.w - 1) != 0) {
                        queue.append(packed);
                    }
                }
            }

            // check if this node is a voxel leaf node...
            if ((fetched & 1) == 1) {
                hit = true;
                //break;
            }
        }

        iterations++;
    }

    return iterations / 1024.0;
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