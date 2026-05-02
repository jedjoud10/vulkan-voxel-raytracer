#ifndef OTHER
#define OTHER
#include <hash.slang>

static const int MIP_LEVELS = 9;
static const int SIZE = 1 << (MIP_LEVELS-1);
static const uint INVALID = 0x3FFFFFF;


float3 per_block_unique_colour(uint3 block_pos) {
    float3 col = normalize(hash33(block_pos * float3(23.231, -435.4354, 9412.1)));
    return col;
}

// TODO: pass this from CPU and don't hard code
static const int SPARSE_GRANULARITY = 64;

struct Fetcher {
    RWTexture3D<uint8_t> svt;
    RWTexture3D<uint8_t> svt_metadata;

    bool is_solid(uint3 pos) {
        return svt_metadata.Load(pos / SPARSE_GRANULARITY) == 255 || svt.Load(pos) == 255;
    }
}

// face: x, y, z
float3 get_accurate_world_position_with_face(float3 world, uint3 block_pos, int face, int3 sign) {
    // sign is either -1 or 1, we need 0 or 1
    sign = max(0, -sign);

    world[face] = block_pos[face] + sign[face];
    return world;
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


float2 flatten_uvs(int face, float3 dir_sign, float3 uvs) {
    if (face == 0) {
        return uvs.yz;
    } else if (face == 1) {
        return uvs.xz;
    } else if (face == 2) {
        return uvs.xy;
    }

    return -1;
}

uint3 unflatten_uvs(int face, bool negate, uint2 flattened) {
    uint val = negate ? 3 : 0;
    if (face == 0) {
        return uint3(val, flattened.x, flattened.y);
    } else if (face == 1) {
        return uint3(flattened.x, val, flattened.y);
    } else if (face == 2) {
        return uint3(flattened.x, flattened.y, val);
    }

    return -1;
}

/*
public static const int3[] FACE_NORMALS = {
    int3(1, 0, 0),
    int3(-1, 0, 0),
    int3(0, 1, 0),
    int3(0, -1, 0),
    int3(0, 0, 1),
    int3(0, 0, -1),
};
*/

float3 transform_face_tangent_to_ws_normal(int global_face, float2 flattened) {
    float val = (global_face % 2 == 0) ? 1 : -1;

    if (global_face == 0 || global_face == 1) {
        return float3(val, flattened.x, flattened.y);
    } else if (global_face == 2 || global_face == 3) {
        return float3(flattened.x, val, flattened.y);
    } else if (global_face == 4 || global_face == 5) {
        return float3(flattened.x, flattened.y, val);
    }

    return 0;
}


#endif