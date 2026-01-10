#ifndef LIGHTING
#define LIGHTING
#include <other.slang>
#include <hash.slang>

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

// Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
float3 aces(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

struct ao_solver {
    Fetcher fetcher;
    uint3 pos;
    float3 uv;
    int face;
    float3 sign;

    bool check(int3 offset, int target_face) {
        if (sign[target_face] < 0) {
            offset[target_face] = -offset[target_face];
        }

        return face == target_face && fetcher.fetch(pos + offset).active;
    }

    float check_axis(int i, int a, int b) {
        bool4 c = false;

        static const int2[] array = {
            int2(-1, -1),
            int2(-1, 1),
            int2(1, -1),
            int2(1, 1),
        };

        static const int2[] array2 = {
            int2(-1, 0),
            int2(1, 0),
            int2(0, -1),
            int2(0, 1),
        };

        static const int[] corner_from_side_lookup = {
            0, 1,
            2, 3,
            0, 2,
            1, 3
        };


        for (int k = 0; k < 4; k++) {
            int3 corner = 0;
            corner[i] = -1;
            corner[a] = array[k].x;
            corner[b] = array[k].y;
            c[k] = check(corner, i);
        }

        for (int s = 0; s < 4; s++) {
            int3 side = 0;
            side[i] = -1;
            side[a] = array2[s].x;
            side[b] = array2[s].y;

            int id1 = corner_from_side_lookup[s * 2];
            int id2 = corner_from_side_lookup[s * 2 + 1];

            if (check(side, i)) {
                c[id1] = true;
                c[id2] = true;
            }
        }

        float4 selected = select(c, 1, 0);
        float interpolated = 1 - lerp(lerp(selected.x, selected.y, uv[b]), lerp(selected.z, selected.w, uv[b]), uv[a]);
        return interpolated;
    }

    float ao() {
        float x = check_axis(0, 1, 2);
        float z = check_axis(2, 1, 0);
        float y = check_axis(1, 0, 2);
        return x * y * z;
    }
}

#endif