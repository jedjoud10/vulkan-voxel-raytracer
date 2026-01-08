#ifndef OTHER
#define OTHER

static const int SIZE = 64;
static const uint INVALID = 0x3FFFFFF;

// Hash function from H. Schechter & R. Bridson, goo.gl/RXiKaH
// https://gist.github.com/keijiro/24f9d505fac238c9a2982c0d6911d8e3
uint hash(uint s)
{
    s ^= 2747636419u;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    return s;
}

// https://www.shadertoy.com/view/4djSRW
float hash12(float2 p) {
    float3 p3  = fract(float3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float hash13(float3 p3)
{
    p3 = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

float3 hash33(float3 p3)
{
    p3 = fract(p3 * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

float noise(float2 p) {
    float zz = hash12(floor(p));
    float zo = hash12(floor(p) + float2(1, 0));
    float oz = hash12(floor(p) + float2(0, 1));
    float oo = hash12(floor(p) + float2(1, 1));

    float2 uv = frac(p);
    return lerp(lerp(zz, zo, uv.x), lerp(oz, oo, uv.x), uv.y);
}

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
    RWTexture3D<uint8_t> voxels;

    Voxel fetch(int3 position) {
        uint8_t raw = 0;

        if (all(position >= 0) && all(position < SIZE)) {
            raw = voxels[position];
        }
        
        return Voxel.from_raw(raw);
    }
}

bool dda(
    RWTexture3D<uint8_t> voxels,
    float3 ray_dir,
    float3 ray_pos,
    out uint iter,
) {
    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);
    Fetcher fetcher = Fetcher(voxels);

    for (int i = 0; i < 16; i++) {
        Voxel voxel = fetcher.fetch((int3)(floored_pos));

        if (voxel.active) {
            iter = i;
            return true;
        }

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    iter = 16;
    return false;
}

float3 normal(int face, float3 sign) {
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
static const uint[4 * 4] null_data2 = { 0 };

struct SurfaceData {
    uint8_t4[4 * 4] colors;
    //uint[4 * 4] colors2;

    __init() {
        colors = null_data;
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