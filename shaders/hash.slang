#ifndef HASH
#define HASH

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

#endif