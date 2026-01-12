#ifndef SKY
#define SKY

#include <thenuclearwolf_sky.slang>
#include <skythedragon_sky.slang>

// the sky colour is a mix of SkyTheDragon's realtistic sky and thenuclearwolf's starnight implementation
float3 sky(float3 sun_dir, float3 ray_dir, bool extraLightEnabled = true) {
    float sunHeight = sun_dir.y;
    float dayTime = smoothstep(-0.1, 0.1, sunHeight);
    float duskDawn = smoothstep(-0.3, 0.3, sunHeight) * (1.0 - dayTime);
    float night = 1.0 - smoothstep(-0.3, 0.0, sunHeight);

    float3 ray_start = cam_pos + float3(0, bottom_radius, 0);
    float2 planet = planet_bounds(ray_start, ray_dir);

    float3 res = scatter(ray_start, ray_dir, normalize(sun_dir), planet.x) * 4 * dayTime;

    if (extraLightEnabled) {
        float sunSpot = max(dot(ray_dir, sun_dir), 0.0);
        float sunMask = pow(sunSpot, 64.0) * dayTime;
        float sun = max(0.0, smoothstep(0.5, 1.5, dot(ray_dir, sun_dir))) * 0.5;
        float sunDisc = smoothstep(0.9999, 0.999935, dot(ray_dir, sun_dir));

        res += sunDisc * dayTime;

        float starsIntensity = stars(ray_dir) * 0.3;
        res += starsIntensity * night;
    }

    return res;
}


// goated approximation for now
// TODO: actually convert this to a proper skybox with proper convolution stuff
// https://www.shadertoy.com/view/3fG3zw

float3 getEnvironmentDiffuse(float3 Dir, float3 SkyColor, float3 GroundColor) {
    float Elevation = Dir.y * 0.5 + 0.5;
	return lerp(GroundColor, SkyColor, Elevation);   
}

float3 getEnvironmentSpecular(float3 Dir, float Roughness, float3 SkyColor, float3 GroundColor)
{
    float Roughness2 = Roughness*Roughness;
    float A = lerp(0.49,-0.5, Roughness2);
    float B = lerp(0.51, 1.5, Roughness2);
    float Elevation = smoothstep(A, B, Dir.y * 0.5 + 0.5);
    float Darken = exp2(-smoothstep(0.0, 1.0, abs(Dir.y)) * (1.0 - Roughness * 0.9) * 3.0);
	return lerp(GroundColor, SkyColor, Elevation) * Darken;   
}

// Golfed, if you care about that kind of stuff
float3 ed(float3 v, float3 sc, float3 gc) {
	return lerp(gc,sc,v.y*.5+.5);   
}

float3 es(float3 v, float r, float3 sc, float3 gc) {
	return lerp(gc,sc,smoothstep(lerp(.49,-.5,r*r),lerp(.51,1.5,r*r),v.y*.5+.5))*
        exp2(-smoothstep(0.,1.,abs(v.y))*(1.-r*.9)*3.);   
}

#endif