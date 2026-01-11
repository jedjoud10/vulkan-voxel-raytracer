#ifndef THENUCLEARWOLF_SKY
#define THENUCLEARWOLF_SKY
#include <hash.slang>


// originally by TheNuclearWolf
// https://www.shadertoy.com/view/lcGfDK

float stars(float3 rd) {
    if(rd.y <= 0.0) return 0.0;
    
    float stars = 0.0;
    float2 uv = rd.xz / (rd.y + 1.0);
    
    float2 id = floor(uv * 700.0 + 234.0);
    float star = hash12(id);
    float brightness = smoothstep(0.98, 1.0, star);
    stars += brightness * 0.5;
    
    return stars * rd.y;
}

float3 sky_color(float3 sunDir, float3 rd, bool extraLightEnabled = true) {
    float sunHeight = sunDir.y;
    float dayTime = smoothstep(-0.1, 0.1, sunHeight);
    float duskDawn = smoothstep(-0.3, 0.3, sunHeight) * (1.0 - dayTime);
    float night = 1.0 - smoothstep(-0.3, 0.0, sunHeight);
    
    // colors
    float3 dayZenith = float3(0.2, 0.4, 0.8);
    float3 dayHorizon = float3(0.8, 0.9, 1.0);
    float3 duskZenith = float3(0.2, 0.15, 0.3);
    float3 duskHorizon = float3(1.0, 0.5, 0.2);
    float3 nightZenith = float3(0.02, 0.02, 0.04);
    float3 nightHorizon = float3(0.04, 0.05, 0.08);
    
    float horizon = pow(1.0 - max(rd.y, 0.0), 6.0 + night);
    
    // lerp colors for each time of day
    float3 dayColor = lerp(dayZenith, dayHorizon, horizon);
    float3 duskColor = lerp(duskZenith, duskHorizon, horizon);
    float3 nightColor = lerp(nightZenith, nightHorizon, horizon);
    
    float3 skyColor = lerp(nightColor, duskColor, duskDawn);
    skyColor = lerp(skyColor, dayColor, dayTime);
    
    float sunSpot = max(dot(rd, sunDir), 0.0);
    float sunMask = pow(sunSpot, 64.0) * dayTime;
    float sun = max(0.0, smoothstep(0.5, 1.5, dot(rd, sunDir))) * 0.5;
    float sunDisc = smoothstep(0.999, 0.99925, dot(rd, sunDir));
    
    float starsIntensity = stars(rd);
    
    float3 moonDir = -sunDir;
    float moonGlow = pow(max(dot(rd, moonDir), 0.0), 32.0) * 0.5;
    float3 moonlight = float3(0.6, 0.7, 0.9) * moonGlow;
    float moonDisc = smoothstep(0.9995, 1.0, dot(rd, moonDir));

    float3 extra = extraLightEnabled ? ((sun + sunDisc) * dayTime) + ((moonlight + moonDisc + starsIntensity) * night) : 0;
    
    return skyColor + extra;
}

#endif