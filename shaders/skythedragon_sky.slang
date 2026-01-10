#ifndef SKY
#define SKY
#include <hash.slang>

// author: SkyTheDragon
// (sky the goat)

// shadertoy version of the flat atmosphere, published at https://www.shadertoy.com/view/t3XBWH

static const float3 solar_irradiance = float3(4.0);
static const float sun_angular_radius = 0.004675;
static const float bottom_radius = 6360.0;
static const float top_radius = 6460.0;

// rayleigh density
static const float ray_width_a = 0.0;
static const float ray_exp_term_a = 0.0;
static const float ray_exp_scale_a = 0.0;
static const float ray_linear_term_a = 0.0;
static const float ray_constant_term_a = 0.0;

static const float ray_width_b = 0.0;
static const float ray_exp_term_b = 1.0;
static const float ray_exp_scale_b = -0.125;
static const float ray_linear_term_b = 0.0;
static const float ray_constant_term_b = 0.0;
 
static const float3 ray_scattering = float3(0.005802, 0.013558, 0.033100);

// mie density
static const float mie_width_a = 0.0;
static const float mie_exp_term_a = 0.0;
static const float mie_exp_scale_a = 0.0;
static const float mie_linear_term_a = 0.0;
static const float mie_constant_term_a = 0.0;

static const float mie_width_b = 0.0;
static const float mie_exp_term_b = 1.0;
static const float mie_exp_scale_b = -0.833333;
static const float mie_linear_term_b = 0.0;
static const float mie_constant_term_b = 0.0;

static const float3 mie_scattering = float3(0.003996);
static const float3 mie_extinction = float3(0.004440);
static const float mie_g = 0.8;

// absorbsion (ozone) density
static const float absorb_width_a = 25.0;
static const float absorb_exp_term_a = 0.0;
static const float absorb_exp_scale_a = 0.0;
static const float absorb_linear_term_a = 0.066667;
static const float absorb_constant_term_a = -0.666667; 

static const float absorb_width_b = 0.0;
static const float absorb_exp_term_b = 0.0;
static const float absorb_exp_scale_b = 0.0;
static const float absorb_linear_term_b = -0.66667;
static const float absorb_constant_term_b = 2.666667;
   
static const float3 absorb_extinction = float3(0.000650, 0.001881, 0.000085);
static const float3 ground_albedo = float3(0.0);

static const float3 cam_pos = float3(0.0, 0.8, 0.0); // camera position
static const float3 sun_dir = normalize(float3(0.0, 0.1, 1.0)); // sun direction

static const float PI = 3.14159265358979323846;

// where does the planet start?
// adapted from: https://iquilezles.org/articles/spherefunctions/
float2 planet_bounds(float3 ray, float3 dir) {
    float b = dot(ray, dir);
    float c = dot(ray, ray) - bottom_radius * bottom_radius;
    float h = b * b - c;
    if (h < 0.0) return float2(-1.0);
    return float2(-b - sqrt(h), -b + sqrt(h));
}

// bruneton uses the cornette-shanks phase function
// so use the same here
// rayleigh assumes g = 0
float phase_ray(float cos_theta) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + cos_theta * cos_theta);
}

// mie assumes g = mie_g
float phase_mie(float cos_theta) {
    float k = 3.0 / (8.0 * PI) * (1.0 - mie_g * mie_g) / (2.0 + mie_g * mie_g);
    return k * (1.0 + cos_theta * cos_theta) / pow(1.0 + mie_g * mie_g - 2.0 * mie_g * cos_theta, 1.5);
}

// optical depth, without the coefficients applied
// rayleigh in x, mie in y, ozone in z
float3 scaled_depth(float3 ray, float3 dir) {
    float b = dot(ray, dir);
    float c = dot(ray, ray);
    float h = sqrt(c);

    // sphere sizes
    float4 r = float4(
        // ray and mie
        max(h - 1.0 / float2(ray_exp_scale_b, mie_exp_scale_b), bottom_radius),
        // ozone, no defined width of the ozone layer, so this is the best next thing
        // height + const / linear gives how far it extends outside the middle of the ozone layer
        max(h, bottom_radius + 1.5 * absorb_width_a + 0.5 * absorb_constant_term_b / absorb_linear_term_b),
        max(h, bottom_radius + 1.5 * absorb_width_a + 0.5 * absorb_constant_term_a / absorb_linear_term_a)
    );

    // scales, clamp to ensure it doesn't reach infinity too fast
    float2 s = exp(max(0.0, h - bottom_radius) * float2(ray_exp_scale_b, mie_exp_scale_b));

    // discriminant, max to ensure no NaN
    float4 d = sqrt(max(b*b + r*r - c, 0.0));

    // scaled optical depths, as length inside a sphere
    return float3(s * (d.xy - b), d.w - d.z);
}

// optical depth, along a ray
float3 optical_depth(float3 ray, float3 dir) {
    // closest point along the ray to the planet center
    float mid = dot(ray, dir);

    // optical depth
    return mid > 0.0
        // looking up
        ? scaled_depth(ray, dir)
        // looking down, use the full ray - what's behind the viewer
        : scaled_depth(ray - dir * mid, dir) * 2.0 - scaled_depth(ray, -dir);
}

// attenuation integral
float3 attenuate(float3 a, float3 b) {
    // prevent division by zero
    return lerp((exp(-a) - exp(-b)) / (b - a), exp(-a), select(a == b, 1, 0));
}

// scattering
float3 scatter(float3 ray, float3 dir, float3 light, float depth) {
    // coefficients, matrix makes it easy later
    float3x3 extinct = float3x3(ray_scattering, mie_extinction, absorb_extinction);

    // depth towards the camera and light
    float3 opt_view_start = optical_depth(ray, dir);
    float3 opt_light_start = optical_depth(ray, light);

    // depth towards the camera and light, at the surface intersection
    float3 opt_view_end = depth < 0.0 ? float3(0.0) : optical_depth(ray + dir * depth, dir);
    float3 opt_light_end = depth < 0.0 ? float3(0.0) : optical_depth(ray + dir * depth, light);

    // attenuation for scattering
    float3 attn = attenuate(
        mul(opt_light_start, extinct),
        mul((opt_light_end + opt_view_start - opt_view_end), extinct)
    );

    // phase function angle
    float cos_gamma = dot(dir, light);

    // combined scattering
    return solar_irradiance * (
        attn * (opt_view_start - opt_view_end).x * ray_scattering * phase_ray(cos_gamma) + 
        attn * (opt_view_start - opt_view_end).y * mie_scattering * phase_mie(cos_gamma)
    );
}

float3 sky(float3 sun_dir, float3 ray_dir, bool extraLightEnabled = true) {
    float3 ray_start = cam_pos + float3(0, bottom_radius, 0);
    float2 planet = planet_bounds(ray_start, ray_dir);

    float3 res = scatter(ray_start, ray_dir, normalize(sun_dir), planet.x) * 4;

    if (extraLightEnabled) {
        float dayTime = 1;
        float sunSpot = max(dot(ray_dir, sun_dir), 0.0);
        float sunMask = pow(sunSpot, 64.0) * dayTime;
        float sun = max(0.0, smoothstep(0.5, 1.5, dot(ray_dir, sun_dir))) * 0.5;
        float sunDisc = smoothstep(0.9999, 0.999935, dot(ray_dir, sun_dir));

        res += sunDisc;
    }

    return res;
}

#endif