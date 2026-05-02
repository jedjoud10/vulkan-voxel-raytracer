#ifndef PBR
#define PBR

#include <sky.slang>

#define PI 3.1415926538


// Literally the whole implementation is stolen from
// https://www.youtube.com/watch?v=RRE-F57fbXw&ab_channel=VictorGordan
// and https://learnopengl.com/PBR/Lighting

// Normal distribution function
// GGX/Trowbridge-reitz model
float ndf(float roughness, float3 n, float3 h) {
	float a = roughness * roughness;
	float a2 = a * a;

	float n_dot_h = max(dot(n, h), 0.0);	
	float n_dot_h2 = n_dot_h * n_dot_h;	

	float semi_denom = n_dot_h2 * (a2 - 1.0) + 1.0;
	float denom = PI * semi_denom * semi_denom;
	return a2 / denom;
}

// Schlick/GGX model
float g1(float k, float3 n, float3 x) {
	float num = max(dot(n, x), 0);
	float denom = num * (1 - k) + k;
	return num / denom;
}

// Smith model
float gsf(float roughness, float3 n, float3 v, float3 l) {
	float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
	return g1(k, n, v) * g1(k, n, l);
}

// Fresnel function
float3 fresnel(float3 f0, float3 h, float3 v) {
	float cosTheta = clamp(dot(h, v), 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// Fresnel function with roughness
float3 fresnelRoughness(float3 f0, float3 v, float3 x, float roughness) {
	float cosTheta = clamp(1.0 - max(dot(v, x), 0), 0, 1);
	return f0 + (max(float3(1.0 - roughness), f0) - f0) * pow(cosTheta, 5.0);
}

// Cook-torrence model for specular
float3 specular(float3 f0, float roughness, float3 v, float3 l, float3 n, float3 h) {
	float3 num = ndf(roughness, n, h) * gsf(roughness, n, v, l) * fresnel(f0, h, v);
	float denom = 4 * max(dot(v, n), 0.0) * max(dot(l, n), 0.0) + 0.0001;
	return num / denom;
}

// Sun data struct
public struct SunData {
	public float3 direction;
	public float3 color;
};

// Sky data struct
public struct SkyData {
	public SamplerCube skybox;
};

// Camera data struct
public struct CameraData {
	public float3 view;
};

// Surface data struct 
public struct PbrSurfaceData {
	public float3 position;
	public float3 albedo;
	public float3 normal;
	public float roughness;
	public float metallic;
	public float visibility;
    public float3 shadows;
};

public float3 lighting(
	PbrSurfaceData surface,
	CameraData camera,
	SunData light,
	SkyData sky,
) {
    float3 half_view = normalize(camera.view + light.direction);
    float3 f0 = lerp(0.04, surface.albedo, surface.metallic);

	// Calculate kS and kD
	float3 ks = fresnelRoughness(f0, half_view, camera.view, surface.roughness);
	float3 kd = (1 - ks) * (1 - surface.metallic);
	

	float3 specular = specular(f0, surface.roughness, camera.view, light.direction, surface.normal, half_view);

	// HACK: this (shadow mult) should be here right? no clue why it wasn't there in the original version
	float3 specular_and_fresnel = (specular + fresnel(f0, half_view, camera.view)) * surface.shadows;

	float n_dot_l = max(dot(light.direction, surface.normal), 0.0);
	
	float3 outgoing_radiance = (kd * surface.albedo / PI + specular) * light.color * n_dot_l * surface.shadows;

	// Diffuse Irradiance IBL
	//float3 skyColor = sky(light.direction, float3(0, 0.8, 0), surface.position, false) + 0.1;
	//float3 groundColor = sky(light.direction, float3(0, 0.05, 0), surface.position, false) + 0.05f;
	float3 skyColor = sky.skybox.Sample(float3(0, 0.8, 0)).xyz + 0.1;
	float3 groundColor = sky.skybox.Sample(float3(0, 0.05, 0)).xyz + 0.05f;
	float3 irradiance = getEnvironmentDiffuse(surface.normal, skyColor, groundColor);

	//float3 irradiance = texture(samplerCube(ibl_diffuse_map, ibl_diffuse_map_sampler), surface.normal).xyz;
	float3 environment_specular = getEnvironmentSpecular(reflect(camera.view, surface.normal), surface.roughness, skyColor, groundColor);
	
	// make the environment specular reflect the actual sky as the roughness decreases
	float smoothness = 1-surface.roughness;
	environment_specular = lerp(environment_specular, sky.skybox.Sample(-reflect(camera.view, surface.normal)).xyz, smoothstep(0.8, 1.0, smoothness));


	//float environment_specular_fresnel_strength = pow(max(dot(reflect(camera.view, surface.normal), camera.view), 0), 4) * surface.sky_visibility;
	float3 ambient = (irradiance * surface.albedo * kd + environment_specular) * surface.visibility;
	// + float3(clamp(dot(reflect(camera.view, surface.normal), camera.view), 0, 1)) * 0.04
	//return getEnvironmentSpecular(surface.normal, 0, skyColor, groundColor);



	//return specular_fresnel_ambient;
	//return surface.sky_visibility;
	//return environment_specular * environment_specular_fresnel_strength;
	return outgoing_radiance + ambient * 0.2;
	//return brdf + ambient * 0.8 + environment_specular * environment_specular_fresnel_strength;
	//return specular;
	//return ambient * 0.5;
}

#endif