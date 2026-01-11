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
	float cosTheta = max(dot(h, v), 0.0);
    return f0 + (1.0 - f0) * pow (1.0 - cosTheta, 5.0);
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
	return num;
}

// Sun data struct
public struct SunData {
	public float3 direction;
	public float3 color;
};

// Camera data struct
public struct CameraData {
	public float3 view;
};

// Surface data struct 
public struct PbrSurfaceData {
	public float3 diffuse;
	public float3 normal;
	public float roughness;
	public float metallic;
	public float visibility;
    public float3 shadows;
};

// Bidirectional reflectance distribution function, aka PBRRRR
public float3 brdf(
	PbrSurfaceData surface,
	CameraData camera,
	SunData light
) {
    float3 half_view = normalize(camera.view + light.direction);
    float3 f0 = lerp(0.04, surface.diffuse, surface.metallic);

	// Calculate kS and kD
	float3 ks = fresnelRoughness(f0, half_view, camera.view, surface.roughness);
	float3 kd = (1 - ks) * (1 - surface.metallic);
	

	float3 specular = specular(f0, surface.roughness, camera.view, light.direction, surface.normal, half_view);
	float3 brdf = kd * (surface.diffuse) + specular + fresnel(f0, half_view, camera.view);
	float3 lighting = float3(max(dot(light.direction, surface.normal), 0.0)) * surface.shadows;
	brdf *= lighting * light.color;

	// Diffuse Irradiance IBL
	// TODO: can we use Slang auto-diff to compute the irradiance and specular IBL in a smarter way?
    //float3 irradiance = sky(light.direction, surface.normal, true);

	float3 skyColor = sky(light.direction, float3(0, 0.8, 0), false) + 0.1;
	float3 groundColor = sky(light.direction, float3(0, 0.05, 0), false) + 0.05f;
	float3 irradiance = getEnvironmentDiffuse(surface.normal, skyColor, groundColor);

	//float3 irradiance = texture(samplerCube(ibl_diffuse_map, ibl_diffuse_map_sampler), surface.normal).xyz;
	float3 ambient = irradiance * surface.diffuse * kd * surface.visibility;
	// + float3(clamp(dot(reflect(camera.view, surface.normal), camera.view), 0, 1)) * 0.04
	//return getEnvironmentSpecular(surface.normal, 0, skyColor, groundColor);
	return brdf + ambient * 0.8;
	//return specular;
	//return ambient * 0.5;
}

#endif