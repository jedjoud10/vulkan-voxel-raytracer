#language slang 2026
#include <lighting.slang>
#include <pbr.slang>
#include <ray_stuff_other.slang>

[[vk::binding(0, 0)]]
readonly RWTexture2D<uint32_t2> rendered;

[[vk::binding(1, 0)]]
RWTexture2D<float4> render_target;

[[vk::binding(0, 1)]]
StructuredBuffer<float4> lights_buffer;

[[vk::binding(1, 1)]]
SamplerCube<float4> skybox_sampler;

[[vk::binding(2, 1)]]
Sampler2D<float4> clouds_sampler;

// https://stackoverflow.com/questions/23975555/how-to-calculate-a-ray-plane-intersection#23976134
Optional<float> ray_plane_intersect_generic(float3 ray_dir, float3 ray_pos, float3 plane_normal, float3 plane_pos) {
    float denom = dot(ray_dir, plane_normal);
    if (abs(denom) > 0.0001f) // your favorite epsilon
    {
        float t = dot((plane_pos - ray_pos), plane_normal) / denom;

        if (t >= 0) {
            return Optional<float>(t);
        } else {
            return none;
        }
    }
    return none;
}

// https://stackoverflow.com/questions/23975555/how-to-calculate-a-ray-plane-intersection#23976134
Optional<float2> ray_plane_intersect_for_clouds(float3 ray_dir, float3 ray_pos, float plane_height) {
    float denom = -ray_dir.y;
    if (abs(denom) > 0.0001f) // your favorite epsilon
    {
        float t = -(plane_height - ray_pos.y) / denom;
        float3 pos = t * ray_dir + ray_pos;

        if (t >= 0) {
            return Optional<float2>(pos.xz);
        } else {
            return none;
        }
    }
    return none;
}


// extra func to calculate sun colour
float3 sun_colour(float3 light) {
    // TODO: use atmo sky scattering instead of this
    float3 midday_colour = pow(float3(252, 232, 212) / 255.0, 1/2.2);
    float3 sunset_colour = pow(float3(249, 128, 7) / 255.0, 1/2.2);

    return lerp(sunset_colour, midday_colour, smoothstep(0, 0.2, light.y));
}

float4 clouds_interesting(float3 sun_dir, float3 ray_dir, float3 ray_pos, Sampler2D<float4> clouds_sampler) {
    float4 clouds_colour_opacity = 0;

    float scale = 8000;
    float height_1 = 800;
    float height_2 = 900;
    int iters = 8;
    

    if (let pos = ray_plane_intersect_for_clouds(ray_dir, ray_pos, height_1)) {
        if (let pos2 = ray_plane_intersect_for_clouds(ray_dir, ray_pos, height_2)) {
            let uvs = pos.xy / scale + 0.5;
            let uvs2 = pos2.xy / scale + 0.5;
            let p1 = float3(pos.x, height_1, pos.y);
            let p2 = float3(pos2.x, height_2, pos2.y);
            let depth = distance(p1, p2);

            if (all(uvs >= 0 & uvs <= 1)) {
                for (int i = 0; i < iters; i++) {
                    float factor = i / (float)iters;
                    float2 lerped = lerp(uvs, uvs2, factor);
                    clouds_colour_opacity += clouds_sampler.Sample(lerped) / (float)iters;
                }
            }

        }   
    }
    return clouds_colour_opacity;
}

float4 clouds(float3 sun_dir, float3 ray_dir, float3 ray_pos, Sampler2D<float4> clouds_sampler) {
    float4 clouds_colour_opacity = 0;
    if (let pos = ray_plane_intersect_for_clouds(ray_dir, ray_pos, 800)) {
        let uvs = pos.xy / 8000 + 0.5;

        if (all(uvs >= 0 & uvs <= 1)) {
            clouds_colour_opacity = clouds_sampler.Sample(uvs);
        }
    }
    return clouds_colour_opacity;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void write_render_texture(uint3 id: SV_DispatchThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform uint debug_type, uniform float time) {
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;

    let ndc = float4(screen_uvs, -1, 1);
    float3 ray_dir = normalize(mul(mat, ndc).xyz);
    float3 inv_dir = 1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    float3 ray_pos = position.xyz;

    uint32_t2 loaded = rendered[id.xy].xy;
    uint32_t flags = loaded.y;
    bool hit = (bool)(flags & 0b1);
    
    float3 colour = 0;
    
    if (hit) {
        //float32_t distance = reinterpret<float32_t>(loaded.x);
        uint32_t face = (flags >> 1) & 0b11;
        float3 normal = get_face_normal(face, sign(ray_dir));
        uint32_t corrected_world_axis_pos = (flags >> 3) & 0b1111111111;

        // we can solve for the missing UV coordinates without needing the distance!!!
        float3 k = 0;
        k[face] = corrected_world_axis_pos;
        float32_t distance = ray_plane_intersect_generic(ray_dir, ray_pos, normal, k+max(normal, 0)).value;

        float3 world_pos = ray_pos + ray_dir * distance;
        uint3 block_pos = floor(world_pos);
        
        block_pos[face] = corrected_world_axis_pos;
        world_pos[face] = corrected_world_axis_pos;
        float3 world = world_pos;

        //normal = normalize(normal + (hash33(floor(world * 16)) - 0.5) * 0.3);
        float3 diffuse = per_block_unique_colour(block_pos);
        float roughness = 0.50;
        float metallic = 0;
        float ao = 1.0;
        float shadow = 1.0;
        //float shadow = (1 - ((flags >> 13) & 1)) * (1-clouds(sun.xyz, sun.xyz, world_pos, clouds_sampler).a);

        PbrSurfaceData pbr_surface_data = PbrSurfaceData(world, diffuse, normal, roughness, metallic, ao, shadow);
        CameraData camera_data = CameraData(-ray_dir);

        float3 sun_colour = 1.0;
        SunData sun_data = SunData(sun.xyz, sun.y > 0 ? (3.2 * sun_colour) : 0);
        SkyData sky_data = SkyData(skybox_sampler, clouds_sampler);

        //colour = (block_pos % 16.0) / 16.0;
        colour = lighting(pbr_surface_data, camera_data, sun_data, sky_data);

        colour = ((flags >> 13) & 0b1);

        /*
        for (int l = 0; l < 100; l++) {
            float3 light_pos = lights_buffer[l].xyz;
            float3 light_towards_position = world - light_pos;
            float d = length(light_towards_position);
            if (d < 10) {
                light_towards_position = normalize(light_towards_position);
                float n_dot_l = saturate(dot(normal, -normalize(light_towards_position)));
                float strength = (1 / (d*d)) * smoothstep(0, 1, 10-d);
                colour += n_dot_l * strength * hash33(l * 3.21312) * 0.3; 
            }
        }
        */
    } else {
        float4 clouds_colour_opacity = clouds(sun.xyz, ray_dir, ray_pos, clouds_sampler);
        float3 sky_colour = skybox_sampler.Sample(ray_dir).xyz;
        float3 composed_sky_colour = lerp(sky_colour, clouds_colour_opacity.xyz, clouds_colour_opacity.w);
        colour = composed_sky_colour;
    }

    render_target[id.xy] = float4(pow(aces(colour), 1/2.2), 0);;
}