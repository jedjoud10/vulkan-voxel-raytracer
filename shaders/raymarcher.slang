#include <lighting.slang>
#include <pbr.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(1, 0)]]
RWTexture3D<uint8_t> voxels;

[[vk::binding(2, 0)]]
RWStructuredBuffer<SurfaceData> surface_data_buffer;

[[vk::binding(3, 0)]]
RWTexture3D<uint> voxels_indices;

[SpecializationConstant] const bool USE_PIXELATED_SHADOWS = true;
[SpecializationConstant] const bool USE_ROUND_NORMALS = false;
[SpecializationConstant] const bool USE_NOISY_REFRACTIONS_REFLECTIONS = false;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun) {
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;

    float3 ray_dir = normalize((mul(mat, float4(screen_uvs, 1, 0))).xyz);
    float3 ray_pos = position.xyz;
    //ray_pos += ray_dir * hash12((float2)(id.xy / 4) * float2(231.2312, 32.32423));

    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);

    Fetcher fetcher = Fetcher(voxels);
    
    float3 color = 0.0;
    bool hit = false;
    float3 tint = 1.0;
    int face = 0;

    float3 world;
    float3 normal;
    float3 block_uvs;
    uint3 block_pos;

    for (int i = 0; i < SIZE*2; i++) {
        Voxel voxel = fetcher.fetch((int3)floored_pos);

        if (voxel.active) {
            if (i == 0) {
                hit = true;
                color = 0.0;
                break;
            }

            float3 test = (floored_pos - ray_pos + 0.5 - 0.5 * dir_sign) * inv_dir;
            float max = max3(test.x, test.y, test.z);
            world = ray_pos + ray_dir * max;
            block_uvs = world - floored_pos;

            if (USE_ROUND_NORMALS) {
                normal = normalize(block_uvs * 2 - 1);
            } else {
                normal = get_face_normal(face, dir_sign);
            }
            block_pos = (uint3)floored_pos;

            if (voxel.refractive || voxel.reflective) {
                //normal += float3(sin(world.x * 10 + 0.2565), cos(world.y * 10 + 0.89684), sin(world.z * 10 - 0.211256)) * 0.12;
                
                if (USE_NOISY_REFRACTIONS_REFLECTIONS) {
                    normal += (hash33(block_uvs * float3(23.231, -435.4354, 9412.1)) - 0.5) * 0.05;
                }

                normal = normalize(normal);

                if (voxel.refractive) {
                    //ray_dir = normalize(ray_dir - normal);
                    //ray_dir = ray_dir;
                    ray_dir = refract(ray_dir, normal, 0.7);
                } else {
                    ray_dir = reflect(ray_dir, normal);
                }
                
                inv_dir = 1 / ray_dir;
                dir_sign = sign(ray_dir);
                side_dist = (floored_pos - world + 0.5 + 0.5 * dir_sign);
                ray_pos = world;
                tint *= normalize(hash33(floor(floored_pos) * float3(23.231, -435.4354, 9412.1)));
            } else {
                hit = true;
                break;
            }
        }

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        face = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    // either compute sky color or PBR color
    if (hit) {
        uint funny_index_magic = voxels_indices[block_pos];

        float3 shadow = 0.0;

        if (USE_PIXELATED_SHADOWS) {
            // if face = 0, enabled faces = 100111, target face is the 1st bit from the right, packed index = 0
            // 100111
            // (100111 << 6) & 0b111111
            // 0
            // count(0) = 0

            // if face = 1, enabled faces = 100111, target face is the 2nd bit from the right, packed index = 1
            // 100111
            // (100111 << 5) & 0b111111
            // count(100) = 1

            // if face = 4, enabled faces = 110011, target face is the 5th bit from the right, packed index = 2

            // first bit (from the left) is the corresponding face, but we count bits from the left
            // count number of valid bits until we reach (6-face) (counting from right)

            // we always only look at one face...
            // we can optimize the surface data fetch by knowing this

            // adds 2 padding bits to the right at the start
            // IT WORKS!!!
            uint funny_index = funny_index_magic & ~(0b111111 << (32 - 6));
            if (funny_index != INVALID) {
                uint enabled_faces = (funny_index_magic >> (32 - 6)) & 0b111111;
                uint face2 = global_face(face, dir_sign);

                enabled_faces = (enabled_faces << (6 - face2)) & 0b111111;
                int packed_face_index = countbits(enabled_faces);

                SurfaceData surface_data = surface_data_buffer[funny_index + packed_face_index];

                float2 flat = flatten_uvs(face, dir_sign, block_uvs);
                uint2 pixels = (uint2)(floor(flat * 4 - 0.002));
                uint converted_to_flat_index = pixels.x + pixels.y * 4;
                shadow = float3(surface_data.colors[converted_to_flat_index].xyz / 255.0);
            }
        } else {
            shadow = dda_shadownate(voxels, sun.xyz, world + sun.xyz * 0.01);
        }
        
        ao_solver solver;
        solver.fetcher = fetcher;
        solver.pos = block_pos;
        solver.uv = floor(block_uvs * 8) / 7;
        solver.face = face;
        solver.sign = dir_sign;
        float ao = solver.ao();
        ao = ao * 0.4f + 0.6;

        uint3 pixels = (uint3)(floor(block_uvs * 8 + 0.001));

        if (hash13(block_pos) > 0.5) {
            pixels.x = 7 - pixels.x;
        }
    
        if (hash13(block_pos + 12.12321) > 0.5) {
            pixels.z = 7 - pixels.z;
        }
    
        normal = normalize(normal + (hash33(pixels * float3(4.5984, 43.2323, -0.1212)) - 0.5) * 0.05);
        float mixer = hash13(pixels * float3(123.321, 21.322, -32.321)) * 0.2 + 0.8;
    
        //float3 grass = 1;
        //float3 dirt = 0.2;
        float3 grass = float3(33, 60, 30) / 255;
        float3 dirt = float3(58, 33, 22) / 255;

        bool top_face = normal.y > 0.5;
        bool top_pixels = pixels.y > 6 && !fetcher.fetch(block_pos + uint3(0, 1, 0)).active;
        float3 diffuse = (top_pixels ? grass : dirt) * (hash13(block_pos + 12) * 0.2 + 0.8);
        
        PbrSurfaceData pbr_surface_data = PbrSurfaceData(diffuse, normal, 0.25, 0.0, ao, shadow);
        CameraData camera_data = CameraData(-ray_dir);
        SunData sun_data = SunData(sun.xyz, sun.y > 0 ? 3 : 0);
        //color = brdf(pbr_surface_data, camera_data, sun_data);
        color = block_pos;
    } else {
        color = sky(sun.xyz, ray_dir);
    }

    color *= tint;

    //color = pow(color, 1 / 2.2);
    color = clamp(pow(aces(color * 1.3), 1 / 2.2), 0, 1);
    output[id.xy] = float4(color, 0);
}