#include <lighting.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(1, 0)]]
RWTexture3D<uint8_t> voxels;

[[vk::binding(2, 0)]]
RWStructuredBuffer<SurfaceData> surface_data_buffer;

[[vk::binding(3, 0)]]
RWTexture3D<uint> voxels_indices;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun) {
    float2 uvs = (float2)id.xy / screen;
    uvs *= 2.0;
    uvs -= 1.0;
    uvs.y = -uvs.y;
    uvs.x = -uvs.x;

    float3 ray_dir = normalize((mul(mat, float4(uvs, 1, 0))).xyz);
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

    for (int i = 0; i < 128; i++) {
        Voxel voxel = fetcher.fetch((int3)floored_pos);

        if (voxel.active) {
            if (i == 0) {
                hit = true;
                color = 0.0;
                break;
            }

            float3 test = (floored_pos - ray_pos + 0.5 - 0.5 * dir_sign) * inv_dir;
            float max = max3(test.x, test.y, test.z);
            float3 world = ray_pos + ray_dir * max;
            float3 uv = world - floored_pos;
            float3 normal = normal(face, dir_sign);

            if (voxel.refractive || voxel.reflective) {
                //normal += float3(sin(world.x * 10 + 0.2565), cos(world.y * 10 + 0.89684), sin(world.z * 10 - 0.211256)) * 0.12;
                normal += (hash33(uv * float3(23.231, -435.4354, 9412.1)) - 0.5) * 0.05;
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
                uint funny_index_magic = voxels_indices[(uint3)floored_pos];

                float3 shadow = 0.0;
                uint tmp = 0;
                shadow = dda_shadownate(voxels, sun.xyz, world + sun.xyz * 0.01);
                /*
                uint funny_index = funny_index_magic & ~(0b111111 << (32 - 6));
                if (funny_index != INVALID) {
                    uint enabled_faces = (funny_index_magic >> (32 - 6)) & 0b111111;
                    uint face2 = global_face(face, dir_sign);

                    enabled_faces = (enabled_faces << (6 - face2)) & 0b111111;
                    int packed_face_index = countbits(enabled_faces);

                    SurfaceData surface_data = surface_data_buffer[funny_index + packed_face_index];

                    float2 flat = flatten_uvs(face, dir_sign, uv);
                    uint2 pixels = (uint2)(floor(flat * 4 - 0.002));
                    uint converted_to_flat_index = pixels.x + pixels.y * 4;
                    shadow = float3(surface_data.colors[converted_to_flat_index].xyz / 255.0);
                }
                */
                
                ao_solver solver;
                solver.fetcher = fetcher;
                solver.pos = (uint3)floored_pos;
                solver.uv = floor(uv * 8) / 7;
                solver.face = face;
                solver.sign = dir_sign;
                float ao = solver.ao();

                //color = (float)i / 128;
                color = light(sun.xyz, fetcher, (uint3)floored_pos, world, ray_dir, uv, normal, ao, shadow);
                break;
            }
        }

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        face = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }

    if (!hit) {
        color = sky(sun.xyz, ray_dir);
    }

    color *= tint;

    color = clamp(pow(aces(color * 1.3), 1 / 2.2), 0, 1);
    output[id.xy] = float4(color, 0);
}