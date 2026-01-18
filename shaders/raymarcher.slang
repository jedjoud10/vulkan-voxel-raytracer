#language slang 2026
#include <lighting.slang>
#include <pbr.slang>

[[vk::binding(0, 0)]]
RWTexture2D<float4> output;

[[vk::binding(1, 0)]]
Texture3D<uint8_t> voxels;

[[vk::binding(2, 0)]]
StructuredBuffer<SurfaceData> surface_data_buffer;

[[vk::binding(3, 0)]]
Texture3D<uint> voxels_indices;

[SpecializationConstant] const bool USE_PIXELATED_SHADOWS = true;
[SpecializationConstant] const bool USE_ROUND_NORMALS = false;
[SpecializationConstant] const bool USE_NOISY_REFRACTIONS_REFLECTIONS = false;

enum DebugType: uint {
    Combined,
    Iterations,
    Recursions,
    FullHalt,
    World,
}

groupshared uint min_clock;
groupshared uint max_clock;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform DebugType debug_type) {
    if (all(lid.xy == 0)) {
        max_clock = 0;
        min_clock = 0xFFFFFFFF;
    }
    GroupMemoryBarrierWithGroupSync();
    
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;

    float3 ray_dir = normalize((mul(mat, float4(screen_uvs, 1, 0))).xyz);
    float3 ray_pos = position.xyz;
    
    //output[id.xy] = float4(sky(sun_dir.xyz, ray_dir, true), 0);
    //return;
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);

    Fetcher fetcher = Fetcher(voxels);



    /*
    float3 col = 0.0;
    int hit2 = 

    //output[id.xy] = float4(hit2 ? (tmp % 1) : 0.0, 0.0);
    //output[id.xy] = float4(iterations / 8.0);
    //output[id.xy] = float4((float)hit2 / 512.0);
    output[id.xy] = float4(col, 0) ;
    return;

    //ray_pos += ray_dir * hash12((float2)(id.xy / 4) * float2(231.2312, 32.32423));

    float3 floored_pos = floor(ray_pos);
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    float3 side_dist = (floored_pos - ray_pos + 0.5 + 0.5 * dir_sign);
    */


    float3 color = 0.0;
    bool hit = false;
    float3 tint = 1.0;
    
    //MultiLevelThingyOutput<true, true, true> thingy;
    TraceStats stats;
    hit = dda_traverse2<true, true, true>(fetcher, ray_dir, lid, ray_pos, stats);
    
    /*
    //output[id.xy] = float4(stats.full_halts / 8.0);
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(data, stats.total_iterations);
    GroupMemoryBarrierWithGroupSync();

    int diff = data;
    //int diff = WaveActiveSum(stats.total_iterations) / 64.0;
    output[id.xy] = float4(diff / 64.0);
    */
    
    //
    switch (debug_type) {
        case DebugType.Combined:
            output[id.xy] = float4(stats.total_iterations / 64.0, stats.recursions / 8.0, stats.full_halts / 8.0, 0);
            return;
        case DebugType.FullHalt:
            output[id.xy] = float4(stats.full_halts / 8.0);
            return;
        case DebugType.Iterations:
            output[id.xy] = float4(stats.total_iterations / 64.0);
            return;
        case DebugType.Recursions:
            output[id.xy] = float4(stats.recursions / 64.0);
            return;
        case DebugType.World:
            uint clock = getRealtimeClock().x;

            GroupMemoryBarrierWithGroupSync();
            InterlockedMax(max_clock, clock);
            InterlockedMin(min_clock, clock);
            GroupMemoryBarrierWithGroupSync();
            //output[id.xy] = float4(stats.last_level_raw_dd / 64.0);
            //output[id.xy] = float4(world % 10.0 / 10.0, 0);
            output[id.xy] = float4((max_clock - min_clock) / 1024.0);
            break;
    }
    //output[id.xy] = float4(hit ? 1 : 0);
    //output[id.xy] = float4(world % 10 / 10.0, 0);
    return;

    int face = 0;
    float3 world = 0;
    float3 normal = 0;
    uint3 block_pos = 0;
    float3 block_uvs = 0;
    /*
    int face = (int)thingy.face.get().value;
    float3 world = thingy.world.get().value;
    float3 normal = get_face_normal(face, dir_sign);
    uint3 block_pos = (uint3)thingy.block_pos.get().value;
    float3 block_uvs = world - block_pos;
    */
    bool lastWasAir = true;

    /*
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

            if ((voxel.refractive || voxel.reflective)) {
                if (lastWasAir) {
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
                }
                
                tint *= per_block_unique_colour(block_pos);
            } else {
                hit = true;
                break;
            }

            lastWasAir = false;
        } else {
            lastWasAir = true;
        }

        float3 reconst = side_dist * inv_dir;
        int3 eqs = select(min3(reconst.x, reconst.y, reconst.z) == reconst, 1, 0);
        face = firstbithigh(eqs.x | eqs.y << 1 | eqs.z << 2);
        floored_pos += dir_sign * eqs;
        side_dist += dir_sign * eqs;
    }
    */

    //int iterations;
    //hit = traverse(fetcher, ray_dir, ray_pos, world, block_pos, block_uvs, face, iterations);
    //normal = get_face_normal(face, dir_sign);

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
            shadow = dda_shadownate(fetcher, sun.xyz, world + sun.xyz * 0.01);
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
    
        //color = normal;
        normal = normalize(normal + (hash33(pixels * float3(4.5984, 43.2323, -0.1212)) - 0.5) * 0.05);
        float mixer = hash13(pixels * float3(123.321, 21.322, -32.321)) * 0.2 + 0.8;
    
        //float3 grass = 1;
        //float3 dirt = 0.2;
        float3 grass = float3(33, 60, 30) / 255;
        float3 dirt = float3(58, 33, 22) / 255;

        float3 sky_reflected_dir = reflect(ray_dir, normal);
        bool top_face = normal.y > 0.5;
        bool top_pixels = pixels.y > 6 && !fetcher.fetch(block_pos + uint3(0, 1, 0)).active;
        float3 diffuse = (top_pixels ? grass : dirt) * (hash13(block_pos + 12) * 0.2 + 0.8);

        float sky_visibility = 1.0;
        /*
        float sky_visibility = 0.0;

        for (int k = 0; k < 8; k++) {
            float3 dir = normalize(sky_reflected_dir + (hash33(world * 23.12123545 + k * 30.43324) * 2.0 - 1.0) * 0.2);
            sky_visibility += dda<8>(fetcher, dir, world + dir * 0.1) ? 0 : 1;
        }

        sky_visibility /= 8.0;
        */
        
        
        PbrSurfaceData pbr_surface_data = PbrSurfaceData(diffuse, normal, 0.95, 0.0, ao, shadow, sky_visibility);
        CameraData camera_data = CameraData(-ray_dir);
        SunData sun_data = SunData(sun.xyz, sun.y > 0 ? 0.8 : 0);
        //color = normal;
        color = brdf(pbr_surface_data, camera_data, sun_data);
        //color = block_pos;
    } else {
        color = sky(sun.xyz, ray_dir);
    }

    color *= tint;

    //color = pow(color, 1 / 2.2);
    color = clamp(pow(aces(color * 1.3), 1 / 2.2), 0, 1);
    output[id.xy] = float4(color, 0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lighting(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform DebugType debug_type) {
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;

    float3 ray_dir = normalize((mul(mat, float4(screen_uvs, 1, 0))).xyz);
    float3 ray_pos = position.xyz;
    
    output[id.xy] = float4(0);
}