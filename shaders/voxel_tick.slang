#include <other.slang>
#include <lighting.slang>

[[vk::binding(0, 0)]]
Texture3D<uint8_t> voxels;

[[vk::binding(1, 0)]]
RWTexture3D<uint> voxel_indices;

[[vk::binding(2, 0)]]
RWStructuredBuffer<SurfaceData> surface_data_buffer;

[[vk::binding(3, 0)]]
RWStructuredBuffer<Atomic<uint>> surface_counter;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> visible_surfaces;

[[vk::binding(5, 0)]]
RWStructuredBuffer<Atomic<uint>> visible_surfaces_counter;

[[vk::binding(6, 0)]]
RWStructuredBuffer<uint32_t3> indirect_dispatch_size;


uint calculate_enabled_faces(uint3 id) {
    uint enabled_faces = 0;

    for (int i = 0; i < 6; i++) {
        Voxel neighbour = Voxel.from_raw(voxels[id + FACE_NORMALS[i]]);
        bool face_visible_neighbour = !neighbour.active || neighbour.refractive;
        if (face_visible_neighbour) {
            enabled_faces |= 1 << i;
        }
    }

    return enabled_faces;
}

static const bool DO_FANCY_STUFF = false;
static const uint SHADOW_SAMPLES_PER_TICK = 32;
static const float SHADOW_TEMPORAL_LERP_ACCUMULATOR_FACTOR = 8.0;
static const bool SHADOW_TEMPORAL = false;
static const bool SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR = true;
static const float SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR_FACTOR = 5.0;
static const float SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR_MARGIN = 0.4;
static const bool SHADOW_USE_RANDOM_ANGLE_SPREAD = false;
static const bool SHADOW_USE_TICK_AS_HASH_INPUT = false;
static const float SHADOW_RANDOM_ANGLE_SPREAD_FACTOR = 0.02;
static const int SAMPLES = (SHADOW_USE_RANDOM_ANGLE_SPREAD ? SHADOW_SAMPLES_PER_TICK : 1);

[shader("compute")]
[numthreads(8, 8, 8)]
void unwrap(uint3 id: SV_DispatchThreadID) {
    if (!DO_FANCY_STUFF) {
        return;
    }
    
    // skip if the block is empty or is reflective or refract
    Voxel self = Voxel.from_raw(voxels[id]);
    if (!self.active || self.reflective || self.refractive) {
        voxel_indices[id] = INVALID;
        return;
    }


    // this calculates the surface texels for each valid voxel
    // it checks the 6 nearby voxels and creates a surface if they are empty / refractive
    uint enabled_faces = calculate_enabled_faces(id);

    if (enabled_faces > 0) {
        uint prev_packed_index = voxel_indices[id];
        uint prev_block_index = prev_packed_index & ~(111111 << 26);
        uint num_enabled_faces = countbits(enabled_faces);

        uint block_index = surface_counter[0].add(num_enabled_faces, MemoryOrder.Relaxed);
        uint packed_index = block_index | enabled_faces << (32 - 6);
        voxel_indices[id] = packed_index;

        for (int i = 0; i < num_enabled_faces; i++) {
            SurfaceData data = SurfaceData();
            //SurfaceData data = surface_data_buffer[prev_block_index + i];
            surface_data_buffer[block_index + i] = data;
        }
    } else {
        // the voxel does not have any valid faces, we can hide it away
        voxel_indices[id] = INVALID;
    }
}


[shader("compute")]
[numthreads(8, 8, 8)]
void unpack(uint3 id: SV_DispatchThreadID, uniform float4 forward, uniform float4 position, uniform float4 sun, uniform uint tick, uniform float delta_raw) {
    if (!DO_FANCY_STUFF) {
        return;
    }

    // TREAT THIS AS -1 SINCE WE DO SOME BIT TWIDDLING AT THE END
    // Contains both the block index and enabled faces bitset
    uint packed_index = voxel_indices[id];

    // skip if the block has no surface
    if (packed_index == INVALID) {
        return;
    }

    float3 diff = normalize((float3)id - position.xyz + 0.5);
    bool block_visible = dot(forward.xyz, diff) > 0.0;

    // Unpack the block index and enabled faces bitset
    uint block_index = packed_index & ~(0b111111 << (32 - 6));
    uint enabled_faces = (packed_index >> (32 - 6)) & 0b111111;

    if (enabled_faces > 0 && block_visible) {
        uint packed_face_index = 0;
        for (int i = 0; i < 6; i++) {
            if (((enabled_faces >> i) & 1) == 1) {
                // More strict check to make sure that the face faces the camera
                // Causes buggy reflections, but we can deal with that for now...
                // TODO: do a cheap "visibility" test that takes account of voxel occlusion and reflections 
                bool face_visible_camera = dot(FACE_NORMALS[i], diff) < 0.0;
                
                if (face_visible_camera) {
                    // write the [world block pos, side index, packed side index] that we will need to read in the next compute shader
                    int write_index = visible_surfaces_counter[0].increment(MemoryOrder.Relaxed);
                    int side_index = i;
                    int packed_side_index = packed_face_index;
                    
                    // encode world block index as 3 bytes
                    int world_block_index = id.x | id.y << 8 | id.z << 16;
                    visible_surfaces[write_index] = world_block_index | (side_index << (32 - 3)) | (packed_side_index << (32 - 6));
                }



                // TODO: Figure out how to do the bit magic but in reverse order?
                packed_face_index++;
            }
        }
    }
}

static const int SHADOW_DISPATCH_GROUP_SIZE = 128;

[shader("compute")]
[numthreads(1, 1, 1)]
void copyDispatchSize(uint3 id: SV_DispatchThreadID) {
    int raw = visible_surfaces_counter[0].load(MemoryOrder.Relaxed);
    int size = (int)ceil((float)raw / SHADOW_DISPATCH_GROUP_SIZE);
    indirect_dispatch_size[0] = uint32_t3(size, 1, 1);
}

[shader("compute")]
[numthreads(SHADOW_DISPATCH_GROUP_SIZE, 1, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform float4 forward, uniform float4 position, uniform float4 sun, uniform uint tick, uniform float delta) {
    if (!DO_FANCY_STUFF) {
        return;
    }

    if (id.x >= visible_surfaces_counter[0].load(MemoryOrder.Relaxed)) {
        return;
    }

    Fetcher fetcher = Fetcher(voxels);

    // packed index that contains the block index, side index, and packed side index
    uint packed_index = visible_surfaces[id.x];

    // unpack...
    uint world_block_index = packed_index & ~(0b111111 << (32 - 6));
    uint side_index = (packed_index >> (32 - 3)) & 0b111;   
    uint packed_face_index = (packed_index >> (32 - 6)) & 0b111;

    uint3 block_pos = uint3(world_block_index & 0xFF, (world_block_index >> 8) & 0xFF, (world_block_index >> 16) & 0xFF);
    uint block_index = voxel_indices[block_pos] & ~(0b111111 << (32 - 6));
    
    SurfaceData data = surface_data_buffer[block_index + packed_face_index];
    uint i = side_index;
    float3 face_normal = FACE_NORMALS[i];

    // Execute the shadow calculation for every possible texel...
    // TODO: ideally do an extra check to verify if this texel is visible by the main camera
    for (int k = 0; k < 16; k++) {
        int p = k;

        /*
        uint2 uv = uint2(p % 4, p / 4);
        uint3 unflattened = unflatten_uvs(i / 2, i % 2 == 0, uv);
        data.colors[p] = uint8_t4((uint8_t3)(unflattened * 255), 0);
        */

        // Skip the texel if it doesn't face the sun
        if (dot(face_normal, sun.xyz) <= 0) {
            data.colors[p] = uint8_t4(0);
            continue;
        }


        uint2 uv = uint2(p % 4, p / 4);
        uint3 unflattened = unflatten_uvs(i / 2, i % 2 == 0, uv);
        
        // ts so slow twin...
        // ts pmo... ong...
        float3 shadow_color = 0.0;
        float spread = SHADOW_USE_RANDOM_ANGLE_SPREAD ? SHADOW_RANDOM_ANGLE_SPREAD_FACTOR : 0; 
        for (int s = 0; s < SAMPLES; s++) {
            // jarvis... randominate this shit...
            float3 hash_id = s * 2432.43243 - (block_pos + unflattened / 3.0) * 232.342;
            hash_id += SHADOW_USE_TICK_AS_HASH_INPUT ? tick * 43.23 : 0.0;
            float3 hashed = (hash33(hash_id) - 0.5) * spread;

            float3 sun_sample = normalize(sun.xyz + hashed);
            float3 world_pos = block_pos + 0.125 + face_normal * 0.15 + unflattened / 4.0 + sun_sample * 0.15;
            shadow_color += dda_shadownate(fetcher, sun_sample, world_pos);
        }


        /*
        // test AO
        shadow_color = 0.0;
        for (int s = 0; s < 8; s++) {
            float3 hash_id = s * 2432.43243 - (block_pos + unflattened / 3.0) * 232.342;
            float2 hashed = (hash33(hash_id) - 0.5).xy * 2.0;
            float3 ray_dir = normalize(transform_face_tangent_to_ws_normal(i, hashed));

            float3 world_pos = block_pos + 0.125 + face_normal * 0.15 + unflattened / 4.0 + ray_dir * 0.15;
            shadow_color += dda<16>(voxels, ray_dir, world_pos) ? 0 : 1;
        }
        */


        uint8_t4 old = data.colors[p];
        float3 a = old.xyz / 255.0;
        float3 b = shadow_color / SAMPLES;
        float3 output = 0.0;

        // TODO: find a better accumulator function...
        if (SHADOW_TEMPORAL) {
            if (SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR) {
                float3 l = SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR_MARGIN;
                output = a + SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR_FACTOR * clamp(b - a, -l, l) * delta;
            } else {
                output = lerp(a, b, clamp(SHADOW_TEMPORAL_LERP_ACCUMULATOR_FACTOR * delta, 0.01, 1.0));
            }
        } else {
            output = b;
        }


        data.colors[p] = uint8_t4((uint8_t3)(clamp(output, 0, 1) * 255), 0);
    }

    surface_data_buffer[block_index + packed_face_index] = data;
}