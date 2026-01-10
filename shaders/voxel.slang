#include <other.slang>
#include <lighting.slang>

[[vk::binding(0, 0)]]
RWTexture3D<uint8_t> voxels;

[[vk::binding(1, 0)]]
RWStructuredBuffer<SurfaceData> surface_data_buffer;

[[vk::binding(2, 0)]]
RWTexture3D<uint> voxel_indices;

[[vk::binding(3, 0)]]
RWStructuredBuffer<Atomic<uint>> counter;


[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 id: SV_DispatchThreadID) {
    int noisy = (int)(noise((float2)id.xz * 0.1) * 10);
    int base = id.y - 15;
    base += noisy * 0.5;

    Voxel voxel;
    voxel.active = base < 0;
    voxel.reflective = false;
    voxel.refractive = false;
    voxels[id] = voxel.into_raw();
}

static const int3[] offsets = {
    int3(1, 0, 0),
    int3(-1, 0, 0),
    int3(0, 1, 0),
    int3(0, -1, 0),
    int3(0, 0, 1),
    int3(0, 0, -1),
};

uint calculate_enabled_faces(uint3 id) {
    uint enabled_faces = 0;

    for (int i = 0; i < 6; i++) {
        Voxel neighbour = Voxel.from_raw(voxels[id + offsets[i]]);
        bool face_visible_neighbour = !neighbour.active || neighbour.refractive;
        if (face_visible_neighbour) {
            enabled_faces |= 1 << i;
        }
    }

    return enabled_faces;
}

[SpecializationConstant] const uint SHADOW_SAMPLES_PER_TICK = 4;
[SpecializationConstant] const float SHADOW_TEMPORAL_LERP_ACCUMULATOR_FACTOR = 8.0;
[SpecializationConstant] const bool SHADOW_TEMPORAL = false;
[SpecializationConstant] const bool SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR = true;
[SpecializationConstant] const float SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR_FACTOR = 5.0;
[SpecializationConstant] const float SHADOW_TEMPORAL_CLAMPED_ACCUMULATOR_MARGIN = 0.4;
[SpecializationConstant] const bool SHADOW_USE_RANDOM_ANGLE_SPREAD = false;
[SpecializationConstant] const float SHADOW_RANDOM_ANGLE_SPREAD_FACTOR = 0.02;

[shader("compute")]
[numthreads(8, 8, 8)]
void update(uint3 id: SV_DispatchThreadID, uniform float4 forward, uniform float4 position, uniform float4 sun, uniform uint tick, uniform float delta_raw) {
    
    // this basically moves the block down if the block underneath it is empty air
    // this simulates gravity
    /*
    if ((voxels[id - uint3(0, 1, 0)] & 1) == 0 && (voxels[id] & 1) == 1 && id.y > 0 && (tick % 128) == 0) {
        voxels[id - uint3(0, 1, 0)] = voxels[id];
        voxels[id] = 0;
        return;
    }
    */

    // skip if the block is empty
    if ((voxels[id] & 1) == 0) {
        voxel_indices[id] = INVALID;
        return;
    }

    // TREAT THIS AS -1 SINCE WE DO SOME BIT TWIDDLING AT THE END
    // Contains both the block index and enabled faces bitset
    uint packed_index = INVALID;

    // this calculates the surface texels for each valid voxel
    // it checks the 6 nearby voxels and creates a surface if they are empty / refractive
    if (tick == 0) {
        uint enabled_faces = calculate_enabled_faces(id);

        /*
        AllMemoryBarrier();

        if (counter[0].load(MemoryOrder.Relaxed) > 0) {
            counter[0];
        }

        AllMemoryBarrier();
        */

        if (enabled_faces > 0) {
            uint prev_packed_index = voxel_indices[id];
            uint prev_block_index = prev_packed_index & ~(111111 << 26);

            uint block_index = counter[0].add(countbits(enabled_faces), MemoryOrder.Relaxed);
            packed_index = block_index;
            packed_index |= enabled_faces << (32 - 6);
            voxel_indices[id] = packed_index;

            for (int i = 0; i < countbits(enabled_faces); i++) {
                SurfaceData data = surface_data_buffer[prev_block_index + i];
                surface_data_buffer[block_index + i] = data;
            }
        } else {
            packed_index = INVALID;
            voxel_indices[id] = INVALID;
        }
    } else {
        packed_index = voxel_indices[id];
    }

    
    float3 diff = normalize((float3)id - position.xyz + 0.5);
    bool block_visible = dot(forward.xyz, diff) > 0.0;

    // Unpack the block index and enabled faces bitset
    uint block_index = packed_index & ~(0b111111 << (32 - 6));
    uint enabled_faces = (packed_index >> (32 - 6)) & 0b111111;

    float delta = delta_raw;

    if (enabled_faces > 0 && block_visible && packed_index != INVALID) {
        uint packed_face_index = 0;
        for (int i = 0; i < 6; i++) {
            if (((enabled_faces >> i) & 1) == 1) {
                // More strict check to make sure that the face faces the camera
                // Causes buggy reflections, but we can deal with that for now...
                // TODO: do a cheap "visibility" test that takes account of voxel occlusion and reflections 
                bool face_visible_camera = dot(offsets[i], diff) < 0.0;
                
                if (face_visible_camera) {
                    // Read cached texel data
                    SurfaceData data = surface_data_buffer[block_index + packed_face_index];

                    // Execute the shadow calculation for every possible texel...
                    // TODO: ideally do an extra check to verify if this texel is visible by the main camera
                    for (int k = 0; k < 16; k++) {
                        int p = k;

                        // Skip the texel if it doesn't face the sun
                        if (dot(offsets[i], sun.xyz) <= 0) {
                            data.colors[p] = uint8_t4(0);
                            continue;
                        }


                        uint2 uv = uint2(p % 4, p / 4);
                        uint3 unflattened = unflatten_uvs(i / 2, i % 2 == 0, uv);
                        
                        // ts so slow twin...
                        // ts pmo... ong...
                        float3 shadow_color = 0.0;
                        float spread = SHADOW_USE_RANDOM_ANGLE_SPREAD ? SHADOW_RANDOM_ANGLE_SPREAD_FACTOR : 0; 
                        for (int s = 0; s < (SHADOW_USE_RANDOM_ANGLE_SPREAD ? SHADOW_SAMPLES_PER_TICK : 1); s++) {
                            // jarvis... randominate this shit...
                            float3 sun_sample = normalize(sun.xyz + (hash33(s * 2432.43243 - (id + unflattened / 3.0) * 232.342 + tick * 43.23) - 0.5) * spread);
                            float3 world_pos = id + 0.125 + offsets[i] * 0.15 + unflattened / 4.0 + sun_sample * 0.15;
                            shadow_color += dda_shadownate(voxels, sun_sample, world_pos);
                        }


                        uint8_t4 old = data.colors[p];
                        float3 a = old.xyz / 255.0;
                        float3 b = shadow_color / SHADOW_SAMPLES_PER_TICK;
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

                    // Write new texel data
                    surface_data_buffer[block_index + packed_face_index] = data;
                }

                // TODO: Figure out how to do the bit magic but in reverse order?
                packed_face_index++;
            }
        }
    }
}