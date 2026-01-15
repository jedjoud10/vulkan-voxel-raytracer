#include <other.slang>
#include <octree.slang>
#include <noises/noise3D.slang>

[[vk::binding(0, 0)]]
RWTexture3D<uint8_t> voxels;

[[vk::binding(1, 0)]]
RWTexture3D<uint8_t> prev_mipmapped_voxels;

[[vk::binding(2, 0)]]
RWTexture3D<uint8_t> next_mipmapped_voxels;
/*
[[vk::binding(1, 0)]]
RWStructuredBuffer<Node> octree;

*/

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 id: SV_DispatchThreadID) {
    float3 p = (float3)id;
    
    float noisy = (noise((float2)id.xz * 0.1) * 10);
    float base = (float)id.y - 15.0;
    base += noisy * 0.5;

    float extra = min(0, noise((float2)id.xz * 0.05) * 20);
    base += extra;

    base += saturate(sin(snoise(p * 0.01) * 0.2)) * 90.5;
    
    float cave = max(snoise(p * 0.02), 0) * 1000;
    base += cave;

    Voxel voxel;
    voxel.active = base < 0.0;
    voxel.reflective = false;
    voxel.refractive = false;
    //voxel.type = cave < 0 ? Type.Grass : Type.Rock;
    //voxel.refractive = extra < 0.0;
    voxels[id] = voxel.into_raw();
}

[shader("compute")]
[numthreads(8, 8, 8)]
void propagateMipMaps(uint3 id: SV_DispatchThreadID) {
    bool has_any = false;
    bool has_all = true;

	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
                uint3 pos = id * 2 + uint3(x, y, z);
                bool is_full_voxel = (prev_mipmapped_voxels[pos] & 1) == 1;
                bool was_propagated_any = (prev_mipmapped_voxels[pos] & 2) == 2;
                bool was_propagated_full = (prev_mipmapped_voxels[pos] & 4) == 4;
                has_any |= is_full_voxel || was_propagated_any;
                has_all &= is_full_voxel || was_propagated_full;
            }
        }
    }

    next_mipmapped_voxels[id] = (uint8_t1)((has_any ? 2 : 0) | (has_all ? 4 : 0));
}

[shader("compute")]
[numthreads(8, 8, 8)]
void generateOctree(uint3 id: SV_DispatchThreadID) {
    float noisy = (noise((float2)id.xz * 0.1) * 10);
    float base = (float)id.y - 15.0;
    base += noisy * 0.5;
    base += min(0, noise((float2)id.xz * 0.05) * 20);

    Voxel voxel;
    voxel.active = base < 0.0;
    voxel.reflective = false;
    voxel.refractive = false;
    voxels[id] = voxel.into_raw();
}