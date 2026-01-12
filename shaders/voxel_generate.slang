#include <other.slang>
#include <octree.slang>

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
    float noisy = (noise((float2)id.xz * 0.1) * 10);
    float base = (float)id.y - 15.0;
    base += noisy * 0.5;

    float extra = min(0, noise((float2)id.xz * 0.05) * 20);
    base += extra;

    Voxel voxel;
    voxel.active = base < 0.0;
    voxel.reflective = false;
    voxel.refractive = extra < 0.0;
    voxels[id] = voxel.into_raw();
}

[shader("compute")]
[numthreads(8, 8, 8)]
void propagateMipMaps(uint3 id: SV_DispatchThreadID) {
    bool has_voxels = false;

	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
                uint3 pos = id * 2 + uint3(x, y, z);
                has_voxels |= (prev_mipmapped_voxels[pos] & 1) == 1;
            }
        }
    }

    next_mipmapped_voxels[id] = (uint8_t1)(has_voxels ? 1 : 0);
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