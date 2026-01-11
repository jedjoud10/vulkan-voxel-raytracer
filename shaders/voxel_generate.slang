#include <other.slang>

[[vk::binding(0, 0)]]
RWTexture3D<uint8_t> voxels;

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