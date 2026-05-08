#language slang 2026
#include <other.slang>

[[vk::binding(0, 0)]]
RWTexture3D<uint8_t4> sparse_voxel_texture;

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 id: SV_DispatchThreadID, uniform uint4 chunk_position) {
    float4 sum = 0;
    uint3 block_pos = id.xyz + chunk_position.xyz;

    if (sparse_voxel_texture[block_pos].w == 255) {
        sparse_voxel_texture[block_pos] = float4(per_block_unique_colour(block_pos) * 255.0, 255);
        return;
    }

    AllMemoryBarrier();


    // unrolling these loops does not help perf much
    // definitely helps occupancy but actually increases overall compute shader execution time 
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                uint4 neighbour_sample = sparse_voxel_texture[block_pos + int3(x,y,z) * 1];
                sum += (float4)neighbour_sample / 255.0;
            }
        }    
    }

    AllMemoryBarrier();

    sparse_voxel_texture[block_pos] = (uint4)((sum / 27.0) * 254);
}
