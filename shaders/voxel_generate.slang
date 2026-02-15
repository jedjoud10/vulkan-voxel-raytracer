#include <other.slang>
#include <octree.slang>
#include <noises.slang>

[[vk::binding(0, 0)]]
RWTexture3D<uint64_t> voxels;

[[vk::binding(1, 0)]]
RWTexture3D<uint64_t> prev_mipmapped_voxels;

[[vk::binding(2, 0)]]
RWTexture3D<uint64_t> next_mipmapped_voxels;
/*
[[vk::binding(1, 0)]]
RWStructuredBuffer<Node> octree;

*/

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 id: SV_DispatchThreadID) {
    uint64_t raw = 0;
    int i = 0;

    float3 p = (float3)id;
    
    float noisy = (snoise((float2)p.xz * 0.03) * 10);
    float base = (float)p.y - 15.0;
    base += noisy * 0.5;

    float extra = min(0, noise((float2)p.xz * 0.05) * 20);
    base += extra;

    //base += saturate(sin(snoise(p * 0.01) * 0.2)) * 90.5;
    float detail = saturate(sin(snoise(p * float3(1, 4, 1) * 0.1)) * 5) * 2;
    base += detail;

    float cave = sign(abs(snoise(p * 0.02)) - 0.2) * 10 - 2;
    cave += snoise(p * 2.8 * float3(1, 2, 1)) * 5.0;
    //cave += max(snoise(p.xz * 0.05123), 0) * 8.0;
    //cave -= max(snoise(p.xz * 0.1312), 0) * 8.0 - p.y;
    //base = cave;

    raw = (base < 0.0) ? 1 : 0; 
    //voxel.type = cave < 0 ? Type.Grass : Type.Rock;
    //voxel.refractive = extra < 0.0;


    /*
	for (int x = 0; x < 4; x++) {
		for (int y = 0; y < 4; y++) {
			for (int z = 0; z < 4; z++) {
                float3 p = (float3)id + float3(x, y, z) / 4.0;
    
                float noisy = (snoise((float2)p.xz * 0.03) * 10);
                float base = (float)p.y - 15.0;
                base += noisy * 0.5;

                float extra = min(0, noise((float2)p.xz * 0.05) * 20);
                base += extra;

                //base += saturate(sin(snoise(p * 0.01) * 0.2)) * 90.5;
                float detail = saturate(sin(snoise(p * float3(1, 4, 1) * 0.1)) * 5) * 2;
                base += detail;

                float cave = sign(abs(snoise(p * 0.02)) - 0.2) * 10 - 2;
                cave += snoise(p * 2.8 * float3(1, 2, 1)) * 5.0;
                //cave += max(snoise(p.xz * 0.05123), 0) * 8.0;
                //cave -= max(snoise(p.xz * 0.1312), 0) * 8.0 - p.y;
                //base = cave;

                raw |= (base < 0.0) ? ((uint64_t)1 << i) : 0; 
                //voxel.type = cave < 0 ? Type.Grass : Type.Rock;
                //voxel.refractive = extra < 0.0;

                i++;
            }            
        }
    }
    */
    
    voxels[id] = raw;
}


public static const uint64_t[] LOCAL_MASKS_2x2x2 = {
            0x330033,
            0xcc00cc,
          0x33003300,
          0xcc00cc00,
    0x33003300000000,
    0xcc00cc00000000,
    0x3300330000000000,
    0xcc00cc0000000000,
};


[shader("compute")]
[numthreads(8, 8, 8)]
void propagateMipMaps(uint3 id: SV_DispatchThreadID) {
    uint64_t mask = 0;

	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
                uint3 pos = id * 2 + uint3(x, y, z);


                /*
                for (int lx = 0; lx < 2; lx++) {
		            for (int ly = 0; ly < 2; ly++) {
			            for (int lz = 0; lz < 2; lz++) {
                            uint64_t local_mask = LOCAL_MASKS_2x2x2[lz + ly * 2 + lx * 2 * 2];

                            if ((prev_mipmapped_voxels[pos] & local_mask) != 0) {
                                int3 fuck = int3(x, y, z) * 2 + int3(lx, ly, lz);
                                int index = fuck.z + fuck.y * 4 + fuck.x * 4 *4;
                                mask |= 1 << index;
                            }
                        }
                    }
                }
                */

                bool was_propagated_any = (prev_mipmapped_voxels[pos] & 1) == 1;
                /*
                bool is_full_voxel = (prev_mipmapped_voxels[pos] & 1) == 1;
                bool was_propagated_full = (prev_mipmapped_voxels[pos] & 4) == 4;
                has_any |= is_full_voxel || was_propagated_any;
                has_all &= is_full_voxel || was_propagated_full;
                */
            }
        }
    }

    next_mipmapped_voxels[id] = mask;
}

[shader("compute")]
[numthreads(8, 8, 8)]
void generateOctree(uint3 id: SV_DispatchThreadID) {
}