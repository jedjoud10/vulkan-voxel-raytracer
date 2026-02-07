#include <stack.slang>

/*
struct Ray {
    float3 pos;
    float3 dir;
    float3 abs_inv_dir;
    int3 dir_sign;
    uint2 coord;
}

struct StackElement {
    float3 side_dist;
    int3 floored_pos;
    int depth;

    StackElement create(float3 ray_pos, float3 dir, float3 inv_dir, int3 dir_sign, float distance, int depth) {
        float3 modified_ray_pos = (ray_pos + dir * distance) / (1 << depth);  
        int3 floored_pos = (int3)floor(modified_ray_pos);
        float3 side_dist = ((dir_sign * (floored_pos - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    }
}

[[vk::binding(0, 0)]]
Texture3D<uint64_t> voxels;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Ray> rays;

[[vk::binding(2, 0)]]
RWStructuredBuffer<StackElement> stacks;

[[vk::binding(3, 0)]]
RWStructuredBuffer<uint8_t> stack_lengths;

[[vk::binding(3, 0)]]
RWStructuredBuffer<Atomic<uint>> ray_counter;

static const int MAX_STACK_SIZE_PER_RAY = 16;

[shader("compute")]
[numthreads(32, 32, 1)]
void generate_rays(uint3 id: SV_DispatchThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun) {
    float2 screen_uvs = (float2)id.xy / screen;
    screen_uvs *= 2.0;
    screen_uvs -= 1.0;
    screen_uvs.y = -screen_uvs.y;
    screen_uvs.x = -screen_uvs.x;

    float3 ray_dir = normalize((mul(mat, float4(screen_uvs, 1, 0))).xyz);
    int index = ray_counter[0].increment(MemoryOrder.Relaxed);
    rays[index] = Ray(position.xyz, ray_dir, 1.0 / ray_dir, id.xy);

    // add the root node to the stack to be traversed
    stack_lengths[index].add(1);
    stacks[index] = StackElement();
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace_rays(uint3 id: SV_DispatchThreadID, uniform float4 position) {
    Ray ray = rays[id.x];

    // do the ray trace for a SINGLE octree level
    // if we hit a node, add it to the stack (for the next iteration) and exit immediately
    // if not, set this ray as "missed"
}

[shader("compute")]
[numthreads(64, 1, 1)]
void repack_rays(uint3 id: SV_DispatchThreadID, uniform float4 position) {
    // check which rays are incomplete. skip complete rays, as they have written their data to the output


    // only start counting for complete rays
}
*/