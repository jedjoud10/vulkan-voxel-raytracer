/*
#include <stack.slang>

[[vk::binding(0, 0)]]
Texture3D<uint8_t> voxels;

struct Ray {
    float3 ray_direction;
    uint16_t2 source_pixel;
    uint packed_target_screen_coordinate; // 16 bits for the x coord, 16 bits for the y coord
}

[[vk::binding(1, 0)]]
RWStructuredBuffer<Ray> rays;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Stack<12, uint>> stacks;

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
    rays[index] = Ray(ray_dir, (uint16_t2)id.xy);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace_rays(uint3 id: SV_DispatchThreadID, uniform float4 position) {
    Ray ray = rays[id.x];

    // check the stack and fetch the closest node

}

[shader("compute")]
[numthreads(64, 1, 1)]
void repack_rays(uint3 id: SV_DispatchThreadID, uniform float4 position) {
    // check which rays are incomplete. skip complete rays, as they have written their data to the output


    // only start counting for complete rays
}

*/