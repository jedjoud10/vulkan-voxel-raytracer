#language slang 2026
#include <ray_stuff_other.slang>

[[vk::binding(0, 0)]]
StructuredBuffer<uint64_t> svo_bitmasks_buffer;

[[vk::binding(1, 0)]]
StructuredBuffer<uint32_t> svo_indices_buffer;

[[vk::binding(2, 0)]]
StructuredBuffer<RayInput> ray_inputs;

[[vk::binding(3, 0)]]
RWStructuredBuffer<RayOutput> ray_outputs;

[[vk::binding(4, 0)]]
StructuredBuffer<Atomic<uint>> ray_counts;

[[vk::binding(5, 0)]]
RWStructuredBuffer<uint32_t3> indirect_dispatch_command;


[shader("compute")]
[numthreads(1, 1, 1)]
void copyDispatchSize(uint3 id: SV_DispatchThreadID) {
    int raw = ray_counts[0].load(MemoryOrder.Relaxed);
    int size = (int)ceil((float)raw / 64);
    indirect_dispatch_command[0] = uint32_t3(size, 1, 1);
}

/*
[shader("compute")]
[numthreads(64, 1, 1)]
void traceRaysRecursiveCoarse(uint3 id: SV_DispatchThreadID) {
    if (id.x >= ray_counts[0].load(MemoryOrder.Relaxed)) {
        return;
    }


}
*/

[shader("compute")]
[numthreads(64, 1, 1)]
void traceRaysRecursive(uint3 id: SV_DispatchThreadID) {
    return;
    if (id.x >= ray_counts[0].load(MemoryOrder.Relaxed)) {
        return;
    }
    
    RayInput input = ray_inputs[id.x];
    float3 ray_pos = WaveReadLaneFirst(input.position.xyz);
    float3 ray_dir = input.direction.xyz;


    uint clock1 = getRealtimeClock().x;
    
    // calculate average ray direction for entire wave
    /*
    float3 avg_ray_dir = WaveActiveSum(input.direction.xyz) / 64.0;
    half3 avg_inv_dir = (half3)1.0 / abs(avg_ray_dir);
    int8_t3 avg_dir_sign = (int8_t3)sign(avg_ray_dir);
    */
    
    SparseVoxelOctree svo = SparseVoxelOctree(svo_indices_buffer, svo_bitmasks_buffer);

    // calculate a "coarse" distance for the entire wave
    //bool hit = any_hit_trace(ray_pos, ray_dir, svo);
    //float min_distance = 999999;
    //aabb_test_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0, 30, min_distance);

    /*
    // trace the main ray, but now advance it by our coarse estimate
    half3 inv_dir = (half3)1.0 / abs(ray_dir);
    int8_t3 dir_sign = (int8_t3)sign(ray_dir);
    DdaTraversalOutput output = dda_trace_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, min_distance, 0, 9999999);
    */
    uint clock2 = getRealtimeClock().x;
    
    //ray_outputs[id.x] = RayOutput(min_distance / 1024 + (output.hit ? 0.5 : 0));
    //ray_outputs[id.x] = RayOutput((clock2 - clock1) / (4096.0  * 8) + (output.hit ? 0.001 : 0) + min_distance * 0.000001);
    //ray_outputs[id.x] = RayOutput(abs((min_distance % 4096) / 4096));
    
    //ray_outputs[id.x] = RayOutput(hit ? 1 : 0);
    //ray_outputs[id.x] = RayOutput(hit ? 1 : 0);

    /*
    float min_distance2 = 999999;
    aabb_test_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0, min_distance2);
    */
    //bool output = aabb_test_recurse<DEPTH-1>(1, svo_bitmasks_buffer[0], ray_dir, ray_pos, inv_dir, dir_sign, 0);
    //ray_outputs[id.x] = RayOutput(output, output2);
}