#language slang 2026
#include "hash.slang"

struct UniformThing {
    float4x4 view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_view_matrix;
    float4x4 inv_projection_matrix;
};

[[vk::binding(0, 0)]]
ParameterBlock<UniformThing> uniform_buffer;

struct VertexInput {
    float3 world;
};

struct VertexOutput {
    float3 color;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;

    float4 world_position = float4(input.world, 1.0);

    float4 t = mul(mul(world_position, uniform_buffer.view_matrix), uniform_buffer.projection_matrix);
    t.y = -t.y;
    t.x = -t.x;

    output.sv_position = t;
    output.color = saturate((world_position.xyz % 64) / 64.0);
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    float3 color = inVert.color;
    return float4(color, 1.0);
}