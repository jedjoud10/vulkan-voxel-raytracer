#language slang 2026
#include "sky.slang"

struct UniformThing {
    float4x4 view_matrix;
    float4x4 projection_matrix;
    float4x4 inv_view_matrix;
    float4x4 inv_projection_matrix;
};

[[vk::binding(0, 0)]]
ParameterBlock<UniformThing> uniform_buffer;

[[vk::binding(0, 1)]]
SamplerCube<float4> skybox_sampler;

[[vk::binding(1, 1)]]
Sampler2D<float4> clouds_sampler;

struct VertexOutput {
    float2 uvs;
    float4 sv_position : SV_Position;
};

static float2 positions[6] = float2[](
    float2(-1, -1), // top left
    float2(1, 1), // bottom right
    float2(-1, 1), // bottom left
    
    float2(-1, -1), // top left 
    float2(1, -1), // top right
    float2(1, 1) // bottom right
);



[shader("vertex")]
VertexOutput vertMain(uint vertex_index: SV_VertexID) {
    VertexOutput output;
    float2 p = positions[vertex_index];
    output.sv_position = float4(p, 1.0, 1.0);
    output.uvs = p;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target {
    float2 screen_uvs = inVert.uvs;
    

    float3 sun = 1.0;
    let ndc = float4(screen_uvs, 1, 1);
    float3 ray_dir = mul(ndc, uniform_buffer.inv_projection_matrix).xyz;
    ray_dir = mul(float4(ray_dir.xy, 1, 0), uniform_buffer.inv_view_matrix).xyz;
    ray_dir = -ray_dir;

    float3 ray_pos = 0.0;
    return float4(sample_skybox_and_clouds(ray_pos, ray_dir, clouds_sampler, skybox_sampler), 1.0);
}