#language slang 2026

[[vk::binding(0, 0)]]
readonly RWTexture2D<float4> pre_compose;

[[vk::binding(1, 0)]]
RWTexture2D<float4> render_target;
[shader("compute")]
[numthreads(8, 8, 1)]
void write_render_texture(uint3 id: SV_DispatchThreadID, uniform float2 screen, uniform matrix<float,4,4> mat, uniform float4 position, uniform float4 sun, uniform uint debug_type, uniform float time) {
    render_target[id.xy] = pre_compose[id.xy];
}