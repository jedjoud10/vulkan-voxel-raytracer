#language slang 2026

[[vk::binding(0, 0)]]
Sampler2D<float4> previous_mip;

[[vk::binding(1, 0)]]
RWTexture2D<float4> next_mip;

[shader("compute")]
[numthreads(8, 8, 1)]
void write_render_texture(uint3 id: SV_DispatchThreadID) {
}