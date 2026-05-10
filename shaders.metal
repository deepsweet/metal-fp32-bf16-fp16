#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint iters;
};

// FP32 kernel
kernel void bench_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float x = a[gid];
    float y = b[gid];
    float c = 1.0009765625f;
    #pragma clang loop unroll(disable)
    for (uint i = 0; i < p.iters; ++i) {
        x = fma(x, y, c);
        y = fma(y, c, x);
        c = c + 0.00006103515625f;
    }
    out[gid] = x + y + c;
}

// BF16 kernel (optimised: keep temporaries as float, convert only when storing back)
#if __METAL_VERSION__ >= 310
kernel void bench_bf16(
    device const bfloat *a [[buffer(0)]],
    device const bfloat *b [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float xf = float(a[gid]);
    float yf = float(b[gid]);
    float cf = 1.0f;
    #pragma clang loop unroll(disable)
    for (uint i = 0; i < p.iters; ++i) {
        xf = fma(xf, yf, cf);
        yf = fma(yf, cf, xf);
        cf = cf + 0.00006103515625f;
    }
    out[gid] = bfloat(xf) + bfloat(yf) + bfloat(cf);
}
#endif

// FP16 kernel
kernel void bench_f16(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    half x = a[gid];
    half y = b[gid];
    half c = half(1.0009765625h);
    #pragma clang loop unroll(disable)
    for (uint i = 0; i < p.iters; ++i) {
        x = fma(x, y, c);
        y = fma(y, c, x);
        c = c + half(0.00006103515625h);
    }
    out[gid] = x + y + c;
}
