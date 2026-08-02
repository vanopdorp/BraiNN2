#include <cuda_fp16.h>

extern "C" __global__
void clamp_update_fp16_kernel(const __half* dw,
                              const __half* max_norm,
                              __half* out,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    __half dw_val = dw[idx];
    __half max_val = max_norm[0];  // broadcast

    // |dw|
    __half abs_val = __habs(dw_val);

    // denom = abs(dw) + 1.0
    __half one = __float2half(1.0f);
    __half denom = __hadd(abs_val, one);

    // scale = max / denom
    __half scale = __hdiv(max_val, denom);

    // out = dw * scale
    out[idx] = __hmul(dw_val, scale);
}
