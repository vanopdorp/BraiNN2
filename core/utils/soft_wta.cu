#include <cuda_fp16.h>

extern "C" __global__
void relu_and_sum_fp16_kernel(const __half* x,
                              __half* relu_out,
                              float* sum_out,
                              int n) {
    __shared__ float shmem[256]; 

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float local_sum = 0.0f;

    if (idx < n) {
        __half x_val = x[idx];
        // ReLU in fp16
        __half zero = __float2half(0.0f);
        __half relu_val = __hgt(x_val, zero) ? x_val : zero;
        relu_out[idx] = relu_val;

        local_sum = __half2float(relu_val);
    }

    shmem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_out, shmem[0]);
    }
}

extern "C" __global__
void soft_wta_fp16_kernel(const __half* relu,
                          __half* out,
                          float sum,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float denom = sum + 1.0f;
    __half relu_val = relu[idx];
    float relu_f = __half2float(relu_val);
    float out_f = relu_f / denom;
    out[idx] = __float2half(out_f);
}
