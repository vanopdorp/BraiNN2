#include <cuda_fp16.h>

extern "C" {
    __global__ void clamp_update_fp16_kernel(const __half*, const __half*, __half*, int);
    __global__ void relu_and_sum_fp16_kernel(const __half*, __half*, float*, int);
    __global__ void soft_wta_fp16_kernel(const __half*, __half*, float, int);
}

#include <ATen/ATen.h>
#include <torch/types.h>
#include <torch/serialize.h>
#include <torch/library.h>
#include <cuda_fp16.h>


void clamp_update_fp16_launcher(torch::Tensor dw,
                                torch::Tensor max_norm,
                                torch::Tensor out) {
    int n = dw.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    clamp_update_fp16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(dw.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(max_norm.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n
    );
}

torch::Tensor clamp_update_fp16(torch::Tensor dw, torch::Tensor max_norm) {
    auto dw16 = dw.to(at::kHalf).contiguous();
    auto max16 = max_norm.to(at::kHalf).contiguous();
    auto out = torch::empty_like(dw16);
    clamp_update_fp16_launcher(dw16, max16, out);
    return out;
}

torch::Tensor soft_wta_fp16(torch::Tensor x) {
    auto x16 = x.to(at::kHalf).contiguous();
    auto relu = torch::empty_like(x16);
    auto sum_buf = torch::zeros({1}, torch::dtype(torch::kFloat32).device(x.device()));

    int n = x16.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    relu_and_sum_fp16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(x16.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(relu.data_ptr<at::Half>()),
        sum_buf.data_ptr<float>(),
        n
    );

    float sum = sum_buf.item<float>();

    auto out = torch::empty_like(x16);
    soft_wta_fp16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(relu.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        sum,
        n
    );

    return out;
}


