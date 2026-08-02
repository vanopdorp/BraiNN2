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

torch::Tensor clamp_update_fp16(torch::Tensor dw, torch::Tensor max_norm);
torch::Tensor soft_wta_fp16(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("clamp_update_fp16", &clamp_update_fp16);
    m.def("soft_wta_fp16", &soft_wta_fp16);
}
