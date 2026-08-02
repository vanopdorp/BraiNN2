#include <torch/extension.h>

torch::Tensor clamp_update_fp16(torch::Tensor dw, torch::Tensor max_norm);
torch::Tensor soft_wta_fp16(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("clamp_update_fp16", &clamp_update_fp16);
    m.def("soft_wta_fp16", &soft_wta_fp16);
}
