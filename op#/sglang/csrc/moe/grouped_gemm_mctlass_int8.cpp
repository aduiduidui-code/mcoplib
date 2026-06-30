// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <torch/extension.h>

int32_t get_block_size_m(int32_t batch_size, int32_t m, int32_t n, int32_t k);
void grouped_gemm_mctlass_kernel_int8(
    torch::Tensor const& a, torch::Tensor const& b, torch::Tensor& c, int32_t batch_size, int32_t m, int32_t n, int32_t k,
    torch::Tensor const& seg_indptr, torch::Tensor const& weight_indices, torch::Tensor const& m_num_tiles_indptr, torch::Tensor const& scale_a, torch::Tensor const& scale_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_block_size_m", &get_block_size_m, "get_grouped_gemm_blocksizem.");
  m.def("grouped_gemm_mctlass_kernel_int8", &grouped_gemm_mctlass_kernel_int8, "grouped_gemm_mctlass_int8");
}
