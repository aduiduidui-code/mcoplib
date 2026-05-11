#include <ATen/ATen.h>

// Gemma RMSNorm variant with optional residual and pack output support
void add_gemma_rms_norm_dynamic_per_token_quant_padding_output(
    at::Tensor& output,
    at::Tensor& output_rms,
    at::Tensor& output_quant_int8,
    at::Tensor& out_scales,
    at::Tensor const& input,
    at::Tensor& residual,
    at::Tensor const& weight,
    const int pad_size,
    const float epsilon,
    const bool bneed_pack);