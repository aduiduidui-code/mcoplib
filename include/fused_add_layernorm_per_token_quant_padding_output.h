#include <ATen/ATen.h>

void add_rms_norm_dynamic_per_token_quant_padding_output(at::Tensor& output,
                                                         at::Tensor& output_rms,
                                                         at::Tensor& output_quant_int8,
                                                         at::Tensor& out_scales,
                                                         at::Tensor const& input,
                                                         at::Tensor& residual,
                                                         at::Tensor const& weight, 
                                                         const int pad_size, 
                                                         const float epsilon);