#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

void gemma_fused_rmsnorm_rope(
    at::Tensor& qkv,              
    at::Tensor const& weight,
    at::Tensor const& positions,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_dim,         
    double eps,              
    at::Tensor const& cos_sin_cache
);