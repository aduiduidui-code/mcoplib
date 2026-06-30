#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> gemma_fused_rmsnorm_rope_no_pack(
    at::Tensor const& qkv,
    at::Tensor const& q_weight,
    at::Tensor const& k_weight,
    at::Tensor const& positions,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_dim,
    double eps,
    at::Tensor const& cos_sin_cache
);
