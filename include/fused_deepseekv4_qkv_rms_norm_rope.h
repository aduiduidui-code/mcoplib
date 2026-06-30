#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

void fused_rms_norm_rope(
    at::Tensor& q,              // Input & output query tensor (modified in-place)
    at::Tensor& kv,             // Input & output key-value tensor (modified in-place)
    at::Tensor const& positions,      // Position indices
    at::Tensor const& freqs_cis,      // Precomputed frequency values (complex)
    int64_t qk_rope_head_dim,         // RoPE dimension (default: 64)
    double eps,                       // RMS norm epsilon (default: 1e-6)
    c10::optional<at::Tensor> weight_q = c10::nullopt,   // Optional weight for q RMS norm
    c10::optional<at::Tensor> weight_kv = c10::nullopt   // Optional weight for kv RMS norm
);