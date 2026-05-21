#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

/**
 * fused_rms_norm_rope - Fused RMS Norm + Rotary Position Embedding for DeepSeekV4
 *
 * This kernel performs in-place operation:
 * 1. RMS normalization on q and kv
 * 2. Apply RoPE on the last qk_rope_head_dim dimensions
 *
 * Mathematical formula:
 *   q_norm = q * rsqrt(mean(q^2) + eps) * weight_q
 *   kv_norm = kv * rsqrt(mean(kv^2) + eps) * weight_kv
 *   q_rope = apply_rotary_embedding(q_norm[..., -qk_rope_head_dim:])
 *   kv_rope = apply_rotary_embedding(kv_norm[..., -qk_rope_head_dim:])
 *
 * Args (in-place modification):
 *   q: [batch_size, num_heads * head_dim] or [batch_size, num_heads, head_dim], bf16/fp16
 *   kv: [batch_size, head_dim], bf16/fp16
 *   positions: [batch_size], int64
 *   freqs_cis: [max_seq_len, qk_rope_head_dim // 2], complex64
 *   qk_rope_head_dim: dimension for RoPE (default: 64)
 *   eps: epsilon for RMS norm (default: 1e-6)
 *   weight_q: optional weight tensor for q RMS norm [head_dim]
 *   weight_kv: optional weight tensor for kv RMS norm [kv_dim]
 *
 * Returns: void (modifies q and kv in-place)
 */

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