// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

#include <torch/library.h>

#include <optional>

void paged_attention_v1(torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache, torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
                        torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len,
                        const c10::optional<torch::Tensor>& alibi_slopes, const std::string& kv_cache_dtype, double k_scale, double v_scale,
                        const int64_t tp_rank, const int64_t blocksparse_local_blocks, const int64_t blocksparse_vert_stride,
                        const int64_t blocksparse_block_size, const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits, torch::Tensor& tmp_out, torch::Tensor& query,
                        torch::Tensor& key_cache, torch::Tensor& value_cache, int64_t num_kv_heads, double scale, torch::Tensor& block_tables,
                        torch::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
                        const std::string& kv_cache_dtype, double k_scale, double v_scale, const int64_t tp_rank, const int64_t blocksparse_local_blocks,
                        const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size, const int64_t blocksparse_head_sliding_step);


void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon);

void lmdeploy_rotary_embedding(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, int64_t head_size, torch::Tensor& cos, torch::Tensor& sin,
                      bool is_neox);

void batched_rotary_embedding(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, int64_t head_size, torch::Tensor& cos_sin_cache, bool is_neox,
                              int64_t rot_dim, torch::Tensor& cos_sin_cache_offsets);

