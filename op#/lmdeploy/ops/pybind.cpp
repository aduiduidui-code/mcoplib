// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include "cache.h"
#include "lm_ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {



    // Attention ops
    // Compute the attention between an input query and the cached
    // keys/values using PagedAttention.
    m.def("paged_attention_v1",
            &paged_attention_v1,
            "paged_attention_v1("
            "    Tensor! out, Tensor query, Tensor key_cache,"
            "    Tensor value_cache, int num_kv_heads, float scale,"
            "    Tensor block_tables, Tensor seq_lens, int block_size,"
            "    int max_seq_len, Tensor? alibi_slopes,"
            "    str kv_cache_dtype, float k_scale, float v_scale,"
            "    int tp_rank, int blocksparse_local_blocks,"
            "    int blocksparse_vert_stride, int blocksparse_block_size,"
            "    int blocksparse_head_sliding_step) -> ()");

    // Rotary embedding
    // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
    m.def("lmdeploy_rotary_embedding",
            &lmdeploy_rotary_embedding,
            "lmdeploy_rotary_embedding(Tensor positions, Tensor! query,"
            "                 Tensor! key, int head_size,"
            "                 Tensor cos, Tensor sin,"
            "                 bool is_neox) -> ()");

    // Cache ops
    m.def("reshape_and_cache_new",
            &reshape_and_cache_new,
            "reshape_and_cache_new(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor slot_mapping,"
            "                  str kv_cache_dtype,"
            "                  float kv_scale,"
            "                  float v_scale) -> ()");


    /*************************lmdeploy*************************************/
}
