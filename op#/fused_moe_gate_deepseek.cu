/*
    Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include "../kernel/fused_moe_gate_deepseek.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

namespace deepseek_moe_gate {

    // aux function for nested AT_DISPATCH
    template<class scalar1_t, class scalar2_t, int NUM_SHARED_EXPERTS, int NUM_EXPERTS, int NUM_EXPERT_GROUP, int TOPK = 8, int MOE_TYPE = 0>
    int launch_fused_moe_gate2(
        torch::Tensor& gating_outputs, //[bs, num_experts], dtype=bf16
        torch::Tensor& correction_bias, //[num_experts], dtype=bf16
        torch::Tensor& out_routing_weights, //[bs, num_selected_experts], dtype=float
        torch::Tensor& out_selected_experts, //[bs, num_selected_experts], dtype=int32
        bool renormalize,
        float scale_factor)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_outputs));
        int dev = gating_outputs.get_device();
        int grid = gating_outputs.size(0);
        int block = NUM_EXPERTS;

        AT_DISPATCH_INTEGRAL_TYPES(out_selected_experts.scalar_type(), "moe_gate fused_topk", [&]{                                  \
            fused_moegate_topk<scalar1_t, scalar2_t, scalar_t, NUM_SHARED_EXPERTS, NUM_EXPERTS, NUM_EXPERT_GROUP, TOPK, MOE_TYPE>   \
                <<<grid, block, 0, at::cuda::getCurrentCUDAStream(dev)>>>(  \
                (const scalar1_t*)gating_outputs.data_ptr<scalar1_t>(),     \
                (const scalar2_t*)correction_bias.data_ptr<scalar2_t>(),    \
                (float*)out_routing_weights.data_ptr(),                     \
                (scalar_t*)out_selected_experts.data_ptr(),                 \
                renormalize,                                                \
                scale_factor);                                      \
        });

        return 0;
    }

    // aux function for nested AT_DISPATCH
    template<class scalar1_t, int NUM_SHARED_EXPERTS, int NUM_EXPERTS, int NUM_EXPERT_GROUP, int TOPK = 8, int MOE_TYPE = 0>
    int launch_fused_moe_gate(
        torch::Tensor& gating_outputs, //[bs, num_experts], dtype=bf16
        torch::Tensor& correction_bias, //[num_experts], dtype=bf16
        torch::Tensor& out_routing_weights, //[bs, num_selected_experts], dtype=float
        torch::Tensor& out_selected_experts, //[bs, num_selected_experts], dtype=int32
        bool renormalize,
        float scale_factor)
    {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, correction_bias.scalar_type(), "moe_gate fused_topk", [&]{            \
            launch_fused_moe_gate2<scalar1_t, scalar_t, NUM_SHARED_EXPERTS, NUM_EXPERTS, NUM_EXPERT_GROUP, TOPK, MOE_TYPE>(    \
                gating_outputs, correction_bias, out_routing_weights, out_selected_experts, renormalize, scale_factor); });
        return 0;
    }
}

// 从256个专家中选择top8个专家，输出专家权重和索引。
int fused_moe_gate_deepseek(
    torch::Tensor& gating_outputs, //[bs, num_experts], dtype=bf16
    torch::Tensor& correction_bias, //[num_experts], dtype=bf16
    torch::Tensor& out_routing_weights, //[bs, num_selected_experts], dtype=float
    torch::Tensor& out_selected_experts, //[bs, num_selected_experts], dtype=int32
    int topk,
    bool renormalize,
    int num_expert_group,
    int topk_group,
    std::optional<int> num_fused_shared_experts,
    std::optional<float> routed_scaling_factor,
    std::optional<int> moegate_type = {0}
) {
    DEBUG_TRACE_PARAMS(gating_outputs, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, topk_group, num_fused_shared_experts,routed_scaling_factor,moegate_type);
    DEBUG_DUMP_PARAMS(gating_outputs, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, topk_group, num_fused_shared_experts,routed_scaling_factor,moegate_type);
    int moe_type = 0;
    if (moegate_type.has_value()){
        TORCH_CHECK(((*moegate_type < 2)), "Expected moegate_type == DEEPSEEK(0) or SGLANG(1), but get ", *moegate_type);
        moe_type = (*moegate_type);
    }
    TORCH_CHECK(((topk == 8) || (topk == 9)), "Expected topk = 8, but get topk = ", topk);
    int num_experts = gating_outputs.size(1);
    float scale_factor = 1.0f;
    if (routed_scaling_factor.has_value())
    {
        if(moe_type == 0) //DEEPSEEK
            scale_factor = *routed_scaling_factor;
        else if(moe_type == 1) //SGLANG
            scale_factor = 1.0f / (*routed_scaling_factor);
    }
    int num_shared_experts = (num_fused_shared_experts.has_value() ? *num_fused_shared_experts : 0);
    if(num_shared_experts > 0)
        TORCH_CHECK(((topk >= 9)), "Expected topk >= 9 when fused_num_shared_experts > 0, but get topk = ", topk);
    if(num_shared_experts == 0)
        TORCH_CHECK(((topk == 8)), "Expected topk == 8 when fused_num_shared_experts = 0 or None, but get topk = ", topk);

#define LAUNCH_MOE_GATE(NUM_SHARED_EXPERTS, NUM_EXPERTS, NUM_EXPERT_GROUP, TOPK, MOE_TYPE)  \
    else if (num_shared_experts == NUM_SHARED_EXPERTS && num_experts == NUM_EXPERTS         \
        && num_expert_group == NUM_EXPERT_GROUP && topk == TOPK && moe_type == MOE_TYPE) {  \
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, gating_outputs.scalar_type(), "moe_gate fused_topk", [&]{ \
            deepseek_moe_gate::launch_fused_moe_gate<scalar_t, NUM_SHARED_EXPERTS, NUM_EXPERTS, NUM_EXPERT_GROUP, TOPK, MOE_TYPE>(gating_outputs,        \
                correction_bias, out_routing_weights, out_selected_experts, renormalize, scale_factor); }); \
    }

    if (false) {
    }
    LAUNCH_MOE_GATE(0, 256, 8, 8, 0)
    LAUNCH_MOE_GATE(0, 320, 1, 8, 0)
    LAUNCH_MOE_GATE(0, 384, 1, 8, 0)
    LAUNCH_MOE_GATE(0, 448, 1, 8, 0)
    LAUNCH_MOE_GATE(0, 256, 8, 8, 1)
    LAUNCH_MOE_GATE(0, 320, 1, 8, 1)
    LAUNCH_MOE_GATE(0, 384, 1, 8, 1)
    LAUNCH_MOE_GATE(0, 448, 1, 8, 1)
    LAUNCH_MOE_GATE(1, 256, 8, 9, 1)
    LAUNCH_MOE_GATE(1, 320, 1, 9, 1)
    LAUNCH_MOE_GATE(1, 384, 1, 9, 1)
    LAUNCH_MOE_GATE(1, 448, 1, 9, 1)
    LAUNCH_MOE_GATE(2, 256, 8, 9, 1)
    LAUNCH_MOE_GATE(2, 320, 1, 9, 1)
    LAUNCH_MOE_GATE(2, 384, 1, 9, 1)
    LAUNCH_MOE_GATE(2, 448, 1, 9, 1)
    LAUNCH_MOE_GATE(3, 256, 8, 9, 1)
    LAUNCH_MOE_GATE(3, 320, 1, 9, 1)
    LAUNCH_MOE_GATE(3, 384, 1, 9, 1)
    LAUNCH_MOE_GATE(3, 448, 1, 9, 1)
    LAUNCH_MOE_GATE(4, 256, 8, 9, 1)
    LAUNCH_MOE_GATE(4, 320, 1, 9, 1)
    LAUNCH_MOE_GATE(4, 384, 1, 9, 1)
    LAUNCH_MOE_GATE(4, 448, 1, 9, 1)
    else {
        TORCH_CHECK(false, "Invalid arguments with TOPK = ", topk, ", NUM_EXPERT_GROUP = ", num_expert_group, ", TOPK_GROUP = ", topk_group, ", NUM_EXPERTS = ", num_experts);
        return 1;
    }
    return 0;
}
