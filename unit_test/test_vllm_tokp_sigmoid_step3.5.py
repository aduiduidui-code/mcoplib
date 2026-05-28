import pytest
import torch

# from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
#     fused_topk_bias,
# )
# from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
# from vllm.platforms import current_platform
#import vllm.custom_ops as ops
from torch.profiler import profile, ProfilerActivity
import mcoplib._moe_C

def torch_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.float(), dim=-1)
    else:
        assert scoring_func == "sigmoid"
        scores = torch.sigmoid(gating_output.float())

    if e_score_correction_bias is not None:
        num_experts = gating_output.shape[-1]
        scores_for_choice = scores.view(
            -1, num_experts
        ) + e_score_correction_bias.unsqueeze(0)
        _, topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_weights = scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids

def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    torch.ops._moe_C.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )

    return topk_weights, topk_indices
def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "softmax",
    indices_type: torch.dtype | None = None,
):
    # if not rocm_aiter_ops.is_fused_moe_enabled():
    assert hidden_states.size(0) == gating_output.size(0), (
        "Number of tokens mismatch"
    )

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        raise AssertionError("step3.5 softmax is not supported")
    elif scoring_func == "sigmoid":
        topk_weights, topk_ids = vllm_topk_sigmoid(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
        )
        ############################################
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ],
            record_shapes=True,
        ) as prof:

            for _ in range(1000):
                vllm_topk_sigmoid(
                    topk_weights,
                    topk_ids,
                    token_expert_indices,
                    gating_output,
                    renormalize,
                    e_score_correction_bias,
                )

        torch.cuda.synchronize()

        print(prof.key_averages().table(
            sort_by="self_cuda_time_total"
        ))

        ############################################

        return topk_weights, topk_ids
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


@pytest.mark.parametrize("num_tokens", [1, 16, 32, 1024, 4096,8192]) #
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("num_experts", [288])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("renormalize", [True])
@pytest.mark.parametrize("scoring_func", ["sigmoid"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_topk_bias(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = torch_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        scoring_func=scoring_func,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )

    torch.testing.assert_close(
        topk_weights_ref.to(torch.float32), topk_weights, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(topk_ids_ref.to(torch.int32), topk_ids, atol=0, rtol=0)