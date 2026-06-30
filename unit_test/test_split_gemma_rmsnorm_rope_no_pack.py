import pytest
import torch
from flashinfer import gemma_rmsnorm
import mcoplib.op as mcops
from vllm import _custom_ops as vllm_ops
from torch.profiler import profile, ProfilerActivity


def rope(query, key, head_size, positions, cos_sin_cache):
    is_neox_style = True

    vllm_ops.rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style,
    )

    return query, key


def ref_gemma_fused_rmsnorm_rope_no_pack(qkv, q_weight, k_weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache):
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q_shape, k_shape = q.shape, k.shape
    q = gemma_rmsnorm(q.reshape(-1, head_dim), q_weight, eps).reshape(q_shape)
    k = gemma_rmsnorm(k.reshape(-1, head_dim), k_weight, eps).reshape(k_shape)

    q, k = rope(q, k, head_dim, positions, cos_sin_cache)
    return q, k, v


def cosine_similarity(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def test_gemma_fused_rmsnorm_rope_no_pack(num_tokens, total_num_heads, tp_size, dtype):
    eps = 1e-6
    head_dim = 128
    num_attention_groups = 8
    max_position_embedding = 262144

    total_num_kv_heads = num_attention_groups
    num_heads = total_num_heads // tp_size
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    torch.manual_seed(42)

    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=dtype, device="cuda")
    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    positions = torch.arange(0, num_tokens, dtype=torch.int64, device="cuda")
    cos_sin_cache = torch.randn(max_position_embedding, head_dim, dtype=dtype, device="cuda")

    assert hasattr(mcops, "gemma_fused_rmsnorm_rope_no_pack"), "mcoplib.op 中没有 gemma_fused_rmsnorm_rope_no_pack，请先确认 pybind 注册和编译是否成功"

    q_ref, k_ref, v_ref = ref_gemma_fused_rmsnorm_rope_no_pack(
        qkv,
        q_weight,
        k_weight,
        positions,
        q_size,
        kv_size,
        head_dim,
        eps,
        cos_sin_cache,
    )

    q_out, k_out, v_out = mcops.gemma_fused_rmsnorm_rope_no_pack(
        qkv,
        q_weight,
        k_weight,
        positions,
        q_size,
        kv_size,
        head_dim,
        eps,
        cos_sin_cache,
    )
    torch.cuda.synchronize()

    q_sim = cosine_similarity(q_out, q_ref)
    k_sim = cosine_similarity(k_out, k_ref)
    v_sim = cosine_similarity(v_out, v_ref)

    print("\naccuracy result:")
    print(f"q similarity: {q_sim:.6f}")
    print(f"k similarity: {k_sim:.6f}")
    print(f"v similarity: {v_sim:.6f}")

    assert q_sim > 0.999, f"q mismatch, similarity={q_sim}"
    assert k_sim > 0.999, f"k mismatch, similarity={k_sim}"
    assert v_sim > 0.999, f"v mismatch, similarity={v_sim}"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(1000):
            q, k, v = mcops.gemma_fused_rmsnorm_rope_no_pack(
                qkv,
                q_weight,
                k_weight,
                positions,
                q_size,
                kv_size,
                head_dim,
                eps,
                cos_sin_cache,
            )
        torch.cuda.synchronize()

    print("\nperf result: ")
    print("iterations: 1000")
    table = prof.key_averages().table(sort_by="device_time_total", row_limit=20)
    print(table)


if __name__ == "__main__":
    dtype = torch.bfloat16
    for tp_size, num_tokens in (
        (8, 2048),
        (8, 4096),
        (8, 8192),
        (1, 1),
        (1, 16),
        (1, 32),
    ):
        for total_num_heads in [96]:  # full attention: 64, sliding attention: 96
            print(f"Testing: {tp_size=}, {num_tokens=}, {total_num_heads=}")
            test_gemma_fused_rmsnorm_rope_no_pack(num_tokens, total_num_heads, tp_size, dtype)
