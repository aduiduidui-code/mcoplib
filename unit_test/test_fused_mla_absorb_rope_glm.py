import os
import time
import torch
import torch.nn.functional as F
from torch import nn
import argparse
import mcoplib.sgl_kernel  

# ============================================================
# Standard GPT-J style rotary embedding (matching kernel implementation)
# ============================================================
def compute_cos_sin_cache(
    max_position_embeddings: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute cos/sin cache for rotary position embedding.

    The cache layout is [max_position_embeddings, head_dim] where:
    - cache[:, :head_dim//2] contains cos values
    - cache[:, head_dim//2:] contains sin values

    This matches the format expected by the CUDA kernel.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))
    t = torch.arange(max_position_embeddings, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat([cos, sin], dim=-1)
    return cache


def torch_rotary_emb_gptj_style(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch reference implementation of GPT-J style rotary position embedding.

    GPT-J style rotates pairs of elements:
    - out[2i] = x[2i] * cos - x[2i+1] * sin
    - out[2i+1] = x[2i+1] * cos + x[2i] * sin

    Args:
        x: Input tensor of shape [..., head_dim], e.g., [q_len, num_heads, head_dim]
        cos_sin_cache: Cache of shape [max_pos, head_dim] containing [cos, sin]
        positions: Position indices of shape [q_len]

    Returns:
        Rotated tensor of same shape as x
    """
    head_dim = x.shape[-1]

    # Get cos/sin for each position
    cos_sin = cos_sin_cache[positions]  # [q_len, head_dim]
    cos = cos_sin[..., :head_dim // 2]  # [q_len, head_dim//2]
    sin = cos_sin[..., head_dim // 2:]  # [q_len, head_dim//2]

    # Reshape cos/sin to broadcast with input tensor
    # x shape: [q_len, num_heads, head_dim]
    # We need cos/sin shape: [q_len, 1, head_dim//2] for proper broadcasting
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # Interleave cos and sin for GPT-J style
    # x1 = x[..., ::2], x2 = x[..., 1::2]
    x1 = x[..., ::2]  # Even indices: [..., head_dim//2]
    x2 = x[..., 1::2]  # Odd indices: [..., head_dim//2]

    # Apply rotation
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    # Interleave output
    out = torch.stack([o1, o2], dim=-1).flatten(-2)
    return out


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.variance_epsilon = eps  # Changed to 1e-6 to match kernel
        self.weight = nn.Parameter(torch.ones(hidden_size).to(torch.bfloat16))

    def forward(
        self,
        x: torch.Tensor,
        residual= None,
    ):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

def fused_forward_absorb(
    q:torch.Tensor, # [bs, 128, 192], dtype=bf16
    w_kc:torch.Tensor, # [128, 128, 512], dtype=bf16
    latent_cache:torch.Tensor, # [bs, 576], dtype=bf16
    cos_sin_cache:torch.Tensor,  # [max_position_embeddings, 64], dtype=float32
    positions:torch.Tensor, # [bs], dtype=int64
    norm_weight:torch.Tensor, # [512], dtype=bf16
    q_input:torch.Tensor, # [bs, 128, 576], dtype=bf16
    k_input:torch.Tensor, # [bs, 1, 576], dtype=bf16
    v_input:torch.Tensor, # [bs, 1, 512]
    q_len:int, #16
    num_local_heads:int, #128,
    kv_lora_rank:int, # 512
    qk_rope_head_dim:int, #64
    qk_nope_head_dim:int, #128
):
    out = torch.ops.sgl_kernel.fused_mla_absorb_rotary_emb(q, w_kc, latent_cache, cos_sin_cache, positions, norm_weight, q_input, k_input, v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
    if out != 0:
        print("Failed to call fusedMLA.[fused_forward_absorb]")
    return q_input, k_input, v_input

def mla_absorb_rotary_emb(
    kv_a_layernorm,
    cos_sin_cache,  # Standard format: [max_pos, head_dim] with [cos, sin]
    q:torch.Tensor, # [bs, 128, 192], dtype=bf16
    w_kc:torch.Tensor, # [128, 128, 512], dtype=bf16
    latent_cache:torch.Tensor, # [bs, 576], dtype=bf16
    positions:torch.Tensor, # [bs], dtype=int64
    q_input:torch.Tensor, # [bs, 128, 576], dtype=bf16
    k_input:torch.Tensor, # [bs, 1, 576], dtype=bf16
    v_input:torch.Tensor, # [bs, 1, 512]
    q_len:int, # 16
    num_local_heads:int, # 128,
    kv_lora_rank:int, # 512
    qk_rope_head_dim:int, # 64
    qk_nope_head_dim:int, # 128
):
    """
    Reference PyTorch implementation that matches the CUDA kernel logic.

    This implementation uses the same cos_sin_cache format and GPT-J style
    rotary embedding as the kernel, ensuring numerical consistency.
    """
    # Step 1: BMM - Compute q_nope @ w_kc
    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)
    q_input[..., : kv_lora_rank] = q_nope_out.transpose(0, 1)

    # Step 2: RMS Norm on latent_cache
    v_input = latent_cache[..., : kv_lora_rank]
    v_input = kv_a_layernorm(v_input.contiguous()).unsqueeze(1)

    # Step 3: Prepare k_input
    k_input = latent_cache.unsqueeze(1)
    k_input[..., : kv_lora_rank] = v_input

    # Step 4: Apply rotary embedding using GPT-J style (matching kernel)
    k_pe = k_input[..., kv_lora_rank :]

    # Apply GPT-J style rotary embedding
    q_pe_rotated = torch_rotary_emb_gptj_style(q_pe, cos_sin_cache, positions)
    k_pe_rotated = torch_rotary_emb_gptj_style(k_pe.squeeze(1), cos_sin_cache, positions).unsqueeze(1)

    # Step 5: Store rotated results
    q_input[..., kv_lora_rank :] = q_pe_rotated
    k_input[..., kv_lora_rank :] = k_pe_rotated

    return q_input, k_input, v_input

with_profile=False

def show_error(golden, v, tag="DIFF ERROR"):
    errors = torch.abs(golden - v)

    errors_max = torch.max(errors)
    errors_ave = torch.sum(errors) / errors.numel()

    max_idx_flat = torch.argmax(errors)
    max_idx = torch.unravel_index(max_idx_flat, errors.shape)

    golden_val = golden[max_idx]
    v_val = v[max_idx]

    print(f"{tag}: error_max={errors_max}, error_ave={errors_ave}, max_error_idx={max_idx}")
    print(f"golden[{max_idx}]={golden_val}, v[{max_idx}]={v_val}")

def print_profiler_summary(prof, max_key_len=50):
    events = prof.key_averages()
    events = sorted(events, key=lambda x: (x.device_time_total / x.count) if x.count > 0 else 0, reverse=True)

    print(f"{'Name':<{max_key_len}} | {'CPU Time Avg (us)':>20} | {'CUDA Time Avg (us)':>20} | {'Count':>10}")

    total_cpu_time = 0.0
    total_cuda_time = 0.0

    for evt in events:
        if evt.count == 0:
            continue

        cpu_time_avg = evt.cpu_time_total / evt.count
        cuda_time_avg = evt.device_time_total / evt.count
        key_str = evt.key

        if len(key_str) > max_key_len:
            key_str = key_str[:max_key_len-3] + '...'

        print(f"{key_str.ljust(max_key_len)} | {cpu_time_avg:20.2f} | {cuda_time_avg:20.2f} | {evt.count:10}")

        total_cpu_time += cpu_time_avg
        total_cuda_time += cuda_time_avg

    print("-" * (max_key_len + 55))
    print(f"{'Total'.ljust(max_key_len)} | {total_cpu_time:20.2f} | {total_cuda_time:20.2f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="profile", choices=["profile", "acc"])
    args = parser.parse_args()

    # MLA parameters (matching GLM5 configuration)
    q_len = 32
    num_local_heads = 4
    kv_lora_rank = 512
    qk_nope_head_dim = 192
    qk_rope_head_dim = 64
    hidden_size = 6144

    # RMS norm with eps=1e-6 to match kernel
    kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-6).cuda()
    max_position_embeddings = 4096  # Standard size for testing
    rope_theta = 10000  # Standard base

    # Create standard format cos_sin_cache (matching kernel expectation)
    # Format: [max_position_embeddings, qk_rope_head_dim]
    # with cos in first half and sin in second half
    cos_sin_cache = compute_cos_sin_cache(
        max_position_embeddings,
        qk_rope_head_dim,
        base=rope_theta,
        device=torch.device("cuda"),
        dtype=torch.float32  # Kernel expects float32
    )
    print(f"cos_sin_cache shape: {cos_sin_cache.shape}, dtype: {cos_sin_cache.dtype}")
    print(f"cos_sin_cache format: cos[:32], sin[32:64] for each position")

    for q_len in [64]:
        print(f"\n\n================ Profiling q_len={q_len} ================")
        q = (torch.rand(q_len, num_local_heads, qk_nope_head_dim+qk_rope_head_dim, dtype=torch.bfloat16).cuda() - 0.5)/10
        w_kc = (torch.rand(num_local_heads, qk_nope_head_dim, kv_lora_rank, dtype=torch.bfloat16).cuda() - 0.5) / 10

        shape = (q_len, kv_lora_rank+qk_rope_head_dim)
        strides = (576, 1)  # contiguous stride for GLM5
        storage_size = (shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1] + 1
        latent_cache_storage = (torch.rand(storage_size, dtype=torch.bfloat16).cuda() - 0.5) / 10
        latent_cache = torch.as_strided(latent_cache_storage, size=shape, stride=strides)
        latent_cache2_storage = latent_cache_storage.clone().detach()
        latent_cache2 = torch.as_strided(latent_cache2_storage, size=shape, stride=strides)

        q_input = torch.zeros(q_len, num_local_heads, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        k_input = torch.zeros(q_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        v_input = torch.zeros(q_len, 1, kv_lora_rank, dtype=torch.bfloat16).cuda()
        fused_q_input = torch.zeros(q_len, num_local_heads, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        fused_k_input = torch.zeros(q_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        fused_v_input = torch.zeros(q_len, 1, kv_lora_rank, dtype=torch.bfloat16).cuda()

        positions = torch.arange(0, q_len, dtype=torch.int64).cuda()  # Start from 0

        print("w_kc stride:", w_kc.stride())
        print("latent_cache stride:", latent_cache.stride())
        print("latent_cache2 stride:", latent_cache2.stride())

        # Run PyTorch reference implementation (using standard GPT-J style rotary)
        legacy_q_input, legacy_k_input, legacy_v_input = mla_absorb_rotary_emb(
            kv_a_layernorm, cos_sin_cache,
            q, w_kc, latent_cache, positions,
            q_input, k_input, v_input,
            q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim
        )

        # Run CUDA kernel
        fused_forward_absorb(
            q, w_kc, latent_cache2, cos_sin_cache, positions, kv_a_layernorm.weight,
            fused_q_input, fused_k_input, fused_v_input,
            q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim
        )

        show_error(legacy_q_input, fused_q_input, "DIFF ERROR OF Q_INPUT")
        show_error(legacy_k_input, fused_k_input, "DIFF ERROR OF K_INPUT")
        show_error(legacy_v_input, fused_v_input, "DIFF ERROR OF V_INPUT")

        # Additional: Check BMM separately
        print("\n--- BMM Verification ---")
        q_nope = q[..., :qk_nope_head_dim]
        q_nope_out_torch = torch.bmm(q_nope.transpose(0, 1).float(), w_kc.float())
        q_nope_out_torch = q_nope_out_torch.transpose(0, 1).bfloat16()

        # Compare with CUDA result
        bmm_error = torch.abs(q_nope_out_torch - fused_q_input[..., :kv_lora_rank])
        print(f"BMM max error: {bmm_error.max().item()}, avg error: {bmm_error.mean().item()}")