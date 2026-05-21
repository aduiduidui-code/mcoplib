import pytest
import torch
import time
import math
import mcoplib.op as ops
from flashinfer import gemma_rmsnorm
from vllm import _custom_ops as vllm_ops
# from torch.profiler import profile, ProfilerActivity
#call gemma_fused_rmsnorm_rope cuda op
#ops.gemma_fused_rmsnorm_rope(xxx)

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

def torch_gemma_fused_rmsnorm_rope(qkv, weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache):
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q_shape, k_shape = q.shape, k.shape
    q = gemma_rmsnorm(q.reshape(-1, head_dim), weight, eps).reshape(q_shape)
    k = gemma_rmsnorm(k.reshape(-1, head_dim), weight, eps).reshape(k_shape)

    q, k = rope(q, k, head_dim, positions, cos_sin_cache)
    output = torch.cat([q, k, v], dim=-1)
    return output
# ============================================================================
# Torch Reference Implementation for Gemma RMSNorm + Neox RoPE
# ============================================================================

def torch_gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Gemma-style RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight)
    Note: Gemma uses (1 + weight) instead of just weight
    """
    # x: [num_tokens * num_heads, head_dim]
    # weight: [head_dim]
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    # Gemma special: (1 + weight)
    return x_normed * (1.0 + weight)


def torch_neox_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Neox-style Rotary Position Embedding

    Neox layout:
    - For dimension i in [0, head_dim/2):
      x'_i = x_i * cos - x_{i+head_dim/2} * sin
      x'_{i+head_dim/2} = x_i * sin + x_{i+head_dim/2} * cos

    Args:
        q, k: [num_tokens, num_heads, head_dim]
        positions: [num_tokens]
        cos_sin_cache: [max_pos, head_dim] - contains cos values followed by sin values
                       cos values: cos_sin_cache[:, 0:head_dim/2]
                       sin values: cos_sin_cache[:, head_dim/2:]
    """
    num_tokens = q.shape[0]
    half_dim = head_dim // 2

    # Get cos and sin for each position
    # cos_sin_cache shape: [max_pos, head_dim]
    # For position p: cos = cos_sin_cache[p, 0:half_dim], sin = cos_sin_cache[p, half_dim:]
    cos = cos_sin_cache[positions, :half_dim]  # [num_tokens, half_dim]
    sin = cos_sin_cache[positions, half_dim:]  # [num_tokens, half_dim]

    # Expand for broadcasting with multi-head tensors
    # q, k shape: [num_tokens, num_heads, head_dim]
    cos = cos.unsqueeze(1)  # [num_tokens, 1, half_dim]
    sin = sin.unsqueeze(1)  # [num_tokens, 1, half_dim]

    # Neox rotation
    # Split into left and right halves
    q_left = q[..., :half_dim]   # [num_tokens, num_heads, half_dim]
    q_right = q[..., half_dim:]  # [num_tokens, num_heads, half_dim]
    k_left = k[..., :half_dim]
    k_right = k[..., half_dim:]

    # Apply rotation
    # x'_i = x_i * cos - x_{i+d/2} * sin (for left half)
    # x'_{i+d/2} = x_i * sin + x_{i+d/2} * cos (for right half)
    q_rotated_left = q_left * cos - q_right * sin
    q_rotated_right = q_left * sin + q_right * cos
    k_rotated_left = k_left * cos - k_right * sin
    k_rotated_right = k_left * sin + k_right * cos

    # Concatenate back
    q_out = torch.cat([q_rotated_left, q_rotated_right], dim=-1)
    k_out = torch.cat([k_rotated_left, k_rotated_right], dim=-1)

    return q_out, k_out


def torch_gemma_fused_rmsnorm_rope_ref(
    qkv: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    q_size: int,
    kv_size: int,
    head_dim: int,
    eps: float,
    cos_sin_cache: torch.Tensor
) -> torch.Tensor:
    """
    Torch reference implementation for gemma_fused_rmsnorm_rope

    This function:
    1. Split qkv into q, k, v
    2. Apply Gemma RMSNorm to q and k (not v)
    3. Apply Neox-style RoPE to q and k
    4. Concatenate back to qkv

    Args:
        qkv: [num_tokens, q_size + 2*kv_size]
        weight: [head_dim] RMSNorm weight
        positions: [num_tokens]
        q_size: size of Q (num_heads * head_dim)
        kv_size: size of K/V (num_kv_heads * head_dim)
        head_dim: head dimension
        eps: RMSNorm epsilon
        cos_sin_cache: [max_pos, head_dim]

    Returns:
        output: [num_tokens, q_size + 2*kv_size] same shape as qkv
    """
    # Split qkv
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    num_tokens = qkv.shape[0]
    num_heads = q_size // head_dim
    num_kv_heads = kv_size // head_dim

    # Reshape for RMSNorm: [num_tokens * num_heads, head_dim]
    q_reshaped = q.reshape(num_tokens * num_heads, head_dim)
    k_reshaped = k.reshape(num_tokens * num_kv_heads, head_dim)

    # Apply Gemma RMSNorm (using float for precision)
    q_normed = torch_gemma_rmsnorm(q_reshaped.float(), weight.float(), eps)
    k_normed = torch_gemma_rmsnorm(k_reshaped.float(), weight.float(), eps)

    # Reshape back: [num_tokens, num_heads, head_dim]
    q_normed = q_normed.reshape(num_tokens, num_heads, head_dim)
    k_normed = k_normed.reshape(num_tokens, num_kv_heads, head_dim)

    # Apply Neox RoPE
    q_rope, k_rope = torch_neox_rope(q_normed, k_normed, positions, cos_sin_cache, head_dim)

    # Flatten back to original shape: [num_tokens, q_size] and [num_tokens, kv_size]
    q_out = q_rope.reshape(num_tokens, q_size).to(qkv.dtype)
    k_out = k_rope.reshape(num_tokens, kv_size).to(qkv.dtype)
    v_out = v  # v is unchanged

    # Concatenate
    output = torch.cat([q_out, k_out, v_out], dim=-1)

    return output


# ============================================================================
# Cosine Similarity for Accuracy Verification
# ============================================================================

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two tensors
    Returns: similarity value in range [-1, 1], expected > 0.9999
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    dot_product = (a_flat * b_flat).sum()
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return similarity.item()


# ============================================================================
# Bandwidth Calculation
# ============================================================================

def calculate_bandwidth(
    num_tokens: int,
    q_size: int,
    kv_size: int,
    head_dim: int,
    dtype: torch.dtype,
    kernel_time_ms: float,
    num_iterations: int
) -> float:
    """
    Calculate effective bandwidth for the kernel

    Memory operations:
    1. Read qkv: [num_tokens, q_size + 2*kv_size]
    2. Read weight: [head_dim]
    3. Read cos_sin_cache: [num_tokens, head_dim] (positions determine access)
    4. Read positions: [num_tokens]
    5. Write qkv (in-place): same as read

    Total data accessed:
    - qkv_size = num_tokens * (q_size + 2*kv_size) * element_size
    - weight_size = head_dim * element_size (small, cached)
    - cos_sin_cache_size = num_tokens * head_dim * element_size
    - positions_size = num_tokens * 8 (int64)

    Since it's in-place, we count both read and write for qkv
    """
    element_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    qkv_bytes = num_tokens * (q_size + 2 * kv_size) * element_size
    weight_bytes = head_dim * element_size
    cos_sin_cache_bytes = num_tokens * head_dim * element_size
    positions_bytes = num_tokens * 8

    # Total bytes: read qkv + read weight + read cos_sin + read positions + write qkv
    total_bytes_per_iter = qkv_bytes * 2 + weight_bytes + cos_sin_cache_bytes + positions_bytes

    total_bytes = total_bytes_per_iter * num_iterations
    total_time_s = kernel_time_ms / 1000.0

    bandwidth_gbps = (total_bytes / total_time_s) / (1024 ** 3)

    return bandwidth_gbps


# ============================================================================
# Performance Benchmark
# ============================================================================

def benchmark_cuda_kernel(
    qkv: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    q_size: int,
    kv_size: int,
    head_dim: int,
    eps: float,
    cos_sin_cache: torch.Tensor,
    num_iterations: int = 1000,
    warmup: int = 10
) -> tuple[float, float]:
    """
    Benchmark CUDA kernel and return avg time per iteration and bandwidth

    Returns:
        (avg_time_ms, bandwidth_gbps)
    """
    # Warmup
    for _ in range(warmup):
        ops.gemma_fused_rmsnorm_rope(
            qkv, weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache
        )
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(num_iterations):
        ops.gemma_fused_rmsnorm_rope(
            qkv, weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000.0
    avg_time_ms = total_time_ms / num_iterations

    bandwidth_gbps = calculate_bandwidth(
        qkv.shape[0], q_size, kv_size, head_dim, qkv.dtype, total_time_ms, num_iterations
    )

    return avg_time_ms, bandwidth_gbps


def benchmark_torch_reference(
    qkv: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    q_size: int,
    kv_size: int,
    head_dim: int,
    eps: float,
    cos_sin_cache: torch.Tensor,
    num_iterations: int = 100,
    warmup: int = 5
) -> float:
    """
    Benchmark Torch reference implementation

    Returns:
        avg_time_ms per iteration
    """
    # Warmup
    for _ in range(warmup):
        torch_gemma_fused_rmsnorm_rope(
            qkv.clone(), weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache
        )
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = torch_gemma_fused_rmsnorm_rope(
            qkv.clone(), weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000.0
    avg_time_ms = total_time_ms / num_iterations

    return avg_time_ms


# ============================================================================
# Unit Test
# ============================================================================

def test_gemma_fused_rmsnorm_rope(
    num_tokens: int,
    total_num_heads: int,
    tp_size: int,
    dtype: torch.dtype,
    verbose: bool = True
):
    """
    Test gemma_fused_rmsnorm_rope CUDA kernel

    1. Accuracy verification: cosine similarity > 0.9999
    2. Performance comparison: CUDA vs Torch reference
    3. Bandwidth calculation
    """
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

    # Create input tensors
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=dtype, device="cuda")
    weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    positions = torch.arange(0, num_tokens, dtype=torch.int64, device="cuda")
    cos_sin_cache = torch.randn(max_position_embedding, head_dim, dtype=dtype, device="cuda")

    # Prepare copies for testing
    qkv_cuda = qkv.clone()
    qkv_torch_ref = qkv.clone()

    # =====================
    # Step 1: Run CUDA kernel (in-place)
    # =====================
    ops.gemma_fused_rmsnorm_rope(
        qkv_cuda, weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache
    )

    # =====================
    # Step 2: Run Torch reference
    # =====================
    qkv_torch_out = torch_gemma_fused_rmsnorm_rope(
        qkv_torch_ref, weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache
    )

    # =====================
    # Step 3: Accuracy verification (Cosine Similarity)
    # =====================
    # Compare Q and K parts (V is unchanged in both)
    q_cuda = qkv_cuda[:, :q_size]
    k_cuda = qkv_cuda[:, q_size:q_size + kv_size]
    q_torch = qkv_torch_out[:, :q_size]
    k_torch = qkv_torch_out[:, q_size:q_size + kv_size]

    q_similarity = cosine_similarity(q_cuda, q_torch)
    k_similarity = cosine_similarity(k_cuda, k_torch)

    # Overall similarity
    overall_similarity = cosine_similarity(qkv_cuda, qkv_torch_out)

    # Precision threshold
    threshold = 0.9999

    passed = True
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test Configuration:")
        print(f"  num_tokens={num_tokens}, total_num_heads={total_num_heads}, tp_size={tp_size}")
        print(f"  num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        print(f"  head_dim={head_dim}, dtype={dtype}")
        print(f"  q_size={q_size}, kv_size={kv_size}")
        print(f"{'='*60}")
        print(f"\nAccuracy Verification (Cosine Similarity):")
        print(f"  Q similarity:  {q_similarity:.6f} (threshold: {threshold})")
        print(f"  K similarity:  {k_similarity:.6f} (threshold: {threshold})")
        print(f"  Overall:       {overall_similarity:.6f} (threshold: {threshold})")

    # Check threshold
    if q_similarity < threshold:
        print(f"  FAILED: Q similarity {q_similarity:.6f} < {threshold}")
        passed = False

    if k_similarity < threshold:
        print(f"  FAILED: K similarity {k_similarity:.6f} < {threshold}")
        passed = False

    if overall_similarity < threshold:
        print(f"  FAILED: Overall similarity {overall_similarity:.6f} < {threshold}")
        passed = False

    if passed and verbose:
        print(f"  PASSED: All similarities >= {threshold}")

    # =====================
    # Step 4: Performance Benchmark
    # =====================
    if verbose:
        print(f"\n{'='*60}")
        print(f"Performance Benchmark:")

    # CUDA kernel benchmark
    cuda_avg_time_ms, cuda_bandwidth_gbps = benchmark_cuda_kernel(
        qkv.clone(), weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache,
        num_iterations=1000, warmup=10
    )

    # Torch reference benchmark (fewer iterations since it's slower)
    torch_avg_time_ms = benchmark_torch_reference(
        qkv.clone(), weight, positions, q_size, kv_size, head_dim, eps, cos_sin_cache,
        num_iterations=100, warmup=5
    )

    # Speedup ratio
    speedup = torch_avg_time_ms / cuda_avg_time_ms

    if verbose:
        print(f"  CUDA kernel avg time:    {cuda_avg_time_ms:.4f} ms/iter")
        print(f"  CUDA kernel bandwidth:   {cuda_bandwidth_gbps:.2f} GB/s")
        print(f"  Torch reference avg time: {torch_avg_time_ms:.4f} ms/iter")
        print(f"  Speedup (Torch/CUDA):     {speedup:.2f}x")
        print(f"{'='*60}")

    # =====================
    # Final assertion
    # =====================
    assert overall_similarity >= threshold, \
        f"Accuracy verification FAILED: cosine similarity {overall_similarity:.6f} < {threshold}"

    return passed, overall_similarity, cuda_avg_time_ms, cuda_bandwidth_gbps, speedup


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    dtype = torch.bfloat16

    print("\n" + "="*70)
    print("  Unit Test: gemma_fused_rmsnorm_rope CUDA Kernel")
    print("  Accuracy: Cosine Similarity >= 0.9999")
    print("  Performance: CUDA vs Torch Reference")
    print("="*70)

    test_configs = [
        # (tp_size, num_tokens)
        (8, 2048),
        (8, 4096),
        (8, 8192),
        (1, 1),
        (1, 16),
        (1, 32),
        (1, 128),
        (1, 256),
    ]

    total_num_heads = 96  # full_attention: 64, sliding_attention: 96

    all_passed = True
    results = []

    for tp_size, num_tokens in test_configs:
        print(f"\n{'='*70}")
        print(f" Running test: tp_size={tp_size}, num_tokens={num_tokens}")
        print(f"{'='*70}")

        try:
            passed, similarity, cuda_time, bandwidth, speedup = test_gemma_fused_rmsnorm_rope(
                num_tokens, total_num_heads, tp_size, dtype, verbose=True
            )
            results.append({
                'tp_size': tp_size,
                'num_tokens': num_tokens,
                'passed': passed,
                'similarity': similarity,
                'cuda_time_ms': cuda_time,
                'bandwidth_gbps': bandwidth,
                'speedup': speedup
            })
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False
            results.append({
                'tp_size': tp_size,
                'num_tokens': num_tokens,
                'passed': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    print(f"\n{'Config':<25} {'Passed':<10} {'Similarity':<12} {'CUDA Time':<12} {'Bandwidth':<12} {'Speedup':<10}")
    print("-"*80)

    for r in results:
        if 'error' in r:
            print(f"tp={r['tp_size']}, tok={r['num_tokens']:<6}  ERROR: {r['error']}")
        else:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"tp={r['tp_size']}, tok={r['num_tokens']:<6}  {status:<10} "
                  f"{r['similarity']:.6f}    {r['cuda_time_ms']:.4f} ms   "
                  f"{r['bandwidth_gbps']:.2f} GB/s   {r['speedup']:.2f}x")

    print("\n" + "="*70)
    if all_passed:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED!")
    print("="*70)

    # Exit with proper status
    import sys
    sys.exit(0 if all_passed else 1)