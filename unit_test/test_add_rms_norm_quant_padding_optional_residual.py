"""
Unit test for add_rms_norm_dynamic_per_token_quant_padding_output kernel.
Tests both scenarios: with residual and without residual.
Tests hidden_size: 7168, 5120, 6144, 4096.
"""

import torch
import torch.nn.functional as F
import math
import time

# Import the CUDA kernel
from mcoplib.op import fused_add_rms_norm_dynamic_per_token_quant_padding_output


def reference_add_rms_norm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor = None
) -> tuple:
    """
    Reference implementation of add_rms_norm + per-token quantization in PyTorch.

    Args:
        input: [num_tokens, hidden_size] - Input tensor
        weight: [hidden_size] - RMS norm weight
        epsilon: RMS norm epsilon
        residual: [num_tokens, hidden_size] or None - Optional residual tensor

    Returns:
        residual_out: residual after add (or None if residual was None)
        out_rms: RMS norm output (bf16)
        output_quant_int8: Quantized output (int8)
        scales: Per-token quantization scales
    """
    # Step 1: Add residual if provided
    if residual is not None:
        x = input.float() + residual.float()
        residual_out = x.to(input.dtype)  # Store back to residual
    else:
        x = input.float()
        residual_out = None

    # Step 2: Compute RMS norm
    # variance = mean(x^2)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    # rms = sqrt(variance + epsilon)
    rms = torch.sqrt(variance + epsilon)
    # normalized = x / rms
    x_norm = x / rms

    # Step 3: Apply weight
    out = x_norm * weight.float()

    # Convert to bf16 for output_rms
    out_rms = out.to(torch.bfloat16)

    # Step 4: Per-token dynamic quantization to int8
    # Compute per-token scale: scale = max(|out|) / 127
    qmax = 127.0  # int8 max
    absmax = out.abs().max(dim=-1, keepdim=True).values
    scales = torch.clamp(absmax / qmax, min=1e-5)  # Avoid zero scale

    # Quantize: quantized = round(out / scale)
    quantized = torch.round(out / scales)
    # Clamp to int8 range
    quantized = torch.clamp(quantized, min=-128, max=127)
    output_quant_int8 = quantized.to(torch.int8)

    # Flatten scales to [num_tokens]
    scales_out = scales.squeeze(-1)

    return residual_out, out_rms, output_quant_int8, scales_out


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1)
    return cos_sim.item()


def compute_bandwidth(hidden_size: int, num_tokens: int, time_ms: float) -> float:
    """
    Compute kernel bandwidth in GB/s.

    Total data read/written:
    - Read: input (bf16), weight (bf16), residual (bf16) if present
    - Write: residual (bf16) if present, out_rms (bf16), output_quant_int8 (int8), scales (float32)

    Each bf16 = 2 bytes, int8 = 1 byte, float32 = 4 bytes
    """
    bytes_per_bf16 = 2
    bytes_per_int8 = 1
    bytes_per_float = 4

    total_bytes = 0

    # Reads
    total_bytes += num_tokens * hidden_size * bytes_per_bf16  # input
    total_bytes += hidden_size * bytes_per_bf16  # weight (shared across tokens)
    # residual read (if present, accounted in test function)

    # Writes
    total_bytes += num_tokens * hidden_size * bytes_per_bf16  # out_rms
    total_bytes += num_tokens * hidden_size * bytes_per_int8  # output_quant_int8
    total_bytes += num_tokens * bytes_per_float  # scales

    bandwidth = total_bytes / (time_ms * 1e-3) / 1e9  # GB/s
    return bandwidth


def benchmark(func, args, warmup=10, rep=100):
    """Benchmark function execution time."""
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_event[i].record()
        func(*args)
        end_event[i].record()

    torch.cuda.synchronize()
    durations = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    return durations


def test_single_hidden_size(hidden_size: int, has_residual: bool, num_tokens: int = 1, epsilon: float = 1e-6):
    """Test a single hidden_size configuration."""
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Create input tensors
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    if has_residual:
        residual = torch.randn_like(input)
        # Clone residual for later comparison (since kernel modifies it in-place)
        residual_original = residual.clone()
    else:
        residual = None
        residual_original = None

    # Determine pad_size (round up to multiple of some value for efficiency)
    pad_size = hidden_size  # Use hidden_size as pad_size for simplicity

    print(f"\n{'='*60}")
    print(f"Test: hidden_size={hidden_size}, has_residual={has_residual}, num_tokens={num_tokens}")
    print(f"{'='*60}")

    # Get reference results
    ref_residual, ref_out_rms, ref_quant, ref_scales = reference_add_rms_norm_quant(
        input, weight, epsilon, residual_original
    )

    # Call CUDA kernel
    # Create output tensors
    shape = list(input.shape)
    shape[-1] = pad_size
    output_buffer = torch.empty(shape, dtype=torch.bfloat16, device=input.device)
    C_output = output_buffer.view(dtype=torch.int8)

    output_rms = torch.empty_like(input)
    output_quant_int8 = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((num_tokens, 1), device=input.device, dtype=torch.float32)

    # Call kernel via ops
    import mcoplib.op as ops
    ops.fused_add_rms_norm_dynamic_per_token_quant_padding_output(
        C_output, output_rms, output_quant_int8, scales, input,
        residual, weight, pad_size, epsilon
    )

    torch.cuda.synchronize()

    # Verify precision using cosine similarity
    # 1. Check out_rms
    cos_sim_rms = cosine_similarity(ref_out_rms, output_rms)
    print(f"out_rms cosine similarity: {cos_sim_rms:.8f}")

    # 2. Check output_quant_int8 (convert to float for comparison)
    cos_sim_quant = cosine_similarity(ref_quant.float(), output_quant_int8.float())
    print(f"output_quant cosine similarity: {cos_sim_quant:.8f}")

    # 3. Check scales
    cos_sim_scales = cosine_similarity(ref_scales.unsqueeze(-1), scales)
    print(f"scales cosine similarity: {cos_sim_scales:.8f}")

    # 4. Check residual if present (kernel modifies residual in-place to input+residual)
    if has_residual and ref_residual is not None:
        # CUDA kernel modifies residual in-place: residual = input + residual
        expected_residual = (input.float() + residual_original.float()).to(dtype)
        cos_sim_residual = cosine_similarity(expected_residual, residual)
        print(f"residual cosine similarity: {cos_sim_residual:.8f}")
        # Assert residual precision
        assert cos_sim_residual > 0.9999, f"residual cosine similarity {cos_sim_residual} < 0.9999"

    # Assert precision requirements
    assert cos_sim_rms > 0.9999, f"out_rms cosine similarity {cos_sim_rms} < 0.9999"
    assert cos_sim_quant > 0.9999, f"output_quant cosine similarity {cos_sim_quant} < 0.9999"
    assert not math.isnan(cos_sim_rms), "Cosine similarity is NaN"

    print("Precision verification PASSED!")

    # Performance benchmark
    # Create fresh tensors for benchmarking
    torch.manual_seed(42)
    input_bench = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    weight_bench = torch.randn(hidden_size, dtype=dtype, device="cuda")
    residual_bench = torch.randn_like(input_bench) if has_residual else None

    output_buffer_bench = torch.empty(shape, dtype=dtype, device="cuda")
    C_output_bench = output_buffer_bench.view(dtype=torch.int8)
    output_rms_bench = torch.empty_like(input_bench)
    output_quant_bench = torch.empty_like(input_bench, dtype=torch.int8)
    scales_bench = torch.empty((num_tokens, 1), device="cuda", dtype=torch.float32)

    def cuda_kernel_func():
        ops.fused_add_rms_norm_dynamic_per_token_quant_padding_output(
            C_output_bench, output_rms_bench, output_quant_bench, scales_bench,
            input_bench, residual_bench, weight_bench, pad_size, epsilon
        )

    # Benchmark CUDA kernel
    dur_cuda = benchmark(cuda_kernel_func, (), warmup=10, rep=100)
    cuda_time_ms = dur_cuda.mean().item()
    print(f"CUDA kernel time: {cuda_time_ms:.4f} ms")

    # Benchmark PyTorch reference
    def torch_reference_func():
        reference_add_rms_norm_quant(input_bench, weight_bench, epsilon, residual_bench)

    dur_torch = benchmark(torch_reference_func, (), warmup=10, rep=100)
    torch_time_ms = dur_torch.mean().item()
    print(f"PyTorch reference time: {torch_time_ms:.4f} ms")

    # Performance ratio
    perf_ratio = torch_time_ms / cuda_time_ms
    print(f"Performance ratio (torch/cuda): {perf_ratio:.2f}x")

    # Compute bandwidth
    bandwidth = compute_bandwidth(hidden_size, num_tokens, cuda_time_ms)
    print(f"CUDA kernel bandwidth: {bandwidth:.2f} GB/s")

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running add_rms_norm_dynamic_per_token_quant_padding_output tests")
    print("="*60)

    hidden_sizes = [7168, 5120, 6144, 4096]
    num_tokens = 1
    epsilon = 1e-6

    results = []

    for hidden_size in hidden_sizes:
        # Test with residual
        try:
            test_name = f"hidden_size={hidden_size}, with residual"
            test_single_hidden_size(hidden_size, has_residual=True, num_tokens=num_tokens, epsilon=epsilon)
            results.append((test_name, True))
        except Exception as e:
            print(f"Test FAILED: {e}")
            results.append((f"hidden_size={hidden_size}, with residual", False))

        # Test without residual
        try:
            test_name = f"hidden_size={hidden_size}, without residual"
            test_single_hidden_size(hidden_size, has_residual=False, num_tokens=num_tokens, epsilon=epsilon)
            results.append((test_name, True))
        except Exception as e:
            print(f"Test FAILED: {e}")
            results.append((f"hidden_size={hidden_size}, without residual", False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")

    return all_passed


if __name__ == "__main__":
    run_all_tests()