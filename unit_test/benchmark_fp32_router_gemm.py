# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
import pandas as pd
import mcoplib._C

from torch.profiler import profile, ProfilerActivity
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Dimensions supported by the DSV3 specialized kernel
DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
DSV3_SUPPORTED_HIDDEN_SIZES = [7168]

# Dimensions supported by the gpt-oss specialized kernel
GPT_OSS_SUPPORTED_NUM_EXPERTS = [32, 128]
GPT_OSS_SUPPORTED_HIDDEN_SIZES = [2880]

# Dimensions supported by the fp32 specialized kernel (MiniMax-M2)
FP32_SUPPORTED_NUM_EXPERTS = [256]
FP32_SUPPORTED_HIDDEN_SIZES = [3072]
FP32_MAX_TOKENS = 32


def fp32_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        hidden_states.shape[0],
        router_weight.shape[0],
        device=hidden_states.device,
        dtype=torch.float32,
    )
    torch.ops._C.fp32_router_gemm(output, hidden_states, router_weight)
    return output


def get_batch_size_range(max_batch_size):
    return [2**x for x in range(14) if 2**x <= max_batch_size]


def get_model_params(config):
    if config.architectures[0] in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
    ):
        num_experts = config.n_routed_experts
        hidden_size = config.hidden_size
    elif config.architectures[0] in ("GptOssForCausalLM",) or config.architectures[
        0
    ] in ("MiniMaxM2ForCausalLM",):
        num_experts = config.num_local_experts
        hidden_size = config.hidden_size
    else:
        raise ValueError(f"Unsupported architecture: {config.architectures}")
    return num_experts, hidden_size


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def estimate_router_gemm_bytes(
    batch_size: int,
    hidden_size: int,
    num_experts: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    bias_dtype: torch.dtype | None = None,
) -> int:
    """
    Estimate total bytes moved by one router GEMM invocation.

    This is an effective bandwidth estimate:
      input read + weight read + output write + optional bias read
    """
    total = batch_size * hidden_size * _dtype_bytes(input_dtype)
    total += num_experts * hidden_size * _dtype_bytes(weight_dtype)
    total += batch_size * num_experts * _dtype_bytes(output_dtype)
    if bias_dtype is not None:
        total += num_experts * _dtype_bytes(bias_dtype)
    return total


def bandwidth_gbps(total_bytes: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return float("inf")
    return total_bytes / (latency_ms * 1e-3) / 1e9


def get_benchmark(model, max_batch_size, trust_remote_code):
    # Collect per-run results for a custom summary table.
    collected_rows = []

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=get_batch_size_range(max_batch_size),
            x_log=False,
            line_arg="provider",
            line_vals=[
                "torch",
                "vllm",
            ],
            line_names=["PyTorch", "vLLM"],
            styles=([("blue", "-"), ("red", "-")]),
            ylabel="TFLOPs",
            plot_name=f"{model} router gemm throughput",
            args={},
        )
    )
    def benchmark(batch_size, provider):
        config = get_config(model=model, trust_remote_code=trust_remote_code)
        num_experts, hidden_size = get_model_params(config)

        is_hopper_or_blackwell = current_platform.is_device_capability(
            90
        ) or current_platform.is_device_capability_family(100)

        allow_dsv3_router_gemm = (
            is_hopper_or_blackwell
            and num_experts in DSV3_SUPPORTED_NUM_EXPERTS
            and hidden_size in DSV3_SUPPORTED_HIDDEN_SIZES
        )
        allow_gpt_oss_router_gemm = (
            is_hopper_or_blackwell
            and num_experts in GPT_OSS_SUPPORTED_NUM_EXPERTS
            and hidden_size in GPT_OSS_SUPPORTED_HIDDEN_SIZES
        )
        is_fp32_router_model = (
            is_hopper_or_blackwell
            and num_experts in FP32_SUPPORTED_NUM_EXPERTS
            and hidden_size in FP32_SUPPORTED_HIDDEN_SIZES
        )
        allow_fp32_router_gemm = is_fp32_router_model and batch_size <= FP32_MAX_TOKENS

        # Weight dtype: fp32 kernel requires fp32 weights; others use bf16.
        weight_dtype = torch.float32 if is_fp32_router_model else torch.bfloat16

        mat_a = torch.randn(
            (batch_size, hidden_size), dtype=torch.bfloat16, device="cuda"
        ).contiguous()
        mat_b = torch.randn(
            (num_experts, hidden_size), dtype=weight_dtype, device="cuda"
        ).contiguous()
        bias = torch.randn(
            num_experts, dtype=torch.bfloat16, device="cuda"
        ).contiguous()

        has_bias = allow_gpt_oss_router_gemm

        print(f"""Running benchmark with batch_size={batch_size}, hidden_size={hidden_size}, num_experts={num_experts}\n"""
              f"""allow_dsv3_router_gemm={allow_dsv3_router_gemm}, allow_gpt_oss_router_gemm={allow_gpt_oss_router_gemm}, allow_fp32_router_gemm={allow_fp32_router_gemm}""")
        print(f"""mat_a: dtype={mat_a.dtype}, shape={mat_a.shape}, stride={mat_a.stride()}""")
        print(f"""mat_b: dtype={mat_b.dtype}, shape={mat_b.shape}, stride={mat_b.stride()}""")
        # These dtypes are used for bandwidth estimation only.
        if provider == "torch":
            if allow_fp32_router_gemm:
                bw_input_dtype = torch.float32
                bw_weight_dtype = torch.float32
                bw_output_dtype = torch.float32
                bw_bias_dtype = None
            elif has_bias:
                bw_input_dtype = torch.bfloat16
                bw_weight_dtype = torch.bfloat16
                bw_output_dtype = torch.bfloat16
                bw_bias_dtype = torch.bfloat16
            else:
                bw_input_dtype = torch.bfloat16
                bw_weight_dtype = torch.bfloat16
                bw_output_dtype = torch.bfloat16
                bw_bias_dtype = None

            def runner():
                if allow_fp32_router_gemm:
                    F.linear(mat_a.float(), mat_b)
                elif has_bias:
                    F.linear(mat_a, mat_b, bias)
                else:
                    F.linear(mat_a, mat_b)

        elif provider == "vllm":
            if allow_dsv3_router_gemm:
                bw_input_dtype = torch.bfloat16
                bw_weight_dtype = torch.bfloat16
                bw_output_dtype = torch.bfloat16
                bw_bias_dtype = None
            elif allow_fp32_router_gemm:
                bw_input_dtype = torch.bfloat16
                bw_weight_dtype = torch.float32
                bw_output_dtype = torch.float32
                bw_bias_dtype = None
            elif allow_gpt_oss_router_gemm:
                bw_input_dtype = torch.bfloat16
                bw_weight_dtype = torch.bfloat16
                bw_output_dtype = torch.bfloat16
                bw_bias_dtype = torch.bfloat16
            elif is_fp32_router_model:
                bw_input_dtype = torch.float32
                bw_weight_dtype = torch.float32
                bw_output_dtype = torch.float32
                bw_bias_dtype = None
            else:
                bw_input_dtype = torch.bfloat16
                bw_weight_dtype = torch.bfloat16
                bw_output_dtype = torch.bfloat16
                bw_bias_dtype = None

            def runner():
                if allow_dsv3_router_gemm:
                    ops.dsv3_router_gemm(mat_a, mat_b, torch.bfloat16)
                elif allow_fp32_router_gemm:
                    fp32_router_gemm(mat_a, mat_b)
                elif allow_gpt_oss_router_gemm:
                    ops.gpt_oss_router_gemm(mat_a, mat_b, bias)
                elif is_fp32_router_model:
                    # batch_size > FP32_MAX_TOKENS: fall back to F.linear
                    F.linear(mat_a.float(), mat_b)
                else:
                    F.linear(mat_a, mat_b)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        WARMUP_ITERS = 10
        BENCH_ITERS = 100

        # Warmup
        for _ in range(WARMUP_ITERS):
            runner()
        torch.cuda.synchronize()

        # Benchmark: iterate BENCH_ITERS times and compute average latency
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(BENCH_ITERS):
            runner()
        end_event.record()
        torch.cuda.synchronize()
        ms = start_event.elapsed_time(end_event) / BENCH_ITERS

        def tflops(t_ms):
            flops = 2 * batch_size * hidden_size * num_experts
            return flops / (t_ms * 1e-3) / 1e12

        total_bytes = estimate_router_gemm_bytes(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            input_dtype=bw_input_dtype,
            weight_dtype=bw_weight_dtype,
            output_dtype=bw_output_dtype,
            bias_dtype=bw_bias_dtype,
        )
        bw = bandwidth_gbps(total_bytes, ms)

        avg_tflops = tflops(ms)

        collected_rows.append(
            {
                "batch_size": int(batch_size),
                "provider": "PyTorch" if provider == "torch" else "vLLM",
                "TFLOPs": avg_tflops,
                "Bandwidth_GBps": bw,
            }
        )

        print(f"  warmup={WARMUP_ITERS}, iters={BENCH_ITERS}, "
              f"avg_latency={ms:.4f} ms, avg_TFLOPs={avg_tflops:.4f}, avg_BW={bw:.2f} GB/s")

        return avg_tflops, avg_tflops, avg_tflops

    return benchmark, collected_rows


def run_fp32_router_gemm_profile():
    """Profile fp32_router_gemm kernel using torch.profiler."""
    NUM_EXPERTS = 256
    HIDDEN_DIM = 3072
    M_VALUES = [1, 2, 4, 8, 16, 32]
    WARMUP_ITERS = 10
    PROFILE_ITERS = 50

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    router_weight = torch.randn(NUM_EXPERTS, HIDDEN_DIM, device=device, dtype=torch.float32)

    for act_dtype in [torch.bfloat16]:
        dtype_name = "bf16" if act_dtype == torch.bfloat16 else "fp32"
        print(f"\n{'='*60}")
        print(f"  fp32_router_gemm Profile — activation dtype: {dtype_name}")
        print(f"{'='*60}")

        for m in M_VALUES:
            hidden_states = torch.randn(m, HIDDEN_DIM, device=device, dtype=act_dtype)

            # Warmup
            for _ in range(WARMUP_ITERS):
                _ = fp32_router_gemm(hidden_states, router_weight)
            torch.cuda.synchronize()

            # Profile
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            ) as prof:
                for _ in range(PROFILE_ITERS):
                    out = fp32_router_gemm(hidden_states, router_weight)
                    #F.linear(hidden_states.float(), router_weight)
                    torch.cuda.synchronize()

            # Print profiler table sorted by device total time
            print(f"\n--- M={m}, dtype={dtype_name} (Profiler, sorted by device total time) ---")
            print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))

            # Print only fp32_router_gemm related entries
            print(f"\n--- M={m}, dtype={dtype_name} (fp32_router_gemm only) ---")
            for e in prof.key_averages():
                if "fp32_router_gemm" in e.key:
                    print(
                        f"{e.key:40} | "
                        f"CPU: {e.cpu_time_total:8.2f} us | "
                        f"Device: {e.device_time_total:8.2f} us | "
                        f"Device avg: {e.device_time_total / PROFILE_ITERS:8.2f} us | "
                        f"calls: {e.count}"
                    )

            # Compute effective bandwidth and TFLOPs
            input_bytes = m * HIDDEN_DIM * hidden_states.element_size()
            weight_bytes = NUM_EXPERTS * HIDDEN_DIM * router_weight.element_size()
            output_bytes = m * NUM_EXPERTS * out.element_size()
            total_bytes = input_bytes + weight_bytes + output_bytes

            top_event = max(prof.key_averages(), key=lambda e: e.device_time_total)
            avg_cuda_us = top_event.device_time_total / PROFILE_ITERS
            avg_cuda_ms = avg_cuda_us / 1000.0

            bandwidth_gbps = total_bytes / (avg_cuda_ms * 1e-3) / 1e9 if avg_cuda_ms > 0 else float("inf")
            flops = 2 * m * HIDDEN_DIM * NUM_EXPERTS
            tflops = flops / (avg_cuda_ms * 1e-3) / 1e12

            print(
                f"\n  M={m}, dtype={dtype_name} | "
                f"data: {total_bytes / 1e6:.2f} MB | "
                f"avg CUDA: {avg_cuda_ms:.4f} ms | "
                f"bandwidth: {bandwidth_gbps:.2f} GB/s | "
                f"TFLOPs: {tflops:.4f}"
            )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--max-batch-size", default=16, type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler profiling for fp32_router_gemm kernel")
    args = parser.parse_args()

    if args.profile:
        run_fp32_router_gemm_profile()
    else:
        # Get the benchmark function and collection buffer
        benchmark, collected_rows = get_benchmark(
            args.model, args.max_batch_size, args.trust_remote_code
        )

        # Run performance benchmark
        benchmark.run(print_data=True)

        # Print a custom summary table that includes both throughput and bandwidth.
        df = pd.DataFrame(collected_rows)
        if not df.empty:
            wide = df.pivot(index="batch_size", columns="provider", values=["TFLOPs", "Bandwidth_GBps"])
            wide.columns = [f"{provider}_{metric}" for metric, provider in wide.columns]
            wide = wide.reset_index().sort_values("batch_size")

            # Reorder columns for readability.
            ordered_cols = ["batch_size"]
            for p in ["PyTorch", "vLLM"]:
                for m in ["TFLOPs", "Bandwidth_GBps"]:
                    col = f"{p}_{m}"
                    if col in wide.columns:
                        ordered_cols.append(col)
            wide = wide[ordered_cols]

            print("\n=== Router GEMM Throughput + Effective Bandwidth ===")
            print(wide.to_string(index=False))