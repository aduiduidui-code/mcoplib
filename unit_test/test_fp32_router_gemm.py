import torch

# 尝试导入自定义的 CUDA op
try:
    import mcoplib._C
except ImportError:
    print("Warning: 无法导入 mcoplib._C，请确保算子已正确编译安装在环境中。")
    print("测试脚本将继续，但在调用 torch.ops._C.fp32_router_gemm 时可能会报错。")


NUM_EXPERTS = 256
HIDDEN_DIM = 3072


def fp32_router_gemm(hidden_states: torch.Tensor,
                     router_weight: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        (hidden_states.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    torch.ops._C.fp32_router_gemm(output, hidden_states, router_weight)
    return output


def run_case(m: int, dtype: torch.dtype, atol: float) -> None:
    torch.manual_seed(0)

    x = torch.randn(m, HIDDEN_DIM, dtype=dtype, device="cuda")
    w = torch.randn(NUM_EXPERTS, HIDDEN_DIM, dtype=torch.float32, device="cuda")

    out = fp32_router_gemm(x, w)
    torch.cuda.synchronize()

    ref = (x.float().cpu().double() @ w.cpu().double().t()).float().to("cuda")
    diff = (out - ref).abs()

    print(
        f"M={m}, dtype={dtype}, "
        f"max_diff={diff.max().item():.8g}, "
        f"mean_diff={diff.mean().item():.8g}"
    )

    assert out.shape == (m, NUM_EXPERTS)
    assert out.dtype == torch.float32
    assert diff.max().item() <= atol


def run_test() -> None:
    assert torch.cuda.is_available(), "需要 CUDA 环境来运行该测试"

    print("has _C:", hasattr(torch.ops, "_C"))
    print("has fp32_router_gemm:", hasattr(torch.ops._C, "fp32_router_gemm"))

    assert hasattr(torch.ops._C, "fp32_router_gemm"), (
        "torch.ops._C.fp32_router_gemm 未注册，请确认已重新编译并安装 mcoplib._C"
    )

    for m in [1, 2, 4, 8, 16, 32]:
        run_case(m, torch.float32, 2e-4)
        run_case(m, torch.bfloat16, 2e-2)

    print("PASS")


if __name__ == "__main__":
    run_test()
