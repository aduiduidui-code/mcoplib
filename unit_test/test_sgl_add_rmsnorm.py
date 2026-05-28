#!/usr/bin/env python3
"""
Minimal reproducer for mcoplib fused_add_rmsnorm bug.

Expected correct semantic:
    residual = input_x + input_residual
    x = RMSNorm(residual, weight, eps)

This test only targets:
    torch.ops.sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps, False)

Run inside the mcoplib Docker.
"""

import argparse
import sys
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--print-items", type=int, default=8)
    return p.parse_args()


def cpu_randn_bf16(shape, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=gen).to(torch.bfloat16)


def rmsnorm_ref(input_x, input_residual, weight, eps):
    residual_f = input_x.float() + input_residual.float()
    variance = residual_f.pow(2).mean(dim=-1, keepdim=True)
    x_f = residual_f * torch.rsqrt(variance + eps) * weight.float()
    return x_f.to(torch.bfloat16), residual_f.to(torch.bfloat16)


def stat_line(name, t, n=8):
    tf = t.detach().float().cpu()
    first = tf.reshape(-1)[:n].tolist()
    print(
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={tf.mean().item():.8e} std={tf.std().item():.8e} "
        f"absmax={tf.abs().max().item():.8e} "
        f"nan={int(torch.isnan(tf).sum().item())} inf={int(torch.isinf(tf).sum().item())}"
    )
    print(f"  first{n}: {first}")


def max_metrics(a, b):
    af = a.detach().float()
    bf = b.detach().float()
    diff = (af - bf).abs()
    rel = diff / af.abs().clamp_min(1e-12)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Importing mcoplib.sgl_kernel registers torch.ops.sgl_kernel.*.
    import mcoplib.sgl_kernel as K  # noqa: F401
    import mcoplib._C

    # Mirror the SGLang-style global redirect, although this repro uses torch.ops directly.
    sys.modules["sgl_kernel"] = K

    print("mcoplib.sgl_kernel file:", getattr(K, "__file__", "<unknown>"))
    try:
        print("schema:", torch.ops.sgl_kernel.fused_add_rmsnorm._schemas)
    except Exception as exc:
        print("schema unavailable:", repr(exc))

    device = "cuda"
    dtype = torch.bfloat16

    input_x = cpu_randn_bf16((args.tokens, args.hidden), args.seed * 1000 + 0).to(device)
    input_residual = cpu_randn_bf16((args.tokens, args.hidden), args.seed * 1000 + 1).to(device)
    weight = (1.0 + 0.1 * cpu_randn_bf16((args.hidden,), 20260512).float()).to(dtype).to(device)

    ref_x, ref_residual = rmsnorm_ref(input_x, input_residual, weight, args.eps)

    # The op is in-place. x and residual are both mutated.
    x = input_x.clone()
    residual = input_residual.clone()

    #torch.ops._C.fused_add_rms_norm(x, residual, weight, args.eps)
    torch.ops.sgl_kernel.fused_add_rmsnorm(x, residual, weight, args.eps, False)
    torch.cuda.synchronize()

    print("\n===== values =====")
    stat_line("input_x", input_x, args.print_items)
    stat_line("input_residual", input_residual, args.print_items)
    stat_line("ref_residual", ref_residual, args.print_items)
    stat_line("actual_residual", residual, args.print_items)
    stat_line("ref_x", ref_x, args.print_items)
    stat_line("actual_x", x, args.print_items)

    residual_ok = torch.allclose(residual, ref_residual, rtol=args.rtol, atol=args.atol)
    x_ok = torch.allclose(x, ref_x, rtol=args.rtol, atol=args.atol)
    x_is_all_zero = bool(torch.all(x == 0).item())

    x_max_abs, x_mean_abs, x_max_rel = max_metrics(ref_x, x)
    r_max_abs, r_mean_abs, r_max_rel = max_metrics(ref_residual, residual)

    print("\n===== checks =====")
    print(f"residual_ok={residual_ok} max_abs={r_max_abs:.8e} mean_abs={r_mean_abs:.8e} max_rel={r_max_rel:.8e}")
    print(f"x_ok={x_ok} max_abs={x_max_abs:.8e} mean_abs={x_mean_abs:.8e} max_rel={x_max_rel:.8e}")
    print(f"x_is_all_zero={x_is_all_zero}")

    # Correct implementation should pass both checks.
    # Current buggy mcoplib raw-5args-false typically has residual_ok=True, x_ok=False, x_is_all_zero=True.
    assert residual_ok, "residual output is wrong; expected residual = input_x + input_residual"
    assert x_ok, (
        "BUG REPRODUCED: fused_add_rmsnorm wrote incorrect x. "
        "Expected x = RMSNorm(input_x + input_residual, weight, eps). "
        f"x_is_all_zero={x_is_all_zero}, max_abs={x_max_abs:.8e}, mean_abs={x_mean_abs:.8e}"
    )

    print("\nPASS: mcoplib fused_add_rmsnorm matches reference.")


if __name__ == "__main__":
    main()