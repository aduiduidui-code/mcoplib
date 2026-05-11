#!/usr/bin/env python
import time
import math
import torch
import torch.nn.functional as F
from mcoplib.op import softplus_sqrt_f16
import torch

def impl_torch(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x).sqrt()

def impl_fused(x: torch.Tensor) -> torch.Tensor:
    return softplus_sqrt_f16(x)

def impl_ref(x: torch.Tensor) -> torch.Tensor:
    x64 = x.double()
    return F.softplus(x64).sqrt()


def validate_accuracy():
    all_pass = True

    ATOL = 2e-4
    RTOL = 1e-3

    boundary_cases = {
        "x = 0.0": torch.tensor([0.0], dtype=torch.float16, device="cuda"),
        "x = 20.0": torch.tensor([20.0], dtype=torch.float16, device="cuda"),
        "x = -20.0": torch.tensor([-20.0], dtype=torch.float16, device="cuda"),
        "x = 18.0": torch.tensor([18.0], dtype=torch.float16, device="cuda"),
        "x = 22.0": torch.tensor([22.0], dtype=torch.float16, device="cuda"),
        "x = -18.0": torch.tensor([-18.0], dtype=torch.float16, device="cuda"),
        "x = -22.0": torch.tensor([-22.0], dtype=torch.float16, device="cuda"),
        "x = 65504": torch.tensor([65504.0], dtype=torch.float16, device="cuda"),
        "x = -65504": torch.tensor([-65504.0], dtype=torch.float16, device="cuda"),
    }

    impls = {
        "torch": impl_torch,
        "fused": softplus_sqrt_f16,
    }

    print(f"[boundary cases] atol={ATOL:.1e}, rtol={RTOL:.1e}")
    for name, x in boundary_cases.items():
        ref = impl_ref(x).double()
        row = f"  {name:18s}"

        for label, fn in impls.items():
            out = fn(x).double()
            abs_err = (out - ref).abs().item()
            rel_err = abs_err / (ref.abs().item() + 1e-12)

            passed = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False

            row += (
                f"  [{label} {status}]"
                f" out={out.item():.2f}"
                f" abs={abs_err:.2e}"
                f" rel={rel_err:.2e}"
            )

        print(row)

    print()
    test_shapes = [
        ((1024, 1024), "(1024, 1024)"),
        ((4096, 4096), "(4096, 4096)"),
        ((1, 1), "(1, 1)"),
        ((1, 65535), "(1, 65535)"),
        ((65535, 1), "(65535, 1)"),
        ((4, 2048, 256), "(4, 2048, 256)"),
        ((8, 4096, 64), "(8, 4096, 64)"),
        ((1, 1, 1), "(1, 1, 1)"),
        ((16, 512, 512), "(16, 512, 512)"),
        ((2, 32, 512, 512), "(2, 32, 512, 512)"),
        ((1, 1, 4096, 4096), "(1, 1, 4096, 4096)"),
        ((4, 8, 128, 128), "(4, 8, 128, 128)"),
        ((1023, 1025), "(1023, 1025)"),
        ((3, 2047, 255), "(3, 2047, 255)"),
        ((2048, 2048),  "(2048, 2048)"),
    ]

    print(f"[random shapes] atol={ATOL:.1e}, rtol={RTOL:.1e}")
    for shape, label in test_shapes:
        x = torch.randn(*shape, dtype=torch.float16, device="cuda") * 10
        ref = impl_ref(x).double()
        row = f"  {label:25s}"

        for impl_label, fn in impls.items():
            out = fn(x).double()
            diff = (out - ref).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()

            passed = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False

            row += (
                f"  [{impl_label} {status}]"
                f" max={max_err:.2e}"
                f" mean={mean_err:.2e}"
            )

        print(row)

    print()
    print("PASS!" if all_pass else "FAILED!")
    print()


def benchmark(fn, x, warmup=100, iters=2000):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) / iters * 1000
    numel = math.prod(x.shape)
    bw = numel * x.element_size() * 2 / (ms * 1e-3) / 1e9
    return ms, bw

def profile_performance():
    shapes = [
        ((1024, 1024),        "(1024, 1024)"),
        ((4096, 4096),        "(4096, 4096)"),
        ((4, 2048, 512),      "(4, 2048, 512)"),
        ((8, 4096, 64),       "(8, 4096, 64)"),
        ((2, 32, 4096, 4096), "(2, 32, 4096, 4096)"),
    ]

    header = f"  {'Shape':30s}  {'torch(ms)':>10}  {'torch(GB/s)':>12}"
    header += f"  {'cuda(ms)':>9}  {'cuda(GB/s)':>11}  {'speedup':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for shape, label in shapes:
        try:
            x = torch.randn(*shape, dtype=torch.float16, device="cuda")
            t_ms, t_bw = benchmark(impl_torch, x)
            row = f"  {label:30s}  {t_ms:10.3f}  {t_bw:12.1f}"
            f_ms, f_bw = benchmark(impl_fused, x)
            row += f"  {f_ms:9.3f}  {f_bw:11.1f}  {t_ms/f_ms:8.2f}x"
            print(row)
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:30s}  OOM — skipped")

    print()

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print(f"\nDevice : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    validate_accuracy()
    profile_performance()
