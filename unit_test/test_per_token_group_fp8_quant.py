import torch

import mcoplib._C  # noqa: F401


def _format_case(
    m: int,
    n: int,
    dtype: torch.dtype,
    group_size: int,
    scale_ue8m0: bool,
    column_major_scales: bool,
) -> str:
    return (
        f"shape=({m}, {n}), dtype={dtype}, group_size={group_size}, "
        f"scale_ue8m0={scale_ue8m0}, column_major_scales={column_major_scales}"
    )


def _check_close_verbose(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    case_desc: str,
    actual_raw: torch.Tensor | None = None,
    expected_raw: torch.Tensor | None = None,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    allowed = atol + rtol * expected_f.abs()
    fail_mask = diff > allowed

    if not fail_mask.any():
        return

    fail_count = int(fail_mask.sum().item())
    total = actual.numel()
    flat_idx = int(diff.reshape(-1).argmax().item())
    max_abs = float(diff.reshape(-1)[flat_idx].item())
    max_allowed = float(allowed.reshape(-1)[flat_idx].item())
    actual_val = float(actual_f.reshape(-1)[flat_idx].item())
    expected_val = float(expected_f.reshape(-1)[flat_idx].item())
    max_index = tuple(int(i) for i in fail_mask.nonzero()[diff[fail_mask].argmax()].tolist())

    print(f"[FAIL] {name}: {case_desc}")
    print(f"  mismatched: {fail_count} / {total}")
    print(f"  max_index: {max_index}")
    print(f"  actual: {actual_val}")
    print(f"  expected: {expected_val}")
    print(f"  abs_diff: {max_abs}")
    print(f"  allowed: {max_allowed} (atol={atol}, rtol={rtol})")

    if actual_raw is not None and expected_raw is not None:
      actual_raw_flat = actual_raw.reshape(-1)
      expected_raw_flat = expected_raw.reshape(-1)
      if actual_raw.dtype == torch.float8_e4m3fn and expected_raw.dtype == torch.float8_e4m3fn:
          actual_bits = int(actual_raw_flat.view(torch.uint8)[flat_idx].item())
          expected_bits = int(expected_raw_flat.view(torch.uint8)[flat_idx].item())
          print(f"  actual_fp8_bits: 0x{actual_bits:02x}")
          print(f"  expected_fp8_bits: 0x{expected_bits:02x}")

    topk = min(5, fail_count)
    if topk > 0:
        fail_flat_idx = torch.nonzero(fail_mask.reshape(-1), as_tuple=False).squeeze(-1)
        top_order = torch.argsort(diff.reshape(-1)[fail_flat_idx], descending=True)[:topk]
        top_idx = fail_flat_idx[top_order]
        print("  top mismatches:")
        for i, idx in enumerate(top_idx.tolist(), start=1):
            coord = []
            tmp = idx
            for size in reversed(actual.shape):
                coord.append(tmp % size)
                tmp //= size
            coord = tuple(int(v) for v in reversed(coord))
            print(
                f"    {i}. index={coord}, actual={float(actual_f.reshape(-1)[idx].item())}, "
                f"expected={float(expected_f.reshape(-1)[idx].item())}, "
                f"abs_diff={float(diff.reshape(-1)[idx].item())}, "
                f"allowed={float(allowed.reshape(-1)[idx].item())}"
            )

    raise AssertionError(f"{name} check failed for {case_desc}")


def ref_per_token_group_fp8_quant(
    x: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    assert n % group_size == 0

    x_view = x.view(m, n // group_size, group_size)
    scales = torch.clamp(x_view.abs().amax(dim=-1).float(), min=eps) / fp8_max
    if scale_ue8m0:
        scales = torch.exp2(torch.ceil(torch.log2(torch.clamp(scales, min=1e-10))))

    q = torch.clamp(x_view / scales.unsqueeze(-1), fp8_min, fp8_max).to(
        torch.float8_e4m3fn
    )
    return q.view_as(x), scales


def run_case(
    m: int,
    n: int,
    dtype: torch.dtype,
    group_size: int,
    scale_ue8m0: bool,
    column_major_scales: bool,
) -> None:
    case_desc = _format_case(
        m, n, dtype, group_size, scale_ue8m0, column_major_scales
    )
    print(f"[RUN] {case_desc}")

    torch.manual_seed(0)
    x = torch.randn(m, n, dtype=dtype, device="cuda")
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    eps = 1e-10

    out_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    if column_major_scales:
        out_s = torch.empty((n // group_size, m), device="cuda",
                            dtype=torch.float32).permute(1, 0)
    else:
        out_s = torch.empty((m, n // group_size), device="cuda",
                            dtype=torch.float32)

    torch.ops._C.per_token_group_fp8_quant(
        x,
        out_q,
        out_s,
        group_size,
        eps,
        fp8_info.min,
        fp8_info.max,
        scale_ue8m0,
        column_major_scales,
        False,
    )

    ref_q, ref_s = ref_per_token_group_fp8_quant(
        x, group_size, eps, fp8_info.min, fp8_info.max, scale_ue8m0
    )

    # Match vLLM upstream test tolerance for the native CUDA path.
    _check_close_verbose("scale", out_s, ref_s, atol=1e-2, rtol=1e-2,
                         case_desc=case_desc)
    _check_close_verbose("quantized_output", out_q.float(), ref_q.float(),
                         atol=1.5e-1, rtol=1.5e-1, case_desc=case_desc,
                         actual_raw=out_q, expected_raw=ref_q)
    print(f"[PASS] {case_desc}")


def test_per_token_group_fp8_quant() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    failures: list[str] = []
    for dtype in (torch.float16, torch.bfloat16):
        for group_size in (64, 128):
            for scale_ue8m0 in (False, True):
                for column_major_scales in (False, True):
                    try:
                        run_case(
                            32,
                            1024,
                            dtype,
                            group_size,
                            scale_ue8m0,
                            column_major_scales,
                        )
                    except AssertionError as exc:
                        failures.append(str(exc))

    if failures:
        print("\nSummary of failed cases:")
        for i, failure in enumerate(failures, start=1):
            print(f"  {i}. {failure}")
        raise AssertionError(f"{len(failures)} case(s) failed")


if __name__ == "__main__":
    test_per_token_group_fp8_quant()
    print("per_token_group_fp8_quant data test passed")
