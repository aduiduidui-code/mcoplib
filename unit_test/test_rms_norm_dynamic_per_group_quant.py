# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import pytest
import torch

import mcoplib.sgl_kernel  # noqa: F401


EPS = 1e-6
GROUP_SIZE = 128

DTYPES = [torch.float16]
QUANT_DTYPES = [torch.int8]
if hasattr(torch, "float8_e4m3fn"):
    QUANT_DTYPES.append(torch.float8_e4m3fn)

NUM_TOKENS_HIDDEN_SIZES = [
    # small / vector coverage
    (1, 128),
    (1, 512),
    (1, 1024),
    (16, 128),
    (16, 512),
    (16, 1024),
    (32, 128),
    (32, 512),
    (32, 1024),

    # production shapes
    (1, 4096),
    (1, 5120),
    (1, 6144),
    (1, 7168),
    (16, 4096),
    (16, 5120),
    (16, 6144),
    (16, 7168),
    (32, 4096),
    (32, 5120),
    (32, 6144),
    (32, 7168),
]

ADD_RESIDUAL = [False, True]
SCALE_UBS = [False, True]
SEEDS = [0]

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() <= 1 else 2)
]


def get_fp8_dtype() -> Optional[torch.dtype]:
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return None


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return get_fp8_dtype() is not None and dtype == get_fp8_dtype()


def as_float32_tensor(x: Union[float, torch.Tensor], device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    x_fp32 = x.float()

    if residual is not None:
        residual_out = x_fp32 + residual.float()
        x_norm_in = residual_out
    else:
        residual_out = None
        x_norm_in = x_fp32

    variance = torch.mean(x_norm_in * x_norm_in, dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)

    out = x_norm_in * rstd * weight.float()
    return out, residual_out


def quantize_int8_ref(
    norm: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, hidden = norm.shape
    groups = (hidden + group_size - 1) // group_size

    out = torch.empty_like(norm, dtype=torch.int8)
    scales = torch.empty((tokens, groups), dtype=torch.float32, device=norm.device)

    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, hidden)
        chunk = norm[:, start:end]

        amax = chunk.abs().amax(dim=-1)
        scale = torch.clamp(amax / 127.0, min=torch.finfo(torch.float32).eps)
        scales[:, g] = scale

        q = torch.round(chunk / scale[:, None]).clamp(-127, 127).to(torch.int8)
        out[:, start:end] = q

    return out, scales


def quantize_fp8_ref(
    norm: torch.Tensor,
    group_size: int,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, hidden = norm.shape
    groups = (hidden + group_size - 1) // group_size

    out = torch.empty_like(norm, dtype=fp8_dtype)
    scales = torch.empty((tokens, groups), dtype=torch.float32, device=norm.device)

    # e4m3 max used by current kernel path.
    qmax = 448.0
    min_scale = 1.0 / (qmax * 512.0)

    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, hidden)
        chunk = norm[:, start:end]

        amax = chunk.abs().amax(dim=-1)
        scale = torch.clamp(amax / qmax, min=min_scale)
        scales[:, g] = scale

        out[:, start:end] = (chunk / scale[:, None]).to(fp8_dtype)

    return out, scales


def dynamic_per_group_quant_ref(
    norm: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if quant_dtype == torch.int8:
        return quantize_int8_ref(norm, group_size)

    if is_fp8_dtype(quant_dtype):
        return quantize_fp8_ref(norm, group_size, quant_dtype)

    raise AssertionError(f"unsupported quant dtype: {quant_dtype}")


def expand_group_scales(scales: torch.Tensor, hidden: int, group_size: int) -> torch.Tensor:
    expanded = scales.repeat_interleave(group_size, dim=-1)
    return expanded[:, :hidden]


def ops_impl(
    weight: torch.Tensor,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: Optional[torch.Tensor],
    scale_ub: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    tokens = x.numel() // x.shape[-1]
    hidden = x.shape[-1]
    groups = (hidden + GROUP_SIZE - 1) // GROUP_SIZE

    out = torch.empty_like(x, dtype=quant_dtype)
    out_norm = torch.empty_like(x)
    scales = torch.empty((tokens, groups), device=x.device, dtype=torch.float32)

    residual_arg = residual.clone() if residual is not None else None

    torch.ops.sgl_kernel.rms_norm_dynamic_per_group_quant(
        out,
        out_norm,
        x,
        weight,
        scales,
        GROUP_SIZE,
        EPS,
        scale_ub,
        residual_arg,
    )

    return out, out_norm, scales, residual_arg


def check_result(
    ref_q: torch.Tensor,
    ref_norm: torch.Tensor,
    ref_scales: torch.Tensor,
    ref_residual: Optional[torch.Tensor],
    ops_q: torch.Tensor,
    ops_norm: torch.Tensor,
    ops_scales: torch.Tensor,
    ops_residual: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    add_residual: bool,
) -> None:
    assert ops_q.dtype == quant_dtype
    assert ops_norm.dtype == ref_norm.dtype
    assert ops_scales.dtype == torch.float32

    torch.testing.assert_close(
        ops_norm.float(),
        ref_norm.float(),
        atol=2e-3,
        rtol=2e-3,
    )

    torch.testing.assert_close(
        ops_scales.float(),
        ref_scales.float(),
        atol=2e-5,
        rtol=2e-4,
    )

    if add_residual:
        assert ops_residual is not None
        assert ref_residual is not None
        torch.testing.assert_close(
            ops_residual.float(),
            ref_residual.float(),
            atol=2e-3,
            rtol=2e-3,
        )
    else:
        assert ops_residual is None

    if quant_dtype == torch.int8:
        # INT8 result should be almost bitwise identical. Allow 1 because
        # different round paths can differ by one integer.
        max_q_err = (ops_q.to(torch.int16) - ref_q.to(torch.int16)).abs().max().item()
        assert max_q_err <= 1
        return

    # FP8: compare kernel cast result against PyTorch FP8 cast reference
    # after dequantization. This validates scale/group/cast behavior.
    hidden = ref_norm.shape[-1]
    expanded_scales = expand_group_scales(ref_scales, hidden, GROUP_SIZE)

    ops_dequant = ops_q.float() * expanded_scales
    ref_dequant = ref_q.float() * expanded_scales

    max_cast_err = (ops_dequant - ref_dequant).abs().max().item()
    assert max_cast_err <= 5e-2

    # Natural FP8 quantization error versus unquantized norm.
    max_abs_ref = ref_norm.float().abs().max().item()
    natural_tol = max(0.20, 0.07 * max_abs_ref)
    max_natural_err = (ops_dequant - ref_norm.float()).abs().max().item()
    assert max_natural_err <= natural_tol


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens,hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("scale_ub_enabled", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_dynamic_per_group_quant(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    scale_ub_enabled: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    if scale_ub_enabled and not is_fp8_dtype(quant_dtype):
        pytest.skip("scale_ub is only valid for fp8 output")

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(device)

    weight = torch.empty(hidden_size, device=device, dtype=dtype)
    weight.normal_(mean=1.0, std=0.1)

    # Keep values moderate to make reference stable and avoid saturation-heavy cases.
    scale = 1.0 / hidden_size
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * scale
    residual = torch.randn_like(x) * scale if add_residual else None

    ref_norm, ref_residual = rms_norm_ref(x, weight, EPS, residual)

    # Current kernel accepts scale_ub for FP8. The reference scale here follows
    # the kernel's dynamic per-group formula. scale_ub is passed only to cover
    # the API path; it should not change expected group scales in this test.
    scale_ub = None
    if scale_ub_enabled:
        scale_ub = torch.mean(ref_norm.abs()).to(dtype=torch.float32, device=device).reshape(1)

    ref_q, ref_scales = dynamic_per_group_quant_ref(ref_norm, quant_dtype, GROUP_SIZE)

    ops_q, ops_norm, ops_scales, ops_residual = ops_impl(
        weight=weight,
        x=x,
        quant_dtype=quant_dtype,
        residual=residual,
        scale_ub=scale_ub,
    )

    check_result(
        ref_q=ref_q,
        ref_norm=ref_norm.to(dtype),
        ref_scales=ref_scales,
        ref_residual=ref_residual.to(dtype) if ref_residual is not None else None,
        ops_q=ops_q,
        ops_norm=ops_norm,
        ops_scales=ops_scales,
        ops_residual=ops_residual,
        quant_dtype=quant_dtype,
        add_residual=add_residual,
    )
