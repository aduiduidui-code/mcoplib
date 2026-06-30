# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch

import mcoplib.sgl_kernel  # noqa: F401


EPS = 1e-6
GROUP_SIZE = 128


def get_fp8_dtype() -> Optional[torch.dtype]:
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return None


@dataclass
class Case:
    name: str
    mutate: Callable[[dict], None]
    match: str


def make_valid_args(
    *,
    device: str,
    input_dtype: torch.dtype = torch.float16,
    quant_dtype: torch.dtype = torch.int8,
    tokens: int = 4,
    hidden: int = 1024,
    with_residual: bool = False,
    with_scale_ub: bool = False,
) -> dict:
    groups = (hidden + GROUP_SIZE - 1) // GROUP_SIZE

    x = torch.randn(tokens, hidden, device=device, dtype=input_dtype) * (1.0 / hidden)
    weight = torch.randn(hidden, device=device, dtype=input_dtype)
    out = torch.empty_like(x, dtype=quant_dtype)
    out_norm = torch.empty_like(x)
    scales = torch.empty((tokens, groups), device=device, dtype=torch.float32)

    residual = torch.randn_like(x) if with_residual else None
    scale_ub = torch.ones((1,), device=device, dtype=torch.float32) if with_scale_ub else None

    return {
        "out": out,
        "out_norm": out_norm,
        "input": x,
        "weight": weight,
        "scales": scales,
        "quant_group_size": GROUP_SIZE,
        "variance_epsilon": EPS,
        "scale_ub": scale_ub,
        "residual": residual,
    }


def call_op(args: dict) -> None:
    torch.ops.sgl_kernel.rms_norm_dynamic_per_group_quant(
        args["out"],
        args["out_norm"],
        args["input"],
        args["weight"],
        args["scales"],
        args["quant_group_size"],
        args["variance_epsilon"],
        args["scale_ub"],
        args["residual"],
    )


def assert_invalid(case: Case, device: str) -> None:
    args = make_valid_args(device=device)
    case.mutate(args)

    with pytest.raises((RuntimeError, AssertionError), match=case.match):
        call_op(args)

    print(f"PASS shape_check name={case.name}")


def set_out_shape_bad(args: dict) -> None:
    x = args["input"]
    args["out"] = torch.empty((x.shape[0], x.shape[1] + 128), device=x.device, dtype=args["out"].dtype)


def set_out_norm_shape_bad(args: dict) -> None:
    x = args["input"]
    args["out_norm"] = torch.empty((x.shape[0], x.shape[1] + 128), device=x.device, dtype=x.dtype)


def set_weight_shape_bad(args: dict) -> None:
    x = args["input"]
    args["weight"] = torch.empty((x.shape[-1] + 1,), device=x.device, dtype=x.dtype)


def set_scales_shape_bad(args: dict) -> None:
    x = args["input"]
    groups = (x.shape[-1] + GROUP_SIZE - 1) // GROUP_SIZE
    args["scales"] = torch.empty((x.shape[0], groups + 1), device=x.device, dtype=torch.float32)


def set_scales_flattened(args: dict) -> None:
    x = args["input"]
    groups = (x.shape[-1] + GROUP_SIZE - 1) // GROUP_SIZE
    args["scales"] = torch.empty((x.shape[0] * groups,), device=x.device, dtype=torch.float32)


def set_residual_shape_bad(args: dict) -> None:
    x = args["input"]
    args["residual"] = torch.empty((x.shape[0], x.shape[1] + 128), device=x.device, dtype=x.dtype)


def set_out_dtype_bad(args: dict) -> None:
    x = args["input"]
    args["out"] = torch.empty_like(x, dtype=x.dtype)


def set_out_norm_dtype_bad(args: dict) -> None:
    x = args["input"]
    args["out_norm"] = torch.empty_like(x, dtype=torch.float32)


def set_weight_dtype_bad(args: dict) -> None:
    x = args["input"]
    args["weight"] = args["weight"].float()


def set_scales_dtype_bad(args: dict) -> None:
    args["scales"] = args["scales"].half()


def set_residual_dtype_bad(args: dict) -> None:
    x = args["input"]
    args["residual"] = torch.empty_like(x, dtype=torch.float32)


def set_input_non_contiguous(args: dict) -> None:
    x = args["input"]
    base = torch.empty(
        (x.shape[0], x.shape[1], 2),
        device=x.device,
        dtype=x.dtype,
    )
    args["input"] = base[:, :, 0]

    assert args["input"].shape == x.shape
    assert not args["input"].is_contiguous()


def set_weight_non_contiguous(args: dict) -> None:
    x = args["input"]
    hidden = x.shape[-1]

    base = torch.empty((hidden, 2), device=x.device, dtype=x.dtype)
    args["weight"] = base[:, 0]

    assert args["weight"].shape == (hidden,)
    assert not args["weight"].is_contiguous()


def set_out_non_contiguous(args: dict) -> None:
    x = args["input"]
    base = torch.empty(
        (x.shape[0], x.shape[1], 2),
        device=x.device,
        dtype=args["out"].dtype,
    )
    args["out"] = base[:, :, 0]

    assert args["out"].shape == x.shape
    assert not args["out"].is_contiguous()


def set_scale_ub_for_int8(args: dict) -> None:
    args["scale_ub"] = torch.ones((1,), device=args["input"].device, dtype=torch.float32)


def set_scale_ub_shape_bad(args: dict) -> None:
    fp8_dtype = get_fp8_dtype()
    if fp8_dtype is None:
        pytest.skip("fp8 dtype is unavailable")
    x = args["input"]
    args["out"] = torch.empty_like(x, dtype=fp8_dtype)
    args["scale_ub"] = torch.ones((2,), device=x.device, dtype=torch.float32)


def set_scale_ub_dtype_bad(args: dict) -> None:
    fp8_dtype = get_fp8_dtype()
    if fp8_dtype is None:
        pytest.skip("fp8 dtype is unavailable")
    x = args["input"]
    args["out"] = torch.empty_like(x, dtype=fp8_dtype)
    args["scale_ub"] = torch.ones((1,), device=x.device, dtype=torch.float16)


def set_epsilon_negative(args: dict) -> None:
    args["variance_epsilon"] = -1.0


def set_epsilon_nan(args: dict) -> None:
    args["variance_epsilon"] = float("nan")


INVALID_CASES = [
    Case(
        "invalid_group_size",
        lambda a: a.__setitem__("quant_group_size", 64),
        "quant_group_size|group",
    ),
    Case(
        "out_shape_mismatch",
        set_out_shape_bad,
        "out.*shape|shape.*out",
    ),
    Case(
        "out_norm_shape_mismatch",
        set_out_norm_shape_bad,
        "out_norm.*shape|shape.*out_norm",
    ),
    Case(
        "weight_shape_mismatch",
        set_weight_shape_bad,
        "weight.*shape|hidden|weight",
    ),
    Case(
        "scales_shape_mismatch",
        set_scales_shape_bad,
        "scales.*shape|scale.*shape",
    ),
    Case(
        "flattened_scales_rejected",
        set_scales_flattened,
        "scales.*dim|scales.*shape|2D",
    ),
    Case(
        "residual_shape_mismatch",
        set_residual_shape_bad,
        "residual.*shape|shape.*residual",
    ),
    Case(
        "out_dtype_bad",
        set_out_dtype_bad,
        "out.*dtype|int8|float8|fp8",
    ),
    Case(
        "out_norm_dtype_bad",
        set_out_norm_dtype_bad,
        "out_norm.*dtype|dtype.*out_norm",
    ),
    Case(
        "weight_dtype_bad",
        set_weight_dtype_bad,
        "weight.*dtype|dtype.*weight",
    ),
    Case(
        "scales_dtype_bad",
        set_scales_dtype_bad,
        "scales.*dtype|float32",
    ),
    Case(
        "residual_dtype_bad",
        set_residual_dtype_bad,
        "residual.*dtype|dtype.*residual",
    ),
    Case(
        "input_non_contiguous",
        set_input_non_contiguous,
        "contiguous",
    ),
    Case(
        "weight_non_contiguous",
        set_weight_non_contiguous,
        "contiguous",
    ),
    Case(
        "out_non_contiguous",
        set_out_non_contiguous,
        "contiguous",
    ),
    Case(
        "scale_ub_for_int8",
        set_scale_ub_for_int8,
        "scale_ub|fp8|float8",
    ),
    Case(
        "scale_ub_shape_bad",
        set_scale_ub_shape_bad,
        "scale_ub.*numel|scale_ub.*shape|scale_ub",
    ),
    Case(
        "scale_ub_dtype_bad",
        set_scale_ub_dtype_bad,
        "scale_ub.*dtype|float32",
    ),
    Case(
        "epsilon_negative",
        set_epsilon_negative,
        "epsilon|variance_epsilon",
    ),
    Case(
        "epsilon_nan",
        set_epsilon_nan,
        "epsilon|variance_epsilon|finite",
    ),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("device", [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() <= 1 else 2)])
@torch.inference_mode()
def test_valid_shape_check(device: str) -> None:
    torch.cuda.set_device(device)

    for input_dtype in [torch.float16, torch.bfloat16]:
        args = make_valid_args(device=device, input_dtype=input_dtype, quant_dtype=torch.int8)
        call_op(args)
        print(f"PASS shape_check name=valid_int8_input_{input_dtype}")

    fp8_dtype = get_fp8_dtype()
    if fp8_dtype is not None:
        args = make_valid_args(
            device=device,
            input_dtype=torch.float16,
            quant_dtype=fp8_dtype,
            with_residual=True,
            with_scale_ub=True,
        )
        call_op(args)
        print("PASS shape_check name=valid_fp8_with_residual_scale_ub")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("device", [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() <= 1 else 2)])
@torch.inference_mode()
def test_invalid_shape_check(case: Case, device: str) -> None:
    torch.cuda.set_device(device)
    assert_invalid(case, device)
