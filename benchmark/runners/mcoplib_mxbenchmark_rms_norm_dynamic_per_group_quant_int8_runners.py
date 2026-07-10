import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel  # noqa: F401; registers torch.ops.sgl_kernel
except ImportError:
    pass


GROUP_SIZE_DEFAULT = 128


def _get_dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    if name == "int8":
        return torch.int8
    if name == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this PyTorch build does not provide float8_e4m3fn")
        return torch.float8_e4m3fn
    if hasattr(torch, name):
        return getattr(torch, name)
    raise ValueError(f"unsupported dtype: {name}")


def _is_fp8_dtype(dtype):
    return hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn


def _make_input(shape, dtype, dev):
    scale = 1.0 / shape[-1]
    return (torch.randn(shape, device=dev, dtype=torch.float32) * scale).to(dtype)


def _make_weight(hidden, dtype, dev):
    weight = torch.empty(hidden, device=dev, dtype=dtype)
    weight.normal_(mean=1.0, std=0.1)
    return weight


def _rms_norm_ref(x, weight, eps, residual):
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


def _quantize_int8_ref(norm, group_size):
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


def _quantize_fp8_ref(norm, group_size, fp8_dtype):
    tokens, hidden = norm.shape
    groups = (hidden + group_size - 1) // group_size

    out = torch.empty_like(norm, dtype=fp8_dtype)
    scales = torch.empty((tokens, groups), dtype=torch.float32, device=norm.device)

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


def _dynamic_per_group_quant_ref(norm, quant_dtype, group_size):
    if quant_dtype == torch.int8:
        return _quantize_int8_ref(norm, group_size)
    if _is_fp8_dtype(quant_dtype):
        return _quantize_fp8_ref(norm, group_size, quant_dtype)
    raise AssertionError(f"unsupported quant dtype: {quant_dtype}")


def _expand_group_scales(scales, hidden, group_size):
    expanded = scales.repeat_interleave(group_size, dim=-1)
    return expanded[:, :hidden]


class Rms_norm_dynamic_per_group_quant_int8_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.hidden_size = config.get("hidden_size", 4096)
        self.group_size = config.get("group_size", GROUP_SIZE_DEFAULT)
        self.epsilon = config.get("epsilon", 1e-6)
        self.add_residual = config.get("add_residual", True)
        self.scale_ub_enabled = config.get("scale_ub_enabled", False)
        self.input_dtype = _get_dtype(config.get("dtype", "float16"))
        self.quant_dtype = _get_dtype(config.get("quant_dtype", "int8"))

        if self.group_size != GROUP_SIZE_DEFAULT:
            raise ValueError("group_size must be 128")
        if self.hidden_size > 16384:
            raise ValueError("hidden_size must be <= 16384")

        self.groups = (self.hidden_size + self.group_size - 1) // self.group_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary(
            "dtype",
            f"{str(self.input_dtype).removeprefix('torch.')}->{str(self.quant_dtype).removeprefix('torch.')}",
        )
        state.add_summary(
            "Shape",
            f"({self.num_tokens} {self.hidden_size}) -> ({self.num_tokens} {self.hidden_size})",
        )

        total_elements = self.num_tokens * self.hidden_size
        element_size = 2 if self.input_dtype in (torch.float16, torch.bfloat16) else 4
        quant_size = 1

        reads = total_elements * element_size
        reads += self.hidden_size * element_size
        if self.add_residual:
            reads += total_elements * element_size
        if self.scale_ub_enabled:
            reads += 4

        writes = total_elements * quant_size
        writes += total_elements * element_size
        writes += self.num_tokens * self.groups * 4
        if self.add_residual:
            writes += total_elements * element_size

        state.add_element_count(total_elements)
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def _make_tensors(self, dev, tokens, hidden):
        groups = (hidden + self.group_size - 1) // self.group_size
        x = _make_input((tokens, hidden), self.input_dtype, dev)
        weight = _make_weight(hidden, self.input_dtype, dev)
        residual = _make_input((tokens, hidden), self.input_dtype, dev) if self.add_residual else None
        out = torch.empty((tokens, hidden), device=dev, dtype=self.quant_dtype)
        out_norm = torch.empty((tokens, hidden), device=dev, dtype=self.input_dtype)
        scales = torch.empty((tokens, groups), device=dev, dtype=torch.float32)
        scale_ub = None
        if self.scale_ub_enabled:
            ref_norm, _ = _rms_norm_ref(x, weight, self.epsilon, residual)
            scale_ub = torch.mean(ref_norm.abs()).to(dtype=torch.float32, device=dev).reshape(1)
        residual_arg = residual.clone() if residual is not None else None
        return out, out_norm, x, weight, scales, scale_ub, residual_arg, residual

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            out, out_norm, x, weight, scales, scale_ub, residual_arg, _ = self._make_tensors(
                dev, self.num_tokens, self.hidden_size
            )

        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.rms_norm_dynamic_per_group_quant,
            out,
            out_norm,
            x,
            weight,
            scales,
            self.group_size,
            self.epsilon,
            scale_ub,
            residual_arg,
        )

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        tokens = 32
        hidden = 512
        out, out_norm, x, weight, scales, scale_ub, residual_arg, residual = self._make_tensors(
            dev, tokens, hidden
        )

        torch.ops.sgl_kernel.rms_norm_dynamic_per_group_quant(
            out,
            out_norm,
            x,
            weight,
            scales,
            self.group_size,
            self.epsilon,
            scale_ub,
            residual_arg,
        )

        ref_norm, ref_residual = _rms_norm_ref(x, weight, self.epsilon, residual)
        ref_q, ref_scales = _dynamic_per_group_quant_ref(ref_norm, self.quant_dtype, self.group_size)

        pass_norm, diff_norm = self.check_diff(out_norm.float(), ref_norm.to(self.input_dtype).float())
        pass_scales, diff_scales = self.check_diff(scales.float(), ref_scales.float())

        pass_residual = True
        diff_residual = 0.0
        if self.add_residual:
            pass_residual, diff_residual = self.check_diff(
                residual_arg.float(),
                ref_residual.to(self.input_dtype).float(),
            )

        if self.quant_dtype == torch.int8:
            max_q_err = (out.to(torch.int16) - ref_q.to(torch.int16)).abs().max().item()
            pass_q = max_q_err <= 1
            diff_q = float(max_q_err)
        else:
            expanded_scales = _expand_group_scales(ref_scales, hidden, self.group_size)
            out_dequant = out.float() * expanded_scales
            ref_dequant = ref_q.float() * expanded_scales
            max_cast_err = (out_dequant - ref_dequant).abs().max().item()

            max_abs_ref = ref_norm.float().abs().max().item()
            natural_tol = max(0.20, 0.07 * max_abs_ref)
            max_natural_err = (out_dequant - ref_norm.float()).abs().max().item()

            pass_q = max_cast_err <= 5e-2 and max_natural_err <= natural_tol
            diff_q = max(float(max_cast_err), float(max_natural_err))

        passed = pass_norm and pass_scales and pass_residual and pass_q
        diff = max(diff_norm, diff_scales, diff_residual, diff_q)
        return passed, diff

