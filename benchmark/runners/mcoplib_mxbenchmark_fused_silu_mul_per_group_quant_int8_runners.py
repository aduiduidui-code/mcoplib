import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel  # noqa: F401; registers torch.ops.sgl_kernel
except ImportError:
    pass


GROUP_SIZE_DEFAULT = 128
COS_EPS = 1.0e-12


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


def _qmax_and_min_scale(quant_dtype):
    if quant_dtype == torch.int8:
        qmax = 127.0
        min_absmax = qmax * torch.finfo(torch.float32).eps
        return qmax, min_absmax / qmax

    qmax = float(torch.finfo(quant_dtype).max)
    return qmax, (1.0 / 512.0) / qmax


def _make_input(shape, dtype, dev):
    return (torch.randn(shape, device=dev, dtype=torch.float32) * 0.5).to(dtype)


def _reference(input_tensor, quant_dtype, group_size):
    input_hidden = input_tensor.size(-1)
    hidden = input_hidden // 2
    tokens = input_tensor.numel() // input_hidden
    groups = hidden // group_size

    x = input_tensor.reshape(tokens, input_hidden).float()
    gate = x[:, :hidden]
    up = x[:, hidden:]

    y = F.silu(gate) * up
    grouped = y.view(tokens, groups, group_size)
    amax = grouped.abs().amax(dim=-1)

    qmax, min_scale = _qmax_and_min_scale(quant_dtype)
    scales_ref = torch.clamp(amax / qmax, min=min_scale)
    scaled = grouped / scales_ref.unsqueeze(-1)

    if quant_dtype == torch.int8:
        out_ref = torch.round(scaled).clamp(-127, 127).to(torch.int8)
    else:
        out_ref = scaled.to(quant_dtype)

    return (
        out_ref.view(*input_tensor.shape[:-1], hidden),
        scales_ref,
        y.view(*input_tensor.shape[:-1], hidden),
    )


def _dequantize(out, scales, tokens, groups, group_size):
    return (
        out.float().reshape(tokens, groups, group_size)
        * scales.unsqueeze(-1)
    )


def _cosine_similarity(actual, expected):
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    numerator = torch.dot(actual_flat, expected_flat)
    denominator = actual_flat.norm() * expected_flat.norm()
    if denominator.item() < COS_EPS:
        return 1.0 if actual_flat.norm().item() < COS_EPS else 0.0
    return (numerator / denominator).item()


class Fused_silu_mul_per_group_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.hidden_size = config.get("hidden_size", 4096)
        self.group_size = config.get("group_size", GROUP_SIZE_DEFAULT)
        self.input_dtype = _get_dtype(config.get("input_dtype", "float16"))
        self.quant_dtype = _get_dtype(config.get("quant_dtype", "int8"))
        self.cos_threshold = config.get("cos_threshold", 0.999)

        if self.hidden_size % self.group_size != 0:
            raise ValueError("hidden_size must be divisible by group_size")

        self.groups = self.hidden_size // self.group_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary(
            "dtype",
            f"{str(self.input_dtype).removeprefix('torch.')}->{str(self.quant_dtype).removeprefix('torch.')}",
        )
        state.add_summary(
            "Shape",
            f"({self.num_tokens} {self.hidden_size * 2}) -> ({self.num_tokens} {self.hidden_size})",
        )

        total_out_elements = self.num_tokens * self.hidden_size
        state.add_element_count(total_out_elements)

        input_elem_size = 2 if self.input_dtype in (torch.float16, torch.bfloat16) else 4
        output_elem_size = 1
        reads = self.num_tokens * self.hidden_size * 2 * input_elem_size
        writes = total_out_elements * output_elem_size
        writes += self.num_tokens * self.groups * 4

        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            input_shape = (self.num_tokens, self.hidden_size * 2)
            input_tensor = _make_input(input_shape, self.input_dtype, dev)
            out = torch.empty(
                (self.num_tokens, self.hidden_size),
                device=dev,
                dtype=self.quant_dtype,
            )
            scales = torch.empty(
                (self.num_tokens, self.groups),
                device=dev,
                dtype=torch.float32,
            )

        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.fused_silu_mul_per_group_quant,
            out,
            scales,
            input_tensor,
        )

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        tokens = 16
        hidden = 512
        groups = hidden // self.group_size
        input_shape = (tokens, hidden * 2)

        input_tensor = _make_input(input_shape, self.input_dtype, dev)
        out = torch.empty((tokens, hidden), device=dev, dtype=self.quant_dtype)
        scales = torch.empty((tokens, groups), device=dev, dtype=torch.float32)

        torch.ops.sgl_kernel.fused_silu_mul_per_group_quant(
            out,
            scales,
            input_tensor,
        )

        _, scales_ref, y_ref = _reference(input_tensor, self.quant_dtype, self.group_size)

        pass_scales, diff_scales = self.check_diff(scales, scales_ref, threshold=0.999999)

        dequant = _dequantize(out, scales, tokens, groups, self.group_size).reshape_as(y_ref)
        cos = _cosine_similarity(dequant, y_ref)
        pass_out = cos >= self.cos_threshold
        diff_out = 1.0 - cos

        return pass_scales and pass_out, max(diff_scales, diff_out)

