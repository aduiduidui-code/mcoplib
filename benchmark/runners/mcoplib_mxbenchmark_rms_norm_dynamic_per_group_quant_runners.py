import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


def _ref_rms_norm(x, weight, eps):
    x_fp32 = x.float()
    variance = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    return x_fp32 * rstd * weight.float()


def _ref_quantize_int8(norm, group_size):
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


class Rms_norm_dynamic_per_group_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 32)
        self.hidden_size = config.get("hidden_size", 7168)
        self.group_size = config.get("group_size", 128)
        self.eps = config.get("eps", 1e-6)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.num_tokens} {self.hidden_size})"
        state.add_summary("Shape", shape_str)

        in_elems = self.num_tokens * self.hidden_size
        groups = (self.hidden_size + self.group_size - 1) // self.group_size
        state.add_element_count(in_elems * 3 + self.num_tokens * groups)

        es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = in_elems * es * 2
        writes = in_elems * 1 + in_elems * es + self.num_tokens * groups * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            x = torch.randn(self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev)
            weight = torch.randn(self.hidden_size, dtype=self.dtype, device=dev)
            out = torch.empty_like(x, dtype=torch.int8)
            out_norm = torch.empty_like(x)
            groups = (self.hidden_size + self.group_size - 1) // self.group_size
            scales = torch.empty((self.num_tokens, groups), dtype=torch.float32, device=dev)
        return self.make_launcher(
            dev_id, torch.ops.sgl_kernel.rms_norm_dynamic_per_group_quant,
            out, out_norm, x, weight, scales, self.group_size, self.eps, None, None
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        N, H = 16, 4096
        torch.manual_seed(42)
        x = torch.randn(N, H, dtype=self.dtype, device=dev)
        weight = torch.randn(H, dtype=self.dtype, device=dev)
        out = torch.empty_like(x, dtype=torch.int8)
        out_norm = torch.empty_like(x)
        groups = (H + self.group_size - 1) // self.group_size
        scales = torch.empty((N, groups), dtype=torch.float32, device=dev)
        torch.ops.sgl_kernel.rms_norm_dynamic_per_group_quant(
            out, out_norm, x, weight, scales, self.group_size, self.eps, None, None
        )
        normed_ref = _ref_rms_norm(x, weight, self.eps)
        ref_out, ref_scales = _ref_quantize_int8(normed_ref, self.group_size)
        q_match = (out == ref_out).float().mean().item()
        scales_match = torch.allclose(scales, ref_scales)
        norm_match = self.check_diff(out_norm.float(), normed_ref.float(), threshold=0.9999)
        passed = q_match >= 0.99 and scales_match and norm_match[0]
        err = (1.0 - q_match) if not passed else 0.0
        return passed, err
