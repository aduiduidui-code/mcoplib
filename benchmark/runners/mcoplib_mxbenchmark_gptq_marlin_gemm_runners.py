import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib
    from mcoplib import op as ops
    from mcoplib.marlin_utils import marlin_quantize
except ImportError:
    pass


class Gptq_marlin_gemm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.size_m = config.get("size_m", 64)
        self.size_k = config.get("size_k", 2048)
        self.size_n = config.get("size_n", 7168)
        self.group_size = config.get("group_size", 64)
        self.num_bits = config.get("num_bits", 4)
        self.scalar_t = torch.bfloat16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.size_m} {self.size_k} {self.size_n})"
        state.add_summary("Shape", shape_str)

        x_elems = self.size_m * self.size_k
        w_elems = self.size_k * self.size_n
        out_elems = self.size_m * self.size_n
        state.add_element_count(x_elems + w_elems + out_elems)

        element_size = 2
        reads = x_elems * element_size + (w_elems * self.num_bits // 8)
        writes = out_elems * element_size
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def _make_inputs(self, dev, size_m, size_k, size_n):
        weight = (torch.rand(size_k, size_n, device=dev) - 0.5) / 10
        x = (torch.rand(size_m, size_k, device=dev) - 0.5) / 10
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            weight, self.num_bits, self.group_size, act_order=False
        )
        w_ref = w_ref.to(self.scalar_t).to(dev)
        marlin_s = marlin_s.to(self.scalar_t).to(torch.half)
        x_half = x.to(self.scalar_t).to(dev).to(torch.half)
        return x_half, marlin_q_w, marlin_s, g_idx, sort_indices, w_ref

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            x, marlin_q_w, marlin_s, g_idx, sort_indices, _ = self._make_inputs(
                dev, self.size_m, self.size_k, self.size_n
            )
            ept = torch.empty(1, device=dev)
            bsz_tensor = torch.tensor([self.size_m], dtype=torch.int, device=dev)
            is_k_full = True
        return self.make_launcher(
            dev_id, ops.gptq_marlin_gemm,
            x, marlin_q_w, marlin_s, g_idx, sort_indices, ept,
            self.num_bits, bsz_tensor, self.size_m, self.size_n, x.shape[-1],
            -1, is_k_full, self.scalar_t, True
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        size_m, size_k, size_n = 64, 2048, 7168
        x, marlin_q_w, marlin_s, g_idx, sort_indices, w_ref = self._make_inputs(
            dev, size_m, size_k, size_n
        )
        ept = torch.empty(1, device=dev)
        bsz_tensor = torch.tensor([size_m], dtype=torch.int, device=dev)
        out = ops.gptq_marlin_gemm(
            x, marlin_q_w, marlin_s, g_idx, sort_indices, ept,
            self.num_bits, bsz_tensor, size_m, size_n, x.shape[-1],
            -1, True, self.scalar_t, True
        )
        out_ref = x @ w_ref.to(torch.half)
        # GPTQ-Marlin is a lossy 4-bit quantized GEMM. The unit test uses
        # max/mean absolute error criterion (error_ave ~0.042, error_max ~0.25).
        mean_err = (out.float() - out_ref.float()).abs().mean().item()
        passed = mean_err < 0.1
        return passed, mean_err
