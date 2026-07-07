import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fp32_router_gemm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 32)
        self.hidden_dim = config.get("hidden_dim", 3072)
        self.num_experts = config.get("num_experts", 256)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.num_tokens} {self.hidden_dim} {self.num_experts})"
        state.add_summary("Shape", shape_str)

        in_elems = self.num_tokens * self.hidden_dim
        w_elems = self.num_experts * self.hidden_dim
        out_elems = self.num_tokens * self.num_experts
        state.add_element_count(in_elems + w_elems + out_elems)

        element_size = 4 if self.dtype == torch.float32 else 2
        reads = (in_elems * element_size) + (w_elems * 4)
        writes = out_elems * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            x = torch.randn(self.num_tokens, self.hidden_dim, dtype=self.dtype, device=dev)
            w = torch.randn(self.num_experts, self.hidden_dim, dtype=torch.float32, device=dev)
            output = torch.empty(self.num_tokens, self.num_experts, dtype=torch.float32, device=dev)
        return self.make_launcher(dev_id, torch.ops._C.fp32_router_gemm, output, x, w)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        M, H, E = 32, 3072, 256
        x = torch.randn(M, H, dtype=self.dtype, device=dev)
        w = torch.randn(E, H, dtype=torch.float32, device=dev)
        output = torch.empty(M, E, dtype=torch.float32, device=dev)
        torch.ops._C.fp32_router_gemm(output, x, w)
        out_ref = (x.float() @ w.float().t()).float()
        return self.check_diff(output, out_ref, threshold=0.9999)
