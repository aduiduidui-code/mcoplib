import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Concat_mla_q_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.num_heads = config.get("num_heads", 128)
        self.nope_dim = config.get("nope_dim", 512)
        self.rope_dim = config.get("rope_dim", 64)
        self.total_dim = self.nope_dim + self.rope_dim

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.num_tokens} {self.num_heads} {self.total_dim})"
        state.add_summary("Shape", shape_str)

        nope_elems = self.num_tokens * self.num_heads * self.nope_dim
        pe_elems = self.num_tokens * self.num_heads * self.rope_dim
        out_elems = self.num_tokens * self.num_heads * self.total_dim
        state.add_element_count(nope_elems + pe_elems + out_elems)

        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = (nope_elems + pe_elems) * element_size
        writes = out_elems * element_size
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            ql_nope = torch.randn(
                self.num_tokens, self.num_heads, self.nope_dim,
                dtype=self.dtype, device=dev
            )
            q_pe = torch.randn(
                self.num_tokens, self.num_heads, self.rope_dim,
                dtype=self.dtype, device=dev
            )
            q_out = torch.empty(
                self.num_tokens, self.num_heads, self.total_dim,
                dtype=self.dtype, device=dev
            )
        return self.make_launcher(
            dev_id, torch.ops._C_cache_ops.concat_mla_q,
            ql_nope, q_pe, q_out
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        num_tokens, num_heads = 64, 16
        ql_nope = torch.randn(
            num_tokens, num_heads, self.nope_dim,
            dtype=self.dtype, device=dev
        )
        q_pe = torch.randn(
            num_tokens, num_heads, self.rope_dim,
            dtype=self.dtype, device=dev
        )
        q_out = torch.empty(
            num_tokens, num_heads, self.total_dim,
            dtype=self.dtype, device=dev
        )
        torch.ops._C_cache_ops.concat_mla_q(ql_nope, q_pe, q_out)
        out_ref = torch.cat([ql_nope, q_pe], dim=-1)
        return self.check_diff(q_out, out_ref)
