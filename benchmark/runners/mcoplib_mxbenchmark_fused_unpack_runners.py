import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fused_unpack_runner(OpBenchmarkBase):
    """Benchmark for torch.ops._C.fused_unpack.

    Unpacks a packed (M, 2*topk + n) fp32 tensor into:
      - weights (M, topk) fp32  (cols 0..topk-1)
      - ids     (M, topk) int32 (cols topk..2*topk-1, float->int32)
      - scale   (M, n)    fp32  (cols 2*topk..2*topk+n-1)

    Reference (from unit_test/test_fused_unpack.py) requires exact equality.
    """

    def __init__(self, name, config):
        super().__init__(name, config)
        self.m = config.get("m", 8192)
        self.topk = config.get("topk", 16)
        self.n = config.get("n", 1)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.m} {self.topk} {self.n})"
        state.add_summary("Shape", shape_str)

        packed_cols = 2 * self.topk + self.n
        in_elems = self.m * packed_cols
        out_w_elems = self.m * self.topk
        out_i_elems = self.m * self.topk
        out_s_elems = self.m * self.n
        state.add_element_count(in_elems + out_w_elems + out_i_elems + out_s_elems)

        # All tensors are fp32 (4 bytes) except ids which is int32 (4 bytes).
        reads = in_elems * 4
        writes = (out_w_elems + out_i_elems + out_s_elems) * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            packed_cols = 2 * self.topk + self.n
            packed = torch.randn(self.m, packed_cols, device=dev, dtype=torch.float32)
            weights = torch.empty(self.m, self.topk, device=dev, dtype=torch.float32)
            ids = torch.empty(self.m, self.topk, device=dev, dtype=torch.int32)
            scale = torch.empty(self.m, self.n, device=dev, dtype=torch.float32)
        return self.make_launcher(
            dev_id, torch.ops._C.fused_unpack,
            packed, self.topk, self.n, weights, ids, scale
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        # Use the configured shape for verification.
        M, topk, n = self.m, self.topk, self.n
        packed_cols = 2 * topk + n
        packed = torch.randn(M, packed_cols, device=dev, dtype=torch.float32)
        weights = torch.empty(M, topk, device=dev, dtype=torch.float32)
        ids = torch.empty(M, topk, device=dev, dtype=torch.int32)
        scale = torch.empty(M, n, device=dev, dtype=torch.float32)
        torch.ops._C.fused_unpack(packed, topk, n, weights, ids, scale)

        # Reference (exact equality per unit test):
        #   weights = packed[:, :topk]
        #   ids     = packed[:, topk:2*topk].to(int32)
        #   scale   = packed[:, 2*topk:]
        ref_w = packed[:, :topk].contiguous()
        ref_i = packed[:, topk:2 * topk].contiguous().to(torch.int32)
        ref_s = packed[:, 2 * topk:].contiguous()

        w_match = torch.all(weights == ref_w).item()
        i_match = torch.all(ids == ref_i).item()
        s_match = torch.all(scale == ref_s).item()
        passed = w_match and i_match and s_match
        return passed, 0.0
