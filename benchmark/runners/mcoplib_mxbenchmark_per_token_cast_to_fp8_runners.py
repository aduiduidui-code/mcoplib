import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


class Per_token_cast_to_fp8_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.m = config.get("m", 4096)
        self.n = config.get("n", 4096)
        self.group_size = 128
        self.fp8_max = 448.0

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bf16->fp8")
        shape_str = f"({self.m} {self.n})"
        state.add_summary("Shape", shape_str)

        in_elems = self.m * self.n
        out_elems = self.m * self.n
        scale_elems = self.m * (self.n // self.group_size)
        state.add_element_count(in_elems + out_elems + scale_elems)

        in_es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = in_elems * in_es
        writes = out_elems * 1 + scale_elems * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            x = torch.randn(self.m, self.n, dtype=self.dtype, device=dev)
            out = torch.empty(self.m, self.n, dtype=torch.float8_e4m3fn, device=dev)
            scale = torch.empty(
                self.m, self.n // self.group_size, device=dev, dtype=torch.float32
            )
        return self.make_launcher(
            dev_id, torch.ops.sgl_kernel.per_token_cast_to_fp8.default,
            out, scale, x
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        m, n = self.m, self.n
        torch.manual_seed(42)
        x = torch.randn(m, n, dtype=self.dtype, device=dev)
        out = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=dev)
        scale = torch.empty(m, n // self.group_size, dtype=torch.float32, device=dev)
        torch.ops.sgl_kernel.per_token_cast_to_fp8.default(out, scale, x)

        # Reference: per-token-group (group=128) amax scaling into fp8_e4m3fn
        x_view = x.view(m, -1, self.group_size)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        scale_ref = (x_amax / self.fp8_max).view(m, -1)

        # The unit test only asserts the scale tensor; replicate that check.
        try:
            torch.testing.assert_close(scale, scale_ref, atol=1e-5, rtol=1e-5)
            return True, 0.0
        except AssertionError:
            return self.check_diff(scale.float(), scale_ref.float(), threshold=0.9999)
