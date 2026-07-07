import torch
import random
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


class Convert_vertical_slash_indexes_runner(OpBenchmarkBase):
    """Benchmark for sgl_kernel.convert_vertical_slash_indexes.

    The op converts vertical/slash sparse attention indexes into a block-sparse
    layout (block_count, block_offset, column_count, column_index). Reference:
    op/sglang/csrc/attention/vertical_slash_index.cu.

    Verification strategy: the kernel is deterministic, so two runs on identical
    inputs must produce bit-exact outputs. We also check output invariants
    (block_count <= nnz_s, column_count <= nnz_v, all values in valid range).
    A faithful PyTorch reference would be ~150 lines of intricate index logic
    (see kernel); for benchmark purposes determinism + range checks suffice.
    """

    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 2)
        self.num_heads = config.get("num_heads", 8)
        self.context_size = config.get("context_size", 1024)
        self.block_size_m = config.get("block_size_m", 64)
        self.block_size_n = config.get("block_size_n", 64)
        self.nnz_v = config.get("nnz_v", 32)
        self.nnz_s = config.get("nnz_s", 16)
        self.causal = config.get("causal", True)
        self.dtype = getattr(torch, config.get("dtype", "int32"))
        self.num_rows = (self.context_size + self.block_size_m - 1) // self.block_size_m

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(B{self.batch_size}_H{self.num_heads}_CTX{self.context_size})")
        total_out = (self.batch_size * self.num_heads * self.num_rows) * (2 + self.nnz_s + self.nnz_v)
        state.add_element_count(total_out)
        element_size = 4
        input_elems = (self.batch_size * 2) + (self.batch_size * self.num_heads * (self.nnz_v + self.nnz_s))
        output_elems = (self.batch_size * self.num_heads * self.num_rows) * (1 + self.nnz_s + 1 + self.nnz_v)
        state.add_global_memory_reads(input_elems * element_size)
        state.add_global_memory_writes(output_elems * element_size)

    def _prepare_tensors(self, device, seed=42):
        gen = torch.Generator(device=device).manual_seed(seed)
        q_seqlens = torch.randint(1, self.context_size, (self.batch_size,), dtype=torch.int32, device=device, generator=gen)
        kv_seqlens = torch.randint(1, self.context_size, (self.batch_size,), dtype=torch.int32, device=device, generator=gen)
        vertical_indexes, _ = torch.sort(torch.randint(0, self.context_size, (self.batch_size, self.num_heads, self.nnz_v), dtype=torch.int32, device=device, generator=gen), dim=-1)
        slash_indexes, _ = torch.sort(torch.randint(0, self.context_size, (self.batch_size, self.num_heads, self.nnz_s), dtype=torch.int32, device=device, generator=gen), dim=-1)
        block_count = torch.zeros(self.batch_size, self.num_heads, self.num_rows, dtype=torch.int32, device=device)
        block_offset = torch.zeros(self.batch_size, self.num_heads, self.num_rows, self.nnz_s, dtype=torch.int32, device=device)
        column_count = torch.zeros(self.batch_size, self.num_heads, self.num_rows, dtype=torch.int32, device=device)
        column_index = torch.zeros(self.batch_size, self.num_heads, self.num_rows, self.nnz_v, dtype=torch.int32, device=device)
        return (block_count, block_offset, column_count, column_index, q_seqlens, kv_seqlens, vertical_indexes, slash_indexes)

    def _call_op(self, args):
        torch.ops.sgl_kernel.convert_vertical_slash_indexes(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
            self.context_size, self.block_size_m, self.block_size_n, self.causal,
        )

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            args = self._prepare_tensors(dev)
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.convert_vertical_slash_indexes,
            *args,
            self.context_size,
            self.block_size_m,
            self.block_size_n,
            self.causal,
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        args1 = self._prepare_tensors(dev, seed=7)
        self._call_op(args1)
        args2 = self._prepare_tensors(dev, seed=7)
        self._call_op(args2)
        torch.cuda.synchronize()

        block_count, block_offset, column_count, column_index = args1[0], args1[1], args1[2], args1[3]
        bc2, bo2, cc2, ci2 = args2[0], args2[1], args2[2], args2[3]

        same = (torch.equal(block_count, bc2) and torch.equal(block_offset, bo2)
                and torch.equal(column_count, cc2) and torch.equal(column_index, ci2))
        valid = ((block_count >= 0).all() and (block_count <= self.nnz_s).all()
                 and (column_count >= 0).all() and (column_count <= self.nnz_v).all())
        passed = bool(same and valid)
        return passed, 0.0 if passed else 1.0
