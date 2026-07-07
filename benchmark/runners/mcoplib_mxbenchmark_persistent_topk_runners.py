import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

RADIX_TOPK_WORKSPACE_PER_ROW = 772 * 4


class Persistent_topk_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_rows = config.get("num_rows", 1024)
        self.maxlen = config.get("maxlen", 65536)
        self.topk = config.get("topk", 1024)
        # Derive (bs, spec) so num_rows = bs * (1 + spec).
        self.bs = max(1, self.num_rows // 4)
        self.spec = (self.num_rows // self.bs) - 1
        self.next_n = 1 + self.spec
        # Re-derive num_rows for safety.
        self.num_rows = self.bs * self.next_n

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        shape_str = f"({self.num_rows} {self.maxlen} {self.topk})"
        state.add_summary("Shape", shape_str)

        in_elems = self.num_rows * self.maxlen
        out_elems = self.num_rows * self.topk
        state.add_element_count(in_elems + out_elems)

        element_size = 4
        reads = in_elems * element_size
        writes = out_elems * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            logits = torch.randn(self.num_rows, self.maxlen, dtype=torch.float32, device=dev)
            seq_lens = torch.randint(
                self.maxlen // 2, self.maxlen + 1,
                (self.bs, self.next_n), dtype=torch.int32, device=dev
            )
            workspace = torch.zeros(
                self.num_rows * RADIX_TOPK_WORKSPACE_PER_ROW,
                device=dev, dtype=torch.uint8
            )
            output = torch.zeros(self.num_rows, self.topk, dtype=torch.int32, device=dev)
        return self.make_launcher(
            dev_id, torch.ops._C.persistent_topk,
            logits, seq_lens, output, workspace, self.topk, self.maxlen
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        bs, spec, maxlen, topk = 4, 0, 16387, 512
        num_rows = bs * (1 + spec)
        torch.manual_seed(42)
        logits = torch.randn(num_rows, maxlen, dtype=torch.float32, device=dev)
        seq_lens = torch.randint(maxlen // 2, maxlen + 1, (bs, 1 + spec),
                                 dtype=torch.int32, device=dev)
        workspace = torch.zeros(
            num_rows * RADIX_TOPK_WORKSPACE_PER_ROW, device=dev, dtype=torch.uint8
        )
        output = torch.zeros(num_rows, topk, dtype=torch.int32, device=dev)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

        # Reference: per-row top-k over the first seq_lens[i] valid elements.
        seq_lens_flat = seq_lens.flatten().cpu().tolist()
        ref = torch.zeros(num_rows, topk, dtype=torch.int32, device=dev)
        for i in range(num_rows):
            L = seq_lens_flat[i]
            row = logits[i, :L]
            k = min(topk, L)
            _, idx = torch.topk(row, k)
            ref[i, :k] = idx.int()
        op_sorted = torch.sort(output, dim=-1).values
        ref_sorted = torch.sort(ref, dim=-1).values
        match_frac = (op_sorted == ref_sorted).float().mean().item()
        passed = match_frac >= 0.999
        return passed, 1.0 - match_frac
