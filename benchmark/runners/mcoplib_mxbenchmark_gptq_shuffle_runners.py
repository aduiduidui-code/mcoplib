import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


def _ref_gptq_shuffle_4bit(q_weight, q_perm):
    """Pure PyTorch reference for _C.gptq_shuffle(q_weight, q_perm, 4).

    Two passes (mirroring op/vllm/quantization/gptq/q_gemm.cu):
      1. make_sequential_4bit: permute rows so that each packed int32 holds
         8 consecutive rows (in original space) according to q_perm.
      2. shuffle_4bit_8: rearrange the 8 4-bit nibbles inside each int32
         from [q0,q1,q2,q3,q4,q5,q6,q7] to [q0,q2,q4,q6,q1,q3,q5,q7].
    """
    height = q_perm.numel()
    out_features = q_weight.size(1)
    pack_factor = 8  # 32 / bits, bits=4
    packed_rows = height // pack_factor

    mask = 0x0F
    q_weight_u32 = q_weight.to(torch.int64) & 0xFFFFFFFF

    # Pass 1: make_sequential. For each output packed row j (0..packed_rows-1),
    # gather 8 source rows q_perm[8j .. 8j+7], pack their nibbles into one int32.
    perm_rows = q_perm.to(torch.int64)
    src_row_idx = perm_rows.view(packed_rows, pack_factor)  # [packed_rows, 8]

    # q_weight_u32: [packed_rows, out_features]. Each int32 holds 8 nibbles of
    # source row r at subrow (r & 7). For source row rsrc = 8*w2_row + w2_subrow,
    # the nibbles for column n are in q_weight_u32[w2_row, n] at bits [w2_subrow*4 .. w2_subrow*4+3].
    src_packed_row = src_row_idx // pack_factor  # [packed_rows, 8]
    src_subrow = src_row_idx % pack_factor       # [packed_rows, 8]
    src_shift = src_subrow * 4                   # [packed_rows, 8]

    # Gather source int32s: [packed_rows, 8, out_features]
    gathered = q_weight_u32[src_packed_row]  # [packed_rows, 8, out_features]
    nibbles = (gathered >> src_shift.unsqueeze(-1)) & mask  # [packed_rows, 8, out_features]
    # Reorder into new int32: nibble i goes to bits [i*4 .. i*4+3]
    out_shift = torch.arange(pack_factor, device=q_weight.device, dtype=torch.int64) * 4
    sequential = (nibbles << out_shift.view(1, pack_factor, 1)).sum(dim=1)  # [packed_rows, out_features]
    sequential = sequential.to(torch.int32)

    # Pass 2: shuffle_4bit_8. Per int32, permute nibbles [0,1,2,3,4,5,6,7] -> [0,2,4,6,1,3,5,7].
    s = sequential.to(torch.int64) & 0xFFFFFFFF
    n0 = (s >> 0) & mask
    n1 = (s >> 4) & mask
    n2 = (s >> 8) & mask
    n3 = (s >> 12) & mask
    n4 = (s >> 16) & mask
    n5 = (s >> 20) & mask
    n6 = (s >> 24) & mask
    n7 = (s >> 28) & mask
    out = (n0 << 0) | (n2 << 4) | (n4 << 8) | (n6 << 12) | \
          (n1 << 16) | (n3 << 20) | (n5 << 24) | (n7 << 28)
    return out.to(torch.int32)


class Gptq_shuffle_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.in_features = config.get("in_features", 1024)
        self.out_features = config.get("out_features", 4096)
        self.bits = config.get("bits", 4)
        assert self.bits == 4, "gptq_shuffle runner currently supports bits=4 only"
        self.pack_factor = 32 // self.bits
        self.packed_rows = self.in_features // self.pack_factor

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Bits", str(self.bits))
        state.add_summary("Shape", f"({self.in_features} {self.out_features})")
        weight_elements = self.packed_rows * self.out_features
        perm_elements = self.in_features
        total_read = (weight_elements + perm_elements) * 4
        total_write = weight_elements * 4
        state.add_global_memory_reads(total_read)
        state.add_global_memory_writes(total_write)
        state.add_element_count(weight_elements)

    def _prepare(self, dev_id, seed=42):
        dev = f'cuda:{dev_id}'
        gen = torch.Generator(device=dev).manual_seed(seed)
        q_weight = torch.randint(
            low=-2147483648,
            high=2147483647,
            size=(self.packed_rows, self.out_features),
            dtype=torch.int32,
            device=dev,
            generator=gen,
        )
        q_perm = torch.randperm(self.in_features, dtype=torch.int32, device=dev, generator=gen)
        return q_weight, q_perm

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            q_weight, q_perm = self._prepare(dev_id)
            # Pass an empty tensor for q_perm during benchmarking to skip the
            # expensive make_sequential pass (which calls cudaMalloc +
            # cudaDeviceSynchronize internally and is a one-shot weight-prep
            # transform, not a hot-loop kernel). Only the bit-shuffle pass runs.
            empty_perm = torch.empty(0, dtype=torch.int32, device=q_weight.device)
        return self.make_launcher(dev_id, torch.ops._C.gptq_shuffle, q_weight, empty_perm, self.bits)

    def run_verification(self, dev_id):
        q_weight, q_perm = self._prepare(dev_id, seed=7)
        q_weight_op = q_weight.clone()
        torch.ops._C.gptq_shuffle(q_weight_op, q_perm, self.bits)
        torch.cuda.synchronize()
        q_weight_ref = _ref_gptq_shuffle_4bit(q_weight, q_perm)
        # Exact bit-exact match expected.
        passed = bool(torch.equal(q_weight_op, q_weight_ref))
        return passed, 0.0 if passed else 1.0
