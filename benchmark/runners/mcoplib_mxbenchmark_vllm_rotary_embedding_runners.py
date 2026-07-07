import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


def _torch_neox_rope(q, k, positions, cos_sin_cache, head_size, inverse=False):
    # cos_sin_cache: [max_position, head_size] (first half = cos, second = sin)
    # Same layout as sgl variant; vllm kernel uses identical NeoX rotation.
    num_tokens = q.shape[0]
    embed_dim = head_size // 2
    cos = cos_sin_cache[:, :embed_dim].float()
    sin = cos_sin_cache[:, embed_dim:2 * embed_dim].float()
    if inverse:
        sin = -sin

    def apply(x, pos):
        x = x.clone().float()
        x1 = x[..., :embed_dim]
        x2 = x[..., embed_dim:2 * embed_dim]
        c = cos[pos]  # [num_tokens, embed_dim]
        s = sin[pos]
        c_b = c.unsqueeze(1).expand_as(x1)
        s_b = s.unsqueeze(1).expand_as(x1)
        o1 = x1 * c_b - x2 * s_b
        o2 = x2 * c_b + x1 * s_b
        out = torch.cat([o1, o2], dim=-1)
        if head_size > 2 * embed_dim:
            out = torch.cat([out, x[..., 2 * embed_dim:].float()], dim=-1)
        return out.to(q.dtype)

    q_out = apply(q, positions)
    k_out = apply(k, positions) if k is not None else None
    return q_out, k_out


class Vllm_rotary_embedding_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.num_heads = config.get("num_heads", 32)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_size = config.get("head_size", 128)
        self.max_position = config.get("max_position", 8192)
        self.is_neox = config.get("is_neox", True)
        self.rope_dim_offset = config.get("rope_dim_offset", 0)
        self.inverse = config.get("inverse", False)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary(
            "Shape",
            f"({self.num_tokens} {self.num_heads} {self.head_size})"
        )
        q_elems = self.num_tokens * self.num_heads * self.head_size
        k_elems = self.num_tokens * self.num_kv_heads * self.head_size
        cache_elems = self.max_position * self.head_size
        pos_elems = self.num_tokens
        state.add_element_count(q_elems + k_elems + cache_elems + pos_elems)

        es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = (q_elems + k_elems) * es + cache_elems * es + pos_elems * 8
        writes = (q_elems + k_elems) * es
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def _prepare(self, dev_id, seed=42):
        dev = f'cuda:{dev_id}'
        torch.manual_seed(seed)
        q = torch.randn(self.num_tokens, self.num_heads, self.head_size,
                        dtype=self.dtype, device=dev)
        k = torch.randn(self.num_tokens, self.num_kv_heads, self.head_size,
                        dtype=self.dtype, device=dev)
        # cos_sin_cache must match query dtype; shape [max_position, head_size].
        cos_sin_cache = torch.randn(self.max_position, self.head_size,
                                    dtype=self.dtype, device=dev)
        positions = torch.randint(0, self.max_position, (self.num_tokens,),
                                  dtype=torch.int64, device=dev)
        return q, k, cos_sin_cache, positions

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            q, k, cos_sin_cache, positions = self._prepare(dev_id)
        return self.make_launcher(
            dev_id, torch.ops._C.rotary_embedding,
            positions, q, k, self.head_size, cos_sin_cache,
            self.is_neox, self.rope_dim_offset, self.inverse
        )

    def run_verification(self, dev_id):
        q, k, cos_sin_cache, positions = self._prepare(dev_id)
        q_in = q.clone()
        k_in = k.clone()
        torch.ops._C.rotary_embedding(
            positions, q, k, self.head_size, cos_sin_cache,
            self.is_neox, self.rope_dim_offset, self.inverse
        )
        q_ref, k_ref = _torch_neox_rope(
            q_in, k_in, positions, cos_sin_cache, self.head_size, self.inverse
        )
        q_match = self.check_diff(q, q_ref, threshold=0.99)
        k_match = self.check_diff(k, k_ref, threshold=0.99)
        return q_match[0] and k_match[0], 0.0 if (q_match[0] and k_match[0]) else 1.0
