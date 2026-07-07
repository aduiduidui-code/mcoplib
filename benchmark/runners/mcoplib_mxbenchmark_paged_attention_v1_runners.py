import torch
import random
import math
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


def _ref_masked_attention(query, key, value, scale, attn_mask=None):
    # query: [1, num_q_heads, head_size]
    # key:   [seq_len, num_q_heads, head_size] (already repeated for GQA)
    # value: [seq_len, num_q_heads, head_size]
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    return torch.einsum("hqk,khd->qhd", attn_weights, value)


def _ref_single_query_cached_kv_attention(
    output, query, num_queries_per_kv, key_cache, value_cache,
    block_tables, seq_lens, scale, alibi_slopes,
):
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst = []
        values_lst = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)
            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)
        alibi_bias = None
        if alibi_slopes is not None:
            position_ids = torch.arange(seq_len, device=query.device).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)
        out = _ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


class Paged_attention_v1_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_seqs = config.get("num_seqs", 7)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_size = config.get("head_size", 128)
        self.block_size = config.get("block_size", 16)
        self.num_query_heads = config.get("num_query_heads", 32)
        self.num_blocks = config.get("num_blocks", 128)
        self.max_seq_len = config.get("max_seq_len", 256)
        self.seed = config.get("seed", 0)
        self.scale = config.get("scale", 1.0 / (self.head_size ** 0.5))
        self.x = 16 // (2 if self.dtype in (torch.float16, torch.bfloat16) else 4)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_seqs} {self.num_query_heads} {self.head_size})")
        element_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        q_elements = self.num_seqs * self.num_query_heads * self.head_size
        kv_elements = self.num_blocks * self.num_kv_heads * self.block_size * self.head_size * 2
        state.add_element_count(q_elements + kv_elements)
        total_read_bytes = (q_elements * element_size) + (kv_elements * element_size)
        total_write_bytes = q_elements * element_size
        state.add_global_memory_reads(total_read_bytes)
        state.add_global_memory_writes(total_write_bytes)

    def _generate_args(self, dev_id, seed=None):
        s = seed if seed is not None else self.seed
        torch.manual_seed(s)
        random.seed(s)
        dev = f'cuda:{dev_id}'
        scale = self.scale
        query = torch.empty(self.num_seqs, self.num_query_heads, self.head_size,
                            dtype=self.dtype, device=dev)
        query.uniform_(-scale, scale)
        key_cache = torch.randn(
            self.num_blocks, self.num_kv_heads, self.head_size // self.x,
            self.block_size, self.x, dtype=self.dtype, device=dev,
        )
        value_cache = torch.randn(
            self.num_blocks, self.num_kv_heads, self.head_size, self.block_size,
            dtype=self.dtype, device=dev,
        )
        seq_lens_list = [random.randint(1, self.max_seq_len) for _ in range(self.num_seqs)]
        seq_lens_list[-1] = self.max_seq_len
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=dev)
        max_seq_len_actual = max(seq_lens_list)
        max_num_blocks_per_seq = (max_seq_len_actual + self.block_size - 1) // self.block_size
        block_tables = torch.randint(
            0, self.num_blocks, (self.num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32, device=dev,
        )
        output = torch.empty_like(query)
        alibi_slopes = None
        kv_cache_dtype = "auto"
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        return {
            "output": output,
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "num_kv_heads": self.num_kv_heads,
            "scale": scale,
            "block_tables": block_tables,
            "seq_lens": seq_lens,
            "block_size": self.block_size,
            "max_seq_len": max_seq_len_actual,
            "alibi_slopes": alibi_slopes,
            "kv_cache_dtype": kv_cache_dtype,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "num_queries_per_kv": self.num_query_heads // self.num_kv_heads,
        }

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            args = self._generate_args(dev_id)
        return self.make_launcher(
            dev_id, torch.ops._C.paged_attention_v1,
            args["output"], args["query"], args["key_cache"], args["value_cache"],
            args["num_kv_heads"], args["scale"], args["block_tables"], args["seq_lens"],
            args["block_size"], args["max_seq_len"], args["alibi_slopes"],
            args["kv_cache_dtype"], args["k_scale"], args["v_scale"],
            0, 0, 0, 0, 0,
        )

    def run_verification(self, dev_id):
        args = self._generate_args(dev_id, seed=7)
        torch.ops._C.paged_attention_v1(
            args["output"], args["query"], args["key_cache"], args["value_cache"],
            args["num_kv_heads"], args["scale"], args["block_tables"], args["seq_lens"],
            args["block_size"], args["max_seq_len"], args["alibi_slopes"],
            args["kv_cache_dtype"], args["k_scale"], args["v_scale"],
            0, 0, 0, 0, 0,
        )
        torch.cuda.synchronize()
        ref_output = torch.empty_like(args["query"])
        _ref_single_query_cached_kv_attention(
            ref_output, args["query"], args["num_queries_per_kv"],
            args["key_cache"], args["value_cache"],
            args["block_tables"], args["seq_lens"], args["scale"], args["alibi_slopes"],
        )
        return self.check_diff(args["output"], ref_output, threshold=0.999)
