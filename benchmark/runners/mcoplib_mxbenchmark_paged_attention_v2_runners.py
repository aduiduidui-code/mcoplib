import torch
import random
import math
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
from runners.mcoplib_mxbenchmark_paged_attention_v1_runners import (
    _ref_single_query_cached_kv_attention,
)


class Paged_attention_v2_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_seqs = config.get("num_seqs", 7)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_size = config.get("head_size", 128)
        self.block_size = config.get("block_size", 16)
        self.num_query_heads = config.get("num_query_heads", 32)
        self.num_blocks = config.get("num_blocks", 128)
        self.max_seq_len = config.get("max_seq_len", 256)
        self.partition_size = config.get("partition_size", 512)
        self.seed = config.get("seed", 0)
        self.scale = 1.0 / (self.head_size ** 0.5)
        self.x = 16 // (2 if self.dtype in (torch.float16, torch.bfloat16) else 4)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(Seqs:{self.num_seqs} QHeads:{self.num_query_heads} HeadSize:{self.head_size})")
        total_elements = self.num_seqs * self.num_query_heads * self.head_size
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        state.add_global_memory_reads(total_elements * 3 * element_size)
        state.add_global_memory_writes(total_elements * 1 * element_size)

    def _prepare_data(self, dev_id, seed=None):
        s = seed if seed is not None else self.seed
        torch.manual_seed(s)
        random.seed(s)
        dev = f'cuda:{dev_id}'
        dtype = self.dtype
        scale = self.scale
        query = torch.empty(self.num_seqs, self.num_query_heads, self.head_size,
                            dtype=dtype, device=dev)
        query.uniform_(-scale, scale)
        key_cache = torch.randn(
            self.num_blocks, self.num_kv_heads, self.head_size // self.x,
            self.block_size, self.x, dtype=dtype, device=dev,
        )
        value_cache = torch.randn(
            self.num_blocks, self.num_kv_heads, self.head_size, self.block_size,
            dtype=dtype, device=dev,
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
        max_num_partitions = (max_seq_len_actual + self.partition_size - 1) // self.partition_size
        output = torch.empty_like(query)
        exp_sums = torch.empty(
            self.num_seqs, self.num_query_heads, max_num_partitions,
            dtype=torch.float32, device=dev,
        )
        max_logits = torch.empty(
            self.num_seqs, self.num_query_heads, max_num_partitions,
            dtype=torch.float32, device=dev,
        )
        tmp_out = torch.empty(
            self.num_seqs, self.num_query_heads, max_num_partitions, self.head_size,
            dtype=dtype, device=dev,
        )
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        return {
            "output": output,
            "exp_sums": exp_sums,
            "max_logits": max_logits,
            "tmp_out": tmp_out,
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "num_kv_heads": self.num_kv_heads,
            "scale": scale,
            "block_tables": block_tables,
            "seq_lens": seq_lens,
            "block_size": self.block_size,
            "max_seq_len": max_seq_len_actual,
            "alibi_slopes": None,
            "kv_cache_dtype": "auto",
            "k_scale": k_scale,
            "v_scale": v_scale,
            "num_queries_per_kv": self.num_query_heads // self.num_kv_heads,
        }

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            args = self._prepare_data(dev_id)
        return self.make_launcher(
            dev_id, torch.ops._C.paged_attention_v2,
            args["output"], args["exp_sums"], args["max_logits"], args["tmp_out"],
            args["query"], args["key_cache"], args["value_cache"],
            args["num_kv_heads"], args["scale"], args["block_tables"], args["seq_lens"],
            args["block_size"], args["max_seq_len"], args["alibi_slopes"],
            args["kv_cache_dtype"], args["k_scale"], args["v_scale"],
            0, 0, 0, 0, 0,
        )

    def run_verification(self, dev_id):
        args = self._prepare_data(dev_id, seed=7)
        torch.ops._C.paged_attention_v2(
            args["output"], args["exp_sums"], args["max_logits"], args["tmp_out"],
            args["query"], args["key_cache"], args["value_cache"],
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
