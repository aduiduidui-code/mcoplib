import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.op as op
except ImportError:
    op = None


def _biased_grouped_topk(gating_output, correction_bias, topk, renormalize,
                         num_expert_group, topk_group, routed_scaling_factor):
    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class Fused_moe_gate_deepseek_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.num_experts = config.get("num_experts", 256)
        self.num_expert_group = config.get("num_expert_group", 8)
        self.topk_group = config.get("topk_group", 4)
        self.topk = config.get("topk", 8)
        self.renormalize = config.get("renormalize", True)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary(
            "Shape",
            f"({self.num_tokens} {self.num_experts})"
        )
        gate_elems = self.num_tokens * self.num_experts
        out_elems = self.num_tokens * self.topk
        state.add_element_count(gate_elems + out_elems * 2 + self.num_experts)

        es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        reads = gate_elems * es + self.num_experts * es
        writes = out_elems * 4 + out_elems * 4
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def _prepare(self, dev_id):
        dev = f'cuda:{dev_id}'
        torch.manual_seed(42)
        gating = torch.rand(self.num_tokens, self.num_experts,
                            dtype=self.dtype, device=dev)
        bias = torch.rand(self.num_experts, dtype=self.dtype, device=dev)
        out_w = torch.zeros(self.num_tokens, self.topk,
                            dtype=torch.float32, device=dev)
        out_idx = torch.zeros(self.num_tokens, self.topk,
                              dtype=torch.int32, device=dev)
        return gating, bias, out_w, out_idx

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            gating, bias, out_w, out_idx = self._prepare(dev_id)
        return self.make_launcher(
            dev_id, op.fused_moe_gate_deepseek,
            gating, bias, out_w, out_idx, self.topk, self.renormalize,
            self.num_expert_group, self.topk_group, None,
            self.routed_scaling_factor, 0
        )

    def run_verification(self, dev_id):
        gating, bias, out_w, out_idx = self._prepare(dev_id)
        op.fused_moe_gate_deepseek(
            gating, bias, out_w, out_idx, self.topk, self.renormalize,
            self.num_expert_group, self.topk_group, None,
            self.routed_scaling_factor, 0
        )
        ref_w, ref_idx = _biased_grouped_topk(
            gating, bias, self.topk, self.renormalize,
            self.num_expert_group, self.topk_group, self.routed_scaling_factor
        )
        sorted_op_w = torch.sort(out_w, dim=1).values
        sorted_ref_w = torch.sort(ref_w, dim=1).values
        w_match = torch.allclose(sorted_op_w, sorted_ref_w, rtol=1e-3, atol=1e-3)
        sorted_op_idx = torch.sort(out_idx, dim=1).values
        sorted_ref_idx = torch.sort(ref_idx, dim=1).values
        idx_match = torch.equal(sorted_op_idx, sorted_ref_idx)
        passed = bool(w_match and idx_match)
        return passed, 0.0 if passed else 1.0
