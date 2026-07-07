# 如何添加新算子到 Benchmark 性能测试框架

本文档介绍如何将一个新的 kernel 算子接入 mcoplib 的 benchmark 性能测试框架，从了解框架结构到编写配置与 runner、注册到白名单、再到测试验证，逐步说明。

## 一、Benchmark 测试框架简介

mcoplib 的 benchmark 框架基于 nvbench 构建，用于对单个 kernel 算子执行：

1. **精度验证（Accuracy Verification）**：调用算子的 `run_verification()`，将算子输出与参考实现做余弦相似度对比，输出 `Pass/Fail` 与 `1-CosSim` 差异值。
2. **性能采样（Performance Sampling）**：精度通过后，注册到 nvbench 执行 warmup + N 次采样，采集 `GPU Time`、`Elem/s`、`GlobalMem BW`、`BWUtil` 等指标。
3. **基准库管理**：通过 `--generate` / `--update` / `--compare` 三种模式将性能数据写入 / 更新 / 对比 CSV 基准库。

框架的核心抽象是 `OpBenchmarkBase` 基类，每个算子实现一个子类，由主入口脚本 `mcoplib_mxbenchmark_ops.py` 统一加载与调度。

## 二、目录结构

```
benchmark/
├── mcoplib_mxbenchmark_ops.py        # 主入口脚本（CLI、SUPPORTED_OPERATORS 白名单、加载逻辑）
├── mcoplib_mxbenchmark_op_wrapper.py # OpBenchmarkBase 基类定义
├── testall.py                        # 批量驱动脚本（逐个调用主入口）
├── config/                           # 每算子一个 JSON 配置文件
│   ├── rms_norm.json
│   ├── silu_and_mul.json
│   └── ...
├── runners/                          # 每算子一个 Python runner 文件
│   ├── mcoplib_mxbenchmark_rms_norm_runners.py
│   ├── mcoplib_mxbenchmark_silu_and_mul_runners.py
│   └── ...
├── statistics/                       # 性能基准 CSV 输出目录
│   └── mcoplib_ops_performance_C500.csv
└── README.md                         # 使用说明
```

### 命名约定（重要）

加载器 `load_operator_runner()` 通过算子名做模糊匹配，但**强烈建议遵循精确命名**以避免歧义：

| 文件 | 命名格式 | 示例 |
|------|---------|------|
| 配置文件 | `<op_name>.json` | `rms_norm.json` |
| Runner 文件 | `mcoplib_mxbenchmark_<op_name>_runners.py` | `mcoplib_mxbenchmark_rms_norm_runners.py` |
| Runner 类名 | 任意（建议 `<Op_name>_runner`） | `Rms_norm_runner` |

`<op_name>` 必须与 `mcoplib_mxbenchmark_ops.py` 中 `SUPPORTED_OPERATORS` 列表的条目一致。

## 三、OpBenchmarkBase 基类

所有 runner 必须继承 `OpBenchmarkBase`（定义于 `mcoplib_mxbenchmark_op_wrapper.py`），并实现三个抽象方法：

```python
class OpBenchmarkBase(ABC):
    def __init__(self, name, config):
        # name: 算子名；config: 从 JSON 加载的 dict
        # 自动解析 config["dtype"] 为 torch.dtype（默认 float16）
        ...

    @abstractmethod
    def define_metrics(self, state):
        """声明性能指标的元信息（Shape、元素数、读写带宽等），供 nvbench 展示。"""
        ...

    @abstractmethod
    def prepare_and_get_launcher(self, dev_id, tc_s):
        """准备输入数据，返回一个 launcher 闭包供 nvbench 反复调用执行算子。"""
        ...

    @abstractmethod
    def run_verification(self, dev_id):
        """精度验证：调用算子 + 参考实现，返回 (passed: bool, diff_val: float)。"""
        ...

    def make_launcher(self, dev_id, op_func, *args):
        """工具方法：把 op_func(*args) 包装成 nvbench 需要的 launcher(stream)。"""
        ...

    def check_diff(self, output_op, output_ref, threshold=0.999999):
        """工具方法：余弦相似度对比，返回 (passed, 1-cos_sim)。"""
        ...
```

## 四、添加流程

下面以添加一个虚构算子 `my_new_op`（假设签名 `torch.ops._C.my_new_op(x, y, out)`）为例，逐步说明。

### 步骤 1：确认算子已注册到 torch.ops

在容器内验证算子可调用：

```bash
docker exec -it vllm-dsv4 /bin/bash
cd /home/metax/mcoplib/github_mcoplib/mcoplib
source env.sh
export PATH=$PATH:/opt/conda/bin
python3 -c "
import torch
import mcoplib._C   # 或 mcoplib.sgl_kernel / mcoplib.op
print(torch.ops._C.my_new_op)
"
```

如果打印出 `<_C.my_new_op>` 表示已注册；若报 `AttributeError`，需先在 C++ 侧完成 pybind 绑定并重新编译 mcoplib。

### 步骤 2：阅读算子源码，记录签名

阅读 `op/vllm/xxx.cu` 或 `op/sglang/csrc/xxx.cu`，记录：
- 函数签名（参数顺序、类型、是否 in-place）
- 输入张量 shape 与 dtype 约束
- 是否有 `is_neox`、`rope_dim_offset` 等可选参数
- 输出是写入预分配张量还是返回新张量

同时阅读对应的 `unit_test/test_xxx.py`，复用其中的参考实现（golden）逻辑用于 `run_verification`。

### 步骤 3：创建配置文件

在 `benchmark/config/` 下新建 `my_new_op.json`：

```json
{
    "device_id": 0,
    "device_name": "MetaX C500",
    "num_tokens": 4096,
    "hidden_size": 4096,
    "dtype": "bfloat16",
    "samples": 1000
}
```

字段说明：
- `device_id` / `device_name`：目标 GPU。
- `dtype`：必填，支持 `"float16"` / `"bfloat16"` / `"float32"`，基类会自动转为 `torch.dtype`。
- `samples`：nvbench 采样次数，越大越稳定但越慢；建议基准库用 1000-2000，调试时用 50-100。
- 其余字段为算子特有参数（如 `num_tokens`、`hidden_size`），由 runner 在 `__init__` 中读取。

### 步骤 4：创建 Runner 文件

在 `benchmark/runners/` 下新建 `mcoplib_mxbenchmark_my_new_op_runners.py`，模板如下：

```python
import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


def _ref_my_new_op(x, y):
    """参考实现：用纯 PyTorch 复现算子语义，供精度验证对比。"""
    return x.float() + y.float()  # 以加法为例，替换为真实语义


class My_new_op_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.hidden_size = config.get("hidden_size", 4096)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_tokens} {self.hidden_size})")
        elems = self.num_tokens * self.hidden_size
        state.add_element_count(elems * 2)  # 读 x+y，写 out
        es = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        state.add_global_memory_reads(elems * es * 2)
        state.add_global_memory_writes(elems * es)

    def _prepare(self, dev_id, seed=42):
        dev = f'cuda:{dev_id}'
        torch.manual_seed(seed)
        x = torch.randn(self.num_tokens, self.hidden_size,
                        dtype=self.dtype, device=dev)
        y = torch.randn(self.num_tokens, self.hidden_size,
                        dtype=self.dtype, device=dev)
        out = torch.empty_like(x)
        return x, y, out

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            x, y, out = self._prepare(dev_id)
        return self.make_launcher(
            dev_id, torch.ops._C.my_new_op, x, y, out
        )

    def run_verification(self, dev_id):
        x, y, out = self._prepare(dev_id)
        torch.ops._C.my_new_op(x, y, out)
        torch.cuda.synchronize()
        out_ref = _ref_my_new_op(x, y).to(out.dtype)
        return self.check_diff(out, out_ref, threshold=0.99)
```

#### 关键要点

1. **`define_metrics`**：通过 `state.add_summary()` 声明展示列；`add_element_count` / `add_global_memory_reads` / `add_global_memory_writes` 用于计算 `Elem/s` 与 `GlobalMem BW`，必须准确反映算子的实际读写量。
2. **`prepare_and_get_launcher`**：在 `tc_s`（test stream）中准备数据，返回 `self.make_launcher(dev_id, op_func, *args)`。launcher 会在 nvbench 采样时被反复调用，**不要在其中做数据初始化**（否则每次采样都重新生成数据）。
3. **`run_verification`**：独立准备一组数据（小规模即可，如 N=16），调用算子 + 参考实现，用 `check_diff` 比对。返回 `(bool, float)`。
4. **`threshold`**：默认 0.999999，对有损算子（如量化）可放宽到 0.99 或更低。

### 步骤 5：注册到 SUPPORTED_OPERATORS 白名单

编辑 `benchmark/mcoplib_mxbenchmark_ops.py`，在 `SUPPORTED_OPERATORS` 列表中按字母序插入 `"my_new_op"`：

```python
SUPPORTED_OPERATORS = [
    ...
    "mx_awq_dequantize",
    "my_new_op",          # ← 新增
    "paged_attention_v1",
    ...
]
```

> 若算子名与已有 runner 冲突（如同名算子在不同框架下有不同实现），可用前缀区分，如 `sgl_rotary_embedding` / `vllm_rotary_embedding`。

### 步骤 6：检查 testall.py 的 IGNORE_OPS（可选）

若该算子在 `testall.py` 的 `IGNORE_OPS` 列表中（批量跑时被跳过），且你希望它参与批量测试，则需从 `IGNORE_OPS` 移除。新算子默认不在该列表，无需操作。

## 五、测试验证

### 5.1 验证算子枚举

```bash
python3 mcoplib_mxbenchmark_ops.py --list | grep my_new_op
```

应输出 `  * my_new_op`。

### 5.2 验证精度（React 模式调试）

先单独跑 `run_verification`，避免直接跑全量 nvbench 浪费时间：

```bash
python3 -u -c "
import torch, json
from runners.mcoplib_mxbenchmark_my_new_op_runners import My_new_op_runner
with open('config/my_new_op.json') as f: cfg = json.load(f)
r = My_new_op_runner('my_new_op', cfg)
passed, diff = r.run_verification(0)
print('VERIFY:', passed, diff)
"
```

期望输出：`VERIFY: True 0.0`（或差异极小如 `1e-07`）。

- 若 `False`：参考实现与算子语义不一致，检查 `_ref_xxx` 逻辑。
- 若抛异常：算子签名或 dtype 不匹配，检查步骤 2 的记录。
- 若 `nan`：输入含 NaN 或算子有 bug，单独用 `torch.isnan(out).any()` 定位。

### 5.3 验证全量基准测试

```bash
# 调试时用小 samples（临时改 config 或用 --generate 走默认）
python3 mcoplib_mxbenchmark_ops.py --op my_new_op --generate --csv statistics/mcoplib_ops_performance_C500.csv
```

期望输出包含：
```
[VERIFY] my_new_op -> PASS (1-CosSim: 0.00e+00)
...
| Acc_Pass | Cos_Dist |    Op    |  dtype  | ... |
|      Yes | 0.00e+00 | my_new_op | bfloat16 | ... |
...
[APPEND] my_new_op
[SUMMARY] Appended: 1, Skipped: 0
```

### 5.4 验证 CSV 写入

```bash
grep "^my_new_op," statistics/mcoplib_ops_performance_C500.csv
```

应能看到一行新数据。

### 5.5 验证 testall.py 批量调度

```bash
python3 testall.py --generate --ops my_new_op
```

期望输出：
```
[1/1] my_new_op ... SUCCESS (appended) [NN.Ns]
SUMMARY | mode=generate | total=1 success=1 failed=0 skipped=0 | ...
```

## 六、完整示例：添加 `fused_moe_gate_deepseek`

以实际接入的 `fused_moe_gate_deepseek` 为例，展示完整流程。

### 6.1 算子签名（来自 `include/fused_moe_gate_deepseek.h`）

```cpp
int fused_moe_gate_deepseek(
    Tensor& gating_outputs,        // [bs, num_experts], bf16/f16/f32
    Tensor& correction_bias,       // [num_experts], bf16/f16/f32
    Tensor& out_routing_weights,   // [bs, topk], float32
    Tensor& out_selected_experts,  // [bs, topk], int32
    int topk,
    bool renormalize,
    int num_expert_group,
    int topk_group,
    std::optional<int> num_fused_shared_experts,
    std::optional<float> routed_scaling_factor,
    std::optional<int> moegate_type
);
```

### 6.2 配置文件 `config/fused_moe_gate_deepseek.json`

```json
{
    "device_id": 0,
    "device_name": "MetaX C500",
    "num_tokens": 4096,
    "num_experts": 256,
    "num_expert_group": 8,
    "topk_group": 4,
    "topk": 8,
    "renormalize": true,
    "routed_scaling_factor": 1.0,
    "dtype": "bfloat16",
    "samples": 1000
}
```

### 6.3 Runner 关键片段

```python
def _biased_grouped_topk(gating_output, correction_bias, topk, renormalize,
                         num_expert_group, topk_group, routed_scaling_factor):
    # 从 unit_test/test_fused_moe_gate_deepseek.py 复刻的参考实现
    scores = gating_output.sigmoid()
    ...
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class Fused_moe_gate_deepseek_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.num_experts = config.get("num_experts", 256)
        ...

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
        # top-k 顺序不保证，按行排序后比对
        sorted_op_w = torch.sort(out_w, dim=1).values
        sorted_ref_w = torch.sort(ref_w, dim=1).values
        w_match = torch.allclose(sorted_op_w, sorted_ref_w, rtol=1e-3, atol=1e-3)
        sorted_op_idx = torch.sort(out_idx, dim=1).values
        sorted_ref_idx = torch.sort(ref_idx, dim=1).values
        idx_match = torch.equal(sorted_op_idx, sorted_ref_idx)
        passed = bool(w_match and idx_match)
        return passed, 0.0 if passed else 1.0
```

注意：该算子输出是 top-k 索引（无序），不能用余弦相似度，需按行排序后用 `allclose` + `equal` 比对，并直接返回 `(passed, 0.0/1.0)` 而非走 `check_diff`。

### 6.4 注册到白名单

```python
SUPPORTED_OPERATORS = [
    ...
    "fused_mla_absorb_rotary_emb",
    "fused_moe_gate_deepseek",    # ← 新增
    "fused_moe_gate_opt",
    ...
]
```

### 6.5 验证结果

```
[VERIFY] fused_moe_gate_deepseek -> PASS (1-CosSim: 0.00e+00)
| Yes | 0.00e+00 | fused_moe_gate_deepseek | bfloat16 | (4096 256) | ... | 5.149G | ... |
[APPEND] fused_moe_gate_deepseek
```

## 七、常见问题与排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `[VERIFY] X -> FAIL (1-CosSim: nan)` | 输出含 NaN，或参考实现与算子语义不符 | 用 `torch.isnan(out).any()` 定位 NaN 位置；核对参考实现 |
| 进程卡在 `Run: [1/1] X [Device=0]` 无后续输出 | 校验失败触发 `os._exit(0)`，stdout 缓冲未刷新 | 单独跑 `run_verification()` 看真实报错 |
| `Acc_Pass=No` 但 CSV 已写入 | 校验失败但 nvbench 仍跑了采样（不应发生） | 检查 runner 是否在 `run_verification` 中吞了异常 |
| `Config file not found` | 配置文件名与算子名不一致 | 确认 `config/<op_name>.json` 命名 |
| `No relevant Python files found in runners` | runner 文件名与算子名不一致 | 确认 `runners/mcoplib_mxbenchmark_<op_name>_runners.py` 命名 |
| 性能数据 `Elem/s` 或 `BWUtil` 异常 | `define_metrics` 的元素数/读写量声明不准 | 核对算子实际读写量，重新计算 |

## 八、总结

添加一个新算子到 benchmark 框架的完整流程：

1. **确认算子已注册**：`torch.ops.<ns>.<op>` 可调用。
2. **阅读源码与单元测试**：记录签名、dtype、shape 约束，复用单元测试的参考实现。
3. **创建配置文件**：`config/<op_name>.json`，含 `dtype`、`samples` 与算子特有参数。
4. **创建 Runner**：`runners/mcoplib_mxbenchmark_<op_name>_runners.py`，继承 `OpBenchmarkBase`，实现 `define_metrics` / `prepare_and_get_launcher` / `run_verification` 三个方法。
5. **注册白名单**：在 `SUPPORTED_OPERATORS` 中按字母序插入算子名。
6. **分步验证**：先 `--list` 确认枚举 → 单独跑 `run_verification` 确认精度 → 跑 `--generate` 确认全流程 → `grep` CSV 确认写入 → `testall.py --ops` 确认批量调度。

核心原则：**参考实现必须忠实复现算子语义**（含边界行为、舍入方式、无序输出处理）；**`define_metrics` 的读写量必须准确**（否则 `Elem/s` 与 `BWUtil` 失真）；**命名必须一致**（配置名 = runner 名 = 白名单条目）。遵循这三点，新算子即可顺利接入框架并参与持续的性能基准管理。
