---
name: mxbench_op_test
description: 为 mcoplib 中的新算子生成基于 nvbench 的性能测试代码 (config JSON + runner Python)，要求精度验证必须通过才会执行性能测试，精度验证失败则报错退出。
allowed-tools: ["Bash", "Read", "Write"]
triggers: ["生成config和runner", "生成算子测试", "生成算子mxbench测试", "添加算子配置", "添加算子benchmark", "新增算子性能测试"]
---

# 为新算子生成 nvbench 性能测试

## 总体原则

mcoplib 的 benchmark 框架基于 nvbench，每个算子实现一个 `OpBenchmarkBase` 子类。**精度验证 (run_verification) 必须先于性能测试 (nvbench 采样) 执行**：
- 精度验证通过 → 进入 warmup + nvbench 采样流程
- 精度验证失败 → `[FATAL]` 报错并调用 `os._exit(0)`，**不执行任何性能采样**

这一 gating 逻辑已内置在 `benchmark/mcoplib_mxbenchmark_ops.py:create_benchmark_wrapper()` 中（约 302-305 行），runner 作者无需重复实现，只需保证 `run_verification()` 正确返回 `(passed: bool, diff_val: float)`。

## 操作步骤

### 第 1 步：定位算子源码

根据单元测试 (`unit_test/test_<op>.py`) 中的 `import` 语句定位算子所属命名空间与源码目录：

| 单元测试中的 import | 调用方式 | 源码目录 | 头文件 |
|----|----|----|----|
| `import mcoplib.op` / `from mcoplib import op` | `op.op_name(...)` | `op/`（不含子目录）+ `kernel/` | `include/` |
| `import mcoplib._C` / `import mcoplib._moe_C` | `torch.ops._C.op_name(...)` / `torch.ops._moe_C.op_name(...)` | `op/vllm/`（MoE 相关在 `op/vllm/moe/`） | `op/vllm/moe/ops.h` / `op/vllm/moe/moe_ops.h` |
| `import mcoplib.sgl_kernel` | `torch.ops.sgl_kernel.op_name(...)` | `op/sglang/csrc/`（MoE 相关在 `op/sglang/csrc/moe/`） | `op/sglang/include/sgl_kernel_ops.h` |

**深度阅读算子源码与单元测试**，记录：
- 函数签名（参数顺序、类型、是否 in-place、optional 参数）
- 输入张量 shape 与 dtype 约束
- 是否有无序输出（top-k 索引等）、是否需要排序后比对
- 单元测试中的参考实现（golden）逻辑——`run_verification` 应尽量复刻

**重要：若同一算子名在多个框架下都有实现**（如 `rotary_embedding`、`topk_softmax` 同时存在于 sgl 与 vllm），需要为每个框架分别建一个 benchmark，命名为 `sgl_<op>` 与 `vllm_<op>`，禁止共用一个 runner。

### 第 2 步：生成 JSON 配置文件

目标路径：`benchmark/config/<op_name>.json`。模板：

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

规则：
1. `device_id` 固定 `0`，`device_name` 默认 `"MetaX C500"`（除非用户明确指定其他显卡）。
2. `dtype` 必填，支持 `"float16"` / `"bfloat16"` / `"float32"` / `"int32"` 等；`OpBenchmarkBase.__init__` 会自动转 `torch.dtype`。
3. `samples` 默认 `1000`（基准库用 1000-2000；调试时可临时改 50-100）。**避免设为 100000 等大值**——某些算子（如 `gptq_shuffle`）每次调用都触发 `cudaMalloc`/`cudaDeviceSynchronize`，采样数过大会导致 nvbench 超时（>600s）。
4. 其余字段为算子特有参数（如 `num_tokens`、`head_size`、`top_k`），由 runner 在 `__init__` 中通过 `config.get(key, default)` 读取。
5. 字段名应与 runner 中 `config.get(...)` 的 key 一致；若 runner 读 `num_query_heads`，config 不能写成 `num_heads`。

### 第 3 步：生成 Runner 文件（核心）

目标路径：`benchmark/runners/mcoplib_mxbenchmark_<op_name>_runners.py`。命名必须严格匹配 `<op_name>`，否则 loader 的精确匹配会失败、走 fuzzy search（容易选错文件）。

#### 3.1 类骨架

```python
import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C          # 或 mcoplib._moe_C / mcoplib.sgl_kernel / mcoplib.op
except ImportError:
    pass


class My_op_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.hidden_size = config.get("hidden_size", 4096)

    def define_metrics(self, state):
        ...

    def prepare_and_get_launcher(self, dev_id, tc_s):
        ...

    def run_verification(self, dev_id):
        ...
```

类名任意（loader 取第一个 `OpBenchmarkBase` 子类），建议 `<Op_name>_runner`。

#### 3.2 define_metrics —— 声明性能指标

通过 `state.add_summary(key, value)` 声明展示列；通过 `add_element_count` / `add_global_memory_reads` / `add_global_memory_writes` 声明计算 `Elem/s` 与 `GlobalMem BW` 所需的元素数与字节数。**读写量必须准确反映算子的实际内存访问量**，否则 `Elem/s` 与 `BWUtil` 失真。

```python
def define_metrics(self, state):
    state.add_summary("Op", self.name)
    state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
    # 注意：Shape 字符串中不要包含逗号，用空格分隔
    state.add_summary("Shape", f"({self.num_tokens} {self.hidden_size})")
    elems = self.num_tokens * self.hidden_size
    state.add_element_count(elems * 2)            # 读 x + 写 out
    es = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
    state.add_global_memory_reads(elems * es)
    state.add_global_memory_writes(elems * es)
```

**禁止**调用 `state.set_blocking_kernel_timeout(-1)` 让 nvbench 无限等待——除非确认算子每次调用都在 15s 内完成。`gptq_shuffle` 这类带 `cudaDeviceSynchronize` 的算子会因 `-1` 导致整个 testall.py 卡死 600s 超时。

#### 3.3 prepare_and_get_launcher —— 准备数据 + 返回 launcher

**关键：数据准备放在 `with torch.cuda.stream(tc_s):` 内**，launcher 闭包只做算子调用。nvbench 会反复调用 launcher 上千次，若每次都重新生成数据会污染性能数据。

```python
def _prepare(self, dev_id, seed=42):
    dev = f'cuda:{dev_id}'
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(self.num_tokens, self.hidden_size,
                    dtype=self.dtype, device=dev, generator=gen)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    return x, y, out

def prepare_and_get_launcher(self, dev_id, tc_s):
    with torch.cuda.stream(tc_s):
        x, y, out = self._prepare(dev_id)
    return self.make_launcher(dev_id, torch.ops._C.my_op, x, y, out)
```

`make_launcher(dev_id, op_func, *args)` 会把 `op_func(*args)` 包装成 `launcher(launch)`，在 nvbench 指定的 stream 上执行。

#### 3.4 run_verification —— 精度验证（必须先于性能测试）

`run_verification` 独立准备一组小规模数据，调用算子 + 参考实现，用 `check_diff` 比对。**返回 `(passed: bool, diff_val: float)`**。

```python
def run_verification(self, dev_id):
    x, y, out = self._prepare(dev_id, seed=7)        # 用与 benchmark 不同的 seed
    torch.ops._C.my_op(x, y, out)
    torch.cuda.synchronize()
    out_ref = _ref_my_op(x, y).to(out.dtype)
    return self.check_diff(out, out_ref, threshold=0.999999)
```

**精度阈值规则**：
- 默认 `threshold=0.999999`（余弦相似度 ≥ 0.999999 才算 PASS）
- 量化类算子（AWQ/GPTQ/FP8）可放宽到 `0.99` 或 `0.999`
- **bit-exact 算子**（如 `gptq_shuffle`、`copy_blocks`）应直接用 `torch.equal` 比对，返回 `(passed, 0.0 if passed else 1.0)`，**不要走 `check_diff`**——余弦相似度对位级 layout 差异不敏感
- 无序输出（top-k 索引）需先排序再比对，或用 `torch.allclose` + `torch.equal` 组合

**框架已保证：`run_verification` 返回 `passed=False` 时，`mcoplib_mxbenchmark_ops.py` 会打印 `[FATAL]` 并 `os._exit(0)`，跳过 nvbench 采样**。runner 无需自己实现这个 gating。

**参考实现 `_ref_*` 的关键要求**：
1. 忠实复现算子语义——含边界行为（padding、mask）、舍入方式、layout 转换
2. **禁止一行流**：解包、反量化、矩阵相乘必须拆为单步
3. **显式对齐维度**：分组量化（AWQ）需先用 `repeat_interleave` 把 `zeros`/`scales` 维度对齐，注意转置
4. **正确处理 KV cache layout**：`paged_attention` 的 `value_cache` shape 是 `[num_blocks, num_kv_heads, head_size, block_size]`（不是 `[block_size, head_size]`），ref 中 V 切片应为 `value_cache[block, head, :, offset]`，详见 references/mcoplib_ops.md

### 第 4 步：注册到 SUPPORTED_OPERATORS 白名单

编辑 `benchmark/mcoplib_mxbenchmark_ops.py`，在 `SUPPORTED_OPERATORS` 列表中按字母序插入算子名：

```python
SUPPORTED_OPERATORS = [
    ...
    "my_op",            # ← 新增
    "paged_attention_v1",
    ...
]
```

**若算子名与已有 runner 冲突**（如同名算子在不同框架下有不同实现），用前缀区分：`sgl_<op>` / `vllm_<op>`。`topk_softmax` 与 `rotary_embedding` 都是这种模式。

### 第 5 步：分步验证（必须按顺序）

#### 5.1 验证算子枚举
```bash
python mcoplib_mxbenchmark_ops.py --list | grep my_op
```
应输出 `  * my_op`。

#### 5.2 单独跑 run_verification（避免直接跑全量 nvbench 浪费时间）
```bash
python -u -c "
import torch, json
from runners.mcoplib_mxbenchmark_my_op_runners import My_op_runner
with open('config/my_op.json') as f: cfg = json.load(f)
r = My_op_runner('my_op', cfg)
passed, diff = r.run_verification(0)
print('VERIFY:', passed, diff)
"
```
期望：`VERIFY: True 0.0`（或差异极小如 `1e-07`）。
- `False` → 参考实现与算子语义不一致，检查 `_ref_*` 逻辑
- 抛异常 → 算子签名或 dtype 不匹配，回到第 1 步核对源码
- `nan` → 输入含 NaN 或算子有 bug，用 `torch.isnan(out).any()` 定位

**这一步是必须的**——直接跑 `--generate` 时若 verify 失败，`os._exit(0)` 会丢掉 stdout 缓冲，看不到 `[VERIFY] FAIL` 行，只表现为 "no result" 的 FAILED。

#### 5.3 验证全量基准测试
```bash
python mcoplib_mxbenchmark_ops.py --op my_op --generate --csv statistics/mcoplib_ops_performance_C500.csv
```
期望输出包含：
```
[VERIFY] my_op -> PASS (1-CosSim: 0.00e+00)
  >> [WARMUP] Running 10 iterations... Done.
...
| Acc_Pass | Cos_Dist |    Op    |  dtype  | ... |
|      Yes | 0.00e+00 |  my_op   | bfloat16 | ... |
...
[APPEND] my_op
[SUMMARY] Appended: 1, Skipped: 0
```

#### 5.4 验证 CSV 写入
```bash
grep "^my_op," statistics/mcoplib_ops_performance_C500.csv
```

#### 5.5 验证 testall.py 批量调度
```bash
python testall.py --generate --ops my_op
```
期望：
```
[1/1] my_op ... SUCCESS (appended) [NN.Ns]
SUMMARY | mode=generate | total=1 success=1 failed=0 skipped=0 | ...
```

### 第 6 步：同步文件到测试环境

- **本地服务器**：确保 `benchmark/config/<op>.json` 与 `benchmark/runners/mcoplib_mxbenchmark_<op>_runners.py` 存在。
- **远程服务器**：用 `scp` 同步两个文件夹到远程对应路径。已配置 SSH Key，可直接 `scp -r ./benchmark/config <user>@<host>:<path>/benchmark/` 与 `scp -r ./benchmark/runners <user>@<host>:<path>/benchmark/`。

### 第 7 步：执行 Benchmark

工作目录必须是 `benchmark/`：

```bash
docker exec -t -w /home/metax/mcoplib/github_mcoplib/mcoplib/benchmark vllm-dsv4 \
  /bin/bash -c "source ../env.sh && export PATH=\$PATH:/opt/conda/bin && \
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/conda/lib/python3.10/site-packages/torch/lib && \
  /opt/conda/bin/python mcoplib_mxbenchmark_ops.py --op <op_name> --csv statistics/mcoplib_ops_performance_C500.csv --generate"
```

容器名与路径以实际环境为准（示例中是 `vllm-dsv4` 容器，路径 `/home/metax/mcoplib/github_mcoplib/mcoplib/benchmark`）。

### 第 8 步：获取输出

**必须等待终端完整运行到下一个命令提示符出现**（类似 `root@...:/workspace#`），获取完整输出返回给用户。若有报错：
1. 根据 `.cu` 源码分析错误原因
2. 修改 runner 文件
3. 重新同步并执行
4. 直到 `Acc_Pass=Yes` 且 `[APPEND]`/`[UPDATE]` 行出现

### 第 9 步：最终检查

**禁止保留调试日志**：除了 nvbench 自身输出，不要在 runner 中留 `print()` / `debug` 语句。调试中途可加，最终成品必须删除。

## 常见问题与排查

| 现象 | 原因 | 解决 |
|----|----|----|
| `[VERIFY] X -> FAIL (1-CosSim: nan)` | 输出含 NaN，或参考实现与算子语义不符 | `torch.isnan(out).any()` 定位；核对 `_ref_*` |
| 进程卡在 `[WARMUP]` 无后续输出，600s 超时 | 算子每次调用都触发 `cudaDeviceSynchronize`/`cudaMalloc`（如 `gptq_shuffle`） | 降低 `samples` 到 50-100；或拆分 benchmark 路径跳过慢分支（如传空 `q_perm`） |
| `set_blocking_kernel_timeout(-1)` 导致 testall.py 卡死 | nvbench 无限等待 | 删除该行，用默认 15s 超时 |
| `Acc_Pass=No` 但 CSV 已写入 | 不应发生（gate 会 `os._exit`） | 检查 runner 是否在 `run_verification` 中吞了异常 |
| `Config file not found` | 配置文件名与算子名不一致 | 确认 `config/<op_name>.json` |
| `No relevant Python files found in runners` | runner 文件名与算子名不一致 | 确认 `runners/mcoplib_mxbenchmark_<op_name>_runners.py` |
| `missing value for argument 'XXX'` | 算子签名已更新，runner 调用缺参 | 回到第 1 步重新核对源码签名 |
| 性能数据 `Elem/s` 或 `BWUtil` 异常 | `define_metrics` 的元素数/读写量声明不准 | 核对算子实际读写量 |
| `value_cache` ref 比对失败（`paged_attention`） | V layout 错误：应为 `[num_blocks, num_kv_heads, head_size, block_size]`，ref 切片 `value_cache[block, head, :, offset]` | 详见 references/mcoplib_ops.md 中 paged_attention 章节 |
| 同名算子在 sgl + vllm 都有 | 不能共用一个 runner | 拆为 `sgl_<op>` + `vllm_<op>`，分别建 config/runner |

## 完整示例参考

已成功接入的算子（可作为模板参考）：
- `fused_moe_gate_deepseek`：top-k 无序输出，排序后 `allclose` + `equal` 比对
- `rotary_embedding` (default/sgl/vllm 三版本)：cos_sin_cache dtype 必须与 query 一致
- `gptq_shuffle`：bit-exact 比对，benchmark 路径跳过慢分支
- `paged_attention_v1`/`v2`：使用 vLLM 官方 `ref_single_query_cached_kv_attention` 模式
- `sgl_topk_softmax` / `vllm_topk_softmax`：同名算子拆分示例
- `top_k_per_row_decode`：注意 8 个位置参数（含 `topK`）

## 关键原则

1. **参考实现必须忠实复现算子语义**（含边界行为、舍入方式、无序输出处理）
2. **`define_metrics` 的读写量必须准确**（否则 `Elem/s` 与 `BWUtil` 失真）
3. **命名必须一致**（配置名 = runner 名 = 白名单条目）
4. **精度验证必须先于性能测试**——框架已内置 gating，runner 只需保证 `run_verification` 返回正确
