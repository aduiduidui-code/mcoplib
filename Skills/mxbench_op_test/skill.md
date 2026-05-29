---
name: mxbench_op_test
description: 为新增的算子生成性能测试所需的 config (JSON) 和 runner (Python) 文件。
allowed-tools: ["Bash", "Read", "Write"]
triggers: ["生成config和runner", "生成算子测试", "生成算子mxbench测试","添加算子配置"]
---

# 生成算子 Benchmark 配置文件
## 操作步骤
### 第1步：收集算子上下文
算子的底层源代码（.cu 文件）和算子文档.md 放在 reference 中
  - 必须先读取 算子文档.md，从中提取：导入语句 (import xxx)、调用示例、参数说明
  - 再读取 .cu 文件，理解算子的计算逻辑用于编写 PyTorch 参考实现
### 第2步：生成 JSON 配置文件
使用 `Write` 工具创建配置文件。
目标路径：`benchmark/config/<op_name>.json`
写入内容模板：
```json
{
    "device_id": 0,
    "device_name": "MetaX C500",
    "batch_size": 16384,
    "top_k": 2,
    "hidden_size": 4096,
    "dtype": "float16",
    "samples": 10000
}
```
1. **基础环境参数**：device_id 默认为 0，device_name 默认为 "MetaX C500"，samples 默认为 10000。除非用户在需求中明确指定了其他显卡（如 "MetaX C280"）或不同的采样数，否则这三个值必须保持不变。
2. **参数值设置**：其余参数根据算子的实际情况来指定合理值，注意json中算子特有参数值的设置要足够大来拉满利用率和带宽。

### 第3步：生成 Runner 测试脚本 (核心步骤)
目标路径：`benchmark/runners/mcoplib_mxbenchmark_<op_name>_runners.py`
**shape格式**
注意define_metrics中指定shape的时候，字符串中不要有逗号存在
**精度验证要求**
根据用户指定的.cu文件名称从reference文件夹中找到对应的.cu文件，根据文件算子源码内容编写pytorch同义实现，同时调用算子，对比两个结果，要求用余弦相似度验证，一般是0.999999，特别算子可以放宽到0.9999
注意：
1. **禁止一行流**：严禁将解包、反量化、矩阵相乘合并在一行写，必须拆解为单步操作。
2. **显式对齐维度**：处理带有 Group 的量化（如 AWQ）时，必须先用 `repeat_interleave` 将 `zeros` 和 `scales` 的维度放大对齐，并注意转置（`.T`），确保相减/相乘前双方的形状**100%一致**。

参考以下模板，保持总体格式不变：
```python
import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
try:
    # 【强制】导入语句必须直接从算子文档.md 的调用示例中复制，禁止照抄下面的示例
  import mcoplib.op  # 示例：根据实际算子文档修改
except ImportError:
    pass
class Moe_sum_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 4096)
        self.top_k = config.get("top_k", 2)
        self.hidden_size = config.get("hidden_size", 4096)
    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.top_k} {self.hidden_size})")
        total_out_elements = self.batch_size * self.hidden_size
        state.add_element_count(total_out_elements)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        reads = (self.batch_size * self.top_k * self.hidden_size) * element_size
        writes = (self.batch_size * self.hidden_size) * element_size
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)
    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_tensor = torch.randn(self.batch_size, self.top_k, self.hidden_size, dtype=self.dtype, device=dev)
            output_tensor = torch.empty(self.batch_size, self.hidden_size, dtype=self.dtype, device=dev)
        return self.make_launcher(dev_id, torch.ops._moe_C.moe_sum, input_tensor, output_tensor)
    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        N, K, H = 16, 2, 128
        input_tensor = torch.randn(N, K, H, dtype=self.dtype, device=dev)
        output_op = torch.empty(N, H, dtype=self.dtype, device=dev)
        torch.ops._moe_C.moe_sum(input_tensor, output_op)
        output_ref = input_tensor.sum(dim=1)
        return self.check_diff(output_op, output_ref)
```
注意：1.禁止去阅读其他文件，只允许查看当前skill同目录下reference文件下的内容
2.生成的json和runner放置在当前项目文件夹下的benchmark下的json和runners文件夹
3.注意服务器一定是可达的，一定要尝试连接服务器上传代码做测试
5.如果上传后测试报错，需要修改runner文件，重新上传检查

### 4. 同步本地文件至服务器
使用 Bash 工具，文件同步到远程服务器。由于已配置 SSH Key，你可以自行思考并构建合适的文件传输命令（如 `scp`）进行同步。

请确保以下两个本地文件夹被完整传输到远程服务器对应的目标路径下：
- **Config 文件夹**:
  - 本地源路径: `./benchmark/config`
  - 远程目标路径: `xinyue@10.6.28.80:/home/xinyue/finale/mcoplib/benchmark`
- **Runners 文件夹** (请确保里面包含所需的 `.py` 运行器文件):
  - 本地源路径: `./benchmark/config`
  - 远程目标路径: `xinyue@10.6.28.80:/home/xinyue/finale/mcoplib/benchmark`

### 5. 远程穿透执行 Benchmark (工作目录修正版)
文件同步完成后，使用 Bash 执行以下指令。

`ssh -tt xinyue@10.6.28.80 "docker exec -t -w /workspace/finale/mcoplib/benchmark mxbench_torch2.8 /opt/conda/bin/python3 -u mcoplib_mxbenchmark_ops.py --op <op-name> --csv statistics/mcoplib_ops_performance_C500.csv --generate 2>&1"`

### 6.获取输出
我只需要终端输出，不用检查别的，必须等待终端完整运行完成到下一个命令提示符出现（类似root@k8s-master:/workspace# ），获取最完整的终端输出返回给用户，同时如果有报错要根据.cu源码修正错误，再次上传修改，直到不报错才可以。

### 7.最终检查
**禁止增加额外日志输出**：除了正常的benchmark的输出日志，不要自行添加一些用于调试的终端打印日志，在调试错误中途可以增加这些打印信息来判断错误原因，但最后的成品必须把这些调试逻辑和输出删除，保持代码纯净