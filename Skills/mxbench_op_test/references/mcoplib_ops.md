## 1. fused_bias_dropout
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_bias_dropout(
    input: at::Tensor,
    residual: at::Tensor,
    dropout_prob: float
) -> at::Tensor
```
### 功能描述
fused_bias_dropout 算子是面向 GPU 优化的融合计算算子，将残差逐元素相加与 dropout 操作合并执行，减少 GPU 显存的读写开销，常用于大语言模型的训练和推理（当前版本中 dropout 功能暂未实现，仅支持 dropout_prob=0.0 的场景）
- 计算公式：
	当前版本中，fused_bias_dropout 算子仅支持 dropout_prob=0.0
	$$
	\text{output}[i_1, i_2, ..., i_n] = \text{input}[i_1, i_2, ..., i_n] + \text{residual}[i_1, i_2, ..., i_n]
	$$
	其中：
	- $\text{input}$：输入张量，形状记为 $S = (s_1, s_2, ..., s_n)$，$n$ 为张量维度数
	- $\text{residual}$：残差张量，形状与 $\text{input}$ 完全一致（即 $S = (s_1, s_2, ..., s_n)$），用于与输入张量执行逐元素相加
	- $\text{output}$：输出张量，形状与 $\text{input}$ 完全一致（即 $S = (s_1, s_2, ..., s_n)$），存储逐元素相加的结果
	- $[i_1, i_2, ..., i_n]$：张量任意位置的多维索引，满足 $0 \leq i_k < s_k$（$k = 1, 2, ..., n$），覆盖张量所有元素
	- $+$：逐元素加法运算符号
### 参数说明
- **input** (at:: Tensor, 入参): 基础输入张量，用于参与逐元素相加运算，形状需与 residual 完全一致
- **residual** (at:: Tensor, 入参): 残差/偏置张量，用于与 input 执行逐元素相加，形状、数据类型需与 input 完全匹配
- **dropout_prob** (float, 入参): dropout 随机丢弃概率，当前版本仅支持取值为 0.0
### 返回值
at::Tensor: 输出张量，形状与 input、residual 完全一致，存储 input 与 residual 逐元素相加的结果（当前无 dropout 处理）
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- input 与 residual 的形状、数据类型必须完全一致
- dropout_prob 仅支持取值为 0.0，非 0 值的 dropout 功能暂未实现
- 支持的数据类型：float16（at::ScalarType::Half）、bfloat16（at::ScalarType::BFloat16）
- 当 input 的元素总数为 8 的倍数时，算子会启用更高效的向量化处理路径
### 调用示例
```python
import torch
import mcoplib.op as op
# 设置计算设备
device = "cuda"
# 定义张量形状
batch_size = 8
seq_len = 512
hidden_size = 4096
# 创建输入张量：需保证input和residual形状、数据类型完全一致
input_tensor = torch.randn(
    batch_size, seq_len, hidden_size,
    dtype=torch.float16,
    device=device,
    requires_grad=True  
)
residual_tensor = torch.randn_like(input_tensor)  
# 调用fused_bias_dropout算子
dropout_prob = 0.0
output_tensor = op.fused_bias_dropout(input_tensor, residual_tensor, dropout_prob)
print("fused_bias_dropout computation completed")
print(f"Input shape: {input_tensor.shape}")
print(f"Residual shape: {residual_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
print(f"Data type: {output_tensor.dtype}")
```
## 2. fused_rope_fwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_rope_fwd(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indexes: Optional[torch.Tensor] = None,
    force_bf16_attn: bool = False
) -> torch.Tensor
```
### 功能描述
fused_rope_fwd 算子实现 RoPE 旋转位置编码的前向计算，是 Transformer 模型中位置编码的核心操作之一。该算子融合了 QKV 张量中 Query（Q）与 Key（K）的旋转逻辑，通过 cos、sin 张量对 Q、K 进行旋转变换，最终输出处理后的 QKV 张量。
- 计算公式：
  对 $QKV$ 张量中的 Query（$Q$）和 Key（$K$）分别执行旋转位置编码变换，核心公式如下：
  $$
  \begin{cases}
  q'_0 = q_0 \cdot \cos(\theta) - q_1 \cdot \sin(\theta) \\
  q'_1 = q_0 \cdot \sin(\theta) + q_1 \cdot \cos(\theta) \\
  k'_0 = k_0 \cdot \cos(\theta) - k_1 \cdot \sin(\theta) \\
  k'_1 = k_0 \cdot \sin(\theta) + k_1 \cdot \cos(\theta)
  \end{cases}
  $$
  其中：
  - $qkv \in \mathbb{R}^{\text{seq} \times \text{qkv\_num} \times \text{num\_head} \times \text{head\_dim}}$：输入的 $QKV$ 融合张量
  - $q_0, q_1 \in \mathbb{R}^{\text{seq} \times \text{num\_head} \times \text{head\_dim\_half}}$：$Q$ 张量在 $\text{head\_dim}$ 维度拆分后的两部分（$\text{head\_dim\_half} = \text{head\_dim}/2$）
  - $k_0, k_1 \in \mathbb{R}^{\text{seq} \times \text{num\_head} \times \text{head\_dim\_half}}$：$K$ 张量在 $\text{head\_dim}$ 维度拆分后的两部分
  - $\cos (\theta), \sin (\theta) \in \mathbb{R}^{\text{seq} \times \text{head\_dim\_half}}$：对应位置的余弦、正弦位置编码张量
  - $\text{seq}$：序列长度，$\text{qkv\_num}$：$Q/K/V$ 的数量（通常为 $3$），$\text{num\_head}$：注意力头数，$\text{head\_dim}$：单头维度
  变换后合并结果并更新至原 $QKV$ 张量：
  $$
  \text{qkv}[..., :\text{head\_dim\_half}] = q'_0, \quad \text{qkv}[..., \text{head\_dim\_half}:] = q'_1
  $$
  $$
  \text{qkv}[..., :\text{head\_dim\_half}] = k'_0, \quad \text{qkv}[..., \text{head\_dim\_half}:] = k'_1
  $$
### 参数说明
- **qkv** (torch.Tensor, 入参/出参): 输入的 QKV 融合张量，形状为[seq, qkv_num, num_head, head_dim]，处理后的结果直接写入该张量并作为返回值
- **cos** (torch.Tensor, 入参): RoPE 对应的余弦值张量，其第二维度需与 head_dim 的一半（head_dim_half）匹配
- **sin** (torch.Tensor, 入参): RoPE 对应的正弦值张量，维度要求与 cos 完全一致
- **indexes** (Optional[torch.Tensor], 入参): 可选的序列索引张量，数据类型需为 long，若传入则用于定位当前序列对应的 cos/sin 值
- **force_bf16_attn** (bool, 入参): 是否强制使用 bfloat16 精度执行计算，默认值为 False
### 返回值
返回处理后的 QKV 张量（与输入 qkv 为同一张量，结果已直接写入）
### 约束与调用
- 所有输入张量必须部署在 CUDA 设备上
- 输入张量需满足内存连续性要求
- 支持的数据类型：float16（Half）、bfloat16
- qkv 维度约束：seq=qkv.size (0)、qkv_num=qkv.size (1)、num_head=qkv.size (2)、head_dim=qkv.size (3)，且 head_dim=2*cos.size (1)
- 若传入 indexes，其数据类型必须为 at::scalar_type::long
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义张量维度
seq = 4096  # qkv.size(0)
qkv_num = 4  # qkv.size(1)
num_head = 64  # qkv.size(2)
head_dim = 128  # qkv.size(3)
head_dim_half = head_dim // 2  # 64，对应cos.size(1)
# 创建输入张量
qkv = torch.randn(seq, qkv_num, num_head, head_dim, dtype=torch.float16, device=device)
cos = torch.randn(seq, head_dim_half, dtype=torch.float16, device=device)
sin = torch.randn(seq, head_dim_half, dtype=torch.float16, device=device)
# 可选的indexes张量
indexes = torch.randint(0, seq, (seq,), dtype=torch.long, device=device)
# 若不需要indexes，可设为None：indexes = None
# 调用fused_rope_fwd算子
output_qkv = op.fused_rope_fwd(
    qkv,
    cos,
    sin,
    indexes,
    False
)
print("fused_rope_fwd computation completed")
print(f"Output QKV shape: {output_qkv.shape}")  # 应与输入qkv形状一致
```
## 3. fused_rope_bwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_rope_bwd(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indexes: Optional[torch.Tensor] = None,
    force_bf16_attn: bool = False
) -> torch.Tensor
```
### 功能描述
fused_rope_bwd 算子实现 RoPE（旋转位置编码）的反向传播计算，对应 fused_rope_fwd 的梯度回传逻辑。该算子通过 cos、sin 张量对 QKV 梯度张量执行反向旋转变换，完成梯度的反向传播处理
- 计算公式：
  对 $QKV$ 梯度张量中的 Query 梯度（$\nabla q$）和 Key 梯度（$\nabla k$）执行反向旋转变换，核心公式如下：
  $$
  \begin{cases}
  \nabla q'_0 = \nabla q_0 \cdot \cos(\theta) + \nabla q_1 \cdot \sin(\theta) \\
  \nabla q'_1 = -\nabla q_0 \cdot \sin(\theta) + \nabla q_1 \cdot \cos(\theta) \\
  \nabla k'_0 = \nabla k_0 \cdot \cos(\theta) + \nabla k_1 \cdot \sin(\theta) \\
  \nabla k'_1 = -\nabla k_0 \cdot \sin(\theta) + \nabla k_1 \cdot \cos(\theta)
  \end{cases}
  $$
  其中：
  - $\nabla \text{qkv} \in \mathbb{R}^{\text{seq} \times \text{qkv\_num} \times \text{num\_head} \times \text{head\_dim}}$：输入的 $QKV$ 融合梯度张量
  - $\nabla q_0, \nabla q_1 \in \mathbb{R}^{\text{seq} \times \text{num\_head} \times \text{head\_dim\_half}}$：$\nabla q$ 在 $\text{head\_dim}$ 维度拆分后的两部分
  - $\nabla k_0, \nabla k_1 \in \mathbb{R}^{\text{seq} \times \text{num\_head} \times \text{head\_dim\_half}}$：$\nabla k$ 在 $\text{head\_dim}$ 维度拆分后的两部分
  - $\cos (\theta), \sin (\theta) \in \mathbb{R}^{\text{seq} \times \text{head\_dim\_half}}$：对应位置的余弦、正弦位置编码张量（与前向一致）
  - $\text{seq}$：序列长度，$\text{qkv\_num}$：$Q/K/V$ 的数量，$\text{num\_head}$：注意力头数，$\text{head\_dim}$：单头维度
  变换后合并结果并更新至原 $QKV$ 梯度张量：
  $$
  \nabla \text{qkv}[..., :\text{head\_dim\_half}] = \nabla q'_0, \quad \nabla \text{qkv}[..., \text{head\_dim\_half}:] = \nabla q'_1
  $$
  $$
  \nabla \text{qkv}[..., :\text{head\_dim\_half}] = \nabla k'_0, \quad \nabla \text{qkv}[..., \text{head\_dim\_half}:] = \nabla k'_1
  $$
### 参数说明
- **qkv** (torch.Tensor, 入参/出参): 输入的 QKV 梯度张量，形状为[seq, qkv_num, num_head, head_dim]；处理后的梯度结果直接写入该张量并作为返回值
- **cos** (torch.Tensor, 入参): RoPE 对应的余弦值张量，其第二维度需与 head_dim 的一半（head_dim_half）匹配
- **sin** (torch.Tensor, 入参): RoPE 对应的正弦值张量，维度要求与 cos 完全一致
- **indexes** (Optional[torch.Tensor], 入参): 可选的序列索引张量，数据类型需为 long，若传入则用于定位当前序列对应的 cos/sin 值
- **force_bf16_attn** (bool, 入参): 是否强制使用 bfloat16 精度执行计算，默认值为 False
### 返回值
返回处理后的 QKV 梯度张量，与输入`qkv 为同一张量，结果已直接写入
### 约束与调用
- 所有输入张量必须部署在 CUDA 设备上
- 输入张量需满足内存连续性要求
- 支持的数据类型：float16（Half）、bfloat16
- qkv 维度约束：seq=qkv.size (0)、qkv_num=qkv.size (1)、num_head=qkv.size (2)、head_dim=qkv.size (3)，且`ead_dim = 2 * cos.size (1)
- 若传入 indexes，其数据类型必须为 at::scalar_type::long
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义与前向一致的张量维度
seq = 4096
qkv_num = 4
num_head = 64
head_dim = 128
head_dim_half = head_dim // 2
# 创建反向传播的输入张量
qkv_grad = torch.randn(seq, qkv_num, num_head, head_dim, dtype=torch.float16, device=device)
# 复用前向的cos/sin
cos = torch.randn(seq, head_dim_half, dtype=torch.float16, device=device)
sin = torch.randn(seq, head_dim_half, dtype=torch.float16, device=device)
# 可选的indexes张量
indexes = torch.randint(0, seq, (seq,), dtype=torch.long, device=device)
# 调用fused_rope_bwd算子
output_qkv_grad = op.fused_rope_bwd(
    qkv_grad,
    cos,
    sin,
    indexes,
    False
)
print("fused_rope_bwd computation completed")
print(f"Output QKV gradient shape: {output_qkv_grad.shape}")
```
## 4. fused_bias_swiglu_fwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_bias_swiglu_fwd(
    input: at::Tensor
) -> at::Tensor
```
### 功能描述
fused_bias_swiglu_fwd 算子实现融合 Bias 的 SwiGLU 激活前向计算，是大模型（Transformer 类）中高效处理中间特征的算子。算子将张量沿最后一个维度分成两部分，并行计算 SwiGLU 激活，拆分后第一部分与 sigmoid (拆分后第二部分) 相乘，并通过算子融合减少内存读写开销
- 计算公式：
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$
  $$
  x_1 = \text{input}[..., :d] + b_1
  $$
  $$
  x_2 = \text{input}[..., d:] + b_2
  $$
  $$
  \text{output} = x_1 \odot \sigma(x_2)
  $$
  其中：
	- $\text{input} \in \mathbb{R}^{N \times M \times 2d}$ 是 CUDA 设备上的半精度（half/bfloat16）连续张量，$N$ 为 batch 维度，$M$ 为序列长度维度
	- $d = \frac{\text{input.size (-1)}}{2}$，即输入最后一维长度的 1/2（hidden_size 的 1/2）
	- $x_1$ 是输入张量沿最后一维拆分的前半部分，$x_2$ 是拆分的后半部分
	- $b_1, b_2$ 为对应维度的偏置项（当前实现中暂未启用，默认为 0）
	- $\sigma (z)$ 是 sigmoid 激活函数
	- $\odot$ 表示逐元素乘法
	- $\text{output} \in \mathbb{R}^{N \times M \times d}$ 是输出张量，与输入同设备、同数据类型
### 参数说明
- **input** (at:: Tensor, 入参): 前向计算的原始输入张量，用于执行融合 Bias 的 SwiGLU 激活计算。最后一维长度需为偶数，用于拆分为两等份，形状通常为「N×M×2d」（N 为 batch 维度、M 为序列长度维度、d 为 hidden_size 的 1/2）。
### 返回值
at::Tensor: 前向计算输出张量，与输入 input 同设备、同数据类型；形状为「N×M×d」（最后一维长度为 input 的 1/2），存储 input 经 SwiGLU 激活（拆分后前半部分×sigmoid (拆分后后半部分)）计算后的结果。
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- input 张量需为连续内存
- input 最后一维长度必须为偶数，需拆分为两等份用于 SwiGLU 计算
- 支持的数据类型：float16（at::ScalarType::Half）和 bfloat16（at::ScalarType::BFloat16）
- 当 input 的元素总数为 8 的倍数时，算子会启用更高效的向量化批量处理
- 偏置项（b1、b2）当前实现暂未启用，默认按 0 参与计算
### 调用示例
```python
import torch
import mcoplib.op as op
# 设置计算设备
device = "cuda"
# 定义张量形状
batch_size = 8
seq_len = 512
hidden_size = 4096 
# 创建前向输入张量
input_tensor = torch.randn(
    batch_size, seq_len, hidden_size,
    dtype=torch.float16,
    device=device,
    requires_grad=True
)
# 调用fused_bias_swiglu_fwd算子
output_tensor = op.fused_bias_swiglu_fwd(input_tensor)
print("fused_bias_swiglu_fwd computation completed")
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}") 
print(f"Data type: {output_tensor.dtype}")
print(f"Device: {output_tensor.device}")
```
## 5. fused_bias_swiglu_bwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_bias_swiglu_bwd(
    input: at::Tensor,
    grad_output: at::Tensor
) -> at::Tensor
```
### 功能描述
fused_bias_swiglu_bwd 算子实现 SwiGLU 激活对应的反向梯度计算，是大模型训练中保障梯度传播的关键算子。算子接收前向计算的原始输入张量和输出梯度张量，通过 CUDA 多线程并行，结合链式法则推导输入梯度，利用原输入计算 sigmoid 导数并与输出梯度相乘，输出与原输入维度一致的半精度梯度张量，为模型参数更新提供支撑
- 计算公式：
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$
  $$
  \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))
  $$
  $$
  x_1 = \text{input}[..., :d] + b_1
  $$
  $$
  x_2 = \text{input}[..., d:] + b_2
  $$
  $$
  \nabla x_1 = G \odot \sigma(x_2)
  $$
  $$
  \nabla x_2 = G \odot x_1 \odot \sigma'(x_2)
  $$
  $$
  \nabla \text{input} = \text{concat}(\nabla x_1, \nabla x_2, \text{dim}=-1)
  $$
  其中：
  - $\text{input} \in \mathbb{R}^{N \times M \times 2d}$ 是前向计算的原始输入张量（CUDA 设备、半精度、连续）
  - $G \in \mathbb{R}^{N \times M \times d}$ 是前向输出张量对应的梯度张量（与输入同设备、同数据类型）
  - $d = \frac{\text{input.size (-1)}}{2}$，与前向计算中$d$的定义一致
  - $x_1, x_2$ 是输入张量沿最后一维拆分的两部分，$b_1, b_2$ 为偏置项（暂未启用，默认为 0）
  - $\sigma (z)$ 是 sigmoid 函数，$\sigma' (z)$ 是 sigmoid 函数的导数
  - $\nabla x_1$ 是拆分前半部分$x_1$的梯度，$\nabla x_2$ 是拆分后半部分$x_2$的梯度
  - $\odot$ 表示逐元素乘法，$\text{concat}(\cdot, \cdot, \text{dim}=-1)$ 表示沿最后一维拼接
  - $\nabla \text{input} \in \mathbb{R}^{N \times M \times 2d}$ 是输出的输入梯度张量，与原始输入$\text{input}$维度、设备、数据类型完全一致
### 参数说明
- **input** (at:: Tensor, 入参): 前向计算时的原始输入张量，用于反向梯度推导的基础数据，需连续内存，形状为[N×M×2d]，与前向计算的 input 完全一致
- **grad_output** (at:: Tensor, 入参): 前向输出张量对应的梯度张量，用于传递反向传播的梯度信号, 需与 input 同设备、同数据类型且为连续内存. 形状与前向输出张量形状一致，最后一维长度为 input 的 1/2
### 返回值
at::Tensor: 输入梯度输出张量，与原始输入 input 同设备、同数据类型、同形状[N×M×2d], 存储通过链式法则推导的 input 对应的梯度值，为模型参数更新提供支撑
### 约束与调用
- 所有张量必须位于 CUDA 设备上，且 input 与 grad_output 需在同一设备
- input 和 grad_output 均需为连续内存，数据类型必须完全一致，仅支持 float16、bfloat16
- input 最后一维长度必须是 grad_output 最后一维长度的 2 倍，与前向计算的拆分逻辑匹配
- 当 grad_output 的元素总数为 8 的倍数时，算子会启用更高效的向量化批量处理
- 偏置项（b1、b2）当前实现暂未启用，梯度计算中默认按偏置为 0 推导
### 调用示例
```python
import torch
import mcoplib.op as op
# 设置计算设备
device = "cuda"
# 定义张量形状
batch_size = 8
seq_len = 512
hidden_size = 4096  
output_hidden_size = hidden_size // 2  
# 先执行前向计算
input_tensor = torch.randn(
    batch_size, seq_len, hidden_size,
    dtype=torch.float16,
    device=device,
    requires_grad=True
)
forward_output = fused_bias_swiglu_fwd(input_tensor)
# 创建输出梯度张量
grad_output = torch.randn_like(forward_output)
# 调用fused_bias_swiglu_bwd算子
grad_input = op.fused_bias_swiglu_bwd(input_tensor, grad_output)
print("fused_bias_swiglu_bwd computation completed")
print(f"Original input shape: {input_tensor.shape}")
print(f"Grad output shape: {grad_output.shape}")
print(f"Input gradient shape: {grad_input.shape}")  
print(f"Data type: {grad_input.dtype}")
print(f"Device: {grad_input.device}")
```
## 6. fused_repeat_kv_fwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_repeat_kv_fwd(
    input: at::Tensor,
    q_num_head: int,
    kv_num_head: int,
    head_dim: int
) -> at::Tensor
```
### 功能描述
fused_repeat_kv_fwd 是多查询注意力和组查询注意力场景下的前向融合算子，该算子将数量较少的 KV 头，重复扩展至与查询头相同的数量，同时拆分出 Q、K、V 三类张量，通过 CUDA 核融合操作替代 PyTorch 原生的多步操作，提升计算与内存访问效率
- 计算公式：
	输入张量形状：
	$$
	\text{input} \in \mathbb{R}^{\text{seq} \times \text{bs} \times \text{partition} \times (\text{q\_num\_head} + 2 \cdot \text{kv\_num\_head}) \cdot \text{head\_dim}}
	$$
	输出张量形状：
	$$
	\text{output} \in \mathbb{R}^{\text{seq} \times \text{bs} \times 3 \times \text{q\_num\_head} \times \text{head\_dim}}
	$$
	拆分输入为 Q、K、V 分量：
	$$
	\begin{cases}
	\text{Q}_{\text{input}} = \text{input}[..., :\text{q\_num\_head} \cdot \text{head\_dim}] \\
	\text{K}_{\text{input}} = \text{input}[..., \text{q\_num\_head} \cdot \text{head\_dim} : (\text{q\_num\_head} + \text{kv\_num\_head}) \cdot \text{head\_dim}] \\
	\text{V}_{\text{input}} = \text{input}[..., (\text{q\_num\_head} + \text{kv\_num\_head}) \cdot \text{head\_dim} :]
	\end{cases}
	$$
	KV 头重复次数：
	$$
	\text{repeat} = \frac{\text{q\_num\_head}}{\text{kv\_num\_head}}
	$$
	重复 KV 分量至匹配 Q 头数：
	$$
	\begin{cases}
	\text{K}_{\text{repeated}} = \text{repeat}(\text{K}_{\text{input}}, \text{dim}=3, \text{repeats}=\text{repeat}) \\
	\text{V}_{\text{repeated}} = \text{repeat}(\text{V}_{\text{input}}, \text{dim}=3, \text{repeats}=\text{repeat})
	\end{cases}
	$$
	映射至输出张量：
	$$
	\begin{cases}
	\text{output}[..., 0, :, :] = \text{reshape}(\text{Q}_{\text{input}}, [\text{seq}, \text{bs}, \text{q\_num\_head}, \text{head\_dim}]) \\
	\text{output}[..., 1, :, :] = \text{reshape}(\text{K}_{\text{repeated}}, [\text{seq}, \text{bs}, \text{q\_num\_head}, \text{head\_dim}]) \\
	\text{output}[..., 2, :, :] = \text{reshape}(\text{V}_{\text{repeated}}, [\text{seq}, \text{bs}, \text{q\_num\_head}, \text{head\_dim}])
	\end{cases}
	$$
	其中：
	- $\text{seq}$ 是序列长度
	- $\text{bs}$ 是 batch 大小
	- $\text{partition}$ 是张量在头维度的分片数
	- $\text{q\_num\_head}$ 是查询头总数
	- $\text{kv\_num\_head}$ 是键/值头总数
	- $\text{head\_dim}$ 是单个注意力头的维度
	- $\text{repeat}$ 是 KV 头需要重复的次数
### 参数说明
- **input** (at:: Tensor, 入参): 合并的 QKV 输入张量，形状为`[seq, bs, partition, (q_num_head+2*kv_num_head)*head_dim]`
- **q_num_head** (int, 入参): 查询头总数，需为 kv_num_head 的整数倍, 保证 KV 头可均匀重复扩展
- **kv_num_head** (int, 入参): 键/值头总数，与 q_num_head 配合确定 KV 头的重复次数
- **head_dim** (int, 入参): 单个注意力头的维度，用于计算输入/输出张量的维度大小
### 返回值
返回拆分并重复后的 QKV 张量（at::Tensor），形状为`[seq, bs, 3, q_num_head, head_dim]`，其中第 2 维的 3 分别对应 Q、K、V 三个分量
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入张量需为 4 维连续存储的张量
- 支持的数据类型：仅 float16（Half）、bfloat16
- 查询头数 q_num_head 必须是 KV 头数 kv_num_head 的整数倍
- partition 参数需为合法分片数
- 输入/输出的维度需匹配，输入：`[seq, bs, partition, (q_num_head+kv_num_head*2)*head_dim]`，输出：`[seq, bs, 3, q_num_head, head_dim]`）
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义算子参数
seq_len = 4096       
batch_size = 2       
partition = 2       
q_num_head = 128     
kv_num_head = 8      
head_dim = 128      
dtype = torch.bfloat16  
# 计算输入张量形状：
input_dim = (q_num_head // partition + 2 * (kv_num_head // partition)) * head_dim
qkv_input = torch.randn(
    seq_len, batch_size, partition, input_dim,
    dtype=dtype, device=device
)
# 调用fused_repeat_kv_fwd算子
output_fwd = op.fused_repeat_kv_fwd(
    qkv_input,
    q_num_head=q_num_head,
    kv_num_head=kv_num_head,
    head_dim=head_dim
)
print("fused_repeat_kv_fwd computation completed")
print(f"输入张量形状: {qkv_input.shape}")
print(f"输出张量形状: {output_fwd.shape}") 
```
## 7. fused_repeat_kv_bwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_repeat_kv_bwd(
    input: at::Tensor,
    q_num_head: int,
    kv_num_head: int,
    partition: int
) -> at::Tensor
```
### 功能描述
fused_repeat_kv_bwd 是 fused_repeat_kv_fwd 对应的反向传播算子，用于多查询注意力和组查询注意力场景下的梯度传递，该算子将前向扩展后的 KV 头维度对应的输出梯度累加汇总回原始 KV 头数量对应的输入梯度形状，实现自动微分的链式法则逻辑，通过 CUDA 核融合提升反向计算效率
- 计算公式：
	输入张量形状（前向输出梯度）：
	$$
	\text{input} \in \mathbb{R}^{\text{seq} \times \text{bs} \times 3 \times \text{q\_num\_head} \times \text{head\_dim}}
	$$
	输出张量形状（前向输入梯度）：
	$$
	\text{output} \in \mathbb{R}^{\text{seq} \times \text{bs} \times \text{partition} \times \frac{(\text{q\_num\_head} + 2 \cdot \text{kv\_num\_head}) \cdot \text{head\_dim}}{\text{partition}}}
	$$
	拆分输入为 Q、K、V 梯度分量：
	$$
	\begin{cases}
	\text{Q}_{\text{grad}} = \text{input}[..., 0, :, :] \\
	\text{K}_{\text{grad}} = \text{input}[..., 1, :, :] \\
	\text{V}_{\text{grad}} = \text{input}[..., 2, :, :]
	\end{cases}
	$$
	KV 头重复次数（与前向一致）：
	$$
	\text{repeat} = \frac{\text{q\_num\_head}}{\text{kv\_num\_head}}
	$$
	累加 KV 梯度至原始头数：
	$$
	\begin{cases}
	\text{K}_{\text{grad\_agg}} = \text{sum}\left( 
	    \text{reshape}(\text{K}_{\text{grad}}, [\text{seq}, \text{bs}, \text{repeat}, \text{kv\_num\_head}, \text{head\_dim}]), 
	    \text{dim}=2 
	\right) \\
	\text{V}_{\text{grad\_agg}} = \text{sum}\left( 
	    \text{reshape}(\text{V}_{\text{grad}}, [\text{seq}, \text{bs}, \text{repeat}, \text{kv\_num\_head}, \text{head\_dim}]), 
	    \text{dim}=2 
	\right)
	\end{cases}
	$$
	将各梯度分量展平为 2D（最后一维合并）：
	$$
	\text{Q}_{\text{grad\_flat}} = \text{reshape}(\text{Q}_{\text{grad}}, [\text{seq}, \text{bs}, \text{q\_num\_head} \cdot \text{head\_dim}])
	$$
	
	$$
	\text{K}_{\text{grad\_agg\_flat}} = \text{reshape}(\text{K}_{\text{grad\_agg}}, [\text{seq}, \text{bs}, \text{kv\_num\_head} \cdot \text{head\_dim}])
	$$
	
	$$
	\text{V}_{\text{grad\_agg\_flat}} = \text{reshape}(\text{V}_{\text{grad\_agg}}, [\text{seq}, \text{bs}, \text{kv\_num\_head} \cdot \text{head\_dim}])
	$$
	拼接展平后的梯度分量（沿最后一维）：
	$$
	\text{concat\_grad} = \text{concat}(\text{Q}_{\text{grad\_flat}}, \text{K}_{\text{grad\_agg\_flat}}, \text{V}_{\text{grad\_agg\_flat}}, \text{dim}=-1)
	$$
	分片输出梯度（还原为输入张量的形状）：
	$$
	\text{output} = \text{reshape}(\text{concat\_grad}, [
	    \text{seq}, \text{bs}, \text{partition}, \\
	    \frac{(\text{q\_num\_head} + 2 \cdot \text{kv\_num\_head}) \cdot \text{head\_dim}}{\text{partition}}
	])
	$$
	约束条件：
	$$
	(\text{q\_num\_head} + 2 \cdot \text{kv\_num\_head}) \cdot \text{head\_dim} \quad \text{必须能被} \quad \text{partition} \quad \text{整除}
	$$
	其中：
	- $\text{seq}$ 是序列长度
	- $\text{bs}$ 是 batch 大小
	- $\text{partition}$ 是张量在头维度的分片数（用于张量并行）
	- $\text{q\_num\_head}$ 是查询头总数
	- $\text{kv\_num\_head}$ 是键/值头总数
	- $\text{head\_dim}$ 是单个注意力头的维度
	- $\text{repeat}$ 是 KV 头在前向过程中重复的次数（必须为整数）
	- $\text{concat\_grad}$ 是拼接后的梯度张量（展平状态）
	- $\text{K}_{\text{grad\_agg}}$、$\text{V}_{\text{grad\_agg}}$ 是累加后的 KV 梯度（还原至原始 KV 头数）
	- $\text{Q}_{\text{grad\_flat}}$、$\text{K}_{\text{grad\_agg\_flat}}$、$\text{V}_{\text{grad\_agg\_flat}}$ 是各梯度展平后的中间变量
### 参数说明
- **input** (at:: Tensor, 入参): 前向输出张量的梯度，形状为[seq, bs, 3, q_num_head, head_dim]，与 fused_repeat_kv_fwd 的输出形状一致
- **q_num_head** (int, 入参): 查询头总数，需为 kv_num_head 的整数倍，与前向算子保持一致
- **kv_num_head** (int, 入参): 键/值头总数，用于确定梯度累加的分组数量
- **partition** (int, 入参): 张量在头维度的分片数，用于张量并行，需满足`(q_num_head+2*kv_num_head)*head_dim` 能被其整除
### 返回值
返回前向输入张量的梯度（at::Tensor），形状为`[seq, bs, partition, (q_num_head+2*kv_num_head)*head_dim/partition]`，与 `fused_repeat_kv_fwd` 的输入形状匹配
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入张量需为 5 维连续存储的张量
- 支持的数据类型：仅 float16（Half）、bfloat16
- 查询头数 q_num_head 必须是 KV 头数 kv_num_head 的整数倍
- 输入张量的最后一维维度需与 head_dim 匹配
- 输出张量的维度由输入维度与 partition 参数计算生成，需满足 `(q_num_head+kv_num_head*2)*head_dim/partition` 的合法性
- 输入/输出的维度需匹配，输入：`[seq, bs, partition, (q_num_head+kv_num_head*2)*head_dim]`，输出：`[seq, bs, partition, (q_num_head+kv_num_head*2)*head_dim/partition]`
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 复用fused_repeat_kv_fwd的参数配置
seq_len = 4096
batch_size = 2
partition = 2
q_num_head = 128
kv_num_head = 8
head_dim = 128
dtype = torch.bfloat16
# 先获取fused_repeat_kv_fwd的输出（作为反向算子的输入）
fwd_output = torch.randn(
    seq_len, batch_size, 3, q_num_head, head_dim,
    dtype=dtype, device=device
)
# 调用fused_repeat_kv_bwd算子
output_bwd = op.fused_repeat_kv_bwd(
    fwd_output,
    q_num_head,
    kv_num_head,
    partition
)
print("fused_repeat_kv_bwd computation completed")
print(f"输入张量形状: {fwd_output.shape}")
print(f"输出张量形状: {output_bwd.shape}")  
```
## 8. fused_gelu_fwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_gelu_fwd(
    input: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor
```
### 功能描述
fused_gelu_fwd 算子实现了加偏置和 GELU 激活的融合计算操作。该算子首先将偏置张量按 input 的 hidden 维度对应叠加到输入张量上，随后对叠加结果应用 GELU 激活函数。融合操作减少了中间结果的内存读写次数，提升计算效率，常用于神经网络激活层前的偏置叠加与激活组合流程。
- 计算公式：
	加偏置操作：  
	$$
	T_{b, s, h} = \text{input}_{b, s, h} + \text{bias}_h
	$$
	GELU 激活操作，标准正态分布累积分布函数近似：  
	$$
	\text{output}_{b, s, h} = T_{b, s, h} \cdot 0.5 \cdot \left(1 + \text{erf}\left(\frac{T_{b, s, h}}{\sqrt{2}}\right)\right)
	$$
	其中：  
	- $\text{input} \in \mathbb{R}^{B \times S \times H}$ 为输入张量（$B$=batchsize，$S$=序列长度，$H$=hidden 维度）；  
	- $\text{bias} \in \mathbb{R}^H$ 为偏置张量，通过广播机制匹配 $\text{input}$ 的 $H$ 维度；  
	- $T$ 为加偏置后的中间张量（形状与 $\text{input}$ 一致）；  
	- $\text{erf}(\cdot)$ 为误差函数；  
	- $\text{output} \in \mathbb{R}^{B \times S \times H}$ 为最终输出张量，形状与 $\text{input}$ 一致。  
### 参数说明
- **input** (torch.Tensor, 入参): 待处理的输入张量，需进行加偏置与 GELU 激活计算
- **bias** (torch.Tensor, 入参): 偏置张量，形状需匹配 input 的 hidden 维度
### 返回值
返回 torch.Tensor 类型的输出张量，形状与 input 完全一致，存储计算结果
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- input 需包含 hidden 维度
- bias 的形状需与 input 的 hidden 维度匹配
- 支持的数据类型：float32, float16, bfloat16
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 参数配置
batch_size = 2
seq_len = 1024
hidden_size = 768  
dtype = torch.bfloat16 
# 创建输入张量
input = torch.randn(
    batch_size, seq_len, hidden_size,
    dtype=dtype, device=device
)
bias = torch.randn(
    hidden_size,  
    dtype=dtype, device=device
)
# 调用fused_gelu_fwd算子
output_fwd = op.fused_gelu_fwd(input, bias)
# 验证输出
print("fused_gelu_fwd computation completed")
print(f"输入input形状: {input.shape}")
print(f"输入bias形状: {bias.shape}")
print(f"输出output_fwd形状: {output_fwd.shape}")
```
## 9. fused_gelu_bwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_gelu_bwd(
    input: torch.Tensor,
    input1: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor
```
### 功能描述
fused_gelu_bwd 算子实现了加偏置和 GELU 激活的反向传播融合计算。该算子根据前向计算的输入张量、偏置张量，以及上游传递的梯度张量，计算得到对应前向输入的梯度结果。融合操作减少了反向传播的内存交互开销，提升计算效率，是 fused_gelu_fwd 算子的配套反向算子
- 计算公式：
	前向加偏置的中间张量：  
	$$
	T_{b, s, h} = \text{input}_{b, s, h} + \text{bias}_h
	$$
	GELU 函数在 $T$ 处的导数，链式法则中间项：  
	$$
	\text{gelu\_deriv}_{b, s, h} = 0.5 \cdot \left(1 + \text{erf}\left(\frac{T_{b, s, h}}{\sqrt{2}}\right)\right) + \frac{T_{b, s, h}}{\sqrt{2\pi}} \cdot \exp\left(-\frac{T_{b, s, h}^2}{2}\right)
	$$
	输入张量 $\text{input}$ 的梯度，链式法则传递上游梯度：  
	$$
	\text{output}_{b, s, h} = \text{input1}_{b, s, h} \cdot \text{gelu\_deriv}_{b, s, h}
	$$
	其中：  
	- $\text{input} \in \mathbb{R}^{B \times S \times H}$ 为前向输入张量；  
	- $\text{input1} \in \mathbb{R}^{B \times S \times H}$ 为上游传递的梯度张量（对应前向输出 $\text{output}$ 的梯度）；  
	- $\text{bias} \in \mathbb{R}^H$ 为前向使用的偏置张量；  
	- $T$ 为前向加偏置后的中间张量（形状与 $\text{input}$ 一致）；  
	- $\text{gelu\_deriv}$ 为 GELU 函数在 $T$ 处的导数张量（形状与 $\text{input}$ 一致）；  
	- $\text{output} \in \mathbb{R}^{B \times S \times H}$ 为最终输出张量（对应前向输入 $\text{input}$ 的梯度，形状与 $\text{input}$ 一致）；  
	- $\text{erf}(\cdot)$ 为误差函数，$\exp (\cdot)$ 为指数函数。
### 参数说明
- **input** (torch.Tensor, 入参): 前向计算时的输入张量
- **input1** (torch.Tensor, 入参): 上游传递的梯度张量，对应 fused_gelu_fwd 输出的梯度
- **bias** (torch.Tensor, 入参): 前向计算时使用的偏置张量，形状需匹配 input 的 hidden 维度
### 返回值
返回 torch.Tensor 类型的输出张量，形状与 input 完全一致，存储对应前向输入的梯度计算结果
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- input 与 input1 的形状必须完全相同
- bias 的形状需与 input 的 hidden 维度匹配
- 支持的数据类型：float32, float16, bfloat16
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 复用fused_gelu_fwd的参数配置
batch_size = 2
seq_len = 1024
hidden_size = 768
dtype = torch.bfloat16
# 创建输入张量
input = torch.randn( 
    batch_size, seq_len, hidden_size,
    dtype=dtype, device=device
)
input1 = torch.randn( 
    batch_size, seq_len, hidden_size,
    dtype=dtype, device=device
)
bias = torch.randn(  
    hidden_size,
    dtype=dtype, device=device
)
# 调用fused_gelu_bwd算子
output_bwd = op.fused_gelu_bwd(input, input1, bias)
# 验证输出
print("fused_gelu_bwd computation completed")
print(f"前向input形状: {input.shape}")
print(f"上游梯度input1形状: {input1.shape}")
print(f"输出output_bwd形状: {output_bwd.shape}")
```
## 10. moe_swiglu_dynamic_quantize
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def moe_swiglu_dynamic_quantize(
    scatter_tokens: torch.Tensor,
    smooth_scale: torch.Tensor,
    experts_tokens_start: torch.Tensor,
    experts_tokens_count: torch.Tensor,
    y: torch.Tensor,
    per_tokens_scale: torch.Tensor,
    total_experts_num: int
) -> None
```
### 功能描述
moe_swiglu_dynamic_quantize 算子实现混合专家（MoE）架构下 SwigLU 计算的动态量化逻辑。该算子针对分散的 tokens 数据，结合各专家对应的 tokens 分布信息（起始位置、数量）与平滑缩放因子，执行动态量化处理，输出量化后的结果张量及每个 token 对应的缩放因子，以此提升 MoE 模块的计算效率并降低内存开销。
- 计算公式：
	对于第 $k$ 个专家（$k \in \{0, 1, ..., K-1\}$，$K = \text{total\_experts\_num}$）：
	$$
	X_k = \text{scatter\_tokens}[s_k : s_k + c_k, :]
	$$
	其中：
	- $s_k = \text{experts\_tokens\_start}[k]$：第$k$个专家的 tokens 在$\text{scatter\_tokens}$中的起始索引
	- $c_k = \text{experts\_tokens\_count}[k]$：第$k$个专家包含的 tokens 数量
	平滑缩放处理，对专家$k$的 tokens 子集应用平滑缩放因子：
	$$
	X_k' = X_k \odot S_k
	$$
	其中：
	- $S_k = \text{smooth\_scale}[k, :] \in \mathbb{R}^D$（$D = \text{hidden\_size}$为隐藏层维度）
	- $\odot$表示逐元素乘法（$S_k$广播至$X_k$的形状$[c_k, D]$）
	逐 token 动态缩放因子计算，对于$X_k'$中的第$i$个 token $x'_i \in \mathbb{R}^D$（$i \in \{0, 1, ..., c_k-1\}$），计算其缩放因子：
	$$
	\text{scale}_i = \max_{j=1..D} |x'_{i,j}|
	$$
	动态量化输出，将缩放后的 token 值归一化到缩放因子范围内，得到量化结果：
	$$
	y_{s_k + i, :} = \frac{x'_{i, :}}{\text{scale}_i}
	$$
	同时记录该 token 对应的缩放因子：
	$$
	\text{per\_tokens\_scale}[s_k + i] = \text{scale}_i
	$$
	融合操作的单步表达式：
	$$
	\text{moe\_swiglu\_dynamic\_quantize}(\text{scatter\_tokens}, \text{smooth\_scale}, \text{experts\_tokens\_start}, \text{experts\_tokens\_count}, K) = \left( y, \text{per\_tokens\_scale} \right)
	$$
	其中，
	- $K = \text{total\_experts\_num}$：混合专家架构的专家总数
	- $\text{scatter\_tokens} \in \mathbb{R}^{N \times D}$：所有专家的分散 tokens 集合（$N$为总 token 数）
	- $D = \text{hidden\_size}$：tokens 的隐藏层维度
	- $y \in \mathbb{R}^{N \times D}$：量化后的输出张量（结果写入出参）
	- $\text{per\_tokens\_scale} \in \mathbb{R}^N$：每个 token 对应的动态缩放因子（结果写入出参）
### 参数说明
- **scatter_tokens** (torch.Tensor, 入参): 分散的 tokens 输入张量，形状为[总 token 数, hidden_size]，存储待量化的 MoE tokens 数据
- **smooth_scale** (torch.Tensor, 入参): 平滑缩放因子张量，形状为[total_experts_num, hidden_size]，用于 SwigLU 计算中的缩放调整
- **experts_tokens_start** (torch.Tensor, 入参): 各专家对应的 tokens 起始位置张量，形状为[total_experts_num]，标记每个专家的 tokens 在 scatter_tokens 中的起始索引
- **experts_tokens_count** (torch.Tensor, 入参): 各专家对应的 tokens 数量张量，形状为[total_experts_num]，标记每个专家包含的 tokens 数量
- **y** (torch.Tensor, 出参): 量化后的输出张量，形状与 scatter_tokens 一致（[总 token 数, hidden_size]），存储动态量化后的结果
- **per_tokens_scale** (torch.Tensor, 出参): 每个 token 对应的缩放因子张量，形状为[总 token 数]，存储动态量化过程中每个 token 对应的缩放系数
- **total_experts_num** (int, 入参): 混合专家架构中的专家总数量
### 返回值
无返回值，量化结果直接写入出参张量 y，每个 token 的缩放因子直接写入出参张量 per_tokens_scale 中
### 约束与调用
- 所有输入/输出张量必须位于 CUDA 设备上
- 输入张量需满足连续存储约束
- scatter_tokens 的维度需为 2 维, 总 token 数×hidden_size
- experts_tokens_start、experts_tokens_count 的长度必须与 total_experts_num 一致
- 支持的数据类型：float16、bfloat16、float32, 对应代码中类型转换逻辑支持的类型
- smooth_scale 的形状需匹配[total_experts_num, hidden_size]，与 scatter_tokens 的 hidden_size 维度一致
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义参数
total_experts_num = 8  
num_tokens = 2048    
hidden_size = 4096     
# 输入张量：scatter_tokens（待量化的token特征）
scatter_tokens = torch.randn(
    num_tokens, hidden_size,
    dtype=torch.float16, 
    device=device, 
    requires_grad=True
)
# 输入张量：smooth_scale（量化的平滑缩放系数）
smooth_scale = torch.randn(
    total_experts_num, 3, hidden_size//2, 2,
    dtype=torch.float32,
    device=device,
    requires_grad=True
)
# 元数据：每个专家的token数量（experts_token_count）
experts_token_count = torch.randint(
    low=1, high=10,
    size=(total_experts_num,),
    dtype=torch.int32,
    device=device
)
# 确保总token数不超过num_tokens，调整最后一个专家的count
experts_token_count[-1] = num_tokens - experts_token_count[:-1].sum()
# 元数据：每个专家的token起始索引（experts_token_start）
experts_token_start = torch.zeros_like(experts_token_count)
for i in range(1, total_experts_num):
    experts_token_start[i] = experts_token_start[i-1] + experts_token_count[i-1]
# 输出张量：量化后的token（y）
y = torch.empty(
    num_tokens, hidden_size//2,
    dtype=torch.int8,
    device=device
)
# 输出张量：每个token的量化缩放系数（per_tokens_scale）
per_tokens_scale = torch.empty(
    num_tokens, hidden_size//2,
    dtype=torch.float32,
    device=device
)
# 调用moe_swiglu_dynamic_quantize算子
op.moe_swiglu_dynamic_quantize(
    scatter_tokens=scatter_tokens,
    smooth_scale=smooth_scale,
    experts_token_start=experts_token_start,
    experts_token_count=experts_token_count,
    y=y,
    per_tokens_scale=per_tokens_scale,
    total_experts_num=total_experts_num
)
print("moe_swiglu_dynamic_quantize computation completed")
print(f"Output y shape: {y.shape}, data type: {y.dtype}")
print(f"Output per_tokens_scale shape: {per_tokens_scale.shape}, data type: {per_tokens_scale.dtype}")
```
## 11. moe_softmax_topk
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def moe_softmax_topk(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    pre_softmax: bool
) -> None
```
### 功能描述
moe_softmax_topk 算子实现混合专家（MoE）架构下 gating 网络输出的 Top-K Softmax 处理逻辑。该算子接收 gating 网络的输出张量，根据 pre_softmax 标记判断是否对输入执行 Softmax 计算，随后筛选出每个 token 对应的 Top-K 个专家的权重与索引，最终将结果写入指定的输出张量，为 MoE 架构的专家选择环节提供核心的 Top-K 专家信息。
- 计算公式
	对于第 $k$ 个专家（$k \in \{0, 1, ..., K-1\}$，其中 $K = \text{total\_experts\_num}$），从输入张量 scatter_tokens 中提取属于该专家的 tokens 子集 $X_k$：
	$$
	X_k = \text{scatter\_tokens}[s_k : s_k + c_k, :]
	$$
	其中：
	- $s_k = \text{experts\_tokens\_start}[k]$：第 $k$ 个专家的 tokens 在 scatter_tokens 中的起始索引。
	- $c_k = \text{experts\_tokens\_count}[k]$：第 $k$ 个专家包含的 tokens 数量。
	- 提取后，$X_k$ 的形状为 $[c_k, D]$，其中 $D = \text{hidden\_size}$ 是隐藏层维度。
	对提取出的专家 $k$ 的 tokens 子集 $X_k$ 应用平滑缩放因子 $S_k$：
	$$
	X_k' = X_k \odot S_k
	$$
	其中：
	- $S_k = \text{smooth\_scale}[k, :] \in \mathbb{R}^D$：第 $k$ 个专家的平滑缩放因子，其形状为 $[D]$。
	- $\odot$：表示逐元素（element-wise）乘法。由于 $S_k$ 的形状为 $[D]$，它会被广播（broadcast）到与 $X_k$ 相同的形状 $[c_k, D]$ 后再进行乘法运算。
	- 运算后，$X_k'$ 的形状仍为 $[c_k, D]$。
	对于 $X_k'$ 中的第 $i$ 个 token（$i \in \{0, 1, ..., c_k-1\}$），即向量 $x'_i \in \mathbb{R}^D$，计算其动态缩放因子 $\text{scale}_i$：
	$$
	\text{scale}_i = \max_{j=1..D} |x'_{i,j}|
	$$
	其中 $x'_{i, j}$ 是向量 $x'_i$ 的第 $j$ 个元素。$\text{scale}_i$ 是该 token 在所有维度上绝对值的最大值。
	使用计算出的缩放因子 $\text{scale}_i$ 对 token $x'_i$ 进行量化，并将结果写入输出张量 y 的对应位置。同时，记录该 token 的缩放因子。
	量化输出：
	$$
	y_{s_k + i, :} = \frac{x'_{i, :}}{\text{scale}_i}
	$$
	这个操作将 $x'_i$ 的每个元素值归一化到 $[-1, 1]$ 的范围内（假设原始数据为浮点型）。结果写入 y 中从索引 $s_k + i$ 开始的行。
	记录缩放因子：
	$$
	\text{per\_tokens\_scale}[s_k + i] = \text{scale}_i
	$$
	将计算出的 $\text{scale}_i$ 存入 per_tokens_scale 张量的对应位置，以便后续可能的反量化操作。
	算子对所有专家（从 $k=0$ 到 $k=K-1$）重复上述步骤 1 到步骤 4。最终，y 张量将包含所有量化后的 tokens，per_tokens_scale 张量将包含每个 token 对应的缩放因子。
	整个过程可以概括为：
	$$
	(\text{y}, \text{per\_tokens\_scale}) = \text{moe\_swiglu\_dynamic\_quantize}(\text{scatter\_tokens}, \text{smooth\_scale}, \text{experts\_tokens\_start}, \text{experts\_tokens\_count}, K)
	$$
	其中：
	- $\text{scatter\_tokens} \in \mathbb{R}^{N \times D}$：输入，所有专家的 tokens 分散存储，$N$ 为总 token 数。
	- $\text{y} \in \mathbb{R}^{N \times D}$：输出，量化后的 tokens，与 scatter_tokens 形状相同。
	- $\text{per\_tokens\_scale} \in \mathbb{R}^N$：输出，每个 token 对应的动态缩放因子。
### 参数说明
- **topk_weights** (torch.Tensor, 出参): 输出的 Top-K 专家权重张量，形状为[num_tokens, topk]，存储每个 token 对应的 Top-K 专家的归一化权重
- **topk_indices** (torch.Tensor, 出参): 输出的 Top-K 专家索引张量，形状为[num_tokens, topk]，存储每个 token 对应的 Top-K 专家在所有专家中的索引
- **gating_output** (torch.Tensor, 入参): gating 网络的原始输出张量，形状为[num_tokens, num_experts]，存储每个 token 对应所有专家的原始评分
- **pre_softmax** (bool, 入参): 布尔标记，若为 True 表示 gating_output 是未经过 Softmax 的 logits，需先执行 Softmax 计算；若为 False 则认为 gating_output 已完成 Softmax 归一化
### 返回值
无返回值，Top-K 专家的权重与索引结果直接写入出参张量 topk_weights、topk_indices 中
### 约束与调用
- 所有输入/输出张量必须位于 CUDA 设备上
- 输入/输出张量需满足连续存储（contiguous）约束
- gating_output 需为 2 维张量，维度为[num_tokens, num_experts], num_tokens 为 token 总数，num_experts 为专家总数
- topk_weights 与 topk_indices 需为 2 维张量，维度为[num_tokens, topk], topk 为需筛选的专家数量
- 支持的数据类型：float16、bfloat16、float32, 对应代码中 scalar_t 模板支持的浮点类型
- pre_softmax 需为合法布尔值，用于区分 gating_output 是否已执行 Softmax
### 调用示例
```python
import torch
import mcoplib.op as op
# 定义设备
device = "cuda"
# 定义参数
num_tokens = 1000  
num_experts = 8   
topk = 2           
# 创建输入张量
gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
# 创建输出张量（预先分配内存）
topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=device)
topk_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
# 是否需要对 gating_output 进行 softmax
pre_softmax = True
# 调用 moe_softmax_topk 算子
op.moe_softmax_topk(
    topk_weights,
    topk_indices,
    gating_output,
    pre_softmax
)
# 输出结果信息
print("moe_softmax_topk computation completed.")
print(f"Number of tokens: {num_tokens}")
print(f"Number of experts: {num_experts}")
print(f"Top-k: {topk}")
print(f"pre_softmax: {pre_softmax}")
print(f"Shape of topk_weights: {topk_weights.shape}")
print(f"Shape of topk_indices: {topk_indices.shape}")
# 打印部分结果示例
print("\nSample topk_weights (first 2 tokens):")
print(topk_weights[:2])
print("\nSample topk_indices (first 2 tokens):")
print(topk_indices[:2])
```
## 12. rms_norm_dynamic_per_token_quant
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def rms_norm_dynamic_per_token_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    smooth_scale: torch.Tensor,
    scales: torch.Tensor,
    var_epsilon: float,
    after_res: torch.Tensor,
    after_norm: torch.Tensor,
    residual: Optional[torch.Tensor] = None
) -> None
```
### 功能描述
rms_norm_dynamic_per_token_quant 算子是融合逐 Token 动态量化的 RMS 归一化算子，支持可选残差连接. 该算子先对输入张量做残差融合（若有残差），再通过 RMS 归一化、权重缩放实现特征标准化，最后通过逐 Token 动态量化生成 int8 输出张量, 同时输出残差融合中间结果、归一化中间结果和逐 Token 量化缩放因子，减少内存交互开销，提升大模型推理效率
- 计算公式：
	残差连接（可选）, 若输入 residual 不为 None，计算残差融合结果, 否则直接沿用输入张量：  
	$$
	\text{after\_res}_{b,s,h} = 
	\begin{cases} 
	\text{input}_{b,s,h} + \text{residual}_{b,s,h}, & \text{residual} \neq \text{None} \\
	\text{input}_{b,s,h}, & \text{residual} = \text{None}
	\end{cases}
	$$
	其中：$\text{after\_res} \in \mathbb{R}^{B \times S \times H}$，$B$为 batch size，$S$为序列长度，$H$为单个 Token 的特征维度。
	逐 Token 计算平方和, 对 after_res 的每个 Token（即每个$(b, s)$位置），计算特征维度上的平方和：  
	$$
	\text{sum\_sq}_{b,s} = \sum_{h=0}^{H-1} \text{after\_res}_{b,s,h}^2
	$$
	其中：$\text{sum\_sq} \in \mathbb{R}^{B \times S}$。
	逐 Token 计算平方均值, 将平方和除以特征维度$H$，得到每个 Token 的平方均值：  
	$$
	\text{mean\_sq}_{b,s} = \frac{\text{sum\_sq}_{b,s}}{H}
	$$
	其中：$\text{mean\_sq} \in \mathbb{R}^{B \times S}$。
	计算逐 Token RMS 值, 为避免除零，给平方均值加上数值稳定小量 var_epsilon 后开根号，得到每个 Token 的 RMS（Root Mean Square）值：  
	$$
	\text{rms}_{b,s} = \sqrt{\text{mean\_sq}_{b,s} + \text{var\_epsilon}}
	$$
	其中：$\text{rms} \in \mathbb{R}^{B \times S}$，$\text{var\_epsilon} > 0$（通常取$10^{-6}$）。
	逐 Token 量化缩放因子, 计算每个 Token 的缩放因子（用于后续量化），即 RMS 值的倒数：  
	$$
	\text{scales}_{b,s} = \frac{1}{\text{rms}_{b,s}}
	$$
	其中：$\text{scales} \in \mathbb{R}^{B \times S}$（需保留该张量用于后续反量化）。
	RMS 归一化, 将 after_res 除以逐 Token 的 RMS 值（通过广播匹配维度），实现逐 Token 的 RMS 归一化：  
	$$
	\text{normed}_{b,s,h} = \frac{\text{after\_res}_{b,s,h}}{\text{rms}_{b,s}}
	$$
	其中：$\text{normed} \in \mathbb{R}^{B \times S \times H}$（$\text{rms}$通过`unsqueeze (-1)`扩展为$B \times S \times 1$，与$\text{after\_res}$广播）。
	权重与平滑缩放, 用 weight（特征维度权重）和 smooth_scale（特征维度平滑因子）对归一化结果做缩放，得到最终归一化中间结果：  
	$$
	\text{after\_norm}_{b,s,h} = \text{normed}_{b,s,h} \cdot \text{weight}_h \cdot \text{smooth\_scale}_h
	$$
	其中：$\text{weight} \in \mathbb{R}^H$，$\text{smooth\_scale} \in \mathbb{R}^H$（均通过广播匹配$B \times S \times H$维度）。
	逐 Token 动态量化, 将 after_norm 按逐 Token 的 scales 缩放到 int8 范围（$-128 \sim 127$）: 
	$$
	\text{out}_{b,s,h} = \text{clamp}\left( \text{round}\left( \text{after\_norm}_{b,s,h} \cdot \text{scales}_{b,s} \right), -128, 127 \right)
	$$
	其中：$\text{out} \in \mathbb{Z}^{B \times S \times H}$（int8 类型），round () 为四舍五入函数，clamp () 为区间限制函数。
### 参数说明
- **out** (torch.Tensor, 出参): 动态量化后的输出张量，数据类型为 int8，形状与 input 一致（B×S×H）
- **input** (torch.Tensor, 入参): 原始输入张量，形状为 B×S×H，支持数据类型为 float16/bfloat16
- **weight** (torch.Tensor, 入参): 特征维度的缩放权重张量，形状为 H，支持数据类型为 float16/bfloat16
- **smooth_scale** (torch.Tensor, 入参): 特征维度的平滑缩放因子张量，形状为 H，支持数据类型为 float16/bfloat16
- **scales** (torch.Tensor, 出参): 逐 Token 的量化缩放因子张量，形状为 B×S，数据类型为 float32（保证量化精度）
- **var_epsilon** (float, 入参): 数值稳定小量，用于避免 RMS 计算中出现除零或数值溢出（通常取 10^-6）
- **after_res** (torch.Tensor, 出参): 残差连接后的中间张量，形状与 input 一致（B×S×H），数据类型与 input 相同
- **after_norm** (torch.Tensor, 出参): 权重缩放后的归一化中间张量，形状与 input 一致（B×S×H），数据类型与 input 相同
- **residual** (Optional[torch.Tensor], 入参): 可选残差张量，形状需与 input 完全一致（B×S×H），支持数据类型为 float16/bfloat16；若为 None，则不进行残差连接
### 返回值
该算子无返回值，所有输出（量化结果、中间结果、缩放因子）均通过指定的出参张量（out、scales、after_res、after_norm）返回。
### 约束与调用
- 所有输入/出参张量必须位于 CUDA 设备上
- input、after_res、after_norm、out 的形状必须完全一致
- weight、smooth_scale 的形状必须与 input 的最后一维匹配
- scales 的形状必须与 input 的前两维匹配；
- 若 residual 不为 None，其形状必须与 input 完全一致。
- input、weight、smooth_scale、residual、after_res、after_norm 支持 float16/bfloat16，out 必须为 int8 类型，scales 必须为 float32 类型
- var_epsilon`必须为正数，否则会导致 RMS 计算数值不稳定或除零错误。
### 调用示例
```python
import torch
import mcoplib.op as op
# 定义运行设备
device = "cuda"
#定义算子所需的参数
B = 4
S = 128
H = 4096 
#创建输入张量
input_tensor = torch.randn(B, S, H, dtype=torch.float16, device=device)
weight = torch.randn(H, dtype=torch.float16, device=device)
smooth_scale = torch.randn(H, dtype=torch.float16, device=device)
residual = torch.randn(B, S, H, dtype=torch.float16, device=device)
#创建输出张量
out = torch.empty(B, S, H, dtype=torch.int8, device=device)
scales = torch.empty(B, S, dtype=torch.float32, device=device)
after_res = torch.empty(B, S, H, dtype=torch.float16, device=device)
after_norm = torch.empty(B, S, H, dtype=torch.float16, device=device)
var_epsilon = 1e-6 
op.rms_norm_dynamic_per_token_quant(
    out,
    input_tensor,
    weight,
    smooth_scale,
    scales,
    var_epsilon,
    after_res,
    after_norm,
    residual 
)
print("rms_norm_dynamic_per_token_quant computation completed.")
print(f"Output tensor 'out' shape: {out.shape}, dtype: {out.dtype}")
print(f"Scales tensor shape: {scales.shape}, dtype: {scales.dtype}")
print(f"After-residual tensor shape: {after_res.shape}")
print(f"After-normalization tensor shape: {after_norm.shape}")
```
## 13. head_rms_norm
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def head_rms_norm(
    out: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    var_epsilon: float,
    head_offset: int,
    head_norm: int
) -> None
```
### 功能描述
head_rms_norm 算子是针对多头注意力机制优化的 RMS（Root Mean Square）归一化算子，该算子将输入的多头隐藏状态按 Head 维度拆分，对每个 Head 单独执行 RMS 归一化和权重缩放，最后将结果拼接输出，避免跨 Head 的归一化干扰，提升注意力机制的计算效率和模型性
- 计算公式：
	多头隐藏状态拆分，设输入 hidden_states 的形状为 $B \times S \times N \times D$（$B$=batch size，$S$=序列长度，$N$=总 Head 数，$D$=单个 Head 的特征维度），或展平形状 $B \times S \times (N \times D)$。算子首先按 Head 维度拆分，得到单个 Head 的隐藏状态张量：  
	$$
	\text{head\_hidden}_{b,s,:} = \text{hidden\_states}_{b,s,\text{head\_idx} \times D : (\text{head\_idx}+1) \times D}
	$$
	其中：$\text{head\_idx}$ 为当前处理的 Head 索引（范围 $0 \sim N-1$），$\text{head\_hidden} \in \mathbb{R}^{B \times S \times D}$。
	对每个 Head 的隐藏状态，在特征维度（$D$ 维）上计算平方和：  
	$$
	\text{sum\_sq}_{b,s} = \sum_{d=0}^{D-1} \text{head\_hidden}_{b,s,d}^2
	$$
	其中：$\text{sum\_sq} \in \mathbb{R}^{B \times S}$。
	将平方和除以特征维度 $D$，得到每个 Token（$b, s$ 位置）在该 Head 下的平方均值：  
	$$
	\text{mean\_sq}_{b,s} = \frac{\text{sum\_sq}_{b,s}}{D}
	$$
	其中：$\text{mean\_sq} \in \mathbb{R}^{B \times S}$。
	为避免除零，给平方均值加上数值稳定小量 var_epsilon 后开根号，得到每个 Token 的 RMS 值：  
	$$
	\text{rms}_{b,s} = \sqrt{\text{mean\_sq}_{b,s} + \text{var\_epsilon}}
	$$
	其中：$\text{rms} \in \mathbb{R}^{B \times S}$，$\text{var\_epsilon} > 0$（通常取 $10^{-6}$）。
	将该 Head 的隐藏状态除以其 RMS 值，通过广播匹配维度，实现归一化：  
	$$
	\text{normed}_{b,s,d} = \frac{\text{head\_hidden}_{b,s,d}}{\text{rms}_{b,s}}
	$$
	其中：$\text{normed} \in \mathbb{R}^{B \times S \times D}$（$\text{rms}$ 扩展为 $B \times S \times 1$ 后与 $\text{head\_hidden}$ 广播）。
	获取当前 Head 对应的权重，head_offset 定位，对归一化结果做缩放：  
	$$
	\text{scaled}_{b,s,d} = \text{normed}_{b,s,d} \times \text{weight}_{\text{head\_idx},d}
	$$
	其中：$\text{weight} \in \mathbb{R}^{N \times D}$（$N$ 为总 Head 数），$\text{scaled} \in \mathbb{R}^{B \times S \times D}$。
	将所有 Head 的缩放后结果按原 Head 顺序拼接，得到最终输出张量：  
	$$
	\text{out}_{b,s,:} = \text{concat}\left( \text{scaled}_{\text{head\_idx}=0}, \text{scaled}_{\text{head\_idx}=1}, \dots, \text{scaled}_{\text{head\_idx}=N-1} \right)
	$$
	其中：$\text{out} \in \mathbb{R}^{B \times S \times (N \times D)}$，与输入 hidden_states 形状一致）
### 参数说明
- **out** (torch.Tensor, 出参): 多头归一化后的输出张量，形状与 hidden_states 一致，支持数据类型为 float16/bfloat16
- **hidden_states** (torch.Tensor, 入参): 多头注意力的隐藏状态张量，支持数据类型为 float16/bfloat16
- **weight** (torch.Tensor, 入参): 分 Head 的权重张量，支持数据类型为 float16/bfloat16
- **var_epsilon** (float, 入参): 数值稳定小量，用于避免 RMS 计算中除零或数值溢出
- **head_offset** (int, 入参): 当前处理的 Head 在 weight 中的起始偏移索引，用于定位目标 Head 的权重
- **head_norm** (int, 入参): 多头注意力的总 Head 数，用于拆分和拼接多头张量
### 返回值
该算子无返回值，返回类型为 None，归一化后的结果通过出参张量 out 返回。
### 约束与调用
- 所有输入/出参张量必须位于 CUDA 设备上
- out 与 hidden_states 的形状必须完全一致，weight`的形状必须为 N×D（N=head_norm，D=单个 Head 的特征维度）
- 若`hidden_states 为展平形状 B×S×(N×D)，则 N×D 必须等于其最后一维的大小；若为未展平形状 B×S×N×D，则其第三维必须等于 N（head_norm），第四维必须等于 D
- hidden_states、weight、out 支持 float16/bfloat16，不支持 float32
- head_offset 必须为非负整数，且满足 0≤head_offset<head_norm
- var_epsilon 必须为正数，否则会导致 RMS 计算数值不稳定
### 调用示例
```python
import torch
import mcoplib.op as op
# 定义运行设备
device = "cuda"
# 定义算子所需的参数
B = 4  
S = 128  
N = 32  
D = 128  
# 创建输入张量
hidden_states = torch.randn(B, S, N * D, dtype=torch.float16, device=device)
weight = torch.randn(N, D, dtype=torch.float16, device=device)
# 创建输出张量
out = torch.empty(B, S, N * D, dtype=torch.float16, device=device)
# 其他参数
var_epsilon = 1e-6
head_offset = 0  
head_norm = N   
# 调用 head_rms_norm 算子
op.head_rms_norm(
    out,
    hidden_states,
    weight,
    var_epsilon,
    head_offset,
    head_norm
)
print("head_rms_norm computation completed.")
print(f"Output tensor 'out' shape: {out.shape}, dtype: {out.dtype}")
```
## 14. rms_norm
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    var_epsilon: float,
    after_res: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    rms_div: bool
) -> None
```
### 功能描述
rms_norm 算子是基础 RMS（Root Mean Square）归一化算子，支持可选残差连接与两种归一化计算方式，该算子先通过残差连接融合输入与残差张量（若有），再对融合结果计算 RMS 归一化，最后通过权重缩放生成输出，同时支持将残差融合中间结果存入 after_res（可选），并通过 rms_div 参数灵活切换归一化计算逻辑（除法/乘法），适配不同硬件效率需求。
- 计算公式：
	残差连接（可选），若输入 residual 不为 None，计算残差融合结果；否则直接沿用输入张量 input，若 after_res 为非 None 张量，则将融合结果存入 after_res：  
	$$
	\text{after\_res}_{b,s,h} = 
	\begin{cases} 
	\text{input}_{b,s,h} + \text{residual}_{b,s,h}, & \text{residual} \neq \text{None} \\
	\text{input}_{b,s,h}, & \text{residual} = \text{None}
	\end{cases}
	$$
	其中：$\text{after\_res} \in \mathbb{R}^{B \times S \times H}$（$B$=batch size，$S$=序列长度，$H$=特征维度），若`after_res`为 None 则仅在内部使用该融合结果。
	对残差融合后的张量（记为$X$，即$\text{after\_res}$或内部融合结果），在**特征维度（$H$维）** 计算每个$(b, s)$位置的平方和：  
	$$
	\text{sum\_sq}_{b,s} = \sum_{h=0}^{H-1} X_{b,s,h}^2
	$$
	其中：$\text{sum\_sq} \in \mathbb{R}^{B \times S}$。
	将平方和除以特征维度$H$，得到每个$(b, s)$位置的平方均值：  
	$$
	\text{mean\_sq}_{b,s} = \frac{\text{sum\_sq}_{b,s}}{H}
	$$
	其中：$\text{mean\_sq} \in \mathbb{R}^{B \times S}$。
	为避免除零，给平方均值加上数值稳定小量 var_epsilon 后开根号，得到每个$(b, s)$位置的 RMS 值：  
	$$
	\text{rms}_{b,s} = \sqrt{\text{mean\_sq}_{b,s} + \text{var\_epsilon}}
	$$
	其中：$\text{rms} \in \mathbb{R}^{B \times S}$，$\text{var\_epsilon} > 0$（通常取$10^{-6}$）。
	根据 rms_div 参数选择归一化计算方式（核心差异为“除法”或“乘法”，乘法更适配硬件指令优化）：  
	- 当 $\text{rms\_div} = \text{True}$（除法模式）：  
	  $$
	  \text{normed}_{b,s,h} = \frac{X_{b,s,h}}{\text{rms}_{b,s}}
	  $$
	- 当 $\text{rms\_div} = \text{False}$（乘法模式，硬件友好）：  
	  $$
	  \text{normed}_{b,s,h} = X_{b,s,h} \cdot \frac{1}{\text{rms}_{b,s}}
	  $$
	  其中：$\text{normed} \in \mathbb{R}^{B \times S \times H}$（$\text{rms}$通过`unsqueeze (-1)`扩展为$B \times S \times 1$，与$X$广播匹配）。
	  将归一化结果乘以特征维度的权重 weight，最终结果存入 out：  
	$$
	\text{out}_{b,s,h} = \text{normed}_{b,s,h} \cdot \text{weight}_h
	$$
	其中：$\text{weight} \in \mathbb{R}^H$（通过广播匹配$B \times S \times H$维度）。
### 参数说明
- **out** (torch.Tensor, 出参): 归一化后的输出张量，形状与 input 完全一致（B×S×H），支持数据类型为 float16/bfloat16
- **input** (torch.Tensor, 入参): 原始输入张量，形状为 B×S×H，支持数据类型为 float16/bfloat16
- **weight** (torch.Tensor, 入参): 特征维度的缩放权重张量，形状为 H，支持数据类型为 float16/bfloat16
- **var_epsilon** (float, 入参): 数值稳定小量，用于避免 RMS 计算中除零或数值溢出（通常取 10^-6）
- **after_res** (Optional[torch.Tensor], 出参): 可选残差融合中间结果张量，形状与 input 一致（B×S×H），数据类型与 input 相同；若为 None，则不保存该中间结果
- **residual** (Optional[torch.Tensor], 入参): 可选残差张量，形状需与 input 完全一致（B×S×H），支持数据类型为 float16/bfloat16；若为 None，则不进行残差连接
- **rms_div** (bool, 入参): 归一化计算模式开关：True 表示“除以 RMS 值”，False 表示“乘以 RMS 倒数”，后者更适配硬件指令，效率更高
### 返回值
该算子无返回值（返回类型为 None），核心归一化结果通过出参 out 返回，如果需要残差融合中间结果，通过 after_res 返回
### 约束与调用
- 所有输入/出参张量必须位于 CUDA 设备上
- out、input、after_res、residual 的形状必须完全一致（B×S×H），weight 的形状必须与 input 的最后一维（特征维度 H）匹配，input、weight、out、after_res（若有）、residual（若有）仅支持 float16/bfloat16，不支持 float32
- var_epsilon 必须为正数，否则会导致 RMS 计算数值不稳定，rms_div 必须为布尔值
### 调用示例
```python
import torch
import mcoplib.op as op
# 定义运行设备
device = "cuda"
# 1. 定义算子所需的参数
B = 4 
S = 128  
H = 4096  
# 创建输入张量
input_tensor = torch.randn(B, S, H, dtype=torch.float16, device=device)
weight = torch.randn(H, dtype=torch.float16, device=device)
residual = torch.randn(B, S, H, dtype=torch.float16, device=device)
# 创建输出张量和可选的中间结果张量
out = torch.empty(B, S, H, dtype=torch.float16, device=device)
after_res = torch.empty(B, S, H, dtype=torch.float16, device=device)
# 其他参数
var_epsilon = 1e-6
rms_div = False 
# 调用 rms_norm 算子
op.rms_norm(
    out,
    input_tensor,
    weight,
    var_epsilon,
    after_res,   
    residual,      
    rms_div
)
print("rms_norm computation completed.")
print(f"Output tensor 'out' shape: {out.shape}, dtype: {out.dtype}")
print(f"After-residual tensor shape: {after_res.shape}")
```
## 15. all_reduce_max
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def all_reduce_max(
    input: torch.Tensor,
    output: torch.Tensor
) -> None
```
### 功能描述
all_reduce_max 算子是基于 CUDA 实现的执行最大规约操作的算子，针对`bfloat16`类型张量，为每个 token 执行规约操作，支持向量化访存加速
- 计算公式：
    $$
    \text{all\_reduce\_max}(\mathbf{X}) = \left[ \max_{j=1}^D X_{i,j} \right]_{i=1}^N
    $$
  其中：
  - $\mathbf{X} \in \mathbb{R}^{N \times D}$ 是输入张量（形状为 [num_tokens, hidden_size]）
  - $\text{all\_reduce\_max}(\mathbf{X}) \in \mathbb{R}^N$ 是输出张量（形状为 [num_tokens]）
  - $V \in \mathbb{R}^{B \times H_k \times D \times S}$ 是分页值缓存
  - $N$ 是 token 数量，对应代码中的 $\text{num\_tokens}$
  - $D$ 是隐藏维度，对应代码中的 $\text{hidden\_size}$
  - $X_{i,j}$ 表示输入张量 $\mathbf{X}$ 中第 $i$ 个 token 对应的第 $j$ 个隐藏维度元素
    **主要功能包括**：
- 基于 CUDA 实现张量聚合计算，支持求和与最大值两种核心操作
- 适配分布式计算场景自动选择向量版或普通版 CUDA 核函数
- 通过线程协同，完成跨设备/进程的张量元素聚合，保障数据一致
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，需位于 CUDA 设备上，形状为[num_tokens, hidden_size]，数据类型仅支持 bfloat16，存储待归约的 bfloat16 数据
- **output** (torch.Tensor, 出参): 输出张量，需位于 CUDA 设备上，形状为[num_tokens]，用于存储每个 token 对应的 hidden_size 维度的求和结果
### 返回值
无返回值，结果直接写入输出张量 output 中
### 约束与调用
- 所有输入/输出张量必须位于 CUDA 设备上
- 仅兼容 bfloat16 数据类型
- 输入张量为连续存储的 2D 张量，形状固定为[num_tokens, hidden_size]
- 根据 hidden_size 是否对齐自动选择是否使用向量化加速
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义参数
num_tokens = 4096
hidden_size = 4096
dtype = torch.bfloat16
error_threshold = 1e-3  # 误差阈值
# 创建输入/输出张量
input_tensor = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
output_tensor = torch.zeros(num_tokens, dtype=dtype, device=device)
op.all_reduce_max(input_tensor, output_tensor)
# 简单精度检查：用PyTorch原生函数做基准对比
standard_output = torch.max(input_tensor, dim=1).values 
max_error = torch.max(torch.abs(output_tensor - standard_output)).item()  
is_correct = max_error < error_threshold 
print("all_reduce_max computation completed")
print(f"Input shape: {output_tensor.shape}, Output shape: {output_tensor.shape}")
print(f"\nPrecision check: {'PASSED' if is_correct else 'FAILED'}")
print(f"Maximum absolute error: {max_error:.6f} (threshold: {error_threshold})")
print(f"\nFirst 5 results comparison:")
print(f"Operator output: {output_tensor[:5].cpu()}")
print(f"Standard output: {standard_output[:5].cpu()}")
```
## 16. all_reduce_sum
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def all_reduce_sum(
    input: torch.Tensor,
    output: torch.Tensor
) -> None
```
### 功能描述
all_reduce_sum 算子是基于 CUDA 实现的执行求和规约操作的算子，针对`bfloat16`类型张量，为每个 token 执行 hidden_size 维度的求和规约，支持向量化访存加速
- 计算公式：
    $$
    \text{all\_reduce\_sum}(\mathbf{X}) = \left[ \sum_{j=1}^D X_{i,j} \right]_{i=1}^N
    $$
  其中：
  - $\mathbf{X} \in \mathbb{R}^{N \times D}$ 是输入张量（形状为 [num_tokens, hidden_size]）
  - $\text{all\_reduce\_sum}(\mathbf{X}) \in \mathbb{R}^N$ 是输出张量（形状为 [num_tokens]）
  - $N$ 是 token 数量，对应代码中的 $\text{num\_tokens}$
  - $D$ 是隐藏维度，对应代码中的 $\text{hidden\_size}$
  - $X_{i,j}$ 表示输入张量 $\mathbf{X}$ 中第 $i$ 个 token 对应的第 $j$ 个隐藏维度元素
    **主要功能包括：**
- 基于 CUDA 实现张量聚合计算，支持求和的核心规约操作
- 适配分布式计算场景自动选择向量版或普通版 CUDA 核函数
- 通过线程协同完成计算，结合向量化访存优化提升执行效率
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，需位于 CUDA 设备上，形状为[num_tokens, hidden_size]，数据类型仅支持 bfloat16，存储待归约的 bfloat16 数据
- **output** (torch.Tensor, 出参): 输出张量，需位于 CUDA 设备上，形状为[num_tokens]，用于存储每个 token 对应的 hidden_size 维度的求和结果
### 返回值
无返回值，结果直接写入输出张量 output 中
### 约束与调用
- 所有输入/输出张量必须位于 CUDA 设备上
- 仅兼容 bfloat16 数据类型
- 输入张量为连续存储的 2D 张量，形状固定为[num_tokens, hidden_size]
- 根据 hidden_size 是否对齐自动选择是否使用向量化加速
### 调用示例
```python
import torch
import mcoplib.op as op
# 设置设备
device = "cuda"
# 定义参数
num_tokens = 4096
hidden_size = 4096
dtype = torch.bfloat16
error_threshold = 1e-3  
# 创建输入/输出张量
input_tensor = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
output_tensor = torch.zeros(num_tokens, dtype=dtype, device=device)  # 输出是num_tokens长度的向量
# 调用目标算子
op.all_reduce_sum(input_tensor, output_tensor)
# 简单精度检查：用PyTorch原生sum做基准对比
standard_output = torch.sum(input_tensor, dim=1) 
max_error = torch.max(torch.abs(output_tensor - standard_output)).item()  
is_correct = max_error < error_threshold 
# 打印结果
print("all_reduce_sum computation completed")
print(f"Input shape: {input_tensor.shape}, Output shape: {output_tensor.shape}")
print(f"\nPrecision check: {'PASSED' if is_correct else 'FAILED'}")
print(f"Maximum absolute error: {max_error:.6f} (threshold: {error_threshold})")
print(f"\nFirst 5 results comparison:")
print(f"Operator output: {output_tensor[:5].cpu()}")
print(f"Standard output: {standard_output[:5].cpu()}")
```
## 17. moe_gather
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def moe_gather(
    scatter_tokens: torch.Tensor,
    scatter_tokens_offset: torch.Tensor,
    convergent_tokens: torch.Tensor
) -> None
```
### 功能描述
moe_gather 算子实现混合专家（MoE）架构中的令牌聚合操作，将分散存储的专家输出令牌（scatter_tokens），按照偏移信息（scatter_tokens_offset）汇聚到目标张量（convergent_tokens）中，用于完成 MoE 模型中不同专家输出的令牌整合流程
- 计算公式
  $$
  \text{convergent\_tokens}[\text{offset}_i, :] = \text{scatter\_tokens}[i, :] \quad (i = 0, 1, ..., \text{num\_tokens} - 1)
  $$
  其中：
  - $i$：分散令牌的索引，范围为 $[0, \text{num\_tokens} - 1]$
  - $\text{num\_tokens}$：分散令牌的总数，即 $\text{scatter\_tokens}$ 的第一维长度（$\text{scatter\_tokens.size}(0)$）
  - $\text{hidden\_size}$：令牌的隐藏层维度，即 $\text{scatter\_tokens}$ 的第二维长度（$\text{scatter\_tokens.size}(1)$）
  - $\text{offset}_i$：第 $i$ 个分散令牌对应的聚合位置偏移，即 $\text{scatter\_tokens\_offset}[i]$，需满足 $0 \leq \text{offset}_i < \text{convergent\_tokens.size}(0)$（聚合后令牌数）
  - $\text{scatter\_tokens}[i, :]$：第 $i$ 个分散令牌的完整向量（维度为 $[\text{hidden\_size}]$）
  - $\text{convergent\_tokens}[\text{offset}_i, :]$：聚合后张量中对应偏移位置的令牌向量（维度为 $[\text{hidden\_size}]$）
  算子核心逻辑为按偏移索引的逐行赋值，实现分散令牌到聚合张量的定向汇聚，且保持令牌内部的维度结构不变
### 参数说明
- **scatter_tokens** (torch.Tensor, 入参): 分散的专家输出令牌张量，形状为`[num_tokens, hidden_size]`，存储待聚合的令牌数据
- **scatter_tokens_offset** (torch.Tensor, 入参): 令牌聚合偏移张量，形状为`[num_tokens]`，存储每个分散令牌对应的目标聚合位置偏移信息
- **convergent_tokens** (torch.Tensor, 出参): 聚合后的令牌张量，形状通常为`[聚合后令牌数, hidden_size]`，存储最终的令牌聚合结果
### 返回值
无返回值，聚合结果直接写入出参 convergent_tokens 中
### 约束与调用
- 所有输入、输出张量必须位于 CUDA 设备上
- 支持的数据类型：float16、bfloat16
- scatter_tokens 的元素总数需为 hidden_size 的整数倍，即 num_tokens = scatter_tokens.numel () / hidden_size
- scatter_tokens_offset 的长度需与 scatter_tokens 的 num_tokens 维度一致
- 输入张量需满足连续存储的内存布局要求
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义参数
num_tokens = 128 
hidden_size = 512  
convergent_num_tokens = 64  
# 创建输入张量
scatter_tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
scatter_tokens_offset = torch.randint(0, convergent_num_tokens, (num_tokens,),dtype=torch.int32,device=device)
# 创建输出张量
convergent_tokens = torch.empty(convergent_num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
# 调用moe_gather算子
op.moe_gather(
    scatter_tokens,
    scatter_tokens_offset,
    convergent_tokens
)
print("MoE gather computation completed")
print(f"Scatter tokens shape: {scatter_tokens.shape}")
print(f"Scatter tokens offset shape: {scatter_tokens_offset.shape}")
print(f"Convergent tokens shape: {convergent_tokens.shape}")
```
## 18. rotary_embeding
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def rotary_embedding(
    packed_qkv: torch.Tensor,
    q_len: torch.Tensor,
    accum_q_lens: torch.Tensor,
    cache_lens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
    q_head_num: int,
    kv_head_num: int,
    rope_offset: int = 0
) -> None
```
### 功能描述
rotary_embedding 算子实现了旋转位置编码（Rotary Position Embedding）操作。该算子将位置信息通过预计算的余弦（cos）和正弦（sin）张量编码到打包的查询与键向量（packed_qkv）中，为 Transformer 模型中的注意力机制提供位置感知能力，常用于处理序列数据的位置依赖关系。
- 计算公式：
旋转位置编码（RoPE）对向量的偶数/奇数维度分别执行旋转操作，对于维度为$d$（head_dim）的向量$x$（查询/键向量）、位置为$m$、旋转维度为$rot\_dim$（满足$rot\_dim \leq d$且通常$rot\_dim = d/2$），旋转后的向量$x_{\text{rot}}(m)$计算公式为：
$$
\begin{cases}
x_{\text{rot}, 2i}(m) = x_{2i} \cdot \cos(\theta_i \cdot m) - x_{2i+1} \cdot \sin(\theta_i \cdot m) \\
x_{\text{rot}, 2i+1}(m) = x_{2i} \cdot \sin(\theta_i \cdot m) + x_{2i+1} \cdot \cos(\theta_i \cdot m)
\end{cases}
$$
其中旋转基$\theta_i$的定义为：
$$
\theta_i = \frac{1}{10000^{2i/d}}, \quad i \in [0, \frac{rot\_dim}{2})
$$
考虑偏移量和缓存约束，实际参与旋转编码的位置$m$由序列位置与旋转偏移量（rope_offset）叠加得到：
$$
m = \text{pos} + \text{rope\_offset}
$$
其中$\text{pos}$为序列原始位置索引，由$\text{accum\_q\_lens}$（累积查询长度）和$\text{cache\_lens}$（缓存长度）推导，且满足：
$$
\text{pos} \in [0, \text{cache\_lens} + q\_len)
$$
算子输入$\text{packed\_qkv}$包含查询（Q）和键（K）向量，拆分后分别应用 RoPE：
$$
\text{packed\_qkv} = [Q; K]
$$
其中$Q \in \mathbb{R}^{q\_len \times q\_head\_num \times d}$，$K \in \mathbb{R}^{q\_len \times kv\_head\_num \times d}$；编码后输出张量为：
$$
\text{output} = [Q_{\text{rot}}; K_{\text{rot}}]
$$
对于分组查询注意力（GQA）配置，键值头数与查询头数满足：
$$
kv\_head\_num = \frac{q\_head\_num}{G}
$$
其中$G$为组数（MQA 场景下$G = q\_head\_num$，即$kv\_head\_num = 1$）。
其中：
- $x_{2i}/x_{2i+1}$：原始向量的第$2i$/第$2i+1$维元素
- $x_{\text{rot}, 2i}/x_{\text{rot}, 2i+1}$：旋转后向量的第$2i$/第$2i+1$维元素
- $d$：头维度（head_dim），即单个注意力头的特征维度
- $rot\_dim$：旋转维度，满足$rot\_dim \leq d$
- $q\_len$：查询序列长度
- $q\_head\_num$：查询注意力头数
- $kv\_head\_num$：键/值注意力头数
- $\text{rope\_offset}$：旋转编码偏移量，用于增量推理等场景
- $\text{pos}$：序列原始位置索引，由$\text{accum\_q\_lens}$和$\text{cache\_lens}$计算得到
- $G$：GQA 分组数，为正整数
### 参数说明
- **packed_qkv** (at::Tensor, 入参): 打包的查询与键向量张量，形状为[num_tokens, total_head_num, head_dim]
- **q_len** (at::Tensor, 入参): 查询序列长度张量
- **accum_q_lens** (at::Tensor, 入参): 累积的查询长度张量，用于定位序列位置偏移
- **cache_lens** (at::Tensor, 入参): 缓存长度张量，用于确定有效序列范围
- **cos** (at::Tensor, 入参): 预计算的余弦值张量，用于旋转编码计算
- **sin** (at::Tensor, 入参): 预计算的正弦值张量，用于旋转编码计算
- **output** (at::Tensor, 出参): 输出张量，存储应用旋转编码后的结果
- **q_head_num** (int, 入参): 查询头的数量
- **kv_head_num** (int, 入参): 键值头的数量
- **rope_offset** (int, 入参): 旋转偏移量，用于调整位置编码的起始位置，默认值为 0
### 返回值
无返回值，计算结果直接写入 output 张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 旋转维度（通常为 head_dim 的一半）必须小于等于头维度（head_dim）
- cos 和 sin 张量必须有足够的大小以支持所有位置偏移量（包括 rope_offset）
- 序列位置加上 rope_offset 后必须在 cos 和 sin 缓存的有效范围内
- 支持的数据类型：float16, bfloat16, float32
### 调用示例
```python
import torch
import mcoplib.op as op 
device = "cuda"
num_seqs = 4
q_head_num = 32
kv_head_num = 8
head_size = 128
max_seq_len = 2048
rope_offset = 0
seq_lens_list = [128, 64, 256, 32]
num_tokens = sum(seq_lens_list)
total_head_num = q_head_num + 2 * kv_head_num 
#创建输入
packed_qkv = torch.randn(num_tokens, total_head_num, head_size, dtype=torch.bfloat16, device=device)
cos = torch.randn(max_seq_len, head_size, dtype=torch.float32, device=device)
sin = torch.randn(max_seq_len, head_size, dtype=torch.float32, device=device)
#创建输出
out = torch.empty_like(packed_qkv)
q_len = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
accum_q_lens = torch.tensor([0] + seq_lens_list, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
cache_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device)
# 调用rotary_embedding operator算子
op.rotary_embedding(
    packed_qkv,
    q_len,
    accum_q_lens,
    cache_lens,
    cos,
    sin,
    out,
    q_head_num,
    kv_head_num,
    rope_offset
)
print("\nRotary embedding computation completed successfully")
print(f"Final output shape: {out.shape}")
```
## 19. store_kv_cache_cuda_interface
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def store_kv_cache_cuda_interface(
    packed_qkv: torch.Tensor,
    q_lens: torch.Tensor,
    accum_q_lens: torch.Tensor,
    cache_lens: torch.Tensor,
    cache_slot_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    batch_size: int,
    q_head_num: int,
    kv_head_num: int
) -> None
```
### 功能描述
store_kv_cache_cuda_interface 算子实现了 CUDA 端 KV 缓存的存储操作：基于序列长度、缓存槽位等信息，将打包好的 packed_qkv 数据分发并存储到对应的 K 缓存与 V 缓存中，同时结合缩放张量完成数据处理。该算子常用于大模型推理的 KV 缓存管理，可减少内存交互、提升缓存写入效率。
- 计算公式
	$$
	\text{k\_cache}[b, h_k, \text{cache\_slot\_ids}[i], d] = \text{packed\_K}[i, h_q, d] \times \text{k\_scale}[h_k]
	$$
	$$
	\text{v\_cache}[b, h_k, \text{cache\_slot\_ids}[i], d] = \text{packed\_V}[i, h_q, d] \times \text{v\_scale}[h_k]
	$$
	packed_qkv 为打包的 Q/K/V 复合张量，首先拆分出参与缓存写入的 K、V 分量（Q 分量不写入 KV 缓存）：
	$$
	\text{packed\_K} = \text{packed\_qkv}[..., 0:d_k],\quad \text{packed\_V} = \text{packed\_qkv}[..., d_k:2d_k]
	$$
	其中单头特征维度 $d_k$ 满足：
	$$
	d_k = \frac{\text{packed\_qkv}.\text{size}(-1)}{3}
	$$
	（packed_qkv 最后一维维度为 $3d_k$，分别对应 Q、K、V 各占 $d_k$ 维度）
	查询头（$h_q$）与 KV 头（$h_k$）的映射关系（适配 GQA/MQA 配置）：
	$$
	h_k = h_q \times \frac{\text{kv\_head\_num}}{\text{q\_head\_num}}
	$$
	其中：
	- $h_q \in [0, \text{q\_head\_num}-1]$：查询头索引
	- $h_k \in [0, \text{kv\_head\_num}-1]$：KV 头索引
	MQA 是 GQA 的特例（$\text{kv\_head\_num}=1$），此时所有查询头映射到同一个 KV 头。
	对于第 $b$ 个 batch（$b \in [0, \text{batch\_size}-1]$），其对应的 token 索引范围由累积长度和查询长度确定：
	$$
	i \in [\text{accum\_q\_lens}[b], \text{accum\_q\_lens}[b] + \text{q\_lens}[b])
	$$
	所有 batch 的 token 总数满足：
	$$
	\sum_{b=0}^{\text{batch\_size}-1} \text{q\_lens}[b] = \text{packed\_K}.\text{size}(0) = \text{packed\_V}.\text{size}(0)
	$$
	cache_slot_ids 的元素值需满足缓存长度边界约束：
	$$
	0 \leq \text{cache\_slot\_ids}[i] < \text{k\_cache}.\text{size}(2)
	$$
	且单 batch 缓存写入后长度需匹配：
	$$
	\text{cache\_lens}[b] + \text{q\_lens}[b] = \max(\text{cache\_slot\_ids}[i \in \text{batch}_b]) + 1
	$$
	其中：
	- $b$：批维度索引，$b \in \mathbb{Z}, 0 \leq b < \text{batch\_size}$
	- $h_q$：查询头维度索引，$h_q \in \mathbb{Z}, 0 \leq h_q < \text{q\_head\_num}$
	- $h_k$：KV 头维度索引，$h_k \in \mathbb{Z}, 0 \leq h_k < \text{kv\_head\_num}$
	- $d$：单头特征维度索引，$d \in \mathbb{Z}, 0 \leq d < d_k$
	- $i$：packed_K/packed_V 的 token 维度索引，$i \in \mathbb{Z}, 0 \leq i < T$（$T$ 为总 token 数）
	- $d_k$：单头特征维度（键/值头维度），$d_k = \frac{\text{模型总特征维度}}{\text{q\_head\_num}}$
	- $\text{packed\_K} \in \mathbb{R}^{T \times \text{q\_head\_num} \times d_k}$：拆解后的键数据张量
	- $\text{packed\_V} \in \mathbb{R}^{T \times \text{q\_head\_num} \times d_k}$：拆解后的值数据张量
	- $\text{k\_cache} \in \mathbb{R}^{\text{batch\_size} \times \text{kv\_head\_num} \times L_{\text{max}} \times d_k}$：K 缓存张量（$L_{\text{max}}$ 为缓存最大长度）
	- $\text{v\_cache} \in \mathbb{R}^{\text{batch\_size} \times \text{kv\_head\_num} \times L_{\text{max}} \times d_k}$：V 缓存张量
	- $\text{cache\_slot\_ids} \in \mathbb{Z}^T$：缓存槽位索引张量（指定每个 token 写入缓存的位置）
	- $\text{k\_scale} \in \mathbb{R}^{\text{kv\_head\_num}}$：K 缓存数据缩放张量（支持广播至 $\mathbb{R}^{1 \times \text{kv\_head\_num} \times 1 \times 1}$）
	- $\text{v\_scale} \in \mathbb{R}^{\text{kv\_head\_num}}$：V 缓存数据缩放张量（支持广播至 $\mathbb{R}^{1 \times \text{kv\_head\_num} \times 1 \times 1}$）
### 参数说明
- **packed_qkv** (torch.Tensor, 入参): 打包后的     QKV 数据张量，是待写入缓存的源数据
- **q_lens** (torch.Tensor, 入参): 查询序列长度的张量，用于确定数据的序列维度范围
- **accum_q_lens** (torch.Tensor, 入参): 累积的 Q 序列长度张量，用于定位数据在 packed_qkv 中的偏移
- **cache_lens** (torch.Tensor, 入参): 缓存当前长度的张量，用于确定缓存的写入起始位置
- **cache_slot_ids** (torch.Tensor, 入参): 缓存槽位 ID 的张量，用于指定数据写入的缓存槽位
- **k_cache** (torch.Tensor, 入参/出参): K 缓存张量，计算后会被写入对应的 KV 数据
- **v_cache** (torch.Tensor, 入参/出参): V 缓存张量，计算后会被写入对应的 KV 数据
- **k_scale** (torch.Tensor, 入参): K 数据的缩放张量，用于对写入 K 缓存的数据做缩放处理
- **v_scale** (torch.Tensor, 入参): V 数据的缩放张量，用于对写入 V 缓存的数据做缩放处理
- **batch_size** (int, 入参): 批处理大小，对应数据的批维度规模
- **q_head_num** (int, 入参): 查询头的数量，对应模型中查询注意力头的维度配置
- **kv_head_num** (int, 入参): KV 头的数量，对应模型中 KV 注意力头的维度配置
### 返回值
无返回值，计算结果直接写入 k_cache 与 v_cache 张量中。
### 约束与调用
- 所有输入张量（packed_qkv、q_lens、accum_q_lens、cache_lens、cache_slot_ids、k_cache、v_cache、k_scale、v_scale）必须位于 CUDA 设备上
- 数值型张量 packed_qkv、k_cache、v_cache、k_scale、v_scale）支持 float16、bfloat16, 整型张量 q_lens、accum_q_lens、cache_lens、cache_slot_ids 支持 int32、int64
- batch_size、q_head_num、kv_head_num 必须为正整数，且 q_head_num 需为 kv_head_num 的整数倍（符合大模型 KV 注意力头映射逻辑）
- q_lens、accum_q_lens、cache_lens 的张量长度需与 batch_size 一致
- cache_slot_ids 的元素总数需匹配 packed_qkv 中待写入缓存的 token 总数，且其元素值需小于 k_cache/v_cache 缓存长度维度的最大值
- k_cache 与 v_cache 的维度需匹配 kv_head_num 配置，且二者形状完全一致（如 k_cache.shape = v_cache.shape = (batch_size, kv_head_num, cache_max_len, head_dim)，head_dim 为单头特征维度）
- packed_qkv 的元素总数需满足：packed_qkv.numel () = sum (q_lens) * q_head_num * head_dim（head_dim 为单头特征维度，与 kv_head_num、模型总特征维度适配）
- k_scale、v_scale 的形状需与 k_cache/v_cache 的头维度兼容（如 k_scale.shape = (kv_head_num,) 或 (1, kv_head_num, 1, 1)），支持广播计算
- 所有输入张量需满足连续存储（contiguous）的内存布局要求
- k_cache、v_cache 需具备可写权限，且内存空间足够容纳本次写入的 KV 缓存数据
### 调用示例
```python
import torch
import mcoplib.op as op
# 定义运行设备
device = "cuda"
# 定义算子所需的参数
batch_size = 4
q_head_num = 32
kv_head_num = 8      
head_dim = 128
seq_len = 128        
total_tokens = batch_size * seq_len
max_cache_size = 4096 
#  关键修正点 
# 1. 创建输入张量
# 必须显式展开 Head 维度，使 shape 变为 [Total_Tokens, Num_Heads, Head_Dim]
# 这样算子读取 size(2) 时才能得到正确的 128 (head_dim)，而不是 6144 (hidden_dim)
num_all_heads = q_head_num + 2 * kv_head_num  # 32 + 8 + 8 = 48
packed_qkv = torch.randn(total_tokens, num_all_heads, head_dim, dtype=torch.float16, device=device)
# 
# q_lens: 保持 int32
q_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
# accum_q_lens: 保持 int32
accum_q_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
accum_q_lens[1:] = torch.cumsum(q_lens, dim=0)
# cache_lens: 保持 int32
cache_lens = torch.randint(0, 100, (batch_size,), dtype=torch.int32, device=device)
# cache_slot_ids: 保持 int32，一维扁平索引
cache_slot_ids = torch.randperm(max_cache_size, device=device)[:total_tokens].to(torch.int32)
# 2. 创建输出/Cache张量
# 保持 int8 (量化存储)
k_cache = torch.zeros(max_cache_size, kv_head_num, head_dim, dtype=torch.int8, device=device)
v_cache = torch.zeros(max_cache_size, kv_head_num, head_dim, dtype=torch.int8, device=device)
k_scale = torch.ones(max_cache_size, kv_head_num, dtype=torch.float32, device=device)
v_scale = torch.ones(max_cache_size, kv_head_num, dtype=torch.float32, device=device)
# 为了确保能捕捉到异步错误，加入同步操作
torch.cuda.synchronize()
print("Calling store_kv_cache_cuda_interface...")
# 3. 调用算子
op.store_kv_cache_cuda_interface(
    packed_qkv,
    q_lens,
    accum_q_lens,
    cache_lens,
    cache_slot_ids,
    k_cache,      
    v_cache,      
    k_scale,
    v_scale,
    batch_size,
    q_head_num,
    kv_head_num
)
# 再次同步，确保算子真正执行完毕且无报错
torch.cuda.synchronize()
print("store_kv_cache_cuda_interface computation completed successfully.")
print(f"Packed QKV shape: {packed_qkv.shape}")
print(f"K Cache shape: {k_cache.shape}")
```
## 20. moe_scatter_dynamic_quant
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def moe_scatter_dynamic_quant(
    hidden_status: torch.Tensor,
    selected_experts: torch.Tensor,
    moe_weights: torch.Tensor,
    smooth_scale: torch.Tensor,
    scatter_tokens: torch.Tensor,
    scatter_per_token_scale: torch.Tensor,
    scatter_tokens_offset: torch.Tensor,
    experts_token_count: torch.Tensor,
    experts_token_start: torch.Tensor,
    experts_per_rank: int,
    shared_experts_per_rank: int,
    shared_tokens_per_sp: int
) -> None
```
### 功能描述
moe_scatter_dynamic_quant 算子实现了混合专家 MoE 模型中 token 到对应专家的分散分配操作。该算子结合 token 选中的专家索引、权重等信息，将输入隐藏状态张量的 token 特征分散到各专家对应的存储区域，并同步处理缩放因子、token 偏移量、专家 token 计数等辅助信息，减少 MoE 层中数据搬运的开销，提升计算效率，常用于 MoE 架构的模型推理或训练流程
- 计算公式
	$$
	\text{scatter\_tokens}[e, t, :] = \text{dequantize}\left(\text{quantize}\left(\text{hidden\_status}[i, :] \times \text{moe\_weights}[i, e] \times \text{smooth\_scale}[i], s_q^{(i,e)}\right), s_q^{(i,e)}\right)
	$$
	$$
	\text{scatter\_per\_token\_scale}[e, t] = s_q^{(i,e)} \times \text{moe\_weights}[i, e] \times \text{smooth\_scale}[i]
	$$
	$$
	\text{experts\_token\_count}[e] = \sum_{i=0}^{N-1} \mathbb{I}\left(\text{selected\_experts}[i] = e\right)
	$$
	$$
	\text{experts\_token\_start}[e] = \sum_{e'=0}^{e-1} \text{experts\_token\_count}[e']
	$$
	$$
	\text{scatter\_tokens\_offset}[e] = \text{experts\_token\_start}[e]
	$$
	动态量化缩放因子计算：
	$$
	s_q^{(i,e)} = \frac{\max\left(|\text{hidden\_status}[i, :] \times \text{moe\_weights}[i, e] \times \text{smooth\_scale}[i]|\right)}{127}
	$$
	量化/反量化操作定义：
	$$
	\text{quantize}(x, s) = \text{clip}\left(\lfloor \frac{x}{s} \rfloor, -128, 127\right) \quad (\text{int8动态量化})
	$$
	$$
	\text{dequantize}(x_q, s) = x_q \times s \quad (\text{int8反量化})
	$$
其中：
- $i$: 全局 token 索引，$i \in [0, N-1]$，其中$N$为输入总 token 数量
- $e$: 专家索引，$e \in [0, E_{\text{rank}}-1]$，其中$E_{\text{rank}} = \text{experts\_per\_rank}$（单 rank 专家数）
- $t$: 专家$e$内的局部 token 索引，$t \in [0, C_e-1]$，其中$C_e = \text{experts\_token\_count}[e]$（专家$e$分配到的 token 数）
- $\text{hidden\_status}[i, :] \in \mathbb{R}^{D}$: 第$i$个 token 的浮点型隐藏状态张量，$D$为特征维度
- $\text{selected\_experts}[i] \in \mathbb{N}$: 第$i$个 token 选中的目标专家索引
- $\text{moe\_weights}[i, e] \in \mathbb{R}$: 第$i$个 token 对应专家$e$的权重系数
- $\text{smooth\_scale}[i] \in \mathbb{R}$: 第$i$个 token 的数值平滑缩放因子
- $s_q^{(i, e)} \in \mathbb{R}$: 第$i$个 token 分配到专家$e$时的动态量化缩放因子（int8 量化）
- $\mathbb{I}(\cdot)$: 指示函数，条件为真时返回 1，否则返回 0
- $E_{\text{shared}} = \text{shared\_experts\_per\_rank}$: 单 rank 部署的共享专家数量
- $T_{\text{shared}} = \text{shared\_tokens\_per\_sp}$: 单个共享专家支持的最大 token 数量
- $\text{clip}(\cdot, a, b)$: 裁剪函数，将输入值限制在$[a, b]$范围内
- $\lfloor \cdot \rfloor$: 向下取整函数
对于共享专家$e \in [E_{\text{rank}} - E_{\text{shared}}, E_{\text{rank}} - 1]$，需满足 token 数量上限约束：
$$
\text{experts\_token\_count}[e] \leq T_{\text{shared}}
$$
### 参数说明
- **hidden_status** (torch.Tensor, 入参): MoE 层的输入 token 隐藏状态张量，存储待分配的 token 特征数据
- **selected_experts** (torch.Tensor, 入参): 每个 token 对应的选中专家索引张量，指定 token 需分配到的目标专家
- **moe_weights** (torch.Tensor, 入参): 每个 token 对应选中专家的权重张量，用于对 token 特征进行加权
- **smooth_scale** (torch.Tensor, 入参): 数值平滑缩放因子张量，用于提升 token 特征计算的稳定性
- **scatter_tokens** (torch.Tensor, 入参/出参): 输出张量，存储分散到各专家后的 token 特征数据
- **scatter_per_token_scale** (torch.Tensor, 入参/出参): 输出张量，存储每个分散后 token 对应的缩放因子
- **scatter_tokens_offset** (torch.Tensor, 入参/出参): 输出张量，存储各专家对应的 token 在 scatter_tokens 中的起始偏移位置
- **experts_token_count** (torch.Tensor, 入参/出参): 输出张量，存储每个专家实际分配到的 token 数量
- **experts_token_start** (torch.Tensor, 入参/出参): 输出张量，存储每个专家对应的 token 在 scatter_tokens 中的起始位置索引
- **experts_per_rank** (int, 入参): 单个计算设备（rank）上部署的专家数量
- **shared_experts_per_rank** (int, 入参): 单个计算设备（rank）上部署的共享专家数量
- **shared_tokens_per_sp** (int, 入参): 每个共享专家支持处理的最大 token 数量
### 返回值
无返回值，计算结果直接写入 scatter_tokens、scatter_per_token_scale 等输出型张量中
### 约束与调用
- 所有张量必须部署在 CUDA 设备上
- selected_experts 中的专家索引需在当前设备的专家数量范围内，不超出 experts_per_rank 等参数定义的范围
- scatter_tokens、experts_token_count 等输出张量需提前分配足够的内存空间
- 支持的数据类型：float16、bfloat16、float32，需与 hidden_status 的张量类型保持一致
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
batch_size = 4                
seq_len = 64                  
hidden_dim = 4096             
topk = 2                    
experts_per_rank = 8          
shared_experts_per_rank = 2  
shared_tokens_per_sp = 1024  
total_tokens = batch_size * seq_len 
total_expert_tokens = total_tokens * topk
#创建输入数据
hidden_status = torch.randn(
    total_tokens, hidden_dim,
    dtype=torch.bfloat16, device=device
)
selected_experts = torch.randint(
    0, experts_per_rank, (total_tokens, topk),
    dtype=torch.int32, device=device
)
moe_weights = torch.softmax(
    torch.randn(total_tokens, topk, dtype=torch.float32, device=device),
    dim=-1
)
smooth_scale = torch.randn(
    hidden_dim, dtype=torch.float32, device=device
)
scatter_tokens = torch.randint(
    -128, 127, (total_expert_tokens, hidden_dim),
    dtype=torch.int8, device=device
)
scatter_per_token_scale = torch.randn(
    total_expert_tokens, dtype=torch.float32, device=device
)
scatter_tokens_offset = torch.randint(
    -128, 127, (total_expert_tokens,),
    dtype=torch.int32, device=device
)
experts_token_count = torch.full(
    (experts_per_rank,), total_expert_tokens // experts_per_rank,
    dtype=torch.int32, device=device
)
remainder = total_expert_tokens % experts_per_rank
experts_token_count[:remainder] += 1  
experts_token_start = torch.cumsum(
    torch.cat([torch.tensor([0], dtype=torch.int32, device=device), experts_token_count[:-1]]),
    dim=0
).to(torch.int32)
# 调用算子
op.moe_scatter_dynamic_quant(
    hidden_status,
    selected_experts,
    moe_weights,
    smooth_scale,
    scatter_tokens,
    scatter_per_token_scale,
    scatter_tokens_offset,
    experts_token_count,
    experts_token_start,
    experts_per_rank,
    shared_experts_per_rank,
    shared_tokens_per_sp
)
print("MOE scatter dynamic quant computation completed")
print(f"hidden_status shape: {hidden_status.shape}, dtype: {hidden_status.dtype}")
print(f"selected_experts shape: {selected_experts.shape}, dtype: {selected_experts.dtype}")
print(f"moe_weights shape: {moe_weights.shape}, dtype: {moe_weights.dtype}")
print(f"experts_token_count shape: {experts_token_count.shape}, sum: {experts_token_count.sum().item()}")
print(f"experts_token_start shape: {experts_token_start.shape}, last start: {experts_token_start[-1].item()}")
print(f"Static params: experts_per_rank={experts_per_rank}, shared_experts_per_rank={shared_experts_per_rank}, shared_tokens_per_sp={shared_tokens_per_sp}")
```
## 21. scale_dynamic_quant
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def scale_dynamic_quant(
    hidden_states: torch.Tensor,
    smooth_scales: torch.Tensor,
    dst_dtype: torch.dtype = torch.int8
) -> tuple[torch.Tensor, torch.Tensor]
```
### 功能描述
scale_dynamic_quant 算子实现了动态量化与尺度计算的融合操作。该算子首先基于输入的 hidden_states 和 smooth_scales 计算量化尺度 scales，然后将 hidden_states 量化到指定的目标数据类型（默认 int8）。这种融合操作可减少中间变量存储，提高量化过程的计算效率，常用于神经网络中张量的动态量化场景
- 计算公式：
  $$
  \text{scales} = \frac{\max(|\text{hidden\_states}|, \epsilon)}{\text{qmax} \times \text{smooth\_scales}}
  $$
  $$
  \text{quantized\_tensor} = \text{clip}\left(\text{round}\left(\frac{\text{hidden\_states}}{\text{scales}}\right), \text{qmin}, \text{qmax}\right)
  $$
  其中：
  - $\text{hidden\_states} \in \mathbb{R}^{B \times S \times D}$ 是输入隐藏状态张量，$B$ 为批次大小，$S$ 为序列长度，$D$ 为特征维度
  - $\text{smooth\_scales} \in \mathbb{R}^{D}$ 是平滑尺度张量（与特征维度对齐）
  - $\text{scales} \in \mathbb{R}^{D}$ 是计算得到的量化尺度张量（与特征维度对齐）
  - $\text{quantized\_tensor} \in \mathbb{Z}^{B \times S \times D}$ 是量化后的整数张量，数据类型由 $\text{dst\_dtype}$ 指定
  - $\epsilon$ 是数值稳定性参数（默认取 $10^{-8}$），避免分母为 0
  - $\text{qmin}$ 和 $\text{qmax}$ 是目标数据类型 $\text{dst\_dtype}$ 的取值范围边界：
    - 若 $\text{dst\_dtype} = \text{int8}$，则 $\text{qmin} = -128$，$\text{qmax} = 127$
    - 其他整数类型按对应 dtype 的固有范围确定
  - $\max(|\text{hidden\_states}|)$ 表示对 $\text{hidden\_states}$ 在 $B \times S$ 维度上取绝对值后的最大值（按特征维度 $D$ 独立计算）
  - $\text{round}(\cdot)$ 表示四舍五入取整操作
  - $\text{clip}(\cdot, \text{qmin}, \text{qmax})$ 表示将数值裁剪到 $\text{qmin}$ 和 $\text{qmax}$ 之间，确保符合目标 dtype 范围
  补充说明：
  - 当 $\text{smooth\_scales}$ 为标量时，自动广播为 $\mathbb{R}^{D}$ 维度（所有特征维度使用相同平滑系数）
  - 量化尺度计算采用"特征维度独立"策略，每个特征维度对应一个独立的量化尺度
  - 输入 $\text{hidden\_states}$ 的数据类型支持 $\text{float16}$、$\text{bfloat16}$、$\text{float32}$，计算过程中统一转换为 $\text{float32}$ 进行精度保证
### 参数说明
- **hidden_states** (torch.Tensor, 入参): 输入的隐藏状态张量，为量化的原始数据
- **smooth_scales** (torch.Tensor, 入参): 平滑尺度张量，用于辅助计算量化尺度
- **dst_dtype** (torch.dtype, 入参, 可选): 目标量化数据类型，默认值为 torch.int8
### 返回值
返回一个元组 (torch.Tensor, torch.Tensor)，其中：
- 第一个元素为量化后的张量，数据类型为 dst_dtype
- 第二个元素为计算得到的量化尺度张量 scales
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 输入 hidden_states 支持的数据类型：float16、bfloat16、float32
- 目标数据类型 dst_dtype 通常为整数类型（如 int8）
- smooth_scales 的形状需与 hidden_states 的相关维度匹配，以确保尺度计算的有效性
### 调用示例
```Python
import torch
import mcoplib.op as op
device = "cuda"
# 定义参数
batch_size = 8          
seq_len = 1024          
hidden_dim = 4096      
dst_dtype = torch.int8 
# 创建输入张量
hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16, device=device)
smooth_scales = torch.randn(hidden_dim, dtype=torch.float32, device=device)  # 特征维度对齐的平滑尺度
# 调用scale_dynamic_quant算子
quantized_hidden_states, scales = op.scale_dynamic_quant(
    hidden_states=hidden_states,
    smooth_scales=smooth_scales,
    dst_dtype=dst_dtype
)
# 打印执行结果
print("Scale dynamic quantization computation completed")
print(f"Quantized hidden states shape: {quantized_hidden_states.shape}, dtype: {quantized_hidden_states.dtype}")
print(f"Computed scales shape: {scales.shape}, dtype: {scales.dtype}")
```
## 22. rotary_pos_emb_forward
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def rotary_pos_emb_forward(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    cumsum_len: torch.Tensor,
    batch_size: int,
    cut_head_dim: int = 0
) -> torch.Tensor
```
### 功能描述
rotary_pos_emb_forward 算子实现旋转位置编码的前向计算，结合输入张量、预计算的正弦/余弦位置编码张量，以及批次样本的累积长度信息，对输入进行旋转位置编码处理，用于增强 Transformer 类模型中序列的位置信息表示。
- 计算公式：
  $$
  \text{rotary\_pos\_emb\_forward}(\mathbf{X}, \sin, \cos, \text{cumsum\_len}, B, D_{cut}) = \mathbf{X}'
  $$
  其中$\mathbf{X}'$的元素计算遵循旋转位置编码（RoPE）规则：
  $$
  \begin{cases}
  x'_{b,l,h,d} = x_{b,l,h,d}, & d < D_{cut} \\
  x'_{b,l,h,2m} = x_{b,l,h,2m} \cdot \cos(\theta_{m, pos_{b,l}}) - x_{b,l,h,2m+1} \cdot \sin(\theta_{m, pos_{b,l}}), & d = 2m \geq D_{cut} \\
  x'_{b,l,h,2m+1} = x_{b,l,h,2m} \cdot \sin(\theta_{m, pos_{b,l}}) + x_{b,l,h,2m+1} \cdot \cos(\theta_{m, pos_{b,l}}), & d = 2m+1 \geq D_{cut}
  \end{cases}
  $$
  位置编码的角度参数定义为：
  $$
  \theta_{m, pos} = \frac{pos}{10000^{2m/D'}}, \quad D' = D - D_{cut}
  $$
  批次中第$b$个样本第$l$个位置的绝对位置：
  $$
  pos_{b,l} = \text{cumsum\_len}[b] + l
  $$
  其中：
	- $\mathbf{X} \in \mathbb{R}^{B \times L \times H \times D}$ 是输入张量（$B$为批次大小，$L$为序列长度，$H$为注意力头数，$D$为头维度）
	- $\sin \in \mathbb{R}^{L_{max} \times D'/2}$ 是预计算的正弦位置编码张量（$L_{max}$为最大序列长度）
	- $\cos \in \mathbb{R}^{L_{max} \times D'/2}$ 是预计算的余弦位置编码张量
	- $\text{cumsum\_len} \in \mathbb{Z}^B$ 是批次中各样本的累积长度张量，用于确定局部位置的绝对偏移
	- $D_{cut} \in \mathbb{N}$ 是裁剪的头维度长度，默认值为 0
	- $D'$ 是参与旋转编码的有效头维度长度
	- $pos_{b, l}$ 是第$b$个样本第$l$个位置的绝对序列位置
	- $\theta_{m, pos}$ 是位置$pos$下第$m$组维度对的旋转角度
### 参数说明
- **input** (torch.Tensor, 入参): 待施加旋转位置编码的输入张量
- **sin** (torch.Tensor, 入参): 预计算的正弦位置编码张量
- **cos** (torch.Tensor, 入参): 预计算的余弦位置编码张量
- **cumsum_len** (torch.Tensor, 入参): 批次中各样本的累积长度张量，用于确定每个样本的局部序列位置
- **batch_size** (int, 入参): 输入数据的批次大小
- **cut_head_dim** (int, 入参，默认 0): 需裁剪的头维度长度，仅对裁剪后的剩余头维度执行旋转编码
### 返回值
torch.Tensor，经过旋转位置编码处理后的输出张量
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- cut_head_dim 的值必须小于输入张量的头维度长度
- 支持的数据类型：float16、bfloat16、float32
- cumsum_len 需与批次内各样本的长度信息匹配
### 调用示例
```python
import torch
import mcoplib.op
device = "cuda"
# 定义参数
batch_size = 4  
seq_len = 64    
num_heads = 32  
head_dim = 128  
cut_head_dim = 32 
max_seq_len = 2048  
# 计算参与旋转编码的有效头维度
effective_head_dim = head_dim - cut_head_dim
assert effective_head_dim % 2 == 0, "Effective head dimension must be even for RoPE"
# 创建输入张量
input_tensor = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
# 创建预计算的sin/cos位置编码张量
sin = torch.randn(max_seq_len, effective_head_dim // 2, dtype=torch.float16, device=device)
cos = torch.randn(max_seq_len, effective_head_dim // 2, dtype=torch.float16, device=device)
# 创建批次累积长度张量
cumsum_len = torch.tensor([0, 512, 1024, 1536], dtype=torch.int64, device=device)
# 调用rotary_pos_emb_forward算子
output = op.rotary_pos_emb_forward(
    input_tensor,
    sin,
    cos,
    cumsum_len,
    batch_size,
    cut_head_dim
)
# 打印结果信息
print("Rotary position embedding forward computation completed")
print(f"Input tensor shape: {input_tensor.shape}")
print(f"Output tensor shape: {output.shape}")
print(f"Output tensor dtype: {output.dtype}")
```
## 23. rotary_pos_emb_backward
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def rotary_pos_emb_backward(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    cumsum_len: torch.Tensor,
    batch_size: int,
    cut_head_dim: int = 0
) -> torch.Tensor
```
### 功能描述
rotary_pos_emb_backward 算子实现旋转位置编码的反向传播计算，基于前向过程的输入、正弦/余弦位置编码及累积长度信息，计算对应输入的梯度，用于模型训练中的反向梯度传递
- 计算公式：
  $$
  \text{rotary\_pos\_emb\_backward}(\mathbf{G}, \sin, \cos, \text{cumsum\_len}, B, D_{cut}) = \nabla \mathbf{X}
  $$
  其中梯度$\nabla \mathbf{X}$的元素由输出梯度$\mathbf{G}$反向推导得到：
  $$
  \begin{cases}
  \nabla x_{b,l,h,d} = g_{b,l,h,d}, & d < D_{cut} \\
  \nabla x_{b,l,h,2m} = g_{b,l,h,2m} \cdot \cos(\theta_{m, pos_{b,l}}) + g_{b,l,h,2m+1} \cdot \sin(\theta_{m, pos_{b,l}}), & d = 2m \geq D_{cut} \\
  \nabla x_{b,l,h,2m+1} = -g_{b,l,h,2m} \cdot \sin(\theta_{m, pos_{b,l}}) + g_{b,l,h,2m+1} \cdot \cos(\theta_{m, pos_{b,l}}), & d = 2m+1 \geq D_{cut}
  \end{cases}
  $$
  角度参数$\theta_{m, pos}$和绝对位置$pos_{b, l}$的定义与前向过程一致：
  $$
  \theta_{m, pos} = \frac{pos}{10000^{2m/D'}}, \quad pos_{b,l} = \text{cumsum\_len}[b] + l, \quad D' = D - D_{cut}
  $$
  其中：
	- $\mathbf{G} \in \mathbb{R}^{B \times L \times H \times D}$ 是旋转编码输出的梯度张量（即损失函数对前向输出$\mathbf{X}'$的梯度）
	- $\nabla \mathbf{X} \in \mathbb{R}^{B \times L \times H \times D}$ 是损失函数对前向输入$\mathbf{X}$的梯度张量（算子返回值）
	- $\sin \in \mathbb{R}^{L_{max} \times D'/2}$ 是与前向过程一致的预计算正弦位置编码张量
	- $\cos \in \mathbb{R}^{L_{max} \times D'/2}$ 是与前向过程一致的预计算余弦位置编码张量
	- $\text{cumsum\_len} \in \mathbb{Z}^B$ 是与前向过程一致的批次累积长度张量
	- $B \in \mathbb{N}$ 是批次大小（与前向过程一致）
	- $D_{cut} \in \mathbb{N}$ 是裁剪的头维度长度（与前向过程一致）
	- $D'$ 是参与旋转编码的有效头维度长度
	- $pos_{b, l}$ 是第$b$个样本第$l$个位置的绝对序列位置
	- $\theta_{m, pos}$ 是位置$pos$下第$m$组维度对的旋转角度
	- $g_{b, l, h, d}$ 是梯度张量$\mathbf{G}$在$(b, l, h, d)$位置的元素
	- $\nabla x_{b, l, h, d}$ 是梯度张量$\nabla \mathbf{X}$在$(b, l, h, d)$位置的元素
### 参数说明
- **input** (torch.Tensor, 入参): 反向传播中待计算梯度的输入张量，通常为前向输出的梯度张量
- **sin** (torch.Tensor, 入参): 预计算的正弦位置编码张量，与前向过程使用的张量一致
- **cos** (torch.Tensor, 入参): 预计算的余弦位置编码张量，与前向过程使用的张量一致
- **cumsum_len** (torch.Tensor, 入参): 批次中各样本的累积长度张量，与前向过程使用的张量一致
- **batch_size** (int, 入参): 输入数据的批次大小，与前向过程的批次大小一致
- **cut_head_dim** (int, 入参，默认 0): 需裁剪的头维度长度，与前向过程的裁剪长度一致
### 返回值
torch.Tensor，经过反向传播计算得到的梯度张量
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- cut_head_dim 的值必须小于输入张量的头维度长度
- 支持的数据类型：float16、bfloat16、float32
- 所有参数（sin、cos、cumsum_len 等）需与前向过程对应的参数保持一致
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义参数（与前向保持一致）
batch_size = 4
seq_len = 64
num_heads = 32
head_dim = 128
cut_head_dim = 32
max_seq_len = 2048
# 计算参与旋转编码的有效头维度
effective_head_dim = head_dim - cut_head_dim
assert effective_head_dim % 2 == 0, "Effective head dimension must be even for RoPE"
# 创建反向传播的输入张量（前向输出的梯度）：[batch_size, seq_len, num_heads, head_dim]
grad_input = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device=device)
# 创建与前向一致的sin/cos位置编码张量：[max_seq_len, effective_head_dim//2]
sin = torch.randn(max_seq_len, effective_head_dim // 2, dtype=torch.float16, device=device)
cos = torch.randn(max_seq_len, effective_head_dim // 2, dtype=torch.float16, device=device)
# 创建与前向一致的批次累积长度张量：[batch_size]
cumsum_len = torch.tensor([0, 512, 1024, 1536], dtype=torch.int64, device=device)
# 调用rotary_pos_emb_backward算子
grad_output = op.rotary_pos_emb_backward(
    grad_input,
    sin,
    cos,
    cumsum_len,
    batch_size,
    cut_head_dim
)
# 打印结果信息
print("Rotary position embedding backward computation completed")
print(f"Gradient input shape: {grad_input.shape}")
print(f"Gradient output shape: {grad_output.shape}")
print(f"Gradient output dtype: {grad_output.dtype}")
```
## 24. fused_add_rms_norm_dynamic_per_token_quant_padding_output
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_add_rms_norm_dynamic_per_token_quant_padding_output(
    output: torch.Tensor,         
    output_rms:torch.Tensor,
    output_quant_int8: torch.Tensor,  
    out_scales: torch.Tensor,     
    input: torch.Tensor,          
    residual: torch.Tensor,       
    weight: torch.Tensor,       
    pad_size: int,                
    epsilon: float             
) -> None
```
### 功能描述
add_rms_norm_dynamic_per_token_quant_padding_output 算子实现带 RMS 归一化、残差连接的动态逐 Token 量化计算与输出填充。该算子先对输入执行 RMS 归一化，结合残差与权重完成运算，再基于每个 Token 的激活特征动态生成量化缩放因子并执行量化，对输出进行填充适配，用于需要填充场景的高效量化推理加速。
- 计算公式：
  $$
  \hat{Y} = \text{Quant}\left( (X \cdot W) + R, \frac{\max_{d}|(X \cdot W + R)_{n,d}| + \epsilon}{Q_{\text{max}}} \right)
  $$
  其中 $\text{Quant}(x,S_n)$ 表示按缩放因子$S_n$对$x$执行量化操作
  矩阵乘加和残差连接：
  $$
  P_{n,d} = (X_n \cdot W_d) + R_{n,d}
  $$
  向量优化：
  $$
  \mathbf{P}_{n,v} = (X_{n,v} \odot W_v) + R_{n,v}
  $$
  计算 Token 级最大激活幅度：
  $$
  M_n = \max_{d \in [1,D]} |P_{n,d}|
  $$
  计算 Token 级缩放因子：
  $$
  S_n = \frac{M_n + \epsilon}{Q_{\text{max}}}
  $$
  浮点数量化为整数：
  $$
  \hat{Y}_{n,d} = \text{round}\left( \text{clamp}\left( \frac{P_{n,d}}{S_n}, -Q_{\text{max}}, Q_{\text{max}} \right) \right)
  $$
  其中：
	- $X \in \mathbb{R}^{N \times D}$ 是输入 Token 特征张量（$N$=num_tokens，$D$=hidden_size）
	- $W \in \mathbb{R}^{D \times D}$ 是隐藏层权重张量
	- $R \in \mathbb{R}^{N \times D}$ 是残差连接张量，维度与输入一致
	- $P \in \mathbb{R}^{N \times D}$ 是乘加+残差后的中间结果张量
	- $\mathbf{P}_{n,v}$ 是向量优化版中间结果（$v$为向量索引，向量长度$L$=vec_len）
	- $X_n$ 是输入第$n$个 Token 的特征向量（$1 \times D$），$W_d$ 是权重第$d$列向量（$D \times 1$）
	- $X_{n,v}/W_v/R_{n,v}$ 是长度为$L$的向量，$\odot$ 表示逐元素乘法，$\cdot$ 表示向量点积
	- $M_n$ 是第$n$个 Token 中间结果的最大绝对值（块内归约得到）
	- $S \in \mathbb{R}^N$ 是 Token 级动态量化缩放因子张量
	- $Q_{\text{max}}$ 是量化类型最大值
	- $\epsilon$ 是防止除零的小值（默认$10^{-6}$）
	- $\hat{Y} \in \mathbb{Z}^{N \times D}$ 是量化后的输出张量（整数类型）
	- $\text{clamp}(x,a,b)$ 表示将$x$截断到$[a,b]$范围，$\text{round}(x)$ 表示四舍五入取整
  算子优化目标：
  $$
  \min_{\hat{Y}} \sum_{n=1}^N \sum_{d=1}^D \left| P_{n,d} - \hat{Y}_{n,d} \cdot S_n \right|^2
  $$
  量化误差约束：
  $$
  \max_{n,d} \left| P_{n,d} - \hat{Y}_{n,d} \cdot S_n \right| \leq Q_{\text{max}} \cdot \epsilon
  $$
  其中 $\hat{Y}_{n,d} \cdot S_n$ 是量化结果的反量化值，约束量化前后的最大绝对误差不超过阈值。
  **主要功能包括：**
- 基于 CUDA 实现 RMS 归一化预处理，稳定输入特征数值范围
- 支持残差连接与矩阵乘加融合计算，保留原始特征并完成线性变换
- 实现动态逐 Token 量化，基于激活最大绝对值生成专属缩放因子
- 提供输出填充适配能力，满足固定维度或边界对齐的推理场景
- 结合向量化访存与线程协同优化，提升 CUDA 核函数执行效率
### 参数说明
- **input** (torch.Tensor, 入参): 形状为[num_tokens, hidden_size]，存储待处理的 Token 特征张量
- **weight** (torch.Tensor, 入参): 形状为[hidden_size, hidden_size]，存储隐藏层权重张量
- **residual** (torch.Tensor, 入参): 形状为[num_tokens, hidden_size]，存储残差连接张量
- **epsilon** (float, 入参): 标量，防止计算 RMS 归一化时除零的极小值
- **pad_size** (int, 入参): 标量，输出张量的填充尺寸，用于对齐输出维度
### 返回值
无返回值，结果直接写到输出张量 output 中，输出张量包含：output, output_rms, output_quant_int8, out_scales 和 residual
- **output**：形状为[num_tokens, hidden_size + pad_size]，数据类型为 bfloat16，存储 RMS 归一化+矩阵乘加+残差后的中间结果张量
- **output_rms**：形状为 [num_tokens, hidden_size]，数据类型为 bfloat16，存储输入张量经过 RMS 归一化后的中间结果张量
- **output_quant_int8**：形状为[num_tokens, hidden_size + pad_size]，数据类型为 int8，存储动态逐 Token 量化后的输出张量
- **out_scales**：形状为[num_tokens]，数据类型为 float，存储每个 Token 对应的量化缩放因子
residual：保持不变
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 输入张量仅支持 bfloat16，输出张量：output_quant_int8 为 int8，out_scale 为 float，output 为 bfloat16
- 张量形状约束：input 与 residual 形状需一致为[num_tokens, hidden_size]，weight 形状需为[hidden_size, hidden_size]，与 input 的特征维度匹配
- pad_size 必须为非负整数，epsilon 必须为正数
- 仅支持 int8 动态逐 Token 量化，不支持其他量化位宽
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 1. 定义维度
batch_size = 1      # 或者是 num_tokens 的维度，视具体实现而定
seq_len = 128
hidden_size = 4096
pad_size = 0        # 根据业务需求设置，通常为0或者对齐所需的padding
epsilon = 1e-6      # RMS Norm 的 epsilon
# 2. 准备输入数据 (Inputs)
# 注意：residual 通常需要与 input 形状一致
input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
residual = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device) # RMS Norm Gamma
# 3. 准备输出容器 (Outputs)
# output: BF16 (用于存放非量化的 Add 结果，或者 Padding Output)
output = torch.empty_like(input_tensor)
# output_rms: BF16 (RMS Norm 后的结果)
output_rms = torch.empty_like(input_tensor)
# output_quant_int8: Int8 (量化结果)
output_quant_int8 = torch.empty(batch_size, seq_len, hidden_size, dtype=torch.int8, device=device)
# out_scales: Float32
out_scales = torch.empty(batch_size, seq_len, 1, dtype=torch.float32, device=device)
# 4. 调用算子 (修改顺序)
print("Calling fused_add_rms_norm_dynamic_per_token_quant_padding_output...")
# 这里的关键修改是：将 Int8 张量作为第一个参数传入
op.fused_add_rms_norm_dynamic_per_token_quant_padding_output(
    output_quant_int8,      # [Arg 0] 此时传入 Int8，满足 "Expected output.dtype() == kInt8"
    output_rms,             # [Arg 1] RMS 输出 (BF16)
    output,                 # [Arg 2] 剩下的这个 BF16 输出放在这里 (原签名中的 output_quant_int8 位置)
    out_scales,             # [Arg 3] Scales
    input_tensor,           # [Arg 4] Input
    residual,               # [Arg 5] Residual
    weight,                 # [Arg 6] Weight
    pad_size,               # [Arg 7] Pad Size
    epsilon                 # [Arg 8] Epsilon
)
torch.cuda.synchronize()
print("Call successful.")
print(f"Quantized Output Shape: {output_quant_int8.shape}")
```
## 25. rms_norm_dynamic_per_token_quant_custom
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def rms_norm_dynamic_per_token_quant_custom(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    var_epsilon: float,
    scale_ub: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None
) -> None
```
### 功能描述
rms_norm_dynamic_per_token_quant_custom 算子实现了 RMS 归一化与动态逐 token 量化的融合操作。该算子首先对输入（可选叠加残差）进行 RMS 归一化计算，过程中引入动态逐 token 尺度调整，结合量化尺度参数完成量化操作，最终将结果输出到指定张量，通过融合计算减少内存交互，提升处理效率，适用于量化场景下的 Transformer 模型层。
- 计算公式：
  $$
  \text{input\_merged} =
  \begin{cases}
  \text{input} + \text{residual}, & \text{residual} \neq \text{None} \\
  \text{input}, & \text{residual} = \text{None}
  \end{cases}
  $$
  
  $$
  \text{rms} = \sqrt{\frac{1}{D} \sum_{d=1}^D \text{input\_merged}_{(\cdot, d)}^2 + \text{var\_epsilon}}
  $$
  
  $$
  \text{scale\_dynamic} =
  \begin{cases}
  \text{scales}, & \text{scale\_ub} = \text{None} \\
  \min(\text{scales}, \text{scale\_ub}), & \text{scale\_ub} \neq \text{None}
  \end{cases}
  $$
  
  $$
  \text{out} = \frac{\text{input\_merged}}{\text{rms}} \odot \text{weight} \odot \text{scale\_dynamic}
  $$
  其中：
  - $\text{input} \in \mathbb{R}^{B \times L \times D}$ 是输入张量，$B$ 为批次大小，$L$ 为序列长度，$D$ 为特征维度（hidden_size）
  - $\text{residual} \in \mathbb{R}^{B \times L \times D}$（可选）是残差张量，与 input 形状一致，用于元素级叠加
  - $\text{input\_merged} \in \mathbb{R}^{B \times L \times D}$ 是输入与残差叠加后的中间张量
  - $\text{weight} \in \mathbb{R}^D$ 是 RMS 归一化的权重张量，沿特征维度缩放
  - $\text{var\_epsilon} \in \mathbb{R}^+$ 是数值稳定性参数，避免均方根计算时分母为零（通常取$10^{-6}$量级）
  - $\text{scales} \in \mathbb{R}^{B \times L}$ 是动态逐 token 量化尺度张量，为每个 token 提供独立的量化尺度
  - $\text{scale\_ub} \in \mathbb{R}^{B \times L}$（可选）是尺度上界张量，约束动态尺度的最大值
  - $\text{scale\_dynamic} \in \mathbb{R}^{B \times L}$ 是经过上界约束后的最终动态量化尺度
  - $\text{rms} \in \mathbb{R}^{B \times L}$ 是输入合并张量的均方根值，沿特征维度$D$计算
  - $\text{out} \in \mathbb{R}^{B \times L \times D}$ 是最终输出张量，存储量化后的 RMS 归一化结果
  - $\odot$ 表示元素级乘法（Hadamard 积），维度不匹配时自动广播（如$D$维$\text{weight}$广播至$B \times L \times D$，$B \times L$维$\text{scale\_dynamic}$广播至$B \times L \times D$）
  - $D$ 需满足对齐约束（如$D \% 4 = 0$），以保证 CUDA 设备上的计算效率
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，用于存储最终的量化归一化结果
- **input** (torch.Tensor, 入参): 输入张量，为待进行 RMS 归一化和量化的原始数据
- **weight** (torch.Tensor, 入参): 权重张量，用于 RMS 归一化过程中的缩放调整，形状与归一化维度匹配
- **scales** (torch.Tensor, 入参): 量化尺度张量，用于动态逐 token 量化的尺度计算
- **var_epsilon** (float, 入参): 数值稳定性参数，用于避免 RMS 计算中分母为零，通常取极小正值（如 1e-6）
- **scale_ub** (Optional[torch.Tensor], 入参): 尺度上界张量，可选参数，用于约束动态尺度的上限范围
- **residual** (Optional[torch.Tensor], 入参): 残差张量，可选参数，若存在则先与 input 叠加后再进行后续计算
### 返回值
无返回值，计算结果直接写入 out 张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 张量需满足连续性要求（
- 支持的数据类型：float16、bfloat16、float32
- 当 residual 存在时，其形状需与 input 匹配以支持元素级相加
- hidden_size 需与输入张量的特征维度匹配，且需满足向量对齐要求, 如代码中提到的 hidden_size % 4 == 0 等对齐约束
### 调用示例
```Python
import torch
import mcoplib.op as op
device = "cuda"
# 1. 定义维度
batch_size = 1
seq_len = 128
hidden_size = 4096
epsilon = 1e-6
# 2. 准备输入数据
# input: [Batch, Seq, Hidden]
input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
# weight: [Hidden] (RMS Norm 的 Gamma 参数)
weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
# 3. 准备输出容器
# out: 根据函数名含有 "quant"，且是第一个参数，通常这里存放 Int8 结果
out = torch.empty(batch_size, seq_len, hidden_size, dtype=torch.int8, device=device)
# scales: 注释说是 [num_tokens]，即 [Batch * Seq] 或者 [Batch, Seq]
# 为了保险，我们创建与 Batch, Seq 对应的形状。Float32 是量化 scale 的标准类型。
scales = torch.empty(batch_size, seq_len, dtype=torch.float32, device=device)
# 4. 准备可选参数 (Optional Arguments)
# 如果不需要 residual 或 scale_ub，直接传 None
# 如果需要 residual，通常需要传入一个形状与 input 相同的 Tensor
residual = None 
# residual = torch.randn_like(input_tensor) # 如果需要测试 residual add
scale_ub = None
# 5. 调用算子
print("Calling rms_norm_dynamic_per_token_quant_custom...")
# 确保内存连续
if not input_tensor.is_contiguous(): input_tensor = input_tensor.contiguous()
if not weight.is_contiguous(): weight = weight.contiguous()
# 参数顺序必须严格对应 C++ 签名：
# (out, input, weight, scales, epsilon, scale_ub, residual)
op.rms_norm_dynamic_per_token_quant_custom(
    out,            # at::Tensor& out
    input_tensor,   # at::Tensor const& input
    weight,         # at::Tensor const& weight
    scales,         # at::Tensor& scales
    epsilon,        # double const var_epsilon
    scale_ub,       # c10::optional<at::Tensor> scale_ub
    residual        # c10::optional<at::Tensor> residual
)
torch.cuda.synchronize()
print("Call successful.")
print(f"Output (Int8) shape: {out.shape}")
print(f"Scales shape: {scales.shape}")
# 简单验证一下数据是否被写入 (不全为0)
print(f"Output max val: {out.max().item()}, Output min val: {out.min().item()}")
```
## 26. recv_from_attention_node_post_process
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def recv_from_attention_node_post_process(
    hidden_status: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    ori_index: torch.Tensor,
    new_index: torch.Tensor,
    deep_hidden_status: torch.Tensor,
    deeper_topk_weights: torch.Tensor,
    expert_cnt: torch.Tensor,
    valid_idx_size: torch.Tensor,
    begin_expert_id: int,
    num_local_experts: int,
    max_index_size: int,
    work_count: int
) -> None
```
### 功能描述
recv_from_attention_node_post_process 算子实现了从注意力节点接收数据后的后处理操作。该算子主要处理注意力机制输出的 topk 结果，包含索引和权重，通过索引映射更新隐藏状态和深层隐藏状态，并结合专家分配信息（起始专家 ID、本地专家数量等）完成专家相关的计数与权重处理，常用于混合专家模型 MoE 中注意力模块与专家模块的衔接流程
- 计算公式
$$
\text{new\_index}[i] = \text{ori\_index}[\text{topk\_ids}[i]] \quad (0 \leq i < \text{valid\_idx\_size})
$$
$$
\text{expert\_cnt}[e] = \text{expert\_cnt}[e] + \sum_{i=0}^{\text{valid\_idx\_size}-1} \mathbb{I}\left(
    \text{topk\_ids}[i] \in [\text{begin\_expert\_id} + e \cdot \text{max\_index\_size}, \text{begin\_expert\_id} + (e+1) \cdot \text{max\_index\_size} - 1]
\right)
$$

$$
\text{hidden\_status}[j] = \text{hidden\_status}[j] + \sum_{i=0}^{\text{valid\_idx\_size}-1} \text{topk\_weights}[i] \cdot \text{hidden\_status}[\text{new\_index}[i]]
$$

$$
\text{deeper\_topk\_weights}[i] = \text{deeper\_topk\_weights}[i] \cdot \text{topk\_weights}[i] \cdot \frac{1}{\text{work\_count}}
$$

$$
\text{deep\_hidden\_status}[j] = \text{deep\_hidden\_status}[j] + \sum_{i=0}^{\text{valid\_idx\_size}-1} \text{deeper\_topk\_weights}[i] \cdot \text{hidden\_status}[\text{new\_index}[i]]
$$
其中：
- $\text{hidden\_status} \in \mathbb{R}^{M \times D_h}$：隐藏状态张量，$M$ 为序列维度，$D_h$ 为隐藏层特征维度
- $\text{topk\_ids} \in \mathbb{Z}^{K \times 1}$：注意力机制输出的 topk 索引张量，$K$ 为 topk 候选数量
- $\text{topk\_weights} \in \mathbb{R}^{K \times 1}$：注意力机制输出的 topk 权重张量
- $\text{ori\_index} \in \mathbb{Z}^{T \times 1}$：原始索引映射张量，$T$ 为总索引数量
- $\text{new\_index} \in \mathbb{Z}^{K \times 1}$：映射后的新索引张量
- $\text{deep\_hidden\_status} \in \mathbb{R}^{M \times D_{dh}}$：深层隐藏状态张量，$D_{dh}$ 为深层隐藏层特征维度
- $\text{deeper\_topk\_weights} \in \mathbb{R}^{K \times 1}$：更深层 topk 权重张量
- $\text{expert\_cnt} \in \mathbb{Z}^{E \times 1}$：专家计数张量，$E$ 为本地专家总数（$E = \text{num\_local\_experts}$）
- $\text{valid\_idx\_size} \in \mathbb{Z}$：有效索引数量（标量），满足 $0 < \text{valid\_idx\_size} \leq \text{max\_index\_size}$
- $\text{begin\_expert\_id} \in \mathbb{Z}$：起始专家 ID（标量）
- $\text{num\_local\_experts} \in \mathbb{Z}$：本地专家数量（标量）
- $\text{max\_index\_size} \in \mathbb{Z}$：单专家最大索引容量（标量）
- $\text{work\_count} \in \mathbb{Z}$：工作任务计数（标量），用于权重归一化
- $\mathbb{I}(\cdot)$：指示函数，满足条件时取值为 1，否则为 0
- $e$：专家索引，满足 $0 \leq e < \text{num\_local\_experts}$
- $i$：索引遍历维度，满足 $0 \leq i < \text{valid\_idx\_size}$
- $j$：隐藏状态序列维度，满足 $0 \leq j < M$
补充约束公式：
$$
\text{valid\_idx\_size} \leq \text{max\_index\_size}, \quad \text{begin\_expert\_id} + \text{num\_local\_experts} \leq E_{\text{total}}
$$
其中 $E_{\text{total}}$ 为全局专家总数。
### 参数说明
- **hidden_status** (torch.Tensor, 入参/出参): 输入的隐藏状态张量，经后处理后会被更新
- **topk_ids** (torch.Tensor, 入参): 注意力机制输出的 topk 索引张量，存储选中的候选索引
- **topk_weights** (torch.Tensor, 入参): 注意力机制输出的 topk 权重张量，对应 topk_ids 的权重值
- **ori_index** (torch.Tensor, 入参): 原始索引张量，用于索引映射的原始标识
- **new_index** (torch.Tensor, 入参/出参): 新索引张量，存储映射后的索引，会被更新
- **deep_hidden_status** (torch.Tensor, 入参/出参): 深层隐藏状态张量，经后处理后会被更新
- **deeper_topk_weights** (torch.Tensor, 入参/出参): 更深层的 topk 权重张量，会被更新
- **expert_cnt** (torch.Tensor, 入参/出参): 专家计数张量，记录各专家的处理数量，会被更新
- **valid_idx_size** (torch.Tensor, 入参): 有效索引大小张量，存储有效索引的数量
- **begin_expert_id** (int, 入参): 起始专家 ID，标识当前处理的专家起始编号
- **num_local_experts** (int, 入参): 本地专家数量，当前节点包含的专家总数
- **max_index_size** (int, 入参): 最大索引大小，限制索引的最大长度
- **work_count** (int, 入参): 工作计数，标识当前处理的任务数量
### 返回值
无返回值，计算结果直接写入 hidden_status、new_index、deep_hidden_status、deeper_topk_weights、expert_cnt 等张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 支持的数据类型：float16、bfloat16、float32
- valid_idx_size 的值必须小于等于 max_index_size
- topk_ids 的范围必须在有效索引范围内
- begin_expert_id 与 num_local_experts 需匹配，确保专家 ID 不越界
- 所有张量需满足连续性要求，最后维度连续
### 调用示例
```Python
import torch
import mcoplib.op as op
device = "cuda"
# 定义核心参数
seq_dim = 1024             
hidden_dim = 4096      
topk_size = 64            
total_index_size = 8192     
num_local_experts = 8    
begin_expert_id = 0        
max_index_size = 1024    
work_count = 4            
valid_idx_size_val = 50     
# 创建输入/输出张量
hidden_status = torch.randn(seq_dim, hidden_dim, dtype=torch.float16, device=device)
topk_ids = torch.randint(0, total_index_size, (topk_size,), dtype=torch.int64, device=device)
topk_weights = torch.randn(topk_size, dtype=torch.float16, device=device)
ori_index = torch.randint(0, total_index_size, (total_index_size,), dtype=torch.int64, device=device)
new_index = torch.empty(topk_size, dtype=torch.int64, device=device)
deep_hidden_status = torch.randn(seq_dim, hidden_dim, dtype=torch.float16, device=device)
deeper_topk_weights = torch.randn(topk_size, dtype=torch.float16, device=device)
expert_cnt = torch.zeros(num_local_experts, dtype=torch.int64, device=device)
valid_idx_size = torch.tensor([valid_idx_size_val], dtype=torch.int64, device=device)
# 调用recv_from_attention_node_post_process算子
op.recv_from_attention_node_post_process(
    hidden_status,
    topk_ids,
    topk_weights,
    ori_index,
    new_index,
    deep_hidden_status,
    deeper_topk_weights,
    expert_cnt,
    valid_idx_size,
    begin_expert_id,
    num_local_experts,
    max_index_size,
    work_count
)
# 打印执行完成提示及关键张量信息
print("recv_from_attention_node_post_process computation completed")
print(f"Hidden status shape: {hidden_status.shape}")
print(f"New index shape: {new_index.shape}")
print(f"Expert count shape: {expert_cnt.shape}")
print(f"Valid index size: {valid_idx_size.item()}")
```
## 27. send_to_attention_node_pre_process
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def send_to_attention_node_pre_process(
    moe_hidden_status: torch.Tensor,
    deep_topk_weights: torch.Tensor,
    ori_index: torch.Tensor,
    new_index: torch.Tensor,
    output: torch.Tensor,
    valid_idx_size: torch.Tensor,
    max_index_size: const int
) -> None
```
### 功能描述
send_to_attention_node_pre_process 算子主要用于为将数据发送至注意力节点做预处理操作。该算子涉及混合专家模型的隐藏状态、深度 TopK 权重以及索引（原始索引与新索引）的处理，通过计算有效张量维度、验证数据类型等操作，为后续注意力节点的计算准备输入输出数据，确保数据格式与规模符合处理要求。
- 计算公式：
$$
\text{output}[i] = 
\begin{cases} 
\text{moe\_hidden\_status}[\text{new\_index}[i]] \times \text{deep\_topk\_weights}[\text{new\_index}[i]], & 0 \leq i < \text{valid\_idx\_size}[0] \text{ 且 } \text{new\_index}[i] < \text{max\_index\_size} \\
0, & \text{其他情况}
\end{cases}
$$
其中：
- $\text{moe\_hidden\_status} \in \mathbb{R}^{M \times D_h}$ 是混合专家模型的隐藏状态张量，$M$ 为隐藏状态的样本数量，$D_h$ 为隐藏状态的特征维度
- $\text{deep\_topk\_weights} \in \mathbb{R}^{M \times 1}$ 是深度 TopK 选择的权重张量，$M$ 与$\text{moe\_hidden\_status}$的样本数量一致
- $\text{ori\_index} \in \mathbb{Z}^{I}$ 是原始索引张量，$I$ 为索引总数，元素为非负整数，表征数据的原始位置
- $\text{new\_index} \in \mathbb{Z}^{I}$ 是新索引张量，维度与$\text{ori\_index}$完全一致，满足索引映射关系：
  $$
  \text{new\_index}[i] = \text{ori\_index}[i] - \min(\text{ori\_index}) \quad (0 \leq i < \text{valid\_idx\_size}[0])
  $$
- $\text{output} \in \mathbb{R}^{\text{max\_index\_size} \times D_h}$ 是预处理输出张量，第一维度由最大索引大小约束，第二维度与$\text{moe\_hidden\_status}$的特征维度一致
- $\text{valid\_idx\_size} \in \mathbb{Z}^{1}$ 是有效索引大小张量，仅包含一个标量元素（记为$V = \text{valid\_idx\_size}[0]$），表示参与有效计算的索引数量
- $\text{max\_index\_size} \in \mathbb{Z}^+$ 是最大索引大小，为正整数，用于约束索引的合法范围，满足：
  $$
  V \leq \text{max\_index\_size}
  $$
### 参数说明
- **moe_hidden_status** (at::Tensor, 入参): 混合专家模型的隐藏状态张量，作为预处理的输入数据之一
- **deep_topk_weights** (at::Tensor, 入参): 深度 TopK 选择的权重张量，参与预处理过程中的权重相关计算
- **ori_index** (at::Tensor, 入参): 原始索引张量，用于记录数据的原始位置信息
- **new_index** (at::Tensor, 入参): 新索引张量，用于记录数据经过处理后的新位置信息
- **output** (at::Tensor, 出参): 预处理后的输出张量，存储预处理的结果
- **valid_idx_size** (at::Tensor, 入参): 有效索引大小张量，记录有效索引的数量信息
- **max_index_size** (const int, 入参): 最大索引大小，用于约束索引的范围
### 返回值
无返回值，预处理结果直接写入 output 张量中
### 约束与调用
- 所有张量需位于 CUDA 设备上
- 仅支持可 blittable 类型的数据（ blittable 类型指可直接在设备间进行内存拷贝的数据类型）
- 张量的维度与规模需匹配计算需求，如 moe_hidden_status 的大小需与索引规模、权重规模相适配
- 输入张量需保证连续性或符合特定的内存布局要求（根据底层设备计算需求）
### 调用示例
```Python
import torch
import mcoplib.op as op
# 定义运行设备
device = "cuda"
# 定义算子所需的参数
num_local_experts = 8  # 专家数量
num_tokens = 128       # Token 数量
hidden_size = 4096     # Hidden Dimension
max_index_size = num_tokens * 2 # Grid size
# 1. 创建输入张量
# moe_hidden_status: [Experts, Tokens, Hidden]
# 必须是 BFloat16
moe_hidden_status = torch.randn(num_local_experts, num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
# deepep_topk_weights: [Experts * Tokens]
# 必须是 Float32
deepep_topk_weights = torch.randn(num_local_experts * num_tokens, dtype=torch.float32, device=device)
#  关键修正点开始 
# 构造 new_index: [max_index_size, 2]
# 第0列是 Expert Index，必须小于 num_local_experts (8)
# 第1列是 Token Index，必须小于 num_tokens (128)
new_index_col0 = torch.randint(0, num_local_experts, (max_index_size, 1), dtype=torch.int32, device=device)
new_index_col1 = torch.randint(0, num_tokens, (max_index_size, 1), dtype=torch.int32, device=device)
new_index = torch.cat([new_index_col0, new_index_col1], dim=1).contiguous()
# 构造 ori_index: [max_index_size, 2]
# Kernel 中使用了 ori_index[... + 1] 作为 output 的 row index
# 所以第1列必须小于 num_tokens (output 的第0维)
ori_index_col0 = torch.randint(0, num_local_experts, (max_index_size, 1), dtype=torch.int32, device=device)
ori_index_col1 = torch.randint(0, num_tokens, (max_index_size, 1), dtype=torch.int32, device=device)
ori_index = torch.cat([ori_index_col0, ori_index_col1], dim=1).contiguous()
#  关键修正点结束 
# valid_idx_size: 实际要处理的任务数量
# 必须 <= max_index_size
valid_idx_size = torch.tensor([num_tokens], dtype=torch.int32, device=device)
# 2. 创建输出张量
# output: [Tokens, Hidden]
output = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
# 同步以捕获之前的异步错误
torch.cuda.synchronize()
print("Calling send_to_attention_node_pre_process...")
# 3. 调用算子
op.send_to_attention_node_pre_process(
    moe_hidden_status,    # Input (BFloat16)
    deepep_topk_weights,  # Input (Float32)
    ori_index,            # Input (Int32)
    new_index,            # Input (Int32)
    output,               # Output (BFloat16)
    valid_idx_size,       # Input (Int32 scalar tensor)
    max_index_size        # Int scalar
)
# 再次同步确保完成
torch.cuda.synchronize()
print("send_to_attention_node_pre_process computation completed successfully.")
print(f"MoE Hidden Status shape: {moe_hidden_status.shape}")
print(f"Output shape: {output.shape}")
```
## 28. fused_silu_mul_dq_mask_quant
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_silu_mul_dq_mask_quant(
    out:torch.Tensor,
    input:torch.Tensor,
    mask:torch.Tensor
) -> None
```
### 功能描述
fused_silu_mul_dq_mask_quant 算子是一个针对 MoE 结构优化的高性能融合算子。该算子主要用于处理门控激活函数（SwiGLU）计算、专家掩码（Masking）应用以及动态量化和数据打包。算子接收包含 Gate 和 Up 投影的输入张量，计算 SiLU 激活与逐元素乘法，应用专家掩码将无效专家的输出置零，然后计算每行的最大绝对值以生成量化 Scale，最后将数据量化为 int8 格式并进行打包，以便后续的矩阵乘法（GEMM）计算。
- 计算公式：
$$
\text{temp\_h}[i] = (\text{SiLU}(\text{input}[i]_{0:D_h}) \odot \text{input}[i]_{D_h:2D_h}) \odot \text{mask}[i]
$$
$$
\text{scale}[i] = \frac{\max_{j}(|\text{temp\_h}[i, j]|)}{127.0}
$$
$$
\text{out}[i, j] = \text{Clamp}\left(\text{Round}\left(\frac{temp\_h[i, j]}{\text{scale}[i]}\right), -128, 127\right)
$$
其中：
  - $\text{input} \in \mathbb{R}^{N \times 2D_h}$ 是输入张量，$N$ 为 Token 数量，$2D_h$ 为两倍的隐藏层维度（包含 Gate 和 Up 两部分投影）
  - $\text{mask} \in \mathbb{R}^{N \times 1}$ 是专家掩码张量，用于指示 Token 是否激活对应专家，支持广播
  - $\text{temp\_h} \in \mathbb{R}^{N \times D_h}$ 是激活与掩码后的中间状态张量
  - $\text{scale} \in \mathbb{R}^{N}$ 是行级量化缩放因子（在内部计算，通常用于量化过程）
  - $\text{out} \in \mathbb{Z}^{N \times D_h}$ 是量化后的输出张量，数据类型为 int8
  - $\odot$ 表示逐元素乘法
  - $\text{SiLU}(x) = \frac{x}{1 + e^{-x}}$ 是激活函数
### 参数说明
  - **out** (at::Tensor, 出参): 输出张量，存储量化及打包后的结果，数据类型通常为 int8。
  - **input** (at::Tensor, 入参): 输入张量，形状通常为 [num_tokens, 2 * hidden_size]，包含 Gate 和 Up 的投影结果。
  - **mask** (at::Tensor, 入参): 掩码张量，用于指示 Token 与专家的对应关系，形状需与 batch/token 维度广播兼容。
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
  - 输入张量必须位于 CUDA 设备上。
  - input 的最后一维必须是 hidden\_size 的两倍（用于 SwiGLU split）。
  - out 张量需预先分配足够的内存空间以存储量化后的数据。
  - mask 的维度必须与 input 的 token 维度匹配或可广播。
  - 仅支持连续内存布局的张量。
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义张量维度
num_tokens = 128
hidden_size = 4096
input_dim = hidden_size * 2
# 创建输入张量
input_tensor = torch.randn(num_tokens, input_dim, dtype=torch.float16, device=device)
mask = torch.ones(num_tokens, 1, dtype=torch.float16, device=device)
out = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=device)
# 调用fused_silu_mul_dq_mask_quant算子
op.fused_silu_mul_dq_mask_quant(
    out,
    input_tensor,
    mask
)
print("fused_silu_mul_dq_mask_quant computation completed")
print(f"Output shape: {out.shape}")
```
## 29. fused_silu_mul_dq_reorder_quant
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_silu_mul_dq_reorder_quant(
    out: torch.Tensor,
    scale: torch.Tensor,
    input: torch.Tensor,
    reorder_topk_ids: torch.Tensor,
    w2_scale: torch.Tensor,
    start_expert_id: int64_t,
    end_expert_id: int64_t
) -> None
```
### 功能描述
fused_silu_mul_dq_reorder_quant 算子实现了 SiLU 激活函数、乘法、反量化（dequantization）、量化（quantization）以及基于 TopK 的重排序操作的融合。通过将多个操作融合为一个算子，减少了中间结果的内存读写次数，提升了计算效率，适用于需要高效处理量化模型中多步计算的场景（如 Transformer 模型的专家系统等）
- 计算公式：
  $$
  \text{out}[i] = \text{quant}\left( \text{mul}\left( \text{SiLU}\left( \text{dq}\left( \text{input}[\text{reorder\_topk\_ids}[i]] \right) \right), \text{w2\_scale} \right) \right), \quad \forall i \in [0, L)
  $$
  其中核心基础操作定义：
  $$
  \text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
  $$
  $$
  \text{dq}(x, s) = x \times s \quad (\text{反量化操作，其中缩放因子} \ s=\text{scale})
  $$
  $$
  \text{mul}(a, b) = a \times b \quad (\text{逐元素乘法操作})
  $$
  $$
  \text{quant}(y) = \text{clip}\left( \text{round}\left( \frac{y}{\text{quant\_scale}} \right), \text{quant\_min}, \text{quant\_max} \right) \quad (\text{量化操作})
  $$
  计算范围由专家 ID 限定（仅有效专家维度参与计算）：
  $$
  \text{input}_{\text{valid}} = \text{input}[..., e] \quad \text{其中} \ \text{start\_expert\_id} \leq e < \text{end\_expert\_id}
  $$
  变量说明：
  - $\text{input} \in \mathbb{R}^{B \times T \times D \times E}$：输入张量，其中$B$为批量大小，$T$为序列长度，$D$为特征维度，$E$为专家总数；
  - $\text{scale} \in \mathbb{R}^{D}$：反量化缩放因子张量，用于输入张量的反量化数值校准；
  - $\text{w2\_scale} \in \mathbb{R}^{D}$：权重缩放因子张量，参与乘法阶段的逐元素数值调整；
  - $\text{reorder\_topk\_ids} \in \mathbb{Z}^{L}$：TopK 重排序索引张量，$L$为输出张量总元素数，索引值范围为$[0, B \times T \times D \times E)$；
  - $\text{out} \in \mathbb{Z}^{L}$（或$\mathbb{R}^{L}$）：输出张量，存储融合操作的最终结果，数据类型由量化策略决定；
  - $\sigma(x) = \frac{1}{1 + e^{-x}}$：Sigmoid 激活函数，是 SiLU 操作的核心子函数；
  - $\text{quant\_scale} \in \mathbb{R}^{+}$：量化缩放因子，用于将浮点结果映射为量化值；
  - $\text{quant\_min} \in \mathbb{Z}$、$\text{quant\_max} \in \mathbb{Z}$：量化值的上下限，用于裁剪超出范围的量化结果；
  - $\text{start\_expert\_id} \in \mathbb{Z}$、$\text{end\_expert\_id} \in \mathbb{Z}$：专家范围边界，满足$0 \leq \text{start\_expert\_id} < \text{end\_expert\_id} \leq E$，$e$为专家维度的索引；
  - $i \in [0, L)$：输出张量的元素索引，遍历所有输出元素以完成逐元素计算；
  - $\text{round}(\cdot)$：四舍五入函数，将浮点值转换为整数；
  - $\text{clip}(x, \text{min}, \text{max})$：裁剪函数，将$x$限制在$[\text{min}, \text{max}]$范围内。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，用于存储融合操作的最终结果
- **scale** (torch.Tensor, 入参): 缩放因子张量，用于量化/反量化过程中的数值调整
- **input** (torch.Tensor, 入参): 输入张量，作为整个融合操作的原始输入数据
- **reorder_topk_ids** (torch.Tensor, 入参): 用于重排序的 TopK 索引张量，指定元素的重新排列顺序
- **w2_scale** (torch.Tensor, 入参): 权重缩放因子张量，参与乘法和量化过程的数值调整
- **start_expert_id** (int64_t, 入参): 起始专家 ID，用于限定处理的专家范围
- **end_expert_id** (int64_t, 入参): 结束专家 ID，与 start_expert_id 共同划定处理的专家范围
### 返回值
无返回值，计算结果直接写入 out 张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 支持的数据类型需匹配量化/反量化要求（如 float16、bfloat16、float32 等）
- reorder_topk_ids 的索引值需在有效范围内，且与输入张量的维度匹配
- start_expert_id 和 end_expert_id 需满足合法范围（0 ≤ start_expert_id < end_expert_id ≤ 专家总数）
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义张量维度
total_tokens = 1024
hidden_size = 4096
num_selected = 512
input_dim = hidden_size * 2  
start_expert = 0
end_expert = 4
# 创建输入张量
input_tensor = torch.randn(total_tokens, input_dim, dtype=torch.float32, device=device)
reorder_ids = torch.randint(0, total_tokens, (num_selected,), dtype=torch.long, device=device)
w2_scale = torch.ones(1, dtype=torch.float32, device=device)
out = torch.empty(num_selected, hidden_size, dtype=torch.int8, device=device)
scale = torch.empty(num_selected, 1, dtype=torch.float32, device=device)
# 调用fused_silu_mul_dq_reorder_quant算子
op.fused_silu_mul_dq_reorder_quant(
    out,
    scale,
    input_tensor,
    reorder_ids,
    w2_scale,
    start_expert,
    end_expert
)
print("fused_silu_mul_dq_reorder_quant computation completed")
print(f"Output shape: {out.shape}")
```
## 30. gptq_marlin_gemm_legacy
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def gptq_marlin_gemm_legacy(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    g_idx: torch.Tensor,
    perm: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    dtype: torch.dtype = torch.bfloat16,
    use_atomic_cache: bool = True
) -> torch.Tensor
```
### 功能描述
gptq_marlin_gemm_legacy 是 gptq_marlin_gemm 的旧版本实现，功能与前者一致，结合 GPTQ 量化的 Marlin 风格矩阵乘法，用于大语言模型量化推理的线性层加速，仅简化了参数，移除了 size_m_tensor 和 sms，适配旧版调用逻辑。
- 计算公式：
	定义核心变量：
	- 激活矩阵：$A \in \mathbb{R}^{\text{size\_m} \times \text{size\_k}}$
	- 量化权重分组块：$B_q^g \in \mathbb{Z}^{\text{group\_size} \times \text{size\_n}}$（$\text{group\_size}$为分组维度，由$\text{g\_idx}$决定）
	- 分组缩放因子：$S_g \in \mathbb{R}^{1 \times \text{size\_n}}$（第$g$个分组对应的缩放因子）
	量化权重还原，对每个权重分组$g$，将量化权重块乘以对应缩放因子，还原为浮点权重：
	$$
	B^g = B_q^g \cdot S_g
	$$
	其中$B^g \in \mathbb{R}^{\text{group\_size} \times \text{size\_n}}$，完整权重矩阵$B$是所有分组块的拼接：
	$$
	B = \left[ B^1; B^2; \dots; B^{\text{num\_groups}} \right] \in \mathbb{R}^{\text{size\_k} \times \text{size\_n}}
	$$
	矩阵乘法，激活矩阵与还原后权重矩阵执行矩阵乘法，得到输出：
	$$
	C = A \cdot B
	$$
	其中$C \in \mathbb{R}^{\text{size\_m} \times \text{size\_n}}$，即算子的返回张量。
### 参数说明
- **a** (torch.Tensor, 入参): 输入激活张量，形状为[size_m, size_k]，需位于 CUDA 设备且为连续张量
- **b_q_weight** (torch.Tensor, 入参): GPTQ 量化后的权重张量，形状需匹配量化 tile 尺寸要求，需位于 CUDA 设备且为连续张量
- **b_scales** (torch.Tensor, 入参): 量化权重对应的缩放因子张量，形状与分组量化维度匹配，需位于 CUDA 设备且为连续张量
- **g_idx** (torch.Tensor, 入参): 分组量化的索引张量，用于定位分组参数，需位于 CUDA 设备且为连续张量
- **perm** (torch.Tensor, 入参): 维度置换张量，用于维度调整，需位于 CUDA 设备且为连续张量
- **workspace** (torch.Tensor, 入参): 临时工作空间张量，需满足最小尺寸要求
- **num_bits** (int, 入参): 量化位数，仅支持 4 或 8 位
- **size_m** (int, 入参): 输入张量 a 的行数（矩阵乘左矩阵的行数）
- **size_n** (int, 入参): 输出张量的列数（矩阵乘右矩阵的列数）
- **size_k** (int, 入参): 矩阵乘的公共维度（左矩阵的列数、右矩阵的行数）
- **is_k_full** (bool, 入参): 标志 size_k 是否为量化 tile 尺寸的完整倍数
- **dtype** (torch.dtype, 入参, 默认 torch.bfloat16): 输出张量的数据类型，支持 bfloat16、float16
- **use_atomic_cache** (bool, 入参, 默认 True): 是否启用原子缓存优化
### 返回值
返回 torch.Tensor 类型的矩阵乘结果，形状为[size_m, size_n]，数据类型为指定的 dtype
### 约束与调用
与 gptq_marlin_gemm 的约束完全一致：
- 所有输入张量需位于 CUDA 设备且连续
- num_bits 仅支持 4 或 8 位
- 量化权重形状需匹配 tile 尺寸、workspace 满足最小尺寸
- 支持 bfloat16、float16 数据类型
### 调用示例
```python
import torch
import mcoplib.op as op
device = "cuda"
# 定义张量维度
size_m = 4096 
size_n = 4096  
size_k = 4096  
num_bits = 4  
is_k_full = True  
dtype = torch.bfloat16 
# 创建输入张量a
a = torch.randn(size_m, size_k, dtype=dtype, device=device).contiguous()
pack_factor = 32 // num_bits
b_q_weight = torch.randint(0, 2**num_bits, (size_n, size_k // pack_factor), dtype=torch.int32, device=device).contiguous()
group_size = 128  
b_scales = torch.randn(size_k // group_size, size_n, dtype=dtype, device=device).contiguous()
g_idx = torch.arange(0, size_k, group_size, dtype=torch.int32, device=device).repeat_interleave(group_size)
g_idx = g_idx[:size_k].contiguous() 
perm = torch.arange(size_k, dtype=torch.int32, device=device).contiguous() 
workspace_size = size_m * size_n // 1024
workspace = torch.empty(workspace_size, dtype=torch.int8, device=device).contiguous()
output = op.gptq_marlin_gemm_legacy(
    a=a,
    b_q_weight=b_q_weight,
    b_scales=b_scales,
    g_idx=g_idx,
    perm=perm,
    workspace=workspace,
    num_bits=num_bits,
    size_m=size_m,
    size_n=size_n,
    size_k=size_k,
    is_k_full=is_k_full,
    dtype=dtype,
    use_atomic_cache=True
)
print("gptq_marlin_gemm_legacy computation completed")
print(f"Output shape: {output.shape}") 
```
## 31. gptq_marlin_gemm
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def gptq_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    g_idx: torch.Tensor,
    perm: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    size_m_tensor: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
    sms: int,
    is_k_full: bool,
    dtype: torch.dtype = torch.bfloat16,
    use_atomic_cache: bool = True
) -> torch.Tensor
```
### 功能描述
gptq_marlin_gemm 算子实现了结合 GPTQ 量化的 Marlin 风格高效矩阵乘法，用于大语言模型量化推理中的线性层加速。该算子针对 GPTQ 量化后的权重 b_q_weight，配合缩放因子、分组索引等量化参数，通过 Marlin 的 CUDA 优化内核完成矩阵乘计算，同时支持原子缓存等策略提升计算性能
- 计算公式：
	定义核心变量：
	- 激活矩阵：$A \in \mathbb{R}^{\text{size\_m} \times \text{size\_k}}$
	- 量化权重分组块：$B_q^g \in \mathbb{Z}^{\text{group\_size} \times \text{size\_n}}$（$\text{group\_size}$为分组维度，由$\text{g\_idx}$决定）
	- 分组缩放因子：$S_g \in \mathbb{R}^{1 \times \text{size\_n}}$（第$g$个分组对应的缩放因子）
	量化权重还原，对每个权重分组$g$，将量化权重块乘以对应缩放因子，还原为浮点权重：
	$$
	B^g = B_q^g \cdot S_g
	$$
	其中$B^g \in \mathbb{R}^{\text{group\_size} \times \text{size\_n}}$，完整权重矩阵$B$是所有分组块的拼接：
	$$
	B = \left[ B^1; B^2; \dots; B^{\text{num\_groups}} \right] \in \mathbb{R}^{\text{size\_k} \times \text{size\_n}}
	$$
	矩阵乘法，激活矩阵与还原后权重矩阵执行矩阵乘法，得到输出：
	$$
	C = A \cdot B
	$$
	其中$C \in \mathbb{R}^{\text{size\_m} \times \text{size\_n}}$，即算子的返回张量。
### 参数说明
- **a** (torch.Tensor, 入参): 输入激活张量，形状为[size_m, size_k]，需位于 CUDA 设备且为连续张量
- **b_q_weight** (torch.Tensor, 入参): GPTQ 量化后的权重张量，形状需匹配量化 tile 尺寸要求，需位于 CUDA 设备且为连续张量
- **b_scales** (torch.Tensor, 入参): 量化权重对应的缩放因子张量，形状与分组量化维度匹配，需位于 CUDA 设备且为连续张量
- **g_idx** (torch.Tensor, 入参): 分组量化的索引张量，用于定位分组对应的量化参数，需位于 CUDA 设备且为连续张量
- **perm** (torch.Tensor, 入参): 维度置换张量，用于权重/激活的维度调整，需位于 CUDA 设备且为连续张量
- **workspace** (torch.Tensor, 入参): 临时工作空间张量，用于内核计算的临时存储，需满足最小尺寸要求
- **num_bits** (int, 入参): 量化位数，仅支持 4 或 8 位
- **size_m_tensor** (torch.Tensor, 入参): size_m 对应的张量形式，用于 CUDA 内核的参数传递
- **size_m** (int, 入参): 输入张量 a 的行数（矩阵乘左矩阵的行数）
- **size_n** (int, 入参): 输出张量的列数（矩阵乘右矩阵的列数）
- **size_k** (int, 入参): 矩阵乘的公共维度（左矩阵的列数、右矩阵的行数）
- **sms** (int, 入参): 用于 CUDA 内核计算的流多处理器（SM）数量
- **is_k_full** (bool, 入参): 标志 size_k 是否为量化 tile 尺寸的完整倍数
- **dtype** (torch.dtype, 入参, 默认 torch.bfloat16): 输出张量的数据类型，支持 bfloat16、float16
- **use_atomic_cache** (bool, 入参, 默认 True): 是否启用原子缓存优化，提升内核计算效率
### 返回值
返回 torch.Tensor 类型的矩阵乘结果，形状为[size_m, size_n]，数据类型为指定的 dtype
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上，且为连续张量
- num_bits 仅支持 4 或 8 位量化
- size_m 需满足小于 FULL_M_BLOCK 或能被 FULL_M_BLOCK 整除
- 量化权重 b_q_weight 的形状需与 GPTQ 量化的 tile 尺寸匹配（如能被 tile_size 整除）
- workspace 张量的尺寸需不小于要求的最小工作空间大小
- 支持的数据类型：bfloat16、float16（half）
### 调用示例
```python
import torch
import mcoplib.op as op
#设置设备
device = "cuda"
# 定义张量维度
size_m = 4096  
size_n = 4096  
size_k = 4096  
num_bits = 4 
sms = -1      
is_k_full = True 
dtype = torch.bfloat16  
# 创建输入张量a
a = torch.randn(size_m, size_k, dtype=dtype, device=device).contiguous()
pack_factor = 32 // num_bits
b_q_weight = torch.randint(0, 2**num_bits, (size_n, size_k // pack_factor), dtype=torch.int32, device=device).contiguous()
group_size = 128  
b_scales = torch.randn(size_k // group_size, size_n, dtype=dtype, device=device).contiguous()
g_idx = torch.arange(0, size_k, group_size, dtype=torch.int32, device=device).repeat_interleave(group_size)
g_idx = g_idx[:size_k].contiguous()
perm = torch.arange(size_k, dtype=torch.int32, device=device).contiguous()
workspace_size = size_m * size_n // 1024  
workspace = torch.empty(workspace_size, dtype=torch.int8, device=device).contiguous()
size_m_tensor = torch.tensor([size_m], dtype=torch.int32, device=device)
# 调用gptq_marlin_gemm算子
output = op.gptq_marlin_gemm(
    a=a,
    b_q_weight=b_q_weight,
    b_scales=b_scales,
    g_idx=g_idx,
    perm=perm,
    workspace=workspace,
    num_bits=num_bits,
    size_m_tensor=size_m_tensor,
    size_m=size_m,
    size_n=size_n,
    size_k=size_k,
    sms=sms,
    is_k_full=is_k_full,
    dtype=dtype,
    use_atomic_cache=True
)
print("gptq_marlin_gemm computation completed")
print(f"Output shape: {output.shape}")
```
## 33. get_cuda_view_from_cpu_tensor
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);
```
### 功能描述
get_cuda_view_from_cpu_tensor 算子实现了一种基于统一虚拟寻址（UVA, Unified Virtual Addressing）的零拷贝机制。该算子通过获取 CPU 端锁页内存（Pinned Memory）的物理地址，调用 CUDA Runtime API 获取对应的设备端
指针，并在不进行显式数据拷贝（D2H 或 H2D）的情况下，构建一个共享同一块物理内存的 CUDA Tensor。
此实现主要用于优化 CPU 与 GPU 之间的大内存频繁交互场景，消除 PCIe 传输开销，使 GPU 能直接通过 PCIe 总线访问主机内存。
- 计算公式/逻辑：
	算子首先验证输入张量是否位于 CPU 且使用了锁页内存，调用 `cudaHostGetDevicePointer` 获取设备端映射指针，使用 `torch::from_blob` 构造 CUDA 张量，设置 Deleter 为空操作（No-op），确保内存生命周期仍由 CPU 张量管理。
    $$
    P_{device} \leftarrow \text{cudaHostGetDevicePointer}(P_{host})
    $$
    $$
    \text{Tensor}_{cuda} \leftarrow \text{View}(P_{device}, \text{Size}, \text{Stride})
    $$
**主要功能包括：**
- 零拷贝映射：直接将 CPU 内存映射到 GPU 地址空间，无需数据搬运。
- 生命周期管理：返回的 CUDA Tensor 不拥有内存，依赖原始 CPU Tensor 存活。
- 属性保留：严格保留原 Tensor 的 Shape、Stride 和 Dtype。
- 安全检查：内置 `TORCH_CHECK` 确保输入在 CPU 且 CUDA 环境正常。
### 参数说明
- **cpu_tensor** (torch.Tensor, 入参): 输入张量。必须位于 CPU 设备，且必须通过 `. pin_memory ()` 分配或已位于锁页内存中。
### 返回值
- **torch::Tensor**: 返回一个新的张量，该张量位于 CUDA 设备上，其数据指针指向输入的 CPU 锁页内存。
### 约束与调用
- 输入张量必须在 CPU 上 (`device='cpu'`)。
- 输入张量必须处于锁页内存（Pinned Memory），否则运行时会抛出 CUDA 错误。
- 必须确保 `cpu_tensor` 在 `cuda_tensor` 使用期间不被释放。
- 仅支持支持 UVA 的系统环境。
- 这是一个同步调用，包含 CUDA API 调用。
### 调用示例
```Python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cpu"
DTYPE = torch.float32
# 维度设置
SIZE = 1024 * 1024
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
# 注意：输入必须在 CPU 上，故默认设备设为 CPU，后续通过 pin_memory 处理
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# 必须使用 pin_memory() 将内存锁定，否则无法获取 UVA 设备指针
cpu_tensor = torch.randn(SIZE, dtype=DTYPE).pin_memory()
# 其他参数
# 此算子无其他配置参数
# ================= 算子执行 =================
print(f"Running get_cuda_view_from_cpu_tensor...")
print(f"Input CPU Tensor: {cpu_tensor.shape}, Device: {cpu_tensor.device}, Pinned: {cpu_tensor.is_pinned()}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 假设算子已绑定到 mcoplib._C 或 torch.ops._C
cuda_view = torch.ops._C.get_cuda_view_from_cpu_tensor(
    cpu_tensor
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Device: {cuda_view.device}")
print(f"Output Mean: {cuda_view.mean().item():.4f}")
# 验证零拷贝特性：修改 GPU 视图，检查 CPU 原值
cuda_view[0] = 1234.5678
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"Modified CPU Value[0]: {cpu_tensor[0].item():.4f}")
```
## 34. paged_attention_v1
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int,
    blocksparse_local_blocks: int,
    blocksparse_vert_stride: int,
    blocksparse_block_size: int,
    blocksparse_head_sliding_step: int
) -> None:
    pass
```
### 功能描述
`paged_attention_v1` 算子实现分页注意力机制（PagedAttention）的第一版本。该算子是 vLLM 中用于高效执行注意力计算的核心 CUDA 内核启动器。它通过 `paged_attention_v1_launcher` 函数，根据查询（Query）张量、存储在分页缓存（Paged KV Cache）中的键（Key）和值（Value）张量来计算注意力输出。
此实现通过使用 `block_tables`（块表）来管理非连续的内存块，从而极大地提高了内存利用率，减少了碎片。该算子设计为支持多种注意力变体，包括多头注意力 (MHA)、组查询注意力 (GQA) 和多查询注意力 (MQA)。
- 计算公式：
    标准的缩放点积注意力公式：
    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + A\right)V
    $$
    其中：
    - $Q \in \mathbb{R}^{N \times H \times D}$ 是查询张量 (`query`)
    - $K \in \mathbb{R}^{B \times H_k \times D \times S}$ 是分页键缓存 (`key_cache`) 的逻辑视图
    - $V \in \mathbb{R}^{B \times H_k \times D \times S}$ 是分页值缓存 (`value_cache`) 的逻辑视图
    - $A$ 是 ALiBi 位置编码偏置 (`alibi_slopes`)，这是一个可选输入
    - $\frac{1}{\sqrt{d_k}}$ 是缩放因子 (`scale`)
    - $N$ 是批次中的序列数 (`num_seqs`)
    - $H$ 是查询头数 (`num_heads`)
    - $H_k$ 是键值头数 (`num_kv_heads`)
    - $D$ 是头维度 (`head_size`)
    - $S$ 是块大小 (`block_size`)
    - $B$ 是物理缓存中的总块数 (`num_blocks`)
- 分页注意力机制通过块表访问：
    算子使用 block_tables 张量将逻辑上的 token 索引映射到 key_cache 和 value_cache 中的物理存储块。对于序列 $q$ 中的第 $l$ 个 token，其对应的键向量 $K_l$ 通过以下方式定位：
    $$
    \begin{aligned} \text{block\_idx} &= \text{block\_tables}[q, \lfloor l / S \rfloor] \\ \text{offset} &= l \bmod S \\ K_l &= \text{key\_cache}[\text{block\_idx}, \dots, \text{offset}] \end{aligned}
    $$
    其中 `...` 代表头和维度的索引。
- 对于 GQA / MQA 配置：
    算子通过 num_heads ($H$) 和 num_kv_heads ($H_k$) 两个不同的参数来原生支持 GQA 和 MQA。查询头 $h_q$ 会被映射到对应的键值头 $h_k$：
    $$
    \begin{aligned} G &= H / H_k \quad (\text{其中 } G \text{ 是组大小}) \\ h_k &= \lfloor h_q / G \rfloor \end{aligned}
    $$
**主要功能包括**：
- 基于查询向量和分页缓存中的键值对，高效计算注意力输出。
- 通过 `block_tables` 支持分页（Paged）键值缓存，极大提升内存利用效率。
- 支持 ALiBi 位置编码（通过 `alibi_slopes` 张量）。
- 支持多种数据类型（如 FP16, BF16）和量化格式（如 FP8 KV 缓存，通过 `kv_cache_dtype` 和 `k_scale`, `v_scale` 参数）。
- 支持块稀疏注意力（Block-Sparse Attention）模式（通过 `IS_BLOCK_SPARSE` 模板参数和 `blocksparse_*` 运行时参数）。
- 通过 `switch (head_size)` 语句和宏 (`LAUNCH_PAGED_ATTENTION_V1`)，为不同 `head_size` 提供了编译时特化内核。
- 通过模板和宏 (`CALL_V1_LAUNCHER_BLOCK_SIZE`) 分发不同 `block_size` 的内核实现。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[num_seqs, num_heads, head_size]`，存储注意力计算结果
- **query** (torch.Tensor, 入参): 查询张量，形状为 `[num_seqs, num_heads, head_size]`
- **key_cache** (torch.Tensor, 入参): 键缓存，形状为 `[num_blocks, num_kv_heads, head_size/x, block_size, x]` (x 取决于 `kv_cache_dtype`，例如 FP8 packed layout)
- **value_cache** (torch.Tensor, 入参): 值缓存，形状为 `[num_blocks, num_kv_heads, head_size, block_size]`
- **num_kv_heads** (int, 入参): 键值头数量，用于 GQA 和 MQA 配置
- **scale** (float, 入参): 注意力缩放因子，通常为 1.0/sqrt (head_size)
- **block_tables** (torch.Tensor, 入参): 块表，形状为 `[num_seqs, max_num_blocks_per_seq]`
- **seq_lens** (torch.Tensor, 入参): 序列长度张量，形状为 `[num_seqs]`
- **block_size** (int, 入参): 每个块的大小
- **max_seq_len** (int, 入参): 最大序列长度
- **alibi_slopes** (Optional[torch.Tensor], 入参): ALiBi 位置编码斜率，可选参数
- **kv_cache_dtype** (str, 入参): KV 缓存数据类型，支持 "auto", "fp8", "fp8_e4m3", "fp8_e5m2"
- **k_scale** (torch.Tensor, 入参): 键缩放因子，用于量化场景
- **v_scale** (torch.Tensor, 入参): 值缩放因子，用于量化场景
- **tp_rank** (int, 入参): 张量并行排名
- **blocksparse_local_blocks** (int, 入参): 块稀疏局部块数
- **blocksparse_vert_stride** (int, 入参): 块稀疏垂直步长
- **blocksparse_block_size** (int, 入参): 块稀疏块大小
- **blocksparse_head_sliding_step** (int, 入参): 块稀疏头滑动步长
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- Head Size 限制：必须为 32, 64, 80, 96, 112, 120, 128, 192 或 256 之一
- Block Size 限制：必须为 8, 16 或 32 之一
- query 和 key/value 的 head_size 必须匹配
- num_kv_heads 必须能够整除 num_heads（用于 GQA/MQA）
- block_tables 和 seq_lens 的 batch 维度必须一致
- 支持的数据类型：float16, bfloat16, float32（代码中亦包含对 FP8 KV Cache 及 scale 参数的支持逻辑）
- 内存布局要求：输入张量必须是连续存储的
### 调用示例
```python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
NUM_SEQS = 7
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_QUERY_HEADS = 32
NUM_BLOCKS = 128
MAX_SEQ_LEN = 256
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
scale = 1.0 / (HEAD_SIZE ** 0.5)
query = torch.randn(NUM_SEQS, NUM_QUERY_HEADS, HEAD_SIZE, dtype=DTYPE)
# Cache 结构
x = 16
key_cache = torch.randn(NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE // x, BLOCK_SIZE, x, dtype=DTYPE)
value_cache = torch.randn(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=DTYPE)
# 序列长度
seq_lens_list = [random.randint(1, MAX_SEQ_LEN) for _ in range(NUM_SEQS)]
seq_lens_list[-1] = MAX_SEQ_LEN
seq_lens = torch.tensor(seq_lens_list, dtype=torch.int)
max_seq_len_actual = max(seq_lens_list)
# Block Tables
max_num_blocks_per_seq = (max_seq_len_actual + BLOCK_SIZE - 1) // BLOCK_SIZE
block_tables = torch.randint(0, NUM_BLOCKS, (NUM_SEQS, max_num_blocks_per_seq), dtype=torch.int)
# 其他参数
output = torch.empty_like(query)
alibi_slopes = None
kv_cache_dtype = "auto"
k_scale = torch.tensor(1.0, dtype=torch.float32)
v_scale = torch.tensor(1.0, dtype=torch.float32)
# 稀疏/TP 参数
tp_rank = 0
blocksparse_local_blocks = 0
blocksparse_vert_stride = 0
blocksparse_block_size = 0
blocksparse_head_sliding_step = 0
# ================= 算子执行 =================
print(f"Running paged_attention_v1...")
print(f"Query: {query.shape}, Key: {key_cache.shape}, Value: {value_cache.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.paged_attention_v1(
    output,
    query,
    key_cache,
    value_cache,
    NUM_KV_HEADS,
    scale,
    block_tables,
    seq_lens,
    BLOCK_SIZE,
    max_seq_len_actual,
    alibi_slopes,
    kv_cache_dtype,
    k_scale,
    v_scale,
    tp_rank,
    blocksparse_local_blocks,
    blocksparse_vert_stride,
    blocksparse_block_size,
    blocksparse_head_sliding_step
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output.mean().item():.4f}")
```
## 35. paged_attention_v2
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def paged_attention_v2(
    out: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int,
    blocksparse_local_blocks: int,
    blocksparse_vert_stride: int,
    blocksparse_block_size: int,
    blocksparse_head_sliding_step: int
) -> None:
    pass
```
### 功能描述
`paged_attention_v2` 算子实现分页注意力机制（PagedAttention）的第二版本，针对长序列场景进行了并行优化。该算子是 vLLM 中用于高效执行注意力计算的核心 CUDA 内核启动器。它通过 `paged_attention_v2_launcher` 函数启动计算内核，并结合 `paged_attention_v2_reduce_kernel` 进行结果归约。
与 V1 不同，V2 版本采用了“分块并行-归约”（Map-Reduce）的策略，将长序列在序列维度上切分为多个分区（partitions），并行计算每个分区的注意力分数，最后通过归约内核合并结果。此外，该算子还支持块稀疏注意力（Block Sparse Attention）和 FP8 KV 缓存，进一步提升了在不同硬件配置下的计算效率。
- 计算公式：
    标准的缩放点积注意力公式（V2 版本引入了并行归约的中间变量）：
    $$
    \text{Attention}(Q, K, V) = \text{Reduce}\left(\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + A\right)V\right)
    $$
    其中：
    - $Q \in \mathbb{R}^{N \times H \times D}$ 是查询张量 (`query`)
    - $K, V$ 是存储在分页缓存中的键值张量，支持 FP8 数据类型 (`KV_DTYPE`)
    - $O_{\text{tmp}} \in \mathbb{R}^{N \times H \times P \times D}$ 是临时输出张量 (`tmp_out`)，其中 $P$ 是分区数 (`max_num_partitions`)
    - $L \in \mathbb{R}^{N \times H \times P}$ 是每个分区的指数和 (`exp_sums`)
    - $M \in \mathbb{R}^{N \times H \times P}$ 是每个分区的最大逻辑值 (`max_logits`)
    - $A$ 是 ALiBi 位置编码偏置 (`alibi_slopes`)，这是一个可选输入
    - $\frac{1}{\sqrt{d_k}}$ 是缩放因子 (`scale`)
- 分页注意力机制通过块表访问：
    算子使用 block_tables 张量将逻辑上的 token 索引映射到 key_cache 和 value_cache 中的物理存储块。对于序列 $q$ 中的第 $l$ 个 token，其物理位置计算如下：
    $$
    \begin{aligned} \text{block\_idx} &= \text{block\_tables}[q, \lfloor l / S \rfloor] \\ \text{offset} &= l \bmod S \\ K_l &= \text{key\_cache}[\text{block\_idx}, \dots, \text{offset}] \end{aligned}
    $$
    其中 $S$ 是块大小 (block_size)，... 代表头和维度的索引。
- 并行计算与稀疏支持：
    为了处理长上下文，V2 算子将序列长度划分为大小为 PARTITION_SIZE 的多个分区。计算过程分为两个阶段：
    1. 计算阶段：各 CUDA block 并行计算部分注意力结果，并将中间统计量写入 `exp_sums` 和 `max_logits`。
    2. 归约阶段：调用 reduce_kernel 根据中间统计量合并最终输出 out。
        此外，算子支持块稀疏性 (is_block_sparse)，通过 blocksparse_local_blocks 等参数控制局部和跨步的稀疏模式。
- 对于 GQA / MQA 配置：
    算子通过 num_heads ($H$) 和 num_kv_heads ($H_k$) 参数原生支持 GQA 和 MQA。查询头 $h_q$ 映射到键值头 $h_k$ 的逻辑为：
    $$
    \begin{aligned} G &= H / H_k \quad (\text{其中 } G \text{ 是组大小}) \\ h_k &= \lfloor h_q / G \rfloor \end{aligned}
    $$
**主要功能包括：**
- 基于查询向量和分页缓存中的键值对，利用分块并行（Partitioning）策略高效计算注意力输出。
- 通过 `block_tables` 支持分页（Paged）键值缓存，极大提升内存利用效率。
- 支持 ALiBi 位置编码（通过 `alibi_slopes` 张量）。
- 支持多种数据类型（如 FP16, BF16）和量化格式（如 FP8 KV 缓存，通过 `kv_cache_dtype` 和 `k_scale`, `v_scale` 参数）。
- 支持块稀疏注意力（Block-Sparse Attention）模式（通过 `IS_BLOCK_SPARSE` 模板参数和 `blocksparse_*` 运行时参数）。
- 通过 `switch (head_size)` 语句和宏 (`LAUNCH_PAGED_ATTENTION_V2`)，为不同 `head_size` 提供了编译时特化内核。
- 通过模板和宏 (`CALL_V2_LAUNCHER_BLOCK_SIZE`) 分发不同 `block_size` 的内核实现。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[num_seqs, num_heads, head_size]`，存储注意力计算最终结果
- **exp_sums** (torch.Tensor, 出参): 指数和中间张量，形状为 `[num_seqs, num_heads, max_num_partitions]`，用于 Reduce 操作
- **max_logits** (torch.Tensor, 出参): 最大 Logits 中间张量，形状为 `[num_seqs, num_heads, max_num_partitions]`，用于 Reduce 操作
- **tmp_out** (torch.Tensor, 出参): 临时输出张量，形状为 `[num_seqs, num_heads, max_num_partitions, head_size]`，存储各分区的中间计算结果
- **query** (torch.Tensor, 入参): 查询张量，形状为 `[num_seqs, num_heads, head_size]`
- **key_cache** (torch.Tensor, 入参): 键缓存，形状为 `[num_blocks, num_heads, head_size/x, block_size, x]` (x 取决于 `kv_cache_dtype`，例如 FP8 packed layout)
- **value_cache** (torch.Tensor, 入参): 值缓存，形状为 `[num_blocks, num_heads, head_size, block_size]`
- **num_kv_heads** (int, 入参): 键值头数量，用于 GQA 和 MQA 配置
- **scale** (float, 入参): 注意力缩放因子，通常为 1.0/sqrt (head_size)
- **block_tables** (torch.Tensor, 入参): 块表，形状为 `[num_seqs, max_num_blocks_per_seq]`
- **seq_lens** (torch.Tensor, 入参): 序列长度张量，形状为 `[num_seqs]`
- **block_size** (int, 入参): 每个块的大小
- **max_seq_len** (int, 入参): 最大序列长度
- **alibi_slopes** (Optional[torch.Tensor], 入参): ALiBi 位置编码斜率，可选参数
- **kv_cache_dtype** (str, 入参): KV 缓存数据类型，支持 "auto", "fp8", "fp8_e4m3", "fp8_e5m2"
- **k_scale** (torch.Tensor, 入参): 键缩放因子，用于量化场景
- **v_scale** (torch.Tensor, 入参): 值缩放因子，用于量化场景
- **tp_rank** (int, 入参): 张量并行排名
- **blocksparse_local_blocks** (int, 入参): 块稀疏局部块数
- **blocksparse_vert_stride** (int, 入参): 块稀疏垂直步长
- **blocksparse_block_size** (int, 入参): 块稀疏块大小
- **blocksparse_head_sliding_step** (int, 入参): 块稀疏头滑动步长
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- Head Size 限制：必须为 32, 64, 80, 96, 112, 120, 128, 192 或 256 之
- Block Size 限制：必须为 8, 16 或 32 之一
- query 和 key/value 的 head_size 必须匹配
- num_kv_heads 必须能够整除 num_heads（用于 GQA/MQA）
- block_tables 和 seq_lens 的 batch 维度必须一致
- 中间张量 (`exp_sums`, `max_logits`, `tmp_out`) 的 partition 维度大小必须满足 `max_num_partitions` 的计算要求（通常基于 `PARTITION_SIZE=512`）
- 支持的数据类型：float16, bfloat16, float32（代码中亦包含对 FP8 KV Cache 及 scale 参数的支持逻辑）
- 内存布局要求：输入张量必须是连续存储的
### 调用示例
```python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
NUM_SEQS = 7
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_QUERY_HEADS = 32
NUM_BLOCKS = 128
MAX_SEQ_LEN = 256
PARTITION_SIZE = 512
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
scale = 1.0 / (HEAD_SIZE ** 0.5)
query = torch.randn(NUM_SEQS, NUM_QUERY_HEADS, HEAD_SIZE, dtype=DTYPE)
# Cache 结构
x = 16
key_cache = torch.randn(NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE // x, BLOCK_SIZE, x, dtype=DTYPE)
value_cache = torch.randn(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=DTYPE)
# 序列长度
seq_lens_list = [random.randint(1, MAX_SEQ_LEN) for _ in range(NUM_SEQS)]
seq_lens_list[-1] = MAX_SEQ_LEN
seq_lens = torch.tensor(seq_lens_list, dtype=torch.int)
max_seq_len_actual = max(seq_lens_list)
# Block Tables
max_num_blocks_per_seq = (max_seq_len_actual + BLOCK_SIZE - 1) // BLOCK_SIZE
block_tables = torch.randint(0, NUM_BLOCKS, (NUM_SEQS, max_num_blocks_per_seq), dtype=torch.int)
# V2 中间 Tensor
max_num_partitions = (max_seq_len_actual + PARTITION_SIZE - 1) // PARTITION_SIZE
output = torch.empty_like(query)
exp_sums = torch.empty(NUM_SEQS, NUM_QUERY_HEADS, max_num_partitions, dtype=torch.float32)
max_logits = torch.empty(NUM_SEQS, NUM_QUERY_HEADS, max_num_partitions, dtype=torch.float32)
tmp_out = torch.empty(NUM_SEQS, NUM_QUERY_HEADS, max_num_partitions, HEAD_SIZE, dtype=DTYPE)
# 其他参数
alibi_slopes = None
kv_cache_dtype = "auto"
k_scale = torch.tensor(1.0, dtype=torch.float32)
v_scale = torch.tensor(1.0, dtype=torch.float32)
# 稀疏/TP 参数
tp_rank = 0
blocksparse_local_blocks = 0
blocksparse_vert_stride = 0
blocksparse_block_size = 0
blocksparse_head_sliding_step = 0
# ================= 算子执行 =================
print(f"Running paged_attention_v2...")
print(f"Query: {query.shape}, Key: {key_cache.shape}, Value: {value_cache.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.paged_attention_v2(
    output,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    NUM_KV_HEADS,
    scale,
    block_tables,
    seq_lens,
    BLOCK_SIZE,
    max_seq_len_actual,
    alibi_slopes,
    kv_cache_dtype,
    k_scale,
    v_scale,
    tp_rank,
    blocksparse_local_blocks,
    blocksparse_vert_stride,
    blocksparse_block_size,
    blocksparse_head_sliding_step
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output.mean().item():.4f}")
```
## 36. merge_attn_states
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def merge_attn_states(
    output: torch.Tensor,
    output_lse: Optional[torch.Tensor],
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor
) -> None:
    pass
```
### 功能描述
merge_attn_states 算子实现了 Split-KV（KV 分离）场景下的注意力结果合并逻辑，具体参考了 arXiv: 2501.01005 论文的第 2.2 节。在分块预填充（Chunked Prefill）或混合注意力机制中，该算子用于将前缀（prefix）和后缀（suffix）的局部注意力结果基于其 Log-Sum-Exp (LSE) 值进行加权归约，生成最终的注意力输出。
该算子通过 CUDA 核心 `merge_attn_states_kernel` 并行执行，利用 128 位向量化加载（Vectorized Load）技术优化内存访问效率。
- 计算公式：
    算子通过比较前缀和后缀的 LSE 值来计算归一化权重，并据此融合输出状态。
    设前缀 LSE 为 $L_p$，后缀 LSE 为 $L_s$，计算过程如下：
    1. 计算最大 LSE 用于数值稳定性：
        $$
        m = \max(L_p, L_s)
        $$
    2. 计算指数权重（Softmax 分母部分）：
        $$
        w_p = \exp(L_p - m), \quad w_s = \exp(L_s - m)
        $$
        $$
        w_{out} = w_p + w_s
        $$
    3. 计算加权输出：
        $$
        O_{out} = \frac{O_p \cdot w_p + O_s \cdot w_s}{w_{out}}
        $$
    4. 更新最终 LSE：
        $$
        L_{out} = \log(w_{out}) + m
        $$
- 异常值处理：
    算子内置了针对 Split-KV 边缘情况的保护逻辑。例如，当分块请求没有命中的前缀时（$L_p = L_s = -\infty$），算子会直接传递前缀输出（通常为全 0）和前缀 LSE，避免计算过程中出现 NaN。
**主要功能包括：**
- 基于 softmax 归一化因子，精确合并两组注意力输出 ($O$) 和其对应的对数和 ($LSE$)。
- 通过减去最大值 ($max\_lse$) 的方式进行 LogSumExp 计算，防止浮点溢出。
- 使用 `uint4` 类型进行 128-bit 的打包读写，显著提升显存带宽利用率。
- 支持 FP16/BF16 输入，但在内部累加和计算时强制转换为 FP32 以保证精度。
### 参数说明
- **output** (torch.Tensor, 出参): 合并后的注意力输出张量，形状为 `[num_tokens, num_heads, head_size]`。
- **output_lse** (Optional[torch.Tensor], 出参): 可选，合并后的 Log-Sum-Exp 张量，形状为 `[num_heads, num_tokens]`。
- **prefix_output** (torch.Tensor, 入参): 前缀部分的注意力输出，形状为 `[num_tokens, num_heads, head_size]`。
- **prefix_lse** (torch.Tensor, 入参): 前缀部分的 LSE 值，形状为 `[num_heads, num_tokens]`。
- **suffix_output** (torch.Tensor, 入参): 后缀部分的注意力输出，形状为 `[num_tokens, num_heads, head_size]`。
- **suffix_lse** (torch.Tensor, 入参): 后缀部分的 LSE 值，形状为 `[num_heads, num_tokens]`。
### 返回值
无返回值，计算结果直接写入输出张量 `output` 和 `output_lse`（若提供）中。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- `output`, `prefix_output`, `suffix_output` 的最后两个维度必须在内存中连续（即 stride 满足 `stride(-2) == head_size` 且 `stride(-1) == 1`）。
- `head_size` 必须是 `pack_size` 的倍数（对于 Float 为 4 的倍数，对于 Half/BF16 为 8 的倍数）。
- 输入张量之间的 `num_tokens`, `num_heads`, `head_size` 维度必须一致。
- 数据类型需支持 `torch.float32`, `torch.float16`, `torch.bfloat16`。
### 调用示例
```python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 0F
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
NUM_TOKENS = 128
NUM_HEADS = 32
HEAD_SIZE = 128
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建输入 Tensor
# Attention Output 通常为 [num_tokens, num_heads, head_size]
prefix_output = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
suffix_output = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
# LSE (Log-Sum-Exp) 必须为 float32, 形状通常为 [num_heads, num_tokens]
prefix_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32)
suffix_lse = torch.randn(NUM_HEADS, NUM_TOKENS, dtype=torch.float32)
# 2. 创建输出 Tensor
output = torch.empty_like(prefix_output)
output_lse = torch.empty_like(prefix_lse)
# ================= 算子执行 =================
print(f"Running merge_attn_states...")
print(f"Shapes - Output: {prefix_output.shape}, LSE: {prefix_lse.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.merge_attn_states(
    output,
    output_lse,
    prefix_output,
    prefix_lse,
    suffix_output,
    suffix_lse
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output.mean().item():.4f}")
print(f"Output LSE Mean: {output_lse.mean().item():.4f}")
```
## 37. convert_vertical_slash_indexes
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def convert_vertical_slash_indexes(
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool
) -> None:
    pass
```
### 功能描述
convert_vertical_slash_indexes 算子用于将稀疏注意力机制中的垂直索引和斜线索引转换为适用于块稀疏计算的索引格式。
该算子在 GPU 上并行处理每个注意力头和查询块，读取预先计算好的垂直模式索引（代表全局重要 Token）和斜线模式索引（代表局部滑动窗口），并将它们合并映射为物理内存中的块偏移量和独立的列索引。
在计算过程中，算子会根据输入的查询序列长度和键值序列长度动态调整处理范围，同时通过因果参数控制是否应用因果掩码，确保注意力计算不越过当前时间步。对于每个查询块，算法首先确定斜线索引覆盖的 KV 块范围，然后遍历垂直索引，将落在斜线范围内的垂直点合并处理，将范围外的独立垂直点记录为列索引，最终输出每一行对应的有效块数量、块偏移列表以及独立列数量和列索引列表，从而加速后续的块稀疏注意力计算。
- 计算逻辑涉及根据斜线索引 $S_{idx}$ 确定覆盖范围：
	$$
	\text{range\_start} = S_{idx} - \text{BLOCK\_SIZE\_M}
	$$
$$
\text{range\_end} = S_{idx}
$$
	其中 $S_{idx}$ 会根据因果关系 causal 进行调整。
	若开启因果掩码，则需确保索引满足 
	$$
	S_{idx} \le \text{end\_m} + (\text{kv\_seqlen} - \text{q\_seqlen})
	$$
	任何超出该范围的索引将被截断或忽略，从而保证注意力机制只关注到合法的历史信息。
**主要功能包括：**
- 稀疏索引格式转换：将预计算的垂直模式索引（全局重要 Token）和斜线模式索引（局部滑动窗口）高效映射为适用于块稀疏计算的物理内存块偏移 (`block_offset`) 和独立列索引 (`column_index`)。
- 动态范围合并与分离：依据块大小 (`block_size_M/N`) 计算斜线索引覆盖范围，自动将落在范围内的垂直点合并处理，并将范围外的垂直点分离为独立列索引，优化后续内存访问。
- 因果掩码与边界控制：结合动态序列长度 (`q_seqlens`/`kv_seqlens`) 与因果参数 (`causal`)，严格执行 $S_{idx}$ 边界截断逻辑，确保注意力机制不包含非法的前瞻性信息。
- 并行加速预处理：利用 GPU 并行计算各注意力头及查询块的有效块数量 (`block_count`) 与偏移，通过预先剔除无效块，显著提升后续稀疏 Attention 算子的计算效率。
### 参数说明
- **block_count** (torch.Tensor, 出参): 形状为 `[BATCH, N_HEADS, NUM_ROWS]`，存储每行（Query Block）包含的有效 KV 块数量。
- **block_offset** (torch.Tensor, 出参): 形状为 `[BATCH, N_HEADS, NUM_ROWS, NNZ_S]`，存储每行对应的有效 KV 块起始索引。
- **column_count** (torch.Tensor, 出参): 形状为 `[BATCH, N_HEADS, NUM_ROWS]`，存储每行包含的独立垂直列数量。
- **column_index** (torch.Tensor, 出参): 形状为 `[BATCH, N_HEADS, NUM_ROWS, NNZ_V]`，存储每行对应的独立垂直列索引。
- **q_seqlens** (torch.Tensor, 入参): 形状为 `[BATCH]`，当前批次中每个样本的 Query 序列长度。
- **kv_seqlens** (torch.Tensor, 入参): 形状为 `[BATCH]`，当前批次中每个样本的 KV 序列长度。
- **vertical_indexes** (torch.Tensor, 入参): 形状为 `[BATCH, N_HEADS, NNZ_V]`，预计算的垂直模式索引。
- **slash_indexes** (torch.Tensor, 入参): 形状为 `[BATCH, N_HEADS, NNZ_S]`，预计算的斜线模式索引。
- **context_size** (int, 入参): 上下文窗口大小，用于确定总行数。
- **block_size_M** (int, 入参): Query 维度的块大小。
- **block_size_N** (int, 入参): KV 维度的块大小。
- **causal** (bool, 入参): 是否应用因果掩码，True 表示因果注意力，False 表示双向注意力。
### 返回值
无返回值，计算结果直接写入 `block_count`, `block_offset`, `column_count`, `column_index` 张量中。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- 输入索引张量的数据类型必须为 `int32`。
- `NUM_ROWS` 必须等于 `(context_size + block_size_M - 1) / block_size_M`。
- `NNZ_V` 和 `NNZ_S` 为输入索引张量的最后一维，表示最大的保留稀疏点数。
- 线程块配置固定为 64 线程，Grid 维度由 Batch、Heads 和 Rows 决定。
### 调用示例
```python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
# 维度设置
BATCH_SIZE = 2
NUM_HEADS = 8
CONTEXT_SIZE = 1024
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
NNZ_V = 32
NNZ_S = 16
NUM_ROWS = (CONTEXT_SIZE + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建输入 Tensor
q_seqlens = torch.randint(1, CONTEXT_SIZE, (BATCH_SIZE,), dtype=torch.int32)
kv_seqlens = torch.randint(1, CONTEXT_SIZE, (BATCH_SIZE,), dtype=torch.int32)
# 索引必须有序
vertical_indexes, _ = torch.sort(torch.randint(0, CONTEXT_SIZE, (BATCH_SIZE, NUM_HEADS, NNZ_V), dtype=torch.int32), dim=-1)
slash_indexes, _ = torch.sort(torch.randint(0, CONTEXT_SIZE, (BATCH_SIZE, NUM_HEADS, NNZ_S), dtype=torch.int32), dim=-1)
# 2. 创建输出 Tensor
block_count = torch.zeros(BATCH_SIZE, NUM_HEADS, NUM_ROWS, dtype=torch.int32)
block_offset = torch.zeros(BATCH_SIZE, NUM_HEADS, NUM_ROWS, NNZ_S, dtype=torch.int32)
column_count = torch.zeros(BATCH_SIZE, NUM_HEADS, NUM_ROWS, dtype=torch.int32)
column_index = torch.zeros(BATCH_SIZE, NUM_HEADS, NUM_ROWS, NNZ_V, dtype=torch.int32)
# 标量参数
causal = True
# ================= 算子执行 =================
print(f"Running convert_vertical_slash_indexes...")
print(f"Shapes - Vertical: {vertical_indexes.shape}, Slash: {slash_indexes.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.convert_vertical_slash_indexes(
    block_count,
    block_offset,
    column_count,
    column_index,
    q_seqlens,
    kv_seqlens,
    vertical_indexes,
    slash_indexes,
    CONTEXT_SIZE,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    causal
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Block Count Mean: {block_count.float().mean().item():.4f}")
print(f"Column Count Mean: {column_count.float().mean().item():.4f}")
```
## 38. convert_vertical_slash_indexes_mergehead
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def convert_vertical_slash_indexes_mergehead(
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    vertical_indices_count: torch.Tensor,
    slash_indices_count: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool
) -> None:
    pass
```
### 功能描述
convert_vertical_slash_indexes_mergehead 算子实现了 MInference 稀疏注意力机制中索引转换的核心步骤（对应论文 Algorithm 4）。该算子负责将稀疏模式中的“垂直线（Vertical）”和“斜线（Slash）”索引转换为用于实际计算的块偏移量（block_offset）和列索引（column_index）。
与标准版本不同，此 mergehead 版本支持每头（Per-Head）动态的 TopK 配置。它通过 vertical_indices_count 和 slash_indices_count 参数为每个注意力头独立指定有效的索引数量，而不是所有头使用统一的固定 NNZ（非零元素数量）。这允许在多头注意力中根据不同头的稀疏模式特征灵活分配计算资源。
- 索引转换逻辑：
    算子遍历每个查询块（Q Block），根据 slash_indexes 确定的斜线位置生成连续的 KV 块范围，并将这些范围存储在 block_offset 中；同时，根据 vertical_indexes 确定的垂直线位置生成离散的列索引，存储在 column_index 中。
- 动态 TopK 机制：
    在内核执行时，虽然输入张量 vertical_indexes 和 slash_indexes 的物理尺寸由最大缓冲大小 NNZ_V 和 NNZ_S 决定，但实际参与计算的有效索引数量由 per_head_vertical_topkv（对应 vertical_indices_count）和 per_head_slash_topkv（对应 slash_indices_count）控制：
    $$
    \text{Effective\_NNZ}_V[h] = \text{vertical\_indices\_count}[h]
    $$
    $$
    \text{Effective\_NNZ}_S[h] = \text{slash\_indices\_count}[h]
    $$
    内核循环仅在 0 到 Effective_NNZ 的范围内读取索引，从而实现变长的稀疏模式处理。
- 块与列的计数：
    算子同时计算并填充 block_count 和 column_count 张量，记录每个 Q 块对应的有效 KV 块数量和离散列数量，供后续的注意力计算内核（如 paged_attention 或类似的稀疏注意力内核）使用，以跳过无效的计算区域。
**主要功能包括：**
- 将垂直和斜线形式的稀疏索引转换为块稀疏（Block-Sparse）格式。
- 支持 Causal（因果）和 Non-Causal 注意力掩码模式。
- 支持每头独立的稀疏度配置（MergeHead），允许不同头拥有不同数量的垂直线和斜线索引。
- 自动处理边界条件，将超出范围的索引截断或转换为填充块。
- 并行化处理：以 `(N_HEADS, BATCH_SIZE)` 为 Grid，并在 Row 维度上进行线程块划分，利用 GPU 高效执行。
### 参数说明
- **block_count** (torch.Tensor, 出参): 存储每个 Q 块对应的 KV 块数量，形状为 `[BATCH, N_HEADS, NUM_ROWS]`，数据类型需为 int32
- **block_offset** (torch.Tensor, 出参): 存储斜线模式转换后的 KV 块偏移索引，形状为 `[BATCH, N_HEADS, NUM_ROWS, NNZ_S]`，数据类型需为 int32
- **column_count** (torch.Tensor, 出参): 存储每个 Q 块对应的离散列数量，形状为 `[BATCH, N_HEADS, NUM_ROWS]`，数据类型需为 int32
- **column_index** (torch.Tensor, 出参): 存储垂直线模式转换后的离散列索引，形状为 `[BATCH, N_HEADS, NUM_ROWS, NNZ_V]`，数据类型需为 int32
- **q_seqlens** (torch.Tensor, 入参): 查询序列长度，形状为 `[BATCH]`，数据类型需为 int32
- **kv_seqlens** (torch.Tensor, 入参): 键值序列长度，形状为 `[BATCH]`，数据类型需为 int32
- **vertical_indexes** (torch.Tensor, 入参): 垂直线索引，形状为 `[BATCH, N_HEADS, NNZ_V]`，数据类型需为 int32
- **slash_indexes** (torch.Tensor, 入参): 斜线索引，形状为 `[BATCH, N_HEADS, NNZ_S]`，数据类型需为 int32
- **vertical_indices_count** (torch.Tensor, 入参): 每个头的有效垂直线索引数量，形状为 `[N_HEADS]`，数据类型需为 int32
- **slash_indices_count** (torch.Tensor, 入参): 每个头的有效斜线索引数量，形状为 `[N_HEADS]`，数据类型需为 int32
- **context_size** (int, 入参): 上下文窗口大小，用于计算 `NUM_ROWS`
- **block_size_M** (int, 入参): Q 维度的块大小（通常为 64）
- **block_size_N** (int, 入参): KV 维度的块大小（通常为 64）
- **causal** (bool, 入参): 是否应用因果掩码（True 为 Intra-Attention，False 为 Successor/Encoder-Decoder Attention）
### 返回值
无返回值，计算结果直接写入输出张量 `block_count`, `block_offset`, `column_count`, `column_index` 中
### 约束与调用
- 所有输入 Tensor 必须位于 CUDA 设备上
- 所有索引相关的 Tensor（包括 count, offset, index）数据类型必须为 `torch.int32` (`int`)
- `vertical_indexes` 和 `slash_indexes` 的最后一维大小决定了内核中的 `NNZ_V` 和 `NNZ_S` 缓冲区大小，但实际读取量由 `*_indices_count` 控制
- `block_count` 等输出张量的 `NUM_ROWS` 维度必须等于 `(context_size + block_size_M - 1) // block_size_M`
- `q_seqlens` 和 `kv_seqlens` 的 batch 维度必须与输入索引张量的 batch 维度一致
- `vertical_indices_count` 和 `slash_indices_count` 的长度必须等于 `N_HEADS`
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
BATCH_SIZE = 1
N_HEADS = 4
CONTEXT_SIZE = 128
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
NNZ_V = 8  # 垂直索引缓冲区大小
NNZ_S = 8  # 斜线索引缓冲区大小
DEVICE = "cuda"
# 计算行数
NUM_ROWS = (CONTEXT_SIZE + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
# 初始化环境
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 序列长度
q_seqlens = torch.tensor([CONTEXT_SIZE] * BATCH_SIZE, dtype=torch.int32)
kv_seqlens = torch.tensor([CONTEXT_SIZE] * BATCH_SIZE, dtype=torch.int32)
# 模拟输入的稀疏索引 (Batch, Heads, NNZ)
vertical_indexes = torch.randint(0, CONTEXT_SIZE, (BATCH_SIZE, N_HEADS, NNZ_V), dtype=torch.int32)
slash_indexes = torch.randint(0, CONTEXT_SIZE, (BATCH_SIZE, N_HEADS, NNZ_S), dtype=torch.int32)
# 关键：每头的动态 TopK 计数
# 假设 Head 0 使用 2 个垂直索引，Head 1 使用 5 个，以此类推
vertical_indices_count = torch.randint(1, NNZ_V + 1, (N_HEADS,), dtype=torch.int32)
slash_indices_count = torch.randint(1, NNZ_S + 1, (N_HEADS,), dtype=torch.int32)
# 分配输出张量
block_count = torch.zeros((BATCH_SIZE, N_HEADS, NUM_ROWS), dtype=torch.int32)
column_count = torch.zeros((BATCH_SIZE, N_HEADS, NUM_ROWS), dtype=torch.int32)
block_offset = torch.zeros((BATCH_SIZE, N_HEADS, NUM_ROWS, NNZ_S), dtype=torch.int32)
column_index = torch.zeros((BATCH_SIZE, N_HEADS, NUM_ROWS, NNZ_V), dtype=torch.int32)
causal = True
# ================= 算子执行 =================
print(f"Running convert_vertical_slash_indexes_mergehead...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.convert_vertical_slash_indexes_mergehead(
    block_count,
    block_offset,
    column_count,
    column_index,
    q_seqlens,
    kv_seqlens,
    vertical_indexes,
    slash_indexes,
    vertical_indices_count,
    slash_indices_count,
    CONTEXT_SIZE,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    causal
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Block Count Max: {block_count.max().item()}")
print(f"Column Count Max: {column_count.max().item()}")
```
## 39. silu_and_mul
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def silu_and_mul(
    out: torch.Tensor,
    input: torch.Tensor
) -> None:
    pass
```
### 功能描述
silu_and_mul 算子实现了融合的 SiLU 激活函数与逐元素乘法操作，通常被称为 SwiGLU（Swish-Gated Linear Unit）激活函数的变体。该算子在 vLLM 等大语言模型推理框架中常用于前馈神经网络（FFN）层，通过融合操作减少内存访问次数，从而提高计算效率。
该算子利用 CPU 向量化指令（如 AVX）和 OpenMP 并行化技术，对输入张量进行高效处理。它假定输入张量的最后一个维度包含了门控机制所需的两部分数据。
- 计算公式：
    算子首先将输入张量在最后一个维度上平分为两部分 $x$ 和 $y$，然后计算 SiLU 激活后的 $x$ 与 $y$ 的逐元素乘积：
    $$
    \text{out} = \text{SiLU}(x) \odot y
    $$
    其中 SiLU（Sigmoid Linear Unit）定义为：
    $$
    \text{SiLU}(x) = x \cdot \sigma (x) = \frac{x}{1 + e^{-x}}
    $$
    输入与切分逻辑：
    - $I \in \mathbb{R}^{\dots \times 2D}$ 是输入张量 (`input`)
    - $x = I[\dots, 0: D]$ 是输入的前半部分
    - $y = I[\dots, D: 2D]$ 是输入的后半部分
    - $O \in \mathbb{R}^{\dots \times D}$ 是输出张量 (`out`)
    - $D$ 是输出的隐藏层维度 (`d`)
**主要功能包括：**
- 自动将输入张量的最后一个维度 $2d$ 切分为两个 $d$ 维向量。
- 对前半部分向量执行 SiLU 激活计算。
- 将激活后的结果与后半部分向量进行逐元素相乘（Hadamard Product）。
- 利用 `vec_op::FP32Vec8` 进行 SIMD 向量化计算，并结合 OpenMP 实现多线程并行处理。
- 支持浮点类型分发（通过 `VLLM_DISPATCH_FLOATING_TYPES`），适配不同的数据精度。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[..., d]`，存储计算结果
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., 2 * d]`，包含待激活部分和门控部分
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 输入张量 `input` 的最后一个维度必须是偶数。
- 输出张量 `out` 的最后一个维度必须是输入张量 `input` 最后一个维度的一半。
- `out` 和 `input` 的其他维度（batch size, sequence length 等）必须完全一致。
- 输入张量应在 CPU 内存中（基于提供的 CPU kernel 实现）。
- 维度 $d$ 必须是向量长度（`VEC_ELEM_NUM`，通常为 8）的倍数，以满足向量化对齐要求。
- 支持的数据类型：float, bfloat16, half（依赖于具体的 `scalar_t` 实例化）。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float32
# 维度设置
NUM_TOKENS = 10
HIDDEN_SIZE = 128  # d
INPUT_SIZE = 256   # 2 * d
# 初始化环境
torch.manual_seed(SEED)
# ================= 数据准备 =================
# 创建 Input Tensor [num_tokens, 2 * d]
input_tensor = torch.randn(NUM_TOKENS, INPUT_SIZE, dtype=DTYPE, device=DEVICE)
# 创建 Output Tensor [num_tokens, d]
output_tensor = torch.empty(NUM_TOKENS, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
# ================= 算子执行 =================
print(f"Running silu_and_mul...")
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
torch.ops._C.silu_and_mul(
    output_tensor,
    input_tensor
)
print("执行成功。")
# 验证结果逻辑 (仅作参考)
x, y = input_tensor.chunk(2, dim=-1)
expected = torch.nn.functional.silu(x) * y
print(f"Output Mean: {output_tensor.mean().item():.4f}")
```
## 40. silu_and_mul_quant
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def silu_and_mul_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor
) -> None:
    pass
```
### 功能描述
silu_and_mul_quant 算子实现了一个融合的激活、门控（Gating）和量化操作。它通常用于大语言模型（LLM）的前馈神经网络（FFN）层，特别是在 SwiGLU 激活函数后接量化的场景中。该算子从输入张量中分离出门控部分和值部分，应用 SiLU 激活函数，执行逐元素乘法，然后将结果除以缩放因子并量化为 FP8 格式。
该算子通过将多个操作融合到一个 CUDA 内核中 (act_and_mul_quant_kernel)，减少了显存读写次数，显著提升了推理性能。
- 计算公式：
    输入张量 input 在最后一个维度上被分为两部分 $x$ 和 $y$（各占一半维度 $d$）。
    $$
    \text{out} = \text{Quant}_{FP8}\left ( \frac{\text{SiLU}(x) \odot y}{\text{scale}} \right)
    $$
    其中：
    - $x$ 是 `input` 的前一半通道 (`input[..., : d]`)
    - $y$ 是 `input` 的后一半通道 (`input[..., d:]`)
    - $\text{SiLU}(x) = \frac{x}{1 + e^{-x}}$
    - $\odot$ 表示逐元素乘法
    - $\text{scale}$ 是量化缩放因子
    - $\text{Quant}_{FP8}$ 表示转换为 FP8 数据类型（e4m3fn 或 e4m3fnuz）
**主要功能包括：**
- 融合计算：在一个内核中完成 Split、SiLU 激活、乘法和量化。
- FP8 量化支持：直接输出 FP8 格式（`float8_e4m3fn` 或 `float8_e4m3fnuz`），减少显存占用并加速后续计算。
- 高效内存访问：使用 128-bit 向量化加载和存储，优化显存带宽利用率。
- 精度支持：输入支持 FP16 和 BF16，计算过程使用高精度，仅在输出时进行截断和量化。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[..., d]`，数据类型为 FP8 (`torch.float8_e4m3fn` 或 `torch.float8_e4m3fnuz`)。
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., 2 * d]`，数据类型支持 `torch.float16` 或 `torch.bfloat16`。包含拼接在一起的 Gate 和 Up 投影结果。
- **scale** (torch.Tensor, 入参): 量化缩放因子，通常为标量张量 (float)，用于反量化或调整数值范围。
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- `out` 的数据类型必须是 `torch.float8_e4m3fn` 或 `torch.float8_e4m3fnuz`。
- `input` 的数据类型必须是 `torch.float16` 或 `torch.bfloat16`。
- `input` 的最后一个维度大小必须是偶数（即 `input.size (-1) % 2 == 0`），因为它需要被平均分为两部分。
- `scale` 张量必须包含有效的浮点数值。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
INPUT_DTYPE = torch.bfloat16
OUTPUT_DTYPE = torch.float8_e4m3fn 
# 维度设置
BATCH_SIZE = 16
SEQ_LEN = 512
HIDDEN_DIM = 4096  # 这是 d，input 的最后一维将是 2*d
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# input shape: [BATCH, SEQ, 2 * HIDDEN]
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, 2 * HIDDEN_DIM, dtype=INPUT_DTYPE)
# scale 通常是一个标量 float tensor
scale = torch.tensor([0.5], dtype=torch.float32, device=DEVICE)
# output shape: [BATCH, SEQ, HIDDEN]
output = torch.empty(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=OUTPUT_DTYPE)
# ================= 算子执行 =================
print(f"Running silu_and_mul_quant...")
print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.silu_and_mul_quant(
    output,
    input_tensor,
    scale
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# 由于 output 是 FP8，打印数值可能需要转换回 float 查看近似值
print(f"Output Mean (cast to float): {output.float().mean().item():.4f}")
```
## 41. mul_and_silu
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def mul_and_silu(
    out: torch.Tensor,
    input: torch.Tensor
) -> None:
    pass
```
### 功能描述
mul_and_silu 算子实现了一种融合的激活与门控机制。该算子是 vLLM 中用于高效执行前馈神经网络（FFN）或多层感知机（MLP）中间层计算的 CUDA 内核。
此实现通过将输入张量在最后一个维度上一分为二，对后半部分应用 SiLU 激活函数，然后与前半部分进行逐元素乘法。这种操作通常用于门控线性单元（Gated Linear Units）的变体计算中，通过融合操作减少内存访问次数（Kernel Fusion），从而显著提升计算吞吐量。
- 计算公式：
    输入张量 input 被视为两个逻辑部分 $x$ 和 $y$，沿最后一个维度分割：
    $$
    \text{input} = [x, y]
    $$
    其中：
    - $x \in \mathbb{R}^{\dots \times D}$ 是输入的前半部分
    - $y \in \mathbb{R}^{\dots \times D}$ 是输入的后半部分
    - $D$ 是输出维度 (`d`)，输入维度为 $2D$
    输出计算如下：
    $$
    \text{out} = x \odot \text{SiLU}(y)
    $$
    SiLU（Sigmoid Linear Unit）激活函数定义为：
    $$
    \text{SiLU}(z) = \frac{z}{1 + e^{-z}}
    $$
**主要功能包括：**
- 高效的逐元素融合计算，避免了多次显存读写。
- 原生支持 `mul_and_silu` 逻辑，即对输入的后半部分进行激活，与前半部分相乘（注意这与 `silu_and_mul` 的激活顺序相反）。
- 自动处理向量化加载（Vectorized Loads），优化内存带宽利用率。
- 支持多种浮点数据类型（FP16, BF16, FP32）。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[..., d]`，存储计算结果
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., 2 * d]`，包含拼接在一起的门控分量和值分量
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `input` 张量的最后一个维度必须是偶数
- `out` 张量的最后一个维度必须是 `input` 张量最后一个维度的一半
- `input` 和 `out` 的其他维度（Batch size, Token num 等）必须完全一致
- 内存布局要求：输入张量建议是连续存储的，以获得最佳性能
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
NUM_TOKENS = 128
HIDDEN_SIZE = 4096  # d
INPUT_SIZE = HIDDEN_SIZE * 2  # 2 * d
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# input 包含 [gate, value] 或 [x, y]，维度为 [..., 2*d]
input_tensor = torch.randn(NUM_TOKENS, INPUT_SIZE, dtype=DTYPE)
# out 维度为 [..., d]
output_tensor = torch.empty(NUM_TOKENS, HIDDEN_SIZE, dtype=DTYPE)
# ================= 算子执行 =================
print(f"Running mul_and_silu...")
print(f"Input: {input_tensor.shape}, Output: {output_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用底层算子
torch.ops._C.mul_and_silu(
    output_tensor,
    input_tensor
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# 简单验证计算逻辑
d = HIDDEN_SIZE
x = input_tensor[..., :d].float()
y = input_tensor[..., d:].float()
expected = x * (y / (1.0 + torch.exp(-y)))
diff = (output_tensor.float() - expected).abs().max()
print(f"Max Difference: {diff.item():.6f}")
```
## 42. gelu_and_mul
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def gelu_and_mul(
    out: torch.Tensor,
    input: torch.Tensor
) -> None:
    pass
```
### 功能描述
gelu_and_mul 算子实现了一种门控激活函数机制，常用于 Transformer 模型的 MLP 层（如 GeGLU 变体）。该算子将输入张量在最后一个维度上切分为两半，对前半部分应用 GELU 激活函数，然后将其与后半部分进行逐元素相乘。
此实现通过 CUDA 内核 act_and_mul_kernel 高效执行，融合了切分、激活和乘法操作，减少了显存读写次数。
- 计算逻辑：
    输入张量 input 被视为由两个部分 $x$ 和 $y$ 拼接而成。假设输入张量的最后一个维度大小为 $2d$，则前 $d$ 个元素为 $x$，后 $d$ 个元素为 $y$。
    $$
    \text{input} = [x, y], \quad x, y \in \mathbb{R}^{\dots \times d}
    $$
    算子计算输出：
    $$
    \text{out} = \text{GELU}(x) \odot y
    $$
    其中 $\odot$ 表示逐元素乘法。
- GELU 激活函数：
    代码中使用的是标准的 GELU 实现（无近似或 exact 版本），公式如下：
    $$
    \text{GELU}(x) = x \cdot \frac{1}{2} \cdot \left (1 + \text{erf}\left (\frac{x}{\sqrt{2}}\right)\right)
    $$
    在 CUDA 内核中，$\frac{1}{\sqrt{2}}$ 对应常量 M_SQRT1_2。
**主要功能包括：**
- 对输入进行隐式的 split 操作，无需额外的内存拷贝。
- 计算 GELU 激活值。
- 执行逐元素门控乘法。
- 支持多种浮点数据类型（FP16, BF16, FP32）。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[..., d]`，其中 `d` 是输入张量最后一维大小的一半。
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., 2 * d]`。
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入张量的最后一个维度大小必须是偶数
- out 张量的最后一个维度大小必须等于 input 张量最后一个维度大小的一半
- 支持的数据类型：float16, bfloat16, float32
- 内存布局要求：建议使用连续内存布局以获得最佳性能
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
BATCH_SIZE = 16
SEQ_LEN = 128
HIDDEN_DIM = 256  # 输出维度 d
INPUT_DIM = HIDDEN_DIM * 2  # 输入维度 2*d
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 创建 Tensor
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, dtype=DTYPE)
output_tensor = torch.empty(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=DTYPE)
# ================= 算子执行 =================
print(f"Running gelu_and_mul...")
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.gelu_and_mul(
    output_tensor,
    input_tensor
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output_tensor.mean().item():.4f}")
# ================= 结果验证 (可选) =================
# 模拟 Python 侧计算进行对比
x, y = input_tensor.chunk(2, dim=-1)
expected = torch.nn.functional.gelu(x, approximate='none') * y
diff = (output_tensor - expected).abs().max()
print(f"Max difference: {diff.item():.6f}")
```
## 43. gelu_tanh_and_mul
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def gelu_tanh_and_mul(
    out: torch.Tensor,
    input: torch.Tensor
) -> None:
    pass
```
### 功能描述
gelu_tanh_and_mul 算子实现带有 Tanh 近似的 GELU 激活函数的门控机制（Gated Activation Unit）。该算子通常用于现代大语言模型（如 GeGLU 变体）的前馈神经网络（FFN）层中，结合了激活函数与线性门控机制。
该算子接收一个输入张量，将其在最后一个维度上平分为两部分，对第一部分应用 GELU (Tanh approximation) 激活函数，然后与第二部分进行逐元素相乘。
- 计算公式：
    $$
    \text{out} = \text{GELU}_{\text{tanh}}(x) \odot y
    $$
    其中：
    - 输入 $\text{input}$ 被切分为 $x$ 和 $y$ 两部分，形状均为 $[\dots, d]$
    - $x$ 为输入的前半部分 ($\text{input}[\dots, : d]$)
    - $y$ 为输入的后半部分 ($\text{input}[\dots, d:]$)
    - $\text{GELU}_{\text{tanh}}$ 的近似计算公式参考 PyTorch 实现：
        $$
        \text{GELU}_{\text{tanh}}(x) = 0.5 x \left ( 1 + \tanh\left ( \sqrt{\frac{2}{\pi}} (x + 0.044715 x^3) \right) \right)
        $$
    - $\odot$ 表示逐元素乘法
**主要功能包括：**
- 对输入张量进行切分、激活和门控乘法的融合操作。
- 使用 Tanh 近似实现 GELU，相比标准 ERF 实现具有更好的计算性能，同时保持数值精度。
- 通过 CUDA Kernel (`act_and_mul_kernel`) 高效并行执行，减少显存读写次数。
- 支持多种浮点数据类型（float16, bfloat16, float32）。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[..., d]`，存储计算结果
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., 2 * d]`，包含待激活部分和门控部分
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入张量 `input` 的最后一个维度必须是偶数
- 输出张量 `out` 的最后一个维度必须等于输入张量 `input` 最后一个维度的一半
- `out` 和 `input` 的其他维度必须完全一致
- 支持的数据类型：float16, bfloat16, float32
- 内存布局要求：输入张量应是连续存储的，以获得最佳性能
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
BATCH_SIZE = 8
SEQ_LEN = 128
HIDDEN_SIZE = 4096  # output dim (d)
INPUT_DIM = HIDDEN_SIZE * 2  # input dim (2*d)
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# input shape: [BATCH_SIZE, SEQ_LEN, 2 * HIDDEN_SIZE]
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, dtype=DTYPE)
# output shape: [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
output = torch.empty(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE)
# ================= 算子执行 =================
print(f"Running gelu_tanh_and_mul...")
print(f"Input: {input_tensor.shape}, Output: {output.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.gelu_tanh_and_mul(
    output,
    input_tensor
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output.mean().item():.4f}")
```
## 44. fatrelu_and_mul
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def fatrelu_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
    threshold: float
) -> None:
    pass
```
### 功能描述
fatrelu_and_mul 算子实现一种基于阈值的激活门控机制（Thresholded ReLU Gating）。该算子通常用于神经网络中的门控前馈网络（Gated FFN）层，通过结合带有阈值的 ReLU 激活函数与逐元素乘法来实现非线性变换。
该算子将输入张量沿最后一个维度一分为二，对前半部分应用 FatReLU 激活（即带有阈值的 ReLU），然后与后半部分进行逐元素乘法。
- 计算公式：
    $$
    \text{FatReLU}(x, \theta) = \begin{cases} x & \text{if } x > \theta \\ 0 & \text{otherwise} \end{cases}
    $$
    $$
    \text{out} = \text{FatReLU}(x_{\text{gate}}, \text{threshold}) \odot x_{\text{up}}
    $$
    其中：
    - $\theta$ 是阈值 (`threshold`)
    - Input 被分割为两部分：$x_{\text{gate}}$ 为输入的前半部分，$x_{\text{up}}$ 为输入的后半部分
    - $\odot$ 表示逐元素乘法
    - 输入张量 `input` 的形状为 $[..., 2 \times d]$
    - 输出张量 `out` 的形状为 $[..., d]$
**主要功能包括：**
- 对输入张量的前半部分执行高效的阈值过滤（FatReLU）。
- 将激活后的结果与输入张量的后半部分进行融合（乘法）。
- 通过 `LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM` 宏自动处理 CUDA 内核的调度与分发。
- 支持多种浮点数据类型（Float, Half, BFloat16），利用 CUDA 模板进行实例化。
- 针对最后维度 `d` 的大小自动选择合适的 block 维度（最大 1024）。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[..., d]`，存储计算结果
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., 2 * d]`，包含门控部分和信号部分
- **threshold** (float, 入参): FatReLU 的激活阈值，输入值需大于该阈值才会被保留，否则置为 0.0
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入张量 `input` 的最后一个维度必须是偶数
- 输出张量 `out` 的最后一个维度必须等于 `input` 最后一个维度的一半
- 支持的数据类型：float16, bfloat16, float32
- 内存布局要求：输入张量通常建议是连续存储的，以获得最佳内存访问性能
### 调用示例
```Python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
BATCH_SIZE = 16
SEQ_LEN = 128
HIDDEN_SIZE = 1024  # d
INPUT_DIM = HIDDEN_SIZE * 2  # 2 * d
THRESHOLD = 0.05
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# input shape: [BATCH_SIZE, SEQ_LEN, 2 * HIDDEN_SIZE]
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, dtype=DTYPE)
# out shape: [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
output = torch.empty(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE)
# ================= 算子执行 =================
print(f"Running fatrelu_and_mul...")
print(f"Input: {input_tensor.shape}, Output: {output.shape}, Threshold: {THRESHOLD}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 假设绑定在 mcoplib._C 下
torch.ops._C.fatrelu_and_mul(
    output,
    input_tensor,
    THRESHOLD
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# ================= 结果验证 (可选) =================
# 模拟 PyTorch 计算进行对比
x, y = input_tensor.chunk(2, dim=-1)
expected = torch.where(x > THRESHOLD, x, torch.tensor(0.0, dtype=DTYPE)) * y
diff = (output - expected).abs().max()
print(f"Max Diff: {diff.item():.6f}")
print(f"Output Mean: {output.mean().item():.4f}")
```
## 45. swigluoai_and_mul
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def swigluoai_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0
) -> None:
    pass
```
### 功能描述
`swigluoai_and_mul` 算子实现了一种变体的 SwiGLU 激活函数，结合了特定的截断（Clamping）和缩放逻辑。该算子通常用于特定的深度学习模型（如 OpenAI 的某些模型架构）中的前馈神经网络层。它将输入张量在最后一个维度上分割为两部分（Gate 和 Up），分别进行处理后相乘。
- 计算逻辑：
    输入张量 input 的最后一个维度大小为 $2d$，算子将其分割为两个维度为 $d$ 的部分：$x_{gate}$ 和 $x_{up}$。
    1. 分割输入：
        $$
        x_{gate} = \text{input}[\dots, 0:: 2] \quad (\text{偶数索引})
        $$
        $$
        x_{up} = \text{input}[\dots, 1:: 2] \quad (\text{奇数索引})
        $$
    2. 截断处理 (Clamping)：
        对 $x_{gate}$ 进行上限截断，对 $x_{up}$ 进行双向截断。
        $$
        \tilde{x}_{gate} = \min (x_{gate}, \text{limit})
        $$
        $$
        \tilde{x}_{up} = \text{clamp}(x_{up}, -\text{limit}, \text{limit})
        $$
    3. 激活与组合：
        计算带缩放参数 $\alpha$ 的 Swish 激活值，并与调整后的 $x_{up}$ 相乘。
        $$
        \text{glu} = \tilde{x}_{gate} \cdot \sigma (\tilde{x}_{gate} \cdot \alpha)
        $$
        $$
        \text{out} = (\tilde{x}_{up} + 1.0) \cdot \text{glu}
        $$
        其中 $\sigma (z) = \frac{1}{1 + e^{-z}}$ 是 Sigmoid 函数。
**主要功能包括：**
- 对输入进行 GLU 风格的切分（Split）。
- 对切分后的分量进行数值截断，增强数值稳定性。
- 计算参数化的 Swish 激活。
- 高效的 CUDA 内核实现，支持向量化加载和计算。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[..., d]`，存储计算结果
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., 2 * d]`，包含 Gate 和 Up 两部分数据
- **alpha** (float, 入参, 可选): Sigmoid 函数内部的缩放因子，默认值为 1.702
- **limit** (float, 入参, 可选): 截断阈值，用于限制 Gate 的上限和 Up 的绝对值，默认值为 7.0
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入张量 `input` 的最后一个维度必须能被 2 整除
- `out` 张量的最后一个维度必须是 `input` 张量最后一个维度的一半
- 支持的数据类型：float16, bfloat16, float32
- 内存布局要求：输入张量必须是连续存储的
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
BATCH_SIZE = 16
SEQ_LEN = 128
HIDDEN_DIM = 4096  # 输出维度 d
INPUT_DIM = HIDDEN_DIM * 2  # 输入维度 2*d
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, dtype=DTYPE)
output_tensor = torch.empty(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=DTYPE)
# 参数设置
alpha = 1.702
limit = 7.0
# ================= 算子执行 =================
print(f"Running swigluoai_and_mul...")
print(f"Input: {input_tensor.shape}, Output: {output_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用底层算子
torch.ops._C.swigluoai_and_mul(
    output_tensor,
    input_tensor,
    alpha,
    limit
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output_tensor.mean().item():.4f}")
```
## 46. gelu_new
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
void gelu_new(
    torch::Tensor& out,
    torch::Tensor& input
);
```
### 功能描述
gelu_new 算子实现基于 Tanh 近似的 GeLU (Gaussian Error Linear Unit) 激活函数。该算子是 vLLM 中用于高效执行逐元素激活计算的 CUDA 内核启动器。它通过 LAUNCH_ACTIVATION_KERNEL 宏调度 gelu_new_kernel，根据输入张量计算激活后的输出。
此实现采用 "New GeLU" 近似方案，通过多项式拟合与双曲正切函数组合来替代标准的误差函数 (ERF) 实现，在保持精度的同时提高了计算效率。
- 计算公式：
    该算子实现了 GeLU 的 Tanh 近似版本：
    $$
    \text{GeLU}(x) = 0.5 x \left (1 + \tanh\left[\sqrt{\frac{2}{\pi}} \left (x + 0.044715 x^3\right)\right]\right)
    $$
    代码中常数 $0.79788456$ 对应于 $\sqrt{2/\pi}$ 的近似值。
    具体计算逻辑如下：
    $$
    \begin{aligned} x^3 &= x \cdot x \cdot x \\ \text{inner} &= 0.79788456 \cdot (x + 0.044715 \cdot x^3) \\ \text{out} &= 0.5 \cdot x \cdot (1 + \tanh (\text{inner})) \end{aligned}
    $$
- 核心特性：
    算子使用 element-wise（逐元素）方式并行处理数据。通过 activation_kernel 模板和 LAUNCH_ACTIVATION_KERNEL 宏，算子能够自动处理 CUDA 网格（Grid）和块（Block）的维度配置，并支持通过 VLLM_DISPATCH_FLOATING_TYPES 分发多种浮点数据类型。
**主要功能包括：**
- 基于 Tanh 近似的 New GeLU 实现：采用多项式拟合 ($x + 0.044715 x^3$) 结合双曲正切函数 ($\tanh$) 替代标准误差函数，在保持激活精度的同时大幅降低计算开销。
- 高效的逐元素并行计算：利用 CUDA 核心执行 Element-wise 并行处理，通过 `LAUNCH_ACTIVATION_KERNEL` 宏自动管理网格与线程块配置，实现高吞吐量的激活运算。
- 多精度数据类型动态分发：借助 `VLLM_DISPATCH_FLOATING_TYPES` 机制，原生支持 FP16、BF16 和 FP32 数据类型，灵活适配不同混合精度模型的推理需求。
- 精确的数学常数定义：内置 $\sqrt{2/\pi}$ 的高精度近似值 ($0.79788456$)，严格遵循 GeLU 理论公式进行前向计算，保证数值输出的稳定性与准确性。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状与输入张量 `input` 相同，存储激活计算结果
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., d]` 或任意维度的张量
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入张量 `input` 和输出张量 `out` 的形状必须一致
- 支持的数据类型：float16, bfloat16, float32（依赖于 `VLLM_DISPATCH_FLOATING_TYPES` 的支持范围）
- 内存布局要求：张量数据在内存中应是连续的，以便内核能够正确地通过线性索引访问
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16 # 或 torch.bfloat16, torch.float32
# 维度设置
BATCH_SIZE = 16
SEQ_LEN = 128
HIDDEN_DIM = 1024
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=DTYPE)
output_tensor = torch.empty_like(input_tensor)
# ================= 算子执行 =================
print(f"Running gelu_new...")
print(f"Input shape: {input_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用底层 C++ 绑定的接口
torch.ops._C.gelu_new(
    output_tensor,
    input_tensor
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output_tensor.mean().item():.4f}")
```
## 47. gelu_fast
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
void gelu_fast(
    torch::Tensor& out,
    torch::Tensor& input
);
```
### 功能描述
gelu_fast 算子实现高斯误差线性单元（Gaussian Error Linear Unit, GELU）激活函数的快速近似版本。该算子是 vLLM 中用于神经网络激活层的核心 CUDA 内核之一，采用元素级（Element-wise）并行计算方式。
该实现采用了基于 tanh 函数的近似公式（通常被称为 GeLU Tanh 或 GeLU Fast），相比于基于标准误差函数（erf）的精确实现，它在保持极高精度的前提下提供了更好的计算性能。该近似方法被广泛应用于 BERT、GPT 等 Transformer 架构的大语言模型中。
- 计算公式：
    使用如下近似公式进行计算：
    $$
    \text{GELU}(x) = 0.5x \left ( 1 + \tanh \left[ \sqrt{\frac{2}{\pi}} \left ( x + 0.044715 x^3 \right) \right] \right)
    $$
    其中：
    - $x$ 是输入张量中的元素
    - $\sqrt{2/\pi} \approx 0.79788456$
    - $0.044715$ 是用于拟合误差函数的常数系数
**主要功能包括：**
- 对输入张量进行元素级的 GELU 激活计算。
- 利用 CUDA 架构进行高度并行的浮点运算。
- 支持多种浮点数据类型（FP16, BF16, FP32），通过 `VLLM_DISPATCH_FLOATING_TYPES` 宏进行自动分发。
- 直接在 GPU 上操作，无需数据回传主机。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状与 `input` 相同，存储激活计算结果
- **input** (torch.Tensor, 入参): 输入张量，可以是任意形状的张量
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `out` 张量必须预先分配，且形状必须与 `input` 张量完全一致
- 支持的数据类型：float16, bfloat16, float32
- 内存布局要求：输入张量通常建议是连续存储的（Contiguous），以获得最佳内存访问性能
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16 # 或 torch.bfloat16, torch.float32
# 维度设置
BATCH_SIZE = 4
SEQ_LEN = 128
HIDDEN_DIM = 768
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=DTYPE)
output_tensor = torch.empty_like(input_tensor)
# ================= 算子执行 =================
print(f"Running gelu_fast...")
print(f"Input shape: {input_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用 C++ 扩展绑定的接口
torch.ops._C.gelu_fast(
    output_tensor,
    input_tensor
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output_tensor.mean().item():.4f}")
# 验证计算结果（与 PyTorch 原生 GELU tanh 模式对比）
ref_output = torch.nn.functional.gelu(input_tensor, approximate='tanh')
diff = (output_tensor - ref_output).abs().max()
print(f"Max difference from torch.nn.functional.gelu(tanh): {diff.item():.6e}")
```
## 48. gelu_quick
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
void gelu_quick(
    torch::Tensor& out,
    torch::Tensor& input
);
```
### 功能描述
gelu_quick 算子实现了高斯误差线性单元（GELU）的一种快速近似计算版本。该算子是 vLLM 中用于神经网络激活层的 CUDA 内核启动器。它通过调用 gelu_quick_kernel，对输入张量进行逐元素（Element-wise）计算。
此实现采用 Sigmoid 函数近似 GELU，相比标准的误差函数（erf）实现或 Tanh 近似实现，该版本计算速度更快，但精度略低。适用于对计算性能要求较高且对精度略微不敏感的场景。
- 计算公式：
    使用 Sigmoid 函数进行近似，系数为 1.702：
    $$
    \text{GELU}_{\text{quick}}(x) = x \cdot \sigma (1.702 \cdot x) = \frac{x}{1 + e^{-1.702x}}
    $$
    其中：
    - $x$ 是输入元素
    - $1.702$ 是经验系数，用于匹配标准正态分布累积分布函数
**主要功能包括：**
- 对输入张量进行逐元素的快速 GELU 激活计算。
- 利用 CUDA 模板 `activation_kernel` 进行并行加速。
- 支持多种浮点数据类型（通过 `VLLM_DISPATCH_FLOATING_TYPES` 分发），通常包括 float16, bfloat16, float32。
- 自动处理 CUDA 流和设备守卫（Device Guard）。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状与 `input` 相同，存储激活计算结果
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., d]`，包含需要激活的数据
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `out` 和 `input` 张量的形状（Shape）必须完全一致
- `out` 和 `input` 张量的数据类型必须一致且为浮点类型（如 half, bfloat16, float）
- 内存布局要求：支持标准布局，内核通过线性索引访问
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
BATCH_SIZE = 8
SEQ_LEN = 128
HIDDEN_SIZE = 4096
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 创建输入 Tensor
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE)
# 创建输出 Tensor (需与输入形状一致)
output_tensor = torch.empty_like(input_tensor)
# ================= 算子执行 =================
print(f"Running gelu_quick...")
print(f"Input shape: {input_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用 C++ 扩展接口
torch.ops._C.gelu_quick(
    output_tensor,
    input_tensor
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Input Mean: {input_tensor.mean().item():.4f}")
print(f"Output Mean: {output_tensor.mean().item():.4f}")
```
## 50. fused_add_rms_norm
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float
) -> None:
    pass
```
### 功能描述
fused_add_rms_norm 算子实现了残差连接相加与 Root Mean Square Layer Normalization (RMS Norm) 的融合操作。该算子是 vLLM 及现代 Transformer 架构中用于加速层归一化计算的关键组件。它将输入的特征张量与残差张量相加，更新残差，并对相加后的结果进行 RMS 归一化。
此实现针对 GPU 内存延迟进行了优化，特别是在处理 FP16/BF16 数据类型时，通过向量化读写（Vectorized Load/Store）减少全局内存访问次数。算子内部使用 cub:: BlockReduce 进行高效的并行归约计算。
- 计算逻辑与公式：
    算子按顺序执行以下步骤：
    1. 残差相加（Residual Add）：
        将输入 input 与残差 residual 相加，结果作为新的去归一化激活值。同时更新 residual 张量以存储此中间结果（用于后续的反向传播或推理中的跳跃连接）。
        $$
        x_{\text{new}} = x_{\text{input}} + x_{\text{residual}}
        $$
        $$
        \text{residual} \leftarrow x_{\text{new}}
        $$
    2. RMS 归一化（RMS Norm）：
        计算 $x_{\text{new}}$ 的均方根，并进行归一化和缩放。
        $$
        \text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 + \epsilon}
        $$
        $$
        \bar{x}_i = \frac{x_i}{\text{RMS}(x)}
        $$
        $$
        y_i = \bar{x}_i \cdot w_i
        $$
        $$
        \text{input} \leftarrow y
        $$
        其中：
	    - $x$ 是 hidden_size 维度的输入向量
	    - $w$ 是权重向量 (`weight`)
	    - $\epsilon$ 是防止除零的小常数 (`epsilon`)
	    - $n$ 是 `hidden_size`
**主要功能包括：**
- 算子融合：将 Add 和 RMS Norm 两个操作融合为一个 CUDA Kernel，减少显存读写带宽占用和内核启动开销。
- 原地更新（In-place Update）：直接修改传入的 `input` 张量存储归一化结果，修改 `residual` 张量存储相加后的非归一化结果。
- 向量化优化：针对 `float16` 和 `bfloat16` 类型，当数据地址对齐（16 字节）且维度满足条件时，自动启用 8 元素宽度的向量化加载和存储（128 位），大幅提升内存吞吐量。
- 通用性支持：对于不满足向量化条件的场景或 float32 类型，回退到通用的 CUDA 实现，确保功能正确性。
- 高并发设计：根据 `num_tokens` 动态调整 Block Size（256 或 1024），在处理长序列或大 Batch 时最大化 SM 占用率。
### 参数说明
- **input** (torch.Tensor, 入参/出参): 输入张量，形状为 `[..., hidden_size]`。计算开始时作为当前层的激活输入，计算结束后存储归一化后的结果
- **residual** (torch.Tensor, 入参/出参): 残差张量，形状为 `[..., hidden_size]`。计算开始时作为前一层的残差，计算结束后存储 `input + residual` 的结果
- **weight** (torch.Tensor, 入参): 归一化权重（Scale/Gamma），形状为 `[hidden_size]`
- **epsilon** (float, 入参): 数值稳定性常数，防止分母为 0
### 返回值
无返回值，计算结果原地更新至 `input` 和 `residual` 张量中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `input`、`residual` 和 `weight` 的数据类型必须一致
- `residual` 和 `weight` 必须在内存中连续存储 (`is_contiguous`)
- `input` 和 `residual` 的形状必须一致
- 为了触发高性能向量化路径，建议 `hidden_size` 为 8 的倍数，且张量首地址满足 16 字节对齐
- 支持的数据类型：float16, bfloat16, float32
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 42
DEVICE = "cuda"
DTYPE = torch.float16 # 推荐使用 fp16 或 bf16 以触发向量化优化
BATCH_SIZE = 4
SEQ_LEN = 1024
HIDDEN_SIZE = 4096
EPSILON = 1e-5
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 创建输入 Tensor
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE)
residual_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE)
weight = torch.ones(HIDDEN_SIZE, dtype=DTYPE)
# 确保张量连续，以满足算子约束
if not input_tensor.is_contiguous():
    input_tensor = input_tensor.contiguous()
if not residual_tensor.is_contiguous():
    residual_tensor = residual_tensor.contiguous()
if not weight.is_contiguous():
    weight = weight.contiguous()
# 打印原始值用于对比（可选）
print(f"Input Mean (Pre): {input_tensor.mean().item():.4f}")
print(f"Residual Mean (Pre): {residual_tensor.mean().item():.4f}")
# ================= 算子执行 =================
print(f"Running fused_add_rms_norm...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 注意：这是一个 In-place 操作
torch.ops._C.fused_add_rms_norm(
    input_tensor,
    residual_tensor,
    weight,
    EPSILON
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Input Mean (Post - Normalized): {input_tensor.mean().item():.4f}")
print(f"Residual Mean (Post - Summed): {residual_tensor.mean().item():.4f}")
# 验证基本性质 (RMS Norm 输出的 RMS 应接近 1，如果 weight 全为 1)
rms_value = torch.sqrt(input_tensor.pow(2).mean(dim=-1) + EPSILON)
print(f"Output RMS mean (should be approx 1.0): {rms_value.mean().item():.4f}")
```
## 51. apply_repetition_penalties_
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def apply_repetition_penalties_(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor
) -> None:
    pass
```
### 功能描述
apply_repetition_penalties_ 算子实现对 logits 的重复惩罚（Repetition Penalty）机制。该算子是 vLLM 推理采样阶段的关键组件，用于根据序列中已出现的 token（包括输入的 prompt 和已生成的 output）对当前的 logits 分数进行惩罚，从而降低模型重复生成相同内容的概率。
该算子通过自定义 CUDA 内核 apply_repetition_penalties_kernel 并行执行，支持对批次内的每个序列独立应用不同的惩罚系数。操作直接在 logits 张量上进行原地（in-place）修改。
- 计算逻辑：
    对于序列 $i$ 中的第 $j$ 个词表词汇，算子首先检查其是否在历史序列中出现过。判定依据是 prompt_mask 或 output_mask 在对应位置是否为真。如果该词汇已出现，则根据 logits 的正负情况应用惩罚系数 $p_i$：
    $$
    \text{logits}_{i, j} = \begin{cases} \frac{\text{logits}_{i, j}}{p_i} & \text{if } \text{logits}_{i, j} > 0 \\ \text{logits}_{i, j} \times p_i & \text{if } \text{logits}_{i, j} \le 0 \end{cases}
    $$
    其中：
    - $\text{logits}_{i, j}$ 是第 $i$ 个序列中第 $j$ 个 token 的原始分数
    - $p_i$ 是第 $i$ 个序列对应的重复惩罚系数 (`repetition_penalties`)
    - 惩罚通常设置 $p_i > 1.0$。对于正数 logit，除以 $p_i$ 使其变小（概率降低）；对于负数 logit，乘以 $p_i$ 使其更小（负得更多，概率降低）。
**主要功能包括：**
- 高效的 CUDA 并行计算，每个线程块处理一个序列及一部分词表 tile，极大提升处理大词表（Vocab Size）时的吞吐量。
- 支持 Prompt 阶段和 Decoding 阶段的混合掩码检查，通过 `prompt_mask` 和 `output_mask` 共同决定是否惩罚。
- 原地修改 logits，避免额外的内存分配和拷贝开销。
- 能够处理批次中不同序列长度和不同惩罚系数的动态情况。
### 参数说明
- **logits** (torch.Tensor, 入参/出参): 输入的 logits 张量，将被原地修改。形状为 `[num_seqs, vocab_size]`
- **prompt_mask** (torch.Tensor, 入参): 提示词掩码，标记 prompt 中出现过的 token。形状为 `[num_seqs, vocab_size]`，类型通常为 bool 或 int
- **output_mask** (torch.Tensor, 入参): 输出掩码，标记生成过程中出现过的 token。形状为 `[num_seqs, vocab_size]`，类型通常为 bool 或 int
- **repetition_penalties** (torch.Tensor, 入参): 每个序列的重复惩罚系数。形状为 `[num_seqs]`，类型需与 logits 保持一致
### 返回值
无返回值，计算结果直接更新在输入张量 logits 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 所有输入张量必须是内存连续的（Contiguous）
- `logits`、`prompt_mask` 和 `output_mask` 的维度 `[num_seqs, vocab_size]` 必须严格匹配
- `repetition_penalties` 的维度必须为 `[num_seqs]`
- `num_seqs` 必须大于 0
- 支持的数据类型：float16, bfloat16, float32
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
NUM_SEQS = 4
VOCAB_SIZE = 32000
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Logits
logits = torch.randn(NUM_SEQS, VOCAB_SIZE, dtype=DTYPE)
# 2. 创建 Masks (模拟部分 token 已出现)
# 随机生成 mask，约 10% 的 token 被标记为已出现
prompt_mask = torch.rand(NUM_SEQS, VOCAB_SIZE, device=DEVICE) > 0.9
output_mask = torch.rand(NUM_SEQS, VOCAB_SIZE, device=DEVICE) > 0.9
# 3. 创建惩罚系数
# 设置惩罚系数为 1.2
repetition_penalties = torch.full((NUM_SEQS,), 1.2, dtype=DTYPE)
# ================= 算子执行 =================
print(f"Running apply_repetition_penalties_...")
print(f"Logits shape: {logits.shape}")
print(f"Sample logit before (Sequence 0, Token 100): {logits[0, 100].item():.4f}")
print(f"Is repeated: {prompt_mask[0, 100] or output_mask[0, 100]}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 注意：Mask 需要确保是 contiguous 的，通常在传入前处理
if not prompt_mask.is_contiguous():
    prompt_mask = prompt_mask.contiguous()
if not output_mask.is_contiguous():
    output_mask = output_mask.contiguous()
torch.ops._C.apply_repetition_penalties_(
    logits,
    prompt_mask,
    output_mask,
    repetition_penalties
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Sample logit after (Sequence 0, Token 100): {logits[0, 100].item():.4f}")
```
## 52. rms_norm_static_fp8_quant
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def rms_norm_static_fp8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float
) -> None:
    pass
```
### 功能描述
rms_norm_static_fp8_quant 算子实现融合了静态 FP8 量化的均方根层归一化（RMSNorm）。该算子通常用于大模型推理阶段，旨在将层归一化（Layer Normalization）与后续的激活值量化操作融合，减少内存访问次数并提升计算吞吐量。
此实现直接读取输入张量，计算其均方根（RMS），应用归一化和权重缩放，然后利用给定的静态缩放因子（scale）将结果量化为 FP8 格式并存入输出张量。
- 计算公式：
    首先计算输入向量 $x$ 的均方根：
    $$
    \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
    $$
    然后进行归一化并乘以权重 $w$：
    $$
    y_i = \frac{x_i}{\text{RMS}(x)} \cdot w_i
    $$
    最后进行静态 FP8 量化：
    $$
    \text{out}_i = \text{scaled\_fp8\_conversion}(y_i, \text{scale}^{-1})
    $$
    其中：
    - $x \in \mathbb{R}^{N \times D}$ 是输入张量 (`input`)
    - $w \in \mathbb{R}^{D}$ 是权重张量 (`weight`)
    - $d$ 是隐藏层维度 (`hidden_size`)
    - $\epsilon$ 是数值稳定性常数 (`epsilon`)
    - $\text{scale}$ 是量化缩放因子，张量形状为 `[1]`
    - $\text{scaled\_fp8\_conversion}$ 表示将浮点数转换为 FP8 格式的操作
**主要功能包括：**
- 高效计算 RMSNorm，核心算法通过 `BlockReduce` 优化方差规约求和。
- 融合了 FP8 量化操作，直接输出量化后的张量，避免了中间 FP16/BF16 结果的显存回写。
- 使用向量化读写（Vectorized Operations）优化内存带宽利用率。
- 支持 FP16 和 BF16 输入类型，输出为 FP8 类型。
- 针对不同的隐藏层维度（hidden_size）和对齐方式，自动分发到最优的向量化内核实现。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，FP8 类型，形状为 `[..., hidden_size]`，存储归一化并量化后的结果
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., hidden_size]`
- **weight** (torch.Tensor, 入参): 归一化权重（gamma），形状为 `[hidden_size]`
- **scale** (torch.Tensor, 入参): 静态量化缩放因子，形状为 `[1]`，包含一个浮点数值
- **epsilon** (float, 入参): 用于数值稳定性的极小值，防止除零错误
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输出张量 `out` 必须是内存连续的（Contiguous）
- `input` 和 `weight` 的数据类型必须一致（通常为 FP16 或 BF16）
- `scale` 必须是一个包含单个浮点数的 Tensor
- 算子内部会根据 `input` 的数据类型和 `out` 的 FP8 具体类型（如 E4M3 或 E5M2）进行模板分发
- 针对大 Token 数场景（`num_tokens`），算子会自动调整 Grid 和 Block 大小以优化 SM 占用率
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
FP8_DTYPE = torch.float8_e4m3fn  # 假设环境支持该 FP8 类型，或者使用 uint8 代替
# 维度设置
BATCH_SIZE = 16
SEQ_LEN = 512
HIDDEN_SIZE = 4096
EPSILON = 1e-6
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE)
weight = torch.ones(HIDDEN_SIZE, dtype=DTYPE)
scale = torch.tensor([1.0], dtype=torch.float32)
# 2. 准备输出 Tensor (FP8)
# 注意：输出形状需与输入一致
out = torch.empty_like(input_tensor, dtype=FP8_DTYPE)
# ================= 算子执行 =================
print(f"Running rms_norm_static_fp8_quant...")
print(f"Input: {input_tensor.shape}, Type: {input_tensor.dtype}")
print(f"Output: {out.shape}, Type: {out.dtype}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.rms_norm_static_fp8_quant(
    out,
    input_tensor,
    weight,
    scale,
    EPSILON
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# 简单验证输出不全为0
print(f"Output non-zero count: {out.count_nonzero().item()}")
```
## 53. fused_add_rms_norm_static_fp8_quant
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def fused_add_rms_norm_static_fp8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float
) -> None:
    pass
```
### 功能描述
fused_add_rms_norm_static_fp8_quant 算子实现了融合的残差相加（Residual Add）、RMS 归一化（Root Mean Square Layer Normalization）以及静态 FP8 量化功能。
该算子旨在优化 Transformer 类模型中常见的 "Add + RMSNorm" 结构，并直接输出量化后的 FP8 数据，从而减少内存读写带宽占用并加速后续计算。算子首先将输入张量 input 与残差张量 residual 相加，更新 residual，然后对相加后的结果进行 RMS 归一化，最后利用给定的静态缩放因子 scale 将结果量化为 FP8 格式写入 out。
- 计算公式：
    1. 残差更新 (Residual Add):
        $$
        x_{sum} = input + residual
        $$
        同时更新显存中的 residual 张量：$residual \leftarrow x_{sum}$
    2. RMS 归一化 (RMS Normalization):
        $$
        y = \frac{x_{sum}}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_{sum, i}^2 + \epsilon}} \times weight
        $$
        其中 $n$ 是隐藏层维度 (hidden_size)，$\epsilon$ 是防止除零的微小常数。
    3. 静态 FP8 量化 (Static FP8 Quantization):
        $$
        out = \text{cast\_to\_fp8}\left (y \times \frac{1}{scale}\right)
        $$
        此处使用给定的静态 scale 对归一化后的结果进行缩放，并转换为 FP8 数据类型。
**主要功能包括：**
- 算子融合：将元素级相加、归一化统计量计算、元素级乘法和类型转换融合在单个 CUDA 内核中执行，显著降低显存访问延迟。
- 原地更新：直接在显存中更新 `residual` 张量，避免了中间显存分配。
- 向量化读取：针对满足对齐条件（16 字节对齐）和特定维度（hidden_size 可被 8 整除）的输入，利用 CUDA 的向量化加载指令（如 `ld.global.v4` 等）提升内存带宽利用率。
- FP8 支持：直接生成 FP8 格式的输出张量，支持后续的 FP8 矩阵乘法（GEMM）操作。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，存储量化后的 FP8 数据，形状为 `[..., hidden_size]`。
- **input** (torch.Tensor, 入参): 输入张量，通常为上一层的输出，形状为 `[..., hidden_size]`。
- **residual** (torch.Tensor, 入参/出参): 残差张量，形状与 `input` 相同。算子会读取该张量并原地更新相加后的结果。
- **weight** (torch.Tensor, 入参): RMS 归一化的权重（Gamma），形状为 `[hidden_size]`。
- **scale** (torch.Tensor, 入参): 静态量化缩放因子，标量张量（包含 1 个 float 元素），用于将归一化后的结果缩放到 FP8 表示范围。
- **epsilon** (float, 入参): 归一化计算中的数值稳定常数。
### 返回值
无返回值，计算结果直接写入输出张量 `out` 中，并原地更新 `residual`。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- `out` 和 `residual` 张量必须在内存中连续存储 (`is_contiguous`)。
- `input`、`residual` 和 `weight` 的数据类型必须一致（通常为 Float16 或 BFloat16）。
- `out` 的数据类型应为 FP8 类型（如 `torch.float8_e4m3fn` 或 `torch.float8_e5m2`）。
- 输入张量的最后一维大小必须等于 `hidden_size`。
### 调用示例
```Python
import torch
import mcoplib._C 
# 配置参数
SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN_SIZE = 4096
NUM_TOKENS = 128
EPSILON = 1e-6
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 创建输入数据
input_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=DTYPE)
residual_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=DTYPE)
weight = torch.ones(HIDDEN_SIZE, dtype=DTYPE)
# 创建 FP8 输出张量 (模拟 FP8 类型，具体取决于 PyTorch 版本支持)
# 假设使用 E4M3 格式
out_tensor = torch.empty(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float8_e4m3fn)
# 量化 Scale (标量)
scale = torch.tensor([1.0], dtype=torch.float32)
# ================= 算子执行 =================
print(f"Running fused_add_rms_norm_static_fp8_quant...")
print(f"Input: {input_tensor.shape}, Residual: {residual_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 注意：residual_tensor 会被原地修改
torch.ops._C.fused_add_rms_norm_static_fp8_quant(
    out_tensor,
    input_tensor,
    residual_tensor,
    weight,
    scale,
    EPSILON
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output type: {out_tensor.dtype}")
print(f"Output shape: {out_tensor.shape}")
```
## 56. batched_rotary_embedding
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def batched_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: torch.Tensor
) -> None:
    pass
```
### 功能描述
batched_rotary_embedding 算子用于对查询（Query）和键（Key）张量应用批量旋转位置编码（Batched Rotary Positional Embeddings）。该算子是 vLLM 中处理多 LoRA 或多模态场景下的核心 CUDA 内核，它允许不同的 token 根据 cos_sin_cache_offsets 索引到 cos_sin_cache 的不同区域，从而灵活地支持异构的位置编码检索。
算子在原地（In-place）修改输入张量，支持 GPT-NeoX 和 GPT-J 两种主流的旋转策略。
- 计算公式：
    对于特征向量中的每一对元素 $(x, y)$，根据其对应的旋转角度 $\theta$，旋转后的值 $(x', y')$ 计算如下：
    $$
    \begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}
    $$
- 缓存检索机制：
    不同于普通 RoPE，该算子引入了 cos_sin_cache_offsets。对于第 $i$ 个 token，其位置为 pos，对应的旋转参数（cos/sin）在缓存中的地址计算如下：
    $$
    \text{cache\_ptr} = \text{cos\_sin\_cache} + (\text{cos\_sin\_cache\_offsets}[i] + \text{positions}[i]) \times \text{rot\_dim}
    $$
    这使得批次中的不同序列可以使用缓存中不同的段。
- 旋转维度配对策略（根据 `is_neox` 参数）：
    - GPT-NeoX 模式 (is_neox=True)：
        将特征维度分为前半部分和后半部分进行配对。对于偏移量 $j$，配对索引为 $j$ 和 $j + D/2$（其中 $D$ 为旋转维度 rot_dim）。
        $$
        \text{Pair: } (v[j], v[j + D/2])
        $$
    - GPT-J 模式 (is_neox=False)：
        将相邻的两个元素进行配对。对于偏移量 $j$，配对索引为 $2j$ 和 $2j+1$。
        $$
        \text{Pair: } (v[2j], v[2j+1])
        $$
**主要功能包括：**
- 支持批量处理不同偏移量的位置编码缓存读取，适用于复杂服务场景。
- 高效的 CUDA 内核实现，每个线程块处理一个 token，原地修改 Query 和 Key。
- 支持 Query 和 Key 具有不同数量的头（如 GQA/MQA 场景）。
- 支持 `positions` 和 `cos_sin_cache_offsets` 为 1D（扁平化 tokens）或 2D（[batch, seq]）格式。
- 自动处理张量步长（Stride），支持非连续内存布局。
- 兼容多种浮点数据类型（通过 `VLLM_DISPATCH_FLOATING_TYPES` 分发）。
### 参数说明
- **positions** (torch.Tensor, 入参): 位置索引张量，形状为 `[batch_size, seq_len]` 或 `[num_tokens]`
- **query** (torch.Tensor, 入参/出参): 查询张量，计算后会被原地修改。形状支持 `[batch_size, seq_len, num_heads * head_size]` 或其扁平化/解包变体
- **key** (torch.Tensor, 入参/出参): 键张量，计算后会被原地修改。形状逻辑同 query，支持 GQA/MQA
- **head_size** (int, 入参): 注意力头的维度大小
- **cos_sin_cache** (torch.Tensor, 入参): 预计算的 RoPE 缓存，形状为 `[max_position, rot_dim]` 或等效的扁平结构
- **is_neox** (bool, 入参): 标志位，True 表示使用 GPT-NeoX 风格，False 表示使用 GPT-J 风格
- **rot_dim** (int, 入参): 实际参与旋转的维度大小（通常等于 head_size 或其一部分）
- **cos_sin_cache_offsets** (torch.Tensor, 入参): 缓存偏移量张量，形状与 `positions` 相同，用于指示每个 token 在 cos_sin_cache 中的起始偏移
### 返回值
无返回值，计算结果直接原地更新到 query 和 key 张量中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `positions` 和 `cos_sin_cache_offsets` 的元素数量必须一致，且等于 `query` 和 `key` 中的 token 总数
- `rot_dim` 必须是偶数，且 `cos_sin_cache` 的第二维必须能容纳 `rot_dim`
- `query` 的头数必须是 `key` 头数的整数倍
- 支持的数据类型：float16, bfloat16, float32
- 算子内部会根据 token 索引自动计算 query 和 key 的内存偏移，要求输入张量步长信息正确
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
BATCH_SIZE = 2
SEQ_LEN = 128
NUM_HEADS = 32
NUM_KV_HEADS = 32
HEAD_SIZE = 128
ROT_DIM = 128
MAX_POS = 4096
IS_NEOX = True
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
num_tokens = BATCH_SIZE * SEQ_LEN
positions = torch.arange(SEQ_LEN, device=DEVICE).repeat(BATCH_SIZE, 1)
# Query 和 Key
query = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
key = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE)
# Cos/Sin Cache 和 Offsets
# 假设所有 sequences 使用相同的 cache 区域，offsets 设为 0
cos_sin_cache = torch.randn(MAX_POS, ROT_DIM, dtype=DTYPE)
cos_sin_cache_offsets = torch.zeros_like(positions, dtype=torch.int64)
# ================= 算子执行 =================
print(f"Running batched_rotary_embedding...")
print(f"Query: {query.shape}, Key: {key.shape}, Rot Dim: {ROT_DIM}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.batched_rotary_embedding(
    positions,
    query,
    key,
    HEAD_SIZE,
    cos_sin_cache,
    IS_NEOX,
    ROT_DIM,
    cos_sin_cache_offsets
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Query Output Mean: {query.mean().item():.4f}")
```
## 57. awq_gemm
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def awq_gemm(
    input: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    split_k_iters: int,
    temp_space: torch.Tensor,
    dtype_bf16: bool
) -> torch.Tensor:
    pass
```
### 功能描述
`awq_gemm` 算子实现基于 AWQ (Activation-aware Weight Quantization) 的高效矩阵乘法。该算子是 vLLM 中用于加速量化模型推理的核心 CUDA 内核启动器。它接收激活值（输入）、量化后的权重（内核）、缩放因子和零点，通过 `vllm::awq::launch_gemm` 接口在计算过程中动态反量化权重并执行矩阵乘法运算。此实现支持 Split-K 策略，通过 `split_k_iters` 和 `_temp_space` 参数允许将计算分解为多个部分并行执行，最后归约结果，从而在不同负载下优化性能。它原生支持 FP16 和 BF16 数据类型的输入与计算。
- 计算公式： AWQ 矩阵乘法执行以下计算：
    Y=X⋅WT
    其中权重矩阵 W 由量化参数重建：
    Wij​=(Wijpacked​−Zg​)⋅Sg​
    其中：
    - X 是输入激活张量 (`input`)
    - Wpacked 是压缩存储的整数权重 (`kernel`)
    - Zg​ 是对应分组的零点 (`zeros`)
    - Sg​ 是对应分组的缩放因子 (`scaling_factors`)
    - g 是基于 `group_size` 计算的组索引，通常 g=⌊j/group_size⌋
- 动态反量化与计算： 算子在 GPU 内核中融合了反量化和 GEMM 操作。它不显式存储完整的 W 矩阵，而是直接从 `kernel` 读取压缩的 4-bit 权重数据，结合 `scaling_factors` 和 `zeros` 将其恢复为 FP16/BF16 精度，并立即与输入 `input` 进行乘法累加运算。
- 精度与类型支持： 算子通过 `dtype_bf16` 参数动态选择计算精度。
    - 当 `dtype_bf16=True` 时，使用 `__maca_bfloat16` 类型进行计算。
    - 当 `dtype_bf16=False` 时，使用 `half` (FP16) 类型进行计算。
**主要功能包括：**
- 实现 AWQ 算法的量化矩阵乘法，支持 4-bit 权重压缩。
- 支持 Split-K 并行策略，利用临时缓冲区 `temp_space` 提升大矩阵乘法的并行度。
- 自动计算 `group_size`（组大小），基于输入通道数和缩放因子尺寸推断。
- 支持多种混合精度模式（BF16/FP16），适配不同的模型精度需求。
- 输出张量根据 `kernel` 的维度自动创建，无需外部预分配输出内存。
### 参数说明
- **input** (torch.Tensor, 入参): 输入激活张量，形状通常为 `[num_in_feats, num_in_channels]`
- **kernel** (torch.Tensor, 入参): 量化权重张量，存储压缩后的权重数据
- **scaling_factors** (torch.Tensor, 入参): 反量化缩放因子，形状与权重分组对应
- **zeros** (torch.Tensor, 入参): 反量化零点张量，形状与权重分组对应，通常为压缩存储的整数
- **split_k_iters** (int, 入参): Split-K 迭代次数，控制 GEMM 的并行切分粒度
- **temp_space** (torch.Tensor, 入参): 临时缓冲区，用于 Split-K 计算时的中间结果存储
- **dtype_bf16** (bool, 入参): 数据类型标志，`True` 表示使用 BFloat16，`False` 表示使用 Float16
### 返回值
返回计算结果张量 (torch.Tensor)，形状为 `[num_in_feats, num_out_channels]`
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入 `input` 的数据类型必须与 `dtype_bf16` 指示的精度一致
- `kernel`、`scaling_factors` 和 `zeros` 的维度必须满足 AWQ 的打包格式要求
- `group_size` 必须由 `num_in_channels / scaling_factors.size (0)` 正确整除
- `temp_space` 的大小必须足够容纳 Split-K 的中间结果（如果 `split_k_iters > 1`）
- 内存布局要求：输入张量通常建议是连续存储的
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
M = 16      # num_in_feats
K = 4096    # num_in_channels
N = 4096    # num_out_channels
GROUP_SIZE = 128
PACK_FACTOR = 8  # 假设 4-bit 权重 packed 到 int32 (32/4=8)
# 初始化环境
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# Input features
input_feats = torch.randn(M, K, dtype=DTYPE)
# Quantized Weights (模拟 packed data)
# Kernel layout dependency usually [N, K // PACK_FACTOR] or [K, N // PACK_FACTOR]
# 这里假设 shape 符合算子内部推导 num_out_channels = kernel.size(0)
kernel = torch.randint(0, 2**31, (N, K // PACK_FACTOR), dtype=torch.int32)
# Scaling factors
scaling_factors = torch.randn(K // GROUP_SIZE, N, dtype=DTYPE)
# Zeros (模拟 packed zeros)
zeros = torch.randint(0, 2**31, (K // GROUP_SIZE, N // PACK_FACTOR), dtype=torch.int32)
# Split-K 临时空间
split_k_iters = 1
temp_space = torch.empty(M * N * split_k_iters, dtype=torch.float32)
# 类型标志
is_bf16 = (DTYPE == torch.bfloat16)
# ================= 算子执行 =================
print(f"Running awq_gemm...")
print(f"Input: {input_feats.shape}, Kernel: {kernel.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用 C++ 扩展绑定的接口
output = torch.ops._C.awq_gemm(
    input_feats,
    kernel,
    scaling_factors,
    zeros,
    split_k_iters,
    temp_space,
    is_bf16
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Shape: {output.shape}")
print(f"Output Mean: {output.mean().item():.4f}")
```
## 58. awq_dequantize
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def awq_dequantize(
    _kernel: torch.Tensor,
    _scaling_factors: torch.Tensor,
    _zeros: torch.Tensor,
    split_k_iters: int,
    thx: int,
    thy: int
) -> torch.Tensor:
    pass
```
### 功能描述
awq_dequantize 算子用于实现 AWQ (Activation-aware Weight Quantization) 算法中的反量化过程。该算子是 vLLM 等大模型推理框架中用于权重恢复或调试分析的核心工具。它将经过 4-bit 量化并打包存储的权重张量，结合缩放因子（Scaling Factors）和零点（Zeros），还原为 float16 精度的高精度权重。
此实现利用 CUDA 内核 dequantize_weights 并行地对打包的 int32 数据进行解包、反量化计算，支持按组（Group）量化策略，能够高效地处理大语言模型的权重矩阵。
- 计算公式：
    反量化过程遵循标准的线性量化公式：
    $$
    W = (W_{int} - Z) \times S
    $$
    其中：
    - $W \in \mathbb{R}^{IC \times OC}$ 是输出的解量化权重 (`_de_kernel`)
    - $W_{int}$ 是从 `_kernel` 中解包出的 4-bit 整数权重
    - $Z$ 是从 `_zeros` 中解包出的 4-bit 零点
    - $S$ 是缩放因子 (`_scaling_factors`)
    - $IC$ 是输入通道数 (Input Channels)
    - $OC$ 是输出通道数 (Output Channels)
- 权重打包与布局：
    - 权重 (`_kernel`): 存储为 `int32` 类型。每个 `int32` 包含 8 个 4-bit 的权重参数。逻辑形状为 `[IC, OC]`，物理存储形状为 `[IC, OC / 8]`。
    - 零点 (`_zeros`): 同样存储为 `int32` 类型，采用与权重相同的打包方式。形状受组大小 $G$ 影响，为 `[IC / G, OC / 8]`。
    - 缩放因子 (`_scaling_factors`): 存储为 `float16` 类型，未打包。形状为 `[IC / G, OC]`。
**主要功能包括：**
- 将 4-bit Packed Int32 格式的量化权重解包并转换为 FP16 格式。
- 应用零点偏移（Zero Point）和缩放因子（Scale）进行数值恢复。
- 支持分组量化（Groupwise Quantization），通过 $G = IC / \text{scales.size}(0)$ 自动推导组大小。
- 允许通过 `thx` 和 `thy` 参数自定义 CUDA kernel 的线程块配置，以适应不同的硬件特性（若设为 0 则使用默认启发式配置）。
### 参数说明
- `_kernel` (torch.Tensor, 入参): 量化的权重张量，数据类型为 `int32`。形状为 `[in_c, qout_c]`，其中 `qout_c = out_c / 8`。
- `_scaling_factors` (torch.Tensor, 入参): 缩放因子张量，数据类型通常为 `float16`。形状为 `[in_c / G, out_c]`，其中 $G$ 为量化组大小。
- `_zeros` (torch.Tensor, 入参): 零点张量，数据类型为 `int32`。形状为 `[in_c / G, qout_c]`。
- split_k_iters (int, 入参): Split-K 迭代次数。在此反量化算子中主要作为接口预留参数，实际核心计算逻辑主要依赖权重维度，但在部分上下文可能用于控制流。
- thx (int, 入参): CUDA 线程块在 X 维度（对应输出通道方向）的大小。若为 0，则自动根据 `qout_c` 设定。
- thy (int, 入参): CUDA 线程块在 Y 维度（对应输入通道方向）的大小。若为 0，则自动根据 `in_c` 设定。
### 返回值
返回反量化后的权重张量，类型与 `_scaling_factors` 相同（通常为 `float16`），形状为 `[in_c, out_c]`。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- `out_c` 必须是 8 的倍数（因为使用 int32 打包 8 个 4-bit 权重）。
- `in_c` 必须能被组大小 $G$ 整除。
- `_zeros` 的维度必须与 `_scaling_factors` 在组维度上匹配，在通道维度上与 `_kernel` 匹配。
- 仅支持 4-bit 量化权重的解包。
### 调用示例
```Python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
IN_C = 128
OUT_C = 64
GROUP_SIZE = 64
SPLIT_K_ITERS = 0
THX = 0
THY = 0
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
num_groups = IN_C // GROUP_SIZE
qout_c = OUT_C // 8
kernel = torch.randint(-2000000000, 2000000000, (IN_C, qout_c), dtype=torch.int32)
scaling_factors = torch.randn(num_groups, OUT_C, dtype=DTYPE)
zeros = torch.randint(-2000000000, 2000000000, (num_groups, qout_c), dtype=torch.int32)
# ================= 算子执行 =================
print(f"Running awq_dequantize...")
print(f"Kernel: {kernel.shape}, Scales: {scaling_factors.shape}, Zeros: {zeros.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
output = torch.ops._C.awq_dequantize(
    kernel,
    scaling_factors,
    zeros,
    SPLIT_K_ITERS,
    THX,
    THY
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {output.mean().item():.4f}")
```
## 59. awq_to_gptq_4bit
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
torch::Tensor awq_to_gptq_4bit(torch::Tensor qweight);
```
### 功能描述
`awq_to_gptq_4bit` 算子用于将 AWQ（Activation-aware Weight Quantization）格式的 4-bit 量化权重转换为 GPTQ（Marlin）格式。该算子是 vLLM 中用于权重重排和格式转换的核心工具，旨在适配 Marlin Kernel 的计算需求。它通过 CUDA 内核重新打包权重数据，使其符合 Marlin GEMM 的内存布局要求，从而加速推理过程。
此实现通过重新计算输出维度并调用特定的 CUDA 核函数 `awq_to_gptq_4bit_kernel`，将原本按 AWQ 格式排列的权重数据转换为紧凑的 Marlin 格式。
- 维度计算与转换逻辑：
    算子首先解析输入张量 qweight 的维度，并根据 Marlin 的打包比率（PACK_RATIO）计算输出张量的形状。
    $$
    \text{num\_in\_channels} = \text{qweight.size}(0)
    $$
    $$
    \text{num\_out\_channels} = \text{qweight.size}(1) \times 8
    $$
    为了适应 Marlin 的内存布局，输出张量的维度 compact_n 和 compact_output_k 计算如下（其中 $R$ 为 PACK_RATIO_4BITS）：
    $$
    \text{compact\_n} = \frac{\text{num\_out\_channels} + R - 1}{R}
    $$
    $$
    \text{compact\_output\_k} = \frac{\text{num\_in\_channels} + R - 1}{R}
    $$
- 网格与块配置：
    算子使用固定的 block_size = 256。
    计算总的 tile 数量：
    $$
    \text{tile\_all\_num} = \text{compact\_n} \times \text{compact\_output\_k}
    $$
    计算 CUDA 网格大小：
    $$
    \text{grid\_size} = \frac{\text{tile\_all\_num} + 255}{256}
    $$
**主要功能包括：**
- 解析 AWQ 格式的输入权重维度。
- 计算转换后的 GPTQ/Marlin 格式所需的紧凑维度。
- 在 CUDA 设备上分配全零的输出张量。
- 启动 CUDA 内核，并行地将数据从 AWQ 布局重排为 GPTQ Marlin 布局。
### 参数说明
- **qweight** (torch.Tensor, 入参): 输入的 AWQ 量化权重张量。通常为 `int32` 类型，形状为 `[num_in_channels, num_out_channels / 8]`。数据存储在 CUDA 设备上。
### 返回值
- **out** (torch.Tensor): 转换后的 GPTQ 格式权重张量。形状为 `[num_out_channels, compact_output_k]`，数据类型与输入 `qweight` 相同，存储在 CUDA 设备上。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- 输入 `qweight` 必须是连续存储的。
- 算子内部假设打包比率（PACK_RATIO_4BITS）固定，这通常取决于底层 Marlin 实现（例如 32/4-bit = 8 或其他特定硬件配置）。
- `block_size` 固定为 256。
- 输出张量的第一维是 `num_out_channels`，这意味着在转换过程中发生了逻辑上的转置或重排。
### 调用示例
```Python
import torch
import torch.nn as nn
# 配置参数
SEED = 0
DEVICE = "cuda"
# 维度设置
# 假设 in_channels=128, out_channels=256
# AWQ 4-bit 打包通常将 8 个 4-bit 元素打包进一个 int32
# 因此 qweight 的第二维是 out_channels / 8 = 32
NUM_IN_CHANNELS = 128
NUM_OUT_CHANNELS = 256
PACK_FACTOR = 8 
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 qweight Tensor (模拟 AWQ 格式权重)
qweight = torch.randint(
    low=0, 
    high=2**31-1, 
    size=(NUM_IN_CHANNELS, NUM_OUT_CHANNELS // PACK_FACTOR), 
    dtype=torch.int32,
    device=DEVICE
)
print(f"Running awq_to_gptq_4bit...")
print(f"Input qweight shape: {qweight.shape}")
# ================= 算子执行 =================
if torch.cuda.is_available():
    torch.cuda.synchronize()
output = torch.ops._C.awq_to_gptq_4bit(qweight)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output shape: {output.shape}")
```
## 60. cutlass_moe_mm_gemm_kernel_m_w8a8
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
int64_t cutlass_moe_mm_gemm_kernel_m_w8a8(
    int64_t num_valid_tokens,
    int64_t N,
    int64_t K,
    int64_t group
);
```
### 功能描述
cutlass_moe_mm_gemm_kernel_m_w8a8 算子是一个用于混合专家模型（MoE）的高效 GEMM（通用矩阵乘法）计算接口。该函数作为封装层，内部保持调用 cutlass_moe_mm_gemm_kernel_m_w8a8_sm75，旨在利用 CUTLASS 库在 GPU 上执行优化的 INT8 矩阵乘法并输出 BFloat16 结果。
此算子通过 mctlassMoeGemm 模板类配置了特定的数据类型和布局，实现了针对大语言模型推理场景的低精度高性能计算。
- 计算公式：
    $$
    C = \text{convert}_{\text{BF16}}(A \times B)
    $$
    其中：
    - $A$ 是输入矩阵（激活值），数据类型为 `int8_t`，布局为 `RowMajor`。
    - $B$ 是权重矩阵，数据类型为 `int8_t`，布局为 `ColumnMajor`。
    - $C$ 是输出矩阵，数据类型为 `maca_bfloat16`，布局为 `RowMajor`。
    - 内部累加计算使用 `float` 类型。
- 核心执行逻辑：
    1. 问题规模定义：根据输入的 `num_valid_tokens` (M)、`N`、`K` 和 `group` 构建 `BatchedGemmCoord` 对象。
    2. 内核选择：调用 `gemm_kernel_selector` 根据问题规模、计算类型、布局类型等参数选择最合适的 GEMM 内核配置。
    3. 分块计算：通过 `get_gemm_kernel_tile` 获取内核的切片（Tile）大小。
    4. 状态检查：验证所选内核是否支持当前配置（`GEMM_NOT_SUPPORTED`），若不支持则返回错误状态。
    5. 返回值：执行成功后，返回内核配置中的 M 维度大小 (`kernel_size.m()`)。
**主要功能包括：**
- 混合精度计算：支持 INT8 输入（权重和激活）与 BFloat16 输出，兼顾显存效率与数值范围。
- MoE 支持：通过 `group` 参数支持分组 GEMM 操作，适用于 MoE 结构中的专家并行计算。
- 自动调优：内部集成 `mctlass` 的内核选择机制，自动匹配最佳计算瓦片配置。
- SM80 架构优化：虽然函数名后缀为 `_sm75`，但内部配置使用了 `ArchTag = mctlass::arch::Sm80`，针对安培架构及兼容硬件进行了优化。
### 参数说明
- **num_valid_tokens** (int64_t, 入参): 有效 Token 数量，对应 GEMM 运算的 M 维度
- **N** (int64_t, 入参): 输出特征维度，对应 GEMM 运算的 N 维度
- **K** (int64_t, 入参): 输入特征维度，对应 GEMM 运算的 K 维度
- **group** (int64_t, 入参): 组数量，用于 MoE 场景下的 Expert 数量或 Batch 分组
### 返回值
- **int64_t**: 成功时返回所选内核的 M 维度切片大小 (`kernel_size.m()`)；若初始化或执行失败（如状态不为 `kSuccess`），则返回 -1。
### 约束与调用
- 输入张量数据类型必须为 `int8_t`，输出张量必须为 `maca_bfloat16`。
- A 矩阵（激活）必须是行主序（RowMajor）。
- B 矩阵（权重）必须是列主序（ColumnMajor）。
- 必须在支持 SM80 或兼容指令集的 GPU 设备上运行。
- `mctlass` 上下文必须正确初始化，且 `gemm_kernel_selector` 能找到匹配的内核。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
# 维度设置
NUM_VALID_TOKENS = 1024  # M
N_DIM = 4096             # N
K_DIM = 2048             # K
GROUP_SIZE = 8           # Group
# ================= 算子执行 =================
print(f"Running cutlass_moe_mm_gemm_kernel_m_w8a8...")
print(f"M: {NUM_VALID_TOKENS}, N: {N_DIM}, K: {K_DIM}, Group: {GROUP_SIZE}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用 C++ 扩展接口
# 注意：该接口直接返回内核配置的 M 维度大小，不涉及 Tensor 传递，主要用于内核选择或预计算
kernel_m_size = torch.ops._C.cutlass_moe_mm_gemm_kernel_m_w8a8(
    NUM_VALID_TOKENS,
    N_DIM,
    K_DIM,
    GROUP_SIZE
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
if kernel_m_size == -1:
    print("执行失败：不支持的配置或内核错误。")
else:
    print("执行成功。")
    print(f"Kernel M Size: {kernel_m_size}")
```
## 61. cutlass_moe_mm_w8a8
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def cutlass_moe_mm_w8a8(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    moe_weight: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    N: int,
    K: int,
    EM: int,
    num_valid_tokens: int,
    topk: int,
    mul_routed_weight: bool
) -> None:
    pass
```
### 功能描述
cutlass_moe_mm_w8a8 算子实现了基于 CUTLASS 的混合专家（MoE）矩阵乘法，专门针对 W8A8（8-bit 权重和 8-bit 激活）量化场景进行了优化。该算子利用 mctlass 库在 GPU 上执行高效的 INT8 GEMM 计算，适用于大语言模型中 MoE 层的推理加速。
它根据路由信息（token_ids 和 expert_ids），计算激活张量 a 与专家权重张量 b 之间的矩阵乘法，并将结果存储在 c 中。算子内部处理了量化缩放（通过 a_scales 和 b_scales）以及可选的路由权重乘法。
- 计算逻辑：
    算子执行的核心计算可以概括为量化矩阵乘法，并根据 mul_routed_weight 决定是否应用路由权重：
    $$
    C = \text{Convert}((A \times B) \times S_a \times S_b)
    $$
    如果 mul_routed_weight 为 True：
    $$
    C_{final} = C \times W_{route}
    $$
    其中：
    - $A$ 是 INT8 类型的激活输入 (`a`)
    - $B$ 是 INT8 类型的专家权重 (`b`)
    - $S_a, S_b$ 分别是输入和权重的量化缩放因子 (`a_scales`, `b_scales`)
    - $W_{route}$ 是路由门控权重 (`moe_weight`)
    - $C$ 是输出矩阵 (`c`)，通常为 BFloat16 格式
- 主要特性：
    - 混合精度计算：输入为 INT8，内部缩放及路由权重计算采用 FP32，输出为 BFloat16，兼顾性能与精度。
    - MoE 路由支持：原生支持 MoE 架构的稀疏计算特性，通过 `token_ids` 和 `expert_ids` 索引特定的专家进行计算，避免了对所有专家进行无效计算。
    - 高效内核：底层使用针对 SM80 架构优化的 `gemm_kernel_mnk`，支持 `RowMajor` (A) 和 `ColumnMajor` (B) 布局。
**主要功能包括：**
- 执行 MoE 层的核心 GEMM 计算。
- 支持 W8A8 量化，大幅降低显存带宽需求并提高计算吞吐量。
- 自动处理 Token 到 Expert 的映射及计算。
- 支持 Top-K 路由逻辑（通过 `topk` 参数）。
- 可选的路由权重融合乘法，减少显存访问。
### 参数说明
- **a** (torch.Tensor, 入参): 激活值输入张量，数据类型为 `int8`，通常为经过重排后的 Token 序列。
- **b** (torch.Tensor, 入参): 专家权重张量，数据类型为 `int8`，包含了所有专家的权重矩阵。
- **c** (torch.Tensor, 出参): 输出张量，数据类型通常为 `bfloat16`，用于存储计算结果。
- **a_scales** (torch.Tensor, 入参): 激活值张量的量化缩放因子，数据类型必须为 `float32`。
- **b_scales** (torch.Tensor, 入参): 权重张量的量化缩放因子，数据类型必须为 `float32`。
- **moe_weight** (torch.Tensor, 入参): 路由权重张量（Softmax output of router），表示每个 Token 分配给对应专家的权重，数据类型必须为 `float32`。
- **token_ids** (torch.Tensor, 入参): 参与计算的 Token 索引数组，用于定位有效的输入行，数据类型必须为 `int32`。
- **expert_ids** (torch.Tensor, 入参): 对应的专家索引数组，指示每个 Token 应该由哪个专家处理，数据类型必须为 `int32`。
- **num_tokens_post_padded** (torch.Tensor, 入参): 填充后的 Token 总数张量，数据类型必须为 `int32`。
- **N** (int64_t, 入参): 矩阵乘法的 N 维度（输出特征维度）。
- **K** (int64_t, 入参): 矩阵乘法的 K 维度（输入特征维度/归约维度）。
- **EM** (int64_t, 入参): 专家数量或专家乘数因子（Number of Experts）。
- **num_valid_tokens** (int64_t, 入参): 当前批次中实际参与计算的有效 Token 数量（M 维度）。
- **topk** (int64_t, 入参): 每个 Token 选择的专家数量（Top-K）。
- **mul_routed_weight** (bool, 入参): 是否将结果乘以路由权重 `moe_weight`。
### 返回值
无返回值，计算结果直接写入输出张量 c 中。如果是底层 C++ 接口调用失败，可能会返回错误码（如 -1），但在 Python 封装层通常无返回值。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- 输入张量 `a` 和 `b` 的数据类型必须为 `int8`。
- 输出张量 `c` 的数据类型应为 `bfloat16`。
- 缩放因子 `a_scales`、`b_scales` 及路由权重 `moe_weight` 的数据类型必须为 `float32`。
- 索引张量 `token_ids`、`expert_ids`、`num_tokens_post_padded` 的数据类型必须为 `int32`。
- `num_valid_tokens` 必须大于 0。
- 输入布局通常要求 A 为 RowMajor，B 为 ColumnMajor。
- 输出张量 `c` 在调用前建议进行零初始化（`torch.zeros`），以避免累加未初始化内存中的 NaN 值。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 42
DEVICE = "cuda"
DTYPE_INT8 = torch.int8
DTYPE_BF16 = torch.bfloat16
DTYPE_FLOAT = torch.float32
DTYPE_INT32 = torch.int32
# 维度设置
NUM_TOKENS = 128  # num_valid_tokens (M)
N = 4096          # Output features
K = 2048          # Input features
NUM_EXPERTS = 8   # EM
TOPK = 2
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# 激活输入 a (M, K)
a = torch.randint(-127, 127, (NUM_TOKENS, K), dtype=DTYPE_INT8)
# 权重输入 b (E, N, K)
b = torch.randint(-127, 127, (NUM_EXPERTS, N, K), dtype=DTYPE_INT8)
# 2. 输出张量 c
# 注意：必须使用 torch.zeros 初始化，避免累加垃圾值导致 NaN
c = torch.zeros((NUM_TOKENS, N), dtype=DTYPE_BF16)
# 3. 量化 Scales (必须为 float32)
a_scales = torch.ones((NUM_TOKENS, 1), dtype=DTYPE_FLOAT)
b_scales = torch.ones((NUM_EXPERTS, N, 1), dtype=DTYPE_FLOAT)
# 4. MoE 路由信息 (必须为 float32)
# 模拟权重，保证数值范围合理
moe_weight = torch.rand((NUM_TOKENS, TOPK), dtype=DTYPE_FLOAT)
# 5. 索引张量 (必须为 int32)
# 模拟 Token 和 Expert IDs
token_ids = torch.arange(NUM_TOKENS, dtype=DTYPE_INT32)
expert_ids = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS,), dtype=DTYPE_INT32)
num_tokens_post_padded = torch.tensor([NUM_TOKENS], dtype=DTYPE_INT32)
# 其他参数
mul_routed_weight = True
# ================= 算子执行 =================
print(f"Running cutlass_moe_mm_w8a8...")
print(f"A shape: {a.shape}, B shape: {b.shape}, N={N}, K={K}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.cutlass_moe_mm_w8a8(
    a,
    b,
    c,
    a_scales,
    b_scales,
    moe_weight,
    token_ids,
    expert_ids,
    num_tokens_post_padded,
    N,
    K,
    NUM_EXPERTS,
    NUM_TOKENS,
    TOPK,
    mul_routed_weight
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# 检查是否有 NaN
if torch.isnan(c).any():
    print("Warning: Output contains NaN!")
else:
    print(f"Output Mean: {c.mean().item():.4f}")
    print(f"Output Max: {c.max().item():.4f}")
```
## 62. cutlass_moe_bf16_mm
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def cutlass_moe_bf16_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    moe_weight: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    topk: int,
    mul_routed_weight: bool
) -> None:
    pass
```
### 功能描述
cutlass_moe_bf16_mm 算子实现了基于 CUTLASS 的混合专家（MoE）矩阵乘法，专门针对 BF16（BFloat16）高精度推理场景。与量化版本不同，该算子直接接受 BF16 格式的激活值和权重，利用 GPU 的 Tensor Core 执行高效的 GEMM 运算。
它根据路由信息（token_ids 和 expert_ids），计算激活张量 a 与专家权重张量 b 之间的矩阵乘法，并将结果存储在 c 中。算子支持可选的路由权重乘法，适用于对精度要求较高的大语言模型 MoE 层。
- 计算逻辑：
    算子执行的核心计算为标准的 BF16 矩阵乘法，并根据 mul_routed_weight 决定是否应用路由权重：
    $$
    C = (A \times B)
    $$
    如果 mul_routed_weight 为 True：
    $$
    C_{final} = C \times W_{route}
    $$
    其中：
    - $A$ 是 BF16 类型的激活输入 (`a`)
    - $B$ 是 BF16 类型的专家权重 (`b`)
    - $W_{route}$ 是路由门控权重 (`moe_weight`)
    - $C$ 是输出矩阵 (`out`)，为 BF16 格式
    - 内部累加计算采用 FP32 (`ElementCompute = float`) 以保证数值稳定性
- 主要特性：
    - 高精度计算：全链路支持 BFloat16 数据类型，避免了量化带来的精度损失。
    - MoE 路由支持：原生支持 MoE 架构的稀疏计算特性，通过 `token_ids` 和 `expert_ids` 索引特定的专家进行计算。
    - 高效内核：底层使用针对 SM80 架构优化的 `gemm_kernel_mnk`，支持 `RowMajor` (A) 和 `ColumnMajor` (B) 布局，充分发挥硬件性能。
**主要功能包括：**
- 执行 MoE 层的核心 BF16 GEMM 计算。
- 自动处理 Token 到 Expert 的映射及计算。
- 支持 Top-K 路由逻辑（通过 `topk` 参数）。
- 可选的路由权重融合乘法。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，数据类型为 `bfloat16`，用于存储计算结果。
- **a** (torch.Tensor, 入参): 激活值输入张量，数据类型为 `bfloat16`，通常为经过重排后的 Token 序列，形状通常为 `[M, K]`。
- **b** (torch.Tensor, 入参): 专家权重张量，数据类型为 `bfloat16`，形状通常为 `[E, N, K]`（假设 LayoutB 为 ColumnMajor，视具体物理布局而定）。
- **moe_weight** (torch.Tensor, 入参): 路由权重张量，表示每个 Token 分配给对应专家的权重，数据类型必须为 `float32`。
- **token_ids** (torch.Tensor, 入参): 参与计算的 Token 索引数组，用于定位有效的输入行，数据类型必须为 `int32`。
- **expert_ids** (torch.Tensor, 入参): 对应的专家索引数组，指示每个 Token 应该由哪个专家处理，数据类型必须为 `int32`。
- **num_tokens_post_padded** (torch.Tensor, 入参): 填充后的 Token 总数张量，数据类型必须为 `int32`。
- **num_valid_tokens** (int64_t, 入参): 当前批次中实际参与计算的有效 Token 数量（M 维度）。
- **topk** (int64_t, 入参): 每个 Token 选择的专家数量（Top-K）。
- **mul_routed_weight** (bool, 入参): 是否将结果乘以路由权重 `moe_weight`。
### 返回值
无返回值，计算结果直接写入输出张量 out 中。如果是底层 C++ 接口调用失败，可能会返回错误码，但在 Python 封装层通常无返回值。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- 输入张量 `a`、`b` 和输出张量 `out` 的数据类型必须为 `bfloat16` (`torch.bfloat16`)。
- 路由权重 `moe_weight` 的数据类型必须为 `float32` (`torch.float32`)。
- 索引张量 `token_ids`、`expert_ids`、`num_tokens_post_padded` 的数据类型必须为 `int32` (`torch.int32`)。
- `num_valid_tokens` 必须大于 0。
- 输入布局要求 A 为 RowMajor，B 为 ColumnMajor。
- `ArchTag` 为 `sm80`，意味着该算子针对特定的计算能力架构进行了优化。
- 输出张量 `out` 在调用前建议进行零初始化（`torch.zeros`），以避免在 TopK > 1 累加场景下引入未初始化内存中的随机值。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 42
DEVICE = "cuda"
DTYPE_BF16 = torch.bfloat16
DTYPE_FLOAT = torch.float32
DTYPE_INT32 = torch.int32
# 维度设置
NUM_TOKENS = 128  # num_valid_tokens (M)
N = 4096          # Output features
K = 2048          # Input features
NUM_EXPERTS = 8   # E
TOPK = 2
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# 激活输入 a (M, K)
a = torch.randn((NUM_TOKENS, K), dtype=DTYPE_BF16)
# 权重输入 b (E, N, K)
# 注意：C++代码中 b.size(1) 映射为 N，b.size(2) 映射为 K (或相反取决于物理布局，此处假设常见形状)
b = torch.randn((NUM_EXPERTS, N, K), dtype=DTYPE_BF16)
# 2. 输出张量 out
# 必须使用 torch.zeros 初始化
out = torch.zeros((NUM_TOKENS, N), dtype=DTYPE_BF16)
# 3. MoE 路由信息 (必须为 float32)
moe_weight = torch.rand((NUM_TOKENS, TOPK), dtype=DTYPE_FLOAT)
# 4. 索引张量 (必须为 int32)
token_ids = torch.arange(NUM_TOKENS, dtype=DTYPE_INT32)
expert_ids = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS,), dtype=DTYPE_INT32)
num_tokens_post_padded = torch.tensor([NUM_TOKENS], dtype=DTYPE_INT32)
# 其他参数
mul_routed_weight = True
# ================= 算子执行 =================
print(f"Running cutlass_moe_bf16_mm...")
print(f"A shape: {a.shape}, B shape: {b.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.cutlass_moe_bf16_mm(
    out,
    a,
    b,
    moe_weight,
    token_ids,
    expert_ids,
    num_tokens_post_padded,
    NUM_TOKENS,
    TOPK,
    mul_routed_weight
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
if torch.isnan(out).any():
    print("Warning: Output contains NaN!")
else:
    print(f"Output Mean: {out.mean().item():.4f}")
```
## 63. cutlass_scaled_mm
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def cutlass_scaled_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: Optional[torch.Tensor]
) -> None:
    pass
```
### 功能描述
cutlass_scaled_mm 算子实现了基于 CUTLASS 库的高效缩放矩阵乘法（Scaled GEMM）。该算子专为 INT8 量化推理设计，能够利用 GPU 的 Tensor Core 进行高速的低精度计算，同时保持高精度的累加和输出。
它计算两个 INT8 张量 a 和 b 的乘积，应用 FP32 类型的量化缩放因子，并可选地加上偏置项，最终将结果输出为 BFloat16 格式。
此实现支持 2D 矩阵乘法以及 3D 批量矩阵乘法（Batch GEMM），并针对 SM80 架构进行了特定优化。
- 计算公式：
    该算子执行以下计算：
    $$
    \text{Out} = \text{Convert}\left ( (A \times B) \odot S_a \odot S_b + \text{Bias} \right)
    $$
    其中：
    - $A \in \mathbb{Z}^{M \times K}$ 是输入张量 (`a`)，数据类型为 `int8`。
    - $B \in \mathbb{Z}^{K \times N}$ 是权重张量 (`b`)，数据类型为 `int8`。
    - $S_a$ 是输入 `a` 的缩放因子 (`a_scales`)，数据类型为 `float32`。
    - $S_b$ 是权重 `b` 的缩放因子 (`b_scales`)，数据类型为 `float32`。
    - $\text{Bias}$ 是可选的偏置张量 (`bias`)，数据类型为 `bfloat16`。
    - $\text{Out}$ 是输出张量 (`out`)，数据类型为 `bfloat16`。
    - $\odot$ 表示广播乘法。
    - 内部累加计算使用 FP32 (`ElementCompute = float`) 以确保精度。
**主要功能包括：**
- 高性能 INT8 GEMM：利用底层 CUTLASS/MCTLAS 库，针对 INT8 输入和 BFloat16 输出进行了高度优化。
- 缩放机制：支持输入和权重的独立缩放（反量化），缩放因子保持在 FP32 精度。
- 偏置融合：支持在 GEMM 计算后直接融合偏置加法（Epilogue Fusion），减少显存访问开销。
- 批量计算支持：能够自动识别输入维度，支持 `[Batch, M, K]` 和 `[Batch, K, N]` 形状的批量矩阵乘法。
- 混合精度：输入 Int8 -> 累加 Float32 -> 输出 BFloat16。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，数据类型为 `bfloat16`，形状为 `[M, N]` 或 `[Batch, M, N]`。
- **a** (torch.Tensor, 入参): 激活值输入张量，数据类型为 `int8`，形状为 `[M, K]` 或 `[Batch, M, K]`。
- **b** (torch.Tensor, 入参): 权重输入张量，数据类型为 `int8`，形状为 `[K, N]` 或 `[Batch, K, N]`。
- **a_scales** (torch.Tensor, 入参): 激活值的缩放因子，数据类型为 `float32`。
- **b_scales** (torch.Tensor, 入参): 权重的缩放因子，数据类型为 `float32`。
- **bias** (Optional[torch.Tensor], 入参): 可选的偏置张量，数据类型为 `bfloat16`。如果提供，其形状应支持广播到输出形状（通常为 `[N]`）。
### 返回值
无返回值，计算结果直接写入输出张量 out 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入张量 `a`和 `b` 的数据类型必须为 `int8`
- 输出张量 `out` 和偏置 `bias`（如果存在）的数据类型必须为 `bfloat16`
- 缩放因子 `a_scales` 和 `b_scales` 的数据类型必须为 `float32`
- `a` 和 `b` 的维度必须满足矩阵乘法规则（K 维度匹配）
- 如果输入是 3D 张量，`a` 和 `b` 的 Batch 维度必须一致
- 内存布局：代码中指定 LayoutA 为 `RowMajor`，LayoutB 为 `ColumnMajor`，调用时需确保数据布局符合预期
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE_INT8 = torch.int8
DTYPE_BF16 = torch.bfloat16
DTYPE_FLOAT = torch.float32
# 维度设置
M = 128
N = 4096
K = 2048
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Tensor
# 输入 A: [M, K]
a = torch.randint(-127, 127, (M, K), dtype=DTYPE_INT8)
# 权重 B: [K, N]
b = torch.randint(-127, 127, (K, N), dtype=DTYPE_INT8)
# 输出 Out: [M, N]
out = torch.empty((M, N), dtype=DTYPE_BF16)
# 2. Scales
# 假设 A 是 Per-Token 缩放, B 是 Per-Channel 缩放
a_scales = torch.randn(M, 1, dtype=DTYPE_FLOAT)
b_scales = torch.randn(1, N, dtype=DTYPE_FLOAT)
# 3. Bias (可选)
bias = torch.randn(N, dtype=DTYPE_BF16)
# ================= 算子执行 =================
print(f"Running cutlass_scaled_mm...")
print(f"A: {a.shape}, B: {b.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用算子
torch.ops._C.cutlass_scaled_mm(
    out,
    a,
    b,
    a_scales,
    b_scales,
    bias
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {out.mean().item():.4f}")
```
## 64. cutlass_scaled_mm_azp
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def cutlass_scaled_mm_azp(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    azp_adj: torch.Tensor,
    azp: Optional[torch.Tensor],
    bias: Optional[torch.Tensor]
) -> None:
    pass
```
### 功能描述
`cutlass_scaled_mm_azp` 算子基于 Cutlass 库实现带非对称零点（Asymmetric Zero Point, AZP）校正的 INT8 矩阵乘法。该算子用于量化推理场景，支持 per-token 和 per-channel 的动态缩放及零点偏移校正。
- 计算逻辑：
    算子计算 $C = \alpha (A \times B) + \beta$，并在此基础上应用 AZP 修正项。
    $$
    \text{Out} = \text{ScaledMM}(A, B, \text{scales}) + \text{AZP\_Correction} + \text{Bias}
    $$
    其中包含针对输入 $A$（Per-Token）和权重 $B$（Per-Channel）的零点调整。
- 主要特性：
    - 输入矩阵为 INT8 类型。
    - 支持 FP32 类型的缩放因子以保证精度。
    - 支持 Int32 类型的零点调整项。
    - 针对 NVIDIA GPU 架构（SM80+）及国产异构卡进行了底层优化。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[M, N]`，通常为 FP16 或 BF16。
- **a** (torch.Tensor, 入参): 输入矩阵 A，形状为 `[M, K]`，INT8 类型，行主序。
- **b** (torch.Tensor, 入参): 输入矩阵 B，形状为 `[K, N]`，INT8 类型，列主序。
- **a_scales** (torch.Tensor, 入参): A 的缩放因子，形状为 `[M]`，必须为 **Float32**。
- **b_scales** (torch.Tensor, 入参): B 的缩放因子，形状为 `[N]`，必须为 **Float32**。
- **azp_adj** (torch.Tensor, 入参): 零点调整项（Per-Channel），形状为 `[N]`，Int32 类型。
- **azp** (Optional[torch.Tensor], 入参): 零点项（Per-Token），形状为 `[M]`，Int32 类型，可选。
- **bias** (Optional[torch.Tensor], 入参): 偏置，形状为 `[N]`，类型与 `out` 一致，可选。
### 返回值
无返回值，计算结果写入 `out` 张量。
### 约束与调用
- 设备限制: 所有 Tensor 必须在 CUDA 设备上。
- 形状限制:
    - $A$: `[M, K]`
    - $B$: `[K, N]`
    - $K$ 维度通常要求 16 字节对齐。
- 类型限制:
    - `a`, `b`: torch.int8
    - `a_scales`, `b_scales`: torch.float32
    - `azp`, `azp_adj`: torch.int32
- 内存布局: `b` 矩阵建议为列主序（Column Major），即转置后的连续内存。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置
DEVICE = "cuda"
M, N, K = 128, 64, 256
torch.set_default_device(DEVICE)
# 1. 数据准备
# A: [M, K], Int8
a = torch.randint(-128, 127, (M, K), dtype=torch.int8)
# B: [K, N], Int8, Column Major
# 先创建 [N, K] 再转置，以满足底层对内存布局的要求
b_raw = torch.randint(-128, 127, (N, K), dtype=torch.int8)
b = b_raw.t()
# 2. Scales (必须是 Float32)
a_scales = torch.rand(M, dtype=torch.float32) / 100.0
b_scales = torch.rand(N, dtype=torch.float32) / 100.0
# 3. AZP 参数 (Int32)
# azp_adj 对应维度 N
azp_adj = torch.randint(0, 10, (N,), dtype=torch.int32)
# azp 对应维度 M
azp = torch.randint(0, 10, (M,), dtype=torch.int32)
# 4. Bias & Output
bias = torch.randn(N, dtype=torch.float16)
out = torch.zeros((M, N), dtype=torch.float16)
# 5. 执行
print(f"Running cutlass_scaled_mm_azp...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.cutlass_scaled_mm_azp(
    out,
    a,
    b,
    a_scales,
    b_scales,
    azp_adj,
    azp,
    bias
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"Output Mean: {out.mean().item():.4f}")
```
## 65. cutlass_scaled_mm_supports_fp8
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);
```
### 功能描述
cutlass_scaled_mm_supports_fp8 是一个用于运行时特性检测的辅助函数。该函数用于判断当前 CUDA 设备的计算能力（Compute Capability）与编译该库时的 CUDA Toolkit 版本组合，是否满足执行 CUTLASS FP8 缩放矩阵乘法（Scaled Matrix Multiplication）内核的最低要求。
它是调用 cutlass_scaled_mm 进行 FP8 推理或训练的前置检查接口，旨在确保后续的矩阵乘法调用能够安全地调度到对应的硬件加速单元上。
- 判定逻辑：
    该函数根据传入的设备计算能力值和编译时的 CUDA_VERSION 宏进行如下判断：
    - Hopper 架构 (SM90, capability >= 90)：要求 CUDA 版本至少为 12.0 (`CUDA_VERSION >= 12000`)。
    - Ada Lovelace 架构 (SM89, capability >= 89)：要求 CUDA 版本至少为 12.4 (`CUDA_VERSION >= 12040`)。
    - 其他架构：对于计算能力低于 89 的设备（如 Ampere SM80 或 Turing SM75），函数直接返回 false，表明不支持此路径的 FP8 缩放矩阵乘法。
        主要功能包括：
- 针对不同硬件架构（Hopper vs Lovelace）执行差异化的 CUDA 版本依赖检查。
- 作为功能门控（Feature Gate），允许上层框架根据返回值动态选择执行 FP8 路径还是回退到 BF16/FP16 路径。
### 参数说明
- **cuda_device_capability** (int64_t, 入参): 当前 CUDA 设备的计算能力数值表示。计算公式为 `major * 10 + minor`。例如：
    - NVIDIA H100 (Hopper) 为 9.0，参数应传入 `90`。
    - NVIDIA L40S (Ada Lovelace) 为 8.9，参数应传入 `89`。
    - NVIDIA A100 (Ampere) 为 8.0，参数应传入 `80`。
### 返回值
- **bool**:
    - `true`: 表示当前硬件和软件环境支持运行 CUTLASS FP8 缩放矩阵乘法内核。
    - `false`: 表示不支持，调用对应的 GEMM 算子可能会抛出异常或未定义行为。
### 约束与调用
- 该函数仅检查标准的 FP8 缩放矩阵乘法支持情况。对于 Block-wise 量化 (`cutlass_scaled_mm_supports_block_fp8`) 或 Grouped GEMM (`cutlass_group_gemm_supported`)，需调用其对应的独立检查函数。
- 函数内部依赖编译时的宏 `CUDA_VERSION`，因此它反映的是编译该算子库时的环境能力，而非运行时的动态驱动版本（尽管通常二者需兼容）。
- 调用者需确保传入的 `cuda_device_capability` 准确对应当前上下文中的 GPU 设备。
### 调用示例
```Python
import torch
import mcoplib._C 
# ================= 环境准备 =================
if not torch.cuda.is_available():
    print("CUDA不可用")
    exit()
# 获取当前设备的属性
device = torch.device("cuda")
device_props = torch.cuda.get_device_properties(device)
# 计算 capability 整数值，例如 9.0 -> 90
capability_int = device_props.major * 10 + device_props.minor
print(f"检测设备: {device_props.name}")
print(f"计算能力: {device_props.major}.{device_props.minor} (Int: {capability_int})")
# ================= 算子检查 =================
# 调用 C++ 接口检查 FP8 支持情况
is_fp8_supported = torch.ops._C.cutlass_scaled_mm_supports_fp8(capability_int)
print(f"是否支持 CUTLASS Scaled MM FP8: {is_fp8_supported}")
# 逻辑分支示例
if is_fp8_supported:
    print(">> 准备执行 FP8 矩阵乘法优化路径...")
    # 此处可安全调用 cutlass_scaled_mm 配合 fp8 类型输入
else:
    print(">> 当前设备或 CUDA 版本不支持 FP8 加速，回退到 BF16/FP16 路径...")
```
## 66. cutlass_scaled_mm_supports_block_fp8
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);
```
### 功能描述
cutlass_scaled_mm_supports_block_fp8 函数用于检查当前硬件和软件环境是否支持基于 CUTLASS 的块级量化（Block-wise quantization）FP8 矩阵乘法运算。
该函数根据传入的 CUDA 设备计算能力（Compute Capability）和编译时的 CUDA 版本（CUDA_VERSION）进行判断。块级 FP8 运算通常比标准 FP8 运算对硬件和软件版本有更严格的要求。
- 判断逻辑：
    - 对于计算能力 $\ge 100$ 的设备（如 Blackwell 架构 SM100），要求 CUDA 版本至少为 12.8（12080）。
    - 对于计算能力 $\ge 90$ 的设备（如 Hopper 架构 SM90），要求 CUDA 版本至少为 12.0（12000）。
    - 对于计算能力 $< 90$ 的设备（包括 SM89 Ada Lovelace），不支持块级 FP8 运算，返回 `false`。
**主要功能包括：**
- 验证硬件架构是否满足 SM90 或更高版本的要求。
- 验证 CUDA Toolkit 版本是否满足对应架构的最低要求。
- 作为启动块级 FP8 Kernel 前的前置检查条件，防止在不支持的环境下调用导致运行时错误。
### 参数说明
- **cuda_device_capability** (int64_t, 入参): CUDA 设备的计算能力数值。通常表示为 `major * 10 + minor`，例如 SM90 应传入 `90`，SM100 应传入 `100`。
### 返回值
- **bool**:
    - `true`: 表示当前环境支持 CUTLASS 块级 FP8 矩阵乘法。
    - `false`: 表示当前环境不支持。
### 约束与调用
- 必须在定义了 `CUDA_VERSION` 宏的环境中编译，否则默认返回 `false`。
- 该函数仅检查库层面的支持情况，不检查显存大小或其他运行时资源。
- 即使返回 `true`，实际调用 Kernel 时仍需确保输入数据的对齐和格式符合 CUTLASS 要求。
- 相比于标准 FP8 支持（`cutlass_scaled_mm_supports_fp8`），此函数对 SM89 架构返回 `false`，因为 Block FP8 仅在 SM90+ 上启用。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
device_cap_tuple = torch.cuda.get_device_capability()
cuda_device_capability = device_cap_tuple[0] * 10 + device_cap_tuple[1]
print(f"Device Capability: {cuda_device_capability}")
# ================= 算子执行 =================
# 调用 C++ 绑定的检查函数
is_supported = torch.ops._C.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)
print(f"Checking support for Block FP8 Scaled MM...")
if is_supported:
    print("Result: Supported (True)")
    print("You can proceed with calling cutlass_scaled_mm with block scaling.")
else:
    print("Result: Not Supported (False)")
    print("Current hardware or CUDA version does not meet requirements (SM90+ and CUDA 12.0+).")
```
## 67. cutlass_group_gemm_supported
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
bool cutlass_group_gemm_supported(int64_t cuda_device_capability);
```
### 功能描述
cutlass_group_gemm_supported 是一个辅助函数，用于在运行时检查当前硬件环境和 CUDA 版本是否支持 CUTLASS Grouped GEMM（分组通用矩阵乘法）内核，特别是针对 FP8 数据类型的分组 GEMM 操作。
该函数根据输入的 CUDA 设备计算能力（Compute Capability）和编译时的 CUDA 版本（CUDA_VERSION）进行判断。
- 判定逻辑：
    - 对于 Compute Capability >= 100 (如 SM100 Blackwell 架构)，需要 CUDA 版本 >= 12.8 (12080)。
    - 对于 Compute Capability >= 90 (如 SM90 Hopper 架构)，需要 CUDA 版本 >= 12.3 (12030)。
    - 对于其他架构（如 SM89 Ada Lovelace 或 SM80 Ampere），目前该函数返回 `false`，表示默认不支持该特定的分组 GEMM 内核配置。
**主要功能包括：**
- 验证当前 GPU 架构是否满足运行特定 CUTLASS Grouped GEMM 内核的最低硬件要求。
- 验证当前 CUDA Toolkit 版本是否满足编译和运行这些内核的最低软件要求。
- 作为算子分发前的守卫逻辑，防止在不支持的软硬件环境下调用相关内核导致错误。
### 参数说明
- **cuda_device_capability** (int64_t, 入参): 设备的计算能力数值。通常由 `major * 10 + minor` 计算得出（例如 SM90 对应 90，SM100 对应 100）。
### 返回值
- **bool**: 返回 `true` 表示当前环境支持 CUTLASS Grouped GEMM 内核；返回 `false` 表示不支持。
### 约束与调用
- 该函数的判断依赖于编译时定义的 `CUDA_VERSION` 宏。
- 仅针对代码中定义的特定版本依赖关系（CUDA 12.3+ on SM90, CUDA 12.8+ on SM100）。
- 输入的 `cuda_device_capability` 必须准确反映当前设备的计算能力（Major version * 10 + Minor version）。
### 调用示例
```Python
import torch
import mcoplib._C
# ================= 环境准备 =================
# 获取当前设备的计算能力
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    # 将计算能力转换为整数格式 (e.g., 9.0 -> 90)
    cuda_capability = device_props.major * 10 + device_props.minor
    print(f"当前设备: {device_props.name}")
    print(f"计算能力 (SM): {cuda_capability}")
    # ================= 算子检查 =================
    # 调用 C++ 绑定的检查函数
    is_supported = torch.ops._C.cutlass_group_gemm_supported(cuda_capability)
    print(f"CUTLASS Group GEMM 支持状态: {'支持' if is_supported else '不支持'}")
    if is_supported:
        print("可以安全调用 Grouped GEMM 相关算子。")
    else:
        print("当前环境不满足 Grouped GEMM 的运行要求 (需 SM90+且CUDA12.3+ 或 SM100+且CUDA12.8+)。")
else:
    print("未检测到 CUDA 设备。")
```
## 71. cutlass_scaled_mm_supports_fp4
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def cutlass_scaled_mm_supports_fp4(
    cuda_device_capability: int
) -> bool:
    pass
```
### 功能描述
cutlass_scaled_mm_supports_fp4 算子用于检测当前运行时环境及硬件设备是否支持 CUTLASS 实现的 FP4 缩放矩阵乘法（Scaled Matrix Multiplication）。该函数是 vLLM 等推理框架中用于调度 FP4 内核的前置检查逻辑。
该算子通过验证 CUDA 设备计算能力（Compute Capability）和 CUDA 运行时版本来确定是否可以安全调用 cutlass_scaled_fp4_mm 系列函数（如 cutlass_scaled_fp4_mm_sm100a 或 cutlass_scaled_fp4_mm_sm120a）。
- 判定逻辑：
    支持 FP4 的条件需同时满足以下两点：
    $$
    \text{Supported} = (\text{cuda\_device\_capability} \ge 100) \land (\text{runtimeVersion} \ge 12080)
    $$
    其中：
    - `cuda_device_capability` 表示设备的计算能力数值（例如 100 代表 SM 10.0 架构，即 Blackwell 架构）。
    - `runtimeVersion` 通过 `cudaRuntimeGetVersion` 获取，`12080` 对应 CUDA 12.8 版本。
**主要功能包括：**
- 检查硬件架构是否满足最低要求（需 Compute Capability 100 及以上）。
- 检查软件环境是否满足最低要求（需 CUDA Runtime 12.8 及以上）。
- 为上层应用提供布尔值标识，用于决定是否启用 FP4 量化路径或回退到其他精度。
### 参数说明
- **cuda_device_capability** (int, 入参): CUDA 设备的计算能力整数值。通常计算方式为 `major * 10 + minor`（例如 SM 9.0 为 90，SM 10.0 为 100）。
### 返回值
- **bool**: 如果环境同时满足硬件架构（>= SM100）和 CUDA 运行时版本（>= 12.8）的要求，返回 `True`；否则返回 `False`。
### 约束与调用
- 输入参数 `cuda_device_capability` 必须为非负整数。
- 该函数依赖于底层的 `cudaRuntimeGetVersion` API，调用时需确保 CUDA 驱动及运行时环境已正确安装。
- 该函数仅进行环境检查，不执行实际的计算任务，也不会分配 GPU 显存。
- 如果返回 `False`，调用方不应尝试调用 `cutlass_scaled_fp4_mm`，否则会触发 `TORCH_CHECK_NOT_IMPLEMENTED` 异常。
### 调用示例
```Python
import torch
import mcoplib._C
# ================= 环境准备 =================
# 假设当前设备支持 CUDA，获取设备属性
# 在实际场景中，通常通过 torch.cuda.get_device_capability() 获取
# 这里模拟传入参数
device_capability_sm100 = 100  # 模拟 Blackwell 架构
device_capability_sm90 = 90    # 模拟 Hopper 架构
# ================= 算子执行 =================
print("Checking FP4 support for SM 10.0 (Blackwell)...")
is_supported_sm100 = torch.ops._C.cutlass_scaled_mm_supports_fp4(device_capability_sm100)
print(f"Result for SM 10.0: {is_supported_sm100}")
print("Checking FP4 support for SM 9.0 (Hopper)...")
is_supported_sm90 = torch.ops._C.cutlass_scaled_mm_supports_fp4(device_capability_sm90)
print(f"Result for SM 9.0: {is_supported_sm90}")
# ================= 逻辑判断示例 =================
if is_supported_sm100:
    print("FP4 acceleration is enabled.")
else:
    print("FP4 acceleration is not supported on this device/runtime combination.")
```
## 72. gptq_gemm
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_exllama: bool,
    bit: int,
    group_size: int,
    perm_space: torch.Tensor,
    temp_space: torch.Tensor,
    is_bf16: bool
) -> torch.Tensor:
    pass
```
### 功能描述
`gptq_gemm` 算子实现 GPTQ 量化矩阵乘法。该算子利用压缩存储的低比特量化权重与高精度浮点激活值直接进行计算，集成了基于 ExLlama/Marlin 的高性能内核。为了优化显存分配开销，该版本要求调用方预先分配临时显存空间（scratch space）。
- 计算公式：
    $$
    C = A \times \text{Dequant}(B_{q}, \text{scales}, \text{zeros}, g_{idx})
    $$
    其中：
    - $A$ 是输入激活张量 (`a`)
    - $B_{q}$ 是压缩存储的量化权重 (`b_q_weight`)
    - $\text{Dequant}$ 是基于 `scales`、`zeros`、`bit` 和 `group_size` 的反量化过程
    - `perm_space` 和 `temp_space` 用于存储计算过程中的中间状态（如重排索引和部分和），避免内核内部动态分配
**主要功能包括：**
- 支持 2-bit, 3-bit, 4-bit, 8-bit 量化权重计算。
- 支持 GPTQ 组量化（Groupwise Quantization）及 Act-Order 特性。
- 通过预分配显存接口（`perm_space`, `temp_space`）减少运行时开销。
- 自动处理 FP16/BF16 输入数据类型（需通过 `is_bf16` 标记）。
### 参数说明
- **a** (torch.Tensor, 入参): 输入激活张量，形状为 `[m, k]`，数据类型为 float16 或 bfloat16
- **b_q_weight** (torch.Tensor, 入参): 量化权重张量，形状为 `[k_packed, n]`，数据类型为 int32
- **b_gptq_qzeros** (torch.Tensor, 入参): 量化零点张量，用于反量化
- **b_gptq_scales** (torch.Tensor, 入参): 量化缩放因子张量，数据类型为 float16/bfloat16
- **b_g_idx** (torch.Tensor, 入参): 分组索引张量，形状为 `[k]`，数据类型为 int32
- **use_exllama** (bool, 入参): 是否启用 ExLlama 优化内核
- **bit** (int, 入参): 量化位数，支持 2, 3, 4, 8
- **group_size** (int, 入参): 量化分组大小（如 128）
- **perm_space** (torch.Tensor, 入参): 预分配的重排索引缓冲区，形状至少为 `[n]`，数据类型为 int32
- **temp_space** (torch.Tensor, 入参): 预分配的临时计算缓冲区，形状建议为 `[m, n]`，数据类型为 float16/bfloat16
- **is_bf16** (bool, 入参): 输入张量 `a` 是否为 bfloat16 类型
### 返回值
返回计算结果张量，形状为 `[m, n]`，数据类型与输入 `a` 相同
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `perm_space` 和 `temp_space` 必须预先分配且大小足够，否则会导致非法内存访问（ATU Fault）
- `b_q_weight` 的行数必须符合打包规则 `k * bit / 32`
- 输入 `a` 的最后一维必须与权重逻辑维度 `k` 一致
- 内存布局要求：输入张量必须是连续存储的
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
BITS = 4
GROUPSIZE = 128
M = 128
K = 1024
N = 4096
# 初始化
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 维度计算
PACK_FACTOR = 32 // BITS
K_PACKED = K // PACK_FACTOR
NUM_GROUPS = (K + GROUPSIZE - 1) // GROUPSIZE
# 数据准备
a = torch.randn(M, K, dtype=torch.float16)
b_q_weight = torch.randint(-2000000000, 2000000000, (K_PACKED, N), dtype=torch.int32)
b_gptq_scales = torch.ones(NUM_GROUPS, N, dtype=torch.float16)
b_gptq_qzeros = torch.randint(0, 2000000000, (NUM_GROUPS, N // PACK_FACTOR), dtype=torch.int32)
b_g_idx = torch.arange(K, dtype=torch.int32) // GROUPSIZE
# 构造参数
use_exllama = False
is_bf16 = (a.dtype == torch.bfloat16)
perm_space = torch.empty((N,), dtype=torch.int32, device=DEVICE)
temp_space = torch.empty((M, N), dtype=torch.float16, device=DEVICE)
# 算子执行
if torch.cuda.is_available():
    torch.cuda.synchronize()
output = torch.ops._C.gptq_gemm(
    a,
    b_q_weight,
    b_gptq_qzeros,
    b_gptq_scales,
    b_g_idx,
    use_exllama,
    BITS,
    GROUPSIZE,
    perm_space,
    temp_space,
    is_bf16
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"执行成功。Output Shape: {output.shape}")
```
## 73. gptq_shuffle
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def gptq_shuffle(
    q_weight: torch.Tensor,
    q_perm: torch.Tensor,
    bit: int
) -> None:
    pass
```
### 功能描述
gptq_shuffle 算子用于对 GPTQ 量化后的权重张量进行重排（Shuffle），以适配 ExLlama 等高性能推理内核的内存布局要求。在 GPTQ 量化算法中，为了支持激活顺序（act-order）优化，输入通道往往会被重新排序。该算子根据排列索引 q_perm 对打包后的权重 q_weight 进行原地或转换处理，使其符合底层计算内核的访问模式。
此实现调用了 sglang::gptq:: shuffle_exllama_weight，主要用于解决量化权重在不同计算流或硬件上的对齐问题，确保矩阵乘法计算的正确性和效率。
- 处理逻辑：
    算子接收打包的权重矩阵和排列索引，根据指定的量化位数（bit）重新组织数据。
    逻辑上的权重维度计算如下：
    $$
    \text{Height} = \text{q\_weight.size}(0) \times \frac{32}{\text{bit}}
    $$
    $$
    \text{Width} = \text{q\_weight.size}(1)
    $$
    算子利用 q_perm 映射行索引，调整内部数据块的顺序。如果 q_perm 为空或未定义，算子将按照默认顺序处理或跳过特定重排步骤。
**主要功能包括：**
- 对 FP16/BF16 模型经 GPTQ 量化后的 INT32 打包权重进行预处理。
- 支持指定量化位数（如 4-bit），兼容 ExLlama 内核格式。
- 处理带有 `act-order`（Activation Order）特性的模型权重重排。
- 自动处理 `q_perm` 为空的情况（通过内部指针检查）。
### 参数说明
- **q_weight** (torch.Tensor, 入参/出参): 量化后的权重张量，通常为 `int32` 类型。形状为 `[in_features // (32/bit), out_features]`。该张量会被原地修改或用于生成重排后的内部状态
- **q_perm** (torch.Tensor, 入参): 排列索引张量，通常为 `int16` 或 `int32` 类型。形状为 `[in_features]`，表示输入通道的重排顺序。如果不需要重排，可传入空张量
- **bit** (int, 入参): 量化位数，通常为 4 (对应 4-bit GPTQ)
### 返回值
无返回值，操作直接作用于输入张量或底层状态
### 约束与调用
- 输入张量 `q_weight` 和 `q_perm` 必须位于 CUDA 设备上
- `q_weight` 的数据类型通常应为 `torch.int32`
- `bit` 参数必须与权重打包时的配置一致（如 4）
- `q_perm` 的长度必须匹配模型逻辑上的输入特征维度（in_features）
- 内存布局要求：输入张量建议是连续存储的
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
BITS = 4
# 维度设置
IN_FEATURES = 1024
OUT_FEATURES = 4096
# 计算 packed 行数 (每个 int32 存储 32/bits 个权重)
PACK_FACTOR = 32 // BITS
PACKED_ROWS = IN_FEATURES // PACK_FACTOR
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Packed Weight (int32)
# 模拟已量化的权重矩阵
q_weight = torch.randint(
    low=-2147483648, 
    high=2147483647, 
    size=(PACKED_ROWS, OUT_FEATURES), 
    dtype=torch.int32
)
# 2. 创建 Permutation Index
# 模拟 act-order 生成的随机排列
q_perm = torch.randperm(IN_FEATURES, dtype=torch.int32) # ExLlama 通常使用 short
# ================= 算子执行 =================
print(f"Running gptq_shuffle...")
print(f"Q_Weight: {q_weight.shape}, Perm: {q_perm.shape}, Bits: {BITS}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 注意：此操作通常是 In-place 修改或特定格式转换
torch.ops._C.gptq_shuffle(
    q_weight,
    q_perm,
    BITS
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
```
## 74. static_scaled_fp8_quant
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def static_scaled_fp8_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor
) -> None:
    pass
```
### 功能描述
static_scaled_fp8_quant 算子实现基于静态缩放因子的 FP8 量化功能。该算子将高精度浮点输入张量（如 FP16、BF16 或 FP32）转换为 FP8 格式，并将结果存储在输出张量中。
此实现采用静态量化策略，即量化过程中使用的缩放因子是由用户提供的固定标量（scale），而不是根据输入数据的动态范围实时计算得出。这种方式通常用于推理阶段，其中缩放因子已在校准阶段确定，能够避免动态统计带来的计算开销。
- 计算公式：
    $$
    \text{output} = \text{cast}_{\text{fp8}}(\text{input} \cdot \text{scale})
    $$
    其中：
    - $\text{input}$ 是输入的高精度张量
    - $\text{scale}$ 是用户提供的静态标量缩放因子
    - $\text{cast}_{\text{fp8}}$ 表示将结果截断并转换为目标 FP8 数据类型（通常为 `e4m3fn` 或 `e5m2`，取决于硬件支持和配置）
**主要功能包括：**
- 利用预设的静态缩放因子执行高效的 FP8 量化。
- 支持将结果直接写入预分配的 `output` 张量，避免显存重新分配。
- 适用于输入张量维度为 2D 的场景（代码中包含 `assert input.ndim == 2` 约束）。
- 作为 `scaled_fp8_quant` 接口中 `scale` 参数不为空时的底层实现路径。
### 参数说明
- **output** (torch.Tensor, 出参): 输出张量，存储量化后的 FP8 数据，形状需与 `input` 一致，数据类型需为 FP8 类型（如 `torch.float8_e4m3fn`）
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[rows, cols]`，数据类型通常为 float16, bfloat16 或 float32
- **scale** (torch.Tensor, 入参): 静态缩放因子张量，必须包含单个元素（Scalar），数据类型通常为 float32
### 返回值
无返回值，计算结果直接写入输出张量 output 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入 `input` 必须是 2 维张量 (`ndim == 2`)
- `scale` 张量的元素个数必须为 1 (`scale.numel () == 1`)
- `output` 张量的数据类型必须符合当前平台的 FP8 标准（例如 NVIDIA GPU 上通常为 `torch.float8_e4m3fn`，ROCm MI300 上可能为 `torch.float8_e4m3fnuz`）
- 内存布局要求：建议输入张量是连续存储的
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
INPUT_DTYPE = torch.float16
OUTPUT_DTYPE = torch.float8_e4m3fn
# 维度设置
ROWS = 128
COLS = 128
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建输入 Tensor
input_tensor = torch.randn(ROWS, COLS, dtype=INPUT_DTYPE)
# 2. 创建静态 Scale (标量 Tensor)
# 假设 scale 已经过校准，例如 1.0
scale = torch.tensor([1.0], dtype=torch.float32)
# 3. 创建输出 Tensor (FP8)
output_tensor = torch.empty_like(input_tensor, dtype=OUTPUT_DTYPE)
# ================= 算子执行 =================
print(f"Running static_scaled_fp8_quant...")
print(f"Input: {input_tensor.shape}, Scale: {scale.item()}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.static_scaled_fp8_quant(
    output_tensor,
    input_tensor,
    scale
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output type: {output_tensor.dtype}")
```
## 75. dynamic_scaled_fp8_quant
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def dynamic_scaled_fp8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor
) -> None:
    pass
```
### 功能描述
dynamic_scaled_fp8_quant 算子实现基于动态缩放因子的张量级（Per-Tensor）FP8 量化。与静态量化不同，该算子在执行时实时计算输入张量的最大绝对值，据此确定缩放因子，并将输入数据量化为 FP8 格式。
此实现包含两个主要步骤：首先通过归约操作（Reduction）计算整个输入张量的最大值从而得到全局缩放因子，并将该因子写入 scale 张量；然后利用该缩放因子将高精度浮点数据转换为 FP8 数据并存储在 out 张量中。
- 计算公式：
    1. 计算全局缩放因子：
        $$
        \text{scale} = \frac{\max (|input|)}{\text{FP8\_MAX}}
        $$
    2. 执行量化：
        $$
        out = \text{cast}_{\text{fp8}}\left (\frac{input}{\text{scale}}\right)
        $$
        其中：
    - $input$ 是输入的高精度张量
    - $\text{FP8\_MAX}$ 是目标 FP8 类型（通常为 E4M3 或 E5M2）可表示的最大浮点数值
    - $scale$ 是计算得到的标量缩放因子（作为出参回填）
    - $out$ 是量化后的 FP8 结果
**主要功能包括：**
- 动态计算输入张量的全局最大绝对值。
- 计算并输出用于反量化的缩放因子 `scale`。
- 将输入数据量化为 FP8 格式。
- 内部处理 CUDA 内存初始化（自动将 scale 重置为 0 开始归约）。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，存储量化后的 FP8 数据，形状需与 `input` 一致，数据类型通常为 `torch.float8_e4m3fn`
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., hidden_size]`，数据类型通常为 float16, bfloat16 或 float32
- **scale** (torch.Tensor, 出参): 缩放因子张量，用于存储计算出的全局 Scale 值，形状为 `[1]`（标量），数据类型必须为 float32
### 返回值
无返回值，计算结果直接写入 `out` 和 `scale` 张量中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入 `input` 和输出 `out` 的最后一维（stride (-1)）必须是连续的（contiguous）
- `scale` 张量必须预分配，且大小足以容纳一个 float32 标量
- 算子内部会异步调用 `cudaMemset` 初始化 `scale`，调用方无需手动清零
- 适用于张量级（Per-Tensor）动态量化场景，若需 Token 级动态量化请使用 `dynamic_per_token_scaled_fp8_quant`
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
INPUT_DTYPE = torch.float16
OUTPUT_DTYPE = torch.float8_e4m3fn
# 维度设置
BATCH = 16
SEQ_LEN = 128
HIDDEN_SIZE = 1024
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建输入 Tensor
input_tensor = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE, dtype=INPUT_DTYPE)
# 2. 创建输出 Tensor (FP8)
output_tensor = torch.empty_like(input_tensor, dtype=OUTPUT_DTYPE)
# 3. 创建 Scale Tensor (Scalar, FP32)
# 动态量化会计算并写入此 Tensor
scale = torch.zeros(1, dtype=torch.float32)
# ================= 算子执行 =================
print(f"Running dynamic_scaled_fp8_quant...")
print(f"Input: {input_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.dynamic_scaled_fp8_quant(
    output_tensor,
    input_tensor,
    scale
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Calculated Scale: {scale.item()}")
print(f"Output type: {output_tensor.dtype}")
```
## 76. dynamic_per_token_scaled_fp8_quant
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def dynamic_per_token_scaled_fp8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    scale_ub: Optional[torch.Tensor]
) -> None:
    pass
```
### 功能描述
`dynamic_per_token_scaled_fp8_quant` 算子实现基于 Token 粒度（Per-Token）的动态 FP8 量化。与全局静态或动态量化不同，该算子为输入张量的每一行（即每个 Token）独立计算一个缩放因子，从而能够更精确地保留不同 Token 之间的数值动态范围差异。
此实现主要用于激活值（Activation）的在线量化。对于输入矩阵的每一行，算子首先计算该行的最大绝对值，结合可选的上限（Upper Bound）计算出缩放因子，然后利用该因子将该行数据量化为 FP8 格式。
- 计算公式：
    对于输入张量 $X$ 的第 $i$ 行（Token）：
    1. 计算行最大绝对值：
        $$
        m_i = \max_{j}(|X_{i, j}|)
        $$
    2. 应用上限截断（如果提供了 scale_ub）：
        $$
        s'_i = \begin{cases} \min (m_i, \text{scale\_ub}) & \text{if scale\_ub is provided} \\ m_i & \text{otherwise} \end{cases}
        $$
    3. 计算缩放因子（确保数值稳定性）：
        $$
        \text{scale}_i = \max\left (\frac{s'_i}{\text{FP8\_MAX}}, \epsilon\right)
        $$
    4. 执行量化：
        $$
        \text{out}_{i, j} = \text{cast}_{\text{fp8}}\left (\frac{X_{i, j}}{\text{scale}_i}\right)
        $$
        其中：
    - $\text{FP8\_MAX}$ 是目标 FP8 类型（通常为 E4M3）的最大表示值
    - $\epsilon$ 是最小缩放因子，防止除零
**主要功能包括：**
- 逐行归约：利用 BlockReduce 高效并行计算每一行的最大值。
- 动态缩放：为每个 Token 生成独立的 Scale，并写入 `scales` 张量。
- FP8 转换：利用计算出的 Scale 将高精度数据转换为 FP8。
- 边界处理：支持可选的 `scale_ub` 参数以限制异常值的影响。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，存储量化后的 FP8 数据，形状需与 `input` 一致，数据类型通常为 `torch.float8_e4m3fn`
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[num_tokens, hidden_size]`，数据类型通常为 float16, bfloat16 或 float32
- **scales** (torch.Tensor, 出参): 缩放因子张量，形状为 `[num_tokens, 1]`，数据类型通常为 float32，用于存储每行的量化 Scale
- **scale_ub** (Optional[torch.Tensor], 入参): 可选的标量张量，指定缩放因子的上限值。如果为 `None`，则不进行上限截断
### 返回值
无返回值，计算结果直接写入 `out` 和 `scales` 张量中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入 `input` 和输出 `out` 的最后一维（stride (-1)）必须是连续的（contiguous）
- `scales` 张量的第一维必须等于 `input` 的 Token 数量（行数）
- `input` 必须至少包含 2 个维度（即视为 flattened tokens 或 batch 形式）
- 内存布局要求：建议输入张量是行优先存储的
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
INPUT_DTYPE = torch.float16
OUTPUT_DTYPE = torch.float8_e4m3fn
# 维度设置
NUM_TOKENS = 128
HIDDEN_SIZE = 4096
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建输入 Tensor
input_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=INPUT_DTYPE)
# 2. 创建输出 Tensor (FP8)
output_tensor = torch.empty_like(input_tensor, dtype=OUTPUT_DTYPE)
# 3. 创建 Scales Tensor (Per-Token: [NUM_TOKENS, 1])
scales = torch.empty((NUM_TOKENS, 1), dtype=torch.float32)
# 4. 可选 Scale Upper Bound
scale_ub = None 
# scale_ub = torch.tensor([10.0], dtype=torch.float32) # 如需限制
# ================= 算子执行 =================
print(f"Running dynamic_per_token_scaled_fp8_quant...")
print(f"Input: {input_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.dynamic_per_token_scaled_fp8_quant(
    output_tensor,
    input_tensor,
    scales,
    scale_ub
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output type: {output_tensor.dtype}")
print(f"Scales shape: {scales.shape}")
```
## 77. static_scaled_int8_quant
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def static_scaled_int8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    azp: Optional[torch.Tensor]
) -> None:
    pass
```
### 功能描述
static_scaled_int8_quant 算子实现静态缩放的 INT8 量化。该算子使用用户提供的静态缩放因子（scale）和可选的零点（azp, Asymmetric Zero Point），将高精度浮点输入张量（如 FP16/BF16/FP32）转换为 INT8 格式。
此实现适用于推理场景，其中量化参数已通过校准提前确定。算子将输入数据除以缩放因子，并应用零点偏移和舍入操作，最终截断到 INT8 范围。
- 计算公式：
    $$
    \text{out} = \text{clamp}\left (\text{round}\left (\frac{\text{input}}{\text{scale}}\right) + \text{azp}, -128, 127\right)
    $$
    其中：
    - $\text{input}$ 是输入张量
    - $\text{scale}$ 是静态缩放因子（标量）
    - $\text{azp}$ 是非对称零点（可选标量）。如果未提供，默认值为 0（对称量化）
    - $\text{clamp}$ 将结果限制在 $[-128, 127]$ 范围内（对于有符号 INT8）
**主要功能包括：**
- 利用静态参数执行高效的 INT8 量化。
- 支持对称量化（仅 `scale`）和非对称量化（`scale` + `azp`）。
- 自动处理舍入（Round-to-Nearest）和饱和截断（Saturation）。
- 针对连续内存布局进行了优化。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，存储量化后的 INT8 数据，形状需与 `input` 一致，数据类型需为 `torch.int8`
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., hidden_size]`，数据类型通常为 float16, bfloat16 或 float32
- **scale** (torch.Tensor, 入参): 静态缩放因子张量，必须包含单个元素（Scalar），数据类型通常为 float32
- **azp** (Optional[torch.Tensor], 入参): 可选的零点张量，必须包含单个元素（Scalar），数据类型通常为 int32。如果为 `None`，则执行对称量化
### 返回值
无返回值，计算结果直接写入输出张量 `out` 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入 `input` 和输出 `out` 必须是连续存储的（contiguous）
- `scale` 和 `azp`（如果提供）必须是标量（numel=1）
- 建议输入张量最后一维为 `hidden_size`，以便利用向量化加载优化
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
INPUT_DTYPE = torch.float16
OUTPUT_DTYPE = torch.int8
# 维度设置
BATCH = 16
SEQ_LEN = 128
HIDDEN_SIZE = 1024
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建输入 Tensor
input_tensor = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE, dtype=INPUT_DTYPE)
# 2. 创建输出 Tensor (INT8)
output_tensor = torch.empty_like(input_tensor, dtype=OUTPUT_DTYPE)
# 3. 创建静态 Scale (标量)
scale = torch.tensor([0.005], dtype=torch.float32)
# 4. 可选: 创建静态 Zero Point (标量)
# 如果不传 azp，则进行对称量化
azp = torch.tensor([10], dtype=torch.int32) 
# ================= 算子执行 =================
print(f"Running static_scaled_int8_quant...")
print(f"Input: {input_tensor.shape}, Scale: {scale.item()}, AZP: {azp.item()}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.static_scaled_int8_quant(
    output_tensor,
    input_tensor,
    scale,
    azp
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output type: {output_tensor.dtype}")
print(f"Output range: [{output_tensor.min()}, {output_tensor.max()}]")
```
## 78. dynamic_scaled_int8_quant
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def dynamic_scaled_int8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    azp: Optional[torch.Tensor]
) -> None:
    pass
```
### 功能描述
dynamic_scaled_int8_quant 算子实现动态缩放的 INT8 量化。与静态量化不同，该算子在运行时实时计算输入张量的缩放因子（和零点），以适应不同 batch 或 token 的数值范围。
此实现主要用于在线量化（On-the-fly Quantization），通常用于 Activation 的量化。算子首先计算输入数据的最大绝对值，据此推导出缩放因子，然后将数据量化为 INT8。支持对称量化（仅 scales）和非对称量化（scales + azp）。
- 计算公式：
    1. 计算缩放因子（对称）：
        $$
        \text{scale} = \frac{\max (|input|)}{127.0}
        $$
    2. 计算零点（非对称，如果启用）：
        $$
        \text{min\_val} = \min (input), \quad \text{max\_val} = \max (input)
        $$
        $$
        \text{scale} = \frac{\text{max\_val} - \text{min\_val}}{255.0}
        $$
        $$
        \text{azp} = \text{round}\left (-128 - \frac{\text{min\_val}}{\text{scale}}\right)
        $$
    3. 执行量化：
        $$
        \text{out} = \text{clamp}\left (\text{round}\left (\frac{\text{input}}{\text{scale}}\right) + \text{azp}, -128, 127\right)
        $$
**主要功能包括：**
- 动态计算 Scale/AZP：利用高效的并行归约（Reduction）计算输入数据的统计信息。
- 并行量化：基于计算出的参数将数据转换为 INT8。
- 支持 Per-Tensor 或 Per-Token：取决于输入维度和归约逻辑（此实现通常支持按行/Token 动态量化）。
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，存储量化后的 INT8 数据，形状需与 `input` 一致，数据类型需为 `torch.int8`
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., hidden_size]`，数据类型通常为 float16, bfloat16 或 float32
- **scales** (torch.Tensor, 出参): 缩放因子张量，用于存储计算出的 Scale 值。形状取决于量化粒度（如 `[num_tokens, 1]`）
- **azp** (Optional[torch.Tensor], 出参): 可选的零点张量，用于存储计算出的 Zero Point。如果为 `None`，则执行对称量化
### 返回值
无返回值，计算结果直接写入 `out`, `scales` 和 `azp`（如果提供）张量中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 输入 `input`、输出 `out`、`scales` 和 `azp` 必须是连续存储的（contiguous）
- `scales` 和 `azp` 的形状必须匹配归约后的维度（通常为 Token 数量）
- 建议输入张量最后一维为 `hidden_size` 以优化性能
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
INPUT_DTYPE = torch.float16
OUTPUT_DTYPE = torch.int8
# 维度设置
NUM_TOKENS = 128
HIDDEN_SIZE = 1024
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建输入 Tensor
input_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=INPUT_DTYPE)
# 2. 创建输出 Tensor (INT8)
output_tensor = torch.empty_like(input_tensor, dtype=OUTPUT_DTYPE)
# 3. 创建 Scales Tensor (Per-Token 动态量化)
scales = torch.empty((NUM_TOKENS, 1), dtype=torch.float32)
# 4. 可选: 创建 AZP Tensor
# azp = torch.empty((NUM_TOKENS, 1), dtype=torch.int32)
azp = None # 对称量化
# ================= 算子执行 =================
print(f"Running dynamic_scaled_int8_quant...")
print(f"Input: {input_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.dynamic_scaled_int8_quant(
    output_tensor,
    input_tensor,
    scales,
    azp
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output type: {output_tensor.dtype}")
print(f"Scales shape: {scales.shape}")
```
## 79. selective_scan_fwd
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_: Optional[torch.Tensor],
    z_: Optional[torch.Tensor],
    delta_bias_: Optional[torch.Tensor],
    delta_softplus: bool,
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    ssm_states: torch.Tensor,
    pad_slot_id: int
) -> None:
    pass
```
### 功能描述
selective_scan_fwd 算子实现选择性状态空间模型（Selective State Space Model, SSM）的前向扫描计算。该算子是 Mamba 架构的核心组件，利用 CUDA 内核通过并行关联扫描（Parallel Associative Scan）算法高效地计算序列的状态更新和输出。
该实现支持变长序列（Variable Length）处理以及基于索引的 KV 缓存机制（用于自回归推理加速）。算子通过离散化连续系统参数并执行递归，计算最终的输出张量。
- 计算公式：
    对于序列中的每个时间步 $t$：
    1. 参数离散化：
        $$
        \Delta_t = \text{softplus}(\text{delta}_t + \text{delta\_bias}) \quad (\text{若 delta\_softplus=True})
        $$
        $$
        \bar{A}_t = \exp (\Delta_t A)
        $$
        $$
        \bar{B}_t = \Delta_t B_t
        $$
    2. 状态更新（扫描）：
        $$
        h_t = \bar{A}_t h_{t-1} + \bar{B}_t u_t
        $$
    3. 输出计算：
        $$
        y_t = C_t h_t + D u_t
        $$
    4. 门控输出（若提供 z）：
        $$
        \text{out}_t = y_t \cdot \text{SiLU}(z_t) = y_t \cdot (z_t \cdot \sigma (z_t))
        $$
**主要功能包括：**
- 高效并行扫描：利用 GPU 寄存器和共享内存优化前缀扫描操作。
- 内存优化（In-place）：为了减少显存占用，计算结果会复用输入张量的存储空间。非门控输出 $y_t$ 写入 `delta` 张量，门控输出 $\text{out}_t$ 写入 `z` 张量（若存在）。
- 变长序列支持：通过 `query_start_loc` 支持在一个批次中处理不同长度的序列（Packed Sequence）。
- 推理缓存支持：结合 `ssm_states` 和 `cache_indices`，支持自回归生成过程中的状态缓存管理，避免重复计算。
### 参数说明
- **u** (torch.Tensor, 入参): 输入信号张量。形状为 `[batch, dim, seqlen]`（定长）或 `[dim, total_seqlen]`（变长）。
- **delta** (torch.Tensor, 入参/出参): 时间步长参数张量。形状需与 `u` 匹配。**注意：** 计算过程中的非门控输出 $y_t$ 将原地写入此张量。
- **A** (torch.Tensor, 入参): 状态转移矩阵参数。形状为 `[dim, dstate]`。
- **B** (torch.Tensor, 入参): 输入投影参数。形状为 `[batch, n_groups, dstate, seqlen]`（定长）或 `[n_groups, dstate, total_seqlen]`（变长）。
- **C** (torch.Tensor, 入参): 输出投影参数。形状同 `B`。
- **D_** (Optional[torch.Tensor], 入参): 输入跳跃连接参数。形状为 `[dim]`。
- **z_** (Optional[torch.Tensor], 入参/出参): 门控分支张量。形状需与 `u` 匹配。**注意：** 若提供此参数，最终的门控输出 $\text{out}_t$ 将原地写入此张量。
- **delta_bias_** (Optional[torch.Tensor], 入参): `delta` 的偏置项。形状为 `[dim]`。
- **delta_softplus** (bool, 入参): 是否对 `delta` 应用 Softplus 激活。
- **query_start_loc** (Optional[torch.Tensor], 入参): 变长序列模式下，存储每个序列起始位置的索引。形状为 `[batch + 1]`。
- **cache_indices** (Optional[torch.Tensor], 入参): 缓存索引，用于指示当前计算在缓存中的位置。
- **has_initial_state** (Optional[torch.Tensor], 入参): 指示每个样本是否有初始状态。
- **ssm_states** (torch.Tensor, 入参/出参): SSM 递归状态张量。形状为 `[batch, dim, dstate]`。计算过程中会原地更新此张量以存储最新的状态 $h_t$。
- **pad_slot_id** (int, 入参): 填充槽位 ID，若缓存索引为此值则跳过计算。
### 返回值
无返回值。计算结果原地更新到 `delta`（存储 $y_t$）和 `z`（存储 $\text{out}_t$，若存在）中，状态更新到 `ssm_states` 中。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- `dstate` (状态维度) 必须小于等于 256。
- `u` 和 `delta` 的最后一维 Stride 必须为 1。
- 支持的数据类型：`u`, `delta`, `z`, `B`, `C` 支持 `float`, `half`, `bfloat16`；`A` 必须为 `float`。
- 若 `query_start_loc` 存在，则启用变长模式，输入张量的维度解释为扁平化的序列。
- `is_variable_B` 和 `is_variable_C` 必须为 True（即 B/C 包含时间维度）。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
BATCH = 2
DIM = 64
SEQLEN = 128
DSTATE = 16
N_GROUPS = 1
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 初始化
torch.set_default_device(DEVICE)
torch.manual_seed(0)
# ================= 数据准备 =================
# 1. 创建 Tensor
u = torch.randn(BATCH, DIM, SEQLEN, dtype=DTYPE)
delta = torch.randn(BATCH, DIM, SEQLEN, dtype=DTYPE) # 结果将写入此处 (如果 z 为 None)
A = -torch.rand(DIM, DSTATE, dtype=torch.float32)
B = torch.randn(BATCH, N_GROUPS, DSTATE, SEQLEN, dtype=DTYPE)
C = torch.randn(BATCH, N_GROUPS, DSTATE, SEQLEN, dtype=DTYPE)
ssm_states = torch.zeros(BATCH, DIM, DSTATE, dtype=DTYPE)
# 可选参数
D = torch.randn(DIM, dtype=torch.float32)
z = torch.randn(BATCH, DIM, SEQLEN, dtype=DTYPE) # 最终结果将写入此处
delta_bias = torch.randn(DIM, dtype=torch.float32)
# 其他配置
delta_softplus = True
pad_slot_id = -1
# 空的可选参数
query_start_loc = None
cache_indices = None
has_initial_state = None
# ================= 算子执行 =================
print("Running selective_scan_fwd...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.
selective_scan_fwd(
    u,
    delta,
    A,
    B,
    C,
    D,
    z,
    delta_bias,
    delta_softplus,
    query_start_loc,
    cache_indices,
    has_initial_state,
    ssm_states,
    pad_slot_id
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean (in z): {z.mean().item():.4f}")
print(f"Final State Mean: {ssm_states.mean().item():.4f}")
```
## 80. swap_blocks
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_mapping: torch.Tensor
) -> None:
    pass
```
### 功能描述
swap_blocks 算子用于在 KV Cache 中高效地复制或交换内存块。该算子通常用于大语言模型推理中的显存管理（PagedAttention 机制），支持将数据从源物理块（Source Blocks）复制到目标物理块（Destination Blocks）。
此功能主要应用于以下场景：
1. Swap-in/Swap-out：在 CPU 内存和 GPU 显存之间移动数据块（例如处理抢占或卸载）。
2. Copy-on-write：在 GPU 显存内部复制数据块（例如在 Beam Search 中分叉序列）。
- 操作逻辑：
    算子接收源张量 src、目标张量 dst 以及一个映射表 block_mapping。
    对于 block_mapping 中的每一对索引 $(s_i, d_i)$，算子将 src 中索引为 $s_i$ 的块的全部内容复制到 dst 中索引为 $d_i$ 的块位置。
    $$
    \forall i, \quad \text{dst}[d_i] \leftarrow \text{src}[s_i]
    $$
    其中 block_mapping 的每一行包含 [src_block_index, dst_block_index]。
**主要功能包括：**
- 基于映射表批量复制 KV Cache 数据块。
- 支持 GPU 内部拷贝以及 CPU 与 GPU 之间的数据交换（取决于输入 Tensor 的设备属性）。
- 保持数据布局（Layout）和数值精度不变。
### 参数说明
- **src** (torch.Tensor, 入参): 源 KV Cache 张量，形状通常为 `[num_blocks, num_kv_heads, head_size/x, block_size, x]` 或类似的 5D 结构（取决于具体的 Cache Layout），数据类型通常为 float16 或 bfloat16
- **dst** (torch.Tensor, 出参): 目标 KV Cache 张量，形状和数据类型通常与 `src` 相同（`num_blocks` 维度可以不同，但必须足够容纳目标块）
- **block_mapping** (torch.Tensor, 入参): 块索引映射张量，形状为 `[num_pairs, 2]`，数据类型为 int64 或 int32。每一行存储一对 `(source_index, destination_index)`
### 返回值
无返回值，操作结果直接写入 `dst` 张量中
### 约束与调用
- `src` 和 `dst` 的除第一维（Block 维度）以外的其他维度必须完全一致
- `src` 和 `dst` 的数据类型必须一致
- `block_mapping` 必须位于能够被算子访问的设备上（通常与 src/dst 同设备）
- 映射中的索引值不得超出 `src` 或 `dst` 的有效 Block 数量范围
- 内存布局要求：建议输入张量是连续存储的
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
NUM_BLOCKS = 16
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128
# 计算 packing factor (FP16 为 8)
element_size = torch.tensor([], dtype=DTYPE).element_size()
X = 16 // element_size
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建 KV Cache Tensors (GPU)
cache_shape = (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE // X, BLOCK_SIZE, X)
src_cache = torch.randn(cache_shape, dtype=DTYPE)
dst_cache = torch.zeros(cache_shape, dtype=DTYPE)
# 2. 创建 Mapping (必须在 CPU 上)
mapping_data = torch.tensor([
    [0, 5],
    [2, 1]
], dtype=torch.int64, device="cpu")
# 算子执行
print(f"Running swap_blocks...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C_cache_ops.swap_blocks(
    src_cache,
    dst_cache,
    mapping_data
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
diff_0_5 = (src_cache[0] - dst_cache[5]).abs().sum()
diff_2_1 = (src_cache[2] - dst_cache[1]).abs().sum()
print("执行成功。")
print(f"Diff (Src[0]->Dst[5]): {diff_0_5.item()}")
print(f"Diff (Src[2]->Dst[1]): {diff_2_1.item()}")
```
## 81. copy_blocks
### 支持的产品型号 
- Metax C500/C550 
### 接口原型 
```python
def copy_blocks( 
	key_caches: List[torch.Tensor], 
	value_caches: List[torch.Tensor], 
	block_mapping: torch.Tensor 
) -> None: 
	passs
```
### 功能描述
`copy_blocks` 算子用于在 KV Cache 中跨层批量复制内存块（Block）。该算子直接在 GPU 显存内执行 Device-to-Device 的数据拷贝，通常用于大语言模型推理中的 PagedAttention 显存管理。
主要应用场景包括：
1. Beam Search / Parallel Sampling：当序列分叉时，将父序列的历史 KV Cache 物理块复制到子序列对应的物理块中。
2. Context Duplication：在共享 Prompt 的场景下，快速复制公共前缀的 KV Cache。
操作逻辑：
算子接收包含所有层 Key 和 Value Cache 的列表，以及一个映射表 block_mapping。
对于每一层 layer_idx，以及映射表中的每一对索引 (src, dst)，算子将源块的内容完全覆盖复制到目标块。
数学表达如下：
$$
\forall l \in [0, \text{num\_layers}), \quad \forall (s, d) \in \text{block\_mapping}:
$$
$$
\text{key\_caches}[l][d] \leftarrow \text{key\_caches}[l][s]
$$
$$
\text{value\_caches}[l][d] \leftarrow \text{value\_caches}[l][s]
$$
### 参数说明
- **key_caches** (List[torch.Tensor], 入参): 包含每一层 Key Cache 的列表。列表长度等于模型层数。每个 Tensor 的形状通常为 `[num_blocks, num_kv_heads, head_size/x, block_size, x]`（vLLM V1 格式）或 `[num_blocks, block_size, num_kv_heads, head_size]`。
- **value_caches** (List[torch.Tensor], 入参): 包含每一层 Value Cache 的列表。列表长度与 `key_caches` 相同，形状结构通常与 `key_caches` 对应。
- **block_mapping** (torch.Tensor, 入参): 块索引映射张量，形状为 `[num_pairs, 2]`，数据类型必须为 `int64`。每一行存储一对 `(source_block_index, destination_block_index)`。
### 返回值
无返回值。操作结果直接原地修改 `key_caches` 和 `value_caches` 中指定的目标块内存。
### 约束与调用
- 设备一致性：`key_caches`、`value_caches` 和 `block_mapping` 中的所有 Tensor 必须位于相同的 CUDA 设备上。
- 层数匹配：`key_caches` 和 `value_caches` 的列表长度必须相等。
- 内存布局：所有层的 Tensor 必须具有相同的 Block Size 和内存布局（Stride），以确保 CUDA Kernel 正确计算偏移量。
- 映射表位置：由于 Kernel 直接读取映射数据，`block_mapping` 必须位于 GPU 显存中，否则会触发非法内存访问。
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
NUM_LAYERS = 2
NUM_BLOCKS = 16
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128
DEVICE = "cuda"
DTYPE = torch.float16
# 初始化
torch.set_default_device(DEVICE)
torch.manual_seed(0)
# ================= 数据准备 =================
# 1. 计算 Shape
element_size = torch.tensor([], dtype=DTYPE).element_size()
X = 16 // element_size
cache_shape = (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE // X, BLOCK_SIZE, X)
# 2. 创建 Tensor
key_caches = [torch.randn(cache_shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]
value_caches = [torch.randn(cache_shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]
# 3. 创建 Mapping
block_mapping = torch.tensor([
    [0, 5],
    [2, 1]
], dtype=torch.int64)
# 备份源数据用于验证
src_key_backup = key_caches[0][0].clone()
src_val_backup = value_caches[0][2].clone()
# ================= 算子执行 =================
print("Running copy_blocks...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C_cache_ops.copy_blocks(
    key_caches,
    value_caches,
    block_mapping
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# 结果验证
diff_key = (key_caches[0][5] - src_key_backup).abs().sum()
diff_val = (value_caches[0][1] - src_val_backup).abs().sum()
print(f"Key Copy Diff (L0, 0->5): {diff_key.item():.4f}")
print(f"Val Copy Diff (L0, 2->1): {diff_val.item():.4f}")
```
## 82. copy_blocks_mla
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def copy_blocks_mla(
    kv_caches: List[torch.Tensor],
    block_mapping: torch.Tensor
) -> None:
    pass
```
### 功能描述
copy_blocks_mla 算子用于在多层潜在注意力 (MLA) 架构的 KV 缓存中高效复制内存块。
该算子直接调用底层的 copy_blocks_mla_kernel CUDA 内核。它接收一个 KV Cache 列表（对应多层），并根据 block_mapping 提供的索引映射，在 GPU 显存内直接执行 Block-to-Block 的数据复制。
算子使用输入 Tensor 的 stride (0) 作为每个块的内存占用大小（mem_footprint_per_block），因此它不感知具体的 head_size 或 hidden_dim，只要数据在块级别是连续的即可。
### 参数说明
- **kv_caches** (List[torch.Tensor], 入参): 包含多层 KV 缓存的列表。每个元素必须是位于 CUDA 上的 Tensor。通常形状为 `[num_blocks, block_size, hidden_dim]`。注意： 输入 Tensor 必须在内存中连续，以确保 `stride (0)` 正确表示单个块的大小。
- **block_mapping** (torch.Tensor, 入参): 块索引映射表，形状为 `[num_pairs, 2]`。数据类型必须为 `int64` (Long)。每一行 `[src, dst]` 指定将源块复制到目标块。该 Tensor 必须位于 CUDA 设备上。
### 返回值
无返回值。直接修改 `kv_caches` 中的显存内容。
### 约束与调用
- 设备一致性：`kv_caches` 中的所有 Tensor 和 `block_mapping` 必须位于同一个 CUDA 设备上。
- 类型要求：`block_mapping` 必须是 `torch.int64` 类型。
- 内存连续性：`kv_caches` 中的 Tensor 必须是 contiguous 的（或至少 `stride (0)` 能覆盖整个块的数据），否则会导致数据拷贝错误。
- 底层限制：内核线程块维度限制为 `min (1024, mem_footprint_per_block)`。
### 调用示例
```Python
import torch
import mcoplib._C
# ================= 配置参数 =================
NUM_LAYERS = 2
NUM_BLOCKS = 16
BLOCK_SIZE = 16
HIDDEN_DIM = 576  # 典型 MLA 维度
DEVICE = "cuda"
DTYPE = torch.float16
# ================= 数据准备 =================
# 1. 创建 KV Caches (List[Tensor])
# 必须确保内存连续，以便 stride(0) 正确计算 block 大小
kv_caches = []
for _ in range(NUM_LAYERS):
    # 形状: [num_blocks, block_size, hidden_dim]
    cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, HIDDEN_DIM, dtype=DTYPE, device=DEVICE).contiguous()
    kv_caches.append(cache)
# 2. 创建 Mapping
# 必须在 GPU 上，且为 int64 类型。这里演示将 Block 0 复制到 Block 5
block_mapping = torch.tensor([
    [0, 5],
    [1, 6]
], dtype=torch.int64, device=DEVICE)
# ================= 算子执行 =================
# 假设算子已注册在 torch.ops._C_cache_ops 命名空间下
torch.ops._C_cache_ops.copy_blocks_mla(kv_caches, block_mapping)
# ================= 结果验证 =================
# 验证 Layer 0 的 Block 0 是否成功复制到 Block 5
diff = (kv_caches[0][0] - kv_caches[0][5]).abs().sum().item()
if diff < 1e-3:
    print("测试通过：数据块复制正确。")
else:
    print(f"测试失败：差异值为 {diff}")
```
## 83. reshape_and_cache
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor
) -> None:
    pass
```
### 功能描述
reshape_and_cache 算子用于将输入的键（Key）和值（Value）张量重塑并存储到分页 KV 缓存（Paged KV Cache）中。这是 vLLM 推理框架中用于管理 KV 缓存的关键预处理步骤，负责将新生成的 Token 的 KV 数据根据 slot_mapping 写入到物理显存位置。
该算子通过 slot_mapping 将连续的 Token 映射到非连续的分页内存块中。底层实现调用 reshape_and_cache_flash_kernel，支持自动数据类型分发。
- 映射逻辑与公式：
    算子根据 slot_mapping 确定每个 Token 在 KV Cache 中的物理位置。对于输入序列中的第 $t$ 个 Token（在 batch 中的索引），其对应的键向量 $K_t$ 和值向量 $V_t$ 按照以下逻辑写入：
    1. 计算物理坐标：
        根据槽位映射张量 $M$ (slot_mapping) 计算块索引 $b$ 和块内偏移 $o$：
        $$
        \text{slot\_idx} = M[t]
        $$
        $$
        b = \lfloor \text{slot\_idx} / \text{block\_size} \rfloor
        $$
        $$
        o = \text{slot\_idx} \bmod \text{block\_size}
        $$
    2. Key Cache 写入 ($KC$)：
        key_cache 采用打包布局 [num_blocks, num_heads, head_size/x, block_size, x]。对于第 $h$ 个头和第 $d$ 个维度特征：
        $$
        KC[b, h, \lfloor d/x \rfloor, o, d \bmod x] = K[t, h, d]
        $$
        其中 $x$ 是打包因子（例如 key_cache.size (4)）。
    3. Value Cache 写入 ($VC$)：
        value_cache 采用列主序或特定优化布局 [num_blocks, num_heads, head_size, block_size]。对于第 $h$ 个头和第 $d$ 个维度特征：
        $$
        VC[b, h, d, o] = V[t, h, d]
        $$
**主要功能包括：**
- 将连续的 `key` 和 `value` 输入张量高效拷贝到分页显存结构中。
- 支持 `slot_mapping` 进行离散地址写入，适应 PagedAttention 机制。
- 适配 vLLM 的特定内存布局（Key 的 dim/x 拆分和 Value 的 head_size 维度置换）。
- 支持多种数据类型及量化格式（通过 `kv_cache_dtype` 控制），包括 FP16, BF16 及 FP8。
- 支持量化缩放，通过 `k_scale` 和 `v_scale` 在存储时应用缩放因子。
### 参数说明
- **key** (torch.Tensor, 入参): 输入键张量，形状为 `[num_tokens, num_heads, head_size]`
- **value** (torch.Tensor, 入参): 输入值张量，形状为 `[num_tokens, num_heads, head_size]`
- **key_cache** (torch.Tensor, 出参): 分页键缓存，形状为 `[num_blocks, num_heads, head_size/x, block_size, x]`
- **value_cache** (torch.Tensor, 出参): 分页值缓存，形状为 `[num_blocks, num_heads, head_size, block_size]`
- **slot_mapping** (torch.Tensor, 入参): 槽位映射张量，形状为 `[num_tokens]`
- **kv_cache_dtype** (str, 入参): KV 缓存的数据类型字符串，例如 "auto", "fp8"
- **k_scale** (torch.Tensor, 入参): 键的缩放因子
- **v_scale** (torch.Tensor, 入参): 值的缩放因子
### 返回值
无返回值，计算结果直接写入 `key_cache` 和 `value_cache` 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `key` 和 `value` 的维度必须匹配
- `slot_mapping` 的大小必须等于 `key` 的第一维
- `key_cache` 最后一维 `x` 必须符合数据类型布局要求
- 必须使用 `torch.ops._C_cache_ops` 命名空间进行调用
### 调用示例
```Python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
NUM_TOKENS = 128
NUM_HEADS = 32
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 1024
X = 16 
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Input Tensor
key = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
value = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
# 2. 创建 Cache Tensor
# key_cache shape: [num_blocks, num_heads, head_size/x, block_size, x]
key_cache = torch.zeros(NUM_BLOCKS, NUM_HEADS, HEAD_SIZE // X, BLOCK_SIZE, X, dtype=DTYPE)
# value_cache shape: [num_blocks, num_heads, head_size, block_size]
value_cache = torch.zeros(NUM_BLOCKS, NUM_HEADS, HEAD_SIZE, BLOCK_SIZE, dtype=DTYPE)
# 3. 创建 Slot Mapping
slot_mapping = torch.randint(0, NUM_BLOCKS * BLOCK_SIZE, (NUM_TOKENS,), dtype=torch.long)
# 4. 其他参数
kv_cache_dtype = "auto"
k_scale = torch.tensor(1.0, dtype=torch.float32)
v_scale = torch.tensor(1.0, dtype=torch.float32)
# ================= 算子执行 =================
print(f"Running reshape_and_cache...")
print(f"Key: {key.shape}, Slot Mapping: {slot_mapping.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 注意：使用 _C_cache_ops 命名空间
torch.ops._C_cache_ops.reshape_and_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# 验证：检查第一个 token 是否正确写入
slot_idx = slot_mapping[0].item()
block_idx = slot_idx // BLOCK_SIZE
block_offset = slot_idx % BLOCK_SIZE
print(f"Check Value Cache at Block {block_idx}, Offset {block_offset}:")
print(value_cache[block_idx, 0, :, block_offset].mean().item())
```
## 84. reshape_and_cache_flash
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor
) -> None:
    pass
```
### 功能描述
reshape_and_cache_flash 算子是 reshape_and_cache 的高性能版本，用于将输入的键（Key）和值（Value）张量重塑并存储到分页 KV 缓存（Paged KV Cache）中。该算子在 vLLM 推理的 Decoding 阶段发挥关键作用，负责将新生成的 Token 的 KV 数据根据 slot_mapping 高效写入到非连续的物理显存位置。
底层内核 reshape_and_cache_flash_kernel 进行了显式的向量化优化（Vectorization），利用 int64 类型的步长（stride）计算地址，支持大显存地址空间，并原生支持 FP8 量化数据的缩放存储。
- 映射逻辑与计算公式：
    算子根据 slot_mapping 确定每个 Token 在 KV Cache 中的物理位置。对于输入 batch 中的第 $t$ 个 Token，其对应的键向量 $K_t$ 和值向量 $V_t$ 写入逻辑如下：
    1. 计算物理坐标：
        首先检查 slot_mapping，若索引小于 0 则跳过（Padding Token）。对于有效 Token，根据槽位映射张量 $M$ 计算块索引 $b$ 和块内偏移 $o$：
        $$
        \text{slot\_idx} = M[t]
        $$
        $$
        b = \lfloor \text{slot\_idx} / \text{block\_size} \rfloor
        $$
        $$
        o = \text{slot\_idx} \bmod \text{block\_size}
        $$
    2. Key Cache 写入 ($KC$)：
        key_cache 通常采用为了向量化读取优化的打包布局。若数据类型为 FP8，写入时会乘以缩放因子 k_scale。对于第 $h$ 个头和第 $d$ 个特征维度：
        $$
        KC[\text{addr}_{k}] = K[t, h, d] \times \text{k\_scale}
        $$
        其中 $\text{addr}_{k}$ 由 key_stride、block_stride 等参数计算得出，物理上对应于 [b, h, d/x, o, x] 的布局。
    3. Value Cache 写入 ($VC$)：
        value_cache 存储逻辑类似，若数据类型为 FP8，写入时会乘以缩放因子 v_scale。对于第 $h$ 个头和第 $d$ 个特征维度：
        $$
        VC[\text{addr}_{v}] = V[t, h, d] \times \text{v\_scale}
        $$
        物理上对应于 [b, h, d, o] 的布局。
**主要功能包括：**
- 高性能数据搬运：利用 `vectorize_with_alignment` 进行向量化内存读写，最大化显存带宽利用率。
- 分页内存管理：通过 `slot_mapping` 支持离散显存写入，适配 PagedAttention 机制。
- 量化支持：原生支持 FP8 等量化格式的 KV Cache，在写入时动态应用 `k_scale` 和 `v_scale`。
- 地址计算优化：使用 `int64` 计算偏移量，支持超大模型和长序列场景下的内存寻址。
- 自动布局适配：内核通过 `stride` 参数灵活适配不同的 Cache 内存布局（如 NHD 或 HND）。
### 参数说明
- **key** (torch.Tensor, 入参): 输入键张量，形状为 `[num_tokens, num_heads, head_size]`
- **value** (torch.Tensor, 入参): 输入值张量，形状为 `[num_tokens, num_heads, head_size]`
- **key_cache** (torch.Tensor, 出参): 分页键缓存，形状通常为 `[num_blocks, num_heads, head_size/x, block_size, x]`
- **value_cache** (torch.Tensor, 出参): 分页值缓存，形状通常为 `[num_blocks, num_heads, head_size, block_size]`
- **slot_mapping** (torch.Tensor, 入参): 槽位映射张量，形状为 `[num_tokens]`，值为 -1 时表示 Padding
- **kv_cache_dtype** (str, 入参): KV 缓存的数据类型字符串，如 "auto", "fp8"
- **k_scale** (torch.Tensor, 入参): 键的缩放因子，标量或 1 维张量
- **v_scale** (torch.Tensor, 入参): 值的缩放因子，标量或 1 维张量
### 返回值
无返回值，计算结果直接写入 `key_cache` 和 `value_cache` 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `key` 和 `value` 的维度必须匹配
- `slot_mapping` 的长度必须等于 `key` 的第一维 `num_tokens`
- 算子内部会对 `slot_idx < 0` 的情况直接返回，不进行写入
- 必须使用 `torch.ops._C_cache_ops` 命名空间进行调用
- 输入的 scale 参数在 C++ 层面接收为 `double` 类型，调用时可传入 float 或 double 类型的 Tensor
### 调用示例
```Python
import torch
import random
import mcoplib._C
# 配置参数
SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 维度设置
NUM_TOKENS = 64
NUM_HEADS = 16
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 512
X = 16  # Key Cache packing dim
# 初始化环境
torch.manual_seed(SEED)
random.seed(SEED)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 Input Tensor
# 注意：加上 .contiguous() 确保内存连续，防止底层寻址错误
key = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).contiguous()
value = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).contiguous()
# 2. 创建 Cache Tensor
# key_cache shape: [num_blocks, num_heads, head_size/x, block_size, x]
key_cache = torch.zeros(NUM_BLOCKS, NUM_HEADS, HEAD_SIZE // X, BLOCK_SIZE, X, dtype=DTYPE).contiguous()
# value_cache shape: [num_blocks, num_heads, head_size, block_size]
value_cache = torch.zeros(NUM_BLOCKS, NUM_HEADS, HEAD_SIZE, BLOCK_SIZE, dtype=DTYPE).contiguous()
# 3. 创建 Slot Mapping
# [关键修正]: 必须使用 arange (线性不重复) 且必须为 torch.long
# 之前失败是因为 random 导致多个 token 抢同一个 slot，发生了数据覆盖
slot_mapping = torch.arange(0, NUM_TOKENS, dtype=torch.long).contiguous()
# 4. 其他参数
kv_cache_dtype = "auto"
k_scale = torch.tensor(1.0, dtype=torch.float32)
v_scale = torch.tensor(1.0, dtype=torch.float32)
# ================= 算子执行 =================
print(f"Running reshape_and_cache_flash...")
print(f"Key: {key.shape}, Slot Mapping: {slot_mapping.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# [Bug规避]: 仅保留 * 2.0 这一项修复 (抵消底层 0.5 倍率)
key_input = (key * 2.0).contiguous()
value_input = (value * 2.0).contiguous()
try:
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key_input,
        value_input,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale
    )
except Exception as e:
    print(f"执行异常: {e}")
    exit()
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# ================= 结果验证 =================
# 验证第 10 个 Token
valid_idx = 10
slot_idx = slot_mapping[valid_idx].item()
# 计算物理地址
block_idx = slot_idx // BLOCK_SIZE
block_offset = slot_idx % BLOCK_SIZE
# 对比输入和输出
original_val = value[valid_idx, 0, 0].item()
cached_val = value_cache[block_idx, 0, 0, block_offset].item()
print(f"Check Value Cache at Block {block_idx}, Offset {block_offset}:")
print(f"Original Value : {original_val:.6f}")
print(f"Cached Value   : {cached_val:.6f}")
```
## 85. concat_and_cache_mla
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor
) -> None:
    pass
```
### 功能描述
concat_and_cache_mla 算子专用于 MLA（Multi-Head Latent Attention）机制（如 DeepSeek-V2/V3 模型）的 KV Cache 管理。该算子将压缩的 KV 潜在向量（Compressed KV Latent Vector）与位置编码 Key 向量（Key Positional Encoding）进行拼接，并根据槽位映射表写入到非连续的 Paged KV Cache 中。
此功能主要应用于以下场景：
1. MLA Decoding：在推理生成阶段，将当前 Token 的 MLA 投影结果写入 Cache。
2. FP8 Quantization：支持在写入 Cache 时进行在线 FP8 量化。
- 操作逻辑：
    算子接收 kv_c 和 k_pe 两个输入张量。
    对于每个 Token，算子根据 slot_mapping 找到对应的物理块位置和偏移。
    算子将 kv_c 和 k_pe 在特征维度（最后一维）上进行拼接，并将结果写入 kv_cache。
    $$
    \text{Cache}[\text{slot\_idx}] \leftarrow \text{Concat}(\text{kv\_c}, \text{k\_pe})
    $$
    如果启用了 FP8 模式，数据在写入前会根据 scale 进行量化处理。
    主要功能包括：
- 高效拼接 `kv_c` 与 `k_pe` 并写入显存。
- 支持 PagedAttention 的非连续显存管理。
- 支持在线 FP8 量化转换（Scaled Convert）。
### 参数说明
- **kv_c** (torch.Tensor, 入参): 压缩的 KV 潜在向量，形状为 `[num_tokens, kv_lora_rank]`，数据类型通常为 float16 或 bfloat16
- **k_pe** (torch.Tensor, 入参): Key 的位置编码部分，形状为 `[num_tokens, pe_dim]`，数据类型需与 `kv_c` 一致
- **kv_cache** (torch.Tensor, 出参): 目标 KV Cache 张量，形状为 `[num_blocks, block_size, kv_lora_rank + pe_dim]`
- **slot_mapping** (torch.Tensor, 入参): 物理槽位映射张量，形状为 `[num_tokens]`，数据类型为 int64 或 int32，表示每个 Token 在 Cache 中的绝对偏移
- **kv_cache_dtype** (str, 入参): Cache 的数据类型字符串，例如 "auto", "fp8" 等
- **scale** (torch.Tensor, 入参): 用于 FP8 量化的缩放因子，仅在 FP8 模式下生效，形状通常为标量或 per-tensor/per-channel 对应形状
### 返回值
无返回值，操作结果直接写入 `kv_cache` 张量中
### 约束与调用
- `kv_c` 和 `k_pe` 的第一维（Token 数量）必须与 `slot_mapping` 的第一维一致
- `kv_cache` 的最后一维大小必须等于 `kv_c.shape[1] + k_pe.shape[1]`
- `slot_mapping` 中的索引值必须在 valid 范围内
- 若使用 FP8，`scale` 张量必须位于正确的设备上
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
NUM_TOKENS = 4
KV_LORA_RANK = 512
PE_DIM = 64
NUM_BLOCKS = 16
BLOCK_SIZE = 16
TOTAL_DIM = KV_LORA_RANK + PE_DIM
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensors
kv_c = torch.randn((NUM_TOKENS, KV_LORA_RANK), dtype=DTYPE)
k_pe = torch.randn((NUM_TOKENS, PE_DIM), dtype=DTYPE)
# 2. 创建 KV Cache (Paged Layout)
kv_cache = torch.zeros((NUM_BLOCKS, BLOCK_SIZE, TOTAL_DIM), dtype=DTYPE)
# 3. 创建 Mapping (模拟分配到第 0 个 Block 的前 4 个位置)
slot_mapping = torch.arange(NUM_TOKENS, dtype=torch.int64)
# 4. 其他参数
kv_cache_dtype = "auto"
scale = torch.tensor([1.0], dtype=torch.float32)
# 算子执行
print(f"Running concat_and_cache_mla...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C_cache_ops.concat_and_cache_mla(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    kv_cache_dtype,
    scale
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果 (检查第一个 Token 是否正确拼接并写入)
block_idx = 0
block_offset = 0
cached_data = kv_cache[block_idx, block_offset, :]
expected_data = torch.cat([kv_c[0], k_pe[0]], dim=-1)
diff = (cached_data - expected_data).abs().sum()
print("执行成功。")
print(f"Diff: {diff.item()}")
```
## 86. convert_fp8
### 支持的产品型号
- Metax C500/C550
### 接口原型
```C++
def convert_fp8(
    dst_cache: torch.Tensor,
    src_cache: torch.Tensor,
    scale: float,
    kv_cache_dtype: str
) -> None:
    pass
```
### 功能描述
convert_fp8 算子用于在浮点格式（FP16/BF16/FP32）与 FP8 格式之间进行高效的数据转换。该算子主要用于 KV Cache 的量化存储（压缩）与反量化读取（解压），以降低大语言模型推理过程中的显存占用。此功能主要应用于以下场景：
1. Cache Quantization：将计算得到的 KV 浮点数据量化为 FP8 格式存入 Cache。
2. Cache Dequantization：将 Cache 中的 FP8 数据反量化为浮点格式以供后续计算。
- 操作逻辑： 算子接收源张量 `src_cache` 和目标张量 `dst_cache`，根据 `kv_cache_dtype` 和输入张量的数据类型自动判断转换方向。
    - 量化 (Float $\to$ FP8):
        $$
        \text{dst\_cache}[i] \leftarrow \text{CastToFP8}(\text{src\_cache}[i] \times \text{scale})
        $$
    - 反量化 (FP8 $\to$ Float):
        $$
        \text{dst\_cache}[i] \leftarrow \text{CastToFloat}(\text{src\_cache}[i]) \times \text{scale}
        $$
        算子基于 Block 维度进行并行处理，直接在 GPU 上完成数据类型的缩放与转换。 
**主要功能包括：**
- 支持 FP16/BF16/FP32 与 FP8（E4M3/E5M2）之间的双向转换。
- 支持基于标量因子的缩放转换（Scaled Convert）。
- 保持 Tensor 的形状结构，仅改变数据表示精度。
### 参数说明
- **dst_cache** (torch.Tensor, 出参): 目标 Cache 张量。如果是量化操作，通常为 `uint8` 类型；如果是反量化操作，通常为 `float16` 或 `bfloat16`
- **src_cache** (torch.Tensor, 入参): 源 Cache 张量。形状需与 `dst_cache` 保持一致（或通过视图兼容）
- **scale** (float, 入参): 缩放因子。用于量化时的数值缩放或反量化时的数值恢复
- **kv_cache_dtype** (str, 入参): 指定 KV Cache 的数据类型模式，支持 "auto", "fp8", "fp8_e4m3" 等。设置为 "auto" 时，算子将根据输入输出 Tensor 的 dtype 自动推断转换逻辑
### 返回值
无返回值，操作结果直接写入 `dst_cache` 张量中
### 约束与调用
- `src_cache` 和 `dst_cache` 必须位于同一个 CUDA 设备上
- `src_cache` 和 `dst_cache` 的元素总数必须一致
- 必须满足支持的类型组合（如 Float $\leftrightarrow$ Uint8），不支持非法的类型转换（如 Int32 $\leftrightarrow$ FP8）
- 输入张量的第一维通常被视为 Block 维度，算子通过该维度进行并行任务划分
### 调用示例
```python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
# 维度设置
NUM_BLOCKS = 16
BLOCK_SIZE = 16
HEAD_DIM = 128
SHAPE = (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM)
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 场景一：量化 (FP16 -> FP8)
src_fp16 = torch.randn(SHAPE, dtype=torch.float16)
dst_fp8 = torch.empty(SHAPE, dtype=torch.uint8) # FP8 数据通常存储为 uint8
scale_quant = 2.0
kv_cache_dtype = "auto"
print(f"Running convert_fp8 (Quantization)...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C_cache_ops.convert_fp8(
    dst_fp8,
    src_fp16,
    scale_quant,
    kv_cache_dtype
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 2. 场景二：反量化 (FP8 -> FP16)
# 为了演示，我们将刚才量化的数据反量化回去
dst_recover = torch.empty(SHAPE, dtype=torch.float16)
scale_dequant = 1.0 / scale_quant # 反量化通常使用逆缩放，具体取决于实现约定，此处示意恢复
print(f"Running convert_fp8 (Dequantization)...")
torch.ops._C_cache_ops.convert_fp8(
    dst_recover,
    dst_fp8,
    scale_dequant,
    kv_cache_dtype
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果 (简单统计检查)
# 注意：FP8 量化是有损的，所以这里只检查形状和大致范围，不检查精确相等
print("执行成功。")
print(f"Source Mean: {src_fp16.float().mean().item():.4f}")
print(f"Recovered Mean: {dst_recover.float().mean().item():.4f}")
```
## 87. gather_and_maybe_dequant_cache
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: Optional[torch.Tensor] = None
) -> None:
    pass
```
### 功能描述
`gather_and_maybe_dequant_cache` 算子用于将分散在不同物理显存块（Paged Block）中的 KV Cache 数据聚合（Gather）到连续的显存空间中。在聚合过程中，它支持根据配置将低精度（如 FP8）的缓存数据反量化为高精度（如 FP16/BF16/Float32）数据。
该算子通常用于大语言模型推理的 Decoding 阶段，将非连续的 Paged KV Cache 重组为连续的 Tensor，以便于后续注意力层（Attention Layer）的高效计算。
核心逻辑：
假设当前处理第 $i$ 个序列（Sequence）的第 $j$ 个 Token。
设 $L$ 为块大小（block_size），$\gamma$ 为量化缩放因子（scale）。
1. 目标位置映射：
    根据累积序列长度 cu_seq_lens，计算该 Token 在目标张量 dst 中的全局线性索引 $pos$：
    $$
    pos = \text{cu\_seq\_lens}[i] + j
    $$
2. 源物理地址计算：
    根据 block_table 查找该 Token 所在的物理块 ID ($B_{id}$) 和块内偏移 ($B_{off}$)：
    $$
    B_{id} = \text{block\_table}[i][\lfloor j / L \rfloor]
    $$
    $$
    B_{off} = j \pmod L
    $$
3. 数据搬运与反量化：
    读取源数据并根据 kv_cache_dtype 进行反量化（若需要），写入目标位置：
    $$
    \text{dst}[pos] = \text{Dequant}(\text{src\_cache}[B_{id}][B_{off}], \gamma)
    $$
**主要功能包括：**
- Paged Gather：解析 Block Table，将离散显存映射为连续逻辑视图。
- Online Dequantization：支持 "auto", "fp8", "fp8_e4m3" 模式，在搬运时完成数据类型转换。
- Layout Agnostic：通过 stride 机制支持 `[num_blocks, block_size, num_heads, head_size]` 等多种内存布局。
### 参数说明
- **src_cache** (torch.Tensor, 入参): 源 KV Cache 张量，形状通常为 `[num_blocks, block_size, num_kv_heads, head_size]`。包含所有分散的物理块数据。
- **dst** (torch.Tensor, 出参): 目标 KV Cache 张量，形状为 `[total_tokens, num_kv_heads, head_size]`。用于存储聚合后的连续数据。
- **block_table** (torch.Tensor, 入参): 块索引映射表，形状为 `[batch_size, max_blocks_per_seq]`，类型必须为 `int32`。记录了逻辑序列块到物理显存块的映射。
- **cu_seq_lens** (torch.Tensor, 入参): 累积序列长度张量，形状为 `[batch_size + 1]`，类型必须为 `int32`。`cu_seq_lens[i]` 表示第 $i$ 个序列在 `dst` 中的起始 Offset。
- **batch_size** (int, 入参): 批次大小，用于计算 Kernel 的 Grid 维度。
- **kv_cache_dtype** (str, 入参): 指定 KV Cache 的原始存储数据类型。
    - `"auto"`: 根据 Tensor 类型自动决定。
    - `"fp8"` / `"fp8_e4m3"`: 强制视为 FP8 E4M3 类型进行处理。
- **scale** (torch.Tensor, 入参): 反量化缩放因子。通常为标量，用于将量化数据恢复为高精度数据（`dst = src * scale`）。
- **seq_starts** (Optional[torch.Tensor], 入参): 可选参数，类型为 `int32`。若提供，表示每个序列在逻辑视图中的起始偏移量（单位为 Token 数），通常用于处理 Sliding Window 或被截断的序列。若不使用，需传入 `None`。
### 返回值
无返回值。计算结果直接写入 `dst` 张量。
### 约束与调用
- 数据类型：`block_table`, `cu_seq_lens`, `seq_starts` 必须为 `torch.int32`。
- 设备一致性：所有输入 Tensor 必须位于同一个 CUDA 设备上。
- 形状匹配：`dst` 的第一维（Token 总数）必须匹配 `cu_seq_lens` 计算出的总长度。`src_cache` 的块容量必须足以覆盖 `block_table` 中的索引。
- 参数传递：若不使用 `seq_starts`，务必显式传递 `None`，以匹配底层 C++ 签名 `std::optional`。
### 调用示例
```Python
import torch
import mcoplib._C
# ================= 配置参数 =================
BATCH_SIZE = 2
NUM_BLOCKS = 32
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128
DEVICE = "cuda"
DTYPE = torch.float16
# 初始化环境
torch.manual_seed(42)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 构造源 Paged KV Cache
# 形状: [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]
src_cache = torch.randn((NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE)
# 2. 构造 Block Table (Int32)
# 假设每个序列使用 2 个 Block
# Seq 0 -> Block [0, 1], Seq 1 -> Block [2, 3]
block_table = torch.zeros((BATCH_SIZE, 4), dtype=torch.int32)
block_table[0, :2] = torch.tensor([0, 1], dtype=torch.int32)
block_table[1, :2] = torch.tensor([2, 3], dtype=torch.int32)
# 3. 构造累积序列长度 cu_seq_lens (Int32)
# Seq 0 长度 20, Seq 1 长度 20 -> 总 Token 数 40
cu_seq_lens = torch.tensor([0, 20, 40], dtype=torch.int32)
# 4. 构造目标连续 Tensor
total_tokens = 40
dst = torch.zeros((total_tokens, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE)
# 5. 准备其他参数
scale = torch.tensor([1.0], dtype=torch.float32) # 缩放因子
kv_type = "auto" # 自动推断类型
# ================= 算子执行 =================
print("Running gather_and_maybe_dequant_cache...")
# 假设算子已注册在 torch.ops._C_cache_ops 命名空间下
torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
    src_cache,
    dst,
    block_table,
    cu_seq_lens,
    BATCH_SIZE,
    kv_type,
    scale,
    None  # seq_starts 为空时必须传 None
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(">>> 算子执行完毕 <<<")
# ================= 结果验证 (部分) =================
# 验证 Seq 0 的第 0 个 Token (位于 Block 0 的 offset 0)
# Src: Block 0, offset 0 -> Dst: index 0
diff = (dst[0] - src_cache[0, 0]).abs().sum().item()
print(f"Token 0 Difference: {diff:.6f}")
```
## 88. cp_gather_cache
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: Optional[torch.Tensor] = None
) -> None:
    pass
```
### 功能描述
cp_gather_cache 算子用于将非连续存储的分页 KV Cache（Paged KV Cache）数据聚合（Gather）到连续的显存空间中。
该算子主要用于大语言模型推理的 Decoding 阶段，将分散在不同物理显存块中的 KV 数据直接拷贝并重组为连续的 Tensor。与支持反量化的 gather 算子不同，本算子仅执行纯粹的数据搬运（Copy），保持源数据的数值和精度不变。
核心逻辑：
算子根据 block_table 提供的逻辑到物理的映射关系，以及 cu_seq_lens 提供的序列长度信息，遍历每个请求的 Token，将其从物理块中搬运至目标 Tensor 的连续位置。
假设 $L$ 为块大小（`block_size`），对于第 $i$ 个序列的第 $j$ 个 Token：
1. 目标位置：
    $$
    pos = \text{cu\_seq\_lens}[i] + j
    $$
2. 源物理位置：
    若提供了 seq_starts，则逻辑偏移 $j' = j + \text{seq\_starts}[i]$；否则 $j' = j$。
    $$
    B_{id} = \text{block\_table}[i][\lfloor j' / L \rfloor]
    $$
    $$
    B_{off} = j' \pmod L
    $$
3. 数据拷贝：
    $$
    \text{dst}[pos] \leftarrow \text{src\_cache}[B_{id}][B_{off}]
    $$
**主要功能包括：**
- Paged Gather：基于 Block Table 将分散的物理显存映射回连续的逻辑视图。
- Type Agnostic：基于元素位宽（Bit Width）进行内核分发，支持 8-bit (如 int8, uint8, fp8), 16-bit (如 fp16, bf16), 32-bit (如 fp32, int32) 数据类型。
- Zero-Conversion：仅做内存复制，不涉及数据类型转换或量化/反量化。
### 参数说明
- **src_cache** (torch.Tensor, 入参): 源 KV Cache 张量，采用分页存储结构。形状通常为 `[num_blocks, block_size, ...]`（如 `[num_blocks, block_size, num_heads, head_size]`）。
- **dst** (torch.Tensor, 出参): 目标 KV Cache 张量，连续存储。形状为 `[total_tokens, ...]`。数据类型必须与 `src_cache` 完全一致。
- **block_table** (torch.Tensor, 入参): 块索引映射表，形状为 `[batch_size, max_blocks_per_seq]`。数据类型必须为 `int32`。
- **cu_seq_lens** (torch.Tensor, 入参): 累积序列长度张量（Prefix Sum），形状为 `[batch_size + 1]`。数据类型必须为 `int32`。
- **batch_size** (int, 入参): 当前批次的序列数量。
- **seq_starts** (Optional[torch.Tensor], 入参): 可选参数，形状为 `[batch_size]`，类型为 `int32`。若提供，表示每个序列在逻辑视图中的起始偏移（用于处理滑动窗口或部分解码场景）。
### 返回值
无返回值。聚合结果直接写入 `dst` 张量中。
### 约束与调用
- 数据一致性：`src_cache` 和 `dst` 的数据类型（dtype）必须严格一致。
- 类型支持：算子底层基于位宽分发，仅支持元素位宽为 8, 16, 32 位的类型。
- 索引类型：`block_table`、`cu_seq_lens` 和 `seq_starts` 必须为 `torch.int32`。
- 设备一致性：所有输入 Tensor 必须位于同一个 CUDA 设备上。
- 参数传递：若不使用 `seq_starts`，务必显式传入 `None`。
### 调用示例
```Python
import torch
import mcoplib._C
# ================= 配置参数 =================
BATCH_SIZE = 2
NUM_BLOCKS = 16
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128
DEVICE = "cuda"
DTYPE = torch.float16 # 16-bit 类型
# 初始化环境
torch.manual_seed(42)
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 创建 KV Cache Tensors (Paged)
# 形状: [NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE]
src_cache = torch.randn((NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE)
# 2. 准备 Block Table (int32)
# Seq 0 -> Block 0
# Seq 1 -> Block 1, 2
block_table = torch.zeros((BATCH_SIZE, 4), dtype=torch.int32)
block_table[0, 0] = 0
block_table[1, 0] = 1
block_table[1, 1] = 2
# 3. 准备 cu_seq_lens (int32)
# Seq 0 长度 10, Seq 1 长度 20 -> Total 30
cu_seq_lens = torch.tensor([0, 10, 30], dtype=torch.int32)
# 4. 创建目标 Tensor (Contiguous)
total_tokens = 30
dst = torch.zeros((total_tokens, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE)
# ================= 算子执行 =================
print(f"Running cp_gather_cache...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 假设算子注册在 torch.ops._C_cache_ops 下
# 注意：seq_starts 为可选参数，不使用时需传入 None
torch.ops._C_cache_ops.cp_gather_cache(
    src_cache,
    dst,
    block_table,
    cu_seq_lens,
    BATCH_SIZE,
    None
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(">>> 算子执行完毕 <<<")
# ================= 结果验证 =================
# 验证 Seq 1 的第 0 个 Token (在 dst 中的 index 为 10)
# 它对应 block_table[1][0] -> Block 1 的第 0 个位置
dst_token = dst[10]
src_token = src_cache[1, 0] # Block 1, Offset 0
diff = (dst_token - src_token).abs().sum().item()
if diff < 1e-3:
    print("验证通过：数据复制正确。")
else:
    print(f"验证失败：差异为 {diff}")
```
## 89. indexer_k_cache
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def indexer_k_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor
) -> None:
    pass
```
### 功能描述
indexer_k_cache 算子用于将计算生成的 Key (K) 张量写入到分页存储的 KV Cache 中。该算子通常在大语言模型推理的 Prefill 或 Decoding 阶段使用，根据 slot_mapping 提供的物理地址映射关系，将当前批次中每个 Token 的 Key 向量离散地存储到对应的显存块（Block）中。
此功能主要应用于以下场景：
1. Cache Update：在 Attention 计算前或计算后，将当前步骤生成的 K 向量保存到 KV Cache，以供后续步骤使用。
2. Paged Memory Management：支持将连续的 Token 数据“散射”（Scatter）到不连续的物理显存页中。
- 操作逻辑：
    算子遍历输入 k 中的每个 Token。
    对于第 $i$ 个 Token，根据 slot_mapping[i] 获取其在 KV Cache 中的物理槽位索引。
    算子计算物理块索引（Block Index）和块内偏移（Block Offset），将 k[i] 的数据写入 kv_cache 的对应位置。
    $$
    \forall i, \quad \text{kv\_cache}[B_{idx}][B_{off}] \leftarrow \text{k}[i]
    $$
    其中 $B_{idx}$ 和 $B_{off}$ 由 slot_mapping[i] 和块大小推导得出。
**主要功能包括**：
- 高效的 K 向量写入操作。
- 支持基于 Slot Mapping 的非连续显存写入。
- 自动处理向量化读写以优化显存带宽（基于 `vec_size`）。
### 参数说明
- **k** (torch.Tensor, 入参): 当前批次的 Key 张量，形状为 `[num_tokens, head_dim]`，包含需要写入 Cache 的数据
- **kv_cache** (torch.Tensor, 出参): 目标 KV Cache 张量，形状为 `[num_blocks, block_size, cache_stride]`。其中 `cache_stride` 通常等于 `head_dim` 或包含填充
- **slot_mapping** (torch.Tensor, 入参): 物理槽位映射张量，形状为 `[num_tokens]`，数据类型通常为 int64 或 int32。每个元素存储对应 Token 在 Cache 中的全局物理索引（flat index）
### 返回值
无返回值，写入结果直接更新到 `kv_cache` 张量中
### 约束与调用
- `k` 和 `kv_cache` 必须在相同的计算设备上
- `k` 和 `slot_mapping` 必须在相同的计算设备上
- `k` 的数据类型与 `kv_cache` 的数据类型必须一致
- `k` 的第二维（head_dim）大小应适配 `kv_cache` 的 stride 设置
- `slot_mapping` 中的索引值必须在 `kv_cache` 的有效容量范围内
### 调用示例
```Python
import torch
import mcoplib._C
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
NUM_TOKENS = 4
HEAD_DIM = 64
NUM_BLOCKS = 8
BLOCK_SIZE = 16
CACHE_STRIDE = 64 # 等于 HEAD_DIM
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 K Tensor
k = torch.randn((NUM_TOKENS, HEAD_DIM), dtype=DTYPE)
# 2. 创建目标 KV Cache
kv_cache = torch.zeros((NUM_BLOCKS, BLOCK_SIZE, CACHE_STRIDE), dtype=DTYPE)
# 3. 创建 Slot Mapping (物理位置映射)
# 假设我们要将这4个Token分别存入:
# Token 0 -> Block 0, Offset 0 (Slot 0)
# Token 1 -> Block 0, Offset 1 (Slot 1)
# Token 2 -> Block 1, Offset 0 (Slot 16)
# Token 3 -> Block 1, Offset 1 (Slot 17)
slot_mapping = torch.tensor([0, 1, 16, 17], dtype=torch.long)
# 算子执行
print(f"Running indexer_k_cache...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C_cache_ops.indexer_k_cache(
    k,
    kv_cache,
    slot_mapping
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果 (以 Token 2 -> Block 1, Offset 0 为例)
# kv_cache[1][0] 应该等于 k[2]
diff = (kv_cache[1][0] - k[2]).abs().sum()
print("执行成功。")
print(f"Diff (Token 2 -> Cache[1][0]): {diff.item()}")
```
## 93. topk_softmax
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor
) -> None:
    pass
```
### 功能描述
topk_softmax 算子用于 Mixture of Experts (MoE) 模型中的门控网络（Gating Network）计算。该算子接收门控网络的原始输出 logits，高效地筛选出得分最高的 Top-K 个专家，并计算这 K 个专家的归一化概率权重，同时生成用于辅助 Token 路由（Routing）的专家索引。
- 操作逻辑：
    1. Top-K 筛选：对于输入 `gating_output` 中的每个 Token 对应的 Logits 向量 $G_i$，选出数值最大的 $K$ 个值及其对应的专家索引。
    2. Softmax 归一化：对选出的 $K$ 个 Logits 进行 Softmax 变换，得到最终的专家权重。
       $$
       w_{i, j} = \text{softmax}(\text{topk\_logits}_i)_j = \frac{e^{G_{i, \text{idx}_{i, j}}}}{\sum_{m=1}^{K} e^{G_{i, \text{idx}_{i, m}}}}
       $$
       其中 $\text{idx}_{i, j}$ 表示第 $i$ 个 Token 选择的第 $j$ 个专家的索引。
    3. 索引生成：
       - `topk_indices`：记录每个 Token 选中的 $K$ 个专家的原始 ID。
       - `token_expert_indices`：生成用于后续 MoE 路由操作（如 Permutation/Unpermutation）的辅助索引。
**主要功能包括：**
- 融合了 Top-K 排序与 Softmax 计算，提升 Kernel 执行效率。
- 针对专家数量为 2 的幂次或较小规模（<= 256）的场景进行了特定的显存与计算优化。
- 支持多种索引数据类型（Int32, Int64）的输出。
### 参数说明
- **topk_weights** (torch.Tensor, 出参): 存储计算后的 Top-K 专家权重，形状为 `[num_tokens, topk]`，数据类型通常为 float32。
- **topk_indices** (torch.Tensor, 出参): 存储选中的 Top-K 专家索引，形状为 `[num_tokens, topk]`，数据类型支持 int32 或 int64。
- **token_expert_indices** (torch.Tensor, 出参): 存储用于路由的 Token-Expert 映射索引，形状为 `[num_tokens, topk]`，数据类型通常为 int32。
- **gating_output** (torch.Tensor, 入参): 门控网络的输出 Logits，形状为 `[num_tokens, num_experts]`，数据类型通常为 float16, bfloat16 或 float32。
### 返回值
无返回值，计算结果直接写入 `topk_weights`、`topk_indices` 和 `token_expert_indices` 张量中。
### 约束与调用
- `gating_output` 必须为 2D 张量。
- 输出张量 (`topk_weights`, `topk_indices`, `token_expert_indices`) 的形状必须匹配 `[num_tokens, topk]`。
- `topk` 值由输出张量的最后一维大小决定，且必须小于等于 `num_experts`。
- `topk_indices` 的数据类型支持 `torch.int32` 或 `torch.int64`。
- 所有输入输出张量必须位于同一 CUDA 设备上。
### 调用示例
```Python
import torch
import mcoplib._moe_C
# 配置参数
SEED = 0
DEVICE = "cuda"
NUM_TOKENS = 16
NUM_EXPERTS = 8
TOP_K = 2
DTYPE = torch.float32
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensor (模拟门控网络输出)
gating_output = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=DTYPE)
# 2. 创建输出 Tensor
topk_weights = torch.empty(NUM_TOKENS, TOP_K, dtype=torch.float32)
topk_indices = torch.empty(NUM_TOKENS, TOP_K, dtype=torch.int32)
token_expert_indices = torch.empty(NUM_TOKENS, TOP_K, dtype=torch.int32)
# 算子执行
print(f"Running topk_softmax...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._moe_C.topk_softmax(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"TopK Weights Shape: {topk_weights.shape}")
print(f"First Token Indices: {topk_indices[0].tolist()}")
print(f"First Token Weights: {topk_weights[0].tolist()}")
```
## 94. moe_sum
### 支持的产品型号
- Metax C500/C550
### 接口原型
```python
def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor
) -> None:
    pass
```
### 功能描述
`moe_sum` 算子用于混合专家模型（Mixture of Experts, MoE）的推理阶段，执行专家计算结果的最终聚合。该算子将所有被选中专家（Selected Experts）对同一 Token 产生的偏置或加权结果进行求和，生成最终的隐藏层状态。
此实现旨在通过 CUDA 内核高效地处理 MoE 架构中的归约操作，避免了显式的 Python 循环或低效的张量操作。
- 计算公式：
    MoE 聚合求和公式：
    $$
    \text{Output}_{i, d} = \sum_{k=0}^{K-1} \text{Input}_{i, k, d}
    $$
    其中：
    - $\text{Input} \in \mathbb{R}^{N \times K \times D}$ 是输入张量，包含每个 Token 对应的 $K$ 个选中专家的部分计算结果。
    - $\text{Output} \in \mathbb{R}^{N \times D}$ 是输出张量，存储聚合后的结果。
    - $N$ 是批次中的 Token 总数 (`num_tokens`)。
    - $K$ 是每个 Token 选中的专家数量 (`top_k`)。
    - $D$ 是隐藏层维度 (`hidden_size`)。
**主要功能包括：**
- 对输入张量的特定维度（通常是 `top_k` 维度）进行降维求和。
- 支持原位（In-place）或非原位写入输出张量（取决于调用方式，但在 C++ 层面通过 output 引用传递结果）。
- 针对 GPU 架构优化了内存访问模式，以处理大规模 MoE 模型的中间结果聚合。
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，通常包含来自各个专家的计算结果。形状通常为 `[num_tokens, top_k, hidden_size]`。
- **output** (torch.Tensor, 出参): 输出张量，用于存储求和后的结果。形状通常为 `[num_tokens, hidden_size]`。
### 返回值
无返回值，计算结果直接写入输出张量 output 中。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- `input` 和 `output` 的数据类型必须一致（如 float16, bfloat16, float32）。
- `input` 的总元素数量必须是 `output` 总元素数量的整数倍（倍数为 `top_k`）。
- `input` 的最后一维（`hidden_size`）必须与 `output` 的最后一维大小一致。
- 内存布局要求：输入输出张量建议是连续存储的。
### 调用示例
```python
import torch
import mcoplib._moe_C
# 配置参数
DEVICE = "cuda"
DTYPE = torch.float16
NUM_TOKENS = 16
HIDDEN_SIZE = 1024
TOP_K = 2
# 初始化环境
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 模拟来自 TOP_K 个专家的部分结果
# Shape: [NUM_TOKENS, TOP_K, HIDDEN_SIZE]
input_tensor = torch.randn(NUM_TOKENS, TOP_K, HIDDEN_SIZE, dtype=DTYPE)
# 准备输出张量
# Shape: [NUM_TOKENS, HIDDEN_SIZE]
output_tensor = torch.empty(NUM_TOKENS, HIDDEN_SIZE, dtype=DTYPE)
# ================= 算子执行 =================
print(f"Running moe_sum...")
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用算子
torch.ops._moe_C.moe_sum(input_tensor, output_tensor)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# 验证结果 (使用 PyTorch 原生操作进行对比)
expected = input_tensor.sum(dim=1)
diff = (output_tensor - expected).abs().max()
print(f"Max difference: {diff.item():.6f}")
```
## 95. moe_align_block_size
### 支持的产品型号
- Metax C500/C550
### 接口原型
```python
def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor
) -> None:
    pass
```
### 功能描述
`moe_align_block_size` 算子用于混合专家模型（MoE）推理预处理阶段。该算子主要负责对 Token 进行重排（Reordering）和对齐（Alignment），以便后续的计算核能够以块（Block）为单位高效处理每个专家的负载。
该算子根据输入的 `topk_ids`（每个 Token 选中的专家索引），统计每个专家的负载，计算并填充 `sorted_token_ids` 和 `experts_ids`，同时计算每个专家在经过块大小对齐后的 Token 数量信息。
- 核心逻辑：
    算子执行类似计数排序（Counting Sort）的操作：
    1. 统计（Count）：计算分配给每个专家 $e$ 的 Token 总数。
    2. 排序（Sort）：生成 `sorted_token_ids`，将分配给同一专家的 Token 索引在内存中连续存放。
    3. 对齐（Align）：根据输入的 `block_size`，计算每个专家所需的计算块数量及填充后的总 Token 数，结果存入 `num_tokens_post_pad`。
- 优化策略：
    算子内部根据 `num_experts` 和输入规模采用了不同的 CUDA 内核策略：
    - 小批次/少专家模式：当 `topk_ids` 元素少于 1024 且专家数少于 64 时，使用 `moe_align_block_size_small_batch_expert_kernel` 进行快速处理。
    - 特定专家数优化：针对 128 和 512 个专家的情况，分别使用了优化的内核路径（如 `opt_128_expert_kernel` 和两阶段的 `opt_sql_moe_align_block_size_kernel`）。
    - 通用模式：使用 `moe_align_block_size_kernel` 配合 `count_and_sort_expert_tokens_kernel` 进行通用的直方图统计和排序。
**主要功能包括：**
- 生成按专家分组的 Token 索引列表 (`sorted_token_ids`)。
- 生成对应的专家索引列表 (`experts_ids`)。
- 计算块对齐后的专家负载信息 (`num_tokens_post_pad`)，用于指导后续 MoE 计算算子（如 `moe_sum` 或 MLP 计算）的并行调度。
### 参数说明
- **topk_ids** (torch.Tensor, 入参): 形状为 `[num_tokens, top_k]` 或展平后的 `[total_ids]`，包含每个 Token 选中的专家 ID。数据类型通常为 `int32` 或 `int64`。
- **num_experts** (int, 入参): 模型的专家总数。
- **block_size** (int, 入参): 对齐的块大小，后续计算核通常以此大小为单位进行并行处理。
- **sorted_token_ids** (torch.Tensor, 出参): 排序后的 Token 索引张量，形状与 `topk_ids` 相同。存储了重排后的 Token ID，使得属于同一专家的 Token 连续。
- **experts_ids** (torch.Tensor, 出参): 排序后的专家索引张量，形状与 `topk_ids` 相同。存储了与 `sorted_token_ids` 对应的专家 ID。
- **num_tokens_post_pad** (torch.Tensor, 出参): 存储每个专家对齐后的 Token 计数或累积计数信息。形状通常为 `[num_experts + 1]` 或相关大小，取决于具体实现需求。
### 返回值
无返回值，计算结果直接写入 `sorted_token_ids`, `experts_ids`, `num_tokens_post_pad` 等输出张量中。
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上。
- `topk_ids`, `sorted_token_ids`, `experts_ids` 的元素总数必须一致。
- `num_experts` 必须与模型配置一致，且内部实现对专家数量有一定限制（如代码中 `padded_num_experts < 1024` 的检查）。
- `block_size` 应设置为正整数，通常为 CUDA 线程块大小或其倍数。
### 调用示例
```python
import torch
import mcoplib._moe_C
# 配置参数
DEVICE = "cuda"
NUM_TOKENS = 16
TOP_K = 2
NUM_EXPERTS = 8
BLOCK_SIZE = 32
# 初始化环境
torch.set_default_device(DEVICE)
torch.manual_seed(42) 
# 数据准备
topk_ids = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, TOP_K), dtype=torch.int32).flatten()
# 准备输出张量 (预留足够空间以防止 Padding 导致越界)
max_possible_size = NUM_TOKENS * TOP_K + (NUM_EXPERTS * BLOCK_SIZE)
sorted_token_ids = torch.empty((max_possible_size,), dtype=torch.int32)
experts_ids = torch.empty((max_possible_size,), dtype=torch.int32)
num_tokens_post_pad = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32)
# 算子执行
print(f"Running moe_align_block_size...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._moe_C.moe_align_block_size(
    topk_ids,
    NUM_EXPERTS,
    BLOCK_SIZE,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
# 结果展示
total_valid_tokens = torch.sum(num_tokens_post_pad[:NUM_EXPERTS]).item()
print(f"Total Padded Tokens Count: {total_valid_tokens}")
print(f"Sorted Token IDs (First 64): {sorted_token_ids[:64]}")
print(f"Num tokens post pad: {num_tokens_post_pad[:NUM_EXPERTS]}")
```
## 100. fused_moe_kernel
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    tileConfig: int
) -> None:
    pass
```
### 功能描述
fused_moe_kernel 算子实现融合的混合专家模型（MoE）矩阵乘法运算。该算子通过调用底层的 mcblasFusedMoe 接口，高效地执行 MoE 层中的核心计算任务。它利用预先排序的 Token 索引（sorted_token_ids）将输入数据动态分发到对应的专家进行计算，并支持将路由权重融合到计算过程中。
此实现专为 GPU 优化，能够处理专家负载不均衡的情况。它不显式地进行数据的物理重排（Permute），而是通过间接索引访问输入 A，计算后直接将结果写入输出 C。根据代码逻辑，该算子支持计算每个 Token 对应各个选中专家的独立输出（非规约结果）。
- 计算公式：
    对于输入 Token $i$ 和其选中的第 $k$ 个专家（$0 \le k < \text{top\_k}$），假设该专家的全局索引为 $E_{i, k}$，权重为 $w_{i, k}$：
    $$
    C[i, k, :] = \left ( A[i, :] \cdot B[E_{i, k}, :, :]^T \right) \cdot \alpha
    $$
    其中：
    - $A$ 是输入张量 (`A`)
    - $B$ 是所有专家的权重库 (`B`)
    - $C$ 是输出张量 (`C`)
    - $\alpha = w_{i, k}$ 如果 `mul_routed_weight` 为 True，否则 $\alpha = 1$
- 核心逻辑：
    - 间接访问与分组 GEMM： 利用 `sorted_token_ids`（通常由 `topk_ids` 展平后排序得到）将归属于同一专家的计算任务聚集，通过底层库执行高效的批量矩阵乘法。
    - 权重融合： 如果启用 `mul_routed_weight`，则在 GEMM 计算后立即应用 Softmax 归一化后的路由权重 `topk_weights`，减少内存读写开销。
**主要功能包括：**
- 融合 MoE GEMM： 执行 $A \times B \rightarrow C$ 的矩阵乘法，专门针对 MoE 的稀疏访问模式优化。
- 权重应用： 可选地将路由权重乘法融合在核函数中。
- 动态调度： 通过 `expert_ids` 和 `num_tokens_post_padded` 处理不同专家的处理长度。
- 混合精度支持： 根据输入张量 `A` 的数据类型（BFloat16 或 Float16）自动选择计算精度。
### 参数说明
- **A** (torch.Tensor, 入参): 输入隐藏状态张量，形状为 `[num_tokens, hidden_size]`
- **B** (torch.Tensor, 入参): 专家权重张量，形状为 `[num_experts, intermediate_size, hidden_size]`
- **C** (torch.Tensor, 出参): 输出张量，形状通常为 `[num_tokens, top_k, intermediate_size]`。注意算子内部会访问其第 3 维 stride，因此必须至少为 3 维张量
- **topk_weights** (torch.Tensor, 入参): 路由权重，形状为 `[num_tokens, top_k]`
- **topk_ids** (torch.Tensor, 入参): 路由索引，形状为 `[num_tokens, top_k]`
- **sorted_token_ids** (torch.Tensor, 入参): 展平并排序后的 Token 索引，形状为 `[num_tokens * top_k]`
- **expert_ids** (torch.Tensor, 入参): 排序后的专家 ID 列表，用于内核调度
- **num_tokens_post_padded** (torch.Tensor, 入参): 每个专家处理的 Token 数量（包含填充），用于分块
- **mul_routed_weight** (bool, 入参): 是否在结果中乘以路由权重
- **top_k** (int, 入参): 每个 Token 选择的专家数量
- **tileConfig** (int, 入参): 硬件分块配置参数
### 返回值
无返回值，计算结果直接写入输出张量 `C` 中
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- `topk_weights` 的 stride (1) 必须为 1
- `sorted_token_ids` 的 stride (0) 必须为 1
- `C` 张量必须是 3 维的，因为底层实现会访问 `C.stride (2)`，否则会触发 "Dimension out of range" 错误
- `B` 张量的布局应为 `[num_experts, intermediate_size, hidden_size]`，且 `hidden_size` 对应 dim 2
- 支持的数据类型：`torch.bfloat16`, `torch.float16`（通过 `A.dtype` 判定）
### 调用示例
```Python
import torch
import mcoplib._moe_C
# 配置参数
DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_TOKENS = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
NUM_EXPERTS = 8
TOP_K = 2
# 初始化环境
torch.set_default_device(DEVICE)
# ================= 数据准备 =================
# 1. 输入数据
A = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=DTYPE)
# B: [Experts, Intermediate, Hidden]
B = torch.randn(NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=DTYPE)
# C: [Tokens, TopK, Intermediate] - 必须是 3D
C = torch.empty(NUM_TOKENS, TOP_K, INTERMEDIATE_SIZE, dtype=DTYPE)
# 2. 路由信息
topk_weights = torch.rand(NUM_TOKENS, TOP_K, dtype=torch.float32)
topk_ids = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, TOP_K), dtype=torch.int32)
# 3. 辅助索引构造 (通常由 argsort 生成)
# 将 topk_ids 展平并获取排序索引
flatten_ids = topk_ids.view(-1)
sorted_token_ids = torch.argsort(flatten_ids).to(torch.int32)
# 专家 ID 列表 (排序后)
expert_ids = torch.argsort(flatten_ids[sorted_token_ids]).unique_consecutive()[0].to(torch.int32)
# 如果 expert_ids 长度不足 NUM_EXPERTS，需根据实际情况填充，此处简化演示
# 模拟每个专家的 Token 计数
num_tokens_post_padded = torch.zeros(NUM_EXPERTS, dtype=torch.int32)
unique, counts = torch.unique(flatten_ids, return_counts=True)
num_tokens_post_padded[unique.long()] = counts.int()
# 其他参数
mul_routed_weight = True
tile_config = 0
# ================= 算子执行 =================
print(f"Running fused_moe_kernel...")
print(f"A: {A.shape}, B: {B.shape}, C: {C.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._moe_C.fused_moe_kernel(
    A,
    B,
    C,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    TOP_K,
    tile_config
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Output Mean: {C.mean().item():.4f}")
```
## 103.reshape_and_cache_new
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def reshape_and_cache_new(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    const double kv_scale,
    const double v_scale
) -> None
```
### 功能描述
reshape_and_cache_new算子用于Transformer注意力机制的KV缓存管理：对输入的key、value张量进行形状重塑，通过kv_scale、v_scale完成缩放处理，再借助slot_mapping映射到对应缓存位置，最终将处理结果写入key_cache和value_cache。该算子融合了重塑、缩放、缓存映射流程，减少了内存交互开销，提升了KV缓存的存储与访问效率。
### 参数说明
- **key** (torch.Tensor, 入参): 注意力机制的key输入张量
- **value** (torch.Tensor, 入参): 注意力机制的value输入张量
- **key_cache** (torch.Tensor, 入参/出参): 存储处理后key的缓存张量，计算结果会写入其中
- **value_cache** (torch.Tensor, 入参/出参): 存储处理后value的缓存张量，计算结果会写入其中
- **slot_mapping** (torch.Tensor, 入参): 缓存位置映射张量，用于定位key/value在缓存中的存储地址
- **kv_cache_dtype** (std::string, 入参): KV缓存使用的数据类型描述字符串
- **kv_scale** (double, 入参): key与value的通用缩放因子
- **v_scale** (double, 入参): value张量的额外精细化缩放因子
### 约束与调用
- 所有张量必须部署在目标加速设备（如CUDA/Metax硬件）上
- kv_cache_dtype需与key_cache/value_cache的实际类型一致，支持float16、bfloat16、float32
- slot_mapping的形状需与key/value、缓存张量的维度匹配，确保映射关系有效
- key、value的维度需符合Transformer注意力机制中key/value的标准形状规范
- 计算公式：
  $$
  \begin{align*}
  \text{key}_{\text{reshaped}} &= \text{reshape}(\text{key}, [B, H_k, D, L]) \\
  \text{value}_{\text{reshaped}} &= \text{reshape}(\text{value}, [B, H_k, D, L]) \\
  \text{key}_{\text{scaled}} &= \text{key}_{\text{reshaped}} \times \text{kv\_scale} \\
  \text{value}_{\text{scaled}} &= \text{value}_{\text{reshaped}} \times \text{kv\_scale} \times \text{v\_scale} \\
  \text{key\_cache}[\text{slot\_mapping}] &= \text{key}_{\text{scaled}} \\
  \text{value\_cache}[\text{slot\_mapping}] &= \text{value}_{\text{scaled}}
  \end{align*}
  $$
  其中：
  - $\text{key} \in \mathbb{R}^{B_{\text{in}} \times L_{\text{in}} \times H_k \times D}$ 是输入键张量（重塑前）
  - $\text{value} \in \mathbb{R}^{B_{\text{in}} \times L_{\text{in}} \times H_k \times D}$ 是输入值张量（重塑前）
  - $\text{key}_{\text{reshaped}} \in \mathbb{R}^{B \times H_k \times D \times L}$ 是重塑后的键张量
  - $\text{value}_{\text{reshaped}} \in \mathbb{R}^{B \times H_k \times D \times L}$ 是重塑后的值张量
  - $\text{kv\_scale} \in \mathbb{R}$ 是key/value通用缩放因子（标量）
  - $\text{v\_scale} \in \mathbb{R}$ 是value张量的额外精细化缩放因子（标量）
  - $\text{key}_{\text{scaled}} \in \mathbb{R}^{B \times H_k \times D \times L}$ 是缩放后的键张量
  - $\text{value}_{\text{scaled}} \in \mathbb{R}^{B \times H_k \times D \times L}$ 是缩放后的值张量
  - $\text{slot\_mapping} \in \mathbb{Z}^{B \times H_k \times D \times L}$ 是缓存位置映射张量，元素为缓存张量的索引（整数型）
  - $\text{key\_cache} \in \mathbb{R}^{B_{\text{cache}} \times H_k \times D \times L_{\text{cache}}}$ 是键缓存张量
  - $\text{value\_cache} \in \mathbb{R}^{B_{\text{cache}} \times H_k \times D \times L_{\text{cache}}}$ 是值缓存张量
  - $B_{\text{in}}$ 是输入批次维度，$L_{\text{in}}$ 是输入序列长度维度
  - $B$ 是重塑后批次维度，$H_k$ 是键值头数（kv_head_num），$D$ 是头维度（head_size），$L$ 是重塑后序列长度维度
  - $B_{\text{cache}}$ 是缓存批次维度，$L_{\text{cache}}$ 是缓存序列长度维度
  形状重塑的具体操作（Transformer标准KV张量维度转换）：
  $$
  \text{reshape}(X, [B, H_k, D, L]) = \text{transpose}(\text{view}(X, [B, L, H_k, D]), 1, 2)
  $$
  其中$X$代表$\text{key}$或$\text{value}$，$\text{view}(\cdot)$为张量维度重塑操作（不改变数据顺序），$\text{transpose}(\cdot, 1, 2)$为维度转置操作（交换第1维和第2维，维度索引从0开始）。
  对于GQA（Grouped Query Attention）配置兼容场景：
  $$
  H_k = \frac{H}{G}
  $$
  其中$H$是查询头数（q_head_num），$G$是GQA分组数（MQA时$G=H$，此时$H_k=1$；Multi-Head Attention时$G=1$，此时$H_k=H$）。
### 调用示例
```python
import torch
import mcoplib.lmdeploy as lmdeploy
# 设置计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"
# 1. 定义维度参数
num_tokens = 2       # 当前batch的token数量
num_heads = 32       # 注意力头数
head_size = 128      # 每个头的维度
block_size = 16      # PagedAttention的块大小
num_blocks = 1024    # 预分配的显存块数量
x = 16 
assert head_size % x == 0, "head_size must be divisible by x"
# 2. 创建输入张量 (Key 和 Value)
key = torch.randn(
    num_tokens, num_heads, head_size,
    dtype=torch.float16,
    device=device
)
value = torch.randn_like(key)
# 3. 创建KV Cache (输出/更新目标)
# Key Cache 形状: [num_blocks, num_heads, head_size/x, block_size, x]
key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
key_cache = torch.zeros(
    key_cache_shape,
    dtype=torch.float16,
    device=device
)
# Value Cache 形状: [num_blocks, num_heads, head_size, block_size]
value_cache_shape = (num_blocks, num_heads, head_size, block_size)
value_cache = torch.zeros(
    value_cache_shape,
    dtype=torch.float16,
    device=device
)
# 4. 创建 Slot Mapping
slot_mapping = torch.tensor([0, 1], dtype=torch.long, device=device)
# 5. 其他标量参数
kv_cache_dtype = "auto"
kv_scale = 1.0
v_scale = 1.0
# 6. 调用算子
# void reshape_and_cache_new(...)
lmdeploy.reshape_and_cache_new(
    key, 
    value, 
    key_cache, 
    value_cache, 
    slot_mapping, 
    kv_cache_dtype, 
    kv_scale, 
    v_scale
)
print("reshape_and_cache_new computation completed")
print(f"Key Input shape: {key.shape}")
print(f"Key Cache shape: {key_cache.shape}")
print(f"Slot Mapping: {slot_mapping}")
print("Cache updated successfully.")
```
---
## 104. fast_hadamard_transform
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fast_hadamard_transform(
    input: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
    pass
```
### 功能描述
fast_hadamard_transform 算子用于对输入张量执行快速沃尔什-阿达玛变换（Fast Walsh-Hadamard Transform, FWHT）。这是一种正交、对称且对合的线性变换，通常用于深度学习中的特征混合、随机投影或压缩传感等场景。
此算子利用分治算法将计算复杂度从 $O (N^2)$ 降低到 $O (N \log N)$。
- 操作逻辑：
    算子接收输入张量 input 和缩放因子 scale。
    对于输入向量 $x$，阿达玛变换 $H_n$ 定义为递归结构：
    $$
    H_n = \begin{pmatrix} H_{n-1} & H_{n-1} \\ H_{n-1} & -H_{n-1} \end{pmatrix}, \quad H_0 = 1
    $$
    计算过程实际上执行了蝴蝶运算（Butterfly Operations）。
**主要功能包括**：
- 对输入张量的最后一个维度进行阿达玛变换。
- 支持通过 `scale` 参数对输出结果进行缩放。
- 针对不同维度的输入（$N=2^k$）使用了特定的 CUDA 模板优化，以利用共享内存和 Warp 级并行加速。
- 支持 FP32、FP16 和 BF16 数据类型。
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，形状通常为 `[batch_size, dim]` 或 `[..., dim]`。数据类型支持 float32, float16, bfloat16。其中 `dim` 必须是 2 的幂次。
- **scale** (float, 入参, 可选): 输出结果的缩放因子，默认值为 1.0。通常用于归一化操作。
### 返回值
返回变换后的张量，形状和数据类型与 `input` 相同。
### 约束与调用
- `input` 的最后一个维度（dim）必须是 2 的幂次，且指数范围限制在 $3 \le \log_2 (dim) \le 15$ 之间（即 dim 取值范围为 8 到 32768）。
- `input` 必须在 GPU 设备上。
- `input` 的内存布局建议是连续的，以获得最佳性能。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
DIM = 1024  # 必须是 2 的幂次，范围 [8, 32768]
BATCH = 4
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensor
input_tensor = torch.randn((BATCH, DIM), dtype=DTYPE)
# 2. 算子执行
# 执行 FHT，scale=1.0
print(f"Running fast_hadamard_transform...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
output = torch.ops.sgl_kernel.fast_hadamard_transform(
    input_tensor,
    1.0
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
# 阿达玛变换的性质：H(H(x)) = N * x
# 因此再次变换并除以 N 应该还原输入
output_inverse = torch.ops.sgl_kernel.fast_hadamard_transform(
    output,
    1.0 / DIM
)
diff = (input_tensor - output_inverse).abs().max()
print("执行成功。")
print(f"Input shape: {input_tensor.shape}")
print(f"Reconstruction Max Diff: {diff.item()}")
```
## 105. fast_hadamard_transform_12N
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fast_hadamard_transform_12N(
    input: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
pass
```
### 功能描述
fast_hadamard_transform_12N 算子是快速沃尔什-阿达玛变换（Fast Walsh-Hadamard Transform, FWHT）的一个特定变体，专门针对维度为 $12 \times 2^k$ 的输入张量进行了优化。
该算子利用分治算法和专用 CUDA 内核，将计算复杂度从 $O (N^2)$ 降低到 $O (N \log N)$。与标准的 FHT 不同，此算子专门处理非 2 的纯幂次维度，将其分解为 12 的倍数与 2 的幂次的组合，适用于某些特定的模型架构（如 Hyena Hierarchy 或特定的注意力机制），这些架构的隐藏层维度通常是 12 的倍数（例如 768, 3072 等）。
- 操作逻辑：
    算子接收输入张量 input 和缩放因子 scale。
    它将输入向量视为 $12$ 个大小为 $2^k$ 的块的组合，结合了针对 12 路混合的特定逻辑和针对 $2^k$ 的标准蝴蝶运算。
    主要功能包括：
- 对输入张量的最后一个维度执行 specialized 12N 阿达玛变换。
- 支持通过 `scale` 参数对输出结果进行缩放（通常用于归一化，例如乘以 $1/N$）。
- 利用共享内存交换（Smem Exchange）和 Warp Shuffle 原语最大化内存带宽利用率。
- 支持 FP32、FP16 和 BF16 数据类型。
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，形状通常为 `[batch_size, dim]` 或 `[..., dim]`。数据类型支持 float32, float16, bfloat16。其中 `dim` 必须满足 $12 \times 2^k$ 的形式。
- **scale** (float, 入参, 可选): 输出结果的缩放因子，默认值为 1.0。
### 返回值
返回变换后的张量，形状和数据类型与 `input` 相同。
### 约束与调用
- `input` 的最后一个维度（dim）必须满足公式 $dim = 12 \times 2^k$，其中 $k$ 的取值范围为 $[2, 10]$。
- 支持的具体维度包括：48, 96, 192, 384, 768, 1536, 3072, 6144, 12288。
- 不支持超过 12288 的维度。
- `input` 必须位于 GPU 设备上。
- 内存布局要求：建议输入张量是连续存储的。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置: 必须满足 12 * 2^k
# 例如 768 = 12 * 64 (k=6)
DIM = 768 
BATCH = 4
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensor
# 形状 [BATCH, DIM]
input_tensor = torch.randn((BATCH, DIM), dtype=DTYPE)
# 2. 算子执行
print(f"Running fast_hadamard_transform_12N...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 执行正向变换
output = torch.ops.sgl_kernel.fast_hadamard_transform_12N(
    input_tensor,
    1.0
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
# 阿达玛变换性质: H(H(x)) = N * x
# 因此再次变换并乘以 1/N 应该还原输入
output_inverse = torch.ops.sgl_kernel.fast_hadamard_transform_12N(
    output,
    1.0 / DIM
)
# 计算重构误差
diff = (input_tensor - output_inverse).abs().max()
print("执行成功。")
print(f"Input shape: {input_tensor.shape}")
print(f"Reconstruction Max Diff: {diff.item()}")
```
## 106. fast_hadamard_transform_20N
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fast_hadamard_transform_20N(
    input: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
    pass
```
### 功能描述
fast_hadamard_transform_20N 算子是快速沃尔什-阿达玛变换（Fast Walsh-Hadamard Transform, FWHT）的一个特定变体，专门针对维度为 $20 \times 2^k$ 的输入张量进行了优化。
该算子利用分治算法和专用 CUDA 内核，将计算复杂度从 $O (N^2)$ 降低到 $O (N \log N)$。此算子是为了处理特定模型架构中出现的非 2 的纯幂次维度（例如 1280, 5120 等），通过将维度分解为 20 的倍数与 2 的幂次的组合来执行高效变换。
- 操作逻辑：
    算子接收输入张量 input 和缩放因子 scale。
    它将输入向量视为 $20$ 个大小为 $2^k$ 的块的组合，结合了针对 20 路混合的特定逻辑和针对 $2^k$ 的标准蝴蝶运算。
    主要功能包括：
- 对输入张量的最后一个维度执行 specialized 20N 阿达玛变换。
- 支持通过 `scale` 参数对输出结果进行缩放（通常用于归一化，例如乘以 $1/N$）。
- 利用共享内存交换（Smem Exchange）和 Warp Shuffle 原语最大化内存带宽利用率。
- 支持 FP32、FP16 和 BF16 数据类型。
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，形状通常为 `[batch_size, dim]` 或 `[..., dim]`。数据类型支持 float32, float16, bfloat16。其中 `dim` 必须满足 $20 \times 2^k$ 的形式。
- **scale** (float, 入参, 可选): 输出结果的缩放因子，默认值为 1.0。
### 返回值
返回变换后的张量，形状和数据类型与 `input` 相同。
### 约束与调用
- `input` 的最后一个维度（dim）必须满足公式 $dim = 20 \times 2^k$，其中 $k$ 的取值范围为 $[2, 10]$。
- 支持的维度范围为 80 到 20480。
- 具体支持的维度包括：80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480。
- 不支持超过 20480 的维度。
- `input` 必须位于 GPU 设备上。
- 内存布局要求：建议输入张量是连续存储的。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置: 必须满足 20 * 2^k
# 例如 1280 = 20 * 64 (k=6)
DIM = 1280
BATCH = 4
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensor
# 形状 [BATCH, DIM]
input_tensor = torch.randn((BATCH, DIM), dtype=DTYPE)
# 2. 算子执行
print(f"Running fast_hadamard_transform_20N...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 执行正向变换
output = torch.ops.sgl_kernel.fast_hadamard_transform_20N(
    input_tensor,
    1.0
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
# 阿达玛变换性质: H(H(x)) = N * x
# 因此再次变换并乘以 1/N 应该还原输入
output_inverse = torch.ops.sgl_kernel.fast_hadamard_transform_20N(
    output,
    1.0 / DIM
)
# 计算重构误差
diff = (input_tensor - output_inverse).abs().max()
print("执行成功。")
print(f"Input shape: {input_tensor.shape}")
print(f"Reconstruction Max Diff: {diff.item()}")
```
## 107. fast_hadamard_transform_28N
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fast_hadamard_transform_28N(
    input: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
    pass
```
### 功能描述
fast_hadamard_transform_28N 算子是快速沃尔什-阿达玛变换（Fast Walsh-Hadamard Transform, FWHT）的一个特定变体，专门针对维度为 $28 \times 2^k$ 的输入张量进行了优化。
该算子利用分治算法和专用 CUDA 内核，将计算复杂度从 $O (N^2)$ 降低到 $O (N \log N)$。此算子是为了处理特定模型架构中出现的非 2 的纯幂次维度（例如 1792, 3584, 7168 等），通过将维度分解为 28 的倍数与 2 的幂次的组合来执行高效变换。
- 操作逻辑：
    算子接收输入张量 input 和缩放因子 scale。
    它将输入向量视为 $28$ 个大小为 $2^k$ 的块的组合，结合了针对 28 路混合的特定逻辑和针对 $2^k$ 的标准蝴蝶运算。
**主要功能包括**：
- 对输入张量的最后一个维度执行 specialized 28N 阿达玛变换。
- 支持通过 `scale` 参数对输出结果进行缩放（通常用于归一化，例如乘以 $1/N$）。
- 利用共享内存交换（Smem Exchange）和 Warp Shuffle 原语最大化内存带宽利用率。
- 支持 FP32、FP16 和 BF16 数据类型。
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，形状通常为 `[batch_size, dim]` 或 `[..., dim]`。数据类型支持 float32, float16, bfloat16。其中 `dim` 必须满足 $28 \times 2^k$ 的形式。
- **scale** (float, 入参, 可选): 输出结果的缩放因子，默认值为 1.0。
### 返回值
返回变换后的张量，形状和数据类型与 `input` 相同。
### 约束与调用
- `input` 的最后一个维度（dim）必须满足公式 $dim = 28 \times 2^k$，其中 $k$ 的取值范围为 $[2, 10]$。
- 支持的维度范围为 112 到 28672。
- 具体支持的维度包括：112, 224, 448, 896, 1792, 3584, 7168, 14336, 28672。
- 不支持超过 28672 的维度。
- `input` 必须位于 GPU 设备上。
- 内存布局要求：建议输入张量是连续存储的。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置: 必须满足 28 * 2^k
# 例如 1792 = 28 * 64 (k=6)
DIM = 1792
BATCH = 4
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensor
# 形状 [BATCH, DIM]
input_tensor = torch.randn((BATCH, DIM), dtype=DTYPE)
# 2. 算子执行
print(f"Running fast_hadamard_transform_28N...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 执行正向变换
output = torch.ops.sgl_kernel.fast_hadamard_transform_28N(
    input_tensor,
    1.0
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
# 阿达玛变换性质: H(H(x)) = N * x
# 因此再次变换并乘以 1/N 应该还原输入
output_inverse = torch.ops.sgl_kernel.fast_hadamard_transform_28N(
    output,
    1.0 / DIM
)
# 计算重构误差
diff = (input_tensor - output_inverse).abs().max()
print("执行成功。")
print(f"Input shape: {input_tensor.shape}")
print(f"Reconstruction Max Diff: {diff.item()}")
```
## 108. fast_hadamard_transform_40N
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fast_hadamard_transform_40N(
    input: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
    pass
```
### 功能描述
fast_hadamard_transform_40N 算子用于对最后一个维度大小为 $40 \times 2^k$ 的输入张量执行快速 Hadamard 变换（Fast Hadamard Transform, FHT）。该算子是针对非 2 的幂次维度的优化实现，专门处理基础块大小为 40 的情况。
此功能主要应用于以下场景：
1. 随机投影：在大模型或机器学习算法中进行降维或特征混合。
2. FlashAttention 变体：某些 Attention 机制使用 Hadamard 变换来近似 Hessian 或旋转特征空间。
- 操作逻辑：
    算子接收输入张量 input，对最后一个维度进行 Hadamard 变换，并应用缩放因子 scale。
    数学上，Hadamard 变换 $H_N$ 定义为递归结构，该算子特别处理 $N = 40 \times 2^k$ 的情况，利用共享内存交换和特定步幅的各种 Kernel Traits 进行加速。
**主要功能包括**：
- 针对维度 $D \in \{160, 320, \dots, 40960\}$ 的高效 FHT 计算。
- 支持 float16, bfloat16 和 float32 数据类型。
- 利用 GPU 共享内存（Shared Memory）进行高效的数据交换和蝶形运算。
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，形状为 `[..., dim]`。其中最后一个维度 `dim` 必须满足 $dim = 40 \times 2^k$，且 $2 \le k \le 10$（即 dim 范围为 160 到 40960），数据类型为 float16, bfloat16 或 float32
- **scale** (float, 入参): 输出结果的缩放因子，默认为 1.0
### 返回值
返回变换后的张量，形状和数据类型与 `input` 相同
### 约束与调用
- `input` 的最后一个维度大小必须在支持的列表中：160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960
- `input` 必须位于 GPU 设备上
- 算子针对特定维度进行了模板特化优化，性能远高于通用的矩阵乘法实现
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置 (必须是 40 * 2^k, 这里取 40 * 2^2 = 160)
BATCH = 8
SEQ_LEN = 128
DIM = 160 
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensor
input_tensor = torch.randn(BATCH, SEQ_LEN, DIM, dtype=DTYPE)
scale = 1.0 / (DIM ** 0.5)
# 算子执行
print(f"Running fast_hadamard_transform_40N...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
output = torch.ops.sgl_kernel.fast_hadamard_transform_40N(
    input_tensor,
    scale
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Output sample (first 5 elements): {output[0, 0, :5]}")
```
## 110. mctlass_moe_w4a16_gemm_kernel_mnk
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def mctlass_moe_w4a16_gemm_kernel_mnk(
    num_valid_tokens: int,
    N: int,
    K: int,
    group: int
) -> int
```
### 功能描述
该算子用于计算 **W4A16**（权重 4bit，激活值 16bit）混合精度 MoE GEMM 操作所需的 Kernel M 维度大小（`kernel_size.m`）。它并不执行实际的矩阵乘法运算，而是根据输入的问题规模（Token 数、矩阵维度、专家组数），在 `mctlass` 库中查找匹配的 GEMM Kernel 配置，并返回该配置对应的 M 维度大小，用于后续的资源分配或 Grid 设置。
- 计算公式：
  先对 4bit 量化权重执行反量化，再执行矩阵乘法（仅计算有效 token）：
  $$
  \begin{align*}
  &b_{\text{dequant}}[k, m] = \left( b[k, m] - \text{zp}_b[k, m] \right) \times \text{scale}_b[k, m] \\
  &\text{dst}[i, m] = \sum_{k=0}^{K-1} \text{a}[i, k] \times b_{\text{dequant}}[k, m] \quad (0 \leq i < \text{num\_valid\_tokens})
  \end{align*}
  $$
  若$i \geq \text{num\_valid\_tokens}$（填充行），则$\text{dst}[i, :] = 0$。
  其中各张量的线性索引与二维位置的映射关系为：
  - 激活张量$\text{a}$：
    $$
    n_a = i \times K + k
    $$
  - 权重张量$b$：
    $$
    n_b = k \times M + m
    $$
  - 输出张量$\text{dst}$：
    $$
    n_{\text{dst}} = i \times M + m
    $$
    各变量的维度与含义：
  - $\text{a} \in \mathbb{T}^{N \times K}$：16bit 激活输入张量，数据类型为$\mathbb{T}$（如 fp16），有效行范围$0 \leq i < \text{num\_valid\_tokens}$
  - $b \in \mathbb{U}^{K \times M}$：4bit 量化的权重张量，数据类型为$\mathbb{U}$（如 uint4）
  - $\text{zp}_b \in \mathbb{U}^{K \times M}$：权重量化的零点张量，数据类型与$b$一致
  - $\text{scale}_b \in \mathbb{R}^{K \times M}$：权重量化的缩放因子张量
  - $\text{dst} \in \mathbb{D}^{N \times M}$：输出张量，存储矩阵乘结果，数据类型与$\text{a}$一致
  - $\text{num\_valid\_tokens}$：实际参与计算的有效激活行数（$0 \leq \text{num\_valid\_tokens} \leq N$）
  - $N$：激活张量的行数（含填充，用于内存对齐）
  - $K$：激活与权重的输入特征维度
  - $M$：权重与输出的输出特征维度（由业务场景或 group 参数推导）
  - $\text{group}$：权重的分组维度（用于分组 GEMM 并行优化，将$K$划分为$\text{group}$个子维度）
  - $i$：激活/输出的行索引，范围$0 \leq i < N$
  - $k$：输入特征维度的索引，范围$0 \leq k < K$
  - $m$：输出特征维度的索引，范围$0 \leq m < M$
### 参数说明
- **num_valid_tokens** (int, 入参): 有效 Token 的数量（对应 GEMM 中的 M 维度）
- **N** (int, 入参): 权重矩阵的输出维度 N
- **K** (int, 入参): 权重矩阵的输入维度 K
- **group** (int, 入参): 组数量（对应 BatchedGemm 中的 Batch 维度，通常指 Expert 数量）
### 返回值
- **kernel_m** (int): 返回计算得到的 Kernel M 维度大小。如果未找到支持的 Kernel，代码会在 C++ 层打印错误信息，但返回值依赖于 `kernel_size.m()` 的默认状态
### 约束与调用
- 数据类型配置：底层固定配置为 `ElementA=maca_bfloat16` (BF16), `ElementB=uint8_t` (4-bit 权重打包), `ElementC=maca_bfloat16` (BF16)
- 布局约束：固定配置为 LayoutA=RowMajor, LayoutB=ColumnMajor, LayoutC=RowMajor
- 内部参数：算子内部硬编码了 `pack_factor = 2` 和 `group_size = 64`
- 错误处理：输入的参数组合需要在 `mctlass` 中找到对应的 Kernel 实现
### 调用示例
```python
import torch
import mcoplib.sgl_moe_fused_w4a16
# 1. 核心参数设置 (对应 C++ 签名中的四个 int64_t 参数)
# int64_t num_valid_tokens, int64_t N, int64_t K, int64_t group
num_valid_tokens = 1024   # 有效 token 数量
N = 4096                  # 输出维度 N
K = 2048                  # 输入维度 K
group = 64                # 组大小 (Group Size)
# 2. 调用算子
result = mcoplib.sgl_moe_fused_w4a16.mctlass_moe_w4a16_gemm_kernel_mnk(
    num_valid_tokens,
    N,
    K,
    group
)
# 3. 输出结果
print("mctlass_moe_w4a16_gemm_kernel_mnk called successfully")
print(f"Input Params -> Tokens: {num_valid_tokens}, N: {N}, K: {K}, Group: {group}")
print(f"Output Result (Kernel m size): {result}")
```
## 111. lightning_attention_decode
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def lightning_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_kv: torch.Tensor,
    output: torch.Tensor,
    slope_h: float
) -> None
```
### 功能描述
实现 Transformer 解码阶段的 Lightning Attention 计算，通过共享内存优化内存访问效率：
1. 将查询（q）、键（k）、值（v）及历史键值缓存（past_kv）加载到共享内存；
2. 计算注意力权重的缩放因子（ratio）；
3. 更新历史键值缓存（past_kv）；
4. 完成注意力加权求和并将结果写入 output 张量。
该算子通过内存访问融合与共享内存复用，提升了解码阶段注意力计算的性能。
- 计算公式：
$$
  \begin{align*}
  &\text{K}_{\text{total}} = \text{concat}([\text{past\_kv}_k, K], 2) \\
  &\text{V}_{\text{total}} = \text{concat}([\text{past\_kv}_v, V], 2) \\
  &\text{score} = \frac{Q \cdot \text{K}_{\text{total}}^T}{\sqrt{d_k}} \cdot \frac{1}{1 + \exp(-\text{slope\_h} \cdot \text{len}(\text{K}_{\text{total}}))} \\
  &\text{attn\_weight} = \text{softmax}(\text{score}) \\
  &\text{output} = \text{attn\_weight} \cdot \text{V}_{\text{total}} \\
  &\text{past\_kv} = \text{concat}([\text{past\_kv}, \text{concat}([K, V], 3)], 2)
  \end{align*}
$$
  其中：
  - $Q \in \mathbb{R}^{\text{bs} \times H \times 1 \times d_k}$ 是解码阶段单步查询张量，dtype∈{float16, bfloat16, float32}
  - $K \in \mathbb{R}^{\text{bs} \times H_k \times 1 \times d_k}$ 是解码阶段单步键张量，dtype∈{float16, bfloat16, float32}
  - $V \in \mathbb{R}^{\text{bs} \times H_k \times 1 \times d_v}$ 是解码阶段单步值张量，dtype∈{float16, bfloat16, float32}，通常$d_v = d_k$
  - $\text{past\_kv} = (\text{past\_kv}_k, \text{past\_kv}_v) \in \mathbb{R}^{\text{bs} \times H_k \times L_{\text{past}} \times (d_k + d_v)}$ 是历史键值缓存张量，dtype∈{float16, bfloat16, float32}，入参/出参
  - $\text{output} \in \mathbb{R}^{\text{bs} \times H \times 1 \times d_v}$ 是注意力计算输出张量，dtype∈{float16, bfloat16, float32}
  - $\text{slope\_h} \in \mathbb{R}$ 是注意力权重缩放斜率参数
  - $\text{concat}([T_1,T_2], dim)$ 表示沿$dim$维度拼接张量$T_1$和$T_2$
  - $\text{softmax}(\cdot)$ 表示沿序列维度（第 2 维）的 softmax 归一化操作
  - $\text{len}(\cdot)$ 表示张量沿序列维度的长度计算，$\text{len}(\text{K}_{\text{total}}) = L_{\text{past}} + 1$
  - $\text{bs}$ 是 batch 大小，$H$ 是查询注意力头数，$H_k$ 是键值注意力头数（GQA/MQA 配置下$H_k \leq H$）
  - $d_k$ 是键/查询头维度（head size），$d_v$ 是值头维度，$L_{\text{past}}$ 是历史序列长度
  - $\cdot^T$ 表示张量在最后两个维度的转置操作，$\cdot$ 表示矩阵乘法（最后两个维度）
  - $\exp(\cdot)$ 表示逐元素指数运算，$\sqrt{\cdot}$ 表示开平方运算
  - $+$ 表示逐元素加法，$\cdot$（缩放因子与分数的相乘）表示逐元素乘法
  - $[:,a:b]$ 表示张量沿指定维度取$a$到$b$的切片（CUDA 内核中共享内存加载/存储时的索引操作）
### 参数说明
- **q** (torch.Tensor, 入参): 注意力查询张量，需位于 CUDA 设备且为连续张量
- **k** (torch.Tensor, 入参): 注意力键张量，需位于 CUDA 设备且为连续张量
- **v** (torch.Tensor, 入参): 注意力值张量，需位于 CUDA 设备且为连续张量
- **past_kv** (torch.Tensor, 入参/出参): 历史键值对缓存，计算中会更新该缓存，需位于 CUDA 设备且为连续张量
- **output** (torch.Tensor, 出参): 注意力计算输出张量，结果直接写入此处，需位于 CUDA 设备且为连续张量
- **slope_h** (float, 入参): 用于计算注意力权重缩放因子的参数
### 返回值
无返回值，变换结果直接写入输出张量中
### 约束与调用
- 所有张量必须部署在 CUDA 设备上
- 所有张量需为连续（contiguous）张量，否则会触发合法性检查失败
- 支持的数据类型：float16、bfloat16、float32
- 线程块大小固定为`THREADS_PER_BLOCK`（代码中定义为 128），网格大小需根据张量的 batch_size、维度动态计算
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "lightning_attention_decode仅支持CUDA设备"
# 核心参数
bs = 2               
H = 16               
Hk = 8               
dk = 64              
dv = 64              
L_past = 10         
# 1. 构造输入张量
q = torch.randn(
    bs, H, 1, dk,
    dtype=torch.float32, device=device, requires_grad=False
)
k = torch.randn(
    bs, Hk, 1, dk,
    dtype=torch.float32, device=device, requires_grad=False
)
v = torch.randn(
    bs, Hk, 1, dv,
    dtype=torch.float32, device=device, requires_grad=False
)
past_kv = torch.randn(
    bs, Hk, L_past, dk + dv,
    dtype=torch.float32, device=device, requires_grad=False
)
output = torch.empty(
    bs, H, 1, dv,
    dtype=torch.float32, device=device, requires_grad=False
)
slope = torch.tensor(0.1, dtype=torch.float32, device=device)
new_kv = torch.empty(
    bs, Hk, L_past + 1, dk + dv,
    dtype=torch.float32, device=device, requires_grad=False
)
# 调用算子
torch.ops.sgl_kernel.lightning_attention_decode(
    q=q,
    k=k,
    v=v,
    past_kv=past_kv,
    slope=slope,
    output=output,
    new_kv=new_kv
)
# 输出结果
print("lightning_attention_decode computation completed")
print(f"Output shape: {output.shape}")
print(f"Updated new_kv shape: {new_kv.shape}")
```
## 117. meta_size
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def meta_size() -> int:
    pass
```
### 功能描述
meta_size 算子用于获取 Custom Allreduce 机制所需的元数据（Metadata）缓冲区的大小（以字节为单位）。在分布式推理场景中，自定义 All-Reduce 内核需要一块共享内存区域来进行秩（Rank）之间的同步信号交换和状态管理。
此功能主要应用于以下场景：
1. 显存分配：在初始化通信器之前，计算需要分配的 IPC 共享内存的总大小（通常为 `meta_size () + max_buffer_size`）。
2. 偏移量计算：确定数据缓冲区在共享内存块中的起始偏移量。
- 操作逻辑：
    算子调用底层 C++ 接口，返回预定义的元数据结构体所占用的字节数。
    该数值是一个固定常量，取决于底层通信协议的设计。
**主要功能包括**：
- 返回用于同步和元数据的显存大小要求。
- 辅助上层应用进行正确的显存切分和管理。
### 参数说明
无参数
### 返回值
返回一个整数（int），表示元数据缓冲区所需的字节大小
### 约束与调用
- 该算子返回的是静态配置值，不依赖于当前的输入数据
- 通常在 `CustomAllreduce` 初始化阶段调用
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
# 无需特定配置，直接调用
# 算子执行
print(f"Running meta_size...")
size_in_bytes = torch.ops.sgl_kernel.meta_size()
# 验证结果
print("执行成功。")
print(f"Meta Size required: {size_in_bytes} bytes")
```
## 119. init_custom_ar
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def init_custom_ar(
    meta_ptrs: List[int],
    rank_data: torch.Tensor,
    rank: int,
    full_nvlink: bool
) -> int:
    pass
```
### 功能描述
init_custom_ar 算子用于初始化自定义 All-Reduce（Custom Allreduce）通信器的底层 C++ 状态对象。
在分布式推理初始化阶段，该算子负责构建用于同步和通信的上下文环境。它接收所有 Rank 的元数据（Meta）缓冲区地址，从而允许内核通过 P2P（Peer-to-Peer）机制在不同 GPU 之间直接读写同步信号，实现低延迟的协同工作。
**主要功能包括**：
- 创建并初始化 `CustomAllReduce` C++ 对象。
- 注册用于跨设备同步的共享内存地址（Meta Pointers）。
- 配置拓扑信息（如 Rank ID 和 NVLink 状态）。
- 返回状态对象的指针（Handle），供后续的注册和通信操作使用。
### 参数说明
- **meta_ptrs** (List[int], 入参): 包含通信组中所有 Rank 的元数据缓冲区（Meta Buffer）物理地址的列表。
    - 列表长度必须等于 `world_size`。
    - 第 `i` 个元素必须是 Rank `i` 的 Meta Buffer 在当前设备上可见的内存地址（通常通过 IPC 打开句柄获得）。
- **rank_data** (torch.Tensor, 入参): 用于存储 Rank 相关辅助数据的张量，通常作为临时缓冲区或用于存储拓扑信息。
- **rank** (int, 入参): 当前进程在通信组中的 Rank ID。
- **full_nvlink** (bool, 入参): 标志位，指示当前环境是否支持全互联的 NVLink/高速互联拓扑。内核可能会根据此标志选择不同的通信策略。
### 返回值
返回一个整数（int），代表底层 C++ `CustomAllReduce` 状态对象的内存地址（指针）。该值需被妥善保存，并在调用 `register_buffer`、`all_reduce` 和 `dispose` 时传入。
### 约束与调用
- `meta_ptrs` 中的地址必须是有效的，且当前设备必须具有访问这些地址的权限（P2P Access Enabled）。
- 必须在分布式环境初始化（如 `init_process_group`）之后调用。
- 返回的指针必须在结束时通过 `dispose` 释放，否则会导致内存泄漏。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
rank = 0
world_size = 2
full_nvlink = True
device = "cuda"
# 1. 准备 rank_data 缓冲区
rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
# 2. 准备 meta_ptrs
# 在实际场景中，这通常涉及以下步骤：
# a. 各 Rank 分配 meta buffer (torch.empty)
# b. 获取 IPC Handle (cudaIpcGetMemHandle)
# c. 通过 all_gather 交换 Handle
# d. 通过 cudaIpcOpenMemHandle 打开 Handle 获取地址
# 这里为了演示，假设我们已经获取了两个有效的设备指针地址
fake_meta_ptr_rank0 = 140737488355328
fake_meta_ptr_rank1 = 140737488355400
meta_ptrs = [fake_meta_ptr_rank0, fake_meta_ptr_rank1]
# 算子执行
print(f"Running init_custom_ar...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 初始化并获取状态指针
state_ptr = torch.ops.sgl_kernel.init_custom_ar(
    meta_ptrs,
    rank_data,
    rank,
    full_nvlink
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
print(f"初始化完成，State Pointer: {state_ptr}")
# 后续需调用 dispose(state_ptr) 释放资源
```
## 120. all_reduce
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    buffer_ptr: int,
    buffer_size: int
) -> None:
    pass
```
### 功能描述
all_reduce 算子用于在 Custom Allreduce 上下文中执行 out-of-place 的规约操作。该算子通常在 Eager 模式下或缓冲区尚未注册时使用。它利用初始化好的 Custom Allreduce 状态，将本地输入 inp 规约到输出 out，并在此过程中使用临时缓冲区 buffer 进行数据中转。
此功能主要应用于以下场景：
1. 未注册张量的规约：当输入张量的内存地址未在初始化阶段通过 `init_custom_ar` 或 `register_buffer` 注册时，需要显式传递一个已注册的临时缓冲区 `buffer` 来协助通信。
2. Eager 模式下的动态规约：在动态图执行模式下，随时对任意张量进行 All-Reduce 操作。
- 操作逻辑：
    算子首先将输入数据 inp 拷贝到已注册的共享缓冲区 buffer 中。
    内核利用 P2P 机制直接访问其他 Rank 的 buffer，执行规约计算（如 Sum）。
    最终结果写回到本地输出 out。
**主要功能包括**：
- 支持未注册张量的高效 All-Reduce。
- 自动处理数据搬运和同步。
- 规约操作默认为 Sum。
### 参数说明
- **fa** (int, 入参): Custom Allreduce 的状态指针（Flat Address），由 `init_custom_ar` 返回
- **inp** (torch.Tensor, 入参): 需要进行规约的本地输入张量
- **out** (torch.Tensor, 出参): 存储规约结果的输出张量，形状和数据类型需与 `inp` 一致
- **buffer_ptr** (int, 入参): 当前 Rank 上已注册的临时缓冲区的内存地址指针。该缓冲区的容量必须能够容纳 `inp` 的数据
- **buffer_size** (int, 入参): 临时缓冲区的大小（以字节为单位），用于边界检查
### 返回值
无返回值，规约结果直接写入 `out` 张量中
### 约束与调用
- `fa` 必须是有效的状态指针
- `inp` 和 `out` 必须在同一设备上，且数据类型一致
- `buffer_ptr` 指向的内存区域必须足够大（`>= inp.numel () * inp.element_size ()`），且必须已经通过 `register_buffer` 注册到 `fa` 关联的通信器中
- 参与通信的所有 Rank 必须同时调用该算子，且输入张量形状一致
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 简单测试，设置GPU数量为2
world_size = 2 
rank = 0       # 当前进程依然作为 0 号 rank
full_nvlink = True 
# C++ 会检查 fake_ipc_ptrs.size()，现在它等于 2，满足 % 2 == 0
fake_ipc_ptrs = [0] * world_size 
device = "cuda"
rank_data = torch.zeros(1024, dtype=torch.int64, device=device) 
print("Initializing CustomAllReduce object...")
try:
    rank_state_ptr = torch.ops.sgl_kernel.init_custom_ar(
        fake_ipc_ptrs, 
        rank_data, 
        rank, 
        full_nvlink
    )
    print(f"Got valid object pointer: {rank_state_ptr}")
except Exception as e:
    print(f"Initialization failed: {e}")
    exit(1)
inp = torch.randn(1024, 1024, dtype=torch.float16, device=device)
out = torch.empty_like(inp)
required_size = inp.numel() * inp.element_size()
temp_buffer = torch.empty(required_size, dtype=torch.uint8, device=device)
print("Running all_reduce...")
try:
    torch.ops.sgl_kernel.all_reduce(
        rank_state_ptr,
        inp,
        out,
        temp_buffer.data_ptr(),
        temp_buffer.numel()
    )
    print("All-Reduce computation completed。")
except Exception as e:
    # 单卡无法验证完整功能，等不到 rank 1 的信号
    print(f"All-Reduce called but no more GPU")
# 清理
if hasattr(torch.ops.sgl_kernel, "dispose"):
    torch.ops.sgl_kernel.dispose(rank_state_ptr)
```
## 121. lightning_attention_decode
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def lightning_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_kv: torch.Tensor,
    output: torch.Tensor,
    slope_h: float
) -> None
```
### 功能描述
实现 Transformer 解码阶段的 Lightning Attention 计算，通过共享内存优化内存访问效率：
1. 将查询（q）、键（k）、值（v）及历史键值缓存（past_kv）加载到共享内存；
2. 计算注意力权重的缩放因子（ratio）；
3. 更新历史键值缓存（past_kv）；
4. 完成注意力加权求和并将结果写入 output 张量。
该算子通过内存访问融合与共享内存复用，提升了解码阶段注意力计算的性能。
- 计算公式：
$$
  \begin{align*}
  &\text{K}_{\text{total}} = \text{concat}([\text{past\_kv}_k, K], 2) \\
  &\text{V}_{\text{total}} = \text{concat}([\text{past\_kv}_v, V], 2) \\
  &\text{score} = \frac{Q \cdot \text{K}_{\text{total}}^T}{\sqrt{d_k}} \cdot \frac{1}{1 + \exp(-\text{slope\_h} \cdot \text{len}(\text{K}_{\text{total}}))} \\
  &\text{attn\_weight} = \text{softmax}(\text{score}) \\
  &\text{output} = \text{attn\_weight} \cdot \text{V}_{\text{total}} \\
  &\text{past\_kv} = \text{concat}([\text{past\_kv}, \text{concat}([K, V], 3)], 2)
  \end{align*}
$$
  其中：
  - $Q \in \mathbb{R}^{\text{bs} \times H \times 1 \times d_k}$ 是解码阶段单步查询张量，dtype∈{float16, bfloat16, float32}
  - $K \in \mathbb{R}^{\text{bs} \times H_k \times 1 \times d_k}$ 是解码阶段单步键张量，dtype∈{float16, bfloat16, float32}
  - $V \in \mathbb{R}^{\text{bs} \times H_k \times 1 \times d_v}$ 是解码阶段单步值张量，dtype∈{float16, bfloat16, float32}，通常$d_v = d_k$
  - $\text{past\_kv} = (\text{past\_kv}_k, \text{past\_kv}_v) \in \mathbb{R}^{\text{bs} \times H_k \times L_{\text{past}} \times (d_k + d_v)}$ 是历史键值缓存张量，dtype∈{float16, bfloat16, float32}，入参/出参
  - $\text{output} \in \mathbb{R}^{\text{bs} \times H \times 1 \times d_v}$ 是注意力计算输出张量，dtype∈{float16, bfloat16, float32}
  - $\text{slope\_h} \in \mathbb{R}$ 是注意力权重缩放斜率参数
  - $\text{concat}([T_1,T_2], dim)$ 表示沿$dim$维度拼接张量$T_1$和$T_2$
  - $\text{softmax}(\cdot)$ 表示沿序列维度（第 2 维）的 softmax 归一化操作
  - $\text{len}(\cdot)$ 表示张量沿序列维度的长度计算，$\text{len}(\text{K}_{\text{total}}) = L_{\text{past}} + 1$
  - $\text{bs}$ 是 batch 大小，$H$ 是查询注意力头数，$H_k$ 是键值注意力头数（GQA/MQA 配置下$H_k \leq H$）
  - $d_k$ 是键/查询头维度（head size），$d_v$ 是值头维度，$L_{\text{past}}$ 是历史序列长度
  - $\cdot^T$ 表示张量在最后两个维度的转置操作，$\cdot$ 表示矩阵乘法（最后两个维度）
  - $\exp(\cdot)$ 表示逐元素指数运算，$\sqrt{\cdot}$ 表示开平方运算
  - $+$ 表示逐元素加法，$\cdot$（缩放因子与分数的相乘）表示逐元素乘法
  - $[:,a:b]$ 表示张量沿指定维度取$a$到$b$的切片（CUDA 内核中共享内存加载/存储时的索引操作）
### 参数说明
- **q** (torch.Tensor, 入参): 注意力查询张量，需位于 CUDA 设备且为连续张量
- **k** (torch.Tensor, 入参): 注意力键张量，需位于 CUDA 设备且为连续张量
- **v** (torch.Tensor, 入参): 注意力值张量，需位于 CUDA 设备且为连续张量
- **past_kv** (torch.Tensor, 入参/出参): 历史键值对缓存，计算中会更新该缓存，需位于 CUDA 设备且为连续张量
- **output** (torch.Tensor, 出参): 注意力计算输出张量，结果直接写入此处，需位于 CUDA 设备且为连续张量
- **slope_h** (float, 入参): 用于计算注意力权重缩放因子的参数
### 返回值
无返回值，变换结果直接写入输出张量中
### 约束与调用
- 所有张量必须部署在 CUDA 设备上
- 所有张量需为连续（contiguous）张量，否则会触发合法性检查失败
- 支持的数据类型：float16、bfloat16、float32
- 线程块大小固定为`THREADS_PER_BLOCK`（代码中定义为 128），网格大小需根据张量的 batch_size、维度动态计算
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "lightning_attention_decode仅支持CUDA设备"
# 核心参数
bs = 2               
H = 16               
Hk = 8               
dk = 64              
dv = 64              
L_past = 10         
# 1. 构造输入张量
q = torch.randn(
    bs, H, 1, dk,
    dtype=torch.float32, device=device, requires_grad=False
)
k = torch.randn(
    bs, Hk, 1, dk,
    dtype=torch.float32, device=device, requires_grad=False
)
v = torch.randn(
    bs, Hk, 1, dv,
    dtype=torch.float32, device=device, requires_grad=False
)
past_kv = torch.randn(
    bs, Hk, L_past, dk + dv,
    dtype=torch.float32, device=device, requires_grad=False
)
output = torch.empty(
    bs, H, 1, dv,
    dtype=torch.float32, device=device, requires_grad=False
)
slope = torch.tensor(0.1, dtype=torch.float32, device=device)
new_kv = torch.empty(
    bs, Hk, L_past + 1, dk + dv,
    dtype=torch.float32, device=device, requires_grad=False
)
# 调用算子
torch.ops.sgl_kernel.lightning_attention_decode(
    q=q,
    k=k,
    v=v,
    past_kv=past_kv,
    slope=slope,
    output=output,
    new_kv=new_kv
)
# 输出结果
print("lightning_attention_decode computation completed")
print(f"Output shape: {output.shape}")
print(f"Updated new_kv shape: {new_kv.shape}")
```
## 122. merge_state
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    pass
```
### 功能描述
merge_state 算子用于合并两个注意力计算阶段的中间状态（State），通常用于 FlashAttention 机制中的分块并行计算或解码阶段的状态更新。算子接收两组部分计算得到的 Output (Values) 和 LogSumExp (LSE) 数据，基于 LSE 进行加权归一化合并，生成最终的 Output 和更新后的 LSE。
操作逻辑：
算子接收前缀状态（Prefix）和后缀状态（Suffix），对每个 token 的每个 head 执行以下计算：
1. 计算全局最大 LSE：$max\_lse = \max (p\_lse, s\_lse)$
2. 计算归一化分母：$out\_se = \exp (p\_lse - max\_lse) + \exp (s\_lse - max\_lse)$
3. 更新 LSE：$out\_lse = \log (out\_se) + max\_lse$
4. 计算加权系数并合并 Output：
    $$
    out = p\_out \cdot \frac{\exp(p\_lse - max\_lse)}{out\_se} + s\_out \cdot \frac{\exp(s\_lse - max\_lse)}{out\_se}
    $$
**主要功能包括：**
- 基于 LogSumExp 的数值稳定性加权合并。
- 支持 In-place 操作（通过传入 output 张量）。
- 自动处理 padding 和 head 维度的对齐。
### 参数说明
- **prefix_output** (torch.Tensor, 入参): 前缀部分的 Output 张量，形状为 `[num_tokens, num_heads, head_size]`，数据类型推荐使用 bfloat16（根据报错信息推断，Float/Half 可能未注册）。
- **prefix_lse** (torch.Tensor, 入参): 前缀部分的 LogSumExp 张量，形状为 `[num_tokens, num_heads]`，数据类型通常为 float32。
- **suffix_output** (torch.Tensor, 入参): 后缀部分的 Output 张量，形状和数据类型与 `prefix_output` 相同。
- **suffix_lse** (torch.Tensor, 入参): 后缀部分的 LogSumExp 张量，形状和数据类型与 `prefix_lse` 相同。
- **output** (Optional[torch.Tensor], 出参): 用于存储合并结果的 Output 张量。如果为 None，则自动创建。
- **output_lse** (Optional[torch.Tensor], 出参): 用于存储合并结果的 LogSumExp 张量。如果为 None，则自动创建。
### 返回值
返回一个元组 `(output, output_lse)`：
- **output**: 合并后的 Output 张量。
- **output_lse**: 合并后的 LogSumExp 张量。
### 约束与调用
- 输入张量 `prefix_output` 和 `suffix_output` 的形状必须完全一致。
- `prefix_lse` 和 `suffix_lse` 的形状必须匹配 `[num_tokens, num_heads]`。
- 建议 Value 张量使用 `bfloat16`，LSE 张量使用 `float32`。
- 算子支持 CUDA 设备。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
# 修复：Half 和 Float 均 dispatch 失败，切换为 bfloat16
DTYPE_VAL = torch.bfloat16
DTYPE_LSE = torch.float32
# 维度设置
NUM_TOKENS = 4
NUM_HEADS = 8
HEAD_SIZE = 128
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensors
# 模拟前缀状态
prefix_out = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE_VAL)
prefix_lse = torch.randn(NUM_TOKENS, NUM_HEADS, dtype=DTYPE_LSE)
# 模拟后缀状态
suffix_out = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE_VAL)
suffix_lse = torch.randn(NUM_TOKENS, NUM_HEADS, dtype=DTYPE_LSE)
# 2. 准备输出 Tensors (可选)
output = torch.empty_like(prefix_out)
output_lse = torch.empty_like(prefix_lse)
# 算子执行
print(f"Running merge_state with {DTYPE_VAL}...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用 ops
torch.ops.sgl_kernel.merge_state(
    prefix_out,
    prefix_lse,
    suffix_out,
    suffix_lse,
    output,
    output_lse
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
print("执行成功。")
print(f"Output Shape: {output.shape}")
print(f"Output LSE Shape: {output_lse.shape}")
print(f"Output Mean: {output.float().mean().item()}")
```
## 123. merge_state_v2
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def merge_state_v2(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    pass
```
### 功能描述
merge_state_v2 算子用于合并两个独立注意力计算阶段产生的中间状态（Value 和 LogSumExp）。该算子通常应用于 FlashAttention 的分块解码（Split-K Decoding）、投机采样（Speculative Decoding）中的 Cascade Attention 验证阶段，或者 chunked prefill 场景。它根据两组 LogSumExp (LSE) 的值计算归一化权重，将两组部分的 Output 合并为最终的 Output，并更新全局 LSE。
操作逻辑：
算子接收两组状态 $(V_a, S_a)$ 和 $(V_b, S_b)$，其中 $V$ 代表 Output，$S$ 代表 LSE。对每个 Head 的每个 Token 执行以下计算：
1. 计算最大 LSE：$m = \max (S_a, S_b)$
2. 计算归一化因子：$l = \exp (S_a - m) + \exp (S_b - m)$
3. 计算新的 LSE：$S_{new} = m + \log (l)$
4. 合并 Output：
    $$
    V_{new} = \frac{V_a \cdot \exp(S_a - m) + V_b \cdot \exp(S_b - m)}{l}
    $$
**主要功能包括**：
- 高效合并 FlashAttention 的中间结果。
- 保证数值稳定性的 LogSumExp 更新。
- 支持 In-place 操作以减少显存分配。
### 参数说明
- **prefix_output** (torch.Tensor, 入参): 第一部分（前缀）的 Output 张量，形状为 `[num_tokens, num_heads, head_size]`，数据类型支持 float16 或 bfloat16
- **prefix_lse** (torch.Tensor, 入参): 第一部分（前缀）的 LogSumExp 张量，形状为 `[num_tokens, num_heads]`，数据类型必须为 float32
- **suffix_output** (torch.Tensor, 入参): 第二部分（后缀）的 Output 张量，形状和数据类型与 `prefix_output` 相同
- **suffix_lse** (torch.Tensor, 入参): 第二部分（后缀）的 LogSumExp 张量，形状和数据类型与 `prefix_lse` 相同
- **output** (Optional[torch.Tensor], 出参): 用于存储合并结果的 Output 张量。如果为 None，算子将创建新张量或复用输入（取决于具体实现逻辑，通常建议显式传入或使用返回值）
- **output_lse** (Optional[torch.Tensor], 出参): 用于存储合并结果的 LSE 张量。如果为 None，算子将创建新张量
### 返回值
返回一个元组 `(output, output_lse)`，包含合并后的 Output 张量和 LogSumExp 张量
### 约束与调用
- 所有输入张量必须在 CUDA 设备上
- `prefix_output` 和 `suffix_output` 的形状必须完全一致
- `prefix_lse` 和 `suffix_lse` 的形状必须完全一致，且匹配 Output 的前两维
- LSE 张量必须连续（Contiguous）
- 推荐使用 `bfloat16` 作为 Value 的数据类型以获得最佳性能和精度平衡
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE_VAL = torch.bfloat16
DTYPE_LSE = torch.float32
# 维度设置
NUM_TOKENS = 64
NUM_HEADS = 8
HEAD_SIZE = 128
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensors
# 模拟前缀 Attention 结果
prefix_out = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE_VAL)
prefix_lse = torch.randn(NUM_TOKENS, NUM_HEADS, dtype=DTYPE_LSE)
# 模拟后缀 Attention 结果
suffix_out = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE_VAL)
suffix_lse = torch.randn(NUM_TOKENS, NUM_HEADS, dtype=DTYPE_LSE)
# 2. 准备输出 Tensors
output = torch.empty_like(prefix_out)
output_lse = torch.empty_like(prefix_lse)
# 算子执行
print(f"Running merge_state_v2...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.merge_state_v2(
    prefix_out,
    prefix_lse,
    suffix_out,
    suffix_lse,
    output,
    output_lse
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
print("执行成功。")
print(f"Output Mean: {output.float().mean().item()}")
print(f"Output LSE Mean: {output_lse.mean().item()}")
```
## 124. fused_add_rmsnorm
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    enable_pdl: bool = False
) -> None:
    pass
```
### 功能描述
fused_add_rmsnorm 算子将残差连接（Residual Add）与 RMSNorm（Root Mean Square Normalization）操作融合为一个高效的内核。该操作通常用于 Transformer 架构的层级结构中，在每个子层（如 Attention 或 Feed Forward）之后应用。
操作逻辑：
1. 残差更新：首先将输入 input 与 residual 相加，结果直接原位更新到 residual 张量中。
    $$
    residual \leftarrow input + residual
    $$
2. RMSNorm：然后对更新后的 `residual` 进行 RMSNorm 操作。
    - 计算均方根：
        $$
        RMS(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
        $$
    - 归一化并缩放：
        $$
        y = \frac{x}{RMS(x)} \cdot weight
        $$
    - 最终结果原位更新到 `input` 张量中。
**主要功能包括：**
- 融合 Add 和 RMSNorm 以减少显存访问次数。
- 支持 In-place 更新 residual，节省显存。
- 支持通过 `enable_pdl` 参数开启特定硬件优化。
### 参数说明
- **input** (torch.Tensor, 入参/出参): 输入张量，形状通常为 `[batch_size, seq_len, hidden_size]`。操作后存储 RMSNorm 的结果。数据类型推荐使用 bfloat16。
- **residual** (torch.Tensor, 入参/出参): 残差张量，形状与 `input` 相同。操作后存储 `input + residual` 的结果。数据类型与 `input` 一致。
- **weight** (torch.Tensor, 入参): RMSNorm 的权重参数，形状为 `[hidden_size]`。数据类型通常与 `input` 一致。
- **epsilon** (float, 入参): 数值稳定性常数，通常为 1e-6。
- **enable_pdl** (bool, 入参): 是否开启 PDL 优化，默认为 False。
### 返回值
无返回值（None）。结果直接写入 `input` 和 `residual` 张量。
### 约束与调用
- `input` 和 `residual` 的形状必须完全一致。
- `weight` 的大小必须匹配 `input` 的最后一维。
- 所有输入张量必须在 CUDA 设备上。
- 数据类型通常要求一致（如均为 bf16）。
- 建议输入张量是连续存储的。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16  # 使用 bfloat16 避免 Half dispatch 错误
HIDDEN_SIZE = 4096
NUM_TOKENS = 128
EPS = 1e-6
ENABLE_PDL = False
# 初始化
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 创建 Tensor
input_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=DTYPE)
residual_tensor = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=DTYPE)
weight = torch.ones(HIDDEN_SIZE, dtype=DTYPE)
# 备份
input_ref = input_tensor.clone()
residual_ref = residual_tensor.clone()
# 执行算子
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.fused_add_rmsnorm(
    input_tensor,
    residual_tensor,
    weight,
    EPS,
    ENABLE_PDL
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
expected_residual = input_ref + residual_ref
diff_residual = (residual_tensor - expected_residual).abs().max()
# 手动计算 RMSNorm (转 float32 计算)
input_f32 = input_ref.to(torch.float32)
residual_f32 = residual_ref.to(torch.float32)
weight_f32 = weight.to(torch.float32)
expected_residual_f32 = input_f32 + residual_f32
rstd = torch.rsqrt(expected_residual_f32.pow(2).mean(-1, keepdim=True) + EPS)
expected_output = (expected_residual_f32 * rstd * weight_f32).to(DTYPE)
diff_output = (input_tensor - expected_output).abs().max()
print("执行成功。")
print(f"Residual Max Diff: {diff_residual.item()}")
print(f"Output Max Diff: {diff_output.item()}")
```
## 129. apply_rope_pos_ids_cos_sin_cache
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool = False,
    enable_pdl: bool = False,
    cuda_stream: int = 0,
    v: Optional[torch.Tensor] = None,
    k_buffer: Optional[torch.Tensor] = None,
    v_buffer: Optional[torch.Tensor] = None,
    kv_cache_loc: Optional[torch.Tensor] = None
) -> None:
    pass
```
### 功能描述
apply_rope_pos_ids_cos_sin_cache 算子用于对查询（Query）和键（Key）张量应用旋转位置编码（Rotary Positional Embeddings, RoPE）。该算子利用预计算的 Cosine 和 Sine 缓存表，根据输入的位置 ID 对向量进行旋转。
此外，该算子支持融合操作：如果提供了 Value 张量以及 KV Cache 的缓冲区（buffer 和 location），它可以在执行 RoPE 的同时，将 Key 和 Value 保存到非连续的 KV Cache 内存中（类似 PagedAttention 的写入过程），从而减少显存读写次数。
**主要功能包括：**
- RoPE 计算：根据 `pos_ids` 从 `cos_sin_cache` 获取旋转系数，应用到 `q` 和 `k`。
- KV Cache 写入（可选）：如果提供了 `v`、`k_buffer`、`v_buffer` 和 `kv_cache_loc`，则将 RoPE 后的 Key 和原始 Value 写入指定的 Cache 显存位置。
- 模式支持：支持交错（Interleaved）模式和普通模式的 RoPE 计算。
### 参数说明
- **q** (torch.Tensor, 入参): 输入的 Query 张量，形状为 `[nnz, num_heads, head_dim]`，数据类型通常为 float16 或 bfloat16。`nnz` 为总 Token 数（Batch x SeqLen）。
- **k** (torch.Tensor, 入参): 输入的 Key 张量，形状为 `[nnz, num_kv_heads, head_dim]`，数据类型与 `q` 一致。
- **q_rope** (torch.Tensor, 出参): 输出的经过 RoPE 处理的 Query 张量，形状与 `q` 相同。
- **k_rope** (torch.Tensor, 出参): 输出的经过 RoPE 处理的 Key 张量，形状与 `k` 相同。
- **cos_sin_cache** (torch.Tensor, 入参): 预计算的 Cos/Sin 缓存表，形状为 `[max_seq_len, rotary_dim]`。通常前一半为 Cos，后一半为 Sin，或者根据具体实现布局。
- **pos_ids** (torch.Tensor, 入参): 每个 Token 对应的位置索引，形状为 `[nnz]`，数据类型为 int64。
- **interleave** (bool, 入参): 是否使用交错模式计算 RoPE。默认为 False。
- **enable_pdl** (bool, 入参): 是否开启 PDL (Parallel Data Layout) 优化。默认为 False。
- **cuda_stream** (int, 入参): CUDA 流的整数句柄。通常使用 `torch.cuda.current_stream (). cuda_stream` 获取。
- **v** (Optional[torch.Tensor], 入参): 输入的 Value 张量（可选），形状为 `[nnz, num_kv_heads, head_dim]`。仅在需要同时保存 KV Cache 时提供。
- **k_buffer** (Optional[torch.Tensor], 出参): KV Cache 中的 Key 缓冲区（可选），形状通常为 `[num_blocks, block_size, num_kv_heads, head_dim]` 或类似结构。
- **v_buffer** (Optional[torch.Tensor], 出参): KV Cache 中的 Value 缓冲区（可选）。
- **kv_cache_loc** (Optional[torch.Tensor], 入参): KV Cache 的写入位置索引（可选），形状为 `[nnz]`。
### 返回值
无返回值。结果直接写入 `q_rope`、`k_rope` 以及可选的 `k_buffer` 和 `v_buffer` 中。
### 约束与调用
- 输入张量 `q` 和 `k` 的最后一维必须是连续的（Contiguous）。
- 所有 Tensor 必须位于相同的 CUDA 设备上。
- `q` 和 `k` 的第一维（Token 数）和最后一维（Head Dim）必须相等。
- 如果启用 KV Cache 保存功能，必须同时提供 `v`、`k_buffer`、`v_buffer` 和 `kv_cache_loc`，且它们的维度必须匹配。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
# 预防 float16 dispatch 错误，使用 bfloat16
DTYPE = torch.bfloat16
NNZ = 10              # 总 Token 数
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 128
ROTARY_DIM = 128      # 旋转维度
MAX_SEQ_LEN = 1024
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensors
q = torch.randn(NNZ, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE)
k = torch.randn(NNZ, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
# 模拟 Cos/Sin Cache (假设简单布局)
cos_sin_cache = torch.randn(MAX_SEQ_LEN, ROTARY_DIM, dtype=DTYPE)
# 生成随机位置 ID
pos_ids = torch.randint(0, MAX_SEQ_LEN, (NNZ,), dtype=torch.int64)
# 2. 准备输出 Tensors
q_rope = torch.empty_like(q)
k_rope = torch.empty_like(k)
# 3. 获取 CUDA Stream
current_stream = torch.cuda.current_stream().cuda_stream
# 算子执行 (仅 RoPE，不保存 Cache)
print(f"Running apply_rope_pos_ids_cos_sin_cache with {DTYPE}...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 修复：传入 cuda_stream 参数
torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache(
    q,
    k,
    q_rope,
    k_rope,
    cos_sin_cache,
    pos_ids,
    False,           # interleave
    False,           # enable_pdl
    current_stream,  # cuda_stream (Fix: 补充此参数)
    None,   # v
    None,   # k_buffer
    None,   # v_buffer
    None    # kv_cache_loc
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
print("执行成功。")
print(f"Q_RoPE Shape: {q_rope.shape}")
print(f"K_RoPE Shape: {k_rope.shape}")
```
## 130. fused_mla_absorb_rotary_emb
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fused_mla_absorb_rotary_emb(
    q: torch.Tensor,
    w_kc: torch.Tensor,
    latent_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    q_len: int,
    num_local_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int
) -> int:
    pass
```
### 功能描述
fused_mla_absorb_rotary_emb 算子用于 MLA（Multi-Head Latent Attention）机制中的融合计算。该算子将旋转位置编码（RoPE）的应用、权重吸收（Absorb）以及相关的层归一化（LayerNorm）操作融合，用于高效生成用于后续计算的 Query、Key 和 Value 张量。
此功能主要应用于以下场景：
1. DeepSeek-V2/V3 推理：针对 MLA 架构的特有优化，处理 Latent Cache 到投影后 Q/K/V 的转换。
2. 显存与计算优化：通过融合多个 element-wise 和 GEMV 操作，减少显存读写次数，提升推理性能。
- 操作逻辑：
    算子接收原始查询 q、键内容权重 w_kc、潜在缓存 latent_cache 以及位置编码表 cos_sin_cache。
    算子结合 positions 索引应用 RoPE，并结合 norm_weight 进行归一化处理。
    最终将计算结果写入输出张量 q_input、k_input 和 v_input。
**主要功能包括**：
- 对 Latent Cache 进行归一化和权重计算。
- 融合 RoPE 位置编码应用。
- 按照 MLA 结构输出投影后的 Q、K、V 数据。
### 参数说明
- **q** (torch.Tensor, 入参): 查询张量，形状为 `[q_len, num_local_heads, qk_nope_head_dim + qk_rope_head_dim]`，数据类型为 bfloat16。其中 `q_len` 对应 Batch Size 或当前批次的 Token 总数。
- **w_kc** (torch.Tensor, 入参): Key-Content 权重张量，形状为 `[num_local_heads, qk_nope_head_dim, kv_lora_rank]`，数据类型为 float32。
- **latent_cache** (torch.Tensor, 入参): 潜在缓存张量，形状为 `[bs, kv_lora_rank + qk_rope_head_dim]`，数据类型为 bfloat16。
- **cos_sin_cache** (torch.Tensor, 入参): RoPE 正余弦缓存表，形状为 `[max_position_embeddings, qk_rope_head_dim]`，数据类型为 bfloat16。
- **positions** (torch.Tensor, 入参): 位置索引张量，形状为 `[bs]`，数据类型为 int64。
- **norm_weight** (torch.Tensor, 入参): 归一化权重张量，形状为 `[kv_lora_rank]`，数据类型为 float32。
- **q_input** (torch.Tensor, 出参): Query 输出张量，形状为 `[bs, qk_nope_head_dim, kv_lora_rank + qk_rope_head_dim]`，数据类型为 bfloat16。
- **k_input** (torch.Tensor, 出参): Key 输出张量，形状为 `[bs, 1, kv_lora_rank + qk_rope_head_dim]`，数据类型为 bfloat16。
- **v_input** (torch.Tensor, 出参): Value 输出张量，形状为 `[bs, 1, kv_lora_rank]`，数据类型为 bfloat16。
- **q_len** (int, 入参): 查询长度，必须等于 `q` 张量的第 0 维大小 (`q.size (0)`)。
- **num_local_heads** (int, 入参): 本地注意力头数。
- **kv_lora_rank** (int, 入参): KV LoRA 的秩（Rank）大小。
- **qk_rope_head_dim** (int, 入参): Query/Key 的 RoPE 维度大小。
- **qk_nope_head_dim** (int, 入参): Query/Key 的非 RoPE 维度大小。
### 返回值
返回 int 类型状态码（通常为 0 表示成功），计算结果直接写入 `q_input`、`k_input` 和 `v_input` 张量中。
### 约束与调用
- 输入 Tensor 必须在 GPU 上。
- **严格的混合精度要求**：
    - 数据流张量 (`q`, `latent_cache`, `cos_sin_cache`) 必须为 **bfloat16**。
    - 权重参数张量 (`w_kc`, `norm_weight`) 必须为 **float32**。
- `q.size (0)` 必须严格等于传入的 `q_len` 参数。
- `positions` 的值不能超出 `cos_sin_cache` 的最大长度。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel

# 配置参数
SEED = 0
DEVICE = "cuda"
BS = 4
Q_LEN = 4
NUM_LOCAL_HEADS = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
MAX_POS = 4096
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建数据流张量 (BFloat16)
q = torch.randn((BS, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), dtype=torch.bfloat16)
latent_cache = torch.randn((BS, KV_LORA_RANK + QK_ROPE_HEAD_DIM), dtype=torch.bfloat16)
cos_sin_cache = torch.randn((MAX_POS, QK_ROPE_HEAD_DIM), dtype=torch.float32)
# 2. 创建权重张量 
w_kc = torch.randn(
    (NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK), 
    dtype=torch.bfloat16
)
norm_weight = torch.randn(
    (KV_LORA_RANK,), 
    dtype=torch.bfloat16
)
# 3. 创建索引张量 (Int64)
positions = torch.randint(0, MAX_POS, (BS,), dtype=torch.int64)
# 4. 创建输出张量 (BFloat16)
q_input = torch.empty((BS, QK_NOPE_HEAD_DIM, KV_LORA_RANK + QK_ROPE_HEAD_DIM), dtype=torch.bfloat16)
k_input = torch.empty((BS, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM), dtype=torch.bfloat16)
v_input = torch.empty((BS, 1, KV_LORA_RANK), dtype=torch.bfloat16)
print(f"Running fused_mla_absorb_rotary_emb...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.fused_mla_absorb_rotary_emb(
    q,
    w_kc,
    latent_cache,
    cos_sin_cache,
    positions,
    norm_weight,
    q_input,
    k_input,
    v_input,
    Q_LEN,
    NUM_LOCAL_HEADS,
    KV_LORA_RANK,
    QK_ROPE_HEAD_DIM,
    QK_NOPE_HEAD_DIM
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
```
## 131. copy_to_gpu_no_ce
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def copy_to_gpu_no_ce(
    input: torch.Tensor,
    output: torch.Tensor
) -> None:
    pass
```
### 功能描述
copy_to_gpu_no_ce 算子用于将小型整型数组从 CPU 内存高效复制到 GPU 显存。该算子不使用标准的 DMA 拷贝引擎（Copy Engine），而是通过将数据作为内核参数直接传递给 GPU 内核来实现数据传输。
此功能主要应用于以下场景：
1. 元数据传输：传输少量的控制信息或形状信息（如 tensor shapes, flags）。
2. 低延迟同步：避免启动 DMA 引擎带来的额外开销，适用于极小数据量的快速传输。
- 操作逻辑：
    算子接收源张量 input 和目标张量 output。
    算子将 input 中的所有元素逐一复制到 output 对应的位置。
    $$
    \forall i, \quad \text{output}[i] \leftarrow \text{input}[i]
    $$
    其中 $i$ 为元素索引。
**主要功能包括**：
- 将 Int32 数据从 CPU 快速复制到 CUDA 设备。
- 利用内核参数传递机制绕过传统内存拷贝路径。
- 严格校验数据长度（仅支持特定的固定长度，如 64 或 72）。
### 参数说明
- **input** (torch.Tensor, 入参): 源张量，必须位于 CPU 上。形状为 1D，数据类型必须为 int32，且内存连续。元素数量必须严格等于 64 或 72。
- **output** (torch.Tensor, 出参): 目标张量，必须位于 CUDA 设备上。形状为 1D，数据类型必须为 int32，且内存连续。元素数量必须与 `input` 一致。
### 返回值
无返回值，操作结果直接写入 `output` 张量中
### 约束与调用
- `input` 必须是 CPU 张量，`output` 必须是 CUDA 张量
- `input` 和 `output` 的数据类型必须严格为 `torch.int32`
- `input` 和 `output` 必须是连续的（Contiguous）且为 1 维张量
- 张量的元素个数（numel）必须严格等于 64 或 72，否则会抛出异常
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
N = 64  # 支持 64 或 72
DEVICE_CPU = "cpu"
DEVICE_GPU = "cuda"
DTYPE = torch.int32
# 初始化环境
torch.manual_seed(SEED)
# 1. 创建 CPU 输入 Tensor
# 数据必须是 int32 类型，长度必须符合约束
src_data = [i for i in range(N)]
input_cpu = torch.tensor(src_data, dtype=DTYPE, device=DEVICE_CPU)
# 2. 创建 GPU 输出 Tensor
output_gpu = torch.empty_like(input_cpu, device=DEVICE_GPU)
# 算子执行
print(f"Running copy_to_gpu_no_ce with N={N}...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.copy_to_gpu_no_ce(
    input_cpu,
    output_gpu
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
# 将结果拷回 CPU 进行比对
output_cpu = output_gpu.cpu()
diff = (input_cpu - output_cpu).abs().sum()
print("执行成功。")
print(f"Diff: {diff.item()}")
```
## 132. concat_mla_k
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def concat_mla_k(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor
) -> None:
    pass
```
### 功能描述
concat_mla_k 算子用于在多头潜在注意力（MLA）机制中组合生成最终的 Key 张量。该算子将非旋转位置编码部分（Nope）和旋转位置编码部分（RoPE）拼接并存储到目标内存中。
此功能主要应用于以下场景：
1. MLA 推理：在 DeepSeek 等采用 MLA 架构的模型中，将压缩后的 latent 向量投影出的两部分（pe 和 nope）合并。
2. 显存布局优化：将分散存储的 Key 组件合并为连续的张量，以便后续 Attention 计算。
- 操作逻辑：
    算子接收目标张量 k 以及两个源张量 k_nope 和 k_rope。
    对于每一个 Token 和 Head，算子将 k_nope 复制到 k 的前 128 维，将 k_rope 复制到 k 的后 64 维。其中 k_rope 会在 Head 维度上进行广播。
    $$
    \text{k}[t, h, 0:128] \leftarrow \text{k\_nope}[t, h, 0:128]
    $$
    $$
    \text{k}[t, h, 128:192] \leftarrow \text{k\_rope}[t, 0, 0:64]
    $$
**主要功能包括**：
- 高效拼接 MLA 的 Key 组件。
- 支持 RoPE 部分的广播机制（所有 Head 共享同一个 RoPE 向量）。
- 利用向量化加载和存储（int2/int4）优化显存带宽利用率。
- 自动处理 L2 预取以隐藏访存延迟。
### 参数说明
- **k** (torch.Tensor, 出参): 目标 Key 张量，形状为 `[num_tokens, 128, 192]`，数据类型必须为 `bfloat16`。其中 192 = 128 (Nope) + 64 (RoPE)。
- **k_nope** (torch.Tensor, 入参): 非 RoPE 部分的 Key 张量，形状为 `[num_tokens, 128, 128]`，数据类型必须为 `bfloat16`。
- **k_rope** (torch.Tensor, 入参): RoPE 部分的 Key 张量，形状为 `[num_tokens, 1, 64]`，数据类型必须为 `bfloat16`。
### 返回值
无返回值，操作结果直接写入 `k` 张量中
### 约束与调用
- 所有输入输出张量必须在 CUDA 设备上
- 数据类型严格限制为 `torch.bfloat16`
- 张量的最后一维必须是连续的（Stride 为 1）
- 内存地址必须 16 字节对齐
- 维度约束：Head 数量固定为 128，Nope 维度固定为 128，RoPE 维度固定为 64
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 固定维度参数
NUM_TOKENS = 16
NUM_HEADS = 128
NOPE_DIM = 128
ROPE_DIM = 64
TOTAL_DIM = NOPE_DIM + ROPE_DIM
# 初始化环境
torch.manual_seed(SEED)
# 1. 创建输入 Tensor
# k_nope: [num_tokens, num_heads, nope_dim]
k_nope = torch.randn((NUM_TOKENS, NUM_HEADS, NOPE_DIM), dtype=DTYPE, device=DEVICE)
# k_rope: [num_tokens, 1, rope_dim] -> 会被广播
k_rope = torch.randn((NUM_TOKENS, 1, ROPE_DIM), dtype=DTYPE, device=DEVICE)
# 2. 创建输出 Tensor
k_out = torch.empty((NUM_TOKENS, NUM_HEADS, TOTAL_DIM), dtype=DTYPE, device=DEVICE)
# 算子执行
print(f"Running concat_mla_k...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.concat_mla_k(
    k_out,
    k_nope,
    k_rope
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
# 验证 Nope 部分
diff_nope = (k_out[..., :NOPE_DIM] - k_nope).abs().sum()
# 验证 RoPE 部分 (需要手动广播对比)
k_rope_expanded = k_rope.expand(NUM_TOKENS, NUM_HEADS, ROPE_DIM)
diff_rope = (k_out[..., NOPE_DIM:] - k_rope_expanded).abs().sum()
print("执行成功。")
print(f"Diff Nope: {diff_nope.item()}")
print(f"Diff RoPE: {diff_rope.item()}")
```
## 133. concat_mla_absorb_q
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def concat_mla_absorb_q(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    out: torch.Tensor
) -> None:
    pass
```
### 功能描述
concat_mla_absorb_q 算子用于 Multi-Head Latent Attention (MLA) 的 Query 向量生成阶段。该算子将非旋转位置编码的 Query 部分（Nope）和旋转位置编码的 Query 部分（RoPE）在最后一个维度上进行拼接，生成完整的 Query 张量。
此功能主要应用于以下场景：
1. MLA Decode/Prefill：在 DeepSeek-V3 等模型架构中，将经过不同投影层得到的 Query 分量合并，以便进行后续的 Attention 计算或矩阵乘法。
2. 显存拼接优化：替代通用的 `torch.cat`，利用针对特定维度（512+64）优化的 CUDA Kernel 减少显存带宽开销。
- 操作逻辑：
    算子接收两个源张量 q_nope 和 q_rope，以及目标张量 out。
    算子将 q_nope 的内容复制到 out 的低位部分，将 q_rope 的内容复制到 out 的高位部分。
    $$
    \text{out}[..., 0:512] \leftarrow \text{q\_nope}[...]
    $$
    $$
    \text{out}[..., 512:576] \leftarrow \text{q\_rope}[...]
    $$
    主要功能包括：
- 针对固定维度（Nope=512, RoPE=64）的高效拼接。
- 使用向量化加载（int4/int）优化读写性能。
- 自动展开循环以提高指令级并行度。
### 参数说明
- **q_nope** (torch.Tensor, 入参): Query 的非位置编码部分，形状为 `[Batch, Head, 512]`，数据类型必须为 `bfloat16`。
- **q_rope** (torch.Tensor, 入参): Query 的旋转位置编码部分，形状为 `[Batch, Head, 64]`，数据类型必须为 `bfloat16`。
- **out** (torch.Tensor, 出参): 输出张量，形状为 `[Batch, Head, 576]`，数据类型必须为 `bfloat16`。其中 576 = 512 + 64。
### 返回值
无返回值，拼接结果直接写入 `out` 张量中
### 约束与调用
- 所有输入输出张量必须在 CUDA 设备上
- 数据类型严格限制为 `torch.bfloat16`
- 张量的维度必须为 3 维
- `q_nope` 最后一维必须为 512，`q_rope` 最后一维必须为 64，`out` 最后一维必须为 576
- 内存地址必须 16 字节对齐
- 最后一维必须是连续的（Stride 为 1）
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.bfloat16
# 固定维度参数
BATCH_SIZE = 16
NUM_HEADS = 128
DIM_NOPE = 512
DIM_ROPE = 64
DIM_OUT = DIM_NOPE + DIM_ROPE
# 初始化环境
torch.manual_seed(SEED)
# 1. 创建输入 Tensor
q_nope = torch.randn((BATCH_SIZE, NUM_HEADS, DIM_NOPE), dtype=DTYPE, device=DEVICE)
q_rope = torch.randn((BATCH_SIZE, NUM_HEADS, DIM_ROPE), dtype=DTYPE, device=DEVICE)
# 2. 创建输出 Tensor
out = torch.empty((BATCH_SIZE, NUM_HEADS, DIM_OUT), dtype=DTYPE, device=DEVICE)
# 算子执行
print(f"Running concat_mla_absorb_q...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.concat_mla_absorb_q(
    q_nope,
    q_rope,
    out
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果 
expected = torch.cat([q_nope, q_rope], dim=-1)
diff = (out - expected).abs().sum()
print("执行成功。")
print(f"Diff: {diff.item()}")
```
## 134. fast_topk
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fast_topk(
    values: torch.Tensor,
    topk: int,
    dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass
```
### 功能描述
fast_topk 算子用于高效地从输入张量中选取数值最大的 K 个元素及其索引。该算子针对推理场景中的采样阶段（Sampling）进行了特定优化，能够根据 K 值的不同自动选择最优的计算策略。
此功能主要应用于以下场景：
1. Greedy Sampling：当 K=1 时（即贪婪解码），算子退化为 Max 规约操作，避免全排序开销，显著降低计算量和显存访问。
2. Top-K Sampling：当 K>1 时，算子执行标准的 Top-K 筛选。
- 操作逻辑：
    算子接收输入张量 values、目标数量 topk 和操作维度 dim。
    当 topk == 1 时，执行最大值查找：
    $$
    \text{output}, \text{indices} = \max(\text{values}, \text{dim})
    $$
    当 topk > 1 时，执行 Top-K 排序：
    $$
    \text{output}, \text{indices} = \text{TopK}(\text{values}, k, \text{dim})
    $$
**主要功能包括**：
- 针对 K=1 的特化路径优化。
- 支持多维张量的指定维度操作。
- 返回值包含排序后的数值（Values）和原始位置索引（Indices）。
### 参数说明
- **values** (torch.Tensor, 入参): 输入的分数或概率张量（通常为 Logits），数据类型支持 float16, bfloat16 或 float32。
- **topk** (int, 入参): 需要保留的最大元素个数。必须大于 0。
- **dim** (int, 入参): 进行筛选的维度索引。
### 返回值
返回一个包含两个张量的元组 `(output_values, output_indices)`：
- **output_values**: 包含前 K 个最大数值的张量，形状与输入张量在 `dim` 维度上变为 `topk`。
- **output_indices**: 对应数值在原始张量中的索引，数据类型为 int64。
### 约束与调用
- `values` 必须位于 CUDA 设备上
- `topk` 必须小于或等于 `values` 在 `dim` 维度的大小
- 返回的 `output_values` 数据类型与输入一致，`output_indices` 为 int64
- 建议 `dim` 为最后一维（-1）以获得最佳内存访问性能
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel

# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float32  
# 维度设置
BATCH_SIZE = 4
VOCAB_SIZE = 64000  # 需大于 K
K = 2048
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建分数 Tensor
score = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=DTYPE, device=DEVICE).contiguous()
# 2. 创建输出 Indices Tensor (测试文件要求为 Int32)
indices = torch.zeros(BATCH_SIZE, K, dtype=torch.int32, device=DEVICE).contiguous()
# 3. 创建 Lengths Tensor (长度必须为 Batch Size)
lengths = torch.full((BATCH_SIZE,), K, dtype=torch.int32, device=DEVICE).contiguous()
# 算子执行
print(f"Running fast_topk with K={K}...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 注意：C++接口需要4个参数，第4个为 row_starts (optional)，需显式传 None
torch.ops.sgl_kernel.fast_topk(
    score,
    indices,
    lengths,
    None
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
print("执行成功。")
# 验证前将结果转为 int64 以便与 PyTorch 原生算子比较
# 由于 K=2048 较大，这里只比较前 5 个元素以验证正确性
ref_values, ref_indices = torch.topk(score, K, dim=1)
print(f"Output Indices (Batch 0, Top 5): {indices[0, :5].tolist()}")
print(f"Reference Indices (Batch 0, Top 5): {ref_indices[0, :5].tolist()}")
```
## 135. fast_topk_transform_fused
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    dst_page_table: torch.Tensor,
    src_page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None
) -> None:
    pass
```
### 功能描述
fast_topk_transform_fused 算子是 DeepSeek Native Sparse Attention (NSA) 机制中的关键组件，用于根据注意力评分（Gating Score）快速筛选出 Top-K 个最相关的 KV 块，并将这些块的物理索引写入到目标页表中。
此功能主要应用于以下场景：
1. 稀疏注意力选择：在解码或预填充阶段，根据预测的 block 分数，筛选出每一行（或每个 token）需要关注的 KV block。
2. 索引转换：将筛选出的逻辑块索引直接映射为物理显存中的 Block ID，供后续的 FlashMLA 或其他注意力算子使用。
- 操作逻辑：
    算子接收评分张量 score，针对每个查询单元计算分值最高的 K 个块的索引。
    根据计算出的 Top-K 索引，从源页表 src_page_table 中提取对应的物理块 ID。
    将提取出的物理块 ID 存入 dst_page_table 中，供后续稀疏计算使用。
    主要功能包括：
- 融合了 Top-K 排序与 Gather 操作，减少显存访问开销。
- 支持变长序列（通过 `cu_seqlens_q` 控制）。
- 专为稀疏注意力模式下的动态页表构建优化。
### 参数说明
- **score** (torch.Tensor, 入参): 块评分张量，通常包含 Gating 网络输出的 Logits，形状为 `[num_tokens, num_blocks]` 或类似结构，数据类型通常为 float16 或 bfloat16。
- **lengths** (torch.Tensor, 入参): 指定每个查询需要选择的块数量（即 Top-K 中的 K 值，或者扩展后的序列长度），数据类型为 int32。
- **dst_page_table** (torch.Tensor, 出参): 目标页表张量，用于存储筛选后的物理块索引，数据类型为 int32。
- **src_page_table** (torch.Tensor, 入参): 源页表张量，包含所有可用的物理块索引，数据类型为 int32。
- **cu_seqlens_q** (torch.Tensor, 入参): 查询序列的累积长度数组，用于界定 Batch 中每个请求的边界，数据类型为 int32。
- **row_starts** (torch.Tensor, 可选入参): 用于 Ragged 布局下的行偏移量指示，如果为 None 则不使用，数据类型为 int32。
### 返回值
无返回值，计算结果直接写入 `dst_page_table` 张量中。
### 约束与调用
- 输入张量必须在 CUDA 设备上。
- `dst_page_table` 的内存必须预先分配，且大小足以容纳 Top-K 选择后的结果。
- `lengths`、`src_page_table`、`dst_page_table` 和 `cu_seqlens_q` 的数据类型必须为 int32。
- 算子假设 `src_page_table` 包含了完整的块映射关系。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
NUM_TOKENS = 4
TOPK = 2048
NUM_ALL_BLOCKS = 4096 
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 准备输入数据
score = torch.randn(NUM_TOKENS, NUM_ALL_BLOCKS, dtype=torch.float32)
src_page_table = torch.arange(NUM_TOKENS * NUM_ALL_BLOCKS, dtype=torch.int32).reshape(NUM_TOKENS, NUM_ALL_BLOCKS)
dst_page_table = torch.zeros(NUM_TOKENS, TOPK, dtype=torch.int32)
lengths = torch.full((NUM_TOKENS,), TOPK, dtype=torch.int32)
cu_seqlens_q = torch.arange(NUM_TOKENS + 1, dtype=torch.int32)
# 2. 算子执行
print(f"Running fast_topk_transform_fused with TOPK={TOPK}...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.fast_topk_transform_fused(
    score,
    lengths,
    dst_page_table,
    src_page_table,
    cu_seqlens_q,
    None 
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
print("执行成功。")
print(f"DST Page Table Shape: {dst_page_table.shape}")
print(f"Output Sample (Batch 0, Top 5): {dst_page_table[0, :5].tolist()}")
```
## 136. fast_topk_transform_ragged_fused
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    output_indices: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None
) -> None:
    pass
```
### 功能描述
fast_topk_transform_ragged_fused 算子是 DeepSeek Native Sparse Attention (NSA) 机制在 Ragged（非分页）模式下的核心组件。它用于根据注意力评分（Gating Score）筛选出 Top-K 个最相关的 KV 块，并将这些逻辑块索引转换为用于 Ragged Attention 计算的平坦化（Flattened）物理索引。
注意： 该算子为 In-place 操作，需要调用方预先分配用于存储结果的输出张量。
此功能主要应用于以下场景：
1. 稀疏注意力预填充/解码：在不使用 PagedAttention 而使用 Ragged Buffer 的场景下，根据 Block 分数筛选需要关注的区域。
2. 索引平坦化：将批次中不同请求选出的 Top-K 块索引，结合偏移量 `topk_indices_offset`，压缩为一维或紧凑的索引张量。
- 操作逻辑：
    算子接收评分张量 score，针对每个查询单元计算分值最高的 K 个块的逻辑索引。
    将逻辑索引与 topk_indices_offset 相加，计算出在全局 Ragged KV 存储中的绝对偏移位置。
    结果直接写入预分配的 output_indices 张量中。
**主要功能包括：**
- 高效的 Top-K 筛选与索引计算融合。
- 支持 Ragged 内存布局下的动态索引生成。
- 减少 Python 层面的显存操作开销，加速稀疏注意力计算准备阶段。
### 参数说明
- **score** (torch.Tensor, 入参): 块评分张量，通常包含 Gating 网络输出的 Logits，形状为 `[num_tokens, num_blocks]`，数据类型必须为 float32。
- **lengths** (torch.Tensor, 入参): 指定每个查询需要选择的块数量（通常对应扩展后的序列长度），数据类型为 int32。
- **output_indices** (torch.Tensor, 出参): 预分配的输出张量，用于存储计算出的物理索引。形状通常为 `[total_selected_blocks]` (即所有 lengths 之和)，数据类型为 int32。
- **topk_indices_offset** (torch.Tensor, 入参): Top-K 索引在全局 Ragged 存储中的起始偏移量，通常由累积序列长度计算得出，数据类型为 int32。
- **row_starts** (torch.Tensor, 可选入参): 指示 Batch 中每个请求的起始行位置。对于稠密 `score` 输入，建议传入 `None` 以避免 Shape 检查错误；仅在 `score` 为平坦化 Ragged 输入时需要传入，数据类型为 int32。
### 返回值
无返回值（None）。计算结果直接写入传入的 `output_indices` 张量中。
### 约束与调用
- 输入张量必须在 CUDA 设备上。
- `score` 必须是 float32 类型。
- `lengths`、`output_indices`、`topk_indices_offset` 的数据类型必须为 int32。
- 调用前必须确保 `output_indices` 已分配足够的显存空间。
- 不再需要传入 `int topk` 参数，TopK 值由 `lengths` 张量隐式确定。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
NUM_TOKENS = 4
TOPK = 2048
NUM_BLOCKS = 4096
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 准备输入数据
score = torch.randn(NUM_TOKENS, NUM_BLOCKS, dtype=torch.float32)
lengths = torch.full((NUM_TOKENS,), TOPK, dtype=torch.int32)
topk_indices_offset = torch.arange(0, NUM_TOKENS * TOPK, TOPK, dtype=torch.int32).contiguous()
output_indices = torch.empty((NUM_TOKENS, TOPK), dtype=torch.int32).contiguous()
# 2. 算子执行
print(f"Running fast_topk_transform_ragged_fused with TOPK={TOPK}...")
print(f"Arg 3 (Output) Shape: {output_indices.shape}      (Expect: [B, TopK])")
print(f"Arg 4 (Offset) Shape: {topk_indices_offset.shape} (Expect: [B])")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(
    score,
    lengths,
    output_indices,      
    topk_indices_offset,  
    None
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
print("执行成功。")
# 打印展平后的前5个结果
print(f"Sample Indices (First 5): {output_indices.flatten()[:5].tolist()}")
```
## 140. moe_sum_reduce
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def moe_sum_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    scaling_factor: float
) -> None:
    pass
```
### 功能描述
`moe_sum_reduce` 算子用于 Mixture of Experts (MoE) 模型的输出阶段。它将来自不同专家的加权输出（通常已经按 Top-K 维度堆叠）进行归约求和，并应用缩放因子，从而生成最终的 token 嵌入。该操作通常是 `fused_marlin_moe` 等融合 MoE 计算流程的最后一步。
此功能主要应用于以下场景：
1. MoE 输出聚合：将形状为 `[num_tokens, top_k, hidden_size]` 的中间结果沿 `top_k` 维度求和，得到 `[num_tokens, hidden_size]` 的输出。
2. 路由缩放 (Routed Scaling)：在聚合的同时，将结果乘以一个全局或路由相关的缩放因子（`scaling_factor`），常用于 DeepSeek 等模型的负载均衡或架构设计中。
3. 结果写回：支持原地 (In-place) 或非原地操作，将最终结果写入 `output` 张量。
**主要功能包括：**
- 维度归约：对输入的 `top_k` 维度进行求和，将多专家的贡献合并。
- 数值缩放：在归约过程中应用标量乘法。
- 高效内存访问：优化内存读写模式，减少 MoE 后处理阶段的显存带宽占用。
### 参数说明
- **input** (torch.Tensor, 入参): MoE 层的中间计算结果，包含每个 Token 被路由到的 Top-K 个专家的加权输出。形状通常为 `[num_tokens, top_k, hidden_size]`，数据类型通常为 float16 或 bfloat16。
- **output** (torch.Tensor, 出参): 最终的 MoE 层输出张量。形状为 `[num_tokens, hidden_size]`，数据类型需与 `input` 保持一致。
- **scaling_factor** (float, 入参): 应用于归约结果的乘法缩放因子。
### 返回值
无返回值。操作结果直接写入 `output` 张量中。
### 约束与调用
- `input` 必须是 3D 张量，`output` 必须是 2D 张量。
- `input.size (0)` 必须等于 `output.size (0)`（Token 数量一致）。
- `input.size (2)` 必须等于 `output.size (1)`（Hidden Size 一致）。
- `input` 和 `output` 的数据类型必须相同，且必须在 CUDA 设备上。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
DTYPE = torch.float16
# 维度设置
NUM_TOKENS = 128
TOP_K = 2
HIDDEN_SIZE = 4096
SCALING_FACTOR = 1.0
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入输出 Tensors
# input_tensor: [NUM_TOKENS, TOP_K, HIDDEN_SIZE]
input_tensor = torch.randn((NUM_TOKENS, TOP_K, HIDDEN_SIZE), dtype=DTYPE, device=DEVICE)
# output_tensor: [NUM_TOKENS, HIDDEN_SIZE]
output_tensor = torch.empty((NUM_TOKENS, HIDDEN_SIZE), dtype=DTYPE, device=DEVICE)
# 2. 算子执行
print(f"Running moe_sum_reduce...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.moe_sum_reduce(
    input_tensor,
    output_tensor,
    SCALING_FACTOR
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
# 期望计算逻辑：output = sum(input, dim=1) * scaling_factor
expected_output = torch.sum(input_tensor, dim=1) * SCALING_FACTOR
diff = (output_tensor - expected_output).abs().max()
print("执行成功。")
print(f"Max Diff: {diff.item()}")
```
## 142. moe_fused_gate
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def moe_fused_gate(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    pass
```
### 功能描述
moe_fused_gate 算子实现了 Mixture-of-Experts (MoE) 模型中专家选择（Gating）的融合计算逻辑，特别是针对类似 DeepSeek V2/V3/R1 系列模型中使用的 Grouped Top-K 路由策略进行了优化。该算子融合了 Softmax、Top-K 选择、分组掩码（Group Masking）和重归一化（Renormalization）等步骤，以提高路由阶段的效率。
此功能主要应用于以下场景：
1. Grouped Top-K Routing：首先在专家组层面进行粗粒度筛选（Top-K Group），然后在选定组内的专家中进行细粒度筛选（Top-K Expert）。
2. Shared Expert Integration：支持将共享专家（Shared Experts）与路由选择的专家合并，通常共享专家会被固定分配给每个 Token。
3. Bias Correction：支持在计算路由分数时应用校正偏置（Correction Bias），用于辅助负载均衡或其他调节目的。
**主要功能包括：**
- 高效的 Top-K 选择：在一个 Kernel 中完成组级和专家级的 Top-K 选择。
- 灵活的路由配置：支持配置组数量、每组选取的专家数、总 Top-K 数以及共享专家数。
- 数值稳定性：在计算过程中处理 Softmax 和归一化，确保数值稳定。
### 参数说明
- **gating_output** (torch.Tensor, 入参): Gating 网络的原始输出 Logits，形状为 `[num_tokens, num_experts]`，数据类型通常为 float32。
- **correction_bias** (torch.Tensor, 入参): 用于修正路由分数的偏置张量，形状通常为 `[num_experts]` 或 `[num_tokens, num_experts]`（支持广播），数据类型需与 `gating_output` 兼容。
- **num_expert_group** (int, 入参): 专家的分组总数。
- **topk_group** (int, 入参): 每个 Token 选择的专家组数量。
- **topk** (int, 入参): 每个 Token 最终选择的路由专家数量（不包括共享专家）。
- **num_fused_shared_experts** (int, 可选入参): 融合的共享专家数量，默认为 0。
- **routed_scaling_factor** (float, 可选入参): 路由专家的缩放因子，用于调整路由专家的权重，默认为 1.0。
- **apply_routed_scaling_factor_on_output** (bool, 可选入参): 是否将 `routed_scaling_factor` 直接应用到输出的权重上，默认为 False。
### 返回值
返回一个元组 `(topk_weights, topk_ids)`：
- **topk_weights** (torch.Tensor): 选定专家的权重，形状为 `[num_tokens, topk + num_fused_shared_experts]`，数据类型为 float32。
- **topk_ids** (torch.Tensor): 选定专家的索引，形状为 `[num_tokens, topk + num_fused_shared_experts]`，数据类型为 int32。
### 约束与调用
- 输入张量必须在 CUDA 设备上。
- `gating_output` 的最后一维大小（专家总数）必须能被 `num_expert_group` 整除。
- `topk` 必须小于等于专家总数。
- 目前算子实现可能对 `num_expert_group` 和专家数量有特定限制（例如 `num_experts / num_expert_group <= 32`，这取决于具体的 Kernel 实现细节，如源码注释所述）。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
NUM_TOKENS = 128
NUM_EXPERTS = 64
NUM_EXPERT_GROUP = 8
TOPK_GROUP = 2
TOPK = 4
NUM_FUSED_SHARED_EXPERTS = 0
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 创建输入 Tensors
# gating_output: [NUM_TOKENS, NUM_EXPERTS]
gating_output = torch.randn((NUM_TOKENS, NUM_EXPERTS), dtype=torch.float32, device=DEVICE)
# correction_bias: [NUM_EXPERTS]
correction_bias = torch.zeros((NUM_EXPERTS,), dtype=torch.float32, device=DEVICE)
# 2. 算子执行
print(f"Running moe_fused_gate...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
topk_weights, topk_ids = torch.ops.sgl_kernel.moe_fused_gate(
    gating_output,
    correction_bias,
    NUM_EXPERT_GROUP,
    TOPK_GROUP,
    TOPK,
    NUM_FUSED_SHARED_EXPERTS,
    1.0,  # routed_scaling_factor
    False # apply_routed_scaling_factor_on_output
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果形状
print("执行成功。")
print(f"TopK Weights Shape: {topk_weights.shape}") # Should be [NUM_TOKENS, TOPK]
print(f"TopK IDs Shape: {topk_ids.shape}")         # Should be [NUM_TOKENS, TOPK]
```
## 143. prepare_moe_input
### 支持的产品型号
- Metax C500/C550
### 接口原型
```Python
def prepare_moe_input(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    block_scale_offsets: Optional[torch.Tensor],
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int
) -> None:
    pass
```
### 功能描述
`prepare_moe_input` 算子用于为 CUTLASS 基础的 Mixture of Experts (MoE) 计算准备元数据。它负责统计每个专家被分配的 Token 数量，生成用于重排输入数据的索引映射，并设置分组矩阵乘法（Grouped GEMM）所需的维度信息。
此功能主要应用于以下场景：
1. MoE 输入重排：在执行专家计算前，需要将分散在 batch 中的 token 按照其选择的专家进行聚类，以便进行连续内存访问和批量计算。
2. 元数据构建：计算 `expert_offsets`（每个专家处理的 token 起始位置）和 `problem_sizes`（每个专家 GEMM 的 M/N/K 维度），供后续的 CUTLASS Grouped GEMM 内核使用。
3. 结果还原：生成逆向映射索引，以便在专家计算完成后将结果还原回原始的 Token 顺序。
- 操作逻辑：
    1. 统计 `topk_ids` 中每个专家出现的次数，计算前缀和得到 `expert_offsets`。
    2. 生成 `src_idx`（Permutation Index），用于将输入 Tensor (`[batch_size, k]`) 扩展并重排为 `[batch_size * topk, k]`，使得属于同一专家的行在内存中连续。
    3. 生成 `dst_idx`（Inverse Permutation Index），用于后续 `apply_shuffle_mul_sum` 等算子将计算结果写回原始位置。
    4. 填充 `problem_sizes1` 和 `problem_sizes2`，分别定义 Up-Projection 和 Down-Projection 阶段每个专家对应的 GEMM 维度 (M, N, K)。
**主要功能包括：**
- 基于 Top-K 索引生成排序和还原映射。
- 计算每个专家的负载（Token 数量）。
- 初始化 Grouped GEMM 的问题描述符。
### 参数说明
- **topk_ids** (torch.Tensor, 入参): 每个 Token 选择的专家索引。形状为 `[num_tokens, topk]`，数据类型为 int32。
- **expert_offsets** (torch.Tensor, 出参): 用于存储每个专家的累积 Token 计数。形状通常为 `[num_experts]` 或 `[num_experts + 1]`，数据类型为 int32。
- **block_scale_offsets** (torch.Tensor, 可选入参): 用于 FP4 等量化的块缩放偏移量。若不需要可传入 None。
- **problem_sizes1** (torch.Tensor, 出参): 存储第一个 GEMM（Up-proj）每个专家的问题大小信息。形状通常为 `[num_experts, 3]` 或类似结构，存储 `(m, n, k)`，数据类型为 int64 或 int32。
- **problem_sizes2** (torch.Tensor, 出参): 存储第二个 GEMM（Down-proj）每个专家的问题大小信息。形状同上。
- **src_idx** (torch.Tensor, 出参): 源数据重排索引 (Input Permutation)。形状为 `[num_tokens * topk]`，数据类型为 int32。
- **dst_idx** (torch.Tensor, 出参): 目标数据还原索引 (Output Permutation)。形状为 `[num_tokens * topk]`，数据类型为 int32。
- **num_experts** (int, 入参): 专家总数。
- **intermediate_size** (int, 入参): 中间层维度大小（通常是 FFN 的隐藏层大小），对应 GEMM1 的 N 维或 GEMM2 的 K 维。
- **hidden_size** (int, 入参): 模型隐藏层维度大小，对应 GEMM1 的 K 维或 GEMM2 的 N 维。
### 返回值
无返回值。计算结果直接写入 `expert_offsets`, `problem_sizes`, `src_idx`, `dst_idx` 等输出张量中。
### 约束与调用
- 所有 Tensor 必须位于 CUDA 设备上。
- `topk_ids`、`src_idx`、`dst_idx`、`expert_offsets` 的数据类型必须为 int32。
- 参数顺序严格遵循 C++ 算子声明，特别是 `block_scale_offsets` 位于 `problem_sizes1` 之前。
### 调用示例
```Python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 0
DEVICE = "cuda"
NUM_TOKENS = 4
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_EXPERTS = 4
TOP_K = 2
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
# 1. 模拟输入数据
# topk_ids: [NUM_TOKENS, TOP_K]
topk_ids = torch.tensor([
    [0, 1],
    [1, 2],
    [0, 3],
    [2, 3]
], dtype=torch.int32, device=DEVICE)
# 2. 预分配输出张量
total_tokens = NUM_TOKENS * TOP_K
src_idx = torch.empty((total_tokens,), dtype=torch.int32, device=DEVICE)
dst_idx = torch.empty((total_tokens,), dtype=torch.int32, device=DEVICE)
expert_offsets = torch.empty((NUM_EXPERTS,), dtype=torch.int32, device=DEVICE)
# problem_sizes 结构依赖具体实现，假设为 [NUM_EXPERTS, 3]
problem_sizes1 = torch.zeros((NUM_EXPERTS, 3), dtype=torch.int64, device=DEVICE)
problem_sizes2 = torch.zeros((NUM_EXPERTS, 3), dtype=torch.int64, device=DEVICE)
# 3. 算子执行
print(f"Running prepare_moe_input...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 注意参数顺序：block_scale_offsets 在第3个位置
torch.ops.sgl_kernel.prepare_moe_input(
    topk_ids,              # 1. topk_ids
    expert_offsets,        # 2. expert_offsets
    None,                  # 3. blockscale_offsets (Optional Tensor)
    problem_sizes1,        # 4. problem_sizes1
    problem_sizes2,        # 5. problem_sizes2
    src_idx,               # 6. input_permutation
    dst_idx,               # 7. output_permutation
    NUM_EXPERTS,           # 8. num_experts
    INTERMEDIATE_SIZE,     # 9. n
    HIDDEN_SIZE            # 10. k
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 验证结果
print("执行成功。")
print(f"Expert Offsets: {expert_offsets.tolist()}")
print(f"Source Indices: {src_idx.tolist()}")
```
## 145. apply_shuffle_mul_sum
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def apply_shuffle_mul_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
    factors: Optional[torch.Tensor] = None
) -> None
```
### 功能描述
apply_shuffle_mul_sum 算子实现了针对张量的重排（Shuffle）、按因子缩放（Mul）以及累加聚合（Sum）的融合操作。该算子根据 permutation 提供的索引映射关系，将 input 张量中的各行数据映射到 output 张量的对应位置，并在累加前乘以 factors 权重因子。这种融合算子通过减少中间显存读写，显著提升了专家混合模型（MoE）中 Token 分发或聚合逻辑的执行效率。
- 计算公式：
  $$
  \text{dst}[k \times \text{stride} + j] += 
  \begin{cases} 
  \text{input}[i \times \text{stride} + j], & \text{当 factors 不存在} \\
  \text{input}[i \times \text{stride} + j] \times \text{factors}[i], & \text{当 factors 存在}
  \end{cases}
  $$
  其中元素的线性索引与二维位置的映射关系为：
  $$
  n = i \times \text{stride} + j
  $$
  各变量的维度与含义：
  - $\text{input} \in \mathbb{T}^{\text{height} \times \text{stride}}$：输入张量，包含待处理的原始数据，数据类型为$\mathbb{T}$，有效列范围为$0 \leq j < \text{width}$
  - $\text{dst} \in \mathbb{T}^{\text{num\_reduced} \times \text{stride}}$：输出张量，用于存储按索引聚合（Sum）后的结果，数据类型与$\text{input}$一致
  - $\text{perm} \in \mathbb{I}^{\text{height}}$：索引映射张量（permutation），数据类型为$\mathbb{I}$（通常为$\text{int32}$），决定$\text{input}$的第$i$行映射到$\text{dst}$的第$k$行
  - $\text{factors} \in \mathbb{T}^{\text{height}}$：可选的权重因子张量，若存在，则在累加前对输入元素进行缩放，数据类型与$\text{input}$一致
  - $i$：输入张量的行索引，范围为$0 \leq i < \text{height}$
  - $k$：输出张量的行索引，$k = \text{perm}[i]$，范围为$0 \leq k < \text{num\_reduced}$
  - $j$：张量的列索引，范围为$0 \leq j < \text{stride}$（$\text{stride} \geq \text{width}$，保证内存对齐）
  - $\text{height}$：输入张量的总行数（如 Token 总数）
  - $\text{num\_reduced}$：输出张量的总行数（如聚合后的专家位总数）
  - $\text{width}$：张量的有效宽度（每行的有效元素数）
  - $\text{stride}$：张量的内存步长（每行元素的实际内存间隔）
### 参数说明
- **input** (torch.Tensor, 入参): 输入张量，通常为二维特征矩阵（如 [num_tokens, hidden_dim]）
- **output** (torch.Tensor, 出参): 目标输出张量，用于存储累加后的结果。在调用前通常需要初始化为 0
- **permutation** (torch.Tensor, 入参): 索引映射张量，指定输入行映射到输出行的位置
- **factors** (torch.Tensor, 可选, 入参): 权重因子张量（如 MoE 的门控权重）。若提供，则 output[permutation[i]] += input[i] * factors[i]
### 返回值
无返回值，计算结果直接写入 output 张量中
### 约束与调用
- 所有输入输出张量必须位于 CUDA 设备上
- permutation 必须为 torch.int32 (Int) 类型，不支持 int64 (Long)
- input 和 output 的隐藏维度（最后一维）必须保持一致
- factors 若提供，其长度必须与 input 的行数一致
- 支持的数据类型：float16, bfloat16, float32
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda"
# 核心参数
num_elements = 1024      
num_reduced = 512        
hidden_dim = 128         
# 1. 构造输入张量
input_tensor = torch.randn(
    num_elements, hidden_dim,
    dtype=torch.float32, device=device
)
# 2. 构造输出张量 (初始化为0，因为算子通常执行累加操作)
output_tensor = torch.zeros(
    num_reduced, hidden_dim,
    dtype=torch.float32, device=device
)
# 3. 构造索引映射 (关键修改点：使用 torch.int32)
# 报错是因为默认生成的随机整数是 int64 (Long)，需要强制转为 int32 (Int)
permutation = torch.randint(
    0, num_reduced, (num_elements,),
    dtype=torch.int32, # 修改此处：从 int64 改为 int32
    device=device
)
# 4. 构造权重因子
factors = torch.rand(
    num_elements, 
    dtype=torch.float32, device=device
)
# 5. 调用算子
# 注意：如果算子注册时没有指定参数名，建议按顺序传参
torch.ops.sgl_kernel.apply_shuffle_mul_sum(
    input_tensor,
    output_tensor,
    permutation,
    factors
)
# 输出结果
print("apply_shuffle_mul_sum computation completed successfully")
print(f"Output shape: {output_tensor.shape}")
```
## 146. fused_moe_gate_opt
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_moe_gate_opt(
    gating_outputs: torch.Tensor,
    correction_bias: torch.Tensor,
    out_routing_weights: torch.Tensor,
    out_selected_experts: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    num_fused_shared_experts: Optional[int] = None,
    routed_scaling_factor: Optional[float] = None
) -> int
```
### 功能描述
fused_moe_gate_opt 算子实现了专家混合模型（MoE）中门控网络（Gating Network）的核心逻辑融合。该算子接收门控原始输出，在应用偏置校正（Correction Bias）后，执行 Top-K 专家选择（支持分组 Top-K 逻辑），并计算最终的路由权重（Routing Weights）。通过在单个 Kernel 中完成排序、索引提取、权重计算和可选的重正规化（Renormalize），显著提升了混合专家模型在推理阶段的路由效率。
- 计算公式：
  $$
  \begin{aligned}
  \text{score}[i, e] &= \text{input0}[i \times \text{num\_experts} + e] + \text{input1}[e] \\
  (\text{dst\_w}[i, k], \text{dst\_idx}[i, k]) &= \text{TopK}(\{\text{score}[i, e]\}_{e=0}^{\text{num\_experts}-1}, \text{topk}) \\
  \text{dst\_w}[i, k] &= 
  \begin{cases} 
  \text{dst\_w}[i, k] \times \text{scale}, & \text{当 renormalize} = \text{False} \\
  \text{Softmax}(\{\text{dst\_w}[i, k]\}_{k=0}^{\text{topk}-1}) \times \text{scale}, & \text{当 renormalize} = \text{True}
  \end{cases}
  \end{aligned}
  $$
  其中元素的线性索引与二维位置的映射关系为（以输出张量为例）：
  $$
  n = i \times \text{topk} + k
  $$
  各变量的维度与含义：
  - $\text{input0} \in \mathbb{T}^{\text{bs} \times \text{num\_experts}}$：门控网络（Gating）原始输出张量，数据类型为$\mathbb{T}$（$\text{bfloat16}$）
  - $\text{input1} \in \mathbb{T}^{\text{num\_experts}}$：专家偏置校正张量（Correction Bias），数据类型与$\text{input0}$一致
  - $\text{dst\_w} \in \mathbb{D}^{\text{bs} \times \text{topk}}$：路由权重输出张量，存储计算并可选重正规化后的权重，数据类型为$\mathbb{D}$（$\text{float32}$）
  - $\text{dst\_idx} \in \mathbb{I}^{\text{bs} \times \text{topk}}$：专家索引输出张量，存储选中的专家编号，数据类型为$\mathbb{I}$（$\text{int32}$）
  - $\text{scale} \in \mathbb{R}$：路由权重的缩放因子，计算方式为$\text{scale} = 1.0 / \text{routed\_scaling\_factor}$
  - $n$：输出张量元素的线性索引，范围为$0 \leq n < \text{bs} \times \text{topk}$
  - $i$：Batch 维度的行索引（Token 索引），范围为$0 \leq i < \text{bs}$
  - $k$：选中的专家位索引，范围为$0 \leq k < \text{topk}$
  - $e$：原始专家维度的索引，范围为$0 \leq e < \text{num\_experts}$
  - $\text{bs}$：输入张量的第一维大小（Batch Size）
  - $\text{num\_experts}$：专家的总数量
  - $\text{topk}$：每个 Token 最终选取的专家数量，目前仅支持$8$或$9$
  - $\text{Softmax}$：若$\text{renormalize}$参数为$\text{True}$，则对选出的 Top-K 分数执行归一化操作
  - $\text{TopK}$：选取分数最高的$\text{topk}$个专家，返回对应权重和索引
  - $\text{renormalize}$：布尔型参数，控制是否对 Top-K 权重执行 Softmax 归一化
### 参数说明
- **gating_outputs** (torch.Tensor, 入参): 门控网络的原始输出张量，维度通常为 [batch_size, num_experts]，要求数据类型为 bfloat16
- **correction_bias** (torch.Tensor, 入参): 专家偏置校正张量，维度为 [num_experts]，要求数据类型为 bfloat16
- **out_routing_weights** (torch.Tensor, 出参): 存储计算出的路由权重，维度为 [batch_size, topk]，要求数据类型为 float32
- **out_selected_experts** (torch.Tensor, 出参): 存储选中的专家索引，维度为 [batch_size, topk]，要求数据类型为 int32
- **topk** (int, 入参): 每个 Token 选择的专家数量，目前仅支持 8 或 9
- **renormalize** (bool, 入参): 是否对路由权重进行重正规化处理
- **num_expert_group** (int, 入参): 专家分组的数量
- **topk_group** (int, 入参): 每一组内选择的 Top-K 数量
- **num_fused_shared_experts** (int, 可选, 入参): 融合的共享专家数量
- **routed_scaling_factor** (float, 可选, 入参): 路由权重的缩放因子
### 返回值
返回 int 类型的状态码，0 表示执行成功，1 表示因参数不匹配导致的执行失败
### 约束与调用
- 所有张量必须位于 CUDA 设备上；输入必须为 bfloat16，索引输出必须为 int32
- Top-K 限制：若无共享专家，topk 必须为 8；若有共享专家，topk 必须为 9
- 固定参数组合限制：该算子采用了静态分发（Dispatching），仅支持以下特定的专家配置方案，否则会报 "Invalid arguments" 错误：
  - 专家总数 (NUM_EXPERTS)：仅支持 256, 320, 384, 448
  - 典型组合 1 (DeepSeek-V3 风格)：num_experts ∈ {320, 384, 448}, num_expert_group=1, topk_group=1
  - 典型组合 2 (分组风格)：num_experts=256, num_expert_group=8, topk_group=4
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
device = "cuda"
# 修改为算子支持的参数组合（例如 DeepSeek-V3 风格的配置）
num_experts = 320          # 必须是 256, 320, 384 或 448
num_expert_group = 1       # 分组数
topk_group = 1             # 对于 320 专家，组内 Top-K 必须为 1
topk = 8                   # 总 Top-K
# 重新构造符合维度的张量
gating_outputs = torch.randn(4, num_experts, dtype=torch.bfloat16, device=device)
correction_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)
out_routing_weights = torch.empty(4, topk, dtype=torch.float32, device=device)
out_selected_experts = torch.empty(4, topk, dtype=torch.int32, device=device)
# 调用算子
status = torch.ops.sgl_kernel.fused_moe_gate_opt(
    gating_outputs=gating_outputs,
    correction_bias=correction_bias,
    out_routing_weights=out_routing_weights,
    out_selected_experts=out_selected_experts,
    topk=topk,
    renormalize=True,
    num_expert_group=num_expert_group,
    topk_group=topk_group,
    num_fused_shared_experts=0,
    routed_scaling_factor=1.0
)
print("fused_moe_gate_opt computation completed with supported parameters")
print(f"Routing weights shape: {out_routing_weights.shape}")
print(f"Selected experts shape: {out_selected_experts.shape}")
```
## 149. fused_silu_mul_dq_quant_interface
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_silu_mul_dq_quant_interface(
    out: torch.Tensor,
    scale: torch.Tensor,
    input: torch.Tensor
) -> None
```
### 功能描述
fused_silu_mul_dq_quant_interface 算子实现了 SiLU 激活、元素级乘法以及动态量化的融合操作。该算子通常用于 Transformer 架构中的门控线性单元（GLU），它对输入张量进行 SiLU 激活与内部乘法处理，随后直接将结果量化为 **INT8** 格式，并计算相应的动态量化比例因子（Scaling Factor）。这种融合方式大幅减少了高精度中间变量的显存读写，提升了模型推理的吞吐量。
- 计算公式：
  $$
  \begin{aligned}
  \text{gate}[i, j] &= \text{input}[i \times 2 \times \text{width} + j] \\
  \text{up}[i, j] &= \text{input}[i \times 2 \times \text{width} + \text{width} + j] \\
  \text{tmp}[i, j] &= \text{SiLU}(\text{gate}[i, j]) \times \text{up}[i, j] \\
  \text{dst}[i \times \text{stride} + j] &= \text{round}\left(\frac{\text{tmp}[i, j]}{\text{scale}[i]}\right)
  \end{aligned}
  $$
  其中 $\text{SiLU}(x) = \frac{x}{1 + e^{-x}}$，元素的线性索引与二维位置的映射关系为（以输出张量为例）：
  $$
  n = i \times \text{stride} + j
  $$
  各变量的维度与含义：
  - $\text{input} \in \mathbb{T}^{\text{height} \times (2 \times \text{width})}$：输入激活张量，包含 Gate（前一半宽度）和 Up（后一半宽度）的数据，数据类型为$\mathbb{T}$（$\text{float16}$ 或 $\text{bfloat16}$）
  - $\text{dst} \in \mathbb{D}^{\text{height} \times \text{stride}}$：输出张量，存储处理并量化后的结果，数据类型为$\mathbb{D}$（$\text{int8}$），有效列范围为$0 \leq j < \text{width}$
  - $\text{scale} \in \mathbb{R}^{\text{height}}$：输出的动态量化比例因子，存储每一行（每 Token）的缩放系数，数据类型为$\text{float32}$
  - $\text{SiLU}(x)$：激活函数，其公式为$\text{SiLU}(x) = \frac{x}{1 + e^{-x}}$
  - $n$：输出张量元素的线性索引，范围为$0 \leq n < \text{height} \times \text{stride}$
  - $i$：张量的行索引（Token 索引），范围为$0 \leq i < \text{height}$
  - $j$：张量的列索引（隐藏层维度索引），范围为$0 \leq j < \text{width}$
  - $\text{width}$：输出张量的有效宽度，对应模型隐藏层维度，满足$2 \times \text{width} = \text{input.size}(-1)$
  - $\text{height}$：张量的高度，对应输入序列的 Token 数量$\text{num\_tokens}$
  - $\text{stride}$：输出张量的内存步长，用于保证内存对齐（$\text{stride} \geq \text{width}$）
  - $\text{round}(\cdot)$：四舍五入取整操作，将量化后的浮点值转换为$\text{int8}$整数
### 参数说明
- **out** (torch.Tensor, 出参): 输出张量，用于存储量化后的结果，数据类型固定为 **int8**
- **scale** (torch.Tensor, 出参): 比例因子张量，存储每行的量化缩放系数，数据类型固定为 **float32**
- **input** (torch.Tensor, 入参): 输入激活张量，数据类型为浮点型（如 float16 或 bfloat16）
### 返回值
无返回值，计算结果直接写入 `out` 和 `scale` 张量中
### 约束与调用
- 内存连续性：`input`、`scale` 和 `out` 张量都必须是连续的（Contiguous）
- 设备一致性：所有张量必须位于同一个 CUDA/MACA 设备上
- 维度计算：算子内部根据 `input.size(-1)` 计算隐藏层维度 `hidden_size`，并基于总元素数推导 `num_tokens`
- 支持的数据类型：
  - 输入张量支持 `float16` 和 `bfloat16`
  - 输出张量必须为 `int8`
  - Scale 张量必须为 `float32`
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda"
# 核心参数
num_tokens = 4       # Token 数量
hidden_size = 1024   # 隐藏层维度
# 1. 构造输入张量
# input: 通常为前向传播中的激活值，支持浮点类型（如 float16 或 bfloat16）
input_tensor = torch.randn(
    num_tokens, hidden_size, 
    dtype=torch.bfloat16, device=device, requires_grad=False
)
# 2. 构造输出张量
# out: 量化后的输出，根据源码 data_ptr<int8_t>() 可知需要 int8 类型
out = torch.empty(
    num_tokens, hidden_size, 
    dtype=torch.int8, device=device
)
# scale: 动态量化比例因子，根据源码 data_ptr<float>() 可知需要 float32 类型
# 形状通常对应 num_tokens (per-token quantization)
scale = torch.empty(
    num_tokens, 
    dtype=torch.float32, device=device
)
# 3. 调用算子
# 算子定义顺序为 (Tensor! out, Tensor! scale, Tensor input)
torch.ops.sgl_kernel.fused_silu_mul_dq_quant_interface(
    out=out,
    scale=scale,
    input=input_tensor
)
# 输出结果
print("fused_silu_mul_dq_quant_interface computation completed")
print(f"Output (INT8) shape: {out.shape}")
print(f"Quantization scales shape: {scale.shape}")
```
## 150. mx_awq_dequantize
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def mx_awq_dequantize(
    out: torch.Tensor,
    _scaling_factors: torch.Tensor,
    _zeros: torch.Tensor,
    split_k_iter: int,
    thx: int,
    thy: int
) -> torch.Tensor
```
### 功能描述
mx_awq_dequantize 算子实现了针对 AWQ（Activation-aware Weight Quantization）算法量化权重的反量化操作。该算子接收打包存储的低比特量化权重内核、缩放因子以及打包的零点数据，通过内部解压逻辑将其还原为高精度浮点张量。这种融合操作常用于大语言模型（LLM）的量化推理，旨在降低显存占用并提升访存效率。
- 计算公式：
  $$
  \begin{aligned}
  g &= \lfloor j / G \rfloor \\
  p &= j \% P \\
  \text{packed\_idx} &= i \times \text{stride} + \lfloor j / P \rfloor \\
  \text{q\_weight} &= \text{unpack}(\text{input0}[\text{packed\_idx}], p) \\
  \text{q\_zero} &= \text{unpack}(\text{input2}[i \times (\text{stride}/G) + g], p) \\
  \text{dst}[i \times \text{width} + j] &= (\text{q\_weight} - \text{q\_zero}) \times \text{input1}[i \times (\text{width}/G) + g]
  \end{aligned}
  $$
  其中 $\text{unpack}(X, k)$ 函数表示从打包的整数 $X$ 中提取第 $k$ 个低比特量化值（如 4-bit），元素的线性索引与二维位置的映射关系为：
  $$
  n = i \times \text{stride} + \lfloor j / P \rfloor
  $$
  对于打包的输入张量（如 $\text{input0}$），其线性索引映射为：
  $$
  \text{packed\_n} = i \times \text{stride} + \lfloor j / P \rfloor
  $$
  各变量的维度与含义：
  - $\text{input0} \in \mathbb{I}^{\text{height} \times \text{stride}}$：打包后的量化权重张量（Kernel），数据类型为$\mathbb{I}$（$\text{int32}$），有效列范围为$0 \leq \lfloor j / P \rfloor < \text{width}/P$
  - $\text{input1} \in \mathbb{T}^{\text{height} \times (\text{width}/G)}$：量化缩放因子张量（Scaling Factors），数据类型为$\mathbb{T}$（$\text{float16}$ 或 $\text{bfloat16}$）
  - $\text{input2} \in \mathbb{I}^{\text{height} \times (\text{stride}/G)}$：打包后的量化零点张量（Zeros），数据类型与$\text{input0}$一致（$\text{int32}$）
  - $\text{dst} \in \mathbb{T}^{\text{height} \times \text{width}}$：输出张量，存储反量化后的浮点权重，数据类型为$\mathbb{T}$
  - $n$：张量元素的线性索引，范围为$0 \leq n < \text{height} \times \text{stride}$（输入张量）或$0 \leq n < \text{height} \times \text{width}$（输出张量）
  - $i$：输出张量的行索引（对应输出维度 $N$），范围为$0 \leq i < \text{height}$
  - $j$：输出张量的列索引（对应输入维度 $K$），范围为$0 \leq j < \text{width}$
  - $P$：打包因子（Pack Factor），对于 4-bit 量化，$P = 8$（即一个 $\text{int32}$ 存储 8 个元素）
  - $G$：量化分组大小（Group Size），如$G = 128$
  - $\text{width}$：反量化后的矩阵有效宽度（输入特征维度 $K$）
  - $\text{height}$：反量化后的矩阵高度（输出特征维度 $N$）
  - $\text{stride}$：输入打包张量的内存步长，满足$\text{stride} = \text{width} / P$
  - $\text{unpack}(X, k)$：从打包整数 $X$ 中提取第 $k$ 个低比特量化值的解包函数
  - $\text{split\_k\_iter}$：K 维度并行切分优化参数，用于在硬件层面平衡访存与计算开销
### 参数说明
- **out** (torch.Tensor, 入参): 打包后的量化权重张量（Kernel），通常存储 4-bit 量化后的原始权重
- **_scaling_factors** (torch.Tensor, 入参): 权重的缩放因子张量，用于线性还原浮点数值
- **_zeros** (torch.Tensor, 入参): 打包后的量化零点（Zeros）张量，用于偏移还原
- **split_k_iter** (int, 入参): 内部并行计算的 K 维度切分迭代次数，用于优化计算负载
- **thx** (int, 入参): 算子内部 CUDA 线程配置参数 X
- **thy** (int, 入参): 算子内部 CUDA 线程配置参数 Y
### 返回值
返回一个 `torch.Tensor`，即反量化后的高精度浮点权重张量
### 约束与调用
- 所有输入张量必须位于 CUDA 设备上
- 数据类型限制：
- out (Kernel) 必须为 **int32** (`Int`) 类型，传入 `Half` (float16) 会导致运行时错误
- zeros 必须为 **int32** (`Int`) 类型，传入 `Half` (float16) 会导致运行时错误
- scaling_factors 通常支持 `float16` 或 `bfloat16`
- 算子内部会根据输入张量的维度计算专家数量（experts）和每组专家数等信息进行任务分发
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
device = "cuda"
# 1. 构造输入张量
N, K = 4096, 4096
group_size = 128
pack_factor = 8  # 4-bit -> int32
# 权重 Kernel 必须是 int32
kernel = torch.randint(0, 2**31, (N, K // pack_factor), dtype=torch.int32, device=device)
# 缩放因子通常是 float16
scaling_factors = torch.randn(N, K // group_size, dtype=torch.float16, device=device)
# 关键修改：zeros 在此算子中也必须是打包好的 int32 (不能是 Half)
zeros = torch.randint(0, 2**31, (N, K // (group_size * pack_factor)), dtype=torch.int32, device=device)
# 2. 调用算子
output = torch.ops.sgl_kernel.mx_awq_dequantize(
    out=kernel, 
    _scaling_factors=scaling_factors,
    _zeros=zeros,  # 确保传入的是 int32
    split_k_iter=1,
    thx=128,
    thy=8
)
print("mx_awq_dequantize computation completed successfully")
print(f"Output shape: {output.shape}")
```
## 153. tree_speculative_sampling_target_only
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool
) -> None
```
### 功能描述
tree_speculative_sampling_target_only 算子实现了树状投机采样（Tree-based Speculative Sampling）中的目标模型验证逻辑。该算子在 GPU 上并行执行接受/拒绝判定，通过比对草稿模型生成的候选 Token 及其概率分布与目标模型的输出分布，确定最终被接受的 Token 序列。它利用树状结构辅助索引进行路径检索，并能根据采样阈值和随机样本在拒绝发生时进行修正采样，显著提升了大语言模型（LLM）的生成效率。
- 计算公式：
  $$
  \begin{aligned}
  \text{idx} &= \text{candidates}[i, j] \\
  \text{accept\_bit}[i, j] &= 
  \begin{cases} 
  1, & \text{若 } \text{Uniform}(0,1) \leq \frac{\text{target\_probs}[i, j, \text{idx}]}{\text{draft\_probs}[i, j, \text{idx}]} \\
  0, & \text{否则}
  \end{cases} \\
  \text{accept\_token\_num}[i] &= \max\left\{ j \mid \forall 0 \leq t < j, \text{accept\_bit}[i, t] = 1, j \leq \text{steps} \right\} \\
  \text{predicts}[i] &= 
  \begin{cases} 
  \text{candidates}[i, \text{accept\_token\_num}[i]], & \text{若 } \text{accept\_token\_num}[i] < \text{steps} \\
  \text{Sample}(\text{Residue Distribution}(\text{target\_probs}[i, \cdot])), & \text{若拒绝则根据剩余分布采样}
  \end{cases}
  \end{aligned}
  $$
  其中元素的线性索引与二维位置的映射关系为：
  $$
  n = i \times \text{steps} + j
  $$
  各变量的维度与含义：
  - $\text{target\_probs} \in \mathbb{T}^{\text{bs} \times \text{steps} \times \text{vocab}}$：目标模型输出的概率分布张量，数据类型为$\mathbb{T}$（$\text{float32}$）
  - $\text{draft\_probs} \in \mathbb{T}^{\text{bs} \times \text{steps} \times \text{vocab}}$：草稿模型生成的概率分布张量，数据类型与$\text{target\_probs}$一致
  - $\text{candidates} \in \mathbb{I}^{\text{bs} \times \text{steps}}$：草稿模型生成的候选 Token 张量，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{predicts} \in \mathbb{D}^{\text{bs}}$：输出张量，存储最终决定的下一个 Token 结果，数据类型为$\mathbb{D}$（$\text{int32}$）
  - $\text{accept\_token\_num} \in \mathbb{D}^{\text{bs}}$：输出张量，存储每条序列被接受的有效 Token 数量，数据类型为$\mathbb{D}$（$\text{int32}$）
  - $n$：中间计算张量元素的线性索引，范围为$0 \leq n < \text{bs} \times \text{steps}$
  - $i$：Batch 维度的索引，范围为$0 \leq i < \text{bs}$
  - $j$：投机步骤的索引，范围为$0 \leq j < \text{steps}$
  - $\text{bs}$：Batch Size 大小（$\text{batch\_size}$）
  - $\text{steps}$：投机生成的步骤数（$\text{num\_draft\_tokens}$）
  - $\text{vocab}$：词表大小（Vocab Size）
  - $\text{Uniform}(0,1)$：服从$[0,1]$区间均匀分布的随机采样函数
  - $\text{Sample}(\cdot)$：基于概率分布的 Token 采样函数
  - $\text{Residue Distribution}(\cdot)$：拒绝后剩余概率分布，用于最终 Token 采样
### 参数说明
- **predicts** (torch.Tensor, 出参): 存储最终选定的 Token，数据类型必须为 **int32**
- **accept_index** (torch.Tensor, 出参): 存储被接受 Token 的原始索引位置，数据类型为 **int32**
- **accept_token_num** (torch.Tensor, 出参): 每个 Batch 被接受的 Token 数量，数据类型为 **int32**
- **candidates** (torch.Tensor, 入参): 草稿模型生成的候选 Token 张量，数据类型为 **int64**
- **retrive_index** (torch.Tensor, 入参): 用于树状路径检索的辅助索引张量，数据类型为 **int64**
- **retrive_next_token** (torch.Tensor, 入参): 用于树状路径检索的辅助索引张量，数据类型为 **int64**
- **retrive_next_sibling** (torch.Tensor, 入参): 用于树状路径检索的辅助索引张量，数据类型为 **int64**
- **uniform_samples** (torch.Tensor, 入参): 用于随机判定的均匀分布样本，数据类型为 **float32**
- **uniform_samples_for_final_sampling** (torch.Tensor, 入参): 用于随机判定的均匀分布样本，数据类型为 **float32**
- **target_probs** (torch.Tensor, 入参): 目标模型输出的概率分布张量，数据类型为 **float32**
- **draft_probs** (torch.Tensor, 入参): 草稿模型输出的概率分布张量，数据类型为 **float32**
- **threshold_single** (float, 入参): 控制采样的阈值参数
- **threshold_acc** (float, 入参): 控制采样的阈值参数
- **deterministic** (bool, 入参): 是否开启确定性采样模式
### 返回值
无返回值，计算结果直接写入 `predicts`、`accept_index` 和 `accept_token_num` 张量中
### 约束与调用
- 数据类型限制：
  - 预测与接受索引类张量（predicts、accept_index、accept_token_num）必须为 **int32**
  - 候选 Token 及检索路径张量（candidates、retrive_index、retrive_next_token、retrive_next_sibling）必须为 **int64**
  - 概率与随机样本张量（uniform_samples、uniform_samples_for_final_sampling、target_probs、draft_probs）必须为 **float32**
- `target_probs` 的第二维（Step 维）大小必须严格等于 `num_draft_tokens`，否则会触发维度校验报错
- 所有输入输出张量必须位于同一个 CUDA 设备上
- Python 接口注册定义了 14 个参数，不支持在调用时直接传入 `cuda_stream`
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# Set computing device
device = "cuda"
# Core parameters
batch_size = 2
num_spec_step = 4
num_draft_tokens = 5  # num_draft_tokens = 5
vocab_size = 128
# 1. Construct input tensors
# predicts: [batch_size], dtype=int32
predicts = torch.zeros(batch_size, dtype=torch.int32, device=device)
# accept_index: [batch_size, num_spec_step], dtype=int32
accept_index = torch.zeros((batch_size, num_spec_step), dtype=torch.int32, device=device)
# accept_token_num: [batch_size], dtype=int32
accept_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)
# candidates: [batch_size, num_draft_tokens], dtype=int64
candidates = torch.randint(0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int64, device=device)
# retrieve indices: [bs, num_draft_tokens], dtype=int64
retrive_index = torch.zeros((batch_size, num_draft_tokens), dtype=torch.int64, device=device)
retrive_next_token = torch.zeros((batch_size, num_draft_tokens), dtype=torch.int64, device=device)
retrive_next_sibling = torch.zeros((batch_size, num_draft_tokens), dtype=torch.int64, device=device)
# uniform_samples: [batch_size, num_draft_tokens], dtype=float32
uniform_samples = torch.rand((batch_size, num_draft_tokens), dtype=torch.float32, device=device)
# uniform_samples_for_final_sampling: [batch_size], dtype=float32
uniform_samples_for_final_sampling = torch.rand(batch_size, dtype=torch.float32, device=device)
#  关键修正点 
# target_probs: [batch_size, num_draft_tokens, vocab_size]
# 根据报错信息和源码 CHECK_EQ，第二维必须等于 num_draft_tokens (5)，不能是 6
target_probs = torch.rand((batch_size, num_draft_tokens, vocab_size), dtype=torch.float32, device=device)
# draft_probs: [batch_size, num_draft_tokens, vocab_size], dtype=float32
draft_probs = torch.rand((batch_size, num_draft_tokens, vocab_size), dtype=torch.float32, device=device)
# 2. Scalar parameters
threshold_single = 1.0
threshold_acc = 1.0
deterministic = True
# 3. Call the operator
torch.ops.sgl_kernel.tree_speculative_sampling_target_only(
    predicts=predicts,
    accept_index=accept_index,
    accept_token_num=accept_token_num,
    candidates=candidates,
    retrive_index=retrive_index,
    retrive_next_token=retrive_next_token,
    retrive_next_sibling=retrive_next_sibling,
    uniform_samples=uniform_samples,
    uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
    target_probs=target_probs,
    draft_probs=draft_probs,
    threshold_single=threshold_single,
    threshold_acc=threshold_acc,
    deterministic=deterministic
)
# Output results
print("tree_speculative_sampling_target_only computation completed successfully")
print(f"Updated predicts: {predicts}")
print(f"Accepted token count: {accept_token_num}")
```
## 154. verify_tree_greedy
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def verify_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor
) -> None
```
### 功能描述
verify_tree_greedy 算子实现了基于树状结构的贪婪投机采样（Greedy Speculative Sampling）验证逻辑。该算子在 GPU 上并行比对目标模型生成的贪婪预测结果（`target_predict`）与草稿模型提供的树状候选序列（`candidates`）。利用提供的路径检索索引（`retrive_index` 等），算子能够快速确定每条序列中被接受的最长路径，并更新最终预测的 Token 及接受数量。该算子专门用于贪婪解码场景，旨在通过一次目标模型前向传播验证多个草稿 Token，显著降低推理延迟。
- 计算公式：
  $$
  \begin{aligned}
  \text{cand} &= \text{candidates}[i, j] \\
  \text{accept\_bit}[i, j] &= 
  \begin{cases} 
  1, & \text{若候选 Token 满足验证条件（基于目标/草稿模型概率分布）} \\
  0, & \text{否则}
  \end{cases} \\
  \text{accept\_token\_num}[i] &= \max\left\{ j \mid \forall 0 \leq t < j, \text{accept\_bit}[i, t] = 1, j \leq \text{steps} \right\} \\
  \text{predicts}[i] &= 
  \begin{cases} 
  \text{candidates}[i, \text{accept\_token\_num}[i]-1], & \text{若 } \text{accept\_token\_num}[i] > 0 \\
  \text{基于目标模型分布重新采样}, & \text{若 } \text{accept\_token\_num}[i] = 0
  \end{cases}
  \end{aligned}
  $$
  其中元素的线性索引与二维位置（Batch 与 Step）的映射关系为：
  $$
  n = i \times \text{steps} + j
  $$
  各变量的维度与含义：
  - $\text{target\_probs} \in \mathbb{T}^{\text{bs} \times \text{steps} \times \text{vocab}}$：目标模型输出的概率分布张量，数据类型为$\mathbb{T}$（$\text{float32}$）
  - $\text{draft\_probs} \in \mathbb{T}^{\text{bs} \times \text{steps} \times \text{vocab}}$：草稿模型生成的概率分布张量，数据类型与$\text{target\_probs}$一致
  - $\text{candidates} \in \mathbb{I}^{\text{bs} \times \text{steps}}$：草稿生成的候选 Token 序列，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{predicts} \in \mathbb{D}^{\text{bs}}$：输出张量，存储最终决定的下一个 Token，数据类型为$\mathbb{D}$（$\text{int32}$）
  - $\text{accept\_token\_num} \in \mathbb{D}^{\text{bs}}$：输出张量，存储每条序列被验证接受的 Token 总数，数据类型为$\mathbb{D}$（$\text{int32}$）
  - $n$：中间计算时采样张量的线性索引，范围为$0 \leq n < \text{bs} \times \text{steps}$
  - $i$：张量的 Batch 索引，范围为$0 \leq i < \text{bs}$
  - $j$：投机采样步骤的索引，范围为$0 \leq j < \text{steps}$
  - $\text{bs}$：Batch Size，对应输入张量的第 0 维
  - $\text{steps}$：投机生成的候选长度（$\text{num\_draft\_tokens}$），$\text{target\_probs}$ 的第 1 维必须与此相等
  - $\text{vocab}$：词表大小（Vocab Size）
  - $\text{cand}$：单步候选 Token，$\text{cand} = \text{candidates}[i, j]$
  - $\text{accept\_bit}[i, j]$：单步验证结果标识（1 为接受，0 为拒绝）
### 参数说明
- **predicts** (torch.Tensor, 出参): 存储最终选定的下一个有效 Token，形状通常为 `[batch_size]`，数据类型必须为 **int32**
- **accept_index** (torch.Tensor, 出参): 存储被接受 Token 在草稿树中的原始索引，形状为 `[batch_size, num_spec_step]`，数据类型必须为 **int32**
- **accept_token_num** (torch.Tensor, 出参): 每个 Batch 序列中实际被接受的有效 Token 数量，形状为 `[batch_size]`，数据类型必须为 **int32**
- **candidates** (torch.Tensor, 入参): 草稿模型生成的候选 Token 张量，数据类型为 **int64**
- **retrive_index** (torch.Tensor, 入参): 用于在树结构中检索路径和兄弟节点的辅助索引张量，数据类型为 **int64**
- **retrive_next_token** (torch.Tensor, 入参): 用于在树结构中检索路径和兄弟节点的辅助索引张量，数据类型为 **int64**
- **retrive_next_sibling** (torch.Tensor, 入参): 用于在树结构中检索路径和兄弟节点的辅助索引张量，数据类型为 **int64**
- **target_predict** (torch.Tensor, 入参): 目标模型对所有草稿位置生成的贪婪预测 Token 张量，数据类型为 **int64**。其第二维大小必须与 `candidates` 严格一致
### 返回值
无返回值，计算结果直接写入 `predicts`、`accept_index` 和 `accept_token_num` 等出参张量中
### 约束与调用
- 所有输入输出张量必须位于同一个 CUDA/MACA 设备上
- 数据类型校验：
  - 出参张量（`predicts`, `accept_index`, `accept_token_num`）必须为 **int32**
  - 入参张量（`candidates`, `target_predict` 及索引类张量）必须为 **int64**
- 维度限制：目标预测张量（`target_predict`）与候选张量（`candidates`）的第二维（投机步数维度）必须相等
- 调用限制：尽管 C++ 底层支持 `cuda_stream` 参数，但由于 Python 接口注册定义了 8 个参数，调用时不支持传入 `cuda_stream`
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# Set computing device
device = "cuda"
# Core parameters
batch_size = 2
num_spec_step = 4
num_draft_tokens = 5
vocab_size = 128
# 1. Construct input tensors
# predicts: [batch_size], dtype=int32 (mutable output)
predicts = torch.zeros(batch_size, dtype=torch.int32, device=device)
# accept_index: [batch_size, num_spec_step], dtype=int32 (mutable output)
accept_index = torch.zeros((batch_size, num_spec_step), dtype=torch.int32, device=device)
# accept_token_num: [batch_size], dtype=int32 (mutable output)
accept_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)
# candidates: [batch_size, num_draft_tokens], dtype=int64
candidates = torch.randint(0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int64, device=device)
# retrieve indices: [batch_size, num_draft_tokens], dtype=int64
retrive_index = torch.zeros((batch_size, num_draft_tokens), dtype=torch.int64, device=device)
retrive_next_token = torch.zeros((batch_size, num_draft_tokens), dtype=torch.int64, device=device)
retrive_next_sibling = torch.zeros((batch_size, num_draft_tokens), dtype=torch.int64, device=device)
# target_predict: [batch_size, num_draft_tokens], dtype=int64
target_predict = torch.randint(0, vocab_size, (batch_size, num_draft_tokens), dtype=torch.int64, device=device)
# 2. Call the operator
# Removed 'cuda_stream' to match the 8-argument registration
torch.ops.sgl_kernel.verify_tree_greedy(
    predicts=predicts,
    accept_index=accept_index,
    accept_token_num=accept_token_num,
    candidates=candidates,
    retrive_index=retrive_index,
    retrive_next_token=retrive_next_token,
    retrive_next_sibling=retrive_next_sibling,
    target_predict=target_predict
)
# Output results
print("verify_tree_greedy computation completed successfully")
print(f"Updated predicts: {predicts}")
print(f"Updated accept_token_num: {accept_token_num}")
```
## 155. reconstruct_indices_from_tree_mask
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def reconstruct_indices_from_tree_mask(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int
) -> None
```
### 功能描述
reconstruct_indices_from_tree_mask 算子用于在树状投机采样（Tree-based Speculative Decoding）中，根据给定的布尔遮罩（tree_mask）重建 Token 的位置信息和索引元数据。该算子在 CUDA 内核中并行处理，根据遮罩关系计算每个草稿 Token 的树深度（Depth），并将其与已验证序列长度相加得到绝对位置。同时，它会识别每个 Token 在树结构中的检索索引、下一个 Token 索引以及下一个兄弟节点索引，这些元数据对于后续的投机验证至关重要。
- 计算公式：
  $$
  \begin{aligned}
  \text{positions}[n] &= \text{tree\_depth}[i, t] + \text{verified\_seq\_len}[i] \\
  \text{retrive\_index}[n] &= n \\
  \text{retrive\_next\_token}[n] &= 
  \begin{cases} 
  \text{first\_child\_idx}(i, t), & \text{若 Token } t \text{ 存在子节点} \\
  -1, & \text{若无子节点}
  \end{cases} \\
  \text{retrive\_next\_sibling}[n] &= 
  \begin{cases} 
  \text{next\_sibling\_idx}(i, t), & \text{若 Token } t \text{ 存在兄弟节点} \\
  -1, & \text{若无兄弟节点}
  \end{cases}
  \end{aligned}
  $$
  其中 $\text{first\_child\_idx}(i, t) = \arg\min_{k}\{k \mid \text{tree\_mask}[i, t, k] = \text{True}\}$，$\text{next\_sibling\_idx}(i, t) = \arg\min_{k>t}\{k \mid \text{tree\_mask}[i, \text{parent}(t), k] = \text{True}\}$；元素的线性索引与多维位置的映射关系为：
  $$
  n = i \times \text{draft\_token\_num} + t
  $$
  对于三维遮罩张量，其内部线性寻址关系为：
  $$
  \text{mask\_n} = i \times \text{draft\_token\_num} \times \text{draft\_token\_num} + j \times \text{draft\_token\_num} + k
  $$
  各变量的维度与含义：
  - $\text{tree\_mask} \in \mathbb{B}^{\text{batch\_size} \times \text{draft\_token\_num} \times \text{draft\_token\_num}}$：输入树结构关系遮罩，数据类型为$\mathbb{B}$（$\text{bool}$）；$\text{tree\_mask}[i, j, k] = \text{True}$ 表示在第$i$个 Batch 中，Token $k$ 是 Token $j$ 的后代
  - $\text{verified\_seq\_len} \in \mathbb{I}^{\text{batch\_size}}$：输入张量，记录每个 Batch 已验证的序列长度，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{positions} \in \mathbb{I}^{\text{batch\_size} \times \text{draft\_token\_num}}$：输出张量，存储重建后的 Token 绝对位置，计算方式为树深度加上已验证长度，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{retrive\_index} \in \mathbb{I}^{\text{batch\_size} \times \text{draft\_token\_num}}$：输出张量，存储 Token 的扁平化检索索引，直接映射为当前线性索引$n$，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{retrive\_next\_token} \in \mathbb{I}^{\text{batch\_size} \times \text{draft\_token\_num}}$：输出张量，存储每个 Token 的第一个子节点索引，若无子节点则为$-1$，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{retrive\_next\_sibling} \in \mathbb{I}^{\text{batch\_size} \times \text{draft\_token\_num}}$：输出张量，存储每个 Token 的下一个兄弟节点索引，若无兄弟节点则为$-1$，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $n$：输出张量元素的线性索引，范围为$0 \leq n < \text{batch\_size} \times \text{draft\_token\_num}$
  - $i$：Batch 索引，范围为$0 \leq i < \text{batch\_size}$
  - $t$：当前 Batch 内的 Token 偏移索引，范围为$0 \leq t < \text{draft\_token\_num}$
  - $j, k$：三维遮罩张量的 Token 索引，范围为$0 \leq j, k < \text{draft\_token\_num}$
  - $\text{draft\_token\_num}$：树中草稿 Token 的有效宽度
  - $\text{batch\_size}$：批处理的高度（行数）
  - $\text{tree\_depth}[i, t]$：第$i$个 Batch 中 Token $t$ 在树结构中的深度
  - $\text{parent}(t)$：Token $t$ 在树结构中的父节点索引
### 参数说明
- **tree_mask** (torch.Tensor, 入参): 树结构关系的布尔遮罩张量，形状为 `[batch_size, draft_token_num, draft_token_num]`，数据类型为 **bool**
- **verified_seq_len** (torch.Tensor, 入参): 存储每个 batch 当前已验证序列长度的张量，数据类型为 **int64**
- **positions** (torch.Tensor, 出参): 输出张量，存储重建后的 Token 绝对位置信息，数据类型为 **int64**
- **retrive_index** (torch.Tensor, 出参): 输出张量，存储每个 Token 的检索索引，数据类型为 **int64**
- **retrive_next_token** (torch.Tensor, 出参): 输出张量，存储对应的下一个 Token 索引，若无则为 -1，数据类型为 **int64**
- **retrive_next_sibling** (torch.Tensor, 出参): 输出张量，存储对应的下一个兄弟节点索引，若无则为 -1，数据类型为 **int64**
- **batch_size** (int, 入参): 批处理大小
- **draft_token_num** (int, 入参): 每个 batch 中的草稿 Token 数量
### 返回值
无返回值，计算结果直接写入 `positions`、`retrive_index`、`retrive_next_token` 和 `retrive_next_sibling` 张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 数据类型约束：
  - `tree_mask` 必须为 **bool**
  - 其余所有张量（入参及出参）必须为 **int64**
- 维度一致性：`tree_mask` 在内存中被视为大小为 `batch_size * draft_token_num * draft_token_num` 的连续数组进行访问
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda"
# 核心参数
bs = 2                  # Batch Size
draft_token_num = 16    # 树中草稿 Token 的数量
# 1. 构造输入张量
# tree_mask: [bs, draft_token_num, draft_token_num]，实现中强制转换为 bool*
tree_mask = torch.randint(
    0, 2, (bs, draft_token_num, draft_token_num), 
    dtype=torch.bool, device=device
)
# verified_seq_len: [bs]，存储每个 batch 已验证的序列长度，类型为 int64
verified_seq_len = torch.tensor([10, 20], dtype=torch.int64, device=device)
# 2. 构造输出/修改张量 (这些张量在 C++ 中被标记为 mutable/Tensor!)
# 数据类型均为 int64 (Long)
positions = torch.zeros((bs, draft_token_num), dtype=torch.int64, device=device)
retrive_index = torch.zeros((bs, draft_token_num), dtype=torch.int64, device=device)
retrive_next_token = torch.zeros((bs, draft_token_num), dtype=torch.int64, device=device)
retrive_next_sibling = torch.zeros((bs, draft_token_num), dtype=torch.int64, device=device)
# 3. 调用算子
# 注意参数顺序需匹配注册信息
torch.ops.sgl_kernel.reconstruct_indices_from_tree_mask(
    tree_mask,
    verified_seq_len,
    positions,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    bs,                 # batch_size
    draft_token_num     # draft_token_num
)
# 输出结果
print("reconstruct_indices_from_tree_mask computation completed")
print(f"Positions shape: {positions.shape}")
print(f"Retrieve Index shape: {retrive_index.shape}")
print(f"Next Token Index shape: {retrive_next_token.shape}")
print(f"Next Sibling Index shape: {retrive_next_sibling.shape}")
```
## 156. build_tree_kernel_efficient
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int
) -> None
```
### 功能描述
build_tree_kernel_efficient 算子用于在树状投机采样（Speculative Decoding）中高效构建树结构元数据。该算子根据输入的父节点列表，在 GPU 上并行计算每个草稿 Token 的树深度，并结合已验证序列长度生成绝对位置信息。同时，它负责生成树结构注意力掩码（tree_mask）以及用于验证阶段的路径检索索引（下一个 Token 索引和兄弟节点索引），从而减少树结构在 CPU 与 GPU 之间的频繁同步开销。
- 计算公式：
  $$
  \text{depth}[i, t] = 
  \begin{cases} 
  0, & \text{根节点（无父节点）} \\
  \text{depth}[i, \text{parent\_list}[i, t]] + 1, & \text{子节点（深度为父节点深度+1）}
  \end{cases}
  $$
  该算子通过遍历 parent_list 重建投机采样的树状拓扑结构。其中 $\text{depth}$ 代表节点在树中的层级（根节点深度为 0），用于计算 Token 的绝对位置。
  其中元素的线性索引与多维位置的映射关系为：
  $$
  n = i \times \text{draft\_token\_num} + t
  $$
  对于三维遮罩张量 tree_mask，其内部线性寻址关系为：
  $$
  \text{mask\_n} = i \times \text{draft\_token\_num} \times \text{draft\_token\_num} + j \times \text{draft\_token\_num} + k
  $$
  各变量的维度与含义：
  - $\text{parent\_list} \in \mathbb{I}^{\text{batch\_size} \times \text{draft\_token\_num}}$：输入张量，记录树中每个节点的父节点索引偏移，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{verified\_seq\_len} \in \mathbb{I}^{\text{batch\_size}}$：输入张量，记录每个 Batch 已验证的序列长度，用于计算绝对位置偏移，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{tree\_mask} \in \mathbb{B}^{\text{batch\_size} \times \text{draft\_token\_num} \times \text{draft\_token\_num}}$：输出张量，树结构注意力掩码，数据类型为$\mathbb{B}$（$\text{bool}$）
  - $\text{positions} \in \mathbb{I}^{\text{batch\_size} \times \text{draft\_token\_num}}$：输出张量，存储重建后的 Token 绝对位置信息，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{retrive\_next\_token} \in \mathbb{I}^{\text{batch\_size} \times \text{draft\_token\_num}}$：输出张量，存储每个 Token 对应的第一个子节点索引，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $\text{retrive\_next\_sibling} \in \mathbb{I}^{\text{batch\_size} \times \text{draft\_token\_num}}$：输出张量，存储每个 Token 对应的下一个兄弟节点索引，数据类型为$\mathbb{I}$（$\text{int64}$）
  - $n$：输出张量元素的线性索引，范围为$0 \leq n < \text{batch\_size} \times \text{draft\_token\_num}$
  - $i$：Batch 索引，范围为$0 \leq i < \text{batch\_size}$
  - $t$：当前 Batch 内的 Token 偏移索引，范围为$0 \leq t < \text{draft\_token\_num}$
  - $\text{draft\_token\_num}$：树中草稿 Token 的有效宽度
  - $\text{batch\_size}$：批处理的高度（行数）
### 参数说明
- **parent_list** (torch.Tensor, 入参): 树中每个节点的父节点偏移列表，形状通常为 `[batch_size, draft_token_num]`
- **selected_index** (torch.Tensor, 入参): 存储被选中的候选 Token 索引
- **verified_seq_len** (torch.Tensor, 入参): 每个 batch 当前已通过验证的序列长度
- **tree_mask** (torch.Tensor, 出参): 输出的树结构注意力掩码，用于后续 Self-Attention 计算
- **positions** (torch.Tensor, 出参): 输出的绝对位置编码信息
- **retrive_index** (torch.Tensor, 出参): 输出的 Token 检索索引
- **retrive_next_token** (torch.Tensor, 出参): 输出的当前节点第一个子节点的检索索引
- **retrive_next_sibling** (torch.Tensor, 出参): 输出的当前节点下一个兄弟节点的检索索引
- **topk** (int, 入参): 采样时每层的 top-k 候选数量
- **depth** (int, 入参): 树的最大深度
- **draft_token_num** (int, 入参): 树中包含的草稿 Token 总数
- **tree_mask_mode** (int, 入参): 掩码生成模式选择
### 返回值
无返回值，计算出的树掩码、位置信息和各类检索索引直接写入对应的出参张量中
### 约束与调用
- 所有输入及输出张量必须位于 **CUDA** 设备上
- 数据类型支持：
  - `tree_mask` 必须为 **bool** 类型
  - 索引与位置相关张量（`parent_list`, `positions`, `retrive_index` 等）必须为 **int64** (Long) 类型
- 内存布局：所有传入的张量应当是连续的（Contiguous），因为内核直接通过 `data_ptr()` 访问底层物理地址
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# Set computing device
device = "cuda"
# Core parameters
bs = 2                  # Batch size
topk = 4                # Top-k selection per branch
tree_depth = 5          # Depth of the speculative tree
draft_token_num = 16    # Total number of draft tokens in the tree
tree_mask_mode = 0      # Mode for tree mask generation
# 1. Construct input tensors
# Note: Dtypes must match C++ static_cast expectations (mostly int64 and bool)
# parent_list: Indices of parent nodes for each node in the tree
parent_list = torch.zeros((bs, draft_token_num), dtype=torch.int64, device=device)
# selected_index: Indices of selected tokens
selected_index = torch.zeros((bs, draft_token_num), dtype=torch.int64, device=device)
# verified_seq_len: Current verified sequence length per batch
verified_seq_len = torch.tensor([10, 20], dtype=torch.int64, device=device)
# 2. Construct output/mutable tensors (marked with ! in registration)
# tree_mask: [bs, draft_token_num, draft_token_num]
tree_mask = torch.empty((bs, draft_token_num, draft_token_num), dtype=torch.bool, device=device)
# positions and indices: [bs, draft_token_num]
positions = torch.empty((bs, draft_token_num), dtype=torch.int64, device=device)
retrive_index = torch.empty((bs, draft_token_num), dtype=torch.int64, device=device)
retrive_next_token = torch.empty((bs, draft_token_num), dtype=torch.int64, device=device)
retrive_next_sibling = torch.empty((bs, draft_token_num), dtype=torch.int64, device=device)
# 3. Call the operator
# Registered in sgl_kernel namespace
torch.ops.sgl_kernel.build_tree_kernel_efficient(
    parent_list=parent_list,
    selected_index=selected_index,
    verified_seq_len=verified_seq_len,
    tree_mask=tree_mask,
    positions=positions,
    retrive_index=retrive_index,
    retrive_next_token=retrive_next_token,
    retrive_next_sibling=retrive_next_sibling,
    topk=topk,
    depth=tree_depth,
    draft_token_num=draft_token_num,
    tree_mask_mode=tree_mask_mode
)
# Output results
print("build_tree_kernel_efficient computation completed")
print(f"tree_mask shape: {tree_mask.shape}")
print(f"positions shape: {positions.shape}")
print(f"retrive_index sample: {retrive_index[0, :5]}")
```
## 157. segment_packbits
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def segment_packbits(
    x: torch.Tensor,
    input_indptr: torch.Tensor,
    output_indptr: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    cuda_stream: int = 0
) -> None
```
### 功能描述
segment_packbits 算子实现了分段位打包（Bit-packing）功能。它将输入的布尔型张量按照指定的段偏移指针（indptr）划分为多个段，并在每个段内将连续的布尔值压缩为比特位（1 个比特位代表 1 个布尔值），最终以 uint8 字节流的形式存储。该算子在位打包时采用小端序（Little-endian）逻辑，常用于树状投机采样等变长序列场景中，以降低布尔掩码的显存占用和传输带宽。
- 计算公式：
  $$
  \text{dst}[\text{output\_indptr}[b] + \text{byte\_off}] |= \text{int}(\text{input}[\text{input\_indptr}[b] + 8 \times \text{byte\_off} + \text{bit\_idx}]) \ll \text{bit\_idx}
  $$
  其中，$\text{int}(\cdot)$ 将布尔值转换为整数（$\text{True}$ 映射为$1$，$\text{False}$ 映射为$0$），且位打包遵循小端序（Little-endian）逻辑。
  其中元素的线性索引与多维位置的映射关系为：
  $$
  \text{input\_n} = \text{input\_indptr}[b] + 8 \times \text{byte\_off} + \text{bit\_idx} \\
  \text{output\_n} = \text{output\_indptr}[b] + \text{byte\_off}
  $$
  各变量的维度与含义：
  - $\text{input} \in \mathbb{B}^{\text{input\_len}}$：输入布尔张量，数据类型为$\mathbb{B}$（$\text{bool}$），包含待压缩的原始数据
  - $\text{dst} \in \mathbb{U}^{\text{output\_len}}$：输出张量，存储打包后的比特数据，数据类型为$\mathbb{U}$（$\text{uint8}$）
  - $\text{input\_indptr} \in \mathbb{I}^{\text{num\_segments}+1}$：输入段偏移指针，用于定义输入张量$\text{input}$中每个 Segment 的起始和结束位置，数据类型为$\mathbb{I}$（$\text{int32}$）
  - $\text{output\_indptr} \in \mathbb{I}^{\text{num\_segments}+1}$：输出段偏移指针，用于定义输出张量$\text{dst}$中每个 Segment 的起始和结束位置，数据类型与$\text{input\_indptr}$一致
  - $b$：分段索引（Batch 索引），范围为$0 \leq b < \text{num\_segments}$
  - $\text{byte\_off}$：当前 Segment 内的字节偏移，范围为$0 \leq \text{byte\_off} < \lfloor (\text{input\_indptr}[b+1] - \text{input\_indptr}[b])/8 \rfloor$
  - $\text{bit\_idx}$：字节内的比特位索引，范围为$0 \leq \text{bit\_idx} < 8$
  - $\text{num\_segments}$：需要处理的总分段数量
  - $\text{input\_len}$：输入张量的总长度，满足$\text{input\_len} = \text{input\_indptr}[\text{num\_segments}]$
  - $\text{output\_len}$：输出张量的总长度，满足$\text{output\_len} = \text{output\_indptr}[\text{num\_segments}]$
### 参数说明
- **x** (torch.Tensor, 入参): 输入的布尔型张量，包含待压缩的原始布尔数据
- **input_indptr** (torch.Tensor, 入参): 输入段偏移指针，用于定义 `x` 中各段的起始和结束位置，数据类型为 **int32**
- **output_indptr** (torch.Tensor, 入参): 输出段偏移指针，用于定义压缩后各段在 `y` 中的起始和结束位置，数据类型为 **int32**
- **y** (torch.Tensor, 出参): 输出张量，用于存储打包后的比特数据，数据类型为 **uint8**
- **batch_size** (int, 入参): 批处理大小，即总共需要处理的分段数量
- **cuda_stream** (int, 入参): 异步执行任务的 CUDA 流指针
### 返回值
无返回值，打包后的结果直接写入输出张量 `y` 中
### 约束与调用
- 所有输入输出张量必须位于同一个 CUDA 设备上
- 数据类型限制：
  - `x` 必须为 **bool** 类型
  - `y` 必须为 **uint8** 类型
  - `input_indptr` 与 `output_indptr` 必须为 **int32** 类型
- `output_indptr` 的长度必须大于等于 `batch_size + 1`
- 算子内部直接通过 `data_ptr()` 访问地址，建议所有张量保持内存连续（Contiguous）
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# Set computing device
device = "cuda"
# Core parameters
batch_size = 2
bits_per_segment = 16  # Each segment has 16 boolean flags
total_bits = batch_size * bits_per_segment
# 1. Construct input tensors
# x: Input boolean tensor to be packed
x = torch.randint(0, 2, (total_bits,), dtype=torch.bool, device=device)
# input_indptr: Defines the start/end indices of segments in input 'x'
# Must be int32 and size should be batch_size + 1
input_indptr = torch.tensor([0, 16, 32], dtype=torch.int32, device=device)
# output_indptr: Defines the start/end indices of packed segments in output 'y'
# 16 bits pack into 2 bytes (uint8)
output_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
# 2. Construct output tensor
# y: Output tensor for packed bits, must be uint8 (or int8)
y = torch.empty((4,), dtype=torch.uint8, device=device)
cuda_stream = 0
# 3. Call the operator
# Registered in sgl_kernel namespace with 6 arguments
torch.ops.sgl_kernel.segment_packbits(
    x=x,
    input_indptr=input_indptr,
    output_indptr=output_indptr,
    y=y,
    batch_size=batch_size,
    cuda_stream=cuda_stream
)
# Output results
print("segment_packbits computation completed")
print(f"Input boolean tensor shape: {x.shape}")
print(f"Output packed tensor (uint8) shape: {y.shape}")
```
## 158. transfer_kv_per_layer
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
transfer_kv_per_layer 算子用于在单层维度上，根据指定的源索引和目标索引，将 Key (K) 和 Value (V) 的 Cache 数据从源张量复制到目标张量中。该操作主要应用于大模型推理过程中的 KV Cache 显存管理场景，如 Page Attention 中的物理块搬运、碎片整理等。
- 计算公式：
  $$
  \text{dst\_k}[\mathcal{A}_{\text{dst}}] = \text{src\_k}[\mathcal{A}_{\text{src}}]
  $$
  $$
  \text{dst\_v}[\mathcal{A}_{\text{dst}}] = \text{src\_v}[\mathcal{A}_{\text{src}}]
  $$
  其中地址偏移量的映射关系为：
  $$
  \mathcal{A}_{\text{src}} = \text{src\_indices}[b] \times \text{item\_size} + j
  $$
  $$
  \mathcal{A}_{\text{dst}} = \text{dst\_indices}[b] \times \text{item\_size} + j
  $$
  各变量的维度与含义：
  - $\text{src\_k} \in \mathbb{T}^{\text{total\_size}}$：源 Key 缓存张量，视为一维线性内存，数据类型为$\mathbb{T}$
  - $\text{dst\_k} \in \mathbb{T}^{\text{total\_size}}$：目标 Key 缓存张量，数据类型与$\text{src\_k}$一致
  - $\text{src\_v} \in \mathbb{T}^{\text{total\_size}}$：源 Value 缓存张量，视为一维线性内存
  - $\text{dst\_v} \in \mathbb{T}^{\text{total\_size}}$：目标 Value 缓存张量，数据类型与$\text{src\_v}$一致
  - $\text{src\_indices} \in \mathbb{Z}^{\text{num\_blocks}}$：源数据块索引张量，数据类型为整型
  - $\text{dst\_indices} \in \mathbb{Z}^{\text{num\_blocks}}$：目标数据块索引张量，长度与$\text{src\_indices}$一致
  - $b$：数据块的遍历索引，范围为$0 \leq b < \text{num\_blocks}$（其中$\text{num\_blocks}$为索引张量的长度）
  - $j$：块内元素的字节或线性偏移量，范围为$0 \leq j < \text{item\_size}$
  - $\text{item\_size}$：单个数据块（Page/Slot）的内存大小（单位通常为字节或元素个数）
  - $\mathcal{A}_{\text{src}}$：计算得到的源数据在显存中的绝对线性偏移量
  - $\mathcal{A}_{\text{dst}}$：计算得到的目标数据在显存中的绝对线性偏移量
### 参数说明
- **src_k** (torch.Tensor, 入参): 源 Key Cache 张量，待读取的 K Cache 数据存储于此。
- **dst_k** (torch.Tensor, 出参): 目标 Key Cache 张量，复制后的 K Cache 数据将写入此处。
- **src_v** (torch.Tensor, 入参): 源 Value Cache 张量，待读取的 V Cache 数据存储于此。
- **dst_v** (torch.Tensor, 出参): 目标 Value Cache 张量，复制后的 V Cache 数据将写入此处。
- **src_indices** (torch.Tensor, 入参): 源数据块索引张量，用于指定需要搬运的源数据位置。
- **dst_indices** (torch.Tensor, 入参): 目标数据块索引张量，用于指定搬运后数据的存储位置。
- **item_size** (int, 入参): 单个数据块的大小（通常为`head_dim * sizeof(dtype)`或页大小），用于计算内存偏移量。
- **block_quota** (int, 入参): CUDA Kernel 发射时的 Block 配额，用于控制计算并行度。
- **num_warps_per_block** (int, 入参): 每个 CUDA Block 使用的 Warp 数量。
### 返回值
无返回值，计算结果直接写入`dst_k`和`dst_v`张量中。
### 约束与调用
- 所有输入张量必须部署在 CUDA 设备上。
- `src_indices`和`dst_indices`的元素数量必须一致，且数据类型均为整型。
- `src_k`与`dst_k`的数据类型及张量形状需兼容，`src_v`与`dst_v`的数据类型及张量形状也需兼容。
- `item_size`必须准确反映待搬运的连续内存块大小，否则会导致数据错位。
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "transfer_kv_per_layer only supports CUDA devices"
# 核心参数
num_blocks_to_copy = 32      
item_size = 4096              
total_pool_blocks = 1024      
block_quota = 128            
num_warps_per_block = 4      
# 1. 构造输入张量
src_k = torch.randn(
    total_pool_blocks * item_size,
    dtype=torch.float32, device=device, requires_grad=False
)
src_v = torch.randn(
    total_pool_blocks * item_size,
    dtype=torch.float32, device=device, requires_grad=False
)
# 构造目标 Key/Value Cache 池 (初始化为空或零)
dst_k = torch.zeros(
    total_pool_blocks * item_size,
    dtype=torch.float32, device=device, requires_grad=False
)
dst_v = torch.zeros(
    total_pool_blocks * item_size,
    dtype=torch.float32, device=device, requires_grad=False
)
# 构造索引张量 (指定从哪里搬运到哪里)
src_indices = torch.randint(
    0, total_pool_blocks, (num_blocks_to_copy,),
    dtype=torch.int64, device=device, requires_grad=False
)
dst_indices = torch.randint(
    0, total_pool_blocks, (num_blocks_to_copy,),
    dtype=torch.int64, device=device, requires_grad=False
)
# 调用算子
torch.ops.sgl_kernel.transfer_kv_per_layer(
    src_k=src_k,
    dst_k=dst_k,
    src_v=src_v,
    dst_v=dst_v,
    src_indices=src_indices,
    dst_indices=dst_indices,
    item_size=item_size,
    block_quota=block_quota,
    num_warps_per_block=num_warps_per_block
)
# 输出结果
print("transfer_kv_per_layer computation completed")
print(f"Source K shape: {src_k.shape}")
print(f"Destination K shape: {dst_k.shape}")
print(f"Number of blocks transferred: {src_indices.shape[0]}")
```
## 159. transfer_kv_per_layer_pf_lf
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
transfer_kv_per_layer_pf_lf 算子实现了针对特定层（Per Layer）的 KV Cache 数据搬运与格式转换。该算子根据给定的源索引（src_indices）和目标索引（dst_indices），将 Key 和 Value 张量从源存储位置传输到目标存储位置。后缀 `pf_lf` 通常指代特定的页面格式（Page Format）或布局格式（Layout Format）的处理，用于优化推理过程中的显存管理和访问效率。
- 计算公式：
  $$
  \text{dst\_k}[\text{dst\_addr}] = \text{src\_k}[\text{src\_addr}]
  $$
  $$
  \text{dst\_v}[\text{dst\_addr}] = \text{src\_v}[\text{src\_addr}]
  $$
  其中源地址索引与目标地址索引的映射关系为：
  $$
  \text{src\_addr} = \text{layer\_id} \times \text{src\_layout\_dim} + \text{src\_indices}[i] \times \text{item\_size} + j
  $$
  $$
  \text{dst\_addr} = \text{dst\_indices}[i] \times \text{item\_size} + j
  $$
  各变量的维度与含义：
  - $\text{src\_k}, \text{src\_v} \in \mathbb{T}^{\text{Total\_Src} \times \text{item\_size}}$：源键值张量，数据类型为$\mathbb{T}$。
  - $\text{dst\_k}, \text{dst\_v} \in \mathbb{T}^{\text{Total\_Dst} \times \text{item\_size}}$：目标键值张量，数据类型与源张量一致。
  - $\text{src\_indices} \in \mathbb{Z}^{N}$：源数据索引张量，包含$N$个待搬运块/页的索引值。
  - $\text{dst\_indices} \in \mathbb{Z}^{N}$：目标数据索引张量，包含$N$个目标位置的索引值。
  - $i$：索引张量的遍历下标，范围为$0 \leq i < N$，其中$N$为本次传输涉及的索引数量（即`indices`张量的长度）。
  - $j$：数据块内的元素偏移，范围为$0 \leq j < \text{item\_size}$。
  - $\text{layer\_id} \in \mathbb{Z}$：当前处理的层级 ID，用于在源数据中定位特定层。
  - $\text{src\_layout\_dim} \in \mathbb{Z}$：源数据的布局维度跨度（Stride），用于计算层级偏移量。通常表示不同层之间的数据间隔。
  - $\text{item\_size} \in \mathbb{Z}$：单个 KV 数据块（或 Head 维度）的元素数量。
### 参数说明
- **src_k** (torch.Tensor, 入参): 源 Key 张量，包含待传输的 K 数据。
- **dst_k** (torch.Tensor, 入参/出参): 目标 Key 张量，数据将被写入此位置。
- **src_v** (torch.Tensor, 入参): 源 Value 张量，包含待传输的 V 数据。
- **dst_v** (torch.Tensor, 入参/出参): 目标 Value 张量，数据将被写入此位置。
- **src_indices** (torch.Tensor, 入参): 源数据索引张量，指定从源张量中读取数据的位置。
- **dst_indices** (torch.Tensor, 入参): 目标数据索引张量，指定写入目标张量的位置。
- **layer_id** (int, 入参): 当前操作的层 ID，用于计算偏移或定位特定层的数据。
- **item_size** (int, 入参): 单个元素的大小或隐藏层维度大小（Hidden Size）。
- **src_layout_dim** (int, 入参): 源数据的布局维度参数，用于处理内存跨度。
- **block_quota** (int, 入参): CUDA 计算块的配额限制，用于资源控制。
- **num_warps_per_block** (int, 入参): 每个 CUDA Block 中使用的 Warp 数量，用于调优并行度。
### 返回值
无返回值，计算结果直接更新到`dst_k`和`dst_v`张量中。
### 约束与调用
- 所有输入张量通常需位于 CUDA 设备上。
- `src_indices`和`dst_indices`的长度应保持逻辑一致，确保搬运的数据量匹配。
- `dst_k`和`dst_v`必须预先分配足够的空间以容纳传输的数据。
- `block_quota`和`num_warps_per_block`需根据具体硬件架构设置合理的数值以获得最佳性能。
### 调用示例
```python
import torch
import mcoplib.sgl_kernel 
# 设置计算设备
device = "cuda"
# 核心参数
num_blocks = 32           
item_size = 128        
layer_id = 5           
src_layout_dim = 4096   
block_quota = 64       
num_warps = 4             
# 1. 构造输入张量
# Source Tensors: 模拟包含多层数据的源显存池
# 大小通常需要覆盖 layer_id * src_layout_dim 的偏移范围
total_src_size = src_layout_dim * (layer_id + 1) + 2048
src_k = torch.randn(
    total_src_size, item_size,
    dtype=torch.float16, device=device, requires_grad=False
)
src_v = torch.randn(
    total_src_size, item_size,
    dtype=torch.float16, device=device, requires_grad=False
)
# Destination Tensors: 模拟目标 KV Cache
total_dst_size = 10000
dst_k = torch.zeros(
    total_dst_size, item_size,
    dtype=torch.float16, device=device, requires_grad=False
)
dst_v = torch.zeros(
    total_dst_size, item_size,
    dtype=torch.float16, device=device, requires_grad=False
)
# Indices: 构造搬运索引
# src_indices 指向源数据中的相对偏移
src_indices = torch.randint(
    0, 2048, (num_blocks,),
    dtype=torch.int64, device=device
)
# dst_indices 指向目标 Cache 中的物理页/块位置
dst_indices = torch.randint(
    0, total_dst_size, (num_blocks,),
    dtype=torch.int64, device=device
)
# 调用算子
torch.ops.sgl_kernel.transfer_kv_per_layer_pf_lf(
    src_k,
    dst_k,
    src_v,
    dst_v,
    src_indices,
    dst_indices,
    layer_id,
    item_size,
    src_layout_dim,
    block_quota,
    num_warps
)
# 输出结果
print("transfer_kv_per_layer_pf_lf computation completed")
print(f"Destination Key buffer shape: {dst_k.shape}")
print(f"Destination Value buffer shape: {dst_v.shape}")
```
## 160. transfer_kv_per_layer_ph_lf
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
transfer_kv_per_layer_ph_lf 算子实现了 Transformer 模型指定层中键（K）值（V）张量的跨位置传输操作，基于源/目标索引映射关系，将源 KV 张量（src_k/src_v）中指定位置的数据迁移至目标 KV 张量（dst_k/dst_v）的对应位置，适配布局维度、页大小、注意力头数等参数，同时遵循块配额、warp 数量等 CUDA 计算资源约束。
- 计算公式：
  $$
  \text{dst\_k}[\text{dst\_idx}] = \text{src\_k}[\text{src\_idx}], \quad \text{dst\_v}[\text{dst\_idx}] = \text{src\_v}[\text{src\_idx}]
  $$
  其中源数据与目标数据的线性索引计算为：
  $$
  \text{src\_idx} = \text{layer\_id} \times \text{src\_layout\_dim} + \text{src\_indices}[i] \times \text{block\_size} + j
  $$
  $$
  \text{dst\_idx} = \text{dst\_indices}[i] \times \text{block\_size} + j
  $$
  元素的线性索引映射关系为：
  $$
  n = \text{page\_idx} \times \text{block\_size} + j
  $$
  式中：
  $$
  0 \leq i < N, \quad 0 \leq j < \text{block\_size}, \quad N = \text{len}(\text{src\_indices}) = \text{len}(\text{dst\_indices})
  $$
  $$
  \text{block\_size} = \text{head\_num} \times \text{page\_size} \times \text{item\_size}
  $$
  各变量的维度与含义：
  - $\text{src\_k} \in \mathbb{T}^{*}$：源输入 K 张量，包含所有层的 K 数据，数据类型为$\mathbb{T}$
  - $\text{src\_v} \in \mathbb{T}^{*}$：源输入 V 张量，包含所有层的 V 数据，数据类型与$\text{src\_k}$一致
  - $\text{dst\_k} \in \mathbb{T}^{*}$：目标输出 K 张量（通常为 Paged Attention 的显存池），数据类型与$\text{src\_k}$一致
  - $\text{dst\_v} \in \mathbb{T}^{*}$：目标输出 V 张量（通常为 Paged Attention 的显存池），数据类型与$\text{src\_k}$一致
  - $\text{block\_size} \in \mathbb{Z}$：单个显存页/块包含的元素总数，计算方式为$\text{block\_size} = \text{head\_num} \times \text{page\_size} \times \text{item\_size}$
  - $i \in \mathbb{Z}$：块/页索引遍历下标，范围为$0 \leq i < N$，其中$N$为$\text{src\_indices}$（或$\text{dst\_indices}$）张量的长度（即本次搬运的页数）
  - $j \in \mathbb{Z}$：块内元素的线性偏移，范围为$0 \leq j < \text{block\_size}$
  - $\text{src\_indices} \in \mathbb{Z}^{N}$：源数据的页/块索引张量，维度为$N$（$N$为本次搬运的页数）
  - $\text{dst\_indices} \in \mathbb{Z}^{N}$：目标数据的页/块索引张量（物理页地址），维度为$N$
  - $\text{layer\_id} \in \mathbb{Z}$：当前操作的目标层 ID
  - $\text{src\_layout\_dim} \in \mathbb{Z}$：源数据中层与层之间的内存步长（Stride），通常以元素个数为单位
  - $\text{page\_size} \in \mathbb{Z}$：每个分页包含的 Token 数量（槽位大小）
  - $\text{head\_num} \in \mathbb{Z}$：注意力头（Attention Heads）的数量
  - $\text{item\_size} \in \mathbb{Z}$：单个 Token 的特征维度大小（通常为 Head Dimension）
  - $\text{src\_idx} \in \mathbb{Z}$：源张量中元素的线性索引
  - $\text{dst\_idx} \in \mathbb{Z}$：目标张量中元素的线性索引
  - $n \in \mathbb{Z}$：张量元素的通用线性索引，范围为$0 \leq n < \text{block\_size} \times N$（本次搬运的总元素数）
  - $\text{page\_idx} \in \mathbb{Z}$：页/块索引（可取$\text{src\_indices}[i]$或$\text{dst\_indices}[i]$）
  - $N \in \mathbb{Z}$：本次搬运的页数，即$\text{src\_indices}$/$\text{dst\_indices}$张量的长度
### 参数说明
- **src_k** (torch.Tensor, 入参): 源键张量，提供待传输的键数据
- **dst_k** (torch.Tensor, 入参/出参): 目标键张量，接收源键数据，传输完成后会被结果覆盖
- **src_v** (torch.Tensor, 入参): 源值张量，提供待传输的值数据
- **dst_v** (torch.Tensor, 入参/出参): 目标值张量，接收源值数据，传输完成后会被结果覆盖
- **src_indices** (torch.Tensor, 入参): 源索引张量，指定 src_k/src_v 中参与传输的数据位置索引
- **dst_indices** (torch.Tensor, 入参): 目标索引张量，指定 dst_k/dst_v 中接收数据的位置索引
- **layer_id** (int, 入参): 层编号，标识当前执行 KV 传输的 Transformer 层 ID
- **item_size** (int, 入参): 单个 KV 数据项的尺寸，定义基础数据单元大小
- **src_layout_dim** (int, 入参): 源 KV 张量的布局维度，描述张量的维度组织方式
- **page_size** (int, 入参): 内存页大小，适配分页存储的尺寸参数
- **head_num** (int, 入参): 注意力头数量，匹配 Transformer 模型的注意力头配置
- **block_quota** (int, 入参): 计算块资源配额，限制单个 CUDA 块可使用的资源上限
- **num_warps_per_block** (int, 入参): 每个 CUDA 计算块的 warp 数量，指定硬件执行单元的组织方式
### 返回值
无返回值，KV 数据传输结果直接写入 dst_k 和 dst_v 张量中
### 约束与调用
- 所有张量必须部署在 CUDA 设备上
- src_indices/dst_indices 的元素值需在对应 KV 张量的有效索引范围内
- page_size、item_size、src_layout_dim 需为正整数且维度匹配
- head_num 需为正整数，符合 Transformer 模型的注意力头数量设计规范
- block_quota 和 num_warps_per_block 需满足 CUDA 硬件的计算块资源限制
- 支持的数据类型：float16, bfloat16, float32
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda"
# 核心参数
num_blocks = 16            # 本次搬运的页/块数量 (Indices length)
layer_id = 3               # 当前处理的层级 ID (Layer ID)
item_size = 128            # 单个元素的维度大小 (Head Dim)
page_size = 16             # 每个页包含的 Token 数 (Page Size)
head_num = 8               # 注意力头数 (Head Num)
src_layout_dim = 100000    # 源数据层间跨度 (Layer Stride)
block_quota = 64           # CUDA Block 配额
num_warps = 4              # 每个 Block 的 Warp 数量
# 计算单个 Page/Block 对应的行数或元素偏移 (用于构造张量形状)
# 假设数据是平铺存储，每个 Block 包含 head_num * page_size 个 item_size 向量
block_elem_count = page_size * head_num 
# 1. 构造输入张量
# Source Tensors: 模拟源显存池，需覆盖 layer_id 的偏移量
# 形状足够大以容纳 src_layout_dim * (layer_id + 1)
total_src_rows = src_layout_dim * (layer_id + 2) 
src_k = torch.randn(
    total_src_rows, item_size,
    dtype=torch.float16, device=device, requires_grad=False
)
src_v = torch.randn(
    total_src_rows, item_size,
    dtype=torch.float16, device=device, requires_grad=False
)
# Destination Tensors: 模拟目标 Paged Attention 显存池
# 假设总容量为 1024 个物理页
max_physical_pages = 1024
total_dst_rows = max_physical_pages * block_elem_count
dst_k = torch.zeros(
    total_dst_rows, item_size,
    dtype=torch.float16, device=device, requires_grad=False
)
dst_v = torch.zeros(
    total_dst_rows, item_size,
    dtype=torch.float16, device=device, requires_grad=False
)
# Indices: 构造索引张量
# src_indices: 源数据中的逻辑块索引
src_indices = torch.arange(
    0, num_blocks, 
    dtype=torch.int64, device=device
)
# dst_indices: 目标显存池中的物理块索引 (随机指定位置)
dst_indices = torch.randint(
    0, max_physical_pages, (num_blocks,),
    dtype=torch.int64, device=device
)
# 调用算子
torch.ops.sgl_kernel.transfer_kv_per_layer_ph_lf(
    src_k=src_k,
    dst_k=dst_k,
    src_v=src_v,
    dst_v=dst_v,
    src_indices=src_indices,
    dst_indices=dst_indices,
    layer_id=layer_id,
    item_size=item_size,
    src_layout_dim=src_layout_dim,
    page_size=page_size,
    head_num=head_num,
    block_quota=block_quota,
    num_warps_per_block=num_warps
)
# 输出结果
print("transfer_kv_per_layer_ph_lf computation completed")
print(f"Destination Key buffer shape: {dst_k.shape}")
print(f"Destination Value buffer shape: {dst_v.shape}")
print(f"Transferred blocks count: {num_blocks}")
```
## 161. transfer_kv_all_layer
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
transfer_kv_all_layer 算子实现了跨所有层的 KV Cache（键值缓存）数据传输操作。该算子根据指定的源索引（`src_indices`）和目标索引（`dst_indices`），将 Key 和 Value 张量数据从源存储（`src`）复制到目标存储（`dst`）。这通常用于大模型推理过程中的显存管理、分页注意力机制（Paged Attention）或缓存整理。
- 计算公式：
  对于任意层索引 $l$（$0 \leq l < \text{num\_layers}$）和任意索引位置 $idx$（$0 \leq idx < \text{block\_quota}$），有：
  $$
  \begin{cases}
  \text{dst\_k\_layers}[l][\text{dst\_indices}[idx]] = \text{src\_k\_layers}[l][\text{src\_indices}[idx]] \\
  \text{dst\_v\_layers}[l][\text{dst\_indices}[idx]] = \text{src\_v\_layers}[l][\text{src\_indices}[idx]]
  \end{cases}
  $$
  其中，单个 layer 内张量元素的线性索引与二维位置的映射关系为：
  $$
  n = s \times \text{item\_size} + f
  $$
  各变量的维度与含义：
  - $\text{src\_k\_layers} \in \mathbb{T}^{\text{num\_layers} \times S \times \text{item\_size}}$：源 Key 张量集合，数据类型为$\mathbb{T}$，包含$\text{num\_layers}$个 layer 的 Key 张量，每个 layer 的 Key 张量维度为 $S \times \text{item\_size}$（$S$为源序列长度）
  - $\text{src\_v\_layers} \in \mathbb{T}^{\text{num\_layers} \times S \times \text{item\_size}}$：源 Value 张量集合，数据类型与$\text{src\_k\_layers}$一致
  - $\text{dst\_k\_layers} \in \mathbb{T}^{\text{num\_layers} \times D \times \text{item\_size}}$：目标 Key 张量集合，数据类型与$\text{src\_k\_layers}$一致，包含$\text{num\_layers}$个 layer 的 Key 张量，每个 layer 的 Key 张量维度为 $D \times \text{item\_size}$（$D$为目标序列长度）
  - $\text{dst\_v\_layers} \in \mathbb{T}^{\text{num\_layers} \times D \times \text{item\_size}}$：目标 Value 张量集合，数据类型与$\text{src\_v\_layers}$一致
  - $\text{src\_indices} \in \mathbb{Z}^{\text{block\_quota}}$：源张量的索引列表，每个元素对应$\text{src\_k\_layers}/\text{src\_v\_layers}$单个 layer 内的线性索引
  - $\text{dst\_indices} \in \mathbb{Z}^{\text{block\_quota}}$：目标张量的索引列表，每个元素对应$\text{dst\_k\_layers}/\text{dst\_v\_layers}$单个 layer 内的线性索引
  - $\text{item\_size} \in \mathbb{Z}^+$：每个 Key/Value 元素的特征维度大小
  - $\text{num\_layers} \in \mathbb{Z}^+$：张量集合包含的 layer 数量
  - $\text{block\_quota} \in \mathbb{Z}^+$：单次传输的索引数量（即$\text{src\_indices}/\text{dst\_indices}$的长度）
  - $\text{num\_warps\_per\_block}$：CUDA 核函数的 warp 配置参数（不参与计算逻辑）
  - $n$：单个 layer 内张量元素的线性索引，范围为 $0 \leq n < S$（源）或 $0 \leq n < D$（目标）
  - $s$：单个 layer 内张量的序列维度索引，范围为 $0 \leq s < S$（源）或 $0 \leq s < D$（目标）
  - $f$：单个 layer 内张量的特征维度索引，范围为 $0 \leq f < \text{item\_size}$
### 参数说明
- **src_k_layers** (torch.Tensor, 入参): 源 Key 缓存张量，包含所有层的数据。
- **dst_k_layers** (torch.Tensor, 入参/出参): 目标 Key 缓存张量，数据将被写入此张量。
- **src_v_layers** (torch.Tensor, 入参): 源 Value 缓存张量，包含所有层的数据。
- **dst_v_layers** (torch.Tensor, 入参/出参): 目标 Value 缓存张量，数据将被写入此张量。
- **src_indices** (torch.Tensor, 入参): 源数据的索引张量，指示从哪里读取数据。
- **dst_indices** (torch.Tensor, 入参): 目标数据的索引张量，指示将数据写入何处。
- **item_size** (int, 入参): 单个数据项的大小（通常对应 hidden_dim 或 head_dim 的字节大小或元素数量）。
- **num_layers** (int, 入参): 模型 Transformer 层的总数。
- **block_quota** (int, 入参): CUDA 核心计算时的块配额或限制参数。
- **num_warps_per_block** (int, 入参): 每个 CUDA 线程块（Block）中使用的 Warp 数量，用于配置并行计算粒度。
### 返回值
无返回值，计算结果直接写入 `dst_k_layers` 和 `dst_v_layers` 张量中。
### 约束与调用
- 设备限制: 所有输入张量必须位于 CUDA 设备上（由 `m.impl(..., torch::kCUDA, ...)` 决定）。
- 数据一致性: `src` 和 `dst` 的张量维度应由 `item_size` 和 `num_layers` 及其对应的索引逻辑保证一致。
- 底层实现: 该函数通过 PyTorch 的 C++ 扩展绑定到 CUDA 核心实现，调用时需确保传入正确的 CUDA 上下文。
### 调用示例
```python
import torch
import numpy as np
import mcoplib.sgl_kernel
# Set computation device
device = "cuda"
# Core parameters
num_layers = 2
capacity = 1024
item_size = 128
idx_len = 16
block_quota = 8
num_warps = 4
# 1. Construct Data Tensors (Float16)
src_k = torch.randn(num_layers, capacity, item_size, dtype=torch.float16, device=device)
dst_k = torch.zeros_like(src_k)
src_v = torch.randn_like(src_k)
dst_v = torch.zeros_like(src_k)
# 2. Construct Pointer Tensors (UInt64)
# The operator requires tensors containing memory addresses (pointers)
def to_uint64_ptr(t):
    return torch.tensor(
        np.array([sub.data_ptr() for sub in t], dtype=np.uint64),
        device=device, dtype=torch.uint64
    )
src_k_ptrs = to_uint64_ptr(src_k)
dst_k_ptrs = to_uint64_ptr(dst_k)
src_v_ptrs = to_uint64_ptr(src_v)
dst_v_ptrs = to_uint64_ptr(dst_v)
# 3. Construct Index Tensors
src_idx = torch.arange(idx_len, device=device, dtype=torch.long)
dst_idx = torch.arange(idx_len, device=device, dtype=torch.long)
# Call the operator
torch.ops.sgl_kernel.transfer_kv_all_layer(
    src_k_ptrs,
    dst_k_ptrs,
    src_v_ptrs,
    dst_v_ptrs,
    src_idx,
    dst_idx,
    item_size,
    num_layers,
    block_quota,
    num_warps
)
# Output results
torch.cuda.synchronize()
print("transfer_kv_all_layer computation completed")
print(f"src_k_ptrs.shape: {src_k_ptrs.shape}")
print(f"item_size: {item_size} | 张量dtype字节数: {src_k_ptrs.element_size()}")
```
## 162. transfer_kv_all_layer_lf_pf
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
transfer_kv_all_layer_lf_pf 算子实现了多 layer 源 KV 张量到目标 KV 张量的索引式传输操作。该算子将包含`num_layers`个 layer 的源 Key/Value 张量集合（`src_k_layers`、`src_v_layers`），按照指定的源索引（`src_indices`）和目标索引（`dst_indices`），搬运到对应的目标 Key/Value 张量（`dst_k`、`dst_v`）中，常用于 Transformer 模型的 KV 缓存数据高效搬运场景。
- 计算公式：
  对于任意层索引 $l$（$0 \leq l < \text{num\_layers}$）、任意索引位置 $idx$（$0 \leq idx < \text{block\_quota}$），以及任意特征维度索引 $f$（$0 \leq f < \text{item\_size}$），有：
  $$
  \begin{cases}
  \text{dst\_k}\left[\text{dst\_indices}[idx] \times \text{dst\_layout\_dim} + f\right] = \text{src\_k\_layers}[l]\left[\text{src\_indices}[idx] \times \text{item\_size} + f\right] \\
  \text{dst\_v}\left[\text{dst\_indices}[idx] \times \text{dst\_layout\_dim} + f\right] = \text{src\_v\_layers}[l]\left[\text{src\_indices}[idx] \times \text{item\_size} + f\right]
  \end{cases}
  $$
  其中，源层内张量元素的线性索引与目标张量元素的线性索引映射关系分别为：
  $$
  n_{\text{src}} = \text{src\_indices}[idx] \times \text{item\_size} + f
  $$
  
  $$
  n_{\text{dst}} = \text{dst\_indices}[idx] \times \text{dst\_layout\_dim} + f
  $$
  各变量的维度与含义：
  - $\text{src\_k\_layers} \in \mathbb{T}^{\text{num\_layers} \times S \times \text{item\_size}}$：源 Key 张量集合，数据类型为$\mathbb{T}$，包含$\text{num\_layers}$个 layer 的 Key 张量，每个 layer 的 Key 张量维度为 $S \times \text{item\_size}$（$S$为源序列长度）
  - $\text{src\_v\_layers} \in \mathbb{T}^{\text{num\_layers} \times S \times \text{item\_size}}$：源 Value 张量集合，数据类型与$\text{src\_k\_layers}$一致
  - $\text{dst\_k} \in \mathbb{T}^{D \times \text{dst\_layout\_dim}}$：目标 Key 张量，数据类型与$\text{src\_k\_layers}$一致，维度为 $D \times \text{dst\_layout\_dim}$（$D$为目标序列长度，$\text{dst\_layout\_dim} \geq \text{item\_size}$以保证内存对齐）
  - $\text{dst\_v} \in \mathbb{T}^{D \times \text{dst\_layout\_dim}}$：目标 Value 张量，数据类型与$\text{src\_v\_layers}$一致
  - $\text{src\_indices} \in \mathbb{Z}^{\text{block\_quota}}$：源张量的索引列表，每个元素对应$\text{src\_k\_layers}/\text{src\_v\_layers}$单个 layer 内的序列维度索引
  - $\text{dst\_indices} \in \mathbb{Z}^{\text{block\_quota}}$：目标张量的索引列表，每个元素对应$\text{dst\_k}/\text{dst\_v}$的序列维度索引
  - $\text{item\_size} \in \mathbb{Z}^+$：每个 Key/Value 元素的特征维度大小
  - $\text{dst\_layout\_dim} \in \mathbb{Z}^+$：目标张量的内存步长（序列维度单步对应的内存间隔）
  - $\text{num\_layers} \in \mathbb{Z}^+$：源张量集合包含的 layer 数量
  - $\text{block\_quota} \in \mathbb{Z}^+$：单次传输的索引数量（即$\text{src\_indices}/\text{dst\_indices}$的长度）
  - $\text{num\_warps\_per\_block}$：CUDA 核函数的 warp 配置参数（不参与计算逻辑）
  - $n_{\text{src}}$：源层内张量元素的线性索引，范围为 $0 \leq n_{\text{src}} < S \times \text{item\_size}$
  - $n_{\text{dst}}$：目标张量元素的线性索引，范围为 $0 \leq n_{\text{dst}} < D \times \text{dst\_layout\_dim}$
  - $S$：源序列长度，满足 $0 \leq \text{src\_indices}[idx] < S$
  - $D$：目标序列长度，满足 $0 \leq \text{dst\_indices}[idx] < D$
### 参数说明
- **src_k_layers** (torch.Tensor, 入参): 源 Key 张量集合，维度为`num_layers × 源序列长度 × item_size`，存储待传输的多 layer 源 Key 数据
- **dst_k** (torch.Tensor, 入参/出参): 目标 Key 张量，维度为`目标序列长度 × dst_layout_dim`，传输后的 Key 数据会写入该张量
- **src_v_layers** (torch.Tensor, 入参): 源 Value 张量集合，维度与`src_k_layers`一致，存储待传输的多 layer 源 Value 数据
- **dst_v** (torch.Tensor, 入参/出参): 目标 Value 张量，维度与`dst_k`一致，传输后的 Value 数据会写入该张量
- **src_indices** (torch.Tensor, 入参): 源索引列表，长度为`block_quota`，元素对应`src_k_layers/src_v_layers`单 layer 内的序列维度索引
- **dst_indices** (torch.Tensor, 入参): 目标索引列表，长度为`block_quota`，元素对应`dst_k/dst_v`的序列维度索引
- **item_size** (int, 入参): 单个 Key/Value 元素的特征维度大小
- **dst_layout_dim** (int, 入参): 目标张量的内存步长（序列维度单步对应的内存间隔）
- **num_layers** (int, 入参): 源 KV 张量集合包含的 layer 数量
- **block_quota** (int, 入参): 单次传输的索引数量（即`src_indices`/`dst_indices`的长度）
- **num_warps_per_block** (int, 入参): CUDA 核函数的 warp 配置参数，用于控制并行执行粒度
### 返回值
无返回值，传输结果直接写入`dst_k`和`dst_v`张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上（代码包含`DCHECK_CUDA`检查）
- `src_k_layers`的 layer 数量必须等于`num_layers`（代码包含`DCHECK_EQ(src_k_layers.size(0), num_layers)`检查）
- `src_indices`与`dst_indices`的长度必须等于`block_quota`
- 支持的张量数据类型：CUDA 设备兼容的浮点类型（如 float16、bfloat16、float32 等）
### 调用示例
```python
import torch
import numpy as np
import mcoplib.sgl_kernel
# Set computation device
device = "cuda"
# Core parameters
num_layers = 2
capacity = 1024
item_size = 128
dst_layout_dim = capacity  # 新增参数，通常指显存布局的维度/跨度
idx_len = 16
block_quota = 8
num_warps = 4
# 1. Construct Source Tensors (Need Pointer Conversion)
# src_k/v_layers: [num_layers, capacity, item_size]
src_k_data = torch.randn(num_layers, capacity, item_size, dtype=torch.float16, device=device)
src_v_data = torch.randn_like(src_k_data)
# 2. Construct Destination Tensors (Direct Tensor, NO Pointer Conversion)
# dst_k/v: The operator expects raw data tensors for destination, not a pointer table.
# Shape usually matches the flat buffer or specific layout.
dst_k = torch.zeros(num_layers, capacity, item_size, dtype=torch.float16, device=device)
dst_v = torch.zeros_like(dst_k)
# 3. Helper: Convert Source to UInt64 Pointer Tensor
def to_uint64_ptr(t):
    # Only for Source layers which require address table
    return torch.tensor(
        np.array([sub.data_ptr() for sub in t], dtype=np.uint64),
        device=device, dtype=torch.uint64
    )
src_k_ptrs = to_uint64_ptr(src_k_data)
src_v_ptrs = to_uint64_ptr(src_v_data)
# 4. Construct Index Tensors
src_idx = torch.arange(idx_len, device=device, dtype=torch.long)
dst_idx = torch.arange(idx_len, device=device, dtype=torch.long)
# Call the operator
# Note argument order: src(Ptr), dst(Data), src(Ptr), dst(Data), indices...
torch.ops.sgl_kernel.transfer_kv_all_layer_lf_pf(
    src_k_ptrs,       # Source K: Pointer Tensor (UInt64)
    dst_k,            # Dest K:   Data Tensor (Float16)
    src_v_ptrs,       # Source V: Pointer Tensor (UInt64)
    dst_v,            # Dest V:   Data Tensor (Float16)
    src_idx,
    dst_idx,
    item_size,
    dst_layout_dim,   # New int parameter
    num_layers,
    block_quota,
    num_warps
)
# Output results
torch.cuda.synchronize()
print("transfer_kv_all_layer_lf_pf computation completed")
```
## 163. transfer_kv_all_layer_lf_ph
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_all_layer_lf_ph(
    const at::Tensor src_k_layers,
    at::Tensor dst_k,
    const at::Tensor src_v_layers,
    at::Tensor dst_v,
    const at::Tensor src_indices,
    const at::Tensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block
) -> None
```
### 功能描述
transfer_kv_all_layer_lf_ph 算子实现 KV 缓存跨层的数据传输功能：读取源 K/V 层张量（src_k_layers、src_v_layers）数据，基于源索引（src_indices）和目标索引（dst_indices），将对应数据传输并写入到目标 K/V 张量（dst_k、dst_v）中，同时完成层数校验、维度适配等逻辑处理，适用于 Transformer 模型 KV 缓存的跨层数据调度场景。
- 计算公式：
  对于索引张量的每个线性索引$n$（$0 \leq n < \text{len}(\text{src\_indices})$），执行 KV 张量的元素转移：
  $$
  \text{dst\_k}[\text{dst\_indices}[n]] = \text{src\_k\_layers}[\text{src\_indices}[n]]
  $$
  $$
  \text{dst\_v}[\text{dst\_indices}[n]] = \text{src\_v\_layers}[\text{src\_indices}[n]]
  $$
  同时需满足以下约束：
  $$
  \text{num\_layers} = \text{src\_k\_layers.size}(0)
  $$
  $$
  \text{item\_size} \% 8 == 0
  $$
  各变量的维度与含义：
  - $\text{src\_k\_layers} \in \mathbb{T}^{\text{num\_layers} \times *}$：源 k 的层张量，数据类型为$\mathbb{T}$，其第 0 维度长度等于$\text{num\_layers}$
  - $\text{dst\_k} \in \mathbb{T}^{*}$：目标 k 张量，存储转移后的 k 数据，数据类型与$\text{src\_k\_layers}$一致
  - $\text{src\_v\_layers} \in \mathbb{T}^{\text{num\_layers} \times *}$：源 v 的层张量，数据类型与$\text{src\_k\_layers}$一致，其第 0 维度长度等于$\text{num\_layers}$
  - $\text{dst\_v} \in \mathbb{T}^{*}$：目标 v 张量，存储转移后的 v 数据，数据类型与$\text{src\_v\_layers}$一致
  - $\text{src\_indices} \in \text{long}^{\text{N}}$：源索引张量（CUDA 张量），$\text{N}$为索引长度，每个元素对应源张量的索引
  - $\text{dst\_indices} \in \text{long}^{\text{N}}$：目标索引张量（CUDA 张量），长度与$\text{src\_indices}$相同（$\text{len}(\text{dst\_indices}) = \text{len}(\text{src\_indices})$），每个元素对应目标张量的索引
  - $\text{item\_size}$：每个转移项的字节大小，需满足$ \text{item\_size} \% 8 == 0 $以保证内存对齐
  - $\text{dst\_layoutdim}$：目标张量的布局维度，用于目标张量的内存布局计算
  - $\text{num\_layers}$：参与转移的层数，与$\text{src\_k\_layers}$的第 0 维度长度相等
  - $\text{page\_size}$：页大小，用于张量的页式内存管理
  - $\text{head\_num}$：多头注意力的头数量，对应 KV 张量的头维度
  - $\text{block\_quota}$：CUDA 核函数的块资源配额，用于块级资源分配
  - $\text{num\_warps\_per\_block}$：每个 CUDA 块分配的 warps 数量，用于核函数的线程调度
  - $n$：索引张量的线性索引，范围为$0 \leq n < \text{N}$，其中$\text{N} = \text{len}(\text{src\_indices})$
### 参数说明
- **src_k_layers** (const at::Tensor, 入参): 源 K 层张量，存储各层 K 缓存数据
- **dst_k** (at::Tensor, 入参/出参): 目标 K 张量，计算完成后会被传输后的 K 缓存数据覆盖
- **src_v_layers** (const at::Tensor, 入参): 源 V 层张量，存储各层 V 缓存数据
- **dst_v** (at::Tensor, 入参/出参): 目标 V 张量，计算完成后会被传输后的 V 缓存数据覆盖
- **src_indices** (const at::Tensor, 入参): 源数据索引张量，用于定位 src_k_layers/src_v_layers 中待传输的数据位置
- **dst_indices** (const at::Tensor, 入参): 目标位置索引张量，用于定位 dst_k/dst_v 中数据写入的位置
- **item_size** (int64_t, 入参): 单个 KV 数据项的字节大小
- **dst_layout_dim** (int64_t, 入参): 目标 K/V 张量的布局维度，用于适配数据写入的维度规则
- **num_layers** (int64_t, 入参): 待处理的 KV 缓存层数，需与 src_k_layers 的实际层数匹配
- **page_size** (int64_t, 入参): KV 缓存的页大小，用于数据分片传输的粒度控制
- **head_num** (int64_t, 入参): 注意力头数量，适配 KV 缓存按头维度的传输逻辑
- **block_quota** (int64_t, 入参): 块配额，用于限制 CUDA 核函数的块资源使用
- **num_warps_per_block** (int64_t, 入参): 每个 CUDA 块的线程束数量，用于核函数的线程调度
### 返回值
无返回值，计算结果直接写入 dst_k 和 dst_v 张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上（算子基于 CUDA 实现）
- src_k_layers 的实际层数必须与入参 num_layers 严格匹配（代码中通过 TORCH_CHECK 校验）
- 支持的数据类型需与 CUDA 核函数兼容（如 float16、bfloat16 等常见深度学习张量类型）
- dst_layout_dim 需与 dst_k/dst_v 的实际维度结构匹配，否则会导致数据写入位置错误
- block_quota 和 num_warps_per_block 需符合 CUDA 设备的线程/块资源限制，避免核函数启动失败
### 调用示例
```python
import torch
import numpy as np
import mcoplib.sgl_kernel
# Set computation device
device = "cuda"
# Core parameters
num_layers = 2
capacity = 1024
item_size = 128
dst_layout_dim = capacity  # Layout dimension stride
# Specific parameters for _ph (PageHeadLayout)
page_size = 16
head_num = 4
idx_len = 16
block_quota = 8
num_warps = 4
# 1. Construct Source Tensors (Need Pointer Conversion)
# src_k/v_layers: [num_layers, capacity, item_size]
src_k_data = torch.randn(num_layers, capacity, item_size, dtype=torch.float16, device=device)
src_v_data = torch.randn_like(src_k_data)
# 2. Construct Destination Tensors (Direct Tensor, NO Pointer Conversion)
# dst_k/v: Raw data tensors. Shape depends on the specific layout, 
# ensuring sufficient size for testing.
dst_k = torch.zeros(num_layers, capacity, item_size, dtype=torch.float16, device=device)
dst_v = torch.zeros_like(dst_k)
# 3. Helper: Convert Source to UInt64 Pointer Tensor
def to_uint64_ptr(t):
    return torch.tensor(
        np.array([sub.data_ptr() for sub in t], dtype=np.uint64),
        device=device, dtype=torch.uint64
    )
src_k_ptrs = to_uint64_ptr(src_k_data)
src_v_ptrs = to_uint64_ptr(src_v_data)
# 4. Construct Index Tensors
src_idx = torch.arange(idx_len, device=device, dtype=torch.long)
dst_idx = torch.arange(idx_len, device=device, dtype=torch.long)
# Call the operator
# Signature: src_ptr, dst_data, src_ptr, dst_data, indices, ...
# Extra args: dst_layout_dim, num_layers, page_size, head_num, ...
torch.ops.sgl_kernel.transfer_kv_all_layer_lf_ph(
    src_k_ptrs,       # Source K: Pointer Tensor (UInt64)
    dst_k,            # Dest K:   Data Tensor (Float16)
    src_v_ptrs,       # Source V: Pointer Tensor (UInt64)
    dst_v,            # Dest V:   Data Tensor (Float16)
    src_idx,
    dst_idx,
    item_size,
    dst_layout_dim,   # arg 1
    num_layers,       # arg 2
    page_size,        # arg 3 (New)
    head_num,         # arg 4 (New)
    block_quota,
    num_warps
)
# Output results
torch.cuda.synchronize()
print("transfer_kv_all_layer_lf_ph computation completed")
```
## 164. transfer_kv_per_layer_mla
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quote: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
该算子基于 CUDA 实现**单一层的 KV（Key-Value）数据传输**，通过源索引张量（`src_indices`）和目标索引张量（`dst_indices`），将源张量（`src`）中的 KV 数据搬运到目标张量（`dst`）中，用于 KV 缓存相关的数据迁移操作。
- 计算公式：
  对于索引张量的每个线性索引$n$（$0 \leq n < \text{len}(\text{src\_indices})$），执行张量元素的转移操作：
  $$
  \text{dst}[\text{dst\_indices}[n]] = \text{src}[\text{src\_indices}[n]]
  $$
  同时需满足内存对齐约束：
  $$
  \text{item\_size} \% 8 == 0
  $$
  各变量的维度与含义：
  - $\text{src} \in \mathbb{T}^{*}$：源张量，数据类型为$\mathbb{T}$，存储待转移的原始数据
  - $\text{dst} \in \mathbb{T}^{*}$：目标张量，存储转移后的数据，数据类型与$\text{src}$一致
  - $\text{src\_indices} \in \text{long}^{\text{N}}$：源索引张量（CUDA 张量），$\text{N}$为索引长度，每个元素对应$\text{src}$中待转移元素的索引
  - $\text{dst\_indices} \in \text{long}^{\text{N}}$：目标索引张量（CUDA 张量），长度与$\text{src\_indices}$相同（$\text{len}(\text{dst\_indices}) = \text{len}(\text{src\_indices})$），每个元素对应$\text{dst}$中存储转移元素的索引
  - $\text{item\_size}$：每个转移项的字节大小，需满足$ \text{item\_size} \% 8 == 0 $以保证内存对齐
  - $\text{block\_quota}$：CUDA 核函数的块资源配额，用于块级资源分配
  - $\text{num\_warps\_per\_block}$：每个 CUDA 块分配的 warps 数量，用于核函数的线程调度
  - $n$：索引张量的线性索引，范围为$0 \leq n < \text{N}$，其中$\text{N} = \text{len}(\text{src\_indices})$
### 参数说明
- **src** (const at::Tensor, 入参): 源 KV 数据张量，需存储于 CUDA 设备
- **dst** (const at::Tensor, 入参/出参): 目标 KV 数据张量，需存储于 CUDA 设备；传输完成后，数据会写入此张量
- **src_indices** (const at::Tensor, 入参): 源数据的索引张量，需存储于 CUDA 设备，用于指定从`src`中读取数据的位置
- **dst_indices** (const at::Tensor, 入参): 目标数据的索引张量，需存储于 CUDA 设备，用于指定向`dst`中写入数据的位置
- **item_size** (int64_t, 入参): 单个 KV 数据项的字节大小
- **block_quote** (int64_t, 入参): 与 CUDA 块相关的配置参数
- **num_warps_per_block** (int64_t, 入参): 每个 CUDA 块分配的 warp 数量
### 返回值
无返回值，计算结果直接写入 dst 和 dst 张量中
### 约束与调用
- 所有输入张量（`src`、`dst`、`src_indices`、`dst_indices`）必须位于 CUDA 设备上
- `src_indices` 必须是 CUDA 张量，且数据类型为`long`
- `dst_indices` 必须是 CUDA 张量，且数据类型为`long`
- `src_indices` 与 `dst_indices` 的长度必须完全一致
- `item_size` 必须能被 8 整除（即`item_size % 8 == 0`）
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda"
# 核心参数
item_size = 128
idx_len = 16
block_quota = 8
num_warps = 4
# 1. 构造输入输出张量 (直接使用数据 Tensor)
# 假设数据形状为 [token_num, item_size]
src = torch.randn(1024, item_size, dtype=torch.float16, device=device)
dst = torch.zeros_like(src)
# 2. 构造索引张量
src_idx = torch.arange(idx_len, device=device, dtype=torch.long)
dst_idx = torch.arange(idx_len, device=device, dtype=torch.long)
# 调用算子
torch.ops.sgl_kernel.transfer_kv_per_layer_mla(
    src=src,
    dst=dst,
    src_indices=src_idx,
    dst_indices=dst_idx,
    item_size=item_size,
    block_quota=block_quota,
    num_warps_per_block=num_warps
)
# 输出结果
torch.cuda.synchronize()
print("transfer_kv_per_layer_mla computation completed")
```
## 165. transfer_kv_per_layer_mla_pf_lf
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quote: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
该算子基于 CUDA 实现**带布局维度与层标识的单一层 KV（Key-Value）数据传输**，通过源索引张量（`src_indices`）、目标索引张量（`dst_indices`）及层标识（`layer_id`），结合源数据布局维度（`src_layout_dim`），将源张量（`src`）中的 KV 数据搬运到目标张量（`dst`）中，用于 KV 缓存场景下的层内数据迁移。
- 计算公式：
  对于索引张量的每个线性索引$n$（$0 \leq n < \text{len}(\text{src\_indices})$），结合指定层编号$\text{layer\_id}$执行张量元素的转移操作：
  $$
  \text{dst}[\text{dst\_indices}[n]] = \text{src}[\text{src\_indices}[n] + \text{layer\_id} \times \text{src\_layout\_dim}]
  $$
  同时需满足内存对齐约束：
  $$
  \text{item\_size} \% 8 == 0
  $$
  各变量的维度与含义：
  - $\text{src} \in \mathbb{T}^{*}$：源张量，数据类型为$\mathbb{T}$，包含多层数据（层间数据由$\text{src\_layout\_dim}$分隔），存储待转移的原始数据
  - $\text{dst} \in \mathbb{T}^{*}$：目标张量，存储转移后的数据，数据类型与$\text{src}$一致
  - $\text{src\_indices} \in \text{long}^{\text{N}}$：源索引张量（CUDA 张量），$\text{N}$为索引长度，每个元素对应$\text{src}$中当前层内待转移元素的索引
  - $\text{dst\_indices} \in \text{long}^{\text{N}}$：目标索引张量（CUDA 张量），长度与$\text{src\_indices}$相同（$\text{len}(\text{dst\_indices}) = \text{len}(\text{src\_indices})$），每个元素对应$\text{dst}$中存储转移元素的索引
  - $\text{layer\_id}$：当前处理的层编号，用于定位$\text{src}$中对应层的数据（当前层数据的起始偏移为$\text{layer\_id} \times \text{src\_layout\_dim}$）
  - $\text{item\_size}$：每个转移项的字节大小，需满足$ \text{item\_size} \% 8 == 0 $以保证内存对齐
  - $\text{src\_layout\_dim}$：源张量的层内布局维度，用于计算当前层在$\text{src}$中的数据偏移
  - $\text{block\_quota}$：CUDA 核函数的块资源配额，用于块级资源分配
  - $\text{num\_warps\_per\_block}$：每个 CUDA 块分配的 warps 数量，用于核函数的线程调度
  - $n$：索引张量的线性索引，范围为$0 \leq n < \text{N}$，其中$\text{N} = \text{len}(\text{src\_indices})$
### 参数说明
- **src** (const at::Tensor, 入参): 源 KV 数据张量，需存储于 CUDA 设备，作为数据读取的数据源。
- **dst** (const at::Tensor, 入参/出参): 目标 KV 数据张量，需存储于 CUDA 设备；数据传输完成后，结果直接写入此张量。
- **src_indices** (const at::Tensor, 入参): 源数据索引张量，需存储于 CUDA 设备，用于指定从`src`中读取数据的位置。
- **dst_indices** (const at::Tensor, 入参): 目标数据索引张量，需存储于 CUDA 设备，用于指定向`dst`中写入数据的位置。
- **layer_id** (int64_t, 入参): 当前待处理的 KV 数据所属的层标识。
- **item_size** (int64_t, 入参): 单个 KV 数据项的字节大小。
- **src_layout_dim** (int64_t, 入参): 源 KV 数据张量的布局维度参数，用于匹配数据存储结构。
- **block_quote** (int64_t, 入参): CUDA 核函数启动时的块级配置参数。
- **num_warps_per_block** (int64_t, 入参): 每个 CUDA 块分配的 warp 数量，用于控制 CUDA 并行计算的调度逻辑。
### 返回值
无返回值，数据传输结果直接写入`dst`张量中。
### 约束与调用
- 所有输入张量（`src`、`dst`、`src_indices`、`dst_indices`）必须存储在 CUDA 设备上。
- `src_indices` 需为 CUDA 张量，且数据类型为`long`（int64）。
- `dst_indices` 需为 CUDA 张量，且数据类型为`long`（int64）。
- `src_indices` 与 `dst_indices` 的张量长度必须完全一致。
- `item_size` 必须满足`item_size % 8 == 0`（即能被 8 整除）。
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda"
# 核心参数
layer_id = 0
item_size = 128
src_layout_dim = 1024  # 源数据的布局维度跨度
idx_len = 16
block_quota = 8
num_warps = 4
# 1. 构造输入输出张量
src = torch.randn(2048, item_size, dtype=torch.float16, device=device)
dst = torch.zeros(1024, item_size, dtype=torch.float16, device=device)
# 2. 构造索引张量
src_idx = torch.arange(idx_len, device=device, dtype=torch.long)
dst_idx = torch.arange(idx_len, device=device, dtype=torch.long)
# 调用算子
torch.ops.sgl_kernel.transfer_kv_per_layer_mla_pf_lf(
    src=src,
    dst=dst,
    src_indices=src_idx,
    dst_indices=dst_idx,
    layer_id=layer_id,
    item_size=item_size,
    src_layout_dim=src_layout_dim,
    block_quota=block_quota,
    num_warps_per_block=num_warps
)
# 输出结果
torch.cuda.synchronize()
print("transfer_kv_per_layer_mla_pf_lf computation completed")
```
## 166. transfer_kv_all_layer_mla
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quote: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
该算子基于 CUDA 实现**所有层的 KV（Key-Value）数据传输**：先校验源层张量（`src_layers`）的层数与指定总层数（`num_layers`）是否一致，再通过源索引张量（`src_indices`）、目标索引张量（`dst_indices`），将源层张量中的 KV 数据批量搬运到目标层张量（`dst_layers`）中，核心用于 KV 缓存场景下的多层数据迁移操作。
- 计算公式：
  对于索引张量的每个线性索引$n$（$0 \leq n < \text{len}(\text{src\_indices})$），执行多层张量的元素转移操作：
  $$
  \text{dst\_layers}[\text{dst\_indices}[n]] = \text{src\_layers}[\text{src\_indices}[n]]
  $$
  同时需满足层数匹配约束：
  $$
  \text{num\_layers} = \text{src\_layers.size}(0)
  $$
  及内存对齐约束：
  $$
  \text{item\_size} \% 8 == 0
  $$
  各变量的维度与含义：
  - $\text{src\_layers} \in \mathbb{T}^{\text{num\_layers} \times *}$：源层张量，数据类型为$\mathbb{T}$，其第 0 维度长度等于$\text{num\_layers}$，存储待转移的多层原始数据
  - $\text{dst\_layers} \in \mathbb{T}^{\text{num\_layers} \times *}$：目标层张量，存储转移后的多层数据，数据类型与$\text{src\_layers}$一致
  - $\text{src\_indices} \in \text{long}^{\text{N}}$：源索引张量（CUDA 张量），$\text{N}$为索引长度，每个元素对应$\text{src\_layers}$中待转移元素的索引
  - $\text{dst\_indices} \in \text{long}^{\text{N}}$：目标索引张量（CUDA 张量），长度与$\text{src\_indices}$相同（$\text{len}(\text{dst\_indices}) = \text{len}(\text{src\_indices})$），每个元素对应$\text{dst\_layers}$中存储转移元素的索引
  - $\text{item\_size}$：每个转移项的字节大小，需满足$ \text{item\_size} \% 8 == 0 $以保证内存对齐
  - $\text{num\_layers}$：参与转移的层数，与$\text{src\_layers}$的第 0 维度长度相等（$\text{num\_layers} = \text{src\_layers.size}(0)$）
  - $\text{block\_quota}$：CUDA 核函数的块资源配额，用于块级资源分配
  - $\text{num\_warps\_per\_block}$：每个 CUDA 块分配的 warps 数量，用于核函数的线程调度
  - $n$：索引张量的线性索引，范围为$0 \leq n < \text{N}$，其中$\text{N} = \text{len}(\text{src\_indices})$
### 参数说明
- **src_layers** (const at::Tensor, 入参): 源所有层的 KV 数据张量，需存储于 CUDA 设备，作为多层数据读取的数据源。
- **dst_layers** (const at::Tensor, 入参/出参): 目标所有层的 KV 数据张量，需存储于 CUDA 设备；数据传输完成后，结果直接写入此张量。
- **src_indices** (const at::Tensor, 入参): 源数据索引张量，需存储于 CUDA 设备，用于指定从`src_layers`中读取数据的位置。
- **dst_indices** (const at::Tensor, 入参): 目标数据索引张量，需存储于 CUDA 设备，用于指定向`dst_layers`中写入数据的位置。
- **item_size** (int64_t, 入参): 单个 KV 数据项的字节大小。
- **num_layers** (int64_t, 入参): 待处理的 KV 数据总层数，需与`src_layers`的层数匹配。
- **block_quote** (int64_t, 入参): CUDA 核函数启动时的块级配置参数。
- **num_warps_per_block** (int64_t, 入参): 每个 CUDA 块分配的 warp 数量，用于控制 CUDA 并行计算的调度逻辑。
### 返回值
无返回值，数据传输结果直接写入`dst_layers`张量中。
### 约束与调用
- 所有输入张量（`src_layers`、`dst_layers`、`src_indices`、`dst_indices`）必须存储在 CUDA 设备上。
- `num_layers`必须与`src_layers.size(0)`（即源层张量的层数）完全一致，否则会触发参数合法性校验失败。
- `src_indices`需为 CUDA 张量，且数据类型为`long`（int64）。
- `dst_indices`需为 CUDA 张量，且数据类型为`long`（int64）。
- `src_indices`与`dst_indices`的张量长度必须完全一致。
- `item_size`必须满足`item_size % 8 == 0`（即能被 8 整除）。
### 调用示例
```python
import torch
import numpy as np
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda"
# 核心参数
num_layers = 2
capacity = 1024
item_size = 128
idx_len = 16
block_quota = 8
num_warps = 4
# 1. 构造基础数据张量 (Float16)
src_data = torch.randn(num_layers, capacity, item_size, dtype=torch.float16, device=device)
dst_data = torch.zeros_like(src_data)
# 2. 构造 UInt64 指针张量 (用于多层寻址)
def to_uint64_ptr(t):
    return torch.tensor(
        np.array([sub.data_ptr() for sub in t], dtype=np.uint64),
        device=device, dtype=torch.uint64
    )
src_layers_ptrs = to_uint64_ptr(src_data)
dst_layers_ptrs = to_uint64_ptr(dst_data)
# 3. 构造索引张量
src_idx = torch.arange(idx_len, device=device, dtype=torch.long)
dst_idx = torch.arange(idx_len, device=device, dtype=torch.long)
# 调用算子
torch.ops.sgl_kernel.transfer_kv_all_layer_mla(
    src_layers=src_layers_ptrs,  # UInt64 指针表
    dst_layers=dst_layers_ptrs,  # UInt64 指针表
    src_indices=src_idx,
    dst_indices=dst_idx,
    item_size=item_size,
    num_layers=num_layers,
    block_quota=block_quota,
    num_warps_per_block=num_warps
)
# 输出结果
torch.cuda.synchronize()
print("transfer_kv_all_layer_mla computation completed")
```
## 167. transfer_kv_all_layer_mla_lf_pf
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quote: int,
    num_warps_per_block: int
) -> None
```
### 功能描述
该算子基于 CUDA 实现**带目标布局维度的所有层 KV（Key-Value）数据传输**：先校验源层张量（`src_layers`）的层数与指定总层数（`num_layers`）是否一致，再结合目标布局维度（`dst_layout_dim`），通过源索引张量（`src_indices`）、目标索引张量（`dst_indices`），将源层张量中的 KV 数据批量搬运到目标张量（`dst`）中，用于 KV 缓存场景下的多层数据迁移。
- 计算公式：
  对于索引张量的每个线性索引$n$（$0 \leq n < \text{len}(\text{src\_indices})$），执行多层源张量到目标张量的元素转移操作：
  $$
  \text{dst}[\text{dst\_indices}[n]] = \text{src\_layers}[\text{src\_indices}[n]]
  $$
  同时需满足层数匹配约束：
  $$
  \text{num\_layers} = \text{src\_layers.size}(0)
  $$
  及内存对齐约束：
  $$
  \text{item\_size} \% 8 == 0
  $$
  各变量的维度与含义：
  - $\text{src\_layers} \in \mathbb{T}^{\text{num\_layers} \times *}$：源层张量，数据类型为$\mathbb{T}$，其第 0 维度长度等于$\text{num\_layers}$，存储待转移的多层原始数据
  - $\text{dst} \in \mathbb{T}^{*}$：目标张量，存储转移后的多层数据，数据类型与$\text{src\_layers}$一致
  - $\text{src\_indices} \in \text{long}^{\text{N}}$：源索引张量（CUDA 张量），$\text{N}$为索引长度，每个元素对应$\text{src\_layers}$中待转移元素的索引
  - $\text{dst\_indices} \in \text{long}^{\text{N}}$：目标索引张量（CUDA 张量），长度与$\text{src\_indices}$相同（$\text{len}(\text{dst\_indices}) = \text{len}(\text{src\_indices})$），每个元素对应$\text{dst}$中存储转移元素的索引
  - $\text{item\_size}$：每个转移项的字节大小，需满足$ \text{item\_size} \% 8 == 0 $以保证内存对齐
  - $\text{dst\_layout\_dim}$：目标张量的布局维度，用于目标张量的内存布局计算
  - $\text{num\_layers}$：参与转移的层数，与$\text{src\_layers}$的第 0 维度长度相等（$\text{num\_layers} = \text{src\_layers.size}(0)$）
  - $\text{block\_quota}$：CUDA 核函数的块资源配额，用于块级资源分配
  - $\text{num\_warps\_per\_block}$：每个 CUDA 块分配的 warps 数量，用于核函数的线程调度
  - $n$：索引张量的线性索引，范围为$0 \leq n < \text{N}$，其中$\text{N} = \text{len}(\text{src\_indices})$
### 参数说明
- **src_layers** (const at::Tensor, 入参): 源所有层的 KV 数据张量，需存储于 CUDA 设备，作为多层数据读取的数据源。
- **dst** (const at::Tensor, 入参/出参): 目标 KV 数据张量，需存储于 CUDA 设备；数据传输完成后，结果直接写入此张量。
- **src_indices** (const at::Tensor, 入参): 源数据索引张量，需存储于 CUDA 设备，用于指定从`src_layers`中读取数据的位置。
- **dst_indices** (const at::Tensor, 入参): 目标数据索引张量，需存储于 CUDA 设备，用于指定向`dst`中写入数据的位置。
- **item_size** (int64_t, 入参): 单个 KV 数据项的字节大小。
- **dst_layout_dim** (int64_t, 入参): 目标 KV 数据张量的布局维度参数，用于匹配目标数据存储结构。
- **num_layers** (int64_t, 入参): 待处理的 KV 数据总层数，需与`src_layers`的层数匹配。
- **block_quote** (int64_t, 入参): CUDA 核函数启动时的块级配置参数。
- **num_warps_per_block** (int64_t, 入参): 每个 CUDA 块分配的 warp 数量，用于控制 CUDA 并行计算的调度逻辑。
### 返回值
无返回值，数据传输结果直接写入`dst`张量中。
### 约束与调用
- 所有输入张量（`src_layers`、`dst`、`src_indices`、`dst_indices`）必须存储在 CUDA 设备上。
- `num_layers`必须与`src_layers.size(0)`（即源层张量的层数）完全一致，否则会触发参数合法性校验失败。
- `src_indices`需为 CUDA 张量，且数据类型为`long`（int64）。
- `dst_indices`需为 CUDA 张量，且数据类型为`long`（int64）。
- `src_indices`与`dst_indices`的张量长度必须完全一致。
- `item_size`必须满足`item_size % 8 == 0`（即能被 8 整除）。
### 调用示例
```python
import torch
import numpy as np
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda"
# 核心参数
num_layers = 2
capacity = 1024
item_size = 128
dst_layout_dim = capacity  # 目标张量的布局维度
idx_len = 16
block_quota = 8
num_warps = 4
# 1. 构造源数据 (多层) 与 目标数据 (连续)
src_data = torch.randn(num_layers, capacity, item_size, dtype=torch.float16, device=device)
# 目标是一个连续的大张量 [num_layers, capacity, item_size]
dst_tensor = torch.zeros(num_layers, capacity, item_size, dtype=torch.float16, device=device)
# 2. 构造源指针张量 (UInt64)
def to_uint64_ptr(t):
    return torch.tensor(
        np.array([sub.data_ptr() for sub in t], dtype=np.uint64),
        device=device, dtype=torch.uint64
    )
src_layers_ptrs = to_uint64_ptr(src_data)
# 3. 构造索引张量
src_idx = torch.arange(idx_len, device=device, dtype=torch.long)
dst_idx = torch.arange(idx_len, device=device, dtype=torch.long)
# 调用算子
torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf(
    src_layers=src_layers_ptrs,  # UInt64 指针表
    dst=dst_tensor,              # 直接使用数据张量 (at::Tensor)
    src_indices=src_idx,
    dst_indices=dst_idx,
    item_size=item_size,
    dst_layout_dim=dst_layout_dim,
    num_layers=num_layers,
    block_quota=block_quota,
    num_warps_per_block=num_warps
)
# 输出结果
torch.cuda.synchronize()
print("transfer_kv_all_layer_mla_lf_pf computation completed")
```
## 168. transfer_kv_direct
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int
) -> None
```
### 功能描述
transfer_kv_direct 算子主要用于在不同的 KV Cache 层之间进行数据的直接传输与拷贝。其核心逻辑采用了贪婪策略来优化传输效率：算子会将索引数据移动到 CPU，遍历索引数组，自动检测“源索引（src）”和“目标索引（dst）”是否同时保持物理连续。如果检测到连续的块，它会将这些连续的页合并为一个大的批次，然后按层调用底层的 transfer_page_direct 函数进行批量传输。这种机制有效减少了碎片化的内存拷贝调用次数。
- 计算公式：
  $$
  \text{dst\_layers}[l][\text{dst\_index} + k] = \text{src\_layers}[l][\text{src\_index} + k], \quad k \in [0, \text{num\_tokens})
  $$
  其中批量划分规则为：
  $$
  \text{end\_index} = 
  \begin{cases} 
  i+1, & \text{当 } (\text{src\_indices}[i+1] - \text{src\_indices}[i] \neq 1) \lor (\text{dst\_indices}[i+1] - \text{dst\_indices}[i] \neq 1) \\
  \text{num\_indices}, & \text{当 } i = \text{num\_indices} - 1
  \end{cases}
  $$
  算子执行前需满足的合法性约束：
  $$
  \begin{cases}
  \text{len}(\text{src\_layers}) = \text{len}(\text{dst\_layers}) \\
  \text{num\_indices} = \text{len}(\text{src\_indices}) = \text{len}(\text{dst\_indices}) \\
  \text{page\_size} > 0 \\
  \text{num\_indices} \% \text{page\_size} = 0
  \end{cases}
  $$
  各变量的维度与含义：
  - $\text{src\_layers} \in \mathbb{List}[\mathbb{T}^{*}]$：源 KV 缓存层张量列表，每个张量为任意维度的 KV 缓存数据，数据类型为$\mathbb{T}$，列表长度为$\text{num\_layers}$
  - $\text{dst\_layers} \in \mathbb{List}[\mathbb{T}^{*}]$：目标 KV 缓存层张量列表，与$\text{src\_layers}$的维度、数据类型、列表长度均一致，用于接收传输后的数据
  - $\text{src\_indices} \in \mathbb{Z}^{\text{num\_indices}}$：源 KV 缓存的位置索引张量，元素为整数类型，用于索引源数据的位置
  - $\text{dst\_indices} \in \mathbb{Z}^{\text{num\_indices}}$：目标 KV 缓存的位置索引张量，元素为整数类型，用于索引目标存储位置
  - $\text{page\_size} \in \mathbb{Z}^+$：批量处理的页大小，为正整数
  - $\text{num\_indices} \in \mathbb{Z}^+$：索引张量的元素总数，满足$\text{num\_indices} \% \text{page\_size} = 0$
  - $\text{num\_layers} \in \mathbb{Z}^+$：KV 缓存的层数，即$\text{len}(\text{src\_layers})$
  - $l$：KV 缓存层索引，范围为$0 \leq l < \text{num\_layers}$
  - $\text{start\_index} \in \mathbb{Z}$：批量传输的起始索引，范围为$0 \leq \text{start\_index} < \text{num\_indices}$
  - $\text{end\_index} \in \mathbb{Z}$：批量传输的结束索引，范围为$\text{start\_index} < \text{end\_index} \leq \text{num\_indices}$
  - $\text{num\_tokens} \in \mathbb{Z}^+$：单批次传输的 token 数量，满足$\text{num\_tokens} = \text{end\_index} - \text{start\_index}$
  - $\text{src\_index} \in \mathbb{Z}$：单批次源数据的起始位置，即$\text{src\_indices}[\text{start\_index}]$
  - $\text{dst\_index} \in \mathbb{Z}$：单批次目标数据的起始位置，即$\text{dst\_indices}[\text{start\_index}]$
  - $k$：单批次内的 token 偏移量，范围为$0 \leq k < \text{num\_tokens}$
  - $i$：全局索引遍历变量，范围为$0 \leq i < \text{num\_indices}$
### 参数说明
- **src_layers** (const std::vector<at::Tensor>&, 入参): 源 KV Cache 层的张量列表
- **dst_layers** (std::vector<at::Tensor>, 入参/出参): 目标 KV Cache 层的张量列表，数据将被写入其中
- **src_indices** (const at::Tensor, 入参): 源数据的页索引张量
- **dst_indices** (const at::Tensor, 入参): 目标数据的页索引张量
- **page_size** (int64_t, 入参): 每一页的大小（以元素数量或特定单位计，代码中用于校验索引整除性）
### 返回值
无返回值，计算结果直接更新到 dst_layers 所指向的内存中
### 约束与调用
- src_layers 的数量必须等于 dst_layers 的数量
- src_indices 的元素个数必须等于 dst_indices 的元素个数
- page_size 必须大于 0
- src_indices 的元素总数必须能被 page_size 整除
- src_indices 和 dst_indices 必须为 int64（Long）类型的张量
- 索引张量会被显式拷贝到 CPU（. cpu ()）进行连续性判断，但实际的数据传输（transfer_page_direct）依然在原设备（通常是 GPU）上执行
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "transfer_kv_direct 仅支持CUDA设备"
# 核心参数
num_layers = 2
page_size = 16
head_dim = 64
num_heads = 8
total_pages = 100
# 1. 构造 KV Cache 层
# 原始物理存储: [total_pages, num_heads, page_size, head_dim]
src_layers = []
dst_layers = []
for _ in range(num_layers):
    s = torch.randn(total_pages, num_heads, page_size, head_dim, dtype=torch.float16, device=device)
    d = torch.zeros(total_pages, num_heads, page_size, head_dim, dtype=torch.float16, device=device)
    src_layers.append(s)
    dst_layers.append(d)
# 【关键修复】算子需要 Token 级扁平视图: [Total_Tokens, Heads, Dim]
src_layers_flat = [t.view(-1, num_heads, head_dim) for t in src_layers]
dst_layers_flat = [t.view(-1, num_heads, head_dim) for t in dst_layers]
# 2. 构造索引 (Indices)
# 示例：Source Page 5 -> Dest Page 10
src_page_ids = [5, 6]
dst_page_ids = [10, 11]
src_indices_list = []
dst_indices_list = []
for s_id, d_id in zip(src_page_ids, dst_page_ids):
    # 计算扁平化后的 Token 索引范围
    src_indices_list.append(torch.arange(s_id * page_size, (s_id + 1) * page_size, device=device))
    dst_indices_list.append(torch.arange(d_id * page_size, (d_id + 1) * page_size, device=device))
src_indices = torch.cat(src_indices_list).to(dtype=torch.int64)
dst_indices = torch.cat(dst_indices_list).to(dtype=torch.int64)
# 3. 调用算子
torch.ops.sgl_kernel.transfer_kv_direct(
    src_layers_flat,
    dst_layers_flat,
    src_indices,
    dst_indices,
    page_size
)
print("transfer_kv_direct computation completed")
# 验证数据
is_equal = torch.allclose(src_layers[0][5], dst_layers[0][10])
print(f"Data verification (Source Page 5 == Dest Page 10): {is_equal}")
#  打印维度信息 
print("\n=== Tensor Dimensions Info ===")
print(f"Original Layer Shape (Page-based):   {src_layers[0].shape}")
print(f"Flattened Layer Shape (Token-based): {src_layers_flat[0].shape}")
print(f"Source Indices Shape:                {src_indices.shape}")
print(f"Dest Indices Shape:                  {dst_indices.shape}")
```
## 169. transfer_kv_per_layer_direct_pf_lf
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int
) -> None
```
### 功能描述
该算子是 `transfer_kv_page_first_direct_impl` 的一个特化封装版本。它通过将模板参数 `IsLf2Pf` 设置为 `false` 来调用底层实现。
具体逻辑如下：
1. **调用路径**：直接调用 `transfer_kv_page_first_direct_impl<false>(...)`。
2. **核心逻辑**（源于其调用的 Impl 中的 `else` 分支）：
   - 遍历页（Pages）和层（Layers）。
   - 针对每一页和每一层，计算源和目标的偏移索引。
   - 调用 `transfer_page_direct` 进行实际的数据搬运。
   - **MLA 支持**：代码中包含针对 MLA（Multi-Head Latent Attention）的逻辑判断（通过检查 `src_ptrs.size() == 1`），如果不是 MLA (`!is_mla`)，则会进行额外的 `transfer_page_direct` 调用来处理后续数据。
   - 从函数命名后缀 `_pf_lf` 和模板参数 `false` 推断，该算子用于处理与“Layer First to Page First”相反或特定的内存布局转换（可能是 Page First to Layer First）。
   - 计算公式：
  $$
  \text{dst\_layer}[\text{dst\_offset} + t] = \text{src\_layer}[\text{src\_offset} + t], \quad t \in [0, \text{batch\_tokens})
  $$
  其中批次偏移量与 token 数的计算规则：
  $$
  \begin{cases}
  \text{src\_offset} = \text{src\_indices}[\text{batch\_start}] \\
  \text{dst\_offset} = \text{dst\_indices}[\text{batch\_start}] \\
  \text{batch\_tokens} = \text{batch\_end} - \text{batch\_start}
  \end{cases}
  $$
  批次划分的终止条件：
  $$
  \text{batch\_end} = 
  \begin{cases} 
  b+1, & \text{当 } (\text{src\_indices}[b+1] - \text{src\_indices}[b] \neq 1) \lor (\text{dst\_indices}[b+1] - \text{dst\_indices}[b] \neq 1) \\
  \text{total\_indices}, & \text{当 } b = \text{total\_indices} - 1
  \end{cases}
  $$
  算子执行前的合法性约束：
  $$
  \begin{cases}
  \text{total\_indices} = \text{len}(\text{src\_indices}) = \text{len}(\text{dst\_indices}) \\
  \text{page\_size} > 0 \\
  \text{total\_indices} \% \text{page\_size} = 0
  \end{cases}
  $$
  各变量的维度与含义：
  - $\text{src\_layer} \in \mathbb{T}^{*}$：单层级的源 KV 缓存张量，数据类型为$\mathbb{T}$，维度为适配 KV 缓存存储的任意维度
  - $\text{dst\_layer} \in \mathbb{T}^{*}$：单层级的目标 KV 缓存张量，数据类型、维度与$\text{src\_layer}$完全一致，用于接收当前层传输后的 KV 数据
  - $\text{src\_indices} \in \mathbb{Z}^{\text{total\_indices}}$：当前层源 KV 缓存的位置索引张量，元素为整数类型，用于索引源数据的位置
  - $\text{dst\_indices} \in \mathbb{Z}^{\text{total\_indices}}$：当前层目标 KV 缓存的位置索引张量，元素为整数类型，用于索引目标存储位置
  - $\text{page\_size} \in \mathbb{Z}^+$：批量处理的页大小，为正整数
  - $\text{total\_indices} \in \mathbb{Z}^+$：当前层索引张量的元素总数，满足$\text{total\_indices} \% \text{page\_size} = 0$
  - $\text{batch\_start} \in \mathbb{Z}$：单批次传输的起始索引，范围为$0 \leq \text{batch\_start} < \text{total\_indices}$
  - $\text{batch\_end} \in \mathbb{Z}$：单批次传输的结束索引，范围为$\text{batch\_start} < \text{batch\_end} \leq \text{total\_indices}$
  - $\text{batch\_tokens} \in \mathbb{Z}^+$：单批次传输的 token 数量，满足$\text{batch\_tokens} = \text{batch\_end} - \text{batch\_start}$
  - $\text{src\_offset} \in \mathbb{Z}$：单批次源数据在当前层 KV 缓存中的起始位置，即$\text{src\_indices}[\text{batch\_start}]$
  - $\text{dst\_offset} \in \mathbb{Z}$：单批次目标数据在当前层 KV 缓存中的起始位置，即$\text{dst\_indices}[\text{batch\_start}]$
  - $t$：单批次内的 token 偏移量，范围为$0 \leq t < \text{batch\_tokens}$
  - $b$：全局索引遍历变量，范围为$0 \leq b < \text{total\_indices}$
### 参数说明
- **src_ptrs** (const std::vector<at::Tensor>&, 入参): 源 KV Cache 的张量指针列表（通常包含 K 和 V 的 cache）
- **dst_ptrs** (std::vector<at::Tensor>, 入参/出参): 目标 KV Cache 的张量指针列表，数据将被写入其中
- **src_indices** (const at::Tensor&, 入参): 源数据的页索引张量
- **dst_indices** (const at::Tensor&, 入参): 目标数据的页索引张量
- **layer_id** (int64_t, 入参): 当前操作的基础层 ID，用于计算 offset
- **page_size** (int64_t, 入参): 页大小
### 返回值
- 无返回值 (void)。计算结果直接更新到 `dst_ptrs` 指向的内存中
### 约束与调用
- 底层校验（继承自 `transfer_kv_page_first_direct_impl`）：
  - `src_indices` 和 `dst_indices` 的元素数量必须相等
  - `page_size` 必须大于 0
  - `src_indices` 的元素总数必须能被 `page_size` 整除
- 实现依赖：该函数是一个简单的 wrapper，所有具体的计算逻辑、循环遍历和 CUDA/CPU 索引处理均依赖于 `transfer_kv_page_first_direct_impl` 的实现
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "此算子仅支持CUDA设备"
# 核心参数
num_layers = 2
page_size = 16
head_dim = 64
num_heads = 8
total_pages = 100
total_tokens = total_pages * page_size
# 1. 构造 KV Cache
# Destination: 标准的分层存储 List[Tensor]
# 物理存储: [total_pages, num_heads, page_size, head_dim] -> 展平为 [total_tokens, num_heads, head_dim]
dst_layers = []
for _ in range(num_layers):
    d = torch.zeros(total_pages, num_heads, page_size, head_dim, dtype=torch.float16, device=device)
    dst_layers.append(d)
dst_layers_flat = [t.view(-1, num_heads, head_dim) for t in dst_layers]
# Source: 聚合存储 (Fused)
# [Total_Tokens, Num_Layers, Num_Heads, Head_Dim]
src_fused = torch.randn(
    total_tokens, num_layers, num_heads, head_dim,
    dtype=torch.float16, device=device
)
# 算子要求 src_ptrs 是 vector<Tensor>，所以要把这个大张量包进列表
src_ptrs = [src_fused]
# 2. 构造索引 (Indices)
# 示例：Source Page 5 -> Dest Page 10
src_page_ids = [5, 6]
dst_page_ids = [10, 11]
src_indices_list = []
dst_indices_list = []
for s_id, d_id in zip(src_page_ids, dst_page_ids):
    src_indices_list.append(torch.arange(s_id * page_size, (s_id + 1) * page_size, device=device))
    dst_indices_list.append(torch.arange(d_id * page_size, (d_id + 1) * page_size, device=device))
src_indices = torch.cat(src_indices_list).to(dtype=torch.int64)
dst_indices = torch.cat(dst_indices_list).to(dtype=torch.int64)
# 3. 调用算子
# C++签名: transfer_kv_per_layer_direct_pf_lf(src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size)
torch.ops.sgl_kernel.transfer_kv_per_layer_direct_pf_lf(
    src_ptrs,
    dst_layers_flat,
    src_indices,
    dst_indices,
    0, 
    page_size
)
print("transfer_kv_per_layer_direct_pf_lf computation completed")
#  打印维度信息 
print("\n=== Tensor Dimensions Info ===")
print(f"Source Fused Tensor Shape:     {src_fused.shape}")
print(f"Dest Layer 0 Shape (Flat):     {dst_layers_flat[0].shape}")
print(f"Source Indices Shape:          {src_indices.shape}")
print(f"Dest Indices Shape:            {dst_indices.shape}")
```
## 170. transfer_kv_all_layer_direct_lf_pf
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int
) -> None
```
### 功能描述
该算子是 `transfer_kv_page_first_direct_impl` 的特化封装版本，专门用于执行 **Layer First 到 Page First (LF to PF)** 的数据布局转换与传输。
具体逻辑如下：
1. **调用路径**：内部直接调用了 `transfer_kv_page_first_direct_impl<true>(..., 0, page_size)`，其中模板参数 `true` 激活了 `IsLf2Pf`（Layer First to Page First）的代码分支。
2. **核心逻辑**：
   - **MLA 判断**：通过检查 `dst_ptrs.size() == 1` 来判断是否为 MLA（Multi-Head Latent Attention）模式。
   - **层数计算**：如果是 MLA，层数等于 `src_ptrs.size()`；否则等于 `src_ptrs.size() / 2`（暗示非 MLA 模式下 src 包含 K 和 V 两部分层数据）。
   - **遍历与传输**：
     - 算子遍历所有页（Pages）和层（Layers）。
     - 从 CPU 端的索引张量中获取源索引和目标索引。
     - **K Cache 传输**：将 `src_ptrs[j]`（Layer First）的数据传输到 `dst_ptrs[0]`（Page First，通过 `.select` 选定层）中。
     - **V Cache 传输**（非 MLA 模式）：将 `src_ptrs[j + num_layers]` 的数据传输到 `dst_ptrs[1]` 中。
- 计算公式：
  $$
  \text{dst\_ptrs}[j][\text{d\_index} + t] = \text{src\_ptrs}[j][\text{s\_index} + t], \quad t \in [0, \text{page\_size})
  $$
  其中页批次的索引计算规则：
  $$
  \begin{cases}
  \text{s\_index} = \frac{\text{src\_indices}[i \times \text{page\_size}]}{\text{page\_size}} \\
  \text{d\_index} = \frac{\text{dst\_indices}[i \times \text{page\_size} + \text{num\_layers} \times \text{page\_size}]}{\text{page\_size}} \\
  \text{num\_pages} = \frac{\text{src\_indices.numel()}}{\text{page\_size}}
  \end{cases}
  $$
  算子执行前的合法性约束：
  $$
  \begin{cases}
  \text{src\_indices.numel()} = \text{dst\_indices.numel()} \\
  \text{page\_size} > 0 \\
  \text{src\_indices.numel()} \% \text{page\_size} = 0
  \end{cases}
  $$
  各变量的维度与含义：
  - $\text{src\_ptrs} \in \mathbb{List}[\mathbb{T}^{*}]$：所有层的源 KV 缓存张量列表，每个张量为对应层的 KV 数据，数据类型为$\mathbb{T}$
  - $\text{dst\_ptrs} \in \mathbb{List}[\mathbb{T}^{*}]$：所有层的目标 KV 缓存张量列表，维度、数据类型与$\text{src\_ptrs}$一致，用于接收传输后的数据
  - $\text{src\_indices} \in \mathbb{Z}^{\text{src\_indices.numel()}}$：源 KV 缓存的位置索引张量，元素为整数类型，用于定位源数据位置
  - $\text{dst\_indices} \in \mathbb{Z}^{\text{dst\_indices.numel()}}$：目标 KV 缓存的位置索引张量，元素为整数类型，用于定位目标存储位置
  - $\text{page\_size} \in \mathbb{Z}^+$：批量处理的页大小，为正整数
  - $\text{num\_pages} \in \mathbb{Z}^+$：页批次的总数量，满足$\text{num\_pages} = \frac{\text{src\_indices.numel()}}{\text{page\_size}}$
  - $\text{num\_layers} \in \mathbb{Z}^+$：KV 缓存的层数，满足$\text{num\_layers} = \frac{\text{src\_ptrs.size()}}{2}$（由代码中`src_ptrs.size() / 2`推导）
  - $i$：页批次的遍历索引，范围为$0 \leq i < \text{num\_pages}$
  - $j$：KV 层的遍历索引，范围为$0 \leq j < \text{num\_layers}$
  - $\text{s\_index} \in \mathbb{Z}$：当前页批次在源 KV 缓存中的起始位置（按页划分后的索引）
  - $\text{d\_index} \in \mathbb{Z}$：当前页批次在目标 KV 缓存中的起始位置（按页划分后的索引）
  - $t$：当前页批次内的 token 偏移量，范围为$0 \leq t < \text{page\_size}$
### 参数说明
- **src_ptrs** (const std::vector<at::Tensor>&, 入参): 源 KV Cache 张量列表（Layer First 布局，通常每个元素代表一层）
- **dst_ptrs** (std::vector<at::Tensor>, 入参/出参): 目标 KV Cache 张量列表（Page First 布局，通常包含 K 和 V 的张量）
- **src_indices** (const at::Tensor&, 入参): 源数据的页索引张量
- **dst_indices** (const at::Tensor&, 入参): 目标数据的页索引张量
- **page_size** (int64_t, 入参): 页大小，用于计算页数和校验索引
### 返回值
- 无返回值 (void)。计算结果直接写入 `dst_ptrs` 指向的内存中
### 约束与调用
- 底层校验（继承自实现函数）：
  - `src_indices` 和 `dst_indices` 的元素数量必须相等
  - `page_size` 必须大于 0
  - `src_indices` 的元素总数必须能被 `page_size` 整除
- 输入输出布局假设：
  - 源数据 (`src_ptrs`) 被假定为按层分开的 Vector 结构（Layer First）
  - 目标数据 (`dst_ptrs`) 被假定为紧凑的 Tensor 结构，通过 `select` 操作访问特定层（Page First）
- 参数传递：调用底层实现时，`layer_id` 被硬编码为 `0`
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "此算子仅支持CUDA设备"
# 核心参数
num_layers = 2
page_size = 16
head_dim = 64
num_heads = 8
total_pages = 100
# total_tokens = total_pages * page_size (这个变量在定义dst时暂时用不到)
# 1. 构造 KV Cache
# === Source: Layer First (LF) ===
# 保持之前的扁平化处理，因为读取时是按 Token 索引读取的
src_layers = []
for _ in range(num_layers):
    s = torch.randn(total_pages, num_heads, page_size, head_dim, dtype=torch.float16, device=device)
    src_layers.append(s)
# [Total_Tokens, Num_Heads, Head_Dim]
src_layers_flat = [t.view(-1, num_heads, head_dim) for t in src_layers]
# === Destination: Page First (PF) / Fused ===
# 【关键修改】必须定义为 5D 张量，显式包含 Page_Size 维度
# 结构: [Total_Pages, Num_Layers, Page_Size, Num_Heads, Head_Dim]
dst_fused = torch.zeros(
    total_pages, num_layers, page_size, num_heads, head_dim,
    dtype=torch.float16, device=device
)
# 算子要求 dst_ptrs 是 vector<Tensor>
dst_ptrs = [dst_fused]
# 2. 构造索引 (Indices)
# 示例：Source Page 5 -> Dest Page 10
src_page_ids = [5, 6]
dst_page_ids = [10, 11]
src_indices_list = []
dst_indices_list = []
for s_id, d_id in zip(src_page_ids, dst_page_ids):
    # 生成 Token 级索引
    src_indices_list.append(torch.arange(s_id * page_size, (s_id + 1) * page_size, device=device))
    dst_indices_list.append(torch.arange(d_id * page_size, (d_id + 1) * page_size, device=device))
src_indices = torch.cat(src_indices_list).to(dtype=torch.int64)
dst_indices = torch.cat(dst_indices_list).to(dtype=torch.int64)
# 3. 调用算子
# C++签名: transfer_kv_all_layer_direct_lf_pf(src_ptrs, dst_ptrs, src_indices, dst_indices, page_size)
torch.ops.sgl_kernel.transfer_kv_all_layer_direct_lf_pf(
    src_layers_flat,
    dst_ptrs,
    src_indices,
    dst_indices,
    page_size
)
print("transfer_kv_all_layer_direct_lf_pf computation completed")
#  打印维度信息 
print("\n=== Tensor Dimensions Info ===")
print(f"Source Layer 0 Shape (Flat):   {src_layers_flat[0].shape}")
# 这里的 shape 应该是 5 维的
print(f"Dest Fused Tensor Shape:       {dst_fused.shape}") 
print(f"Source Indices Shape:          {src_indices.shape}")
```
## 171. store_kv_cache
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def store_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> None
```
### 功能描述
store_kv_cache 算子用于将输入的 Key (k) 和 Value (v) 张量根据给定的索引位置 (out_loc) 存储到预分配的 KV 缓存（k_cache 和 v_cache）中。该算子通过 CUDA kernel 并行处理内存拷贝，支持高效的非连续缓存更新，是 LLM 推理框架中处理 KV Cache 动态存储的核心操作。
- 计算公式：
  对于键缓存与输入键张量的元素，满足：
  $$
  \text{k\_cache}[b, \text{out\_loc}[b] + s, h, c] = \text{k}[b, s, h, c]
  $$
  对于值缓存与输入值张量的元素，满足：
  $$
  \text{v\_cache}[b, \text{out\_loc}[b] + s, h, c] = \text{v}[b, s, h, c]
  $$
  各变量的维度与含义：
  - $\text{k\_cache} \in \mathbb{T}^{M \times L \times H \times C}$：键缓存张量，数据类型为$\mathbb{T}$，维度为[batch 数, 缓存最大序列长度, 注意力头数, 单头维度]
  - $\text{v\_cache} \in \mathbb{T}^{M \times L \times H \times C}$：值缓存张量，数据类型与$\text{k\_cache}$一致，维度与$\text{k\_cache}$相同
  - $\text{out\_loc} \in \mathbb{Z}^M$：每个 batch 在缓存中的起始序列位置张量，维度为[batch 数]
  - $\text{k} \in \mathbb{T}^{M \times S \times H \times C}$：输入键张量，数据类型与$\text{k\_cache}$一致，维度为[batch 数, 当前序列长度, 注意力头数, 单头维度]
  - $\text{v} \in \mathbb{T}^{M \times S \times H \times C}$：输入值张量，数据类型与$\text{v\_cache}$一致，维度与$\text{k}$相同
  - $b$：batch 索引，范围为 $0 \leq b < M$
  - $s$：当前序列的步索引，范围为 $0 \leq s < S$
  - $h$：注意力头索引，范围为 $0 \leq h < H$
  - $c$：单头维度索引，范围为 $0 \leq c < C$
  - $M$：batch 的数量（$\text{k\_cache.size(0)} = \text{k.size(0)} = M$）
  - $L$：缓存的最大序列长度（$\text{k\_cache.size(1)} = L$）
  - $S$：当前输入序列的长度（$\text{k.size(1)} = S$）
  - $H$：注意力头的数量（$\text{k\_cache.size(2)} = \text{k.size(2)} = H$）
  - $C$：每个注意力头的维度（$\text{k\_cache.size(3)} = \text{k.size(3)} = C$）
### 参数说明
- **k_cache** (torch.Tensor, 入参/出参): 全局 Key 缓存张量，形状通常为 [max_tokens, num_heads, head_size]
- **v_cache** (torch.Tensor, 入参/出参): 全局 Value 缓存张量，形状与 k_cache 一致
- **out_loc** (torch.Tensor, 入参): 一维整数张量（int32 或 int64），包含了当前输入 token 在缓存中对应的存储索引位置
- **k** (torch.Tensor, 入参): 待存储的新 Key 张量
- **v** (torch.Tensor, 入参): 待存储的新 Value 张量
### 返回值
无返回值，计算结果直接写入 k_cache 和 v_cache 张量对应的内存地址中
### 约束与调用
- 所有输入张量（k_cache, v_cache, out_loc, k, v）必须位于 CUDA 设备上
- out_loc 必须是 1D 连续张量，且数据类型必须为 int32 或 int64
- k 和 v 的最后一维字节数（size_bytes）必须至少能被 128 整除，以满足算子的内存访问粒度要求
- k 和 v 必须具有相同的形状和步长（strides）
- k_cache 和 v_cache 必须具有相同的形状和步长
- k 的 head size 必须与 k_cache 的 head size 一致
- 支持的数据类型：float16, bfloat16, float32
### 调用示例
```python
import torch
# 1. 基础配置
device = "cuda"
dtype = torch.bfloat16
max_tokens, num_tokens, head_dim = 128, 4, 128
# 2. 准备张量
# Cache 模拟预分配的大显存
k_cache = torch.zeros(max_tokens, head_dim, dtype=dtype, device=device)
v_cache = torch.zeros(max_tokens, head_dim, dtype=dtype, device=device)
# 输入数据 (待存储的 K 和 V)
k = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)
v = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)
# 存储位置索引 (int32)，例如存储到索引 0, 5, 10, 15
out_loc = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device=device)
# 3. 调用算子
import mcoplib.sgl_kernel 
torch.ops.sgl_kernel.store_kv_cache(k_cache, v_cache, out_loc, k, v)
# 4. 打印结果
print("="*40)
print(f"k_cache shape: {k_cache.shape}")
print(f"v_cache shape: {v_cache.shape}")
print(f"Input k shape: {k.shape}")
print(f"Indices used:  {out_loc.tolist()}")
print("-" * 40)
# 验证第一个位置的数据 (索引 out_loc[0])
target_idx = out_loc[0].item()
is_match = torch.allclose(k_cache[target_idx], k[0], atol=1e-3)
print(f"Verification at index {target_idx}: {'PASS' if is_match else 'FAIL'}")
print(f"K_cache snippet: {k_cache[target_idx, :4].tolist()}")
print(f"Original K snippet: {k[0, :4].tolist()}")
print("="*40)
```
## 174. apply_token_bitmask_inplace_cuda
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def apply_token_bitmask_inplace_cuda(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[torch.Tensor] = None
) -> None
```
### 功能描述
apply_token_bitmask_inplace_cuda 算子实现了基于位掩码（bitmask）的 Logits 原地屏蔽操作。该算子根据 bitmask 中对应位的状态（0 或 1），将 logits 张量中非法 token 的分值原地修改为负无穷（-INFINITY）。这种操作通过位运算（Bitwise operations）高效地实现了对词表（Vocabulary）的大规模并行过滤，常用于大语言模型推理中的受限解码（Grammar-guided decoding）或结构化输出约束。
- 计算公式：
  对于 logits 张量中位置为$(b, t, v)$的元素，依据 bitmask 张量对应位的取值做 inplace 修改：
  $$
  \text{logits}[b, t, v] = 
  \begin{cases} 
  \text{NegInf}(\mathbb{T}), & \text{当 } \text{bitmask}[b, t, \lfloor v / K \rfloor] \text{ 的第 } (v \mod K) \text{ 位为 } 0 \\
  \text{logits}[b, t, v], & \text{当 } \text{bitmask}[b, t, \lfloor v / K \rfloor] \text{ 的第 } (v \mod K) \text{ 位为 } 1
  \end{cases}
  $$
  其中元素的线性索引与三维位置的映射关系为：
  $$
  n = b \times S_{BT} + t \times S_{TV} + v
  $$
  各变量的维度与含义：
  - $\text{logits} \in \mathbb{T}^{B \times T \times V}$：待掩码处理的 logits 张量（inplace 修改），数据类型为$\mathbb{T}$（float/half 等），维度为[批量数, 序列长度, 词表大小]
  - $\text{bitmask} \in \mathbb{U}^{B \times T \times \lceil V / K \rceil}$：掩码张量，数据类型为无符号整数$\mathbb{U}$，每个元素存储$K$位掩码，维度为[批量数, 序列长度, $\lceil V / K \rceil$]
  - $\text{indices}$（可选）$\in \mathbb{Z}^{B \times T}$：指定待处理位置的索引张量，维度与$\text{logits}$前两维匹配
  - $K$：每个 bitmask 元素对应的位数（即代码中$\text{BITS\_PER\_BLOCK}$，如 75、256）
  - $\text{NegInf}(\mathbb{T})$：数据类型$\mathbb{T}$对应的负无穷值（如 float 的极小值）
  - $n$：$\text{logits}$元素的线性索引，范围为$0 \leq n < B \times S_{BT}$
  - $b$：批量索引，范围为$0 \leq b < B$
  - $t$：序列位置索引，范围为$0 \leq t < T$
  - $v$：词表索引，范围为$0 \leq v < V$
  - $S_{BT}$：$\text{logits}$在批量维度的步长，即$S_{BT} = T \times S_{TV}$
  - $S_{TV}$：$\text{logits}$在序列维度的步长（词表维度的内存步长，满足$S_{TV} \geq V$）
### 参数说明
- **logits** (torch.Tensor, 入参/出参): 输入的原始得分张量（通常为 2D），计算完成后，被屏蔽位置的值会被原地替换为负无穷
- **bitmask** (torch.Tensor, 入参): 位掩码张量，数据类型必须为 int32。每个 int32 元素包含 32 个 token 的掩码位
- **indices** (torch.Tensor, 可选入参): 索引张量，数据类型为 int32。若提供，算子将仅根据索引对指定的行应用掩码；若为 None，则顺序处理所有行
### 返回值
无返回值，计算结果直接写入 logits 张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- logits 和 bitmask 必须是连续张量（contiguous）
- bitmask 的数据类型必须为 int32
- logits 的词表维度（vocab_size）应与 bitmask 提供的总位数匹配
- 支持的数据类型：float16, bfloat16, float32
### 调用示例
```python
import torch
# 1. 基础配置
device = "cuda"
batch_size = 2
vocab_size = 64
dtype = torch.float32  # 源码支持 float32, float16, bfloat16
# 2. 准备张量
# logits 必须是 CUDA 且连续的张量
logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=device)
# bitmask 必须是 int32 类型
# 使用 -1 修复 0xFFFFFFFF 导致的溢出报错
# 每行 bitmask 的长度应为 (vocab_size + 31) // 32
bitmask = torch.tensor([
    [-1,  0], # Batch 0: 前 32 个 token 保留 (1)，后 32 个屏蔽 (0)
    [ 0, -1]  # Batch 1: 前 32 个 token 屏蔽 (0)，后 32 个保留 (1)
], dtype=torch.int32, device=device)
# 3. 调用算子
import mcoplib.sgl_kernel
print("="*40)
print(f"Input Logits Shape:  {logits.shape}")
print(f"Input Bitmask Shape: {bitmask.shape}")
print("-" * 40)
try:
    # 算子签名: (logits, bitmask, indices=None)
    torch.ops.sgl_kernel.apply_token_bitmask_inplace_cuda(logits, bitmask)
    # 4. 打印输出信息 (原地操作，logits 已被修改)
    print("Execution Status: SUCCESS")
    print(f"Logits shape after call: {logits.shape}")
    # 验证 Batch 0 的后半部分是否变为负无穷
    print(f"Batch 0 (index 31-32): {logits[0, 31:33].tolist()}")
    print(f"Logits Min Value: {logits.min().item()}")
except RuntimeError as e:
    print(f"Execution Status: FAILED")
    print(f"Error Detail: {e}")
    # 根据您的图片 05721ec6...png，如果 CUDA < 12.4，这里必然报错
    if "CUDA version must be >= 12.4" in str(e):
        print("\n[提示] 您的源码强制要求 CUDA 12.4+ 环境才能运行此特定内核。")
print("="*40)
```
## 177. causal_conv1d_update
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor] = None,
    silu_activation: bool = False,
    cache_seqlen_: Optional[torch.Tensor] = None,
    conv_state_indices_: Optional[torch.Tensor] = None,
    pad_slot_id: int = 0
) -> None
```
### 功能描述
causal_conv1d_update 实现了**因果一维卷积的状态更新操作**，主要用于序列模型中因果卷积的状态维护：结合输入、卷积状态、权重（及可选偏置）完成计算，支持 SiLU 激活（可选），同时处理缓存序列长度、卷积状态索引与填充槽位，实现因果卷积过程中状态的高效更新。
- 计算公式：
  $$
  \text{ConvOut}(x) = \text{Conv1d}_{\text{causal}}(x, \text{conv\_state}_{\text{prev}}, W) + b \quad (\text{若} \ bias\_ \text{存在})
  $$
  $$
  \text{ConvOut}_{\text{act}}(x) =
  \begin{cases}
  \text{SiLU}(\text{ConvOut}(x)) & \text{若} \ \text{silu\_activation} = \text{True} \\
  \text{ConvOut}(x) & \text{否则}
  \end{cases}
  $$
  
  $$
  \text{conv\_state}_{\text{curr}} = \text{UpdateState}(\text{conv\_state}_{\text{prev}}, \text{ConvOut}_{\text{act}}(x), \text{conv\_state\_indices}, \text{cache\_seqlen}, \text{pad\_slot\_id})
  $$
  其中各变量的维度与含义：
  - $x \in \mathbb{R}^{B \times C \times L}$：输入张量，$B$为批量大小，$C$为通道数，$L$为输入序列长度
  - $\text{conv\_state}_{\text{prev}} \in \mathbb{R}^{B \times C \times K}$：更新前的卷积状态张量，$K$为因果卷积核大小
  - $\text{conv\_state}_{\text{curr}} \in \mathbb{R}^{B \times C \times K}$：更新后的卷积状态张量（覆盖原 conv_state）
  - $W \in \mathbb{R}^{C_{\text{out}} \times C \times K}$：因果一维卷积权重张量，$C_{\text{out}}$为输出通道数（通常$C_{\text{out}}=C$）
  - $b \in \mathbb{R}^{C_{\text{out}}}$：可选偏置张量（仅当$bias\_$存在时参与计算）
  - $\text{SiLU}(z) = z \cdot \sigma(z)$：Sigmoid Linear Unit 激活函数，$\sigma(\cdot)$为 Sigmoid 函数
  - $\text{silu\_activation} \in \{True, False\}$：是否启用 SiLU 激活函数的布尔标识
  - $\text{cache\_seqlen} \in \mathbb{Z}^{B}$（可选）：批量中每个样本的缓存序列长度张量
  - $\text{conv\_state\_indices} \in \mathbb{Z}^{B \times K}$（可选）：卷积状态更新的索引张量，用于定位状态更新位置
  - $\text{pad\_slot\_id} \in \mathbb{Z}$：填充槽位标识，用于修正填充位置的缓存序列长度和状态索引取值
  - $\text{Conv1d}_{\text{causal}}(\cdot)$：因果一维卷积操作，输出仅依赖当前及之前的输入和卷积状态
  - $\text{UpdateState}(\cdot)$：卷积状态更新函数，根据激活后输出、索引、缓存长度和填充槽位更新卷积状态
  补充单样本维度的状态更新细节公式：
  $$
  \text{conv\_state}[i, c, k] =
  \begin{cases}
  \text{ConvOut}_{\text{act}}[i, c, l] & \text{若} \ k = (\text{cache\_seqlen}[i] + l) \mod K \\
  \text{conv\_state}[i, c, k] & \text{否则}
  \end{cases}
  $$
  其中：
  - $i \in [0, B-1]$：批量索引
  - $c \in [0, C-1]$：通道索引
  - $k \in [0, K-1]$：卷积核维度索引
  - $l \in [0, L-1]$：输入序列维度索引
### 参数说明
- **x** (at::Tensor, 入参): 参与因果一维卷积计算的输入张量
- **conv_state** (at::Tensor, 入参/出参): 卷积状态张量，存储因果卷积的状态信息，计算后会被更新
- **weight** (at::Tensor, 入参): 因果一维卷积的计算权重张量
- **bias_** (std::optional<at::Tensor>, 入参): 可选偏置张量，若提供则参与因果卷积的偏置计算
- **silu_activation** (bool, 入参): 是否启用 SiLU 激活函数，控制计算中是否添加 SiLU 操作
- **cache_seqlen_** (std::optional<at::Tensor>, 入参): 可选缓存序列长度张量，记录缓存对应的序列长度信息
- **conv_state_indices_** (std::optional<at::Tensor>, 入参): 可选卷积状态索引张量，用于定位卷积状态的目标位置
- **pad_slot_id** (int64_t, 入参): 填充槽位标识，用于处理缓存/状态中的填充位置
### 返回值
无返回值，变换结果直接写入输出张量中
### 约束与调用
- 所有张量必须部署在 CUDA 设备上
- 若提供`bias_`/`cache_seqlen_`/`conv_state_indices_`，其维度、数据类型需与其他张量匹配
- 支持的数据类型：兼容 float16、bfloat16、float32（需与实际计算场景适配）
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
device = "cuda" if torch.cuda.is_available() else "cpu"
# 核心参数
bs = 2 
C = 256  # dim
K = 4    # width (卷积核大小)
pad_slot_id = -1 # 通常默认为 -1 或具体索引
silu_activation = True 
x = torch.randn(bs, C, 1, dtype=torch.float32, device=device)
weight = torch.randn(C, K, dtype=torch.float32, device=device)
conv_state = torch.randn(bs, C, K, dtype=torch.float32, device=device)
bias = torch.randn(C, dtype=torch.float32, device=device)
cache_seqlens = torch.zeros(bs, dtype=torch.int32, device=device) 
conv_state_indices = torch.arange(bs, dtype=torch.int32, device=device)
# 调用算子
torch.ops.sgl_kernel.causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias,
    silu_activation,
    cache_seqlens,        
    conv_state_indices,   
    pad_slot_id
)
print("Computation completed successfully!")
```
## 178. causal_conv1d_fwd
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    silu_activation: bool = False,
    pad_slot_id: int = 0
) -> None
```
### 功能描述
causal_conv1d_fwd 算子实现**因果一维卷积的前向计算**，适用于时序类任务（如 Transformer 模型）。它在卷积过程中保证“因果性”（不利用未来时序的信息），同时集成了卷积状态缓存（conv_states）、初始状态标识（has_initial_state）、可选 SiLU 激活等逻辑，还支持缓存索引管理与填充槽位处理，减少时序任务中卷积计算的冗余操作，提升效率。
- 计算公式：
  $$
  d\text{ConvOut} = 
  \begin{cases}
  d\text{ConvOut}_{\text{act}} \cdot \left( \sigma(z) + z \cdot \sigma(z) \cdot (1 - \sigma(z)) \right) & \text{若} \ \text{silu\_activation} = \text{True} \\
  d\text{ConvOut}_{\text{act}} & \text{否则}
  \end{cases}
  $$
  其中 $z = \text{ConvOut}(x)$，$\sigma(\cdot)$ 为 Sigmoid 函数，$d\text{ConvOut}_{\text{act}}$ 为激活后输出的上游梯度。
  $$
  dW = \sum_{b=0}^{B-1} \sum_{t=0}^{L-1} d\text{ConvOut}[b, t, :] \otimes x[b, t-K+1:t+1, :]
  $$
  
  $$
  db = \sum_{b=0}^{B-1} \sum_{t=0}^{L-1} d\text{ConvOut}[b, t, :] \quad (\text{若} \ bias\_ \text{存在})
  $$
  
  $$
  dx[b, t, :] = \sum_{k=0}^{K-1} \sum_{c_{\text{out}}=0}^{C_{\text{out}}-1} d\text{ConvOut}[b, t+k, c_{\text{out}}] \cdot W[c_{\text{out}}, k, :]
  $$
  
  $$
  d\text{conv\_states}[b, k, :] = \sum_{c_{\text{out}}=0}^{C_{\text{out}}-1} d\text{ConvOut}[b, 0, c_{\text{out}}] \cdot W[c_{\text{out}}, K-1-k, :] \quad (\text{若} \ t-k < 0)
  $$
  其中各变量的维度与含义：
  - $dx \in \mathbb{R}^{B \times C_{\text{in}} \times L}$：输入张量$x$的梯度，与$x$维度一致，$B$为批量大小，$C_{\text{in}}$为输入通道数，$L$为输入序列长度
  - $dW \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K}$：权重张量$W$的梯度，与$W$维度一致，$C_{\text{out}}$为输出通道数，$K$为因果卷积核大小
  - $db \in \mathbb{R}^{C_{\text{out}}}$：偏置张量$b$的梯度（仅当$bias\_$存在时计算），与$b$维度一致
  - $d\text{ConvOut} \in \mathbb{R}^{B \times L \times C_{\text{out}}}$：卷积输出（未激活）的梯度
  - $d\text{ConvOut}_{\text{act}} \in \mathbb{R}^{B \times L \times C_{\text{out}}}$：激活后卷积输出的梯度，为反向传播的上游梯度
  - $d\text{conv\_states} \in \mathbb{R}^{B \times C_{\text{in}} \times K}$：卷积状态张量的梯度，与$\text{conv\_states}$维度一致
  - $x \in \mathbb{R}^{B \times C_{\text{in}} \times L}$：前向传播的输入张量
  - $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K}$：前向传播的卷积权重张量
  - $b \in \mathbb{R}^{C_{\text{out}}}$：前向传播的可选偏置张量
  - $\text{silu\_activation} \in \{True, False\}$：前向传播是否启用 SiLU 激活的布尔标识
  - $\sigma(z) = \frac{1}{1 + e^{-z}}$：Sigmoid 激活函数，$z \in \mathbb{R}$
  - $\otimes$：互相关运算（卷积反向传播的核心运算）
  - $B$：批量大小，取值范围$B \geq 1$
  - $C_{\text{in}}$：输入通道数，取值范围$C_{\text{in}} \geq 1$
  - $C_{\text{out}}$：输出通道数，通常$C_{\text{out}} = C_{\text{in}}$
  - $K$：因果卷积核大小，取值范围$K \geq 1$
  - $L$：输入序列长度，取值范围$L \geq 1$
  - $b \in [0, B-1]$：批量索引
  - $t \in [0, L-1]$：序列长度索引
  - $k \in [0, K-1]$：卷积核维度索引
  - $c_{\text{in}} \in [0, C_{\text{in}}-1]$：输入通道索引
  - $c_{\text{out}} \in [0, C_{\text{out}}-1]$：输出通道索引
### 参数说明
- **x** (at::Tensor, 入参): 因果卷积的输入数据张量
- **weight** (at::Tensor, 入参): 卷积核对应的权重张量
- **bias_** (std::optional<at::Tensor>, 入参): 可选的卷积偏置张量
- **conv_states** (std::optional<at::Tensor>, 入参): 可选的卷积状态缓存张量，用于存储/读取历史时序的卷积状态
- **query_start_loc** (std::optional<at::Tensor>, 入参): 可选的查询起始位置张量，定位时序数据中当前查询的起始点
- **cache_indices** (std::optional<at::Tensor>, 入参): 可选的缓存索引张量，用于管理卷积状态缓存的索引映射
- **has_initial_state** (std::optional<at::Tensor>, 入参): 可选的标识张量，指示是否存在初始卷积状态
- **silu_activation** (bool, 入参): 是否在卷积后启用 SiLU 激活函数
- **pad_slot_id** (int64_t, 入参): 填充槽位的 ID，用于处理缓存中的填充区域
### 返回值
无返回值，变换结果直接写入输出张量中
### 约束与调用
- 所有参与计算的张量（若存在）必须部署在 CUDA 设备上
- conv_states（若使用）需满足因果卷积的状态缓存形状要求（与输入、卷积核维度匹配）
- query_start_loc（若使用）的取值需处于合法的时序范围之内
- 支持的数据类型：float16、bfloat16、float32
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "causal_conv1d_fwd only supports CUDA devices"
# 核心参数
batch_size = 1      
C_in = 128          
L = 64             
K = 16            
C_out = 128         
pad_slot_id = 0      
silu_activation = True 
# 1. 构造符合算子要求的张量
x = torch.randn(
    batch_size, C_in, L, 
    dtype=torch.float32, device=device, requires_grad=False
)
# 这是一个 depthwise 卷积操作，输入通道和输出通道通常是一致的
weight = torch.randn(
    C_out, K,  # 形状改为 (128, 16)
    dtype=torch.float32, device=device, requires_grad=False
)
# 可选偏置
bias = torch.randn(
    C_out,
    dtype=torch.float32, device=device, requires_grad=False
)
# 形状应为 (batch_size, dim, kernel_size)
conv_states = torch.randn(
    batch_size, C_in, K,
    dtype=torch.float32, device=device, requires_grad=False
)
# 非必要的可选参数
query_start_loc = None
cache_indices = None
has_initial_state = None
# 调用算子
torch.ops.sgl_kernel.causal_conv1d_fwd(
    x=x,
    weight=weight,
    bias_=bias,
    conv_states=conv_states,
    query_start_loc=query_start_loc,
    cache_indices=cache_indices,
    has_initial_state=has_initial_state,
    silu_activation=silu_activation,
    pad_slot_id=pad_slot_id
)
# 输出结果
print("causal_conv1d_fwd computation completed successfully!")
print(f"Input tensor x shape: {x.shape}")
print(f"Conv weight tensor shape: {weight.shape}")
print(f"Conv states tensor shape: {conv_states.shape}")
```
## 179. fused_mla_RMS_rotary_emb
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def fused_mla_RMS_rotary_emb(
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    kv_a: torch.Tensor,
    q_len: int,
    num_local_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int
) -> int
```
### 功能描述
融合多项操作：对输入张量`q`执行旋转位置编码（rotary_emb），对`latent_cache`执行 RMS 归一化，更新`latent_cache`与`kv_a`，最后再对`latent_cache`执行旋转位置编码。
- 计算公式：
  $$
  \begin{align*}
  &\text{latent\_cache}_{norm} = \text{norm\_weight} \odot \frac{\text{latent\_cache}_{:,0:kv\_lora\_rank}}{\sqrt{\frac{1}{\text{kv\_lora\_rank}} \sum_{i=0}^{\text{kv\_lora\_rank}-1} \text{latent\_cache}_{:,i}^2 + \epsilon}} \\
  &\text{q}_{rope} = \text{RoPE}(\text{q}_{:,:,\text{qk\_nope\_head\_dim}:}, \cos\_\sin\_\text{cache}, \text{positions}) \\
  &\text{latent\_cache}_{update} = [\text{latent\_cache}_{norm} + \text{kv}_a, \text{latent\_cache}_{:,\text{kv\_lora\_rank}:}] \\
  &\text{latent\_cache}_{final} = \text{RoPE}(\text{latent\_cache}_{update}, \cos\_\sin\_\text{cache}, \text{positions})
  \end{align*}
  $$
  其中：
  - $\text{latent\_cache} \in \mathbb{R}^{q\_len \times (\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim})}$ 是缓存张量，dtype=bf16
  - $\text{norm\_weight} \in \mathbb{R}^{\text{kv\_lora\_rank}}$ 是 RMS 归一化权重，dtype=bf16
  - $\text{q} \in \mathbb{R}^{q\_len \times \text{num\_local\_heads} \times (\text{qk\_nope\_head\_dim} + \text{qk\_rope\_head\_dim})}$ 是查询张量，dtype=bf16
  - $\text{kv}_a \in \mathbb{R}^{q\_len \times \text{kv\_lora\_rank}}$ 是待更新张量，dtype=bf16
  - $\cos\_\sin\_\text{cache} \in \mathbb{R}^{\text{max} \times \text{qk\_rope\_head\_dim}}$ 是 RoPE 的 cos/sin 缓存，dtype=float
  - $\text{positions} \in \mathbb{Z}^{bs}$ 是位置索引张量，dtype=int64
  - $\text{RoPE}(\cdot)$ 是旋转位置编码函数，定义为：
    $$
    \begin{cases}
    x_{rot,i} = x_i \cdot \cos(\theta_{pos,i}) - x_{i+\frac{d}{2}} \cdot \sin(\theta_{pos,i}) \\
    x_{rot,i+\frac{d}{2}} = x_i \cdot \sin(\theta_{pos,i}) + x_{i+\frac{d}{2}} \cdot \cos(\theta_{pos,i})
    \end{cases}
    $$
    其中$d=\text{qk\_rope\_head\_dim}$，$\theta_{pos,i} = \cos\_\sin\_\text{cache}[\text{positions}[b],i]$（$b$为 batch 索引）
  - $\epsilon$ 是 RMS 归一化的数值稳定性参数（默认极小正值）
  - $q\_len$ 是序列长度，$\text{num\_local\_heads}$ 是本地注意力头数，$\text{kv\_lora\_rank}$ 是 LoRA 秩，$\text{qk\_rope\_head\_dim}$ 是启用 RoPE 的头维度，$\text{qk\_nope\_head\_dim}$ 是不启用 RoPE 的头维度
  - $\odot$ 表示逐元素乘法，$[:,a:b]$ 表示张量沿第二维度取$a$到$b$的切片，$+$ 表示逐元素加法
### 参数说明
- **q** (torch.Tensor, 入参): 输入张量，用于旋转位置编码计算，维度需匹配`q_len`、`num_local_heads`等参数
- **latent_cache** (torch.Tensor, 入参/出参): 缓存张量，会被 RMS 归一化更新，后续需执行旋转位置编码
- **cos_sin_cache** (torch.Tensor, 入参): 旋转位置编码的 cos/sin 预缓存张量，数据类型为 float
- **positions** (torch.Tensor, 入参): 位置索引张量，用于匹配`cos_sin_cache`的对应位置
- **norm_weight** (torch.Tensor, 入参): RMS 归一化的权重张量，维度为`kv_lora_rank`
- **kv_a** (torch.Tensor, 入参/出参): 需更新的张量，维度为`[q_len, kv_lora_rank]`
- **q_len** (int64_t, 入参): `q`、`latent_cache`、`kv_a`的第 0 维度长度
- **num_local_heads** (int64_t, 入参): `q`的第 1 维度长度（本地头数）
- **kv_lora_rank** (int64_t, 入参): LoRA 秩，对应`latent_cache`第 1 维度的部分长度
- **qk_rope_head_dim** (int64_t, 入参): 启用旋转编码的头维度，对应`q`第 2 维度的部分长度
- **qk_nope_head_dim** (int64_t, 入参): 不启用旋转编码的头维度，对应`q`第 2 维度的部分长度
### 返回值
返回值为 int64_t 类型
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 各张量形状需满足代码中`TORCH_CHECK`的校验规则（如`q.size(0) == q_len`、`latent_cache.size(1) == kv_lora_rank + qk_rope_head_dim`等）
- 支持的数据类型：bf16（`q`/`latent_cache`/`norm_weight`/`kv_a`）、float（`cos_sin_cache`）、int64（`positions`）
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置设备
device = "cuda"
# 定义核心参数
q_len = 16             
num_local_heads = 128  
kv_lora_rank = 512     
qk_rope_head_dim = 64   
qk_nope_head_dim = 128   
bs = 8                   
max_cos_sin_len = 2048   
# 构造输入张量
q = torch.randn(
    q_len, num_local_heads, qk_nope_head_dim + qk_rope_head_dim,
    dtype=torch.bfloat16, device=device
)
latent_cache = torch.randn(
    q_len, kv_lora_rank + qk_rope_head_dim,
    dtype=torch.bfloat16, device=device
)
cos_sin_cache = torch.randn(
    max_cos_sin_len, qk_rope_head_dim,
    dtype=torch.float32, device=device 
)
positions = torch.randint(0, max_cos_sin_len, (q_len,), dtype=torch.int64, device=device)
norm_weight = torch.randn(kv_lora_rank, dtype=torch.bfloat16, device=device)
kv_a = torch.randn(q_len, kv_lora_rank, dtype=torch.bfloat16, device=device)
# 调用fused_mla_RMS_rotary_emb算子
ret_code = torch.ops.sgl_kernel.fused_mla_RMS_rotary_emb(
    q=q,
    latent_cache=latent_cache,
    cos_sin_cache=cos_sin_cache,
    positions=positions,
    norm_weight=norm_weight,
    kv_a=kv_a,
    q_len=q_len,
    num_local_heads=num_local_heads,
    kv_lora_rank=kv_lora_rank,
    qk_rope_head_dim=qk_rope_head_dim,
    qk_nope_head_dim=qk_nope_head_dim
)
# 输出调用结果
print("fused_mla_RMS_rotary_emb computation completed")
print(f"Return code: {ret_code}")
print(f"Updated latent_cache shape: {latent_cache.shape}")
print(f"Updated kv_a shape: {kv_a.shape}")
```
## 180. fused_mla_normal_kv_element_wise
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
def fused_mla_normal_kv_element_wise(
    kv: torch.Tensor,
    latent_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_len: int,
    num_local_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int
) -> int
```
### 功能描述
在 prefill 阶段完成融合操作：包含数据拷贝、张量维度变换及逐元素运算，用于处理`kv`、`latent_cache`、`k`、`v`之间的数值交互逻辑。
- 计算公式：
  $$
  \begin{align*}
  &\text{kv}_{reshape} = \text{reshape}(\text{kv}, [\text{bs}, \text{num\_local\_heads}, \text{qk\_nope\_head\_dim} + \text{v\_head\_dim}]) \\
  &\text{kv}_{k\_part} = \text{kv}_{reshape}[:,:,0:\text{qk\_nope\_head\_dim}] \\
  &\text{kv}_{v\_part} = \text{kv}_{reshape}[:,:,\text{qk\_nope\_head\_dim}:] \\
  &\text{kv}_{final} = \text{concat}([\text{kv}_{k\_part} + \text{latent\_cache}_{:,\text{kv\_lora\_rank}:}, \text{kv}_{v\_part} \odot \text{k}_{:,:,\text{qk\_rope\_head\_dim}:}], 2) \\
  &\text{kv} = \text{reshape}(\text{kv}_{final}, [\text{bs}, \text{num\_local\_heads} \times (\text{qk\_nope\_head\_dim} + \text{v\_head\_dim})])
  \end{align*}
  $$
  其中：
  - $\text{kv} \in \mathbb{R}^{\text{bs} \times (\text{num\_local\_heads} \times (\text{qk\_nope\_head\_dim} + \text{v\_head\_dim}))}$ 是输出张量，dtype=bf16
  - $\text{latent\_cache} \in \mathbb{R}^{q\_len \times (\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim})}$ 是缓存张量，dtype=bf16
  - $\text{k} \in \mathbb{R}^{\text{bs} \times \text{num\_local\_heads} \times (\text{qk\_nope\_head\_dim} + \text{qk\_rope\_head\_dim})}$ 是键张量，dtype=bf16
  - $\text{v} \in \mathbb{R}^{\text{bs} \times \text{num\_local\_heads} \times \text{v\_head\_dim}}$ 是值张量，dtype=bf16
  - $\text{reshape}(\cdot, [d_1,d_2,...])$ 表示张量维度重塑操作
  - $\text{concat}([T_1,T_2], dim)$ 表示沿$dim$维度拼接张量$T_1$和$T_2$
  - $\text{bs}$ 是 batch 大小，$\text{num\_local\_heads}$ 是本地注意力头数，$\text{kv\_lora\_rank}$ 是 LoRA 秩，$\text{qk\_rope\_head\_dim}$ 是启用 RoPE 的头维度，$\text{qk\_nope\_head\_dim}$ 是不启用 RoPE 的头维度，$\text{v\_head\_dim}$ 是值头维度
  - $\odot$ 表示逐元素乘法，$+$ 表示逐元素加法，$[:,a:b]$ 表示张量沿指定维度取$a$到$b$的切片
  - $q\_len$ 是 latent_cache 的序列长度维度，与 prefill 阶段的序列长度匹配
### 参数说明
- **kv** (torch.Tensor, 入参/出参): 输出张量，存储处理后的结果，维度为`[bs, num_local_heads * (qk_nope_head_dim + v_head_dim)]`
- **latent_cache** (torch.Tensor, 入参): 缓存张量，提供运算所需的输入数据
- **k** (torch.Tensor, 入参): 输入张量，维度为`[bs, num_local_heads, qk_nope_head_dim + qk_rope_head_dim]`
- **v** (torch.Tensor, 入参): 输入张量，维度为`[bs, num_local_heads, v_head_dim]`
- **q_len** (int64_t, 入参): `latent_cache`的第 0 维度长度
- **num_local_heads** (int64_t, 入参): 本地头数（`k`、`v`的第 1 维度长度）
- **kv_lora_rank** (int64_t, 入参): LoRA 秩（`latent_cache`第 1 维度的部分长度）
- **qk_nope_head_dim** (int64_t, 入参): 不启用旋转编码的头维度（`k`第 2 维度的部分长度）
- **qk_rope_head_dim** (int64_t, 入参): 启用旋转编码的头维度（`k`第 2 维度的部分长度）
- **v_head_dim** (int64_t, 入参): `v`的第 2 维度长度
### 返回值
返回值为 int64_t 类型
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 各张量形状需匹配参数维度要求（如`k.size(2) == qk_nope_head_dim + qk_rope_head_dim`、`v.size(2) == v_head_dim`等）
- 支持的数据类型：bf16（`kv`/`latent_cache`/`k`/`v`）
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 设置设备
device = "cuda"
# 定义核心参数
bs = 8             
q_len = 16             
num_local_heads = 128    
kv_lora_rank = 512      
qk_rope_head_dim = 64    
qk_nope_head_dim = 128   
v_head_dim = 128         
# 构造输入张量（符合参数维度&类型要求）
kv = torch.empty(
    q_len, num_local_heads * (qk_nope_head_dim + v_head_dim),
    dtype=torch.bfloat16, device=device
)
latent_cache = torch.randn(
    q_len, kv_lora_rank + qk_rope_head_dim,
    dtype=torch.bfloat16, device=device
)
k = torch.randn(
    q_len, num_local_heads, qk_nope_head_dim + qk_rope_head_dim,
    dtype=torch.bfloat16, device=device
)
v = torch.randn(
    q_len, num_local_heads, v_head_dim,
    dtype=torch.bfloat16, device=device
)
# 调用fused_mla_normal_kv_element_wise算子
ret_code = torch.ops.sgl_kernel.fused_mla_normal_kv_element_wise(
    kv=kv,
    latent_cache=latent_cache,
    k=k,
    v=v,
    q_len=q_len,
    num_local_heads=num_local_heads,
    kv_lora_rank=kv_lora_rank,
    qk_nope_head_dim=qk_nope_head_dim,
    qk_rope_head_dim=qk_rope_head_dim,
    v_head_dim=v_head_dim
)
# 输出调用结果
print("fused_mla_normal_kv_element_wise computation completed")
print(f"Return code: {ret_code}")
print(f"Updated kv shape: {kv.shape}")
```
## 186. per_token_cast_to_fp8
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def per_token_cast_to_fp8(
    out: torch.Tensor,
    scale: torch.Tensor,
    input: torch.Tensor
) -> None
```
### 功能描述
per_token_cast_to_fp8 算子实现了按 Token（行）进行的动态 FP8 量化转换。该算子首先计算输入张量每一行（Token 维度）元素的绝对最大值（abs_max），根据 FP8 (E4M3) 的表示范围计算出对应的缩放因子（scale），然后将输入数据缩放并原地或异地转换为 FP8 格式。这种逐 Token 量化的方式能有效保留异常值信息，减少量化误差，提高 FP8 推理时的精度。
- 计算公式：
  首先逐 token 计算缩放因子（将输入缩放到 FP8 动态范围内）：
  $$
  \text{scale}[i] = \frac{\max_{j \in [0, \text{dim}-1]} \left| \text{input}[i, j] \right|}{\text{max\_fp8}}
  $$
  （注：$\text{max\_fp8}$为目标 FP8 格式的最大可表示正值，如 E4M3 格式对应$65504.0$，E5M2 格式对应$57344.0$）
  再逐元素执行缩放与 FP8 量化：
  $$
  \text{out}[i, j] = \text{quantize}_{\text{fp8}} \left( \frac{\text{input}[i, j]}{\text{scale}[i]} \right)
  $$
  其中$\text{quantize}_{\text{fp8}}(\cdot)$是将浮点值映射为目标 FP8 格式（E4M3/E5M2）的量化操作。
  其中元素的线性索引与二维位置的映射关系为：
  $$
  n = i \times \text{stride} + j
  $$
  各变量的维度与含义：
  - $\text{input} \in \mathbb{T}^{\text{num\_tokens} \times \text{dim}}$：输入张量，数据类型为$\mathbb{T}$（如 fp16/fp32），每行对应一个 token 的特征
  - $\text{out} \in \text{FP8}^{\text{num\_tokens} \times \text{dim}}$：输出张量，数据类型为 FP8（E4M3/E5M2），维度与$\text{input}$一致
  - $\text{scale} \in \mathbb{R}^{\text{num\_tokens}}$：逐 token 的缩放因子张量，每个元素对应一个 token 的缩放系数
  - $\text{num\_tokens}$：输入张量的 token 数量（行数）
  - $\text{dim}$：每个 token 的特征维度（列数）
  - $\text{max\_fp8}$：目标 FP8 格式的最大可表示正值（由 FP8 子类型决定）
  - $\text{stride}$：输入/输出张量的内存步长（每行元素的实际内存间隔，满足$\text{stride} \geq \text{dim}$以保证内存对齐）
  - $i$：token 的行索引，范围为$0 \leq i < \text{num\_tokens}$
  - $j$：token 的特征列索引，范围为$0 \leq j < \text{dim}$
  - $n$：张量元素的线性索引，范围为$0 \leq n < \text{num\_elems}$（$\text{num\_elems} = \text{num\_tokens} \times \text{stride}$）
### 参数说明
- **out** (torch.Tensor, 出参): 转换后的 FP8 张量，类型为 torch.float8_e4m3fn
- **scale** (torch.Tensor, 出参): 存储每一行计算得出的缩放因子张量，通常为 torch.float32 类型
- **input** (torch.Tensor, 入参): 待转换的输入张量，通常为 bfloat16 类型
### 返回值
无返回值，计算结果直接写入 out 和 scale 张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- input、scale 和 out 张量必须是内存连续的（contiguous）
- 输入张量的最后一维（hidden_size）必须是 128 的整数倍
- 算子内部采用 FP8 E4M3 标准，缩放因子计算逻辑中包含针对 448.0（E4M3 最大可表示值）的归一化处理
- 支持的数据类型：输入目前仅支持 bfloat16，输出支持 float8_e4m3fn
### 调用示例
```python
import torch
# 1. 基础配置
device = "cuda"  # 环境中可能是 MACA 设备，但 PyTorch 接口通常映射为 cuda
num_tokens = 8
hidden_size = 512  # 必须是 128 的倍数
dtype_in = torch.bfloat16
dtype_out = torch.float8_e4m3fn  # 对应源码中的 maca_fp8_e4m3
# 2. 准备张量
# 输入张量 (BF16)
input_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype_in, device=device)
# 输出张量 (FP8) - 必须是连续的 (contiguous)
out_tensor = torch.empty(num_tokens, hidden_size, dtype=dtype_out, device=device)
# Scale 张量 (FP32) 
# 源码逻辑：每 128 个元素计算一个 scale
# 因此 scale 长度为 总元素量 / 128
scale_size = (num_tokens * hidden_size) // 128
scale_tensor = torch.empty(scale_size, dtype=torch.float32, device=device)
# 3. 调用算子
import mcoplib.sgl_kernel
print("="*40)
print(f"Input shape:  {input_tensor.shape} ({input_tensor.dtype})")
print(f"Output shape: {out_tensor.shape} ({out_tensor.dtype})")
print(f"Scale shape:  {scale_tensor.shape} ({scale_tensor.dtype})")
print("-" * 40)
try:
    # 算子注册签名: (out, scale, input)
    torch.ops.sgl_kernel.per_token_cast_to_fp8(out_tensor, scale_tensor, input_tensor)
    print("Execution Status: SUCCESS")
    # 打印部分结果
    print(f"Scale snippet: {scale_tensor[:4].tolist()}")
except Exception as e:
    print(f"Execution Status: FAILED")
    print(f"Error Detail: {e}")
print("="*40)
```
## 201. CalSumOp
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
template<typename T, typename D>
void CalSumOp(
    const T* __restrict input,
    const unsigned char* __restrict mask,
    D* __restrict dst,
    int width,
    int height,
    int stride,
    cudaStream_t stream
);
```
### 功能描述
CalSumOp 是基于 CUDA 并行实现的**带掩码筛选的张量求和算子**。该算子先通过`unsigned char`类型的 mask 筛选输入张量中的有效元素，再对有效元素进行求和计算，最终将结果写入输出张量。同时利用`cudaStream_t`管理异步执行，减少内存访问开销，提升计算效率，适用于需要按掩码过滤后求和的张量处理场景（如特征筛选后的聚合计算）。
- 计算公式：
  $$
  \text{dst}[0] = \sum_{i=0}^{\text{height}-1} \sum_{j=0}^{\text{width}-1} \left( \text{input}[i \times \text{stride} + j] \times \mathbb{I}\{\text{mask}[i \times \text{stride} + j] \neq 0\} \right)
  $$
  其中：
  - $\text{input} \in \mathbb{R}^{\text{height} \times \text{width}}$ 是二维输入张量（实际内存布局由 stride 决定，支持非连续存储）
  - $\text{mask} \in \{0, 1\}^{\text{height} \times \text{width}}$ 是二进制掩码张量，与 input 的维度和内存布局完全对齐
  - $\text{dst} \in \mathbb{R}^1$ 是标量输出张量，唯一元素存储掩码筛选后的求和结果
  - $\text{width} \in \mathbb{N}^+$ 是输入张量的宽度维度（列数），表示每行的元素数量
  - $\text{height} \in \mathbb{N}^+$ 是输入张量的高度维度（行数），表示张量的总行数
  - $\text{stride} \in \mathbb{N}^+$ 是输入/掩码张量的内存步长，即相邻行首元素的内存偏移量（以元素个数为单位）
  - $\mathbb{I}\{\cdot\}$ 是指示函数，满足$\mathbb{I}\{P\}=1$（当条件$P$为真），$\mathbb{I}\{P\}=0$（当条件$P$为假）
  - $i \times \text{stride} + j$ 是张量中位置$(i,j)$对应的一维内存偏移索引（$i$为行索引，$j$为列索引）
  CUDA 并行计算的线程映射规则：
  $$
  \text{global\_tid} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
  $$
  $$
  i = \lfloor \text{global\_tid} / \text{width} \rfloor, \quad j = \text{global\_tid} \bmod \text{width}
  $$
  $$
  \text{partial\_sum}[\text{global\_tid}] = \text{input}[i \times \text{stride} + j] \times \mathbb{I}\{\text{mask}[i \times \text{stride} + j] \neq 0\}
  $$
  $$
  \text{dst}[0] = \text{blockReduceSum}(\text{partial\_sum}, \text{stream})
  $$
  其中：
  - $\text{global\_tid} \in \mathbb{N}$ 是 CUDA 线程的全局唯一索引
  - $\text{blockIdx.x} \in \mathbb{N}$ 是 CUDA 线程块的一维索引
  - $\text{blockDim.x} \in \mathbb{N}^+$ 是每个 CUDA 线程块的线程数量
  - $\text{threadIdx.x} \in \mathbb{N}$ 是线程块内的线程索引
  - $\text{partial\_sum}$ 是线程级局部求和缓存张量
  - $\text{blockReduceSum}(\cdot, \cdot)$ 是 CUDA 块级归约求和函数，基于指定的`stream`异步执行全局求和
### 参数说明
- **input** (const T* __restrict, 入参): 输入张量（模板类型 T），存储待求和的原始数据。
- **mask** (const unsigned char* __restrict, 入参): 掩码张量，通过元素值标记`input`中参与求和的有效元素（通常非 0 为有效）。
- **dst** (D* __restrict, 出参): 输出张量（模板类型 D），存储掩码筛选后的求和结果。
- **width** (int, 入参): 输入张量的宽度维度，用于确定维度范围。
- **height** (int, 入参): 输入张量的高度维度，用于确定维度范围。
- **stride** (int, 入参): 输入张量的步长，用于计算元素的内存偏移地址。
- **stream** (cudaStream_t, 入参): CUDA 流，用于管理算子的异步执行流程。
### 约束与调用
- 所有张量（`input`/`mask`/`dst`）必须存储在 CUDA 设备内存中。
- 支持的模板类型：`T`（输入类型）包括`char`、`short`、`float`等；`D`（输出类型）包括`double`等（以实际模板实例化为准）。
- `mask`的元素类型固定为`unsigned char`，有效元素的标记规则需与业务逻辑一致。
- `width`/`height`/`stride`的取值需匹配输入张量的实际维度布局，避免内存越界。
- 需依赖 CUDA 运行时环境，支持 CUDA 流操作及并行核函数执行。
### 调用示例
```C++
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"
#include "calsum.h"  
// 封装CUDA错误检查宏（避免隐藏bug）
#define CHECK_CUDA_ERR(err) \
    do { \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)
int main() {
    // 1. 定义输入参数（示例：3x2的矩阵，连续存储，stride=宽度）
    const int width = 2;     // 每行元素个数
    const int height = 3;    // 总行数
    const int stride = width;// 行步长（连续存储时等于width）
    const int num_elems = stride * height; // 总元素数（内存维度）
    // 2. 构造Host端输入数据
    // input: [1,2, 3,4, 5,6] → 3行2列的矩阵
    std::vector<float> input_h = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    // mask: [1,0, 1,1, 0,1] → 仅标记部分元素参与求和
    std::vector<unsigned char> mask_h = {1, 0, 1, 1, 0, 1};
    std::vector<double> dst_h(1, 0.0); // 输出为标量（仅1个元素）
    // 3. 分配Device端内存（带错误检查）
    float* input_d = nullptr;
    unsigned char* mask_d = nullptr;
    double* dst_d = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&input_d, num_elems * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&mask_d, num_elems * sizeof(unsigned char)));
    CHECK_CUDA_ERR(cudaMalloc(&dst_d, dst_h.size() * sizeof(double)));
    // 4. Host → Device 拷贝数据（带错误检查）
    CHECK_CUDA_ERR(cudaMemcpy(input_d, input_h.data(), num_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(mask_d, mask_h.data(), num_elems * sizeof(unsigned char), cudaMemcpyHostToDevice));
    // 初始化输出显存为0（可选，确保初始值干净）
    CHECK_CUDA_ERR(cudaMemset(dst_d, 0, dst_h.size() * sizeof(double)));
    // 5. 调用CalSumOp<float, double>算子（指定输入float，输出double，使用默认流）
    CalSumOp<float, double>(
        input_d,
        mask_d,
        dst_d,
        width,
        height,
        stride,
        (cudaStream_t)0  // 使用CUDA默认流
    );
    // 等待CUDA核函数执行完成（必须加，避免结果未计算完成就读取）
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    // 6. Device → Host 拷贝结果（带错误检查）
    CHECK_CUDA_ERR(cudaMemcpy(dst_h.data(), dst_d, dst_h.size() * sizeof(double), cudaMemcpyDeviceToHost));
    // 7. 验证结果（预期求和：1 + 3 + 4 + 6 = 14）
    std::cout << "CalSumOp<float, double> 运算结果：" << std::endl;
    std::cout << "掩码筛选后的求和值 = " << dst_h[0] << std::endl;
    // 8. 释放Device内存
    cudaFree(input_d);
    cudaFree(mask_d);
    cudaFree(dst_d);
    return 0;
}
```
## 202. CountNoZeroOp
### 支持的产品型号
- - Metax C500/C550/C588/C600
### 接口原型
```cpp
template<typename T, typename D>
void CountNoZeroOp(
    const T* __restrict input,
    const unsigned char* __restrict mask,
    D* __restrict dst,
    int width,
    int height,
    int stride,
    cudaStream_t stream
);
```
### 功能描述
CountNoZeroOp 算子实现了输入张量中非零元素的并行统计功能，支持结合掩码（mask）筛选特定位置后统计对应区域的非零元素数量。算子通过双版本 CUDA 核函数（带 mask/不带 mask）、共享内存缓存、线程同步与归约等优化手段，减少全局内存访问开销，提升并行计算效率，适用于张量有效元素计数的场景。
- 计算公式：
  $$
  \text{count}_{\text{non-zero}} = \sum_{i=0}^{\text{height}-1} \sum_{j=0}^{\text{width}-1} \mathbb{I}\left( \text{input}[i \times \text{stride} + j] \neq 0 \right)
  $$
  若启用掩码（mask）筛选统计区域，计算公式为：
  $$
  \text{count}_{\text{non-zero}}^{\text{masked}} = \sum_{i=0}^{\text{height}-1} \sum_{j=0}^{\text{width}-1} \mathbb{I}\left( \text{mask}[i \times \text{stride} + j] = 1 \land \text{input}[i \times \text{stride} + j] \neq 0 \right)
  $$
  其中：
  - $\mathbb{I}(\cdot)$ 是指示函数，满足条件时取值为$1$，否则为$0$
  - $\text{input}[i \times \text{stride} + j] \in \mathbb{T}$ 是输入张量在$(i,j)$位置的元素，$\mathbb{T}$ 为支持的数值类型（int、short、float、double、unsigned char 等）
  - $\text{mask}[i \times \text{stride} + j] \in \{0,1\}$ 是掩码张量在$(i,j)$位置的元素，$1$表示统计该位置，$0$表示跳过
  - $\text{count}_{\text{non-zero}} \in \mathbb{D}$ 是无掩码时的非零元素总数，$\mathbb{D}$ 为整数类型（如 int32、int64）
  - $\text{count}_{\text{non-zero}}^{\text{masked}} \in \mathbb{D}$ 是带掩码时的非零元素总数
  - $\text{height} \in \mathbb{N}^+$ 是输入张量的高度维度大小
  - $\text{width} \in \mathbb{N}^+$ 是输入张量的宽度维度大小
  - $\text{stride} \in \mathbb{N}^+$ 是输入张量的内存步长（单行列数对应的内存偏移量，满足$\text{stride} \geq \text{width}$）
  并行计算时，算子将二维统计任务拆分到线程块/线程中，单线程负责的子区域计数归约公式为：
  $$
  \text{count}_{\text{thread}} = \sum_{(i,j) \in \Omega_{\text{thread}}} \mathbb{I}\left( \text{cond}(i,j) \right)
  $$
  $$
  \text{count}_{\text{non-zero}} = \sum_{\text{thread} \in \text{all\_threads}} \text{count}_{\text{thread}}
  $$
  其中：
  - $\Omega_{\text{thread}}$ 是单个线程负责统计的$(i,j)$位置集合
  - $\text{cond}(i,j)$ 是统计条件（无掩码时为$\text{input}[i \times \text{stride} + j] \neq 0$，带掩码时为$\text{mask}[i \times \text{stride} + j] = 1 \land \text{input}[i \times \text{stride} + j] \neq 0$）
  - $\text{count}_{\text{thread}} \in \mathbb{D}$ 是单个线程统计得到的子区域非零元素数
### 参数说明
- **input** (const T* __restrict, 入参): 待统计的输入张量设备指针，存储原始数据
- **mask** (const unsigned char* __restrict, 入参): 掩码张量设备指针，用于指定需统计的元素位置（可选，未启用时执行全量非零统计）
- **dst** (D* __restrict, 出参): 输出张量设备指针，存储最终统计得到的非零元素总数
- **width** (int, 入参): 输入张量的宽度维度大小
- **height** (int, 入参): 输入张量的高度维度大小
- **stride** (int, 入参): 输入张量的内存步长（单行列数对应的内存偏移量）
- **stream** (cudaStream_t, 入参): CUDA 异步流，用于控制算子的执行流
### 返回值
无返回值，统计结果直接写入`dst`指针指向的设备存储区域
### 约束与调用
- 所有张量（input、mask、dst）必须分配在 CUDA 设备内存上
- 模板参数`T`支持的数据类型：int、short、float、double、unsigned char 等（需与输入数据类型匹配）
- 模板参数`D`需为整数类型（用于存储计数结果）
- 若使用 mask，mask 的元素数量需与 input 的统计区域维度匹配
- stride 需大于等于 width，确保张量内存布局的有效性
- 核函数线程配置需符合模板参数（如 NM_THREADS）的约束
### 调用示例
```C++
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"
#include "count_nozero.h"  

// 封装CUDA错误检查宏
#define CHECK_CUDA_ERR(err) \
    do { \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {
    // 1. 定义输入参数
    const int width = 3;    // 每行3个元素
    const int height = 2;   // 共2行
    const int stride = width;
    const int num_elems = stride * height;
    // 2. 构造Host端输入数据
    // Input: [ 0.0, 1.5, 0.0, 
    //         -2.0, 0.0, 3.0 ]
    std::vector<float> input_h = {0.0f, 1.5f, 0.0f, -2.0f, 0.0f, 3.0f};
    // Mask: [ 1, 1, 1,
    //         1, 1, 0 ]  -> 最后一个元素(3.0)被mask掉，应该不参与计数
    std::vector<unsigned char> mask_h = {1, 1, 1, 1, 1, 0};
    // Output: 注意这里改为 int，因为截图显示 CountNoZeroOp<float, int>
    std::vector<int> dst_h(1, 0); 
    // 3. 分配Device端内存
    float* input_d = nullptr;
    unsigned char* mask_d = nullptr;
    int* dst_d = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&input_d, num_elems * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&mask_d, num_elems * sizeof(unsigned char)));
    CHECK_CUDA_ERR(cudaMalloc(&dst_d, dst_h.size() * sizeof(int)));
    // 4. Host → Device 拷贝数据
    CHECK_CUDA_ERR(cudaMemcpy(input_d, input_h.data(), num_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(mask_d, mask_h.data(), num_elems * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemset(dst_d, 0, dst_h.size() * sizeof(int))); // 初始化输出为0
    // 5. 调用 CountNoZeroOp<float, int> 算子
    // 根据截图第157行：template void CountNoZeroOp<float, int>(...)
    CountNoZeroOp<float, int>(
        input_d,
        mask_d,     // 如果传 nullptr，则对应截图代码里的 else 分支（无mask模式）
        dst_d,
        width,
        height,
        stride,
        (cudaStream_t)0
    );
    // 等待核函数完成
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    // 6. Device → Host 拷贝结果
    CHECK_CUDA_ERR(cudaMemcpy(dst_h.data(), dst_d, dst_h.size() * sizeof(int), cudaMemcpyDeviceToHost));
    // 预期结果: 2
    std::cout << "CountNoZeroOp<float, int> 运算结果：" << std::endl;
    std::cout << "非零元素个数 (经过Mask筛选) = " << dst_h[0] << std::endl;
    // 8. 释放内存
    cudaFree(input_d);
    cudaFree(mask_d);
    cudaFree(dst_d);
    return 0;
}
```
## 203. MeanStdevOp
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
template<typename T, typename D>
void MeanStdevOp(
    const T* __restrict input,
    D* __restrict mean,
    D* __restrict std,
    int width,
    int height,
    int stride,
    cudaStream_t stream
);
```
### 功能描述
MeanStdevOp 算子实现了输入张量的**均值与标准差**的 CUDA 并行计算。该算子分两步完成：
1. 对输入数据做并行求和，计算出张量的均值；
2. 基于均值计算每个元素与均值差的平方和，进一步得到标准差。
这种并行化实现减少了计算耗时，常用于数据预处理、模型特征归一化等需要统计量的场景。
- 计算公式：
  $$
  \mu = \frac{1}{W \times H} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} x_{i,j}
  $$
  $$
  \sigma = \sqrt{\frac{1}{W \times H} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} (x_{i,j} - \mu)^2}
  $$
  其中：
  - $x_{i,j} \in \mathbb{R}$ 是输入张量$\text{input}$中第$i$行第$j$列的有效元素，实际存储位置为$\text{input}[i \times \text{stride} + j]$
  - $\mu \in \mathbb{R}$ 是输入张量所有有效元素的均值，计算结果写入$\text{mean}$指针指向的 CUDA 设备内存
  - $\sigma \in \mathbb{R}$ 是输入张量所有有效元素的总体标准差，计算结果写入$\text{std}$指针指向的 CUDA 设备内存
  - $W \in \mathbb{N}^+$ 是输入张量的有效宽度（对应算子参数$\text{width}$）
  - $H \in \mathbb{N}^+$ 是输入张量的高度（对应算子参数$\text{height}$）
  - $\text{stride} \in \mathbb{N}^+$ 是输入张量的存储步长（每行实际存储的元素数，需满足$\text{stride} \geq W$）
  - $\mathbb{N}^+$ 表示正整数集合
  若需计算样本标准差（除以$W \times H - 1$），算子可扩展为：
  $$
  \sigma_{\text{sample}} = \sqrt{\frac{1}{W \times H - 1} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} (x_{i,j} - \mu)^2}
  $$
  其中$\sigma_{\text{sample}} \in \mathbb{R}$ 为样本标准差，适用于小批量数据统计场景。
### 参数说明
- **input** (const T* __restrict, 入参): 待计算统计量的设备端输入张量指针
- **mean** (D* __restrict, 出参): 存储计算结果的设备端均值指针
- **std** (D* __restrict, 出参): 存储计算结果的设备端标准差指针
- **width** (int, 入参): 输入张量的有效宽度（每行有效元素数）
- **height** (int, 入参): 输入张量的高度（行数）
- **stride** (int, 入参): 输入张量的存储步长（每行实际存储的元素数，含填充）
- **stream** (cudaStream_t, 入参): 执行计算的 CUDA 流（管理异步任务队列）
### 约束与调用
- 所有张量（input、mean、std）必须位于 CUDA 设备内存中
- 支持的数据类型：模板参数 T/D 兼容 float 等数值类型（代码包含 float 的特化实现）
- 需满足`stride ≥ width`（步长需不小于有效宽度，匹配存储布局）
- **stream**需为有效的 CUDA 流对象（可传`cudaStreamDefault`使用默认流）
### 调用示例
```C++
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"
#include "meanstdev.h"  
// 封装CUDA错误检查宏（避免隐藏bug）
#define CHECK_CUDA_ERR(err) \
    do { \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)
int main() {
    // 1. 定义输入参数（2x2矩阵，连续存储无填充）
    const int width = 2;     // 有效宽度
    const int height = 2;    // 高度
    const int stride = width;// 存储步长（无填充，等于有效宽度）
    const int num_elems = stride * height; // 总存储元素数
    // 2. 构造Host端输入数据（2x2矩阵：[[1.0, 2.0], [3.0, 4.0]]）
    // 预期均值：(1+2+3+4)/4 = 2.5；预期总体标准差：sqrt(5/4) ≈ 1.11803
    std::vector<float> input_h = {1.0f, 2.0f, 3.0f, 4.0f};
    // Host端存储均值/标准差的变量
    float mean_h = 0.0f;
    float std_h = 0.0f;
    // 3. 分配Device端内存（带错误检查）
    float* input_d = nullptr;  // 输入张量设备内存
    float* mean_d = nullptr;   // 均值结果设备内存（单个值）
    float* std_d = nullptr;    // 标准差结果设备内存（单个值）
    CHECK_CUDA_ERR(cudaMalloc(&input_d, num_elems * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&mean_d, sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&std_d, sizeof(float)));
    // 4. Host → Device 拷贝输入数据（带错误检查）
    CHECK_CUDA_ERR(cudaMemcpy(input_d, input_h.data(), num_elems * sizeof(float), cudaMemcpyHostToDevice));
    // 5. 调用MeanStdevOp算子（模板参数T=float, D=float；使用默认CUDA流）
    MeanStdevOp<float, float>(
        input_d, 
        mean_d, 
        std_d, 
        width, 
        height, 
        stride, 
        cudaStreamDefault  // 显式使用默认流，替代(0)更规范
    );
    // 等待CUDA核函数执行完成（必须加，确保结果就绪）
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    // 6. Device → Host 拷贝结果（均值+标准差）
    CHECK_CUDA_ERR(cudaMemcpy(&mean_h, mean_d, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(&std_h, std_d, sizeof(float), cudaMemcpyDeviceToHost));
    // 7. 验证/打印结果
    std::cout << "MeanStdevOp<float, float> 运算结果：" << std::endl;
    std::cout << "输入矩阵：" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << input_h[i * stride + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "计算得到的均值：" << mean_h << "（预期：2.5）" << std::endl;
    std::cout << "计算得到的总体标准差：" << std_h << "（预期≈1.11803）" << std::endl;
    // 8. 释放Device内存
    CHECK_CUDA_ERR(cudaFree(input_d));
    CHECK_CUDA_ERR(cudaFree(mean_d));
    CHECK_CUDA_ERR(cudaFree(std_d));
    return 0;
}
```
## 204. fused_postprocess_yuv420
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```c
void fused_postprocess_yuv420(
    const float* __restrict inputData,
    uint8_t* __restrict yuv420Data,
    int width,
    int height,
    float scale,
    cudaStream_t stream
);
```
### 功能描述
fused_postprocess_yuv420 是融合式后处理算子，负责将输入的 float 类型数据转换为 YUV420 格式的图像数据。过程集成了颜色空间转换、数据缩放等步骤，通过融合操作减少内存访问次数，提升计算效率；基于 CUDA 执行，支持通过 CUDA 流实现异步计算。
- 计算公式：
  该算子实现$float$类型 RGB 数据到$uint8_t$类型 YUV420 格式的转换，包含数据缩放、颜色空间转换与格式排布：
  $$
  R' = \text{round}(inputData_{(y,x,0)} \times scale),\ G' = \text{round}(inputData_{(y,x,1)} \times scale),\ B' = \text{round}(inputData_{(y,x,2)} \times scale)
  $$
  $$
  R' = \max(0, \min(255, R')),\ G' = \max(0, \min(255, G')),\ B' = \max(0, \min(255, B'))
  $$
  $$
  Y(x,y) = 0.299 \times R' + 0.587 \times G' + 0.114 \times B'
  $$
  $$
  U(x,y) = -0.147 \times R' - 0.289 \times G' + 0.436 \times B' + 128
  $$
  $$
  V(x,y) = 0.615 \times R' - 0.515 \times G' - 0.100 \times B' + 128
  $$
  $$
  \text{yuv420Data}[y \times width + x] = \text{round}(Y(x,y))
  $$
  $$
  \text{yuv420Data}[height \times width + (y//2) \times (width//2) + (x//2)] = \text{round}(\text{avg}_{2×2}U(x,y))
  $$
  $$
  \text{yuv420Data}[height \times width + \frac{height \times width}{4} + (y//2) \times (width//2) + (x//2)] = \text{round}(\text{avg}_{2×2}V(x,y))
  $$
  其中：
  - $inputData \in \mathbb{R}^{height \times width \times 3}$：$float$类型输入张量，存储每个像素的$R、G、B$通道值（对应索引$0、1、2$）
  - $yuv420Data \in \mathbb{U}^{height \times width + \frac{height \times width}{2}}$：$uint8_t$类型输出张量，存储 YUV420 格式数据
  - $width, height \in \mathbb{N}^+$：目标图像的宽度与高度
  - $scale \in \mathbb{R}^+$：输入数据的缩放因子
  - $R', G', B' \in \mathbb{N}$：缩放并裁剪后的 RGB 通道值，范围$[0,255]$
  - $Y(x,y), U(x,y), V(x,y) \in \mathbb{R}$：转换后的 YUV 分量值
  - $\text{avg}_{2×2}U(x,y)$：$2×2$像素块内$U$分量的均值
  - $\text{avg}_{2×2}V(x,y)$：$2×2$像素块内$V$分量的均值
### 参数说明
- **inputData** (const float* __restrict, 入参): 待处理的 float 类型输入张量，存储原始数据
- **yuv420Data** (uint8_t* __restrict, 出参): 输出缓冲区，存储转换后的 YUV420 格式图像数据（uint8_t 类型）
- **width** (int, 入参): 目标图像的宽度
- **height** (int, 入参): 目标图像的高度
- **scale** (float, 入参): 数据缩放因子，用于调整输入数据的数值范围
- **stream** (cudaStream_t, 入参): CUDA 流对象，指定算子异步执行的流
### 约束与调用
- 所有指针对应的内存必须位于 CUDA 设备上
- inputData 的维度需与 width、height 对应的图像数据量匹配
- 支持的输入类型：float；输出类型：uint8_t
### 调用示例
```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"
#include "process_interface.h"  
int main() {
    // 1. 定义输入参数（2x2 RGB图像，简化验证）
    const int width = 2;                 // 图像宽度
    const int height = 2;                // 图像高度
    const float scale = 255.0f;          // 缩放因子（将float[0,1]映射到uint8[0,255]）
    const int input_elems = height * width * 3;  // RGB输入元素数（2*2*3=12）
    // YUV420输出大小：Y(2*2) + U(1*1) + V(1*1) = 6字节
    const int yuv420_elems = height * width + (height/2)*(width/2) + (height/2)*(width/2);
    // 2. 构造Host端输入数据（RGB全1.0f，缩放后为255）
    std::vector<float> inputData_h(input_elems, 1.0f);  
    // 存储YUV420结果的Host数组
    std::vector<uint8_t> yuv420Data_h(yuv420_elems, 0);  
    // 3. 分配Device端内存（带错误检查）
    float* inputData_d = nullptr;
    uint8_t* yuv420Data_d = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&inputData_d, input_elems * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&yuv420Data_d, yuv420_elems * sizeof(uint8_t)));
    // 4. Host → Device 拷贝数据
    CHECK_CUDA_ERR(cudaMemcpy(inputData_d, inputData_h.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
    // 5. 创建CUDA流并调用算子
    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERR(cudaStreamCreate(&stream));
    fused_postprocess_yuv420(inputData_d, yuv420Data_d, width, height, scale, stream);
    // 等待CUDA流执行完成
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
    // 6. Device → Host 拷贝结果
    CHECK_CUDA_ERR(cudaMemcpy(yuv420Data_h.data(), yuv420Data_d, yuv420_elems * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // 7. 验证结果（预期：Y全255，U/V均为128）
    std::cout << "fused_postprocess_yuv420 运算结果：" << std::endl;
    std::cout << "Y分量：";
    for (int i = 0; i < height*width; ++i) {
        std::cout << (int)yuv420Data_h[i] << " ";
    }
    std::cout << "\nU分量：" << (int)yuv420Data_h[height*width] << std::endl;
    std::cout << "V分量：" << (int)yuv420Data_h[height*width + 1] << std::endl;
    // 8. 释放资源
    cudaStreamDestroy(stream);
    cudaFree(inputData_d);
    cudaFree(yuv420Data_d);
    return 0;
}
```
## 205. fused_postprocess_nv12
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
void fused_postprocess_nv12(
    const float* __restrict inputData,
    uint8_t* __restrict nv12Data,
    int width,
    int height,
    float scale,
    cudaStream_t stream
);
```
### 功能描述
fused_postprocess_nv12 是融合式后处理算子，负责将输入的 float 类型数据转换为 NV12 格式的图像数据。集成了颜色空间转换、数据缩放等步骤，通过融合操作优化性能；基于 CUDA 执行，支持通过 CUDA 流实现异步计算。
- 计算公式：
  该算子实现$float$类型 RGB 数据到$uint8_t$类型 NV12 格式的转换，包含数据缩放、颜色空间转换与格式排布：
  $$
  R' = \text{round}(inputData_{(y,x,0)} \times scale),\ G' = \text{round}(inputData_{(y,x,1)} \times scale),\ B' = \text{round}(inputData_{(y,x,2)} \times scale)
  $$
  $$
  R' = \max(0, \min(255, R')),\ G' = \max(0, \min(255, G')),\ B' = \max(0, \min(255, B'))
  $$
  $$
  Y(x,y) = 0.299 \times R' + 0.587 \times G' + 0.114 \times B'
  $$
  $$
  U(x,y) = -0.147 \times R' - 0.289 \times G' + 0.436 \times B' + 128
  $$
  $$
  V(x,y) = 0.615 \times R' - 0.515 \times G' - 0.100 \times B' + 128
  $$
  $$
  \text{nv12Data}[y \times width + x] = \text{round}(Y(x,y))
  $$
  $$
  \text{nv12Data}[height \times width + 2 \times [(y//2) \times (width//2) + (x//2)]] = \text{round}(\text{avg}_{2×2}U(x,y))
  $$
  $$
  \text{nv12Data}[height \times width + 2 \times [(y//2) \times (width//2) + (x//2)] + 1] = \text{round}(\text{avg}_{2×2}V(x,y))
  $$
  其中：
  - $inputData \in \mathbb{R}^{height \times width \times 3}$：$float$类型输入张量，存储每个像素的$R、G、B$通道值（对应索引$0、1、2$）
  - $nv12Data \in \mathbb{U}^{height \times width + \frac{height \times width}{2}}$：$uint8_t$类型输出张量，存储 NV12 格式数据
  - $width, height \in \mathbb{N}^+$：目标图像的宽度与高度
  - $scale \in \mathbb{R}^+$：输入数据的缩放因子
  - $R', G', B' \in \mathbb{N}$：缩放并裁剪后的 RGB 通道值，范围$[0,255]$
  - $Y(x,y), U(x,y), V(x,y) \in \mathbb{R}$：转换后的 YUV 分量值
  - $\text{avg}_{2×2}U(x,y)$：$2×2$像素块内$U$分量的均值
  - $\text{avg}_{2×2}V(x,y)$：$2×2$像素块内$V$分量的均值
### 参数说明
- **inputData** (const float* __restrict, 入参): 待处理的 float 类型输入张量，存储原始数据
- **nv12Data** (uint8_t* __restrict, 出参): 输出缓冲区，存储转换后的 NV12 格式图像数据（uint8_t 类型）
- **width** (int, 入参): 目标图像的宽度
- **height** (int, 入参): 目标图像的高度
- **scale** (float, 入参): 数据缩放因子，用于调整输入数据的数值范围
- **stream** (cudaStream_t, 入参): CUDA 流对象，指定算子异步执行的流
### 约束与调用
- 所有指针对应的内存必须位于 CUDA 设备上
- inputData 的维度需与 width、height 对应的图像数据量匹配
- 支持的输入类型：float；输出类型：uint8_t
### 调用示例
```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"
#include "process_interface.h"  
int main() {
    // 1. 定义输入参数（2x2 RGB图像，简化验证）
    const int width = 2;                 // 图像宽度
    const int height = 2;                // 图像高度
    const float scale = 255.0f;          // 缩放因子（将float[0,1]映射到uint8[0,255]）
    const int input_elems = height * width * 3;  // RGB输入元素数（2*2*3=12）
    // NV12输出大小：Y(2*2) + UV(1*1*2) = 6字节
    const int nv12_elems = height * width + (height/2)*(width/2)*2;
    // 2. 构造Host端输入数据（RGB全1.0f，缩放后为255）
    std::vector<float> inputData_h(input_elems, 1.0f);  
    // 存储NV12结果的Host数组
    std::vector<uint8_t> nv12Data_h(nv12_elems, 0);  
    // 3. 分配Device端内存（带错误检查）
    float* inputData_d = nullptr;
    uint8_t* nv12Data_d = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&inputData_d, input_elems * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&nv12Data_d, nv12_elems * sizeof(uint8_t)));
    // 4. Host → Device 拷贝数据
    CHECK_CUDA_ERR(cudaMemcpy(inputData_d, inputData_h.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
    // 5. 创建CUDA流并调用算子
    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERR(cudaStreamCreate(&stream));
    fused_postprocess_nv12(inputData_d, nv12Data_d, width, height, scale, stream);
    // 等待CUDA流执行完成
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
    // 6. Device → Host 拷贝结果
    CHECK_CUDA_ERR(cudaMemcpy(nv12Data_h.data(), nv12Data_d, nv12_elems * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // 7. 验证结果（预期：Y全255，U=128、V=128）
    std::cout << "fused_postprocess_nv12 运算结果：" << std::endl;
    std::cout << "Y分量：";
    for (int i = 0; i < height*width; ++i) {
        std::cout << (int)nv12Data_h[i] << " ";
    }
    std::cout << "\nU分量：" << (int)nv12Data_h[height*width] << std::endl;
    std::cout << "V分量：" << (int)nv12Data_h[height*width + 1] << std::endl;
    // 8. 释放资源
    cudaStreamDestroy(stream);
    cudaFree(inputData_d);
    cudaFree(nv12Data_d);
    return 0;
}
```
## 206. fused_preprocess_yuv420
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
void fused_preprocess_yuv420(
    const uint8_t* __restrict yuv420Data,
    float* __restrict outputData,
    int width,
    int height,
    float scale,
    cudaStream_t stream
);
```
### 功能描述
fused_preprocess_yuv420 算子实现了 YUV420 格式图像数据的融合预处理操作，包含 YUV 颜色空间转换、数值归一化（偏移、范围裁剪）及缩放等步骤。通过融合多步操作降低内存访问开销，提升了图像预处理效率，适用于图像/视频类 AI 模型的输入预处理环节。
- 计算公式：
  $$
  \begin{align*}
  &\text{Step 1: 提取YUV420格式Y/U/V分量} \\
  &Y(i,j) = \text{yuv420Data}[i \times \text{width} + j] \\
  &U(i,j) = \text{yuv420Data}\left[\text{width} \times \text{height} + \left\lfloor \frac{i}{2} \right\rfloor \times \frac{\text{width}}{2} + \left\lfloor \frac{j}{2} \right\rfloor\right] \\
  &V(i,j) = \text{yuv420Data}\left[\text{width} \times \text{height} \times 5/4 + \left\lfloor \frac{i}{2} \right\rfloor \times \frac{\text{width}}{2} + \left\lfloor \frac{j}{2} \right\rfloor\right] \\
  &\text{Step 2: YUV420转RGB（BT.601标准）} \\
  &R(i,j) = 1.164 \times (Y(i,j)-16) + 1.596 \times (V(i,j)-128) \\
  &G(i,j) = 1.164 \times (Y(i,j)-16) - 0.813 \times (V(i,j)-128) - 0.391 \times (U(i,j)-128) \\
  &B(i,j) = 1.164 \times (Y(i,j)-16) + 2.018 \times (U(i,j)-128) \\
  &\text{Step 3: 数值裁剪与缩放} \\
  &R_{\text{norm}}(i,j) = \text{clip}\left(R(i,j), 0, 255\right) \times \frac{\text{scale}}{255} \\
  &G_{\text{norm}}(i,j) = \text{clip}\left(G(i,j), 0, 255\right) \times \frac{\text{scale}}{255} \\
  &B_{\text{norm}}(i,j) = \text{clip}\left(B(i,j), 0, 255\right) \times \frac{\text{scale}}{255} \\
  &\text{Step 4: 输出赋值} \\
  &\text{outputData}[i \times \text{width} \times 3 + j \times 3 + c] = 
  \begin{cases}
  R_{\text{norm}}(i,j) & c=0 \\
  G_{\text{norm}}(i,j) & c=1 \\
  B_{\text{norm}}(i,j) & c=2
  \end{cases}
  \end{align*}
  $$
  其中：
  - $Y(i,j) \in \mathbb{Z}_{[0,255]}$ 是 YUV420 格式中位置$(i,j)$的亮度分量
  - $U(i,j) \in \mathbb{Z}_{[0,255]}$ 是 YUV420 格式中位置$(i,j)$的蓝色色度分量（色度 U）
  - $V(i,j) \in \mathbb{Z}_{[0,255]}$ 是 YUV420 格式中位置$(i,j)$的红色色度分量（色度 V）
  - $i \in [0, \text{height}-1]$ 是图像行索引，$j \in [0, \text{width}-1]$ 是图像列索引
  - $\text{clip}(x, a, b) = \max\left(\min(x, b), a\right)$ 是数值裁剪函数，限制$x$在$[a,b]$范围内
  - $c \in \{0,1,2\}$ 是输出通道索引（0=R、1=G、2=B）
  - $\text{width} \in \mathbb{Z}_{>0}$ 是输入图像宽度，$\text{height} \in \mathbb{Z}_{>0}$ 是输入图像高度
  - $\text{scale} \in \mathbb{R}$ 是输出数据的缩放系数
  - $\text{yuv420Data} \in \mathbb{U8}^{(\text{width} \times \text{height} \times 3/2)}$ 是 YUV420 格式输入数据（CUDA 设备内存）
  - $\text{outputData} \in \mathbb{R}^{(\text{width} \times \text{height} \times 3)}$ 是预处理后输出数据（CUDA 设备内存）
### 参数说明
- **yuv420Data** (const uint8_t* __restrict, 入参): YUV420 格式的输入图像数据指针，需位于 CUDA 设备内存
- **outputData** (float* __restrict, 出参): 预处理后的输出数据指针，结果写入该 CUDA 设备内存地址
- **width** (int, 入参): 输入图像的宽度
- **height** (int, 入参): 输入图像的高度
- **scale** (float, 入参): 预处理后数据的缩放系数
- **stream** (cudaStream_t, 入参): CUDA 流，用于指定算子的异步执行流
### 返回值
无返回值，预处理结果直接写入`outputData`指针指向的内存。
### 约束与调用
- 所有数据指针（`yuv420Data`、`outputData`）必须位于 CUDA 设备内存
- 输入数据需严格符合 YUV420 格式（Y 分量占`width×height`字节，U、V 分量各占`width×height/4`字节）
- `width`、`height`需为正偶数（符合 YUV420 分辨率要求）
- `stream`需为有效的`cudaStream_t`对象
- 支持输入类型：`uint8_t`；输出类型：`float`
### 调用示例
```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"
#include "process_interface.h"  
int main() {
    // 1. 定义输入参数（选偶数分辨率，符合YUV420格式要求）
    const int width = 32;          // 图像宽度（偶数）
    const int height = 32;         // 图像高度（偶数）
    const float scale = 1.0f / 255.0f;  // 归一化缩放系数
    // YUV420数据大小：Y(width*height) + U(width*height/4) + V(width*height/4)
    const size_t yuv420_data_size = width * height * 3 / 2;
    // 输出数据大小：RGB三通道，float类型
    const size_t output_data_size = width * height * 3 * sizeof(float);
    // 2. 构造Host端输入数据（简单初始化YUV420数据）
    std::vector<uint8_t> yuv420_data_h(yuv420_data_size, 128);  // 初始化为灰度值128
    // 输出Host数组（存储RGB结果）
    std::vector<float> output_data_h(width * height * 3, 0.0f);
    // 3. 分配Device端内存
    uint8_t* yuv420_data_d = nullptr;
    float* output_data_d = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&yuv420_data_d, yuv420_data_size));
    CHECK_CUDA_ERR(cudaMalloc(&output_data_d, output_data_size));
    // 4. Host → Device 拷贝数据
    CHECK_CUDA_ERR(cudaMemcpy(yuv420_data_d, yuv420_data_h.data(), yuv420_data_size, 
                              cudaMemcpyHostToDevice));
    // 5. 创建CUDA流（可选，也可传0使用默认流）
    cudaStream_t stream;
    CHECK_CUDA_ERR(cudaStreamCreate(&stream));
    // 6. 调用fused_preprocess_yuv420算子
    fused_preprocess_yuv420(yuv420_data_d, output_data_d, width, height, scale, stream);
    // 7. 等待CUDA流执行完成
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
    // 8. Device → Host 拷贝结果
    CHECK_CUDA_ERR(cudaMemcpy(output_data_h.data(), output_data_d, output_data_size, 
                              cudaMemcpyDeviceToHost));
    // 9. 验证结果（打印前10个输出值）
    std::cout << "fused_preprocess_yuv420 运算结果（前10个RGB值）：" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << output_data_h[i] << " ";
    }
    std::cout << std::endl;
    // 10. 释放资源
    CHECK_CUDA_ERR(cudaStreamDestroy(stream));
    cudaFree(yuv420_data_d);
    cudaFree(output_data_d);
    return 0;
}
```
## 207. fused_preprocess_nv12
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```cpp
void fused_preprocess_nv12(
    const uint8_t* __restrict nv12Data,
    float* __restrict outputData,
    int width,
    int height,
    float scale,
    cudaStream_t stream
);
```
### 功能描述
fused_preprocess_nv12 算子实现了 NV12 格式图像数据的融合预处理操作，包含 YUV 颜色空间转换、数值归一化（偏移、范围裁剪）及缩放等步骤。通过融合多步操作减少内存访问次数，提升了图像预处理的计算效率，常用于图像/视频类 AI 模型的输入预处理阶段。
- 计算公式：
  $$
  \begin{align*}
  &\text{Step 1: 提取NV12格式Y/U/V分量} \\
  &Y(i,j) = \text{nv12Data}[i \times \text{width} + j] \\
  &\text{uv\_idx} = \text{width} \times \text{height} + \left\lfloor \frac{i}{2} \right\rfloor \times \text{width} + \left\lfloor \frac{j}{2} \right\rfloor \times 2 \\
  &U(i,j) = \text{nv12Data}[\text{uv\_idx}] \\
  &V(i,j) = \text{nv12Data}[\text{uv\_idx}+1] \\
  &\text{Step 2: YUV420(NV12)转RGB（BT.601标准）} \\
  &R(i,j) = 1.164 \times (Y(i,j)-16) + 1.596 \times (V(i,j)-128) \\
  &G(i,j) = 1.164 \times (Y(i,j)-16) - 0.813 \times (V(i,j)-128) - 0.391 \times (U(i,j)-128) \\
  &B(i,j) = 1.164 \times (Y(i,j)-16) + 2.018 \times (U(i,j)-128) \\
  &\text{Step 3: 数值裁剪与缩放} \\
  &R_{\text{norm}}(i,j) = \text{clip}\left(R(i,j), 0, 255\right) \times \frac{\text{scale}}{255} \\
  &G_{\text{norm}}(i,j) = \text{clip}\left(G(i,j), 0, 255\right) \times \frac{\text{scale}}{255} \\
  &B_{\text{norm}}(i,j) = \text{clip}\left(B(i,j), 0, 255\right) \times \frac{\text{scale}}{255} \\
  &\text{Step 4: 输出赋值} \\
  &\text{outputData}[i \times \text{width} \times 3 + j \times 3 + c] = 
  \begin{cases}
  R_{\text{norm}}(i,j) & c=0 \\
  G_{\text{norm}}(i,j) & c=1 \\
  B_{\text{norm}}(i,j) & c=2
  \end{cases}
  \end{align*}
  $$
  其中：
  - $Y(i,j) \in \mathbb{Z}_{[0,255]}$ 是 NV12 格式中位置$(i,j)$的亮度分量
  - $U(i,j) \in \mathbb{Z}_{[0,255]}$ 是 NV12 格式中位置$(i,j)$的蓝色色度分量（色度 U）
  - $V(i,j) \in \mathbb{Z}_{[0,255]}$ 是 NV12 格式中位置$(i,j)$的红色色度分量（色度 V）
  - $i \in [0, \text{height}-1]$ 是图像行索引，$j \in [0, \text{width}-1]$ 是图像列索引
  - $\text{clip}(x, a, b) = \max\left(\min(x, b), a\right)$ 是数值裁剪函数，限制$x$在$[a,b]$范围内
  - $c \in \{0,1,2\}$ 是输出通道索引（0=R、1=G、2=B）
  - $\text{width} \in \mathbb{Z}_{>0}$ 是输入图像宽度，$\text{height} \in \mathbb{Z}_{>0}$ 是输入图像高度
  - $\text{scale} \in \mathbb{R}$ 是输出数据的缩放系数
  - $\text{nv12Data} \in \mathbb{U8}^{(\text{width} \times \text{height} \times 3/2)}$ 是 NV12 格式输入数据（CUDA 设备内存）
  - $\text{outputData} \in \mathbb{R}^{(\text{width} \times \text{height} \times 3)}$ 是预处理后输出数据（CUDA 设备内存）
### 参数说明
- **nv12Data** (const uint8_t* __restrict, 入参): NV12 格式的输入图像数据指针，需位于 CUDA 设备内存
- **outputData** (float* __restrict, 出参): 预处理后的输出数据指针，结果写入该 CUDA 设备内存地址
- **width** (int, 入参): 输入图像的宽度
- **height** (int, 入参): 输入图像的高度
- **scale** (float, 入参): 预处理后数据的缩放系数
- **stream** (cudaStream_t, 入参): CUDA 流，用于指定算子的异步执行流
### 返回值
无返回值，预处理结果直接写入`outputData`指针指向的内存。
### 约束与调用
- 所有数据指针（`nv12Data`、`outputData`）必须位于 CUDA 设备内存
- 输入数据需严格符合 NV12 格式（Y 分量占`width×height`字节，UV 分量交替存储占`width×height/2`字节）
- `width`、`height`需为正偶数（符合 NV12 分辨率要求）
- `stream`需为有效的`cudaStream_t`对象
- 支持输入类型：`uint8_t`；输出类型：`float`
### 调用示例
```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"
#include "process_interface.h"  
int main() {
    // 1. 定义输入参数（选偶数分辨率，符合NV12格式要求）
    const int width = 32;          // 图像宽度（偶数）
    const int height = 32;         // 图像高度（偶数）
    const float scale = 1.0f / 255.0f;  // 归一化缩放系数
    // NV12数据大小：Y分量(width*height) + UV分量(width*height/2)
    const size_t nv12_data_size = width * height * 3 / 2;
    // 输出数据大小：RGB三通道，float类型
    const size_t output_data_size = width * height * 3 * sizeof(float);
    // 2. 构造Host端输入数据（简单初始化NV12数据）
    std::vector<uint8_t> nv12_data_h(nv12_data_size, 128);  // 初始化为灰度值128
    // 输出Host数组（存储RGB结果）
    std::vector<float> output_data_h(width * height * 3, 0.0f);
    // 3. 分配Device端内存
    uint8_t* nv12_data_d = nullptr;
    float* output_data_d = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&nv12_data_d, nv12_data_size));
    CHECK_CUDA_ERR(cudaMalloc(&output_data_d, output_data_size));
    // 4. Host → Device 拷贝数据
    CHECK_CUDA_ERR(cudaMemcpy(nv12_data_d, nv12_data_h.data(), nv12_data_size, 
                              cudaMemcpyHostToDevice));
    // 5. 创建CUDA流（可选，也可传0使用默认流）
    cudaStream_t stream;
    CHECK_CUDA_ERR(cudaStreamCreate(&stream));
    // 6. 调用fused_preprocess_nv12算子
    fused_preprocess_nv12(nv12_data_d, output_data_d, width, height, scale, stream);
    // 7. 等待CUDA流执行完成
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
    // 8. Device → Host 拷贝结果
    CHECK_CUDA_ERR(cudaMemcpy(output_data_h.data(), output_data_d, output_data_size, 
                              cudaMemcpyDeviceToHost));
    // 9. 验证结果（打印前10个输出值）
    std::cout << "fused_preprocess_nv12 运算结果（前10个RGB值）：" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << output_data_h[i] << " ";
    }
    std::cout << std::endl;
    // 10. 释放资源
    CHECK_CUDA_ERR(cudaStreamDestroy(stream));
    cudaFree(nv12_data_d);
    cudaFree(output_data_d);
    return 0;
}
```
## 332. fused_moe_gate_deepseek
### 支持的产品型号
- 支持CUDA的GPU设备（如NVIDIA A100、H100等）
### 接口原型
```python
def fused_moe_gate_deepseek(
    gating_outputs: torch.Tensor,
    correction_bias: torch.Tensor,
    out_routing_weights: torch.Tensor,
    out_selected_experts: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    num_fused_shared_experts: int,
    scale_factor: float,
    moegate_type: Optional[int] = None
) -> None
```
### 功能描述
fused_moe_gate_deepseek是DeepSeek混合专家（MoE）模型的融合门控算子，集成了门控输出校正、TopK专家选择、路由权重计算及重归一化等步骤。该算子通过融合MoE门控阶段的多个子操作，减少内存交互开销，提升大模型MoE层门控阶段的计算效率，是MoE模型推理中的核心门控计算组件。
- 计算公式：
    输入门控对数概率张量定义, 算子核心输入为MoE专家的门控对数概率张量，无额外残差输入：
    $$
    \text{gate\_logits}_{b,s,e} \in \mathbb{R}^{B \times S \times E}
    $$
    其中：$B$为batch size，$S$为序列长度，$E$为MoE结构的专家总数，$b \in [0,B-1], s \in [0,S-1], e \in [0,E-1]$。算子超参包含：$K$为激活专家数$\text{top\_k}$（正整数，$K \ll E$），$\epsilon$为数值稳定小量（$\epsilon>0$，通常取$10^{-6}$）。
    逐Token筛选Top-K门控对数概率与专家索引, 对每个Token（即每个$(b, s)$位置），在专家维度$E$上筛选出数值最大的$K$个门控对数概率值，同时获取其对应的专家索引：
    $$
    \begin{cases}
    \text{topk\_logits}_{b,s,k} = \text{TopK}(\text{gate\_logits}_{b,s,:}, K) \\
    \text{topk\_indices}_{b,s,k} = \text{ArgTopK}(\text{gate\_logits}_{b,s,:}, K)
    \end{cases}
    $$
    其中：$\text{topk\_logits} \in \mathbb{R}^{B \times S \times K}$，为筛选后的Top-K门控对数概率；$\text{topk\_indices} \in \mathbb{Z}^{B \times S \times K}$，为对应Top-K概率的专家编号（整数类型），$k \in [0,K-1]$。
    逐Token数值稳定的Top-K Softmax归一化, 为避免指数运算数值溢出，对每个Token的Top-K门控对数概率做去中心化处理后，执行Softmax计算，得到原始专家权重，同时加入稳定小量防止除零：
    $$
    \begin{align}
    \text{shifted\_logits}_{b,s,k} &= \text{topk\_logits}_{b,s,k} - \max_{k'=0}^{K-1}\text{topk\_logits}_{b,s,k'} \\
    \text{exp\_logits}_{b,s,k} &= \exp(\text{shifted\_logits}_{b,s,k}) \\
    \text{raw\_weights}_{b,s,k} &= \frac{\text{exp\_logits}_{b,s,k}}{\sum_{k'=0}^{K-1}\text{exp\_logits}_{b,s,k'} + \epsilon}
    \end{align}
    $$
    其中：$\text{raw\_weights} \in \mathbb{R}^{B \times S \times K}$，为Softmax归一化后的原始专家权重，权重取值范围$[0,1]$，$\exp(\cdot)$为自然指数函数，$\max(\cdot)$为取最大值函数。
    逐Token门控权重归一化校准, 对原始专家权重做二次归一化，保证每个Token的激活专家权重之和严格为1，是DeepSeek-MoE门控的核心融合逻辑，进一步提升数值稳定性：
    $$
    \text{gate\_weights}_{b,s,k} = \frac{\text{raw\_weights}_{b,s,k}}{\sum_{k'=0}^{K-1}\text{raw\_weights}_{b,s,k'} + \epsilon}
    $$
    其中：$\text{gate\_weights} \in \mathbb{R}^{B \times S \times K}$，为算子最终输出的专家门控权重，满足$\sum_{k=0}^{K-1}\text{gate\_weights}_{b,s,k} \approx 1$。
    算子最终双输出, 该融合算子为**双输出张量**，无额外中间张量输出，最终输出为：
    $$
    \text{outputs} = (\text{gate\_weights}, \text{topk\_indices})
    $$
    其中：主输出$\text{gate\_weights}$为浮点型专家权重，用于后续MoE专家输出的加权融合；辅输出$\text{topk\_indices}$为整型专家索引，用于筛选激活的MoE专家计算结果。
### 参数说明
- **gating_outputs** (torch.Tensor, 入参): 门控网络的输出张量，形状通常为（B, num_experts）（B为批大小，num_experts为专家总数），支持数据类型为float16/bfloat16
- **correction_bias** (torch.Tensor, 入参): 门控输出的校正偏置张量，形状为（num_experts），支持数据类型为float16/bfloat16
- **out_routing_weights** (torch.Tensor, 出参): 输出的专家路由权重张量，形状为（B, num_selected_experts）（num_selected_experts为单样本选中的专家数），数据类型为float16/bfloat16
- **out_selected_experts** (torch.Tensor, 出参): 输出的选中专家索引张量，形状为（B, num_selected_experts），数据类型为int32
- **topk** (int, 入参): 单样本全局选择的专家数量
- **renormalize** (bool, 入参): 是否对路由权重执行重归一化的标志位
- **num_expert_group** (int, 入参): 专家的分组数量
- **topk_group** (int, 入参): 每个专家分组内选择的专家数量
- **num_fused_shared_experts** (int, 入参): 融合的共享专家数量
- **scale_factor** (float, 入参): 路由权重的缩放因子
- **moegate_type** (Optional[int], 入参): MoE门控类型标识，仅支持取值0（DEEPSEEK）或1（SOFTMAX）；默认值为0
### 返回值
该算子无返回值，所有输出（路由权重、选中专家索引）均通过指定的出参张量（out_routing_weights、out_selected_experts）返回。
### 约束与调用
- 所有输入/出参张量必须部署在CUDA设备上
- gating_outputs的形状需与correction_bias的形状（num_experts）匹配，即gating_outputs最后一维为num_experts
- out_routing_weights、out_selected_experts的形状需为（B, num_selected_experts），其中num_selected_experts由topk、topk_group等参数共同确定
- 数据类型约束：gating_outputs、correction_bias支持float16/bfloat16；out_routing_weights支持float16/bfloat16；out_selected_experts必须为int32类型
- moegate_type仅允许取值0或1，传入其他值会触发参数校验错误
- topk的取值需符合逻辑约束（如num_fused_shared_experts>0时，topk需为0），具体以代码内的TORCH_CHECK校验规则为准
- scale_factor需为有效的浮点数值，用于路由权重的缩放计算
### 调用示例
```python
import torch
import mcoplib.op as op
# 设置计算设备
device = "cuda"
# 1. 定义超参数 
batch_size = 4
seq_len = 1024
num_tokens = batch_size * seq_len
num_experts = 256    
num_expert_group = 8      
topk = 8                  
topk_group = 3           
num_shared_experts = 0    
moegate_type = 0     
# 2. 创建输入张量
gating_outputs = torch.randn(
    num_tokens, num_experts,
    dtype=torch.float16,
    device=device,
    requires_grad=True
)
# correction_bias: [num_experts], 用于修正 gating 的偏置
correction_bias = torch.zeros(
    num_experts,
    dtype=torch.float16,
    device=device
)
# 3. 预分配输出张量 (根据 C++ 签名，需要传入 output buffer)
out_routing_weights = torch.empty(
    num_tokens, topk,
    dtype=torch.float32,
    device=device
)
# out_selected_experts: [tokens, topk], 索引必须是 int32
out_selected_experts = torch.empty(
    num_tokens, topk,
    dtype=torch.int32,
    device=device
)
# 其他标量参数
renormalize = True
scale_factor = 1.0 # 缩放因子，moegate_type=0 时通常配合使用
# 4. 调用算子
op.fused_moe_gate_deepseek(
    gating_outputs,         # 输入：门控分数
    correction_bias,        # 输入：偏置
    out_routing_weights,    # 输出：路由权重 (In-place 修改)
    out_selected_experts,   # 输出：选中专家索引 (In-place 修改)
    topk,                   # int: TopK 数量
    renormalize,            # bool: 是否重新归一化
    num_expert_group,       # int: 专家组数
    topk_group,             # int: 组内 TopK
    num_shared_experts,     # int: 共享专家数
    scale_factor,           # float: 缩放因子
    moegate_type            # int: Gate 类型 (0=DeepSeek)
)
print("fused_moe_gate_deepseek computation completed")
print(f"Input shape: {gating_outputs.shape}")
print(f"Output Weights shape: {out_routing_weights.shape}, dtype: {out_routing_weights.dtype}")
print(f"Output Indices shape: {out_selected_experts.shape}, dtype: {out_selected_experts.dtype}")
# 结果验证
print(f"\nExample output (first token):")
print(f"Selected Experts: {out_selected_experts[0].tolist()}")
print(f"Routing Weights: {out_routing_weights[0].tolist()}")
```
## 349. top_k_per_row
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def top_k_per_row(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int
) -> None
```
### 功能描述
top_k_per_row 是面向大语言模型计算流程的算子，用于对logits张量按行提取top k对应的索引：结合`rowStarts`与`rowEnds`限定每行的有效数据范围，针对logits的每一行计算top k结果，并将对应的索引写入`indices`张量，支撑模型中token候选集的选择等场景。
- 计算公式：
	当前版本中，top_k_per_row算子用于对每一行指定区间内的logits值，筛选出top-k的元素并记录其对应的索引。
	对于第 $r$ 行（$0 \leq r < \text{numRows}$），设该行在$\text{logits}$中的区间为 $[\text{start}_r, \text{end}_r]$（其中$\text{start}_r = \text{rowStarts}[r]$，$\text{end}_r = \text{rowEnds}[r]$），则：
	$$
	\text{indices}[r, i] = \arg\max_{\substack{x \in [\text{start}_r, \text{end}_r) \\ x \notin \{ \text{indices}[r, 0], ..., \text{indices}[r, i-1] \}}} \text{logits}[x]
	$$
	其中 $i$ 为top-k的位次索引，满足 $0 \leq i < k$（$k$ 为预设的top-k数量）。
	- $\text{logits}$：输入张量，存储待筛选的logits数值，其布局由$\text{stride0}$、$\text{stride1}$适配
	- $\text{rowStarts}$：输入张量，形状为 $(\text{numRows})$，$\text{rowStarts}[r]$ 表示第 $r$ 行在$\text{logits}$中的起始位置
	- $\text{rowEnds}$：输入张量，形状为 $(\text{numRows})$，$\text{rowEnds}[r]$ 表示第 $r$ 行在$\text{logits}$中的结束位置（开区间）
	- $\text{indices}$：输出张量，形状为 $(\text{numRows}, k)$，$\text{indices}[r, i]$ 存储第 $r$ 行第 $i$ 大logits对应的索引
	- $\text{numRows}$：标量，代表待处理的行数
	- $\text{stride0}$、$\text{stride1}$：标量，$\text{logits}$张量的存储步幅，用于定位元素的实际位置
	- $r$：行索引，满足 $0 \leq r < \text{numRows}$
	- $i$：top-k的位次索引，满足 $0 \leq i < k$
	- $\arg\max$：取最大值对应的索引，且需排除已选入top-k的索引
### 参数说明
- **logits** (torch.Tensor, 入参): 待计算的logits张量，存储各位置的概率输出数据，其内存布局需与`stride0`、`stride1`匹配
- **rowStarts** (torch.Tensor, 入参): 每行有效数据的起始位置张量，形状为`[numRows]`，标识logits中每一行的计算起始边界
- **rowEnds** (torch.Tensor, 入参): 每行有效数据的结束位置张量，形状为`[numRows]`，标识logits中每一行的计算结束边界
- **indices** (torch.Tensor, 出参): 输出张量，用于存储每行top k对应的索引结果，形状需匹配`numRows`与top k的数量要求
- **numRows** (int, 入参): logits张量的行数，即待计算top k的行数量
- **stride0** (int, 入参): logits张量的第一维度步幅，用于匹配其内存布局
- **stride1** (int, 入参): logits张量的第二维度步幅，用于匹配其内存布局
### 返回值
算子无返回值，top k对应的索引结果直接写入`indices`张量中
### 约束与调用
- 所有张量必须部署在Metax系列对应的加速计算设备上
- `logits`的行数必须与`numRows`的取值一致，且其内存布局需与`stride0`、`stride1`匹配
- `rowStarts`与`rowEnds`的元素数量需等于`numRows`，且每行的`rowStarts[i] < rowEnds[i]`需成立
- `indices`的形状需满足：行数等于`numRows`、列数等于top k的数量（需与算子内部top k配置匹配）
- 支持的数据类型：logits需为浮点类类型（如float16（at::ScalarType::Half）、bfloat16（at::ScalarType::BFloat16））
- `rowStarts`与`rowEnds`的取值需在logits对应维度的有效范围内
### 调用示例
```python
import torch
import mcoplib._C
# 配置参数
DEVICE = "cuda"
NUM_ROWS = 4        # 对应 C++ 中的 numRows
VOCAB_SIZE = 32000  # 词表大小（每行的长度）
TOPK = 2048         # 算子硬编码的 kTopK 值
# 初始化环境
torch.set_default_device(DEVICE)
# 1. 准备输入数据
# logits: [numRows, vocab_size]
logits = torch.randn(NUM_ROWS, VOCAB_SIZE, dtype=torch.float32)
# rowStarts / rowEnds: 每一行在当前批次中的起始和结束位置
row_starts = torch.zeros(NUM_ROWS, dtype=torch.int32)
row_ends = torch.full((NUM_ROWS,), VOCAB_SIZE, dtype=torch.int32)
# 2. 准备输出数据
indices = torch.empty((NUM_ROWS, TOPK), dtype=torch.int32)
# 3. 提取步长 (Strides)
stride0 = logits.stride(0)
stride1 = logits.stride(1)
# 4. 算子执行
print(f"Running top_k_per_row with NUM_ROWS={NUM_ROWS}...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用算子 (对应截图中的 top_k_per_row 签名)
torch.ops._C.top_k_per_row(
    logits,
    row_starts,
    row_ends,
    indices,
    NUM_ROWS,
    stride0,
    stride1
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 5. 打印输出结果
print("执行成功。")
print(f"Input Logits Shape: {logits.shape}")
print(f"Output Indices Shape: {indices.shape} (Expect: [{NUM_ROWS}, {TOPK}])")
```
## 354. top_k_per_row_decode
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int
) -> None
```
### 功能描述
top_k_per_row_decode 是面向大语言模型解码阶段的算子，用于对logits张量按行提取top k对应的索引：结合解码相关的next_n参数与序列长度张量seq_lens，针对logits的每一行（对应解码过程中各位置的概率输出）计算top k结果，将索引写入indices张量，支撑解码阶段的候选token选择流程。
- 计算公式：
	当前版本中，top_k_per_row_decode算子用于解码阶段，对每个序列对应的logits子区间，筛选出top-k的元素并记录其对应的索引。
	对于第 $r$ 个序列（行索引，$0 \leq r < \text{numRows}$），设其在$\text{logits}$中的目标区间为 $[\text{start}_r, \text{start}_r + \text{next\_n}]$（其中$\text{start}_r = \text{seq\_lens}[r]$），则：
	$$
	\text{indices}[r, i] = \arg\max_{\substack{x \in [\text{start}_r, \text{start}_r + \text{next\_n}) \\ x \notin \{ \text{indices}[r, 0], ..., \text{indices}[r, i-1] \}}} \text{logits}[x]
	$$
	其中 $i$ 为top-k的位次索引，满足 $0 \leq i < k$（$k$ 为预设的top-k筛选数量）。
	- $\text{logits}$：输入张量，存储待筛选的logits数值，其元素实际位置由$\text{stride0}$、$\text{stride1}$适配
	- $\text{next\_n}$：标量，每个序列对应的logits子区间的长度
	- $\text{seq\_lens}$：输入张量，形状为 $(\text{numRows})$，$\text{seq\_lens}[r]$ 表示第 $r$ 个序列在$\text{logits}$中的起始位置
	- $\text{indices}$：输出张量，形状为 $(\text{numRows}, k)$，$\text{indices}[r, i]$ 存储第 $r$ 个序列第 $i$ 大logits对应的索引
	- $\text{numRows}$：标量，代表待处理的序列（行）总数
	- $\text{stride0}$、$\text{stride1}$：标量，$\text{logits}$张量的存储步幅，用于定位元素的实际存储位置
	- $r$：序列（行）索引，满足 $0 \leq r < \text{numRows}$
	- $i$：top-k的位次索引，满足 $0 \leq i < k$
	- $\arg\max$：取最大值对应的索引，且需排除已选入top-k的索引
### 参数说明
- **logits** (torch.Tensor, 入参): 待计算的logits张量，存储各位置的概率输出数据，需与numRows、stride0/stride1匹配内存布局
- **next_n** (int, 入参): 解码阶段的步长控制参数，用于限定解码相关的计算范围
- **seq_lens** (torch.Tensor, 入参): 序列长度张量，标识每个样本的有效序列长度，支撑解码过程的长度边界判断
- **indices** (torch.Tensor, 出参): 输出张量，用于存储每行top k对应的索引结果，形状需匹配numRows与top k的数量要求
- **numRows** (int, 入参): logits张量的行数，即待计算top k的行数量
- **stride0** (int, 入参): logits张量的第一维度步幅，用于匹配其内存布局
- **stride1** (int, 入参): logits张量的第二维度步幅，用于匹配其内存布局
### 返回值
算子无返回值，top k对应的索引结果直接写入`indices`张量中
### 约束与调用
- 所有张量必须部署在Metax系列对应的加速计算设备上
- `logits`的行数必须与`numRows`的取值一致，且其内存布局需与`stride0`、`stride1`匹配
- `seq_lens`的元素数量需与解码阶段的样本数量（batch size）对应
- `indices`的形状需满足：行数等于`numRows`、列数等于top k的数量（需与算子内部top k配置匹配）
- 支持的数据类型：logits需为浮点类类型（如float16（at::ScalarType::Half）、bfloat16（at::ScalarType::BFloat16））
- `next_n`的取值需为正整数，且不超过解码阶段的有效计算范围
### 调用示例
```python
import torch
import mcoplib._C
# --- 配置参数 ---
DEVICE = "cuda"
NUM_ROWS = 4        # 批次大小 (Batch Size)
VOCAB_SIZE = 32000  # 词表大小
NEXT_N = 50         # 对应 C++ 签名的 next_n，即解码时每行取前多少个结果
torch.set_default_device(DEVICE)
# 1. 准备输入数据
logits = torch.randn(NUM_ROWS, VOCAB_SIZE, dtype=torch.float32)
seq_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
# 2. 准备输出数据
indices = torch.empty((NUM_ROWS, NEXT_N), dtype=torch.int32)
# 3. 提取步长 (Strides)
stride0 = logits.stride(0)
stride1 = logits.stride(1)
# 4. 算子执行
print(f"Running top_k_per_row_decode with NUM_ROWS={NUM_ROWS}, NEXT_N={NEXT_N}...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用算子
torch.ops._C.top_k_per_row_decode(
    logits,      # const torch::Tensor& logits
    NEXT_N,      # int64_t next_n
    seq_lens,    # const torch::Tensor& seq_lens
    indices,     # torch::Tensor& indices
    NUM_ROWS,    # int64_t numRows
    stride0,     # int64_t stride0
    stride1      # int64_t stride1
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 5. 打印输出结果
print("执行成功。")
print("-" * 30)
print(f"Input Logits Shape:   {logits.shape}")
print(f"Input Seq Lens:       {seq_lens.tolist()}")
print(f"Output Indices Shape: {indices.shape} (Expect: [{NUM_ROWS}, {NEXT_N}])")
# 简单验证一下输出是否由 tensor 填充
print(f"Sample Output (Row 0, first 10 indices): {indices[0, :10].tolist()}")
```
## 355. cp_gather_indexer_k_cache
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def cp_gather_indexer_k_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor
) -> None
```
### 功能描述
cp_gather_indexer_k_cache 算子是用于收集量化K缓存的提取函数，主要服务于大语言模型的推理流程：从KV缓存中针对性地读取量化后的K部分数据，实现缓存数据的高效提取，支撑模型推理过程中的K缓存调用。
- 计算公式：
	当前版本中，cp_gather_indexer_k_cache算子用于从K缓存中，结合块表与序列长度信息，收集对应位置的K数据，生成目标K张量。
	$$
	\text{dst\_k}[t, :] = \text{kv\_cache}[\text{block\_idx}, \text{block\_inner\_pos}, :\text{head\_dim}]
	$$
	其中，对于任意token全局索引 $t$（$0 \leq t < \text{num\_tokens}$），需通过以下步骤确定索引：
	1. 确定token所属batch索引：$\text{batch\_idx}$ 满足 $\text{seq\_lens}[\text{batch\_idx}] \leq t < \text{seq\_lens}[\text{batch\_idx} + 1]$
	2. 计算batch内局部偏移：$\text{batch\_token\_offset} = t - \text{seq\_lens}[\text{batch\_idx}]$
	3. 定位缓存块索引：$\text{block\_idx} = \text{block\_table}[\text{batch\_idx}, \text{batch\_token\_offset} // \text{block\_size}]$
	4. 定位块内token位置：$\text{block\_inner\_pos} = \text{batch\_token\_offset} \% \text{block\_size}$
	参数说明：
	- $\text{kv\_cache}$：输入张量，形状记为 $(\text{num\_blocks}, \text{block\_size}, \text{cache\_stride})$，存储K缓存数据
	- $\text{dst\_k}$：输出张量，形状为 $(\text{num\_tokens}, \text{head\_dim})$，存储收集到的K数据
	- $\text{block\_table}$：输入张量，形状为 $(\text{batch\_size}, \text{num\_blocks})$，记录每个batch对应的缓存块索引映射
	- $\text{seq\_lens}$：输入张量，形状为 $(\text{batch\_size} + 1)$，标记各batch的token起始/结束位置（$\text{seq\_lens}[0] = 0$，$\text{seq\_lens}[\text{batch\_size}] = \text{num\_tokens}$）
	- $t$：token全局索引，满足 $0 \leq t < \text{num\_tokens}$
	- $\text{block\_size}$：缓存块的token容量，为标量正整数
	- $\text{head\_dim}$：注意力头的维度，为标量正整数，用于截取$\text{kv\_cache}$的对应维度区间
### 参数说明
- **kv_cache** (torch.Tensor, 入参): 量化的KV缓存张量，存储待提取的K缓存数据，形状为 `[num_blocks, block_size, cache_stride]`
- **dst_k** (torch.Tensor, 出参): 输出张量，用于存储收集后的K缓存数据，形状为 `[num_tokens, head_dim]`
- **block_table** (torch.Tensor, 入参): 块索引表张量，用于定位KV缓存中的目标块，形状为 `[batch_size, num_blocks]`
- **cu_seq_lens** (torch.Tensor, 入参): 序列长度累积张量，标识每个batch的序列长度边界，形状为 `[batch_size + 1]`
### 返回值
算子无返回值，收集后的K缓存数据直接写入`dst_k`张量中
### 约束与调用
- 所有张量必须部署在Metax系列对应的加速计算设备上
- `kv_cache`的`num_blocks`维度大小需与`block_table`的`num_blocks`维度大小一致
- `cu_seq_lens`的长度必须等于`block_table`的`batch_size` + 1
- 支持的数据类型：需匹配量化K缓存对应的类型（如量化整数类型、压缩浮点类型等）
- `dst_k`的形状需与实际待收集的token数量、头维度要求一致（即`[num_tokens, head_dim]`需满足推理流程的数据维度约束）
### 调用示例
```python
import torch
import mcoplib._C 
# --- 配置参数 ---
DEVICE = "cuda"
BATCH_SIZE = 2        
HEAD_DIM = 64        
BLOCK_SIZE = 16        
NUM_PHYSICAL_BLOCKS = 100 
CACHE_STRIDE = HEAD_DIM 
torch.set_default_device(DEVICE)
torch.manual_seed(0) 
# 1. 准备输入数据
# 1.1 构造序列长度信息
seq_lens_list = [10, 20]
total_tokens = sum(seq_lens_list)
# cu_seq_lens: [0, 10, 30] -> Shape: [batch_size + 1]
cu_seq_lens = torch.tensor([0, 10, 30], dtype=torch.int32)
# 1.2 构造 Block Table (逻辑块 -> 物理块 ID 的映射)
max_blocks = 2
block_table = torch.full((BATCH_SIZE, max_blocks), -1, dtype=torch.int32)
block_table[0, 0] = 7
block_table[1, 0] = 3
block_table[1, 1] = 5
# 1.3 构造物理 KV Cache (Source)
# Shape: [num_blocks, block_size, cache_stride]
# 这里用 float16 模拟半精度缓存，如果是量化缓存(int8/fp8)请修改 dtype
kv_cache = torch.randn(NUM_PHYSICAL_BLOCKS, BLOCK_SIZE, CACHE_STRIDE, dtype=torch.float16)
# 2. 准备输出数据 (Destination)
dst_k = torch.empty((total_tokens, HEAD_DIM), dtype=torch.float16)
# 3. 算子执行
print(f"Running cp_gather_indexer_k_cache...")
print(f"  Batch Size: {BATCH_SIZE}, Total Tokens: {total_tokens}")
print(f"  KV Cache Shape: {kv_cache.shape}")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用算子
torch.ops._C_cache_ops.cp_gather_indexer_k_cache(
    kv_cache,      # const torch::Tensor& kv_cache
    dst_k,         # torch::Tensor& dst_k (输出)
    block_table,   # const torch::Tensor& block_table
    cu_seq_lens    # const torch::Tensor& cu_seq_lens
)
if torch.cuda.is_available():
    torch.cuda.synchronize()

# 4. 打印输出结果与验证
print("执行成功。")
print("-" * 30)
print(f"Dst K Shape: {dst_k.shape} (Expect: [{total_tokens}, {HEAD_DIM}])")
```
## 368. cp_gather_indexer_k_quant_cache
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor
) -> void
```
### 功能描述
cp_gather_indexer_k_quant_cache 是面向 GPU 优化的量化K缓存提取算子，用于从分块存储的量化KV缓存中，收集对应token的K数据及配套缩放因子，减少显存冗余读写开销，常用于大语言模型的推理流程（需配合量化KV缓存的存储格式使用）。
- 计算公式：
	当前版本中，cp_gather_indexer_k_quant_cache算子用于从量化的K缓存中，结合块表与序列长度信息，收集对应位置的量化数据，生成目标K张量及配套缩放因子张量。
	对于任意token全局索引 $t$（$0 \leq t < \text{num\_tokens}$）：
	1. 确定token所属batch索引 $\text{batch\_idx}$，满足 $\text{seq\_lens}[\text{batch\_idx}] \leq t < \text{seq\_lens}[\text{batch\_idx} + 1]$
	2. 计算token在所属batch内的局部偏移：$\text{batch\_token\_offset} = t - \text{seq\_lens}[\text{batch\_idx}]$
	3. 定位缓存块索引：$\text{block\_idx} = \text{block\_table}[\text{batch\_idx}, \text{batch\_token\_offset} // \text{block\_size}]$
	4. 定位块内token位置：$\text{block\_inner\_pos} = \text{batch\_token\_offset} \% \text{block\_size}$
	5. 收集量化数据与缩放因子：
	$$
	\text{dst\_k}[t, :] = \text{kv\_cache}[\text{block\_idx}, \text{block\_inner\_pos}, :\text{head\_dim / quant\_block\_size * 4}]
	$$
	$$
	\text{dst\_scale}[t, :, :] = \text{kv\_cache}[\text{block\_idx}, \text{block\_inner\_pos}, \text{head\_dim / quant\_block\_size * 4}:]
	$$
	其中：
	- $\text{kv\_cache}$：输入张量，形状为 $(\text{num\_blocks}, \text{block\_size}, \text{cache\_stride})$，存储量化后的K缓存（包含量化值与对应缩放因子）
	- $\text{dst\_k}$：输出张量，形状为 $(\text{num\_tokens}, \text{head\_dim / quant\_block\_size * 4})$，存储收集到的量化K数据
	- $\text{dst\_scale}$：输出张量，形状为 $(\text{num\_tokens}, \text{heads}, \text{dim / quant\_block\_size})$，存储与量化K数据匹配的缩放因子
	- $\text{block\_table}$：输入张量，形状为 $(\text{batch\_size}, \text{num\_blocks})$，记录每个batch对应的缓存块索引映射
	- $\text{seq\_lens}$：输入张量，形状为 $(\text{batch\_size} + 1)$，标记各batch的token起始/结束位置（$\text{seq\_lens}[0]=0$，$\text{seq\_lens}[\text{batch\_size}]=\text{num\_tokens}$）
	- $t$：token全局索引，满足 $0 \leq t < \text{num\_tokens}$
	- $\text{batch\_idx}$：token所属batch的索引，满足 $0 \leq \text{batch\_idx} < \text{batch\_size}$
	- $\text{block\_size}$：缓存块的token容量，为标量正整数
	- $\text{head\_dim}$：注意力头的维度，$\text{quant\_block\_size}$为量化块尺寸，二者共同划分$\text{kv\_cache}$的维度区间
### 参数说明
- **kv_cache** (torch.Tensor, 入参): 量化后的KV缓存张量，形状为 `[num_blocks, block_size, cache_stride]`，存储分块形式的量化K缓存数据
- **dst_k** (torch.Tensor, 出参): 输出的K数据张量，形状为 `[num_tokens, head_dim]`，用于存储收集后的K数据
- **dst_scale** (torch.Tensor, 出参): 输出的量化缩放因子张量，形状为 `[num_tokens, head_dim / quant_block_size * 4]`，用于存储量化K对应的缩放信息
- **block_table** (torch.Tensor, 入参): 分块索引表张量，形状为 `[batch_size, num_blocks]`，记录各批次对应的缓存分块索引
- **cu_seq_lens** (torch.Tensor, 入参): 序列长度累积张量，形状为 `[batch_size + 1]`，记录各批次序列长度的累积信息
### 返回值
无返回值，收集后的K数据与缩放因子直接存储在 `dst_k`、`dst_scale` 出参张量中
### 约束与调用
- 所有张量必须位于 CUDA 设备上
- 各张量的维度需严格匹配（如 `num_tokens` 需与序列长度、分块信息保持一致）
- 需配合预设量化规则使用：`quant_block_size` 需为 `head_dim` 的约数
- 数据类型约束：`kv_cache`/`dst_k` 为低精度量化类型（如 int4/int8），`dst_scale` 为浮点类型（如 float32），`block_table`/`cu_seq_lens` 为 int64（torch.int64）
### 调用示例
```python
import torch
import mcoplib._C 

# --- 配置参数 ---
DEVICE = "cuda"
BATCH_SIZE = 2          # 批次大小
HEAD_DIM = 64           # 注意力头维度
BLOCK_SIZE = 16         # 每个 Block 包含的 Token 数
NUM_PHYSICAL_BLOCKS = 100 
CACHE_STRIDE = HEAD_DIM # Cache Stride
QUANT_BLOCK_SIZE = 32   

torch.set_default_device(DEVICE)
torch.manual_seed(0) 

# 1. 准备输入数据
seq_lens_list = [10, 20]
total_tokens = sum(seq_lens_list)
cu_seq_lens = torch.tensor([0, 10, 30], dtype=torch.int32)

# 1.2 构造 Block Table
max_blocks = 2
block_table = torch.full((BATCH_SIZE, max_blocks), -1, dtype=torch.int32)
block_table[0, 0] = 7
block_table[1, 0] = 3
block_table[1, 1] = 5

# 1.3 构造物理 KV Cache (Source) - 量化版
kv_cache = torch.randint(-127, 127, (NUM_PHYSICAL_BLOCKS, BLOCK_SIZE, CACHE_STRIDE), dtype=torch.int8)

# 2. 准备输出数据 (Destination)
dst_k = torch.empty((total_tokens, HEAD_DIM), dtype=torch.int8)

# 2.2 dst_scale (量化缩放因子)
scale_dim_bytes = (HEAD_DIM // QUANT_BLOCK_SIZE) * 4
dst_scale = torch.empty((total_tokens, scale_dim_bytes), dtype=torch.uint8)

# 3. 算子执行
print(f"Running cp_gather_indexer_k_quant_cache...")
print(f"  Shape Info:")
print(f"  - KV Cache: {kv_cache.shape} (int8)")
print(f"  - Dst K:    {dst_k.shape} (int8)")
print(f"  - Dst Scale:{dst_scale.shape} (uint8 view of float32 scales)")
print(f"    (Calculated from HEAD_DIM={HEAD_DIM}, QUANT_BLOCK_SIZE={QUANT_BLOCK_SIZE})")

if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._C.cp_gather_indexer_k_quant_cache(
    kv_cache,      # const torch::Tensor& kv_cache
    dst_k,         # torch::Tensor& dst_k
    dst_scale,     # torch::Tensor& dst_scale
    block_table,   # const torch::Tensor& block_table
    cu_seq_lens    # const torch::Tensor& cu_seq_lens
)

if torch.cuda.is_available():
    torch.cuda.synchronize()

# 4. 打印输出结果与验证
print("执行成功。")
print("-" * 30)
print(f"Output Dst K Sample (Row 0, first 10): {dst_k[0, :10].tolist()}")
print(f"Output Dst Scale Shape: {dst_scale.shape}")
```
## 396. batched_moe_align_block_size
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def batched_moe_align_block_size(
    max_tokens_per_batch: int64_t,
    block_size: int64_t,
    num_experts: int64_t,
    sorted_ids: torch::Tensor,
    expert_ids: torch::Tensor,
    num_tokens_post_pad: torch::Tensor
) -> None
```
### 功能描述
batched_moe_align_block_size 是面向MoE（混合专家模型）场景优化的批处理算子，用于对齐token分块大小，同步调整token的排序索引、对应专家ID及padding后的token数量，减少MoE计算中数据读写与分块匹配的开销，适用于大语言模型MoE架构的训练与推理流程。
- 计算公式：
	当前版本中，batched_moe_align_block_size算子用于混合专家（MoE）场景下，将每个专家的token数量对齐到指定block_size（不足block_size整数倍的部分填充），同时调整token索引与专家归属的对应关系。
	$$
	\text{num\_tokens\_post\_pad}[e] = \left\lceil \frac{\text{expert\_num\_tokens}[e]}{\text{block\_size}} \right\rceil \times \text{block\_size}
	$$
	对于任意位置索引 $i$（$0 \leq i < \text{max\_tokens\_per\_batch}$），若 $i$ 属于专家 $e$ 的token区域（即 $\sum_{e'=0}^{e-1} \text{num\_tokens\_post\_pad}[e'] \leq i < \sum_{e'=0}^{e} \text{num\_tokens\_post\_pad}[e']$）：
	$$
	\begin{cases}
	\text{sorted\_ids}[i] = \text{原专家}e\text{的token索引}, & i - \sum_{e'=0}^{e-1} \text{num\_tokens\_post\_pad}[e'] < \text{expert\_num\_tokens}[e] \\
	\text{sorted\_ids}[i] = \text{填充值}, & \text{否则}
	\end{cases}
	$$
	$$
	\text{expert\_ids}[i] = e
	$$
	其中：
	- $\text{max\_tokens\_per\_batch}$：标量，代表单个batch允许的最大token总数，用于约束张量维度规模
	- $\text{block\_size}$：标量，代表分块对齐的目标尺寸，为正整数
	- $\text{expert\_num\_tokens}$：输入张量，形状为 $(E)$（$E$ 为专家总数），$\text{expert\_num\_tokens}[e]$ 表示专家 $e$ 初始的token数量
	- $\text{sorted\_ids}$：输入输出张量，形状为 $(\text{max\_tokens\_per\_batch})$，存储按专家分组并对齐block_size后的token索引，填充区域取值为预设填充值
	- $\text{expert\_ids}$：输入输出张量，形状为 $(\text{max\_tokens\_per\_batch})$，$\text{expert\_ids}[i]$ 表示位置 $i$ 的token对应的专家id
	- $\text{num\_tokens\_post\_pad}$：输出张量，形状为 $(E)$，$\text{num\_tokens\_post\_pad}[e]$ 表示专家 $e$ 对齐block_size后的token数量
	- $e$：专家索引，满足 $0 \leq e < E$
	- $\lceil \cdot \rceil$：向上取整运算，用于确保 $\text{num\_tokens\_post\_pad}[e]$ 为 $\text{block\_size}$ 的整数倍
	- $\sum_{e'=0}^{e-1} \text{num\_tokens\_post\_pad}[e']$：表示专家 $e$ 对应的token区域在全局张量中的起始位置（当 $e=0$ 时，求和结果为0）
### 参数说明
- **max_tokens_per_batch** (int64_t, 入参): 单个批次支持的最大token数量，需为正整数
- **block_size** (int64_t, 入参): MoE计算中每个分块的token数量，需为正整数
- **num_experts** (int64_t, 入参): MoE架构中专家的总数量，需为正整数
- **sorted_ids** (torch::Tensor, 入参): 存储token排序索引的张量，操作会直接修改该张量内容
- **expert_ids** (torch::Tensor, 入参): 存储每个token对应专家ID的张量，操作会直接修改该张量内容
- **num_tokens_post_pad** (torch::Tensor, 入参): 存储padding后各分块token数量的张量，操作会直接修改该张量内容
### 返回值
void: 无返回值，所有操作结果直接作用于输入的张量参数
### 约束与调用
- 所有张量必须位于CUDA设备上
- sorted_ids、expert_ids、num_tokens_post_pad的数据类型需为int64（即at::ScalarType::Long）
- max_tokens_per_batch、block_size、num_experts需为大于0的整数
- 张量的形状需与批处理规模、token数量及专家数量匹配
### 调用示例
```python
import torch
import mcoplib._moe_C
# --- 配置参数 ---
DEVICE = "cuda"
BLOCK_SIZE = 8       
NUM_EXPERTS = 4      
NUM_TOKENS = 32        
DTYPE_INT = torch.int32
torch.set_default_device(DEVICE)
# 1. 准备输入数据
expert_num_tokens = torch.tensor([7, 11, 6, 8], dtype=DTYPE_INT)
# 2. 准备输出数据
max_capacity_arg = NUM_TOKENS
aligned_capacity = (max_capacity_arg + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
rect_sorted_size = NUM_EXPERTS * aligned_capacity
rect_block_size = rect_sorted_size // BLOCK_SIZE
sorted_ids = torch.empty(rect_sorted_size, dtype=DTYPE_INT)
block_ids_out = torch.empty(rect_block_size, dtype=DTYPE_INT)
num_tokens_post_pad = torch.empty(1, dtype=DTYPE_INT) # 必须是大小为 1 的标量 Tensor
# 3. 算子执行
print(f"Running batched_moe_align_block_size with NUM_EXPERTS={NUM_EXPERTS}, BLOCK_SIZE={BLOCK_SIZE}...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 调用算子
torch.ops._moe_C.batched_moe_align_block_size(
    max_capacity_arg,    # int scalar
    BLOCK_SIZE,          # int scalar
    expert_num_tokens,   # const torch::Tensor& (int32)
    sorted_ids,          # torch::Tensor& (int32)
    block_ids_out,       # torch::Tensor& (int32)
    num_tokens_post_pad  # torch::Tensor& (int32, size=[1])
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
# 4. 打印输出结果
print("执行成功。")
print("-" * 30)
print(f"Input Expert Tokens:      {expert_num_tokens.tolist()}")
print(f"Output Sorted IDs Shape:  {sorted_ids.shape} (Expect: [{rect_sorted_size}])")
print(f"Output Block IDs Shape:   {block_ids_out.shape} (Expect: [{rect_block_size}])")
print(f"Total Post-Pad Tokens:    {num_tokens_post_pad.item()}")
# 简单验证一下输出
print(f"Sample Sorted IDs (First 10): {sorted_ids[:10].tolist()}")
```
## 397. moe_lora_align_block_size
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor
) -> void
```
### 功能描述
moe_lora_align_block_size 是面向 MoE-LoRA 混合架构优化的批处理算子，用于对齐token分块大小，同步处理token的LoRA映射关系、专家ID、排序索引及padding后token数量等信息，减少MoE-LoRA场景下数据分块与映射的显存读写开销，适用于大语言模型MoE-LoRA架构的训练与推理流程。
- 计算公式：
	当前版本中，moe_lora_align_block_size算子用于MoE结合LoRA的场景，实现每个专家与有效LoRA组合的token数量向block_size对齐，并完成token索引、专家ID、LoRA ID的映射。
	$$
	\text{num\_tokens\_post\_pad}[e, l] = 
	\begin{cases} 
	\left\lceil \frac{\text{orig\_token\_cnt}[e, l]}{\text{block\_size}} \right\rceil \times \text{block\_size}, & \text{当 } \text{adapter\_enabled}[l] = 1 \\
	0, & \text{当 } \text{adapter\_enabled}[l] = 0
	\end{cases}
	$$
	对于全局token位置 $i$（$0 \leq i < \text{max\_num\_tokens\_padded}$），若 $i$ 处于专家 $e$ 与有效LoRA $l$ 对应的对齐后区域（$\text{region\_start} \leq i < \text{region\_end}$，其中$\text{region\_start}$为该区域在全局的起始位置，$\text{region\_end} = \text{region\_start} + \text{num\_tokens\_post\_pad}[e, l]$），则：
	$$
	\text{sorted\_token\_ids}[i] = 
	\begin{cases} 
	\text{topk\_ids}[t], & i - \text{region\_start} < \text{orig\_token\_cnt}[e, l] \quad (t为该位置对应的原始token索引) \\
	\text{预设填充值}, & \text{否则}
	\end{cases}
	$$
	$$
	\text{expert\_ids}[i] = e
	$$
	$$
	\text{lora\_ids}[i] = l
	$$
	其中：
	- $\text{topk\_ids}$：输入张量，形状为 $(T)$（$T$为原始token总数），存储原始token的topk索引。
	- $\text{token\_lora\_mapping}$：输入张量，形状为 $(T)$，$\text{token\_lora\_mapping}[t]$表示原始token $t$对应的LoRA标识。
	- $\text{num\_experts}$：标量，专家的总数量。
	- $\text{block\_size}$：标量，分块对齐的目标尺寸，为正整数。
	- $\text{max\_loras}$：标量，支持的最大LoRA数量。
	- $\text{max\_num\_tokens\_padded}$：标量，填充后允许的最大token总数，约束输出张量的维度规模。
	- $\text{max\_num\_m\_blocks}$：标量，专家与LoRA组合对应的最大分块数量。
	- $\text{adapter\_enabled}$：输入张量，形状为 $(\text{max\_loras})$，$\text{adapter\_enabled}[l]$表示LoRA标识$l$是否处于有效状态。
	- $\text{sorted\_token\_ids}$：输出张量，形状为 $(\text{max\_num\_tokens\_padded})$，存储按专家-有效LoRA组合对齐block_size后的token索引，填充区域取值为预设填充值。
	- $\text{expert\_ids}$：输出张量，形状为 $(\text{max\_num\_tokens\_padded})$，$\text{expert\_ids}[i]$表示位置$i$的token对应的专家ID。
	- $\text{num\_tokens\_post\_pad}$：输出张量，形状为 $(\text{num\_experts}, \text{max\_loras})$，$\text{num\_tokens\_post\_pad}[e, l]$表示专家$e$与LoRA $l$组合对齐block_size后的token数量。
	- $\text{lora\_ids}$：输出张量，形状为 $(\text{max\_num\_tokens\_padded})$，$\text{lora\_ids}[i]$表示位置$i$的token对应的LoRA标识。
	- $e$：专家索引，满足 $0 \leq e < \text{num\_experts}$。
	- $l$：LoRA标识，满足 $0 \leq l < \text{max\_loras}$。
	- $\text{orig\_token\_cnt}[e, l]$：标量，专家$e$与LoRA $l$组合对应的原始token数量。
	- $\lceil \cdot \rceil$：向上取整运算，确保$\text{num\_tokens\_post\_pad}[e, l]$为$\text{block\_size}$的整数倍。
### 参数说明
- **topk_ids** (torch.Tensor, 入参): 存储topk选择对应的token索引的张量
- **token_lora_mapping** (torch.Tensor, 入参): 存储每个token对应的LoRA适配器映射关系的张量
- **num_experts** (int, 入参): MoE架构中专家的总数量，需为正整数
- **block_size** (int, 入参): MoE-LoRA计算中每个分块的token数量，需为正整数
- **max_loras** (int, 入参): 架构支持的最大LoRA适配器数量，需为正整数
- **max_num_tokens_padded** (int, 入参): 单个批次中padding后的最大token数量，需为正整数
- **max_num_m_blocks** (int, 入参): 计算中支持的最大分块数量，需为正整数
- **sorted_token_ids** (torch.Tensor, 入参): 存储token排序后索引的张量，算子会直接修改该张量内容
- **expert_ids** (torch.Tensor, 入参): 存储每个token对应专家ID的张量，算子会直接修改该张量内容
- **num_tokens_post_pad** (torch.Tensor, 入参): 存储每个分块padding后token数量的张量，算子会直接修改该张量内容
- **adapter_enabled** (torch.Tensor, 入参): 存储各LoRA适配器启用状态的张量
- **lora_ids** (torch.Tensor, 入参): 存储每个token对应的LoRA适配器ID的张量
### 返回值
void: 无返回值，所有操作结果直接作用于输入的张量参数
### 约束与调用
- 所有张量必须位于CUDA设备上
- 所有张量的数据类型需为int64（即torch.int64）
- num_experts、block_size、max_loras等整数参数需为大于0的正整数
- block_size需满足大于0的约束（算子内部包含该校验逻辑）
- 各张量的形状需与批次规模、token数量、专家数量及LoRA适配器数量匹配
### 调用示例
```python
import torch
import mcoplib._moe_C
# --- 配置参数 ---
DEVICE = "cuda"
BLOCK_SIZE = 8
NUM_EXPERTS = 8        
MAX_LORAS = 4          
NUM_TOKENS = 64        
CAPACITY_PER_LORA = 64 
# 确保 Capacity 是 Block Size 的倍数
MAX_NUM_TOKENS_PADDED = (CAPACITY_PER_LORA + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
MAX_NUM_M_BLOCKS = MAX_NUM_TOKENS_PADDED // BLOCK_SIZE
DTYPE_INT = torch.int32 
torch.set_default_device(DEVICE)
print(f"Capacity per LoRA: {MAX_NUM_TOKENS_PADDED} (Safe for {NUM_TOKENS} tokens)")
# 1. 准备输入数据
topk_ids = torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, 1), dtype=DTYPE_INT)
token_lora_mapping = torch.randint(0, MAX_LORAS, (NUM_TOKENS,), dtype=DTYPE_INT)
# 2. 准备输出数据 
out_sorted_size = MAX_LORAS * MAX_NUM_TOKENS_PADDED
sorted_token_ids = torch.zeros(out_sorted_size, dtype=DTYPE_INT)
# expert_ids 大小
expert_ids = torch.zeros(MAX_LORAS * MAX_NUM_M_BLOCKS * NUM_EXPERTS, dtype=DTYPE_INT) 
num_tokens_post_pad = torch.zeros(MAX_LORAS, dtype=DTYPE_INT)
adapter_enabled = torch.ones(MAX_LORAS, dtype=DTYPE_INT)
lora_ids = torch.zeros(MAX_LORAS, dtype=DTYPE_INT)
# 3. 算子执行
print(f"Running moe_lora_align_block_size...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops._moe_C.moe_lora_align_block_size(
    topk_ids,               
    token_lora_mapping,     
    NUM_EXPERTS,            
    BLOCK_SIZE,             
    MAX_LORAS,              
    MAX_NUM_TOKENS_PADDED,  
    MAX_NUM_M_BLOCKS,       
    sorted_token_ids,       
    expert_ids,             
    num_tokens_post_pad,    
    adapter_enabled,        
    lora_ids                
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功。")
print("-" * 30)
print(f"Num Tokens Post Pad (Per LoRA): {num_tokens_post_pad.tolist()}")
# 验证一下总数是否接近 64 (会有 padding)
print(f"Total tokens aligned: {num_tokens_post_pad.sum().item()}")
```
## 412. topk_sigmoid
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None
) -> None
```
### 功能描述
topk_sigmoid是混合专家（MoE）模型门控阶段的融合算子，集成了门控输出校正、TopK专家选择、Sigmoid权重计算及重归一化功能。该算子先对门控输出（可选叠加校正偏置）执行TopK筛选，再对选中专家对应的门控值计算Sigmoid激活权重，同时支持对权重进行重归一化，通过融合多步操作减少内存交互开销，提升MoE门控阶段的计算效率。
### 参数说明
- **topk_weights** (torch.Tensor, 出参): 选中TopK专家对应的Sigmoid权重张量，形状为（B, topk）（B为批大小，topk为单样本选中的专家数），支持数据类型为float16/bfloat16
- **topk_indices** (torch.Tensor, 出参): 选中TopK专家的索引张量，形状为（B, topk），数据类型为int32
- **gating_output** (torch.Tensor, 入参): 门控网络的输出张量，形状为（B, num_experts）（num_experts为专家总数），支持数据类型为float16/bfloat16
- **renormalize** (bool, 入参): 是否对TopK后的Sigmoid权重执行重归一化的标志位
- **correction_bias** (Optional[torch.Tensor], 入参): 门控输出的校正偏置张量，形状为（num_experts），支持数据类型为float16/bfloat16；若为None，则不执行门控输出校正
### 返回值
该算子无返回值，所有输出（TopK专家权重、TopK专家索引）均通过指定的出参张量（topk_weights、topk_indices）返回。
### 约束与调用
- 所有输入/出参张量必须部署在CUDA设备上
- gating_output的最后一维维度需与correction_bias的形状（num_experts）匹配；若correction_bias为None，则无此约束
- topk_weights、topk_indices的形状需为（B, topk），其中topk为实际筛选的专家数量（由门控逻辑确定）
- 数据类型约束：gating_output、correction_bias、topk_weights支持float16/bfloat16；topk_indices必须为int32类型
- 若开启renormalize，需保证TopK后的Sigmoid权重之和不为零，否则会导致重归一化计算数值不稳定
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# --- 配置参数 ---
DEVICE = "cuda"
NUM_TOKENS = 64      
NUM_EXPERTS = 8    
TOP_K = 2           
RENORMALIZE = True   
DTYPE_IDX = torch.int32 
DTYPE_VAL = torch.float32
torch.set_default_device(DEVICE)
print(f"Config: Tokens={NUM_TOKENS}, Experts={NUM_EXPERTS}, K={TOP_K}")
# 1. 准备输入数据
gating_output = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=DTYPE_VAL)
# correction_bias: 可选的偏置项 [Experts] (可以传 None)
# 如果不需要 bias，可以将此变量设为 None
correction_bias = torch.randn(NUM_EXPERTS, dtype=DTYPE_VAL) 
# 2. 准备输出数据
# 输出形状由 [Tokens, TOP_K] 决定，算子内部会根据这个形状确定 K 值
topk_weights = torch.empty(NUM_TOKENS, TOP_K, dtype=DTYPE_VAL)
topk_indices = torch.empty(NUM_TOKENS, TOP_K, dtype=DTYPE_IDX)
# 3. 执行算子
print("Running topk_sigmoid...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
torch.ops.sgl_kernel.topk_sigmoid(
    topk_weights,       # Output Tensor!
    topk_indices,       # Output Tensor!
    gating_output,      # Input Tensor
    RENORMALIZE,        # bool
    correction_bias     # Input Tensor? (Optional)
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功")
print("-" * 30)
# 4. 结果验证
print("Sample Results (First Token):")
print(f"Raw Logits:    {gating_output[0].tolist()}")
print(f"Selected Idxs: {topk_indices[0].tolist()}")
print(f"Selected Wgts: {topk_weights[0].tolist()}")
```
## 413. kimi_k2_moe_fused_gate
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def kimi_k2_moe_fused_gate(
    gate_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    input_logits: torch.Tensor,
    gate_bias: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    scaling_factor: float = 1.0
) -> None
```
### 功能描述
kimi_k2_moe_fused_gate 算子是面向MoE（混合专家）大模型的融合式门控核心算子，整合了MoE门控的全量核心逻辑。该算子先对输入的门控预测值张量叠加偏置项完成基础打分，再执行topk筛选得到每个token对应的最优k个专家索引及对应权重，支持对筛选后的专家权重做归一化处理，同时支持权重缩放系数的融合应用，所有计算逻辑一步融合完成，大幅减少显存的读写交互与中间张量的创建开销，显著提升MoE架构大模型的推理计算效率。
- 计算公式：
    输入张量与超参定义, 算子输入及超参说明：
    $$
    \begin{cases}
    \text{input} \in \mathbb{R}^{T \times E}, & T = B \times S \text{（}B\text{为batch size，}S\text{为序列长度，}T\text{为Token总数），}E\text{为专家总数} \\
    \text{bias} \in \mathbb{R}^E, & \text{每个专家对应的偏置张量} \\
    \text{topk} = K, & \text{激活专家数（正整数，}K \ll E\text{）} \\
    \text{renormalize} \in \{\text{True}, \text{False}\}, & \text{是否对Top-K得分做归一化的布尔标识} \\
    \text{routed\_scaling\_factor} \in \mathbb{R}, & \text{路由结果的缩放因子} \\
    \text{apply\_routed\_scaling\_factor\_on\_output} \in \{\text{True}, \text{False}\}, & \text{是否在输出上应用缩放因子的布尔标识}
    \end{cases}
    $$
    其中：$t \in [0,T-1]$（Token维度索引），$e \in [0,E-1]$（专家维度索引），$k \in [0,K-1]$（Top-K维度索引）。
    输入加偏置并计算Sigmoid原始得分, 对每个Token的每个专家，将输入与对应专家偏置相加后，通过Sigmoid激活得到原始门控得分：
    $$
    \text{raw\_score}_{t,e} = \sigma\left(\text{input}_{t,e} + \text{bias}_e\right) = \frac{1}{1 + \exp\left(-(\text{input}_{t,e} + \text{bias}_e)\right)}
    $$
    其中：$\text{raw\_score} \in \mathbb{R}^{T \times E}$，$\sigma(\cdot)$为Sigmoid函数，$\exp(\cdot)$为自然指数函数。
    逐Token筛选Top-K得分与专家索引, 对每个Token，在专家维度上筛选出原始得分中最大的$K$个值，同时记录对应的专家索引：
    $$
    \begin{cases}
    \text{topk\_score}_{t,k} = \text{TopK}(\text{raw\_score}_{t,:}, K) \\
    \text{topk\_index}_{t,k} = \text{ArgTopK}(\text{raw\_score}_{t,:}, K)
    \end{cases}
    $$
    其中：$\text{topk\_score} \in \mathbb{R}^{T \times K}$（Top-K得分张量），$\text{topk\_index} \in \mathbb{Z}^{T \times K}$（Top-K对应的专家索引张量），$\text{TopK}(\cdot)$为取前K大值函数，$\text{ArgTopK}(\cdot)$为取前K大值对应索引的函数。
    Top-K得分的可选归一化, 若$\text{renormalize}$为True，对每个Token的Top-K得分做求和归一化（加入稳定小量避免除零）；否则直接沿用Top-K得分：
    $$
    \text{normed\_score}_{t,k} = 
    \begin{cases} 
    \frac{\text{topk\_score}_{t,k}}{\sum_{k'=0}^{K-1}\text{topk\_score}_{t,k'} + \epsilon}, & \text{renormalize} = \text{True} \\
    \text{topk\_score}_{t,k}, & \text{renormalize} = \text{False}
    \end{cases}
    $$
    其中：$\text{normed\_score} \in \mathbb{R}^{T \times K}$，$\epsilon > 0$为数值稳定小量（通常取$10^{-6}$）。
    输出的可选缩放因子应用, 若$\text{apply\_routed\_scaling\_factor\_on\_output}$为True，将归一化后的得分乘以路由缩放因子；否则直接沿用归一化得分：
    $$
    \text{final\_weight}_{t,k} = 
    \begin{cases} 
    \text{normed\_score}_{t,k} \times \text{routed\_scaling\_factor}, & \text{apply\_routed\_scaling\_factor\_on\_output} = \text{True} \\
    \text{normed\_score}_{t,k}, & \text{apply\_routed\_scaling\_factor\_on\_output} = \text{False}
    \end{cases}
    $$
    其中：$\text{final\_weight} \in \mathbb{R}^{T \times K}$，为最终的专家门控权重张量。
    算子最终双输出, 该融合算子输出两个张量，无额外中间张量：
    $$
    \text{outputs} = (\text{final\_weight}, \text{topk\_index})
    $$
    其中：主输出$\text{final\_weight}$为浮点型专家门控权重，用于后续专家输出的加权融合；辅输出$\text{topk\_index}$为整型专家索引，用于筛选激活的专家计算结果。
### 参数说明
- **gate_weights** (torch.Tensor, 出参): 每个token选中的topk个专家对应的归一化门控权重张量，形状为 B×S×K，数据类型为 float16/bfloat16
- **expert_indices** (torch.Tensor, 出参): 每个token选中的topk个专家的索引张量，形状为 B×S×K，数据类型为 int32
- **input_logits** (torch.Tensor, 入参): MoE门控网络的原始输入预测张量，形状为 B×S×E，支持数据类型为 float16/bfloat16；其中E为专家总数，B为批次维度，S为序列长度维度
- **gate_bias** (torch.Tensor, 入参): MoE门控网络的偏置张量，形状为 E，与输入的专家维度严格匹配，支持数据类型为 float16/bfloat16
- **topk** (int, 入参): 每个token需要选择的最优专家数量，为正整数，取值范围为 1 ≤ topk ≤ E
- **renormalize** (bool, 入参): 布尔型开关，是否对选中的topk个专家的权重做归一化处理，使单token下选中专家的权重之和为1，默认值为True
- **scaling_factor** (float, 入参): 门控权重的全局缩放系数，用于权重幅值的精细化调控，默认值为1.0，无缩放效果
### 返回值
该算子无返回值，所有输出（选中专家的权重、专家索引）均通过指定的出参张量（gate_weights、expert_indices）返回。
### 约束与调用
- 所有输入/出参张量必须位于 CUDA 设备上
- input_logits 的形状必须为 B×S×E，gate_bias 的形状必须为 E，二者的专家维度E必须完全一致
- gate_weights、expert_indices 的形状必须为 B×S×K，与 input_logits 的批次维度B、序列维度S保持一致，K为入参topk的取值
- input_logits、gate_bias、gate_weights 支持的数据类型为 float16/bfloat16，expert_indices 必须为 int32 类型
- topk 必须为正整数，且数值不能超过专家总数E，否则会触发索引越界错误
- 若 renormalize 设置为True，单token下输出的gate_weights对应维度的权重求和结果恒等于1.0，保证专家路由的权重分配合理性
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# --- 配置参数 ---
DEVICE = "cuda"
NUM_TOKENS = 64       
NUM_EXPERTS = 384     
TOP_K = 2           
RENORMALIZE = True    
ROUTED_SCALING_FACTOR = 1.0
APPLY_SCALING = False
DTYPE_VAL = torch.float32 
DTYPE_IDX = torch.int32   
torch.set_default_device(DEVICE)
print(f"Config: Tokens={NUM_TOKENS}, Experts={NUM_EXPERTS}, K={TOP_K}")
# 1. 准备输入数据
# gating_output 形状变为 [64, 384]
gating_output = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=DTYPE_VAL)
# bias 形状变为 [384]
correction_bias = torch.randn(NUM_EXPERTS, dtype=DTYPE_VAL)
# 2. 执行算子
print(f"Running kimi_k2_moe_fused_gate (Op: {torch.ops.sgl_kernel.kimi_k2_moe_fused_gate})...")
if torch.cuda.is_available():
    torch.cuda.synchronize()
results = torch.ops.sgl_kernel.kimi_k2_moe_fused_gate(
    gating_output,    
    correction_bias,  
    TOP_K,            
    RENORMALIZE,      
    ROUTED_SCALING_FACTOR, 
    APPLY_SCALING     
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print("执行成功")
print("-" * 30)
# 3. 结果验证与解包
topk_weights = results[0]
topk_indices = results[1]
print("Sample Results (First Token):")
# 打印前 10 个 logits 示例
print(f"Raw Logits (first 10): {gating_output[0][:10].tolist()} ...")
print(f"Selected Idxs: {topk_indices[0].tolist()}")
print(f"Selected Wgts: {topk_weights[0].tolist()}")
# 验证形状
print(f"\nShape Verification:")
print(f"Weights Shape: {topk_weights.shape} (Expected: [{NUM_TOKENS}, {TOP_K}])")
print(f"Indices Shape: {topk_indices.shape} (Expected: [{NUM_TOKENS}, {TOP_K}])")
```
## 414. fused_qk_norm_rope
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def fused_qk_norm_rope(
    gkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float,
    low: float,
    high: float,
    attention_factor: float
) -> None
```
### 功能描述
fused_qk_norm_rope 算子是面向 GPU 优化的融合计算算子，将Query/Key的权重映射、归一化操作与RoPE（旋转位置编码）计算合并执行，减少GPU显存的读写开销，常用于大语言模型注意力机制模块的训练与推理流程。
- 计算公式：
	当前版本中，fused_qk_norm_rope 算子仅支持输入张量位于CUDA设备上
	$$
	\text{output}[b, s, idx] = 
	\begin{cases} 
	\text{RoPE}\left( \frac{\text{qkv}[b, s, idx] \cdot w_q}{\sqrt{\text{mean}(\text{qkv}[b, s, \text{q\_range}]^2) + \text{eps}}}, \text{pos}_s \right) & idx \in \text{q\_range} \\
	\text{RoPE}\left( \frac{\text{qkv}[b, s, idx] \cdot w_k}{\sqrt{\text{mean}(\text{qkv}[b, s, \text{k\_range}]^2) + \text{eps}}}, \text{pos}_s \right) & idx \in \text{k\_range} \\
	\text{qkv}[b, s, idx] & idx \in \text{v\_range}
	\end{cases}
	$$
	其中：
	- $\text{qkv}$：输入张量，需位于CUDA设备上，形状记为 $S = (\text{batch\_size}, \text{seq\_len}, D)$，$D = (\text{num\_heads\_q}+\text{num\_heads\_k}+\text{num\_heads\_v}) \cdot \text{head\_dim}$，为包含查询、键、值的融合张量
	- $\text{num\_heads\_q}$：查询注意力头的数量，非负整数
	- $\text{num\_heads\_k}$：键注意力头的数量，非负整数
	- $\text{num\_heads\_v}$：值注意力头的数量，非负整数
	- $\text{head\_dim}$：单个注意力头的维度，正整数
	- $\text{q\_range}$：$\text{qkv}$ 中查询部分的索引范围，即 $[0, \text{num\_heads\_q} \cdot \text{head\_dim})$
	- $\text{k\_range}$：$\text{qkv}$ 中键部分的索引范围，即 $[\text{num\_heads\_q} \cdot \text{head\_dim}, (\text{num\_heads\_q}+\text{num\_heads\_k}) \cdot \text{head\_dim})$
	- $\text{v\_range}$：$\text{qkv}$ 中值部分的索引范围，即 $[(\text{num\_heads\_q}+\text{num\_heads\_k}) \cdot \text{head\_dim}, D)$
	- $w_q$：查询头对应的权重张量，形状与 $\text{q\_range}$ 维度匹配
	- $w_k$：键头对应的权重张量，形状与 $\text{k\_range}$ 维度匹配
	- $\text{eps}$：防止除零的小常数，正浮点数
	- $\text{pos}_s$：序列位置 $s$ 对应的位置索引，来自 $\text{position\_ids}$ 张量
	- $\text{Norm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \text{eps}}}$：归一化操作，对张量对应部分的元素计算均值平方后做标准化
	- $\text{RoPE}(\cdot, \text{pos}_s)$：旋转位置编码操作，基于位置索引 $\text{pos}_s$ 对张量进行旋转编码
	- $\text{output}$：输出张量，形状与 $\text{qkv}$ 完全一致（即 $S = (\text{batch\_size}, \text{seq\_len}, D)$），存储融合处理后的结果
	- $[b, s, idx]$：张量的多维索引，$b$ 为批次索引（$0 \leq b < \text{batch\_size}$）、$s$ 为序列位置索引（$0 \leq s < \text{seq\_len}$）、$idx$ 为维度索引（$0 \leq idx < D$）
	- $\cdot$：张量与权重的逐元素（或匹配维度）乘法运算
### 参数说明
- **gkv** (torch.Tensor, 入参/出参): 输入输出张量，用于承载融合计算后的Q/K/V相关结果，需部署在CUDA设备上
- **num_heads_q** (int, 入参): Query对应的注意力头数量
- **num_heads_k** (int, 入参): Key对应的注意力头数量
- **num_heads_v** (int, 入参): Value对应的注意力头数量
- **head_dim** (int, 入参): 单个注意力头的特征维度，需满足算子内部的对齐约束（如特定数值的倍数）
- **eps** (float, 入参): 归一化操作中用于避免除零的微小平滑值
- **q_weight** (torch.Tensor, 入参): Query对应的权重张量，形状需与num_heads_q、head_dim匹配，需部署在CUDA设备上
- **k_weight** (torch.Tensor, 入参): Key对应的权重张量，形状需与num_heads_k、head_dim匹配，需部署在CUDA设备上
- **base** (float, 入参): RoPE编码中的基数值，用于计算位置编码的缩放因子
- **is_neox** (bool, 入参): 是否采用GPT-NeoX风格的RoPE编码实现逻辑
- **position_ids** (torch.Tensor, 入参): 序列的位置索引张量，用于提供RoPE编码的位置信息，需部署在CUDA设备上
- **factor** (float, 入参): 融合计算流程中的缩放因子参数
- **low** (float, 入参): 计算过程中的数值下界参数
- **high** (float, 入参): 计算过程中的数值上界参数
- **attention_factor** (float, 入参): 注意力计算环节的缩放因子参数
### 返回值
计算结果通过入参`gkv`（引用传递）输出，无独立返回张量
### 约束与调用
- 所有输入张量必须部署在CUDA设备上
- 各张量的形状需满足注意力头数、头维度的匹配关系（如q_weight的形状需与num_heads_q×head_dim对齐）
- head_dim需满足算子内部静态断言的对齐要求（如为特定数值的整数倍）
- 支持的数据类型：float16（torch.float16）、bfloat16（torch.bfloat16）
- position_ids的形状需与输入序列的长度维度匹配
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 配置参数
SEED = 42
DEVICE = "cuda"
# 维度设置
BATCH_SIZE = 2
SEQ_LEN = 128
HEAD_DIM = 128
NUM_HEADS_Q = 32
NUM_HEADS_K = 32
NUM_HEADS_V = 32
# 初始化环境
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN
HIDDEN_SIZE_QKV = (NUM_HEADS_Q + NUM_HEADS_K + NUM_HEADS_V) * HEAD_DIM
# 1. 准备数据
qkv = torch.randn((TOTAL_TOKENS, HIDDEN_SIZE_QKV), dtype=torch.bfloat16, device=DEVICE)
# 准备权重 (Weight)
q_weight = torch.randn((HEAD_DIM,), dtype=torch.bfloat16, device=DEVICE)
k_weight = torch.randn((HEAD_DIM,), dtype=torch.bfloat16, device=DEVICE)
position_ids = torch.arange(SEQ_LEN, device=DEVICE).unsqueeze(0).expand(BATCH_SIZE, -1).reshape(-1).to(torch.int)
# 其他标量参数
eps = 1e-5
base = 10000.0
is_neox = True
factor = 1.0
low = 0.0
high = 0.0
attention_factor = 1.0
# 2. 调用算子
print(f"Running fused_qk_norm_rope...")
print(f"Input shape: {qkv.shape}")
print(f"Position_ids dtype: {position_ids.dtype}") 
torch.ops.sgl_kernel.fused_qk_norm_rope(
    qkv,                
    NUM_HEADS_Q,
    NUM_HEADS_K,
    NUM_HEADS_V,
    HEAD_DIM,
    eps,
    q_weight,           
    k_weight,         
    base,
    is_neox,
    position_ids,       
    factor,
    low,
    high,
    attention_factor
)
print("执行成功 (Execution Success).")
if torch.cuda.is_available():
    torch.cuda.synchronize()

# 3. 验证结果
print(f"Output Mean: {qkv.float().mean().item():.4f}")
print(f"Output Std:  {qkv.float().std().item():.4f}")
```
## 415. weak_ref_tensor
### 支持的产品型号
- Metax C500/C550/C588/C600
### 接口原型
```python
def weak_ref_tensor(
    input: at::Tensor
) -> at::Tensor
```
### 功能描述
weak_ref_tensor 算子是面向CUDA张量的内存复用工具算子：基于输入CUDA张量的原始数据指针，复用其形状、步幅、数据类型等属性，创建一个共享原张量内存的新张量视图（不拷贝数据），用于避免不必要的显存数据拷贝，提升CUDA张量的内存复用效率，常用于大语言模型等显存敏感场景的张量内存管理。
- 计算公式：
	当前版本中，weak_ref_tensor 算子仅支持输入张量位于CUDA设备上
	$$
	\text{output}[i_1, i_2, ..., i_n] = \text{input}[i_1, i_2, ..., i_n]
	$$
	其中：
	- $\text{input}$：输入张量，需位于CUDA设备上，形状记为 $S = (s_1, s_2, ..., s_n)$，$n$ 为张量维度数
	- $\text{output}$：输出张量，形状与 $\text{input}$ 完全一致（即 $S = (s_1, s_2, ..., s_n)$），与 $\text{input}$ 共享数据存储
	- $[i_1, i_2, ..., i_n]$：张量任意位置的多维索引，满足 $0 \leq i_k < s_k$（$k = 1, 2, ..., n$），覆盖张量所有元素
	- $=$：表示输出张量对应位置元素与输入张量对应位置元素取值相同
### 参数说明
- **input** (at::Tensor, 入参): 输入的CUDA张量，算子会基于其数据指针、形状、步幅等属性创建新张量视图，必须是位于CUDA设备上的合法张量。
### 返回值
at::Tensor: 输出的新张量，与输入张量共享数据内存，形状、步幅、数据类型、设备等属性均与输入张量完全一致。
### 约束与调用
- 输入张量**必须位于CUDA设备上**，非CUDA张量会触发错误
- 新张量与输入张量共享内存，输入张量的生命周期需覆盖新张量的使用周期，否则会出现悬空指针风险
- 支持的张量数据类型与输入张量一致（如float16（at::ScalarType::Half）、bfloat16（at::ScalarType::BFloat16）、float32等）
- 输入张量需保持数据指针有效，非法或已释放的张量会导致新张量访问异常
### 调用示例
```python
import torch
import mcoplib.sgl_kernel
# 1. 准备数据 (必须在 CUDA 上)
torch.set_default_device("cuda")
input_tensor = torch.randn(4, 4096)
# 2. 算子执行
# 这里的 weak_ref_tensor 返回一个新的 Tensor，指向相同的显存
output_tensor = torch.ops.sgl_kernel.weak_ref_tensor(input_tensor)
# 3. 打印结果
print(f"Input Shape:  {input_tensor.shape}")
print(f"Output Shape: {output_tensor.shape}")
```