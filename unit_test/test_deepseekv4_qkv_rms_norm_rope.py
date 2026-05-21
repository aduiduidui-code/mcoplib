#!/usr/bin/env python3
"""
Fused RMS Norm + RoPE for DeepSeekV4 单元测试

测试目标：
1. 验证精度 (cos_sim > 0.9999)
2. 性能对比 (CUDA vs PyTorch)
3. 计算加速比和带宽
4. 测试多组不同shape并统计性能
5. 测试不同的weight配置（q有weight、kv有weight、两者都有）
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List

# Import kernel
import mcoplib.op as ops


def torch_rms_norm(x: torch.Tensor, eps: float, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    RMS 归一化
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)

    if weight is not None:
        x_normed = x_normed * weight

    return x_normed


def apply_rotary_emb_torch_interleaved(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """
    交错式 (Interleaved) 旋转位置编码实现。
    完全对齐提供的 CUDA Kernel (相邻元素作为实部和虚部配对)。
    """
    # 1. 获取当前 token 对应的频率并提取实部和虚部
    freqs = freqs_cis[positions]
    cos = freqs.real
    sin = freqs.imag

    if inverse:
        sin = -sin

    # 针对 attention heads 维度进行广播 (假设 x 形状为 [Batch, Heads, D])
    # 如果 x 的形状是 [Batch, SeqLen, Heads, D]，这里需要改为 .unsqueeze(1).unsqueeze(2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # ================= 核心修改点 =================

    # 2. 将 x 重塑，以提取交错的实部(偶数位)和虚部(奇数位)
    # x 形状变化: [..., rope_dim] -> [..., rope_dim // 2, 2]
    x_reshaped = x.view(*x.shape[:-1], -1, 2)

    # 提取实部 (索引 0) 和虚部 (索引 1)
    # 取出后的形状为: [..., rope_dim // 2]
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]

    # 3. 执行复数乘法 (RoPE 旋转)
    x_out_real = x_real * cos - x_imag * sin
    x_out_imag = x_real * sin + x_imag * cos

    # 4. 将结果重新交错打包回原有的内存布局
    # 将实部和虚部堆叠，形状恢复到 [..., rope_dim // 2, 2]
    x_out = torch.stack([x_out_real, x_out_imag], dim=-1)

    # 展平最后两个维度，最终输出形状恢复为 [..., rope_dim]
    return x_out.flatten(-2)


def torch_fused_forward_prepare_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    freqs_cis: torch.Tensor,
    qk_rope_head_dim: int,
    eps: float,
    weight_q: Optional[torch.Tensor] = None,
    weight_kv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用纯 PyTorch 实现的 _forward_prepare 融合函数 (Reference实现)
    支持 weight_q 和 weight_kv 参数
    """
    if q.dim() == 2:
        batch_size = q.size(0)
        total_dim = q.size(1)
        num_heads = total_dim // 64
        head_dim = 64
        q_reshaped = q.view(batch_size, num_heads, head_dim)
    else:
        q_reshaped = q

    # RMS Norm with optional weight
    q_normed = torch_rms_norm(q_reshaped, eps, weight=weight_q)
    kv_normed = torch_rms_norm(kv, eps, weight=weight_kv)

    # Apply RoPE on the last qk_rope_head_dim dimensions
    q_rope = q_normed[..., -qk_rope_head_dim:]
    q_rope_out = apply_rotary_emb_torch_interleaved(q_rope, freqs_cis, positions)
    q_normed[..., -qk_rope_head_dim:] = q_rope_out

    kv_rope = kv_normed[..., -qk_rope_head_dim:]
    kv_rope_unsqueezed = kv_rope.unsqueeze(1)
    kv_rope_out = apply_rotary_emb_torch_interleaved(kv_rope_unsqueezed, freqs_cis, positions)
    kv_normed[..., -qk_rope_head_dim:] = kv_rope_out.squeeze(1)

    if q.dim() == 2:
        q_out = q_normed.view(batch_size, total_dim)
    else:
        q_out = q_normed

    return q_out, kv_normed


def cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """计算两个张量的余弦相似度"""
    flat1 = t1.flatten().float()
    flat2 = t2.flatten().float()
    return F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1).item()


def benchmark(func, args, warmup=10, rep=100):
    """性能测试函数"""
    for _ in range(warmup):
        func(*args)

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]

    for i in range(rep):
        start_event[i].record()
        func(*args)
        end_event[i].record()

    torch.cuda.synchronize()
    dur = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    return dur.mean().item()


def calculate_bandwidth(q_size: int, batch_size: int, kv_dim: int, time_ms: float,
                         qk_rope_head_dim: int = 64,
                         weight_q_size: int = 0,
                         weight_kv_size: int = 0) -> Tuple[float, float]:
    """计算带宽

    Args:
        q_size: q tensor 的元素总数
        batch_size: 批次大小 (直接传入，无需推断)
        kv_dim: kv tensor 的最后一维大小
        time_ms: 执行时间 (毫秒)
        qk_rope_head_dim: rope 维度大小
        weight_q_size: weight_q tensor 的元素总数 (如果有)
        weight_kv_size: weight_kv tensor 的元素总数 (如果有)

    Returns:
        (带宽 GB/s, 数据传输量 MB)
    """
    # 数据读取: q + kv + freqs (只读rope部分) + positions + weights (如果有)
    q_bytes = q_size * 2  # bf16 = 2 bytes
    kv_bytes = batch_size * kv_dim * 2  # kv shape: [batch_size, kv_dim]

    # freqs 只读 rope 维度的一半，每个元素 8 bytes (complex64)
    rope_half_dim = qk_rope_head_dim // 2
    freqs_bytes = batch_size * rope_half_dim * 8
    positions_bytes = batch_size * 8  # int64

    # weight 数据读取 (如果有)
    weight_q_bytes = weight_q_size * 2 if weight_q_size > 0 else 0
    weight_kv_bytes = weight_kv_size * 2 if weight_kv_size > 0 else 0

    total_read_bytes = q_bytes + kv_bytes + freqs_bytes + positions_bytes + weight_q_bytes + weight_kv_bytes

    # 数据写入: q + kv (in-place)
    total_write_bytes = q_bytes + kv_bytes

    total_bytes = total_read_bytes + total_write_bytes

    bandwidth_gbps = total_bytes / (time_ms * 1e-3) / 1e9
    return bandwidth_gbps, total_bytes / 1e6


def test_single_shape(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    kv_dim: int,
    qk_rope_head_dim: int,
    eps: float,
    test_name: str,
) -> Tuple[bool, float, float, float, float]:
    """测试单个shape配置（无weight），返回精度、性能信息"""
    return test_single_shape_with_weight(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_dim=kv_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        eps=eps,
        test_name=test_name,
        weight_q=None,
        weight_kv=None,
    )


def test_single_shape_with_weight(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    kv_dim: int,
    qk_rope_head_dim: int,
    eps: float,
    test_name: str,
    weight_q: Optional[torch.Tensor] = None,
    weight_kv: Optional[torch.Tensor] = None,
) -> Tuple[bool, float, float, float, float]:
    """测试单个shape配置（支持weight），返回精度、性能信息"""
    device = "cuda"
    max_seq_len = 1048576

    print(f"\n{'='*80}")
    print(f"测试: {test_name}")
    print(f"{'='*80}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  kv_dim: {kv_dim}")
    print(f"  qk_rope_head_dim: {qk_rope_head_dim}")
    print(f"  eps: {eps}")
    print(f"  weight_q: {'None' if weight_q is None else f'shape={weight_q.shape}'}")
    print(f"  weight_kv: {'None' if weight_kv is None else f'shape={weight_kv.shape}'}")

    # 创建测试数据
    if head_dim == 64:
        q = torch.randn(batch_size, num_heads * head_dim, dtype=torch.bfloat16, device=device)
    else:
        q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device)

    kv = torch.randn(batch_size, kv_dim, dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, max_seq_len, (batch_size,), dtype=torch.int64, device=device)
    freqs_cis = torch.randn(max_seq_len, qk_rope_head_dim // 2, dtype=torch.complex64, device=device)

    print(f"  q shape: {q.shape}, dtype: {q.dtype}")
    print(f"  kv shape: {kv.shape}, dtype: {kv.dtype}")

    # ==================== 精度验证 ====================
    q_ref, kv_ref = torch_fused_forward_prepare_reference(
        q.clone(), kv.clone(), positions, freqs_cis, qk_rope_head_dim, eps,
        weight_q=weight_q,
        weight_kv=weight_kv,
    )

    q_cuda = q.clone()
    kv_cuda = kv.clone()
    ops.fused_rms_norm_rope(
        q_cuda, kv_cuda, positions, freqs_cis, qk_rope_head_dim, eps,
        weight_q=weight_q,
        weight_kv=weight_kv,
    )

    q_cos_sim = cosine_similarity(q_ref, q_cuda)
    kv_cos_sim = cosine_similarity(kv_ref, kv_cuda)

    print(f"\n精度验证:")
    print(f"  Query 余弦相似度: {q_cos_sim:.6f}")
    print(f"  KV 余弦相似度: {kv_cos_sim:.6f}")

    if q_cos_sim > 0.9999 and kv_cos_sim > 0.9999:
        print(f"  ✓ 精度验证通过！")
        precision_pass = True
    else:
        print(f"  ✗ 精度验证失败！")
        q_diff = (q_ref - q_cuda).abs()
        kv_diff = (kv_ref - kv_cuda).abs()
        print(f"  q最大差异: {q_diff.max().item():.6f}")
        print(f"  q平均差异: {q_diff.mean().item():.6f}")
        print(f"  kv最大差异: {kv_diff.max().item():.6f}")
        print(f"  kv平均差异: {kv_diff.mean().item():.6f}")
        precision_pass = False

    # ==================== 性能测试 ====================
    print(f"\n性能测试:")

    # PyTorch Reference性能
    def ref_func(q_in, kv_in, pos, freq, rope_dim, epsilon, w_q, w_kv):
        return torch_fused_forward_prepare_reference(q_in, kv_in, pos, freq, rope_dim, epsilon, w_q, w_kv)

    ref_time = benchmark(
        ref_func,
        (q.clone(), kv.clone(), positions, freqs_cis, qk_rope_head_dim, eps, weight_q, weight_kv),
        warmup=10, rep=100
    )

    # CUDA Kernel性能
    def cuda_func(q_in, kv_in, pos, freq, rope_dim, epsilon, w_q, w_kv):
        ops.fused_rms_norm_rope(q_in, kv_in, pos, freq, rope_dim, epsilon, w_q, w_kv)

    cuda_time = benchmark(
        cuda_func,
        (q.clone(), kv.clone(), positions, freqs_cis, qk_rope_head_dim, eps, weight_q, weight_kv),
        warmup=10, rep=100
    )

    # 计算加速比
    speedup = ref_time / cuda_time if cuda_time > 0 else 0

    # 计算带宽（考虑weight的数据量）
    weight_q_size = weight_q.numel() if weight_q is not None else 0
    weight_kv_size = weight_kv.numel() if weight_kv is not None else 0
    bandwidth_gbps, data_size_mb = calculate_bandwidth(
        q.numel(), batch_size, kv_dim, cuda_time, qk_rope_head_dim,
        weight_q_size, weight_kv_size
    )

    # 理论带宽（C500 GPU）
    theoretical_bandwidth = 400.0  # GB/s
    bandwidth_efficiency = (bandwidth_gbps / theoretical_bandwidth) * 100

    print(f"  PyTorch Reference 平均耗时: {ref_time:.3f} ms")
    print(f"  CUDA Kernel 平均耗时: {cuda_time:.3f} ms")
    print(f"  加速比: {speedup:.2f}x")
    print(f"  数据传输量: {data_size_mb:.3f} MB")
    print(f"  计算带宽: {bandwidth_gbps:.2f} GB/s")
    print(f"  理论带宽 (C500): {theoretical_bandwidth:.2f} GB/s")
    print(f"  带宽效率: {bandwidth_efficiency:.2f}%")

    return precision_pass, speedup, ref_time, cuda_time, bandwidth_gbps


def main():
    """主测试函数 - 测试多组典型shape并统计性能，包括weight测试"""
    eps = 1e-6
    qk_rope_head_dim = 64
    device = "cuda"

    print("=" * 80)
    print("Fused RMS Norm + RoPE for DeepSeekV4 - 多shape精度与性能测试")
    print("=" * 80)

    # ==================== 第一部分：无weight的测试 ====================
    print("\n" + "=" * 80)
    print("第一部分：无weight的测试（基准测试）")
    print("=" * 80)

    # 定义典型shape配置
    test_configs: List[dict] = [
        {
            "batch_size": 1,
            "num_heads": 64,
            "head_dim": 512,
            "kv_dim": 512,
            "test_name": "batch1 q[1,64,512] kv[1,512] (无weight)"
        },
        {
            "batch_size": 2,
            "num_heads": 64,
            "head_dim": 512,
            "kv_dim": 512,
            "test_name": "batch2 q[2,64,512] kv[2,512] (无weight)"
        },
        {
            "batch_size": 4,
            "num_heads": 64,
            "head_dim": 512,
            "kv_dim": 512,
            "test_name": "batch4 q[4,64,512] kv[4,512] (无weight)"
        },
        {
            "batch_size": 8,
            "num_heads": 64,
            "head_dim": 512,
            "kv_dim": 512,
            "test_name": "batch8 q[8,64,512] kv[8,512] (无weight)"
        },
        {
            "batch_size": 16,
            "num_heads": 64,
            "head_dim": 512,
            "kv_dim": 512,
            "test_name": "batch16 q[16,64,512] kv[16,512] (无weight)"
        },
        {
            "batch_size": 32,
            "num_heads": 64,
            "head_dim": 512,
            "kv_dim": 512,
            "test_name": "batch32 q[32,64,512] kv[32,512] (无weight)"
        },
    ]

    # 运行无weight测试
    results = []
    performance_data = []

    for config in test_configs:
        precision_pass, speedup, ref_time, cuda_time, bandwidth = test_single_shape(
            batch_size=config["batch_size"],
            num_heads=config["num_heads"],
            head_dim=config["head_dim"],
            kv_dim=config["kv_dim"],
            qk_rope_head_dim=qk_rope_head_dim,
            eps=eps,
            test_name=config["test_name"],
        )
        results.append((config["test_name"], precision_pass))
        performance_data.append({
            "name": config["test_name"],
            "precision_pass": precision_pass,
            "speedup": speedup,
            "ref_time": ref_time,
            "cuda_time": cuda_time,
            "bandwidth": bandwidth,
            "batch_size": config["batch_size"],
            "kv_dim": config["kv_dim"],
            "weight_type": "none",
        })

    # ==================== 第二部分：weight测试 ====================
    print("\n" + "=" * 80)
    print("第二部分：带weight的测试")
    print("=" * 80)

    # 选取一个代表性的batch_size进行weight测试
    weight_test_batch = 4
    weight_test_num_heads = 64
    weight_test_head_dim = 512
    weight_test_kv_dim = 512

    # 创建weight tensor（使用接近1.0的随机值，模拟实际RMS Norm weight）
    # RMS Norm weight 通常初始化为1.0附近
    weight_q_tensor = (1.0 + 0.1 * torch.randn(weight_test_head_dim, dtype=torch.bfloat16, device=device)).to(torch.bfloat16)
    weight_kv_tensor = (1.0 + 0.1 * torch.randn(weight_test_kv_dim, dtype=torch.bfloat16, device=device)).to(torch.bfloat16)

    # 测试 Case 1: 只有 q 有 weight
    print("\n" + "-" * 80)
    print("Case 1: 只有 Q 有 weight")
    print("-" * 80)
    precision_pass, speedup, ref_time, cuda_time, bandwidth = test_single_shape_with_weight(
        batch_size=weight_test_batch,
        num_heads=weight_test_num_heads,
        head_dim=weight_test_head_dim,
        kv_dim=weight_test_kv_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        eps=eps,
        test_name=f"batch{weight_test_batch} q有weight kv无weight",
        weight_q=weight_q_tensor,
        weight_kv=None,
    )
    results.append((f"batch{weight_test_batch} q有weight kv无weight", precision_pass))
    performance_data.append({
        "name": f"batch{weight_test_batch} q有weight kv无weight",
        "precision_pass": precision_pass,
        "speedup": speedup,
        "ref_time": ref_time,
        "cuda_time": cuda_time,
        "bandwidth": bandwidth,
        "batch_size": weight_test_batch,
        "kv_dim": weight_test_kv_dim,
        "weight_type": "q_only",
    })

    # 测试 Case 2: 只有 kv 有 weight
    print("\n" + "-" * 80)
    print("Case 2: 只有 KV 有 weight")
    print("-" * 80)
    precision_pass, speedup, ref_time, cuda_time, bandwidth = test_single_shape_with_weight(
        batch_size=weight_test_batch,
        num_heads=weight_test_num_heads,
        head_dim=weight_test_head_dim,
        kv_dim=weight_test_kv_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        eps=eps,
        test_name=f"batch{weight_test_batch} q无weight kv有weight",
        weight_q=None,
        weight_kv=weight_kv_tensor,
    )
    results.append((f"batch{weight_test_batch} q无weight kv有weight", precision_pass))
    performance_data.append({
        "name": f"batch{weight_test_batch} q无weight kv有weight",
        "precision_pass": precision_pass,
        "speedup": speedup,
        "ref_time": ref_time,
        "cuda_time": cuda_time,
        "bandwidth": bandwidth,
        "batch_size": weight_test_batch,
        "kv_dim": weight_test_kv_dim,
        "weight_type": "kv_only",
    })

    # 测试 Case 3: q 和 kv 都有 weight
    print("\n" + "-" * 80)
    print("Case 3: Q 和 KV 都有 weight")
    print("-" * 80)
    precision_pass, speedup, ref_time, cuda_time, bandwidth = test_single_shape_with_weight(
        batch_size=weight_test_batch,
        num_heads=weight_test_num_heads,
        head_dim=weight_test_head_dim,
        kv_dim=weight_test_kv_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        eps=eps,
        test_name=f"batch{weight_test_batch} q和kv都有weight",
        weight_q=weight_q_tensor,
        weight_kv=weight_kv_tensor,
    )
    results.append((f"batch{weight_test_batch} q和kv都有weight", precision_pass))
    performance_data.append({
        "name": f"batch{weight_test_batch} q和kv都有weight",
        "precision_pass": precision_pass,
        "speedup": speedup,
        "ref_time": ref_time,
        "cuda_time": cuda_time,
        "bandwidth": bandwidth,
        "batch_size": weight_test_batch,
        "kv_dim": weight_test_kv_dim,
        "weight_type": "both",
    })

    # ==================== 最终总结 ====================
    print("\n")
    print("=" * 80)
    print("最终测试结果汇总")
    print("=" * 80)

    all_passed = True
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False

    # 打印性能统计表（按weight类型分组）
    print("\n")
    print("=" * 80)
    print("性能对比统计表（无weight测试）")
    print("=" * 80)
    print(f"{'测试名称':<45} {'batch':<8} {'kv_dim':<8} {'PyTorch(ms)':<12} {'CUDA(ms)':<10} {'加速比':<8} {'带宽(GB/s)':<10}")
    print("-" * 100)

    no_weight_data = [d for d in performance_data if d['weight_type'] == 'none']
    for data in no_weight_data:
        print(f"{data['name']:<45} {data['batch_size']:<8} {data['kv_dim']:<8} "
              f"{data['ref_time']:<12.3f} {data['cuda_time']:<10.3f} "
              f"{data['speedup']:<8.2f} {data['bandwidth']:<10.2f}")

    print("-" * 100)

    if no_weight_data:
        avg_speedup = sum([d['speedup'] for d in no_weight_data]) / len(no_weight_data)
        avg_bandwidth = sum([d['bandwidth'] for d in no_weight_data]) / len(no_weight_data)
        avg_ref_time = sum([d['ref_time'] for d in no_weight_data]) / len(no_weight_data)
        avg_cuda_time = sum([d['cuda_time'] for d in no_weight_data]) / len(no_weight_data)

        print(f"{'平均值':<45} {'-':<8} {'-':<8} "
              f"{avg_ref_time:<12.3f} {avg_cuda_time:<10.3f} "
              f"{avg_speedup:<8.2f} {avg_bandwidth:<10.2f}")

    # 打印带weight的性能统计表
    print("\n")
    print("=" * 80)
    print("性能对比统计表（带weight测试）")
    print("=" * 80)
    print(f"{'测试名称':<45} {'weight类型':<12} {'PyTorch(ms)':<12} {'CUDA(ms)':<10} {'加速比':<8} {'带宽(GB/s)':<10}")
    print("-" * 100)

    weight_data = [d for d in performance_data if d['weight_type'] != 'none']
    for data in weight_data:
        weight_label = {
            "q_only": "q有weight",
            "kv_only": "kv有weight",
            "both": "两者都有",
        }.get(data['weight_type'], data['weight_type'])
        print(f"{data['name']:<45} {weight_label:<12} "
              f"{data['ref_time']:<12.3f} {data['cuda_time']:<10.3f} "
              f"{data['speedup']:<8.2f} {data['bandwidth']:<10.2f}")

    print("-" * 100)

    # 计算全部测试的平均值
    if performance_data:
        avg_speedup_all = sum([d['speedup'] for d in performance_data]) / len(performance_data)
        avg_bandwidth_all = sum([d['bandwidth'] for d in performance_data]) / len(performance_data)

        print(f"\n全部测试性能总结:")
        print(f"  总测试数: {len(results)}")
        print(f"  平均加速比: {avg_speedup_all:.2f}x")
        print(f"  平均带宽: {avg_bandwidth_all:.2f} GB/s")
        print(f"  最高加速比: {max([d['speedup'] for d in performance_data]):.2f}x")
        print(f"  最高带宽: {max([d['bandwidth'] for d in performance_data]):.2f} GB/s")

        # 无weight测试的平均值
        if no_weight_data:
            print(f"\n无weight测试性能总结:")
            print(f"  平均加速比: {avg_speedup:.2f}x")
            print(f"  平均带宽: {avg_bandwidth:.2f} GB/s")

        # weight测试的平均值
        if weight_data:
            avg_speedup_weight = sum([d['speedup'] for d in weight_data]) / len(weight_data)
            avg_bandwidth_weight = sum([d['bandwidth'] for d in weight_data]) / len(weight_data)
            print(f"\n带weight测试性能总结:")
            print(f"  平均加速比: {avg_speedup_weight:.2f}x")
            print(f"  平均带宽: {avg_bandwidth_weight:.2f} GB/s")

    print("\n")
    if all_passed:
        print("=" * 80)
        print(f"✓✓✓ 所有 {len(results)} 组测试全部通过！✓✓✓")
        print("=" * 80)
        print("精度验证成功，所有余弦相似度 > 0.9999")
        print("包含：无weight测试 + q有weight测试 + kv有weight测试 + 两者都有weight测试")
        print(f"性能测试完成，平均加速比 {avg_speedup_all:.2f}x")
    else:
        print("=" * 80)
        print("✗✗✗ 部分测试失败！✗✗✗")
        print("=" * 80)
        print("请检查失败的测试配置")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)