#!/usr/bin/env python3
"""
性能对比：torch.compile (3 Triton kernels) vs 手写 CUDA (1 kernel)
M = batch_size × dp_size (dp_size = 4)
误差要求：0（完全相等）
"""

import torch
import mcoplib._C
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# 方案A：原始代码 + torch.compile
# ============================================================================
@torch.compile(dynamic=True)
def original_unpack(packed, topk, n):
    weights = packed[:, :topk].contiguous()
    ids = packed[:, topk:2*topk].contiguous().to(torch.int32)
    scale = packed[:, 2*topk:].contiguous()
    return weights, ids, scale


# ============================================================================
# 方案B：手写融合 CUDA kernel
# ============================================================================
def fused_unpack_wrapper(packed, topk, n):
    M = packed.size(0)
    weights = torch.empty(M, topk, device='cuda', dtype=torch.float32)
    ids = torch.empty(M, topk, device='cuda', dtype=torch.int32)
    scale = torch.empty(M, n, device='cuda', dtype=torch.float32)
    torch.ops._C.fused_unpack(packed, topk, n, weights, ids, scale)
    return weights, ids, scale


def benchmark(func, packed, topk, n, num_warmup=100, num_iter=1000):
    for _ in range(num_warmup):
        func(packed, topk, n)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iter):
        func(packed, topk, n)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / num_iter


def main():
    dp_size = 4
    
    # 测试配置：场景, bs, topk, n
    configs = [
        # decode 场景
        ("decode", 32, 8, 1),
        ("decode", 32, 8, 4),
        ("decode", 32, 4, 1),
        ("decode", 32, 16, 1),
        # prefill 场景
        ("prefill", 2048, 8, 1),
        ("prefill", 2048, 8, 4),
        ("prefill", 2048, 4, 1),
        ("prefill", 2048, 16, 1),
        # large 场景
        ("large", 8192, 8, 1),
        ("large", 8192, 8, 4),
        ("large", 8192, 4, 1),
        ("large", 8192, 16, 1),
    ]
    
    print(f"{'场景':<8} {'bs':<6} {'M':<8} {'topk':<6} {'n':<4} {'torch.compile(ms)':<18} {'手写CUDA(ms)':<18} {'加速比':<8} {'精度误差'}")
    print("-" * 110)
    
    for name, bs, topk, n in configs:
        M = bs * dp_size
        packed = torch.randn(M, 2*topk + n, device='cuda', dtype=torch.float32)
        
        # 正确性验证（误差为 0）
        with torch.no_grad():
            w1, i1, s1 = original_unpack(packed, topk, n)
            w2, i2, s2 = fused_unpack_wrapper(packed, topk, n)
        
        weights_equal = torch.all(w1 == w2)
        ids_equal = torch.all(i1 == i2)
        scale_equal = torch.all(s1 == s2)
        all_equal = weights_equal and ids_equal and scale_equal
        
        if not all_equal:
            print(f"{name:<8} {bs:<6} {M:<8} {topk:<6} {n:<4} {'-':<18} {'-':<18} {'-':<8} FAIL")
            continue
        
        # 性能测试
        time_original = benchmark(original_unpack, packed, topk, n)
        time_fused = benchmark(fused_unpack_wrapper, packed, topk, n)
        speedup = time_original / time_fused
        
        print(f"{name:<8} {bs:<6} {M:<8} {topk:<6} {n:<4} {time_original:<18.4f} {time_fused:<18.4f} {speedup:<8.2f}x 0")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    
    main()