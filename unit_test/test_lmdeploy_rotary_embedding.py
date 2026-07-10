#!/usr/bin/env python3
"""
OP算子性能测试benchmark代码
用于对比mcoplib_ops和maca_extension_ops两个算子库的性能和精度
"""

import os
import sys
import time
import torch
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
import argparse
from functools import partial
import mcoplib.lmdeploy
import mcoplib._C
# 添加项目路径到sys.path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class OpsBenchmark:
    """算子性能测试基类"""
    
    def __init__(self):

        print("starting to setting device and dtype")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16  # 默认使用半精度

        try:
            from mcoplib import lmdeploy as mcoplib_ops
            self.mcoplib_ops = mcoplib_ops
            print("✓ Successfully imported mcoplib ops")
        except ImportError as e:
            print(f"✗ Failed to import mcoplib ops: {e}")
            self.mcoplib_ops = None
        try:
            from mcoplib import op as op_rms_norm
            self.op_rms_norm = op_rms_norm
            print("✓ Successfully imported mcoplib rms_norm")
        except ImportError as e:
            print(f"✗ Failed to import mcoplib rms_norm: {e}")
            self.op_rms_norm = None

    
    def measure_time(self, func: Callable, *args, warmup: int = 3, repeat: int = 10, **kwargs) -> Tuple[float, Any]:
        """测量函数执行时间"""
        # 预热
        for _ in range(warmup):
            _ = func(*args, **kwargs)
        
        # 正式测量
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(repeat):
            result = func(*args, **kwargs)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / repeat * 1e6  # 转换为微秒
        return avg_time, result
    
    def compare_ops(self, op_name: str, op_func1: Callable, input_data: Dict[str, Any], shapes_info: str = "") -> Dict[str, Any]:
        """对比两个算子的性能和精度"""
        if self.mcoplib_ops is None:
            return {"error": "One or both ops libraries not available"}
        
        print(f"\n{'='*60}")
        print(f"Testing {op_name} - {shapes_info}")
        print(f"{'='*60}")
        
        try:
            # 测试maca_ext_ops
            time1, result1 = self.measure_time(op_func1, **input_data)
            print(f"mcoplib_time {op_name}: {time1:.2f} μs ")
            

            return {
                "op_name": op_name,
                "shapes_info": shapes_info,
                "mcoplib_time": time1,
                "shapes": {k: v.shape if hasattr(v, 'shape') else str(v) for k, v in input_data.items()}
            }

                
        except Exception as e:
            print(f"Error testing {op_name}: {e}")
            return {"op_name": op_name, "error": str(e)}
    
    def generate_test_data(self, shape: Tuple[int, ...], dtype: torch.dtype = None) -> torch.Tensor:
        """生成测试数据"""
        if dtype is None:
            dtype = self.dtype
        
        # 使用固定随机种子生成数据
        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)
        
        return torch.randn(shape, dtype=dtype, device=self.device, generator=generator)
    
    def run_benchmark(self) -> List[Dict[str, Any]]:
        """运行所有算子的benchmark测试"""
        results = []
        
        # 需要测试的算子列表
        ops_to_test = [
            "apply_rotary_pos_emb", 
        ]
        
        for op_name in ops_to_test:
            try:
                op_results = self.test_single_op(op_name)
                results.extend(op_results)
            except Exception as e:
                print(f"Failed to test {op_name}: {e}")
                results.append({
                    "op_name": op_name,
                    "error": str(e)
                })
        
        return results
    
    def test_single_op(self, op_name: str) -> List[Dict[str, Any]]:
        """测试单个算子"""
        results = []
        
        # 定义不同算子的测试shape组合
        test_shapes = {

            "apply_rotary_pos_emb": [
                (128, 8, 64),     # 小shape (batch, num_heads, head_dim)
                (1024, 32, 128),  # 中等shape
                (4096, 64, 256),  # 大shape
            ]
        }
        
        # 获取当前算子的测试shapes
        shapes = test_shapes.get(op_name, [(128, 512)])  # 默认使用小shape
        
        for shape in shapes:
            try:
                if op_name == "apply_rotary_pos_emb":
                    result = self.test_apply_rotary_pos_emb(shape)
                else:
                    print(f"Unknown op: {op_name}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                print(f"Failed to test {op_name} with shape {shape}: {e}")
                results.append({
                    "op_name": op_name,
                    "shapes_info": str(shape),
                    "error": str(e)
                })
        
        return results
    
    
    def test_apply_rotary_pos_emb(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """测试apply_rotary_pos_emb算子"""
        batch_size, num_heads, head_dim = shape
        
        # 生成测试数据
        query = self.generate_test_data((batch_size * num_heads, head_dim))
        key = self.generate_test_data((batch_size * num_heads, head_dim))
        cos = self.generate_test_data((batch_size * num_heads, head_dim // 2))
        sin = self.generate_test_data((batch_size * num_heads, head_dim // 2))
        
        # 测试数据
        input_data = {
            "query": query,
            "key": key,
            "cos": cos,
            "sin": sin
        }
        
        def test_mcoplib(**kwargs):
            query = kwargs["query"].clone()
            key = kwargs["key"].clone()
            cos = kwargs["cos"]
            sin = kwargs["sin"]
            query =query.contiguous().unsqueeze(0)
            key = key.contiguous().unsqueeze(0)
            position_ids_1d = torch.arange(0, query.size(1), device=query.device)
            head_size = query.size(-1)
            query = query.flatten(-2, -1)
            key = key.flatten(-2, -1)
            rot_dim = cos.size(-1)
            self.mcoplib_ops.lmdeploy_rotary_embedding(
                    position_ids_1d,
                    query,
                    key,
                    head_size,
                    cos.view(-1, rot_dim),
                    sin.view(-1, rot_dim),
                    True,
            )
            result = query
            return result  # 返回query用于对比
        
        return self.compare_ops("apply_rotary_pos_emb", test_mcoplib, input_data, str(shape))

def main():
    """主函数"""
    
    # 创建benchmark实例
    benchmark = OpsBenchmark()
    results = benchmark.run_benchmark()
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = [r for r in results if 'error' not in r]
    failed_tests = [r for r in results if 'error' in r]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")

if __name__ == "__main__":
    main()