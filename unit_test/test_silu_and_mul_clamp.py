#!/usr/bin/env python3
"""
OP算子性能测试benchmark代码
用于测试silu_and_mul_clamp算子性能
"""

import os
import sys
import time
import torch
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Callable

import mcoplib.lmdeploy

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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        try:
            from mcoplib import lmdeploy as mcoplib_ops
            self.mcoplib_ops = mcoplib_ops
            import mcoplib._C
            print("√ Successfully imported mcoplib ops")
        except ImportError as e:
            print(f"X Failed to import mcoplib ops: {e}")
            self.mcoplib_ops = None

        try:
            from mcoplib import op as op_rms_norm
            self.op_rms_norm = op_rms_norm
            print("√ Successfully imported mcoplib rms_norm")
        except ImportError as e:
            print(f"X Failed to import mcoplib rms_norm: {e}")
            self.op_rms_norm = None

    def measure_time(self,
                     func: Callable,
                     *args,
                     warmup: int = 3,
                     repeat: int = 10,
                     **kwargs):

        for _ in range(warmup):
            _ = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(repeat):
            result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / repeat * 1e6

        return avg_time, result

    def compare_ops(self,
                    op_name: str,
                    op_func1: Callable,
                    input_data: Dict[str, Any],
                    shapes_info: str = ""):

        if self.mcoplib_ops is None:
            return {"error": "One or both ops libraries not available"}

        print(f"\n{'='*60}")
        print(f"Testing {op_name} - {shapes_info}")
        print(f"{'='*60}")

        try:

            time1, result1 = self.measure_time(op_func1, **input_data)

            print(f"mcoplib_time {op_name}: {time1:.2f} μs ")

            return {
                "op_name": op_name,
                "shapes_info": shapes_info,
                "mcoplib_time": time1,
                "shapes": {
                    k: v.shape if hasattr(v, "shape") else str(v)
                    for k, v in input_data.items()
                },
            }

        except Exception as e:

            print(f"Error testing {op_name}: {e}")

            return {
                "op_name": op_name,
                "error": str(e),
            }

    def generate_test_data(self,
                           shape: Tuple[int, ...],
                           dtype: torch.dtype = None):

        if dtype is None:
            dtype = self.dtype

        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)

        return torch.randn(
            shape,
            dtype=dtype,
            device=self.device,
            generator=generator,
        )

    def run_benchmark(self):

        results = []

        ops_to_test = [
            "silu_and_mul_clamp",
        ]

        for op_name in ops_to_test:

            try:

                op_results = self.test_single_op(op_name)

                results.extend(op_results)

            except Exception as e:

                print(f"Failed to test {op_name}: {e}")

                results.append(
                    {
                        "op_name": op_name,
                        "error": str(e),
                    })

        return results

    def test_single_op(self, op_name):

        results = []

        test_shapes = {
            "silu_and_mul_clamp": [
                (128, 1024),
                (1024, 2048),
                (4096, 4096),
            ]
        }

        shapes = test_shapes.get(op_name, [(128, 512)])

        for shape in shapes:

            try:

                if op_name == "silu_and_mul_clamp":
                    result = self.test_silu_and_mul_clamp(shape)
                else:
                    print(f"Unknown op: {op_name}")
                    continue

                results.append(result)

            except Exception as e:

                print(
                    f"Failed to test {op_name} with shape {shape}: {e}")

                results.append({
                    "op_name": op_name,
                    "shapes_info": str(shape),
                    "error": str(e)
                })

        return results

    def test_silu_and_mul_clamp(self, shape):

        x = self.generate_test_data((shape[0], shape[1] * 2))

        input_data = {
            "x": x,
            "limit": 7.0,
            "alpha": 1.0,
            "beta": 0.0,
        }

        def test_mcoplib(**kwargs):

            x = kwargs["x"].clone()

            limit = kwargs["limit"]
            alpha = kwargs["alpha"]
            beta = kwargs["beta"]

            d = x.shape[-1] // 2

            output_shape = x.shape[:-1] + (d,)

            out = torch.empty(
                output_shape,
                dtype=x.dtype,
                device=x.device)

            torch.ops._C.silu_and_mul_with_clamp(
                out,
                x,
                limit,
                alpha,
                beta)

            return out

        return self.compare_ops(
            "silu_and_mul_clamp",
            test_mcoplib,
            input_data,
            str(shape))

def main():

    benchmark = OpsBenchmark()

    results = benchmark.run_benchmark()

    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")

    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]

    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")


if __name__ == "__main__":
    main()