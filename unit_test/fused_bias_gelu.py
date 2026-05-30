# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

# from megatron.core.jit import jit_fuser
#from mcoplib.op import fused_gelu_fwd, fused_gelu_bwd
from mcoplib import op

def benchmark(func, args, warmup=2, rep=10):
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
    return dur

def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return ff * g

class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp

def test_gelu(m, n, dtype,warmup=10, rep=100):
    input = torch.randn(m, 1, n, dtype=dtype, device='cuda', requires_grad=True)
    # [TODO] support bias with non-zero value
    bias = torch.zeros(n, dtype=dtype, device='cuda', requires_grad=True)
    grad_output = torch.randn(m, 1, n, dtype=dtype, device='cuda', requires_grad=True)
    
    output_ref = bias_gelu(bias, input)
    output_flash = op.fused_gelu_fwd(input, bias)
    all_close = torch.allclose(output_ref, output_flash, atol=1e-2, rtol=1e-2)
    assert all_close
    grad_input_ref = bias_gelu_back(grad_output, bias, input)
    grad_input_flash = op.fused_gelu_bwd(input, grad_output, bias)
    all_close = torch.allclose(grad_input_ref, grad_input_flash, atol=1e-2, rtol=1e-2)
    assert all_close

    dur_ref = benchmark(bias_gelu, (input, bias), warmup, rep)
    dur_infi = benchmark(op.fused_gelu_fwd, (input,bias), warmup, rep)
    print(f"fwd: m={m}, n={n}, dtype={dtype}, ref={dur_ref.mean().item()}, infi={dur_infi.mean().item()}")
    
    dur_ref = benchmark(bias_gelu_back, (grad_output, bias, input), warmup, rep)
    dur_infi = benchmark(op.fused_gelu_bwd, (input,grad_output,bias), warmup, rep)
    print(f"bwd: m={m}, n={n}, dtype={dtype}, ref={dur_ref.mean().item()}, infi={dur_infi.mean().item()}")
    
if __name__ == "__main__":
    test_gelu(4096, 28672 // 2, torch.float)
    test_gelu(4096, 117, torch.float)
