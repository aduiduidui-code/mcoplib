import torch
from mcoplib.custom_ops import fused_add_rms_norm_dynamic_per_token_quant_padding_output, fused_fp8_add_rms_norm_dynamic_per_token_quant_padding_output

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()

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

def torch_func(x, residual, weight, eps):
    sum = x + residual
    # import pdb
    # pdb.set_trace()
    rms_res = torch.rms_norm(sum, sum.shape, weight=weight.unsqueeze(0).expand(x.shape), eps=eps)
    return sum, rms_res

def tmp_compare(tensor_packed, tensor_tuple):
    valid_size = hidden_size //2
    x_quant = tensor_packed[:, :valid_size].view(torch.int8).contiguous()
    x_scale = tensor_packed[:, valid_size:valid_size+2].view(torch.float32).contiguous()
    ref_quant = tensor_tuple[0]
    ref_scale = tensor_tuple[1]
    print(f"x_quant diff: {calc_diff(x_quant, ref_quant)}  x_scale diff: {calc_diff(x_scale, ref_scale)}")

def test_rms_norm_custom(warmup=10, rep=100):
    eps = 1e-6
    dtype = torch.bfloat16
    batch_size = 2048
    hidden_size = 3072
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    
    dur_forward = benchmark(fused_add_rms_norm_dynamic_per_token_quant_padding_output, (x, residual, weight, 1792, eps), warmup, rep)
    print(f"batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype}, infi_forward={dur_forward.mean().item()}")

    sum_torch, rms_torch = torch_func(x, residual, weight,eps)
    output_, residual_, output_rms_, output_quant_int8_, scales_ = fused_add_rms_norm_dynamic_per_token_quant_padding_output(x, residual, weight, 1792, eps)
    tmp_compare(output_, (output_quant_int8_, scales_))
    torch.allclose(sum_torch, residual_)
    torch.allclose(rms_torch, output_rms_)
    print("precision check pass")

def test_rms_norm_fp8_custom(warmup=10, rep=100):
    eps = 1e-6
    dtype = torch.bfloat16
    batch_size = 1
    hidden_size = 7168
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    
    dur_forward = benchmark(fused_fp8_add_rms_norm_dynamic_per_token_quant_padding_output, (x, residual, weight, 3840, eps), warmup, rep)
    print(f"batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype}, infi_forward={dur_forward.mean().item()}")

    sum_torch, rms_torch = torch_func(x, residual, weight,eps)
    output_, residual_, output_rms_, output_quant_int8_, scales_ = fused_fp8_add_rms_norm_dynamic_per_token_quant_padding_output(x, residual, weight, 3840, eps)

    torch.allclose(sum_torch, residual_)
    torch.allclose(rms_torch, output_rms_)
    print("precision check pass")

if __name__ == "__main__":
    packed_size = 1792
    hidden_size = 3072
    # In this case, Megatron foward cost is 700 us , inficom forward cost is 200 us 
    # we assume the similar speedup in backward 
    test_rms_norm_custom(10, 100)
    # test_rms_norm_fp8_custom(10,100) 
