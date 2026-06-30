#include <cmath>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
//#include "type_convert.cuh"

namespace vllm {

// ---------------------------------------------------------------------------
// Fast path: topk == 8 — float4 vectorized, one thread per row.
// ---------------------------------------------------------------------------
__global__ void fused_unpack_topk8_kernel(
    const float*  packed,
    float*        topk_weights,
    int*          topk_ids,
    float*        scale,
    int M, int n, int packed_stride)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float*  src   = packed + row * packed_stride;
    float*        dst_w = topk_weights + row * 8;
    int*          dst_id = topk_ids + row * 8;
    float*        dst_s = scale + row * n;

    // weights: cols 0-7, 2×float4
    float4 w0 = reinterpret_cast<const float4*>(src)[0];
    float4 w1 = reinterpret_cast<const float4*>(src)[1];
    reinterpret_cast<float4*>(dst_w)[0] = w0;
    reinterpret_cast<float4*>(dst_w)[1] = w1;

    // ids: cols 8-15, float→int32, 2×float4→2×int4
    float4 id0_f = reinterpret_cast<const float4*>(src + 8)[0];
    float4 id1_f = reinterpret_cast<const float4*>(src + 8)[1];
    int4 id0_i = make_int4(static_cast<int>(id0_f.x), static_cast<int>(id0_f.y),
                           static_cast<int>(id0_f.z), static_cast<int>(id0_f.w));
    int4 id1_i = make_int4(static_cast<int>(id1_f.x), static_cast<int>(id1_f.y),
                           static_cast<int>(id1_f.z), static_cast<int>(id1_f.w));
    reinterpret_cast<int4*>(dst_id)[0] = id0_i;
    reinterpret_cast<int4*>(dst_id)[1] = id1_i;

    // scale: cols 16..16+n-1
    for (int i = 0; i < n; ++i) {
        dst_s[i] = src[16 + i];
    }
}

// ---------------------------------------------------------------------------
// Generic fallback for any topk.
// ---------------------------------------------------------------------------
__global__ void fused_unpack_generic_kernel(
    const float*  packed,
    float*        topk_weights,
    int*          topk_ids,
    float*        scale,
    int M, int topk, int n, int packed_stride)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float*  src   = packed + row * packed_stride;
    float*        dst_w = topk_weights + row * topk;
    int*          dst_id = topk_ids + row * topk;
    float*        dst_s = scale + row * n;

    for (int i = 0; i < topk; ++i) {
        dst_w[i] = src[i];
    }
    for (int i = 0; i < topk; ++i) {
        dst_id[i] = static_cast<int>(src[topk + i]);
    }
    for (int i = 0; i < n; ++i) {
        dst_s[i] = src[2 * topk + i];
    }
}
} // namespace vllm

// ---------------------------------------------------------------------------
// Host (called from torch)
// ---------------------------------------------------------------------------
void fused_unpack(
    const torch::Tensor& packed,
    int64_t topk,
    int64_t n,
    torch::Tensor& topk_weights,
    torch::Tensor& topk_ids,
    torch::Tensor& scale)
{
    // Params check
    TORCH_CHECK(packed.is_cuda(), "packed must be on CUDA");
    TORCH_CHECK(packed.dtype() == torch::kFloat32, "packed must be float32");

    auto packed_contiguous = packed.contiguous();
    int M = static_cast<int>(packed_contiguous.size(0));
    int packed_stride = static_cast<int>(packed_contiguous.size(1));

    // Reallocate if output tensor size mismatches
    if (topk_weights.numel() != M * topk) {
        topk_weights = torch::empty({M, topk}, packed.options());
        topk_ids = torch::empty({M, topk}, packed.options().dtype(torch::kInt32));
        scale = torch::empty({M, n}, packed.options());
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;

        if (topk == 8) {
        vllm::fused_unpack_topk8_kernel<<<blocks, threads, 0, stream>>>(
            packed_contiguous.data_ptr<float>(),
            topk_weights.data_ptr<float>(),
            topk_ids.data_ptr<int>(),
            scale.data_ptr<float>(),
            M, static_cast<int>(n), packed_stride);
    } else {
        vllm::fused_unpack_generic_kernel<<<blocks, threads, 0, stream>>>(
            packed_contiguous.data_ptr<float>(),
            topk_weights.data_ptr<float>(),
            topk_ids.data_ptr<int>(),
            scale.data_ptr<float>(),
            M, static_cast<int>(topk), static_cast<int>(n), packed_stride);
    }
}

