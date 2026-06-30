// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace fused_mla {
// RMS(x) = sqrt(sum(x[i]*x[i]) / KV_LORA_RANK)
// OUT(x) = x / (RMS(x) + eps)
template<typename scalar_t, int KV_LORA_RANK=512>
__device__ __forceinline__ void __rms_norm(int tid, scalar_t (&w)[8], scalar_t (&v)[8], float eps = 1e-06) {
    //Step 1: read cache, here we assume that each block have 256 threads, and KV_LORA_RANK=512
    float r[8];
    float rr[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = __bfloat162float(v[i]);
        r[i] = val;
        rr[i] = val * val;
    }

    float local_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        local_sum += rr[i];
    }
    //Now shlf on threads
    constexpr int kWarpSize = 64;
    #pragma unroll
    for (int32_t mask = 32; mask >= 1; mask /= 2) {
        local_sum += __shfl_xor_sync(0xffffffffffffffff, local_sum, mask, kWarpSize);
    }

    float rms_x_f = rsqrtf(local_sum / KV_LORA_RANK + eps);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] *= rms_x_f;
        v[i] = __float2bfloat16_rn(r[i] * __bfloat162float(w[i]));
    }
}

template<typename scalar_t, int KV_LORA_RANK=512>
__device__ __forceinline__ void rms_norm(int tid, const scalar_t* norm_weight, const scalar_t* cache, scalar_t* dst_cache0, float eps = 1e-06) {
    scalar_t v[8];
    scalar_t w[8];
    *(float4*)v = *((float4*)cache + tid);
    *(float4*)w = *((float4*)norm_weight + tid);
    __rms_norm(tid, w, v, eps);
    *((float4*)dst_cache0 + tid) = *(float4*)v;
}

template<typename scalar_t, int KV_LORA_RANK=512>
__device__ __forceinline__ void rms_norm(int tid, const scalar_t* norm_weight, const scalar_t* cache, scalar_t* dst_cache0, scalar_t* dst_cache1, float eps = 1e-06) {
    scalar_t v[8];
    scalar_t w[8];
    *(float4*)v = *((float4*)cache + tid);
    *(float4*)w = *((float4*)norm_weight + tid);
    __rms_norm(tid, w, v, eps);
    *((float4*)(dst_cache0) + tid) = *(float4*)v;
    *((float4*)(dst_cache1) + tid) = *(float4*)v;
}

}