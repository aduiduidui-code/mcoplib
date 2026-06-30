// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Router GEMM: activation(T) x weight(fp32) -> fp32, H=3072, E=256, M<=32.
// Supports bf16 or fp32 activation; weight is always fp32.
// Extremely Optimized for Metax C500 (sm_80, 104 SMs, 1.55 TB/s).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <stdexcept>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ---------------------------------------------------------------------------
// 128-bit Vectorized Load Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void load_weight_8(float const* __restrict__ ptr, float* dst) {
  float4 v0 = *reinterpret_cast<float4 const*>(ptr);
  float4 v1 = *reinterpret_cast<float4 const*>(ptr + 4);
  dst[0] = v0.x; dst[1] = v0.y; dst[2] = v0.z; dst[3] = v0.w;
  dst[4] = v1.x; dst[5] = v1.y; dst[6] = v1.z; dst[7] = v1.w;
}

__device__ __forceinline__ void load_activation_fp32_8(float const* __restrict__ ptr, float* dst) {
  float4 v0 = *reinterpret_cast<float4 const*>(ptr);
  float4 v1 = *reinterpret_cast<float4 const*>(ptr + 4);
  dst[0] = v0.x; dst[1] = v0.y; dst[2] = v0.z; dst[3] = v0.w;
  dst[4] = v1.x; dst[5] = v1.y; dst[6] = v1.z; dst[7] = v1.w;
}

__device__ __forceinline__ void load_activation_bf16_8(__nv_bfloat16 const* __restrict__ ptr, float* dst) {
  uint4 v = *reinterpret_cast<uint4 const*>(ptr);
  __nv_bfloat162 const* bf16x2_ptr = reinterpret_cast<__nv_bfloat162 const*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f2 = __bfloat1622float2(bf16x2_ptr[i]);
    dst[i * 2] = f2.x;
    dst[i * 2 + 1] = f2.y;
  }
}

// ---------------------------------------------------------------------------
// SIMD-16 Reduction for Metax C500 micro-architecture
// ---------------------------------------------------------------------------

__device__ __forceinline__ float simd16_reduce_sum(float val) {
  val += __shfl_down_sync_16(0xffffffffffffffff, val, 8);
  val += __shfl_down_sync_16(0xffffffffffffffff, val, 4);
  val += __shfl_down_sync_16(0xffffffffffffffff, val, 2);
  val += __shfl_down_sync_16(0xffffffffffffffff, val, 1);
  return val;
}

// ---------------------------------------------------------------------------
// BF16 Activation Path - K-Loop Eliminated (One-Wave Execution)
// ---------------------------------------------------------------------------

template <int kBlockSize, int kNumTokens, int kNumExperts, int kHiddenDim>
__global__ void fp32_router_gemm_kernel_bf16(
    float* __restrict__ out, __nv_bfloat16 const* __restrict__ mat_a,
    float const* __restrict__ mat_b) {
  
  constexpr int VPT = 8;
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;

  int const simd16_lane = tid & 15;
  int const simd16_group = tid >> 4; // Max 23 for BlockSize=384

  float acc[kNumTokens] = {};

  int const k_base = tid * VPT;
  float const* b_col = mat_b + n_idx * kHiddenDim;
  
  // SREG Caching: Weight loaded once and kept physically in registers
  float b_float[8];
  load_weight_8(b_col + k_base, b_float);

  // M-loop completely unrolled for high density FFMA
#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
    float a_float[8];
    load_activation_bf16_8(mat_a + m_idx * kHiddenDim + k_base, a_float);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      acc[m_idx] += a_float[k] * b_float[k];
    }
  }

  // 384 threads = 24 SIMD-16 groups
  __shared__ float sm_reduction[kNumTokens][24];

#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = simd16_reduce_sum(acc[m]);
    if (simd16_lane == 0) {
      sm_reduction[m][simd16_group] = sum;
    }
  }

  __syncthreads();

  // Phase 2 Reduction: First 32 threads compress the 24 partial values
  if (tid < 32) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float val = (tid < 24) ? sm_reduction[m][tid] : 0.0f;
      val = simd16_reduce_sum(val);
      if (simd16_lane == 0) {
        sm_reduction[m][simd16_group] = val; // Writes to indices 0 and 1
      }
    }
  }

  __syncthreads();

  // Final scalar add and global memory commit
  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      out[m * kNumExperts + n_idx] = sm_reduction[m][0] + sm_reduction[m][1];
    }
  }
}

// ---------------------------------------------------------------------------
// FP32 Activation Path - K-Loop Eliminated (One-Wave Execution)
// ---------------------------------------------------------------------------

template <int kBlockSize, int kNumTokens, int kNumExperts, int kHiddenDim>
__global__ void fp32_router_gemm_kernel_fp32(
    float* __restrict__ out, float const* __restrict__ mat_a,
    float const* __restrict__ mat_b) {
  
  constexpr int VPT = 8;
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;

  int const simd16_lane = tid & 15;
  int const simd16_group = tid >> 4;

  float acc[kNumTokens] = {};

  int const k_base = tid * VPT;
  float const* b_col = mat_b + n_idx * kHiddenDim;
  
  float b_float[8];
  load_weight_8(b_col + k_base, b_float);

#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
    float a_float[8];
    load_activation_fp32_8(mat_a + m_idx * kHiddenDim + k_base, a_float);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      acc[m_idx] += a_float[k] * b_float[k];
    }
  }

  __shared__ float sm_reduction[kNumTokens][24];

#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = simd16_reduce_sum(acc[m]);
    if (simd16_lane == 0) {
      sm_reduction[m][simd16_group] = sum;
    }
  }

  __syncthreads();

  if (tid < 32) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float val = (tid < 24) ? sm_reduction[m][tid] : 0.0f;
      val = simd16_reduce_sum(val);
      if (simd16_lane == 0) {
        sm_reduction[m][simd16_group] = val;
      }
    }
  }

  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      out[m * kNumExperts + n_idx] = sm_reduction[m][0] + sm_reduction[m][1];
    }
  }
}

// ---------------------------------------------------------------------------
// C500 Tuned Launcher
// ---------------------------------------------------------------------------

template <typename InputT, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeFp32RouterGemm(float* output, InputT const* mat_a,
                          float const* mat_b, cudaStream_t stream) {
  // OPTIMIZED for C500: Replaced 128 with 384 for one-wave grid execution
  constexpr int kBlockSize = 384; 
  if constexpr (std::is_same_v<InputT, __nv_bfloat16>) {
    fp32_router_gemm_kernel_bf16<kBlockSize, kNumTokens, kNumExperts, kHiddenDim>
        <<<kNumExperts, kBlockSize, 0, stream>>>(output, mat_a, mat_b);
  } else {
    fp32_router_gemm_kernel_fp32<kBlockSize, kNumTokens, kNumExperts, kHiddenDim>
        <<<kNumExperts, kBlockSize, 0, stream>>>(output, mat_a, mat_b);
  }
}

#define INSTANTIATE(T, M)                                      \
  template void invokeFp32RouterGemm<T, M, 256, 3072>(         \
      float*, T const*, float const*, cudaStream_t);

#define INSTANTIATE_ALL(T) \
  INSTANTIATE(T, 1)        \
  INSTANTIATE(T, 2)        \
  INSTANTIATE(T, 3)        \
  INSTANTIATE(T, 4)        \
  INSTANTIATE(T, 5)        \
  INSTANTIATE(T, 6)        \
  INSTANTIATE(T, 7)        \
  INSTANTIATE(T, 8)        \
  INSTANTIATE(T, 9)        \
  INSTANTIATE(T, 10)       \
  INSTANTIATE(T, 11)       \
  INSTANTIATE(T, 12)       \
  INSTANTIATE(T, 13)       \
  INSTANTIATE(T, 14)       \
  INSTANTIATE(T, 15)       \
  INSTANTIATE(T, 16)       \
  INSTANTIATE(T, 17)       \
  INSTANTIATE(T, 18)       \
  INSTANTIATE(T, 19)       \
  INSTANTIATE(T, 20)       \
  INSTANTIATE(T, 21)       \
  INSTANTIATE(T, 22)       \
  INSTANTIATE(T, 23)       \
  INSTANTIATE(T, 24)       \
  INSTANTIATE(T, 25)       \
  INSTANTIATE(T, 26)       \
  INSTANTIATE(T, 27)       \
  INSTANTIATE(T, 28)       \
  INSTANTIATE(T, 29)       \
  INSTANTIATE(T, 30)       \
  INSTANTIATE(T, 31)       \
  INSTANTIATE(T, 32)

INSTANTIATE_ALL(float)
INSTANTIATE_ALL(__nv_bfloat16)

#undef INSTANTIATE_ALL
#undef INSTANTIATE