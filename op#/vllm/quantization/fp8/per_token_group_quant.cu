#include "common.cuh"
#include "dispatch_utils.h"
#include "../vectorization_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

namespace vllm {

__device__ __forceinline__ float group_reduce_max_16(float val) {
#ifdef USE_ROCM
  const int lane_in_wave = threadIdx.x % warpSize;
  const unsigned long long mask = 0xFFFFull << ((lane_in_wave / 16) * 16);
  val = fmaxf(val, __shfl_xor_sync(mask, val, 8, 16));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4, 16));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2, 16));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1, 16));
#else
  const unsigned mask = (threadIdx.x % 32 >= 16) ? 0xffff0000u : 0x0000ffffu;
  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
#endif
  return val;
}

template <typename T, bool SCALE_UE8M0>
__device__ __forceinline__ float compute_group_scale(
    const T* __restrict__ group_input, T* __restrict__ smem_group,
    int group_size, int lane_id, int threads_per_group, float eps,
    float max_8bit) {
  float local_absmax = eps;
  constexpr int vec_size = 16 / sizeof(T);

  auto scalar_op_cache = [&] __device__(T & dst, const T& src) {
    float abs_v = fabsf(static_cast<float>(src));
    local_absmax = fmaxf(local_absmax, abs_v);
    dst = src;
  };

  vectorize_with_alignment<vec_size>(group_input, smem_group, group_size,
                                     lane_id, threads_per_group,
                                     scalar_op_cache);

  local_absmax = group_reduce_max_16(local_absmax);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }
  return y_s;
}

template <typename T, typename dst_t>
__device__ __forceinline__ void quantize_group(
    const T* __restrict__ smem_group, dst_t* __restrict__ group_output,
    int group_size, int lane_id, int threads_per_group, float y_s,
    float min_8bit, float max_8bit) {
  constexpr int vec_size = 16 / sizeof(T);

  auto scalar_op_quant = [&] __device__(dst_t & dst, const T& src) {
    float q = fminf(fmaxf(static_cast<float>(src) / y_s, min_8bit), max_8bit);
    dst = static_cast<dst_t>(q);
  };

  vectorize_with_alignment<vec_size>(smem_group, group_output, group_size,
                                     lane_id, threads_per_group,
                                     scalar_op_quant);
}

template <typename T, typename dst_t, bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    float* __restrict__ output_s, int group_size, int groups_per_block,
    float eps, float min_8bit, float max_8bit, int scale_num_rows = 0,
    int64_t scale_stride = 0) {
  constexpr int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = static_cast<int64_t>(blockIdx.x) * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;

  const T* group_input = input + block_group_offset;
  dst_t* group_output = static_cast<dst_t*>(output_q) + block_group_offset;
  float* scale_output = nullptr;

  if constexpr (IS_COLUMN_MAJOR) {
    const int row_idx = global_group_id / scale_num_rows;
    const int col_idx = global_group_id % scale_num_rows;
    scale_output = output_s + col_idx * scale_stride + row_idx;
  } else {
    scale_output = output_s + global_group_id;
  }

  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;

  const float y_s = compute_group_scale<T, SCALE_UE8M0>(
      group_input, smem_group, group_size, lane_id, threads_per_group, eps,
      max_8bit);

  if (lane_id == 0) {
    *scale_output = y_s;
  }

  __syncthreads();

  quantize_group<T, dst_t>(smem_group, group_output, group_size, lane_id,
                           threads_per_group, y_s, min_8bit, max_8bit);
}

inline int get_groups_per_block(int64_t num_groups) {
  if (num_groups % 16 == 0) {
    return 16;
  }
  if (num_groups % 8 == 0) {
    return 8;
  }
  if (num_groups % 4 == 0) {
    return 4;
  }
  if (num_groups % 2 == 0) {
    return 2;
  }
  return 1;
}

}  // namespace vllm

void per_token_group_quant_fp8(torch::Tensor const& input,
                               torch::Tensor& output_q,
                               torch::Tensor& output_s, int64_t group_size,
                               double eps, double fp8_min, double fp8_max,
                               bool scale_ue8m0,
                               bool dummy_is_scale_transposed,
                               bool dummy_is_tma_aligned) {
  (void)dummy_is_scale_transposed;
  (void)dummy_is_tma_aligned;

  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(output_q.is_cuda(), "output_q must be a CUDA tensor");
  TORCH_CHECK(output_s.is_cuda(), "output_s must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output_q.is_contiguous(), "output_q must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  TORCH_CHECK(output_s.dim() == 2, "output_s must be a 2D tensor");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(input.numel() % group_size == 0,
              "input.numel() must be divisible by group_size");
  TORCH_CHECK(input.stride(-1) == 1, "last dimension of input must be contiguous");
  TORCH_CHECK(output_q.stride(-1) == 1,
              "last dimension of output_q must be contiguous");
  TORCH_CHECK(output_s.scalar_type() == at::ScalarType::Float,
              "output_s must have dtype float32");

  const int64_t num_rows = input.size(0);
  const int64_t hidden_size = input.size(1);
  TORCH_CHECK(hidden_size % group_size == 0,
              "input.size(1) must be divisible by group_size");
  TORCH_CHECK(output_q.sizes() == input.sizes(),
              "output_q must have the same shape as input");

  const int64_t groups_per_row = hidden_size / group_size;
  TORCH_CHECK(output_s.size(0) == num_rows && output_s.size(1) == groups_per_row,
              "output_s must have shape [", num_rows, ", ", groups_per_row, "]");

  if (input.numel() == 0) {
    return;
  }

  const int64_t num_groups = input.numel() / group_size;
  const int groups_per_block = vllm::get_groups_per_block(num_groups);
  TORCH_CHECK(num_groups % groups_per_block == 0,
              "num_groups must be divisible by groups_per_block");

  constexpr int threads_per_group = 16;
  const int num_blocks = static_cast<int>(num_groups / groups_per_block);
  const int num_threads = groups_per_block * threads_per_group;
  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = static_cast<int>(output_s.size(1));
  const int64_t scale_stride = output_s.stride(1);

  if (is_column_major) {
    TORCH_CHECK(output_s.stride(0) == 1,
                "column-major output_s must have stride(0) == 1");
  } else {
    TORCH_CHECK(output_s.is_contiguous(),
                "row-major output_s must be contiguous");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_KERNEL(T, DST_DTYPE)                                        \
  do {                                                                     \
    dim3 grid(num_blocks);                                                 \
    dim3 block(num_threads);                                               \
    size_t smem_bytes =                                                    \
        static_cast<size_t>(groups_per_block) * group_size * sizeof(T);    \
    if (is_column_major) {                                                 \
      if (scale_ue8m0) {                                                   \
        vllm::per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true>  \
            <<<grid, block, smem_bytes, stream>>>(                         \
                input.data_ptr<T>(), output_q.data_ptr(),                  \
                output_s.data_ptr<float>(), static_cast<int>(group_size),   \
                groups_per_block, static_cast<float>(eps),                  \
                static_cast<float>(fp8_min), static_cast<float>(fp8_max),   \
                scale_num_rows, scale_stride);                             \
      } else {                                                             \
        vllm::per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false> \
            <<<grid, block, smem_bytes, stream>>>(                         \
                input.data_ptr<T>(), output_q.data_ptr(),                  \
                output_s.data_ptr<float>(), static_cast<int>(group_size),   \
                groups_per_block, static_cast<float>(eps),                  \
                static_cast<float>(fp8_min), static_cast<float>(fp8_max),   \
                scale_num_rows, scale_stride);                             \
      }                                                                    \
    } else {                                                               \
      if (scale_ue8m0) {                                                   \
        vllm::per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, true> \
            <<<grid, block, smem_bytes, stream>>>(                         \
                input.data_ptr<T>(), output_q.data_ptr(),                  \
                output_s.data_ptr<float>(), static_cast<int>(group_size),   \
                groups_per_block, static_cast<float>(eps),                  \
                static_cast<float>(fp8_min), static_cast<float>(fp8_max));  \
      } else {                                                             \
        vllm::per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, false>\
            <<<grid, block, smem_bytes, stream>>>(                         \
                input.data_ptr<T>(), output_q.data_ptr(),                  \
                output_s.data_ptr<float>(), static_cast<int>(group_size),   \
                groups_per_block, static_cast<float>(eps),                  \
                static_cast<float>(fp8_min), static_cast<float>(fp8_max));  \
      }                                                                    \
    }                                                                      \
  } while (0)

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_fp8_input_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            output_q.scalar_type(), "per_token_group_quant_fp8_output_type",
            [&] { LAUNCH_KERNEL(scalar_t, fp8_t); });
      });

#undef LAUNCH_KERNEL

  AT_CUDA_CHECK(cudaGetLastError());
}
