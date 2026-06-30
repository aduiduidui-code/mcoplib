// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#ifndef USE_ROCM
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
#endif

#include <cmath>
#include <torch/types.h>

#ifndef USE_ROCM
  #include <c10/util/Float8_e4m3fn.h>
  #define MAYBE_HOST_DEVICE C10_HOST_DEVICE
#else
  #include <ATen/hip/HIPContext.h>
  #include <c10/util/Float8_e4m3fn.h>
  #include <c10/util/Float8_e4m3fnuz.h>
  // ROCm doesn't seem to need C10_HOST_DEVICE for static constexpr
  #define MAYBE_HOST_DEVICE
#endif

static bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;
#else
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string device_arch = dprops->gcnArchName;
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
#endif
}

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

// Using the default max value from pytorch (240.0 0x7F) will cause accuracy
// issues when running dynamic quantization. Here use 224.0 0x7E for rocm.
template <>
struct quant_type_max<c10::Float8_e4m3fnuz> {
  static constexpr c10::Float8_e4m3fnuz val() {
    return c10::Float8_e4m3fnuz(0x7E, c10::Float8_e4m3fnuz::from_bits());
  }
};

template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

template <>
struct min_scaling_factor<int8_t> {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return std::numeric_limits<float>::epsilon();
  }
};

static __forceinline__ __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  //uint32_t dst;
  //asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  //return reinterpret_cast<const int8_t&>(dst);
  int32_t dst;
  dst = __float2int_rn(x);
  dst = min(dst, 127);
  dst = max(dst, -128);
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

template <typename fp8_type>
static __device__ __forceinline__ fp8_type float_to_fp8(float const x) {
  float const r =
      fmax(-quant_type_max_v<fp8_type>, fmin(x, quant_type_max_v<fp8_type>));
  return static_cast<fp8_type>(r);
}

template <typename quant_type_t, bool is_scale_inverted, typename enable = void>
struct ScaledQuant;

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<
    quant_type_t, is_scale_inverted,
    typename std::enable_if_t<std::is_same_v<quant_type_t, int8_t>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_int8_rn(x * scale);
    } else {
      return float_to_int8_rn(x / scale);
    }
  }
};

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<quant_type_t, is_scale_inverted,
                   typename std::enable_if_t<
                       std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                       std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_fp8<quant_type_t>(x * scale);
    } else {
      return float_to_fp8<quant_type_t>(x / scale);
    }
  }
};

template <typename scalar_t, typename quant_type_t, bool is_scale_inverted>
__device__ void scaled_quant_conversion(quant_type_t* __restrict__ output,
                                        scalar_t const* __restrict__ input,
                                        float const scale, int const tid,
                                        int const num_elements,
                                        int const step) {
  for (int i = tid; i < num_elements; i += step) {
    output[i] = ScaledQuant<quant_type_t, is_scale_inverted>(input[i], scale);
  }
}

// Vectorization containers
template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename quant_type_t>
struct __align__(4) q8x4_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>);
  quant_type_t x;
  quant_type_t y;
  quant_type_t z;
  quant_type_t w;
};

template<typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = 64) {
    return __shfl_xor_sync(0xffffffffffffffff, value, laneMask, width);
}

template<typename T, int32_t block_dim>
__device__ __forceinline__ void reduce_max_sum(T& val_max, T& val_sum){
  __shared__ T shared_s[64];
  __shared__ T shared_m[64];
  T val_m = val_max;
  T val_s = val_sum;
  int lane = threadIdx.x & 0x3f;
  int wid = threadIdx.x >> 6;

  for (int mask = 32; mask > 0; mask >>= 1) {
    val_s += WARP_SHFL_XOR<T>(val_s, mask);
    val_m = max(val_m, WARP_SHFL_XOR<T>(val_m, mask));
  }

  if(lane == 0) {
    shared_s[wid] = val_s;
    shared_m[wid] = val_m;
  }
  __syncthreads();
  val_s = (lane < ((block_dim + 63) >> 6)) ? shared_s[lane] : static_cast<T>(0.0f);
  val_m = (lane < ((block_dim + 63) >> 6)) ? shared_m[lane] : static_cast<T>(-9999);
  __syncthreads();

  for (int mask = 32; mask > 0; mask >>= 1) {
    val_s += WARP_SHFL_XOR<T>(val_s, mask);
    val_m = max(val_m, WARP_SHFL_XOR<T>(val_m, mask));
  }
  val_max = val_m;
  val_sum = val_s;
}

namespace vllm {

// has_residual must be true, if residual is not a nullptr
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  // sum of squares
  float ss = 0.0f;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    ss += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, blockDim.x);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  float block_absmax_val_maybe = 0.0f;
  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(x));
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  block_absmax_val_maybe =
      BlockReduce(reduceStore)
          .Reduce(block_absmax_val_maybe, cub::Max{}, blockDim.x);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / 127, 1.5e-6);
    s_token_scale = scale;                 // Shared memory store
    all_token_scales[blockIdx.x] = scale;  // Global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}


template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    // Norm
    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    // Quant
    output[token_offset + i] =
        ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(x, scale);
  }
}

template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_t* __restrict__ output_bf16,
                               scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    // Norm
    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    // Quant
    output_bf16[token_offset + i] = static_cast<scalar_t>(x);
    output[token_offset + i] =
        ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(x, scale);
  }
}

namespace vectorized
{
    
// Compute 1.0/rms(input)
// hidden_size must be a multiple of 4
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  // sum of squares
  float ss = 0.0f;

  int32_t const num_vec_elems = hidden_size >> 2;

#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
    }

    ss += x.x * x.x;
    ss += x.y * x.y;
    ss += x.z * x.z;
    ss += x.w * x.w;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, blockDim.x);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

// Vectorized version of vllm::compute_dynamic_per_token_scales
// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  // Vectorized input/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }


  int32_t const num_vec_elems = hidden_size >> 2;
  float block_absmax_val_maybe = 0.0f;

#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
    }

    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.x * rms) * w.x));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.y * rms) * w.y));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.z * rms) * w.z));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.w * rms) * w.w));
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  block_absmax_val_maybe =
      BlockReduce(reduceStore)
          .Reduce(block_absmax_val_maybe, cub::Max{}, blockDim.x);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / 127, 1.5e-6);
    s_token_scale = scale;                 // shared memory store
    all_token_scales[blockIdx.x] = scale;  // global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}

// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  // Vectorized input/output/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&output[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  int32_t const num_vec_elems = hidden_size >> 2;

// TODO(luka/varun) extract into type-agnostic vectorized quant function to
//  replace scaled_fp8_conversion_vec
#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> const in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
      // Update residual
      r.x = static_cast<scalar_t>(x.x);
      r.y = static_cast<scalar_t>(x.y);
      r.z = static_cast<scalar_t>(x.z);
      r.w = static_cast<scalar_t>(x.w);
      vec_residual[i] = r;
    }

    q8x4_t<scalar_out_t> out;
    out.x = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        static_cast<scalar_t>(x.x * rms) * w.x, scale);
    out.y = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        static_cast<scalar_t>(x.y * rms) * w.y, scale);
    out.z = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        static_cast<scalar_t>(x.z * rms) * w.z, scale);
    out.w = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        static_cast<scalar_t>(x.w * rms) * w.w, scale);
    vec_output[i] = out;
  }
}


template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_t* __restrict__ output_bf16,
                               scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  // Vectorized input/output/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  vec4_t<scalar_t>* vec_output_bf16 =
      reinterpret_cast<vec4_t<scalar_t>*>(&output_bf16[token_offset]);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&output[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  int32_t const num_vec_elems = hidden_size >> 2;

// TODO(luka/varun) extract into type-agnostic vectorized quant function to
//  replace scaled_fp8_conversion_vec
#pragma unroll 4
  for (auto i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> const in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
      // Update residual
      r.x = static_cast<scalar_t>(x.x);
      r.y = static_cast<scalar_t>(x.y);
      r.z = static_cast<scalar_t>(x.z);
      r.w = static_cast<scalar_t>(x.w);
      vec_residual[i] = r;
    }

    q8x4_t<scalar_out_t> out;
    vec4_t<scalar_t> out_bf16;

    out_bf16.x = static_cast<scalar_t>(x.x * rms) * w.x;
    out.x = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        out_bf16.x, scale);
    out_bf16.y = static_cast<scalar_t>(x.y * rms) * w.y;
    out.y = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        out_bf16.y, scale);
    out_bf16.z = static_cast<scalar_t>(x.z * rms) * w.z;
    out.z = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        out_bf16.z, scale);
    out_bf16.w = static_cast<scalar_t>(x.w * rms) * w.w;
    out.w = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
        out_bf16.w, scale);
    vec_output[i] = out;
    vec_output_bf16[i] = out_bf16;
  }
}



} // namespace vectorized

} // namespace vllm