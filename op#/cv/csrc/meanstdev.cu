#include "utils.h"
#include "calsum.h"

template<typename T>
__device__ __forceinline__ T ScanWarp(T val) {
  int32_t lane = threadIdx.x & 63;
  T tmp = __shfl_up_sync(0xffffffffffffffff, val, 1);
  if (lane >= 1) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 2);
  if (lane >= 2) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 4);
  if (lane >= 4) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 8);
  if (lane >= 8) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 16);
  if (lane >= 16) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 32);
  if (lane >= 32) {
    val += tmp;
  }
  return val;
}

template<typename T, typename D, int NUM_THREADS>
__global__ void mean_no_mask_kernel(const T* __restrict input, D* __restrict mean, int width, int height, int stride, D scale) {
    constexpr int N = 16 / sizeof(T);
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    int num_elems = height * stride;
    D sum = (D)0;
    if(index < num_elems) {
        if(index >= num_elems - N) {
            for(int i = index; i < num_elems; i++) {
                int h, w;
                h = i / stride;
                w = i % stride;
                if(w < width) {
                    sum += (D)input[i] * scale;
                }
            }
        } else {
            float4 reg_input = *(float4*)(input + index);
            T* ptr_reg_input = (T*)&reg_input;
            #pragma unroll N
            for(int i = 0; i < N; i++) {
                int j = index + i;
                int h, w;
                h = j / stride;
                w = j % stride;
                if(w < width) {
                    sum += (D)ptr_reg_input[i] *scale;
                }
            }
        }
    }
    constexpr int WARPS_NUM = NUM_THREADS >> 6;
    __shared__ D acc_buffer[WARPS_NUM];
    sum = ScanWarp<D>(sum);
    int warp_id = threadIdx.x >> 6;
    int lane_id = threadIdx.x & 63;
    if(lane_id == 0) {
        acc_buffer[warp_id] = sum;
    }
    __syncthreads();
    for(int stride = WARPS_NUM >> 1; stride > 0; stride = stride >> 1) {
        if(threadIdx.x < stride) {
            acc_buffer[threadIdx.x] += acc_buffer[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        atomicAdd(mean, acc_buffer[0]);
    }
}

template<typename T, typename D, int NUM_THREADS>
__global__ void std_no_mask_kernel(const T* __restrict input, const D* __restrict mean, D* __restrict std, int width, int height, int stride, D scale) {
    constexpr int N = 16 / sizeof(T);
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    int num_elems = height * stride;
    D sum = (D)0;
    D value_mean = mean[0];
    if(index < num_elems) {
        if(index >= num_elems - N) {
            for(int i = index; i < num_elems; i++) {
                int h, w;
                h = i / stride;
                w = i % stride;
                if(w < width) {
                    D value = ((D)input[i] - value_mean);
                    sum += value * value * scale;
                }
            }
        } else {
            float4 reg_input = *(float4*)(input + index);
            T* ptr_reg_input = (T*)&reg_input;
            #pragma unroll N
            for(int i = 0; i < N; i++) {
                int j = index + i;
                int h, w;
                h = j / stride;
                w = j % stride;
                if(w < width) {
                    D value = ((D)ptr_reg_input[i] - value_mean);
                    sum += value * value *scale;
                }
            }
        }
    }
    constexpr int WARPS_NUM = NUM_THREADS >> 6;
    __shared__ D acc_buffer[WARPS_NUM];
    sum = ScanWarp<D>(sum);
    int warp_id = threadIdx.x >> 6;
    int lane_id = threadIdx.x & 63;
    if(lane_id == 0) {
        acc_buffer[warp_id] = sum;
    }
    __syncthreads();
    for(int stride = WARPS_NUM >> 1; stride > 0; stride = stride >> 1) {
        if(threadIdx.x < stride) {
            acc_buffer[threadIdx.x] += acc_buffer[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        atomicAdd(std, acc_buffer[0]);
    }
}

template<typename T, typename D>
void MeanStdevOp(const T* __restrict input, D* __restrict mean, D* __restrict std, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    D scale = (D)1.0 / (D)(width*height);
    mean_no_mask_kernel<T, D, num_threads><<<gridsize, num_threads, 0, stream>>>(input, mean, width, height, stride, scale);
    std_no_mask_kernel<T, D, num_threads><<<gridsize, num_threads, 0, stream>>>(input, mean, std, width, height, stride, scale);
}

template void MeanStdevOp<uchar, float>(const uchar* __restrict input, float* __restrict mean, float* __restrict std, int width, int height, int stride, cudaStream_t stream);
template void MeanStdevOp<float, float>(const float* __restrict input, float* __restrict mean, float* __restrict std, int width, int height, int stride, cudaStream_t stream);
