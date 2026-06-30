#include "utils.h"
#include "count_nozero.h"

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
__global__ void count_nonzero_no_mask_kernel(const T* __restrict input, D* __restrict dst, int width, int height, int stride) {
    constexpr int N = 16 / sizeof(T);
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    int num_elems = height * stride;
    D count = (D)0;
    if(index < num_elems) {
        if(index >= num_elems - N) {
            for(int i = index; i < num_elems; i++) {
                int h, w;
                h = i / stride;
                w = i % stride;
                if(w < width) {
                    count += (input[i] != (T)0);
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
                    count += (ptr_reg_input[i] != (T)0);
                }
            }
        }
    }
    constexpr int WARPS_NUM = NUM_THREADS >> 6;
    __shared__ D acc_buffer[WARPS_NUM];
    count = ScanWarp<D>(count);
    int warp_id = threadIdx.x >> 6;
    int lane_id = threadIdx.x & 63;
    if(lane_id == 0) {
        acc_buffer[warp_id] = count;
    }
    __syncthreads();
    for(int stride = WARPS_NUM >> 1; stride > 0; stride = stride >> 1) {
        if(threadIdx.x < stride) {
            acc_buffer[threadIdx.x] += acc_buffer[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        atomicAdd(dst, acc_buffer[0]);
    }
}

template<typename T, typename D, int NUM_THREADS>
__global__ void count_nonzero_mask_kernel(const T* __restrict input, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride) {
    constexpr int N = 16 / sizeof(T);
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    int num_elems = height * stride;
    D count = (D)0;
    if(index < num_elems) {
        if(index >= num_elems - N) {
            for(int i = index; i < num_elems; i++) {
                int h, w;
                h = i / stride;
                w = i % stride;
                if(w < width && mask[i]) {
                    count += (input[i] != (T)0);
                }
            }
        } else {
            float4 reg_input = *(float4*)(input + index);
            T* ptr_reg_input = (T*)&reg_input;
            unsigned char mask_buffer[N];
            copy<N>((void *)(mask + index), (void*)mask_buffer);
            #pragma unroll N
            for(int i = 0; i < N; i++) {
                int j = index + i;
                int h, w;
                h = j / stride;
                w = j % stride;
                if(w < width && mask_buffer[i]) {
                    count += (ptr_reg_input[i] != (T)0);
                }
            }
        }
    }
    constexpr int WARPS_NUM = NUM_THREADS >> 6;
    __shared__ D acc_buffer[WARPS_NUM];
    count = ScanWarp<D>(count);
    int warp_id = threadIdx.x >> 6;
    int lane_id = threadIdx.x & 63;
    if(lane_id == 0) {
        acc_buffer[warp_id] = count;
    }
    __syncthreads();
    for(int stride = WARPS_NUM >> 1; stride > 0; stride = stride >> 1) {
        if(threadIdx.x < stride) {
            acc_buffer[threadIdx.x] += acc_buffer[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        atomicAdd(dst, acc_buffer[0]);
    }
}

template<typename T, typename D>
void CountNoZeroOp(const T* __restrict input, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    if(mask) {
        count_nonzero_mask_kernel<T, D, num_threads><<<gridsize, num_threads, 0, stream>>>(input, mask, dst, width, height, stride);
    } else {
        count_nonzero_no_mask_kernel<T, D, num_threads><<<gridsize, num_threads, 0, stream>>>(input, dst, width, height, stride);
    }
}

template void CountNoZeroOp<uchar, int>(const uchar* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void CountNoZeroOp<schar, int>(const schar* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void CountNoZeroOp<ushort, int>(const ushort* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void CountNoZeroOp<short, int>(const short* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void CountNoZeroOp<int, int>(const int* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void CountNoZeroOp<float, int>(const float* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void CountNoZeroOp<double, int>(const double* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);