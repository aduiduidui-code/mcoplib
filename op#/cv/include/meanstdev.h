#ifndef _MEANSTDEV_
#define _MEANSTDEV_
#include <stdint.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T, typename D>
void MeanStdevOp(const T* __restrict input, D* __restrict mean, D* __restrict std, int width, int height, int stride, cudaStream_t stream);
template void MeanStdevOp<uchar, float>(const uchar* __restrict input, float* __restrict mean, float* __restrict std, int width, int height, int stride, cudaStream_t stream);
template void MeanStdevOp<float, float>(const float* __restrict input, float* __restrict mean, float* __restrict std, int width, int height, int stride, cudaStream_t stream);

#endif//_MEANSTDEV_