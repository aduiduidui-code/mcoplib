#ifndef _COUNT_NOZERO_
#define _COUNT_NOZERO_
#include <stdint.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename T, typename D>
void CountNoZeroOp(const T* __restrict input, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CountNoZeroOp<uchar, int>(const uchar* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CountNoZeroOp<schar, int>(const schar* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CountNoZeroOp<ushort, int>(const ushort* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CountNoZeroOp<short, int>(const short* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CountNoZeroOp<int, int>(const int* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CountNoZeroOp<float, int>(const float* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CountNoZeroOp<double, int>(const double* __restrict input, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
#endif//_COUNT_NOZERO_