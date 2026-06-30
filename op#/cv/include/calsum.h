#ifndef _CALSUM_
#define _CALSUM_
#include <stdint.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename T, typename D>
void CalsumOp(const T* __restrict input, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CalsumOp<uchar, double>(const uchar* __restrict input, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CalsumOp<schar, double>(const schar* __restrict input, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CalsumOp<ushort, double>(const ushort* __restrict input, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CalsumOp<short, double>(const short* __restrict input, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CalsumOp<int, double>(const int* __restrict input, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CalsumOp<float, double>(const float* __restrict input, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void CalsumOp<double, double>(const double* __restrict input, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
#endif//_CALSUM_