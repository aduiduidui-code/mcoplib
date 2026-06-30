#ifndef _SPLIT_
#define _SPLIT_
#include <stdint.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename T>
void Split2Op(const T* __restrict input, T* __restrict output0, T* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template<typename T>
void Split3Op(const T* __restrict input, T* __restrict output0, T* __restrict output1, T* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template<typename T>
void Split4Op(const T* __restrict input, T* __restrict output0, T* __restrict output1, T* __restrict output2, T* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);

extern template void Split2Op<uchar>(const uchar* __restrict input, uchar* __restrict output0, uchar* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split2Op<ushort>(const ushort* __restrict input, ushort* __restrict output0, ushort* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split2Op<int>(const int* __restrict input, int* __restrict output0, int* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split2Op<double>(const double* __restrict input, double* __restrict output0, double* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);

extern template void Split3Op<uchar>(const uchar* __restrict input, uchar* __restrict output0, uchar* __restrict output1,uchar* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split3Op<ushort>(const ushort* __restrict input, ushort* __restrict output0, ushort* __restrict output1, ushort* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split3Op<int>(const int* __restrict input, int* __restrict output0, int* __restrict output1, int* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split3Op<double>(const double* __restrict input, double* __restrict output0, double* __restrict output1, double* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);

extern template void Split4Op<uchar>(const uchar* __restrict input, uchar* __restrict output0, uchar* __restrict output1,uchar* __restrict output2,uchar* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split4Op<ushort>(const ushort* __restrict input, ushort* __restrict output0, ushort* __restrict output1, ushort* __restrict output2,ushort* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split4Op<int>(const int* __restrict input, int* __restrict output0, int* __restrict output1, int* __restrict output2,int* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
extern template void Split4Op<double>(const double* __restrict input, double* __restrict output0, double* __restrict output1, double* __restrict output2,double* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);

#endif//_SPLIT_