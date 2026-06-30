#ifndef _ARITHM_
#define _ARITHM_
#include <stdint.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T, typename D>
void AddOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<uchar, uchar>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<uchar, char>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<uchar, ushort>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<uchar, short>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<uchar, int>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<uchar, float>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<uchar, double>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void AddOp<char, uchar>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<char, char>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<char, ushort>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<char, short>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<char, int>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<char, float>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<char, double>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void AddOp<ushort, ushort>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<ushort, short>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<ushort, int>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<ushort, float>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<ushort, double>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void AddOp<short, ushort>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<short, short>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<short, int>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<short, float>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<short, double>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void AddOp<int, int>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<int, float>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<int, double>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void AddOp<float, float>(const float * __restrict input0, const float* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<float, double>(const float * __restrict input0, const float* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void AddOp<double, double>(const double * __restrict input0, const double* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T, typename D>
void SubOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<uchar, uchar>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<uchar, char>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<uchar, ushort>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<uchar, short>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<uchar, int>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<uchar, float>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<uchar, double>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void SubOp<char, uchar>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<char, char>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<char, ushort>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<char, short>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<char, int>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<char, float>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<char, double>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void SubOp<ushort, ushort>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<ushort, short>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<ushort, int>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<ushort, float>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<ushort, double>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void SubOp<short, ushort>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<short, short>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<short, int>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<short, float>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<short, double>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void SubOp<int, int>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<int, float>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<int, double>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

extern template void SubOp<float, float>(const float * __restrict input0, const float* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<float, double>(const float * __restrict input0, const float* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void SubOp<double, double>(const double * __restrict input0, const double* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void LessOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void LargerOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void EqualOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void EqualOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void EqualOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void EqualOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void EqualOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void EqualOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void EqualOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void EqualOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void LessEqualOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessEqualOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessEqualOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessEqualOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessEqualOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessEqualOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessEqualOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LessEqualOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void LargerEqualOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerEqualOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerEqualOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerEqualOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerEqualOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerEqualOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerEqualOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void LargerEqualOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void NotEqualOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void NotEqualOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void NotEqualOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void NotEqualOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void NotEqualOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void NotEqualOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void NotEqualOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void NotEqualOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T, typename S, typename D>
void MulOp(const T* __restrict input0, const T* __restrict input1, D* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<uchar, float, uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<uchar, float, schar>(const uchar* __restrict input0, const uchar* __restrict input1, schar* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<uchar, float, ushort>(const uchar* __restrict input0, const uchar* __restrict input1, ushort* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<uchar, float, short>(const uchar* __restrict input0, const uchar* __restrict input1, short* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<uchar, float, int>(const uchar* __restrict input0, const uchar* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<uchar, float, float>(const uchar* __restrict input0, const uchar* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<uchar, double, double>(const uchar* __restrict input0, const uchar* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

extern template void MulOp<schar, float, uchar>(const schar* __restrict input0, const schar* __restrict input1, uchar* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<schar, float, schar>(const schar* __restrict input0, const schar* __restrict input1, schar* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<schar, float, ushort>(const schar* __restrict input0, const schar* __restrict input1, ushort* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<schar, float, short>(const schar* __restrict input0, const schar* __restrict input1, short* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<schar, float, int>(const schar* __restrict input0, const schar* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<schar, float, float>(const schar* __restrict input0, const schar* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<schar, double, double>(const schar* __restrict input0, const schar* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

extern template void MulOp<ushort, float, ushort>(const ushort* __restrict input0, const ushort* __restrict input1, ushort* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<ushort, float, short>(const ushort* __restrict input0, const ushort* __restrict input1, short* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<ushort, float, int>(const ushort* __restrict input0, const ushort* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<ushort, float, float>(const ushort* __restrict input0, const ushort* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<ushort, double, double>(const ushort* __restrict input0, const ushort* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

extern template void MulOp<short, float, ushort>(const short* __restrict input0, const short* __restrict input1, ushort* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<short, float, short>(const short* __restrict input0, const short* __restrict input1, short* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<short, float, int>(const short* __restrict input0, const short* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<short, float, float>(const short* __restrict input0, const short* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<short, double, double>(const short* __restrict input0, const short* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

extern template void MulOp<int, float, int>(const int* __restrict input0, const int* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<int, float, float>(const int* __restrict input0, const int* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<int, double, double>(const int* __restrict input0, const int* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

extern template void MulOp<float, float, float>(const float* __restrict input0, const float* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<float, double, double>(const float* __restrict input0, const float* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
extern template void MulOp<double, double, double>(const double* __restrict input0, const double* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

template<typename T, typename D>
void BitAndOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template<typename T, typename D>
void BitOrOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template<typename T, typename D>
void BitXorOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitAndOp<uchar, uchar>(const uchar* __restrict input0, const uchar* __restrict input1, const unsigned char* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitAndOp<ushort, ushort>(const ushort* __restrict input0, const ushort* __restrict input1, const unsigned char* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitAndOp<uint, uint>(const uint* __restrict input0, const uint* __restrict input1, const unsigned char* __restrict mask, uint* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitOrOp<uchar, uchar>(const uchar* __restrict input0, const uchar* __restrict input1, const unsigned char* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitOrOp<ushort, ushort>(const ushort* __restrict input0, const ushort* __restrict input1, const unsigned char* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitOrOp<uint, uint>(const uint* __restrict input0, const uint* __restrict input1, const unsigned char* __restrict mask, uint* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitXorOp<uchar, uchar>(const uchar* __restrict input0, const uchar* __restrict input1, const unsigned char* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitXorOp<ushort, ushort>(const ushort* __restrict input0, const ushort* __restrict input1, const unsigned char* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
extern template void BitXorOp<uint, uint>(const uint* __restrict input0, const uint* __restrict input1, const unsigned char* __restrict mask, uint* __restrict dst, int width, int height, int stride, cudaStream_t stream);

#endif//_ARITHM_