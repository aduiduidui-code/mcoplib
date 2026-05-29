#include "utils.h"
#include "arithm.h"

template <typename T> __device__ __forceinline__ T saturate_cast(uchar v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(schar v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(ushort v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(short v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(uint v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(int v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(float v) { return T(v); }
template <typename T> __device__ __forceinline__ T saturate_cast(double v) { return T(v); }

template <> __device__ __forceinline__ uchar saturate_cast<uchar>(schar v)
{
    uint res = 0;
    //int vi = v;
    //asm("cvt.sat.u8.s8 %0, %1;" : "=r"(res) : "r"(vi));
    return res;
}
template <> __device__ __forceinline__ uchar saturate_cast<uchar>(short v)
{
    uint res = 0;
    //asm("cvt.sat.u8.s16 %0, %1;" : "=r"(res) : "h"(v));
    return res;
}
template <> __device__ __forceinline__ uchar saturate_cast<uchar>(ushort v)
{
    uint res = 0;
    //asm("cvt.sat.u8.u16 %0, %1;" : "=r"(res) : "h"(v));
    return res;
}
template <> __device__ __forceinline__ uchar saturate_cast<uchar>(int v)
{
    uint res = 0;
    //asm("cvt.sat.u8.s32 %0, %1;" : "=r"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ uchar saturate_cast<uchar>(uint v)
{
    uint res = 0;
    //asm("cvt.sat.u8.u32 %0, %1;" : "=r"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ uchar saturate_cast<uchar>(float v)
{
    uint res = 0;
    //asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(res) : "f"(v));
    return res;
}
template <> __device__ __forceinline__ uchar saturate_cast<uchar>(double v)
{
    uint res = 0;
    //asm("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(res) : "d"(v));
    return res;
}

template <> __device__ __forceinline__ schar saturate_cast<schar>(uchar v)
{
    uint res = 0;
    //uint vi = v;
    //asm("cvt.sat.s8.u8 %0, %1;" : "=r"(res) : "r"(vi));
    return res;
}
template <> __device__ __forceinline__ schar saturate_cast<schar>(short v)
{
    uint res = 0;
    //asm("cvt.sat.s8.s16 %0, %1;" : "=r"(res) : "h"(v));
    return res;
}
template <> __device__ __forceinline__ schar saturate_cast<schar>(ushort v)
{
    uint res = 0;
    //asm("cvt.sat.s8.u16 %0, %1;" : "=r"(res) : "h"(v));
    return res;
}
template <> __device__ __forceinline__ schar saturate_cast<schar>(int v)
{
    uint res = 0;
    //asm("cvt.sat.s8.s32 %0, %1;" : "=r"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ schar saturate_cast<schar>(uint v)
{
    uint res = 0;
    //asm("cvt.sat.s8.u32 %0, %1;" : "=r"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ schar saturate_cast<schar>(float v)
{
    uint res = 0;
    //asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(v));
    return res;
}
template <> __device__ __forceinline__ schar saturate_cast<schar>(double v)
{
    uint res = 0;
    //asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(res) : "d"(v));
    return res;
}

template <> __device__ __forceinline__ ushort saturate_cast<ushort>(schar v)
{
    ushort res = 0;
    //int vi = v;
    //asm("cvt.sat.u16.s8 %0, %1;" : "=h"(res) : "r"(vi));
    return res;
}
template <> __device__ __forceinline__ ushort saturate_cast<ushort>(short v)
{
    ushort res = 0;
    //asm("cvt.sat.u16.s16 %0, %1;" : "=h"(res) : "h"(v));
    return res;
}
template <> __device__ __forceinline__ ushort saturate_cast<ushort>(int v)
{
    ushort res = 0;
    //asm("cvt.sat.u16.s32 %0, %1;" : "=h"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ ushort saturate_cast<ushort>(uint v)
{
    ushort res = 0;
    //asm("cvt.sat.u16.u32 %0, %1;" : "=h"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ ushort saturate_cast<ushort>(float v)
{
    ushort res = 0;
    //asm("cvt.rni.sat.u16.f32 %0, %1;" : "=h"(res) : "f"(v));
    return res;
}
template <> __device__ __forceinline__ ushort saturate_cast<ushort>(double v)
{
    ushort res = 0;
    //asm("cvt.rni.sat.u16.f64 %0, %1;" : "=h"(res) : "d"(v));
    return res;
}

template <> __device__ __forceinline__ short saturate_cast<short>(ushort v)
{
    short res = 0;
    //asm("cvt.sat.s16.u16 %0, %1;" : "=h"(res) : "h"(v));
    return res;
}
template <> __device__ __forceinline__ short saturate_cast<short>(int v)
{
    short res = 0;
    //asm("cvt.sat.s16.s32 %0, %1;" : "=h"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ short saturate_cast<short>(uint v)
{
    short res = 0;
    //asm("cvt.sat.s16.u32 %0, %1;" : "=h"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ short saturate_cast<short>(float v)
{
    short res = 0;
    //asm("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(res) : "f"(v));
    return res;
}
template <> __device__ __forceinline__ short saturate_cast<short>(double v)
{
    short res = 0;
    //asm("cvt.rni.sat.s16.f64 %0, %1;" : "=h"(res) : "d"(v));
    return res;
}

template <> __device__ __forceinline__ int saturate_cast<int>(uint v)
{
    int res = 0;
    //asm("cvt.sat.s32.u32 %0, %1;" : "=r"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ int saturate_cast<int>(float v)
{
    return __float2int_rn(v);
}
template <> __device__ __forceinline__ int saturate_cast<int>(double v)
{
#if CV_CUDEV_ARCH >= 130
    return __double2int_rn(v);
#else
    return saturate_cast<int>((float) v);
#endif
}

template <> __device__ __forceinline__ uint saturate_cast<uint>(schar v)
{
    uint res = 0;
    //int vi = v;
    //asm("cvt.sat.u32.s8 %0, %1;" : "=r"(res) : "r"(vi));
    return res;
}
template <> __device__ __forceinline__ uint saturate_cast<uint>(short v)
{
    uint res = 0;
    //asm("cvt.sat.u32.s16 %0, %1;" : "=r"(res) : "h"(v));
    return res;
}
template <> __device__ __forceinline__ uint saturate_cast<uint>(int v)
{
    uint res = 0;
    //asm("cvt.sat.u32.s32 %0, %1;" : "=r"(res) : "r"(v));
    return res;
}
template <> __device__ __forceinline__ uint saturate_cast<uint>(float v)
{
    return __float2uint_rn(v);
}
template <> __device__ __forceinline__ uint saturate_cast<uint>(double v)
{
#if CV_CUDEV_ARCH >= 130
    return __double2uint_rn(v);
#else
    return saturate_cast<uint>((float) v);
#endif
}

template <typename _Arg, typename _Result> struct unary_function
{
    typedef _Arg    argument_type;
    typedef _Result result_type;
};

template <typename _Arg1, typename _Arg2, typename _Result> struct binary_function
{
    typedef _Arg1   first_argument_type;
    typedef _Arg2   second_argument_type;
    typedef _Result result_type;
};

template <typename T, typename D> struct AddOp1 : binary_function<T, T, D>
{
    __device__ __forceinline__ D operator ()(T a, T b) const
    {
        return saturate_cast<D>(a + b);
    }
};

template <typename T, typename D> struct SubOp1 : binary_function<T, T, D>
{
    __device__ __forceinline__ D operator ()(T a, T b) const
    {
        return saturate_cast<D>(a - b);
    }
};

template<typename T> struct less
{
    __device__ __forceinline__ unsigned char operator ()(T a, T b) const
    {
        return (unsigned char)(a < b);
    }
};

template<typename T> struct larger
{
    __device__ __forceinline__ unsigned char operator ()(T a, T b) const
    {
        return (unsigned char)(a > b);
    }
};

template<typename T> struct equal
{
    __device__ __forceinline__ unsigned char operator ()(T a, T b) const
    {
        return (unsigned char)(a == b);
    }
};

template<typename T> struct lt
{
    __device__ __forceinline__ unsigned char operator ()(T a, T b) const
    {
        return (unsigned char)(a <= b);
    }
};

template<typename T> struct gt
{
    __device__ __forceinline__ unsigned char operator ()(T a, T b) const
    {
        return (unsigned char)(a >= b);
    }
};

template<typename T> struct noteq
{
    __device__ __forceinline__ unsigned char operator ()(T a, T b) const
    {
        return (unsigned char)(a != b);
    }
};

template<typename T, typename D, class Op>
__global__ void arithm_no_mask_kernel_no_padding(const T* __restrict input0, const T* __restrict input1, D* __restrict dst, const Op &op, int64_t num_elems)
{
    constexpr int N = 16 / sizeof(T);
    int64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if(index >= num_elems) return;
    if(index >= num_elems - N) {
        for(int64_t i = index; i < num_elems; i++) {
            dst[i] = op(input0[i], input1[i]);
        }
    } else {
        float4 reg_input0 = *(float4*)(input0 + index);
        float4 reg_input1 = *(float4*)(input1 + index);
        T* ptr_reg_input0 = (T*)&reg_input0;
        T* ptr_reg_input1 = (T*)&reg_input1;
        D reg_dst[N];
        #pragma unroll N
        for(int i = 0; i < N; i++) {
            reg_dst[i] = op(ptr_reg_input0[i], ptr_reg_input1[i]);
        }
        D* ptr_dst = dst + index;
        copy<sizeof(D)*N>((void*)ptr_dst, (void*)reg_dst);
    }
}

template<typename T, typename D, class Op>
__global__ void arithm_mask_kernel_no_padding(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, const Op &op, int64_t num_elems) 
{
    constexpr int N = 16 / sizeof(T);
    int64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if(index >= num_elems) return;
    if(index >= num_elems - N) {
        for(int64_t i = index; i < num_elems; i++) {
            if(mask[i]) {
                dst[i] = op(input0[i], input1[i]);
            }
        }
    } else {
        float4 reg_input0 = *(float4*)(input0 + index);
        float4 reg_input1 = *(float4*)(input1 + index);
        T* ptr_reg_input0 = (T*)&reg_input0;
        T* ptr_reg_input1 = (T*)&reg_input1;
        D reg_dst[N];
        D* ptr_dst = dst + index;
        unsigned char reg_mask[N];
        copy<N>((void*)reg_mask, (void*)(mask + index));
        copy<sizeof(D)*N>((void*)reg_dst, (void*)ptr_dst);
        #pragma unroll N
        for(int i = 0; i < N; i++) {
            if(reg_mask[i]) {
                reg_dst[i] = op(ptr_reg_input0[i], ptr_reg_input1[i]);
            }
        }
        copy<sizeof(D)*N>((void*)ptr_dst, (void*)reg_dst);
    }
}

template<typename T, typename D>
void AddOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    if(mask) {
        arithm_mask_kernel_no_padding<T, D, AddOp1<T,D>><<<gridsize, num_threads, 0, stream>>>(input0, input1, mask, dst, AddOp1<T,D>(), num_elems);
    } else {
        arithm_no_mask_kernel_no_padding<T, D, AddOp1<T,D>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, AddOp1<T,D>(), num_elems);
    }
}

template void AddOp<uchar, uchar>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<uchar, char>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<uchar, ushort>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<uchar, short>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<uchar, int>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<uchar, float>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<uchar, double>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void AddOp<char, uchar>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<char, char>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<char, ushort>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<char, short>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<char, int>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<char, float>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<char, double>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void AddOp<ushort, ushort>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<ushort, short>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<ushort, int>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<ushort, float>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<ushort, double>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void AddOp<short, ushort>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<short, short>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<short, int>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<short, float>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<short, double>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void AddOp<int, int>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<int, float>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<int, double>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void AddOp<float, float>(const float * __restrict input0, const float* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<float, double>(const float * __restrict input0, const float* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void AddOp<double, double>(const double * __restrict input0, const double* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T, typename D>
void SubOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    if(mask) {
        arithm_mask_kernel_no_padding<T, D, SubOp1<T,D>><<<gridsize, num_threads, 0, stream>>>(input0, input1, mask, dst, SubOp1<T,D>(), num_elems);
    } else {
        arithm_no_mask_kernel_no_padding<T, D, SubOp1<T,D>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, SubOp1<T,D>(), num_elems);
    }
}

template void SubOp<uchar, uchar>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<uchar, char>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<uchar, ushort>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<uchar, short>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<uchar, int>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<uchar, float>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<uchar, double>(const uchar * __restrict input0, const uchar* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void SubOp<char, uchar>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<char, char>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, char* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<char, ushort>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<char, short>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<char, int>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<char, float>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<char, double>(const char * __restrict input0, const char* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void SubOp<ushort, ushort>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<ushort, short>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<ushort, int>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<ushort, float>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<ushort, double>(const ushort * __restrict input0, const ushort* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void SubOp<short, ushort>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<short, short>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, short* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<short, int>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<short, float>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<short, double>(const short * __restrict input0, const short* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void SubOp<int, int>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, int* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<int, float>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<int, double>(const int * __restrict input0, const int* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template void SubOp<float, float>(const float * __restrict input0, const float* __restrict input1, const uchar* __restrict mask, float* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<float, double>(const float * __restrict input0, const float* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void SubOp<double, double>(const double * __restrict input0, const double* __restrict input1, const uchar* __restrict mask, double* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void LessOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    arithm_no_mask_kernel_no_padding<T, unsigned char, less<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, less<T>(), num_elems);
}

template void LessOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void LargerOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    arithm_no_mask_kernel_no_padding<T, unsigned char, larger<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, larger<T>(), num_elems);
}

template void LargerOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void EqualOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    arithm_no_mask_kernel_no_padding<T, unsigned char, equal<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, equal<T>(), num_elems);
}

template void EqualOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void EqualOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void EqualOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void EqualOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void EqualOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void EqualOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void EqualOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void LessEqualOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    arithm_no_mask_kernel_no_padding<T, unsigned char, lt<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, lt<T>(), num_elems);
}

template void LessEqualOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessEqualOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessEqualOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessEqualOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessEqualOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessEqualOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LessEqualOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void LargerEqualOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    arithm_no_mask_kernel_no_padding<T, unsigned char, gt<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, gt<T>(), num_elems);
}

template void LargerEqualOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerEqualOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerEqualOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerEqualOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerEqualOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerEqualOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void LargerEqualOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template<typename T>
void NotEqualOp(const T* __restrict input0, const T* __restrict input1, unsigned char* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    arithm_no_mask_kernel_no_padding<T, unsigned char, noteq<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, noteq<T>(), num_elems);
}

template void NotEqualOp<uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void NotEqualOp<char>(const char* __restrict input0, const char* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void NotEqualOp<ushort>(const ushort* __restrict input0, const ushort* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void NotEqualOp<short>(const short* __restrict input0, const short* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void NotEqualOp<int>(const int* __restrict input0, const int* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void NotEqualOp<float>(const float* __restrict input0, const float* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void NotEqualOp<double>(const double* __restrict input0, const double* __restrict input1, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);

template <typename T, typename D> struct Mul : binary_function<T, T, D>
{
    __device__ __forceinline__ D operator ()(T a, T b) const
    {
        return saturate_cast<D>(a * b);
    }
};

template <typename T, typename S, typename D> struct MulScale : binary_function<T, T, D>
{
    S scale;

    __device__ __forceinline__ D operator ()(T a, T b) const
    {
        return saturate_cast<D>(scale * a * b);
    }
};

template<typename T, typename S, typename D>
void MulOp(const T* __restrict input0, const T* __restrict input1, D* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    if(scale == 1) {
        Mul<T, D> op;
        arithm_no_mask_kernel_no_padding<T, D, Mul<T,D>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, op, num_elems);
    } else {
        MulScale<T, S, D> op;
        arithm_no_mask_kernel_no_padding<T, D, MulScale<T,S,D>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, op, num_elems);
    }
}

template void MulOp<uchar, float, uchar>(const uchar* __restrict input0, const uchar* __restrict input1, uchar* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<uchar, float, schar>(const uchar* __restrict input0, const uchar* __restrict input1, schar* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<uchar, float, ushort>(const uchar* __restrict input0, const uchar* __restrict input1, ushort* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<uchar, float, short>(const uchar* __restrict input0, const uchar* __restrict input1, short* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<uchar, float, int>(const uchar* __restrict input0, const uchar* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<uchar, float, float>(const uchar* __restrict input0, const uchar* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<uchar, double, double>(const uchar* __restrict input0, const uchar* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

template void MulOp<schar, float, uchar>(const schar* __restrict input0, const schar* __restrict input1, uchar* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<schar, float, schar>(const schar* __restrict input0, const schar* __restrict input1, schar* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<schar, float, ushort>(const schar* __restrict input0, const schar* __restrict input1, ushort* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<schar, float, short>(const schar* __restrict input0, const schar* __restrict input1, short* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<schar, float, int>(const schar* __restrict input0, const schar* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<schar, float, float>(const schar* __restrict input0, const schar* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<schar, double, double>(const schar* __restrict input0, const schar* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

template void MulOp<ushort, float, ushort>(const ushort* __restrict input0, const ushort* __restrict input1, ushort* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<ushort, float, short>(const ushort* __restrict input0, const ushort* __restrict input1, short* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<ushort, float, int>(const ushort* __restrict input0, const ushort* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<ushort, float, float>(const ushort* __restrict input0, const ushort* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<ushort, double, double>(const ushort* __restrict input0, const ushort* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

template void MulOp<short, float, ushort>(const short* __restrict input0, const short* __restrict input1, ushort* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<short, float, short>(const short* __restrict input0, const short* __restrict input1, short* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<short, float, int>(const short* __restrict input0, const short* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<short, float, float>(const short* __restrict input0, const short* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<short, double, double>(const short* __restrict input0, const short* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

template void MulOp<int, float, int>(const int* __restrict input0, const int* __restrict input1, int* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<int, float, float>(const int* __restrict input0, const int* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<int, double, double>(const int* __restrict input0, const int* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

template void MulOp<float, float, float>(const float* __restrict input0, const float* __restrict input1, float* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<float, double, double>(const float* __restrict input0, const float* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);
template void MulOp<double, double, double>(const double* __restrict input0, const double* __restrict input1, double* __restrict dst, const double scale, int width, int height, int stride, cudaStream_t stream);

template <typename T> struct bit_and : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(T a,
                                             T b) const
    {
        return a & b;
    }
};

template <typename T> struct bit_or : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(T a,
                                             T b) const
    {
        return a | b;
    }
};

template <typename T> struct bit_xor : binary_function<T, T, T>
{
    __device__ __forceinline__ T operator ()(T a,
                                             T b) const
    {
        return a ^ b;
    }
};

template<typename T, typename D>
void BitAndOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    if(mask) {
        arithm_mask_kernel_no_padding<T, D, bit_and<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, mask, dst, bit_and<T>(), num_elems);
    } else {
        arithm_no_mask_kernel_no_padding<T, D, bit_and<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, bit_and<T>(), num_elems);
    }
}

template<typename T, typename D>
void BitOrOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    if(mask) {
        arithm_mask_kernel_no_padding<T, D, bit_or<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, mask, dst, bit_or<T>(), num_elems);
    } else {
        arithm_no_mask_kernel_no_padding<T, D, bit_or<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, bit_or<T>(), num_elems);
    }
}

template<typename T, typename D>
void BitXorOp(const T* __restrict input0, const T* __restrict input1, const unsigned char* __restrict mask, D* __restrict dst, int width, int height, int stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int64_t num_elems = (int64_t)stride * height;
    constexpr int N = 16 / sizeof(T);
    int64_t gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    if(mask) {
        arithm_mask_kernel_no_padding<T, D, bit_xor<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, mask, dst, bit_xor<T>(), num_elems);
    } else {
        arithm_no_mask_kernel_no_padding<T, D, bit_xor<T>><<<gridsize, num_threads, 0, stream>>>(input0, input1, dst, bit_xor<T>(), num_elems);
    }
}

template void BitAndOp<uchar, uchar>(const uchar* __restrict input0, const uchar* __restrict input1, const unsigned char* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void BitAndOp<ushort, ushort>(const ushort* __restrict input0, const ushort* __restrict input1, const unsigned char* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void BitAndOp<uint, uint>(const uint* __restrict input0, const uint* __restrict input1, const unsigned char* __restrict mask, uint* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void BitOrOp<uchar, uchar>(const uchar* __restrict input0, const uchar* __restrict input1, const unsigned char* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void BitOrOp<ushort, ushort>(const ushort* __restrict input0, const ushort* __restrict input1, const unsigned char* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void BitOrOp<uint, uint>(const uint* __restrict input0, const uint* __restrict input1, const unsigned char* __restrict mask, uint* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void BitXorOp<uchar, uchar>(const uchar* __restrict input0, const uchar* __restrict input1, const unsigned char* __restrict mask, uchar* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void BitXorOp<ushort, ushort>(const ushort* __restrict input0, const ushort* __restrict input1, const unsigned char* __restrict mask, ushort* __restrict dst, int width, int height, int stride, cudaStream_t stream);
template void BitXorOp<uint, uint>(const uint* __restrict input0, const uint* __restrict input1, const unsigned char* __restrict mask, uint* __restrict dst, int width, int height, int stride, cudaStream_t stream);