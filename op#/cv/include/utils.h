#ifndef UTILS
#define UTILS
#include <cuda_fp16.h>
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef char schar;
typedef unsigned int uint;

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ __forceinline__ void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template<>
__device__ __forceinline__ void copy<32>(const void* local, void* data)
{
    const float4* in = static_cast<const float4*>(local);
    float4* out = static_cast<float4*>(data);
    out[0] = in[0];
    out[1] = in[1];
}

template<>
__device__ __forceinline__ void copy<64>(const void* local, void* data)
{
    const float4* in = static_cast<const float4*>(local);
    float4* out = static_cast<float4*>(data);
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];
}

template<>
__device__ __forceinline__ void copy<128>(const void* local, void* data)
{
    const float4* in = static_cast<const float4*>(local);
    float4* out = static_cast<float4*>(data);
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];
    out[4] = in[4];
    out[5] = in[5];
    out[6] = in[6];
    out[7] = in[7];
}
#endif//UTILS