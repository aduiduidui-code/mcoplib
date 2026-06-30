// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once
#ifdef __HPCC_ARCH__
#include "hpcc_fp16.h"
#include "hpcc_bfloat16.h"
#elif defined(__MACA_ARCH__)
#include "maca_fp16.h"
#include "maca_bfloat16.h"
#endif

using b32VecType = uint32_t;
using b64VecType = __NATIVE_VECTOR__(2, uint32_t);
using b128VecType = __NATIVE_VECTOR__(4, uint32_t);
using b128VecType_i = __NATIVE_VECTOR__(4, int32_t);
using Float4VecType = __NATIVE_VECTOR__(4, float);


using PackTypeInt4 = b128VecType;
using PackTypeInt2 = b64VecType;
using PackType = uint32_t;

typedef enum
{
    OP_N = 0,
    OP_T = 1
} Operation_t;

#define cast_half(ptr) reinterpret_cast<__half *>(ptr)
#define cast_b16(ptr) reinterpret_cast<uint16_t *>(ptr)
#define cast_b32(ptr) reinterpret_cast<uint32_t *>(ptr)
#define cast_b64(ptr) reinterpret_cast<b64VecType *>(ptr)
#define cast_u64(ptr) reinterpret_cast<uint64_t *>(ptr)
#define cast_b128(ptr) reinterpret_cast<b128VecType *>(ptr)
#define cast_b128_i(ptr) reinterpret_cast<b128VecType_i *>(ptr)

#ifdef __HPCC_ARCH__
#define ldg_b32_reg_noasync(dst, base, pred, ret0_en)                                                \
    dst = __builtin_htc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, false, false, pred, 1, \
                                           HPCC_ICMP_EQ)[0];
#define ldg_b64_reg_noasync(dst, base, pred, ret0_en)                                                \
    dst = __builtin_htc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, false, pred, 1, \
                                           HPCC_ICMP_EQ);
#define ldg_b128_reg_noasync(dst, base, pred, ret0_en)                                                \
    dst = __builtin_htc_ldg_b128_predicator(cast_b128(base), 0, ret0_en, true, false, false, pred, 1, \
                                           HPCC_ICMP_EQ);

#define ldg_b32_reg_async(dst, base, pred, ret0_en)                                                \
    dst = __builtin_htc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, false, true, pred, 1, \
                                           HPCC_ICMP_EQ)[0];
#define ldg_b64_reg_async(dst, base, pred, ret0_en)                                                \
    dst = __builtin_htc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, true, pred, 1, \
                                           HPCC_ICMP_EQ);
#define ldg_b128_reg_async(dst, base, pred, ret0_en)                                                 \
    dst = __builtin_htc_ldg_b128_predicator(cast_b128(base), 0, ret0_en, true, false, true, pred, 1, \
                                            HPCC_ICMP_EQ);
#define ldg_b64_v4h_reg_async(dst, base, pred, ret0_en)                                                \
    dst = __builtin_htc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, true, pred, 1, \
                                           HPCC_ICMP_EQ);

#define ldg_b32_bsm_noasync(saddr, base, pred, ret0_en)                                                    \
    __builtin_htc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0, ret0_en, true, false, false, \
                                         pred, 1, HPCC_ICMP_EQ);
#define ldg_b64_bsm_noasync(saddr, base, pred, ret0_en)                                                    \
    __builtin_htc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0, ret0_en, true, false, false, \
                                         pred, 1, HPCC_ICMP_EQ);
#define ldg_b128_bsm_noasync(saddr, base, pred, ret0_en)                                                \
    __builtin_htc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, ret0_en, true, false, \
                                          false, pred, 1, HPCC_ICMP_EQ);

#define ldg_b32_bsm_async(saddr, base, pred, ret0_en)                                                    \
    __builtin_htc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0, ret0_en, true, false, true, \
                                         pred, 1, HPCC_ICMP_EQ);
#define ldg_b64_bsm_async(saddr, base, pred, ret0_en)                                                    \
    __builtin_htc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0, ret0_en, true, false, true, \
                                         pred, 1, HPCC_ICMP_EQ);
#define ldg_b128_bsm_async(saddr, base, pred, ret0_en)                                                \
    __builtin_htc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, ret0_en, true, false, \
                                          true, pred, 1, HPCC_ICMP_EQ);

#define stg_b32_async(data, base, pred)                                                   \
    __builtin_htc_stg_b32_predicator(cast_b32(base), 0, data, true, false, true, pred, 1, \
                                     HPCC_ICMP_EQ);
#define stg_b64_async(data, base, pred)                                                   \
    __builtin_htc_stg_b64_predicator(cast_b64(base), 0, data, true, false, true, pred, 1, \
                                     HPCC_ICMP_EQ);
#define stg_b128_async(data, base, pred)                                                    \
    __builtin_htc_stg_b128_predicator(cast_b128(base), 0, data, true, false, true, pred, 1, \
                                      HPCC_ICMP_EQ);

#elif defined(__MACA_ARCH__)

#define ldg_b32_reg_noasync(dst, base, pred, ret0_en)                                                \
    dst = __builtin_mxc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, false, false, pred, 1, \
                                           MACA_ICMP_EQ)[0];
#define ldg_b64_reg_noasync(dst, base, pred, ret0_en)                                                \
    dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, false, pred, 1, \
                                           MACA_ICMP_EQ);
#define ldg_b128_reg_noasync(dst, base, pred, ret0_en)                                                \
    dst = __builtin_mxc_ldg_b128_predicator(cast_b128(base), 0, ret0_en, true, false, false, pred, 1, \
                                           MACA_ICMP_EQ);

#define ldg_b32_reg_async(dst, base, pred, ret0_en)                                                \
    dst = __builtin_mxc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, false, true, pred, 1, \
                                           MACA_ICMP_EQ)[0];
#define ldg_b64_reg_async(dst, base, pred, ret0_en)                                                \
    dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, true, pred, 1, \
                                           MACA_ICMP_EQ);
#define ldg_b128_reg_async(dst, base, pred, ret0_en)                                                 \
    dst = __builtin_mxc_ldg_b128_predicator(cast_b128(base), 0, ret0_en, true, false, true, pred, 1, \
                                            MACA_ICMP_EQ);
#define ldg_b64_v4h_reg_async(dst, base, pred, ret0_en)                                                \
    dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, true, pred, 1, \
                                           MACA_ICMP_EQ);

#define ldg_b32_bsm_noasync(saddr, base, pred, ret0_en)                                                    \
    __builtin_mxc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0, ret0_en, true, false, false, \
                                         pred, 1, MACA_ICMP_EQ);
#define ldg_b64_bsm_noasync(saddr, base, pred, ret0_en)                                                    \
    __builtin_mxc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0, ret0_en, true, false, false, \
                                         pred, 1, MACA_ICMP_EQ);
#define ldg_b128_bsm_noasync(saddr, base, pred, ret0_en)                                                \
    __builtin_mxc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, ret0_en, true, false, \
                                          false, pred, 1, MACA_ICMP_EQ);

#define ldg_b32_bsm_async(saddr, base, pred, ret0_en)                                                    \
    __builtin_mxc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0, ret0_en, true, false, true, \
                                         pred, 1, MACA_ICMP_EQ);
#define ldg_b64_bsm_async(saddr, base, pred, ret0_en)                                                    \
    __builtin_mxc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0, ret0_en, true, false, true, \
                                         pred, 1, MACA_ICMP_EQ);
#define ldg_b128_bsm_async(saddr, base, pred, ret0_en)                                                \
    __builtin_mxc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, ret0_en, true, false, \
                                          true, pred, 1, MACA_ICMP_EQ);

#define stg_b32_async(data, base, pred)                                                   \
    __builtin_mxc_stg_b32_predicator(cast_b32(base), 0, data, true, false, true, pred, 1, \
                                     MACA_ICMP_EQ);
#define stg_b64_async(data, base, pred)                                                   \
    __builtin_mxc_stg_b64_predicator(cast_b64(base), 0, data, true, false, true, pred, 1, \
                                     MACA_ICMP_EQ);
#define stg_b128_async(data, base, pred)                                                    \
    __builtin_mxc_stg_b128_predicator(cast_b128(base), 0, data, true, false, true, pred, 1, \
                                      MACA_ICMP_EQ);

#endif

#ifdef __HPCC_ARCH__
#define mma_16x16x16f16(a_reg, b_reg, c_reg) \
    c_reg = __builtin_htc_mma_16x16x16f16(a_reg, b_reg, c_reg)

#define mma_16x16x16bf16(a_reg, b_reg, c_reg) \
    c_reg = __builtin_htc_mma_16x16x16bf16(a_reg, b_reg, c_reg)
#elif defined(__MACA_ARCH__)
#define mma_16x16x16f16(a_reg, b_reg, c_reg) \
    c_reg = __builtin_mxc_mma_16x16x16f16(a_reg, b_reg, c_reg)

#define mma_16x16x16bf16(a_reg, b_reg, c_reg) \
    c_reg = __builtin_mxc_mma_16x16x16bf16(a_reg, b_reg, c_reg)
#endif

template<class scalar_t>
__device__ __forceinline__ void mma_16x16x16(PackTypeInt2& a, PackTypeInt2& b, PackTypeInt4& c) {
}

template<>
__device__ __forceinline__ void mma_16x16x16<half>(PackTypeInt2& a, PackTypeInt2& b, PackTypeInt4& c) {
    mma_16x16x16f16(a, b, c);
}

#ifdef __HPCC_ARCH__
template<>
__device__ __forceinline__ void mma_16x16x16<__hpcc_bfloat16>(PackTypeInt2& a, PackTypeInt2& b, PackTypeInt4& c) {
    mma_16x16x16bf16(a, b, c);
}
#elif defined(__MACA_ARCH__)
template<>
__device__ __forceinline__ void mma_16x16x16<__maca_bfloat16>(PackTypeInt2& a, PackTypeInt2& b, PackTypeInt4& c) {
    mma_16x16x16bf16(a, b, c);
}
#endif