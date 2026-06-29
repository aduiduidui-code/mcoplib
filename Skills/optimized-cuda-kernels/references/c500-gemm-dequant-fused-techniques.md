# C500 GEMM Dequant-Fused Optimization Techniques

Optimization techniques extracted from `op/vllm/quantization/gptq/hgemm_gptq.h` (launch_gemm_gptq_kernel).
These patterns are applicable to any fused dequant+GEMM kernel on Metax C500 GPU.

---

## Technique 1: C500 专用 MMA 指令替代标准 wmma/cublas

### 原理

C500 提供专有的 `mma_16x16x16` 指令，比标准 CUDA wmma API 更贴合 C500 PEU 硬件，
可直接使用 `__builtin_mxc_mma_16x16x16f16` 和 `__builtin_mxc_mma_16x16x16bf16` 内置函数。

### 代码模式

```cuda
// 在 Hgemm_common.cuh 中定义
#define mma_16x16x16f16(a_reg, b_reg, c_reg) \
  c_reg = __builtin_mxc_mma_16x16x16f16(a_reg, b_reg, c_reg)

#define mma_16x16x16bf16(a_reg, b_reg, c_reg) \
  c_reg = __builtin_mxc_mma_16x16x16bf16(a_reg, b_reg, c_reg)

// 使用示例：一次 mma 计算 16×16×16 的子块
template <class scalar_t>
__device__ __forceinline__ void mma_16x16x16(PackTypeInt2& a, PackTypeInt2& b,
                                             PackTypeInt4& c) {}

template <>
__device__ __forceinline__ void mma_16x16x16<half>(PackTypeInt2& a,
                                                   PackTypeInt2& b,
                                                   PackTypeInt4& c) {
  mma_16x16x16f16(a, b, c);  // C500 专用 FP16 MMA
}

template <>
__device__ __forceinline__ void mma_16x16x16<__maca_bfloat16>(PackTypeInt2& a,
                                                              PackTypeInt2& b,
                                                              PackTypeInt4& c) {
  mma_16x16x16bf16(a, b, c);  // C500 专用 BF16 MMA
}

// 在 matmul 中两次 mma 覆盖 SLICE_K=32
__device__ __forceinline__ void matmul(int mdx) {
  #pragma unroll
  for (int i = 0; i < N_ITERS; i++) {
    mma_16x16x16<scalar_t>(local_a[mdx][0],            // K[0:15]
                           *((PackTypeInt2*)dequant_b[i]),
                           *((PackTypeInt4*)output[mdx][i]));
    mma_16x16x16<scalar_t>(local_a[mdx][1],            // K[16:31]
                           *((PackTypeInt2*)dequant_b[i] + 1),
                           *((PackTypeInt4*)output[mdx][i]));
  }
}
```

### 适用场景

所有需要在 C500 上做 FP16/BF16 矩阵乘法的 kernel，尤其是：
- 量化权重 GEMM（GPTQ/AWQ/FP8）
- MoE 推理中的小批量 GEMM
- 注意力机制中的 QKV 投影

---

## Technique 2: C500 专用带谓词向量化加载指令

### 原理

C500 提供 `__builtin_mxc_ldg_b32/b64/b128_predicator` 内置函数，
支持带条件谓词的向量化加载，避免 warp-divergent 分支导致的加载低效。
比标准 CUDA 的条件加载 + `reinterpret_cast` 更高效。

### 代码模式

```cuda
// 在 Hgemm_common.cuh 中定义
#define ldg_b32_reg_noasync(dst, base, pred, ret0_en) \
  dst = __builtin_mxc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, \
                                         false, false, pred, 1, MACA_ICMP_EQ)[0];

#define ldg_b64_reg_noasync(dst, base, pred, ret0_en) \
  dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, \
                                         false, false, pred, 1, MACA_ICMP_EQ);

#define ldg_b128_reg_noasync(dst, base, pred, ret0_en) \
  dst = __builtin_mxc_ldg_b128_predicator(cast_b128(base), 0, ret0_en, true, \
                                           false, false, pred, 1, MACA_ICMP_EQ);

// 使用示例：带边界检查的 A 矩阵加载
__device__ __forceinline__ void ldg_a(int k_idx) {
  int t = tv.tid;
  #pragma unroll LOADING_A_LOOP
  for (int i = 0; i < LOADING_A_LOOP; i++) {
    int reading_m = t / (SLICE_K / FragACount);
    int reading_k = t % (SLICE_K / FragACount);
    FragA* gvm_addr = (FragA*)A_loading + reading_m * k / FragACount + reading_k;
    if constexpr (HAS_M_PRED && HAS_NK_PRED) {
      bool pred = reading_m < m && k_broad + reading_k * FragACount < k;
      ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred, true);  // 谓词控制
    }
    t += THREADS;
  }
}
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `dst` | 目标寄存器（b32/b64/b128） |
| `base` | 全局内存地址 |
| `pred` | 条件谓词，false 时写入 ret0 值 |
| `ret0_en` | true 时谓词为 false 写入 0，false 时不写 |

### 适用场景

- 需要边界检查的 tile 加载（M/N/K 不整除 TILE 大小时）
- 任何需要条件加载的 kernel
- 替代 `if (pred) dst = *ptr; else dst = 0;` 模式

---

## Technique 3: C500 专用 Bit-Cast 指令替代移位+转换

### 原理

4-bit 量化反量化需要从 uint32_t 中提取 4-bit 值并转为 FP32。
标准做法是移位+掩码+`int2float`，C500 提供专用的 `__builtin_mxc_b0_cast_to_f32` 等
内置函数，一条指令完成 byte 提取+类型转换。

### 代码模式

```cuda
// 定义 Bit-Cast 宏
#define CVT_B0TOF32(q, out) out = __builtin_mxc_b0_cast_to_f32(q);
#define CVT_B1TOF32(q, out) out = __builtin_mxc_b1_cast_to_f32(q);
#define CVT_B2TOF32(q, out) out = __builtin_mxc_b2_cast_to_f32(q);
#define CVT_B3TOF32(q, out) out = __builtin_mxc_b3_cast_to_f32(q);

// 4-bit 反量化使用示例
template <class scalar_t>
__device__ __forceinline__ void dequant_gptq_4bits(const PackType& p,
                                                   scalar_t (&out)[8],
                                                   const v2f& scale,
                                                   const v2f& scale_zero) {
  v2f a0;
  int p0 = p & 0x0f0f0f0f;          // 提取低 4-bit（每 byte 的低 nibble）
  CVT_B0TOF32(p0, a0.x);             // byte0 的低 4-bit → FP32
  CVT_B2TOF32(p0, a0.y);             // byte2 的低 4-bit → FP32
  a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);  // packed FMA
  out[0] = (scalar_t)a0.x;
  out[1] = (scalar_t)a0.y;

  CVT_B1TOF32(p0, a0.x);             // byte1 的低 4-bit → FP32
  CVT_B3TOF32(p0, a0.y);             // byte3 的低 4-bit → FP32
  a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
  out[4] = (scalar_t)a0.x;
  out[5] = (scalar_t)a0.y;

  p0 = (p >> 4) & 0x0f0f0f0f;       // 提取高 4-bit（每 byte 的高 nibble）
  CVT_B0TOF32(p0, a0.x);
  CVT_B2TOF32(p0, a0.y);
  a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
  out[2] = (scalar_t)a0.x;
  out[3] = (scalar_t)a0.y;

  CVT_B1TOF32(p0, a0.x);
  CVT_B3TOF32(p0, a0.y);
  a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
  out[6] = (scalar_t)a0.x;
  out[7] = (scalar_t)a0.y;
}
```

### Nibble 提取与 Byte 位置对应关系

```
uint32_t = [byte3][byte2][byte1][byte0]

p0 = p & 0x0f0f0f0f  → 提取每 byte 的低 4-bit:
  byte0 低 nibble → CVT_B0TOF32
  byte1 低 nibble → CVT_B1TOF32
  byte2 低 nibble → CVT_B2TOF32
  byte3 低 nibble → CVT_B3TOF32

p0 = (p >> 4) & 0x0f0f0f0f  → 提取每 byte 的高 4-bit:
  byte0 高 nibble → CVT_B0TOF32
  byte1 高 nibble → CVT_B1TOF32
  byte2 高 nibble → CVT_B2TOF32
  byte3 高 nibble → CVT_B3TOF32
```

### 适用场景

- GPTQ 4-bit 反量化
- AWQ 4-bit 反量化
- 任何需要从 packed 整数中提取子字节值并转为 FP32 的场景
- 8-bit 反量化也可使用（`CVT_B0/B1/B2/B3TOF32` 对 byte 级别同样有效）

---

## Technique 4: C500 专用 Packed FMA 指令

### 原理

`__builtin_mxc_pk_fma_f32` 是 C500 专用的 packed FMA 指令，
一次执行 2 个 FP32 FMA：`{a0×b0+c0, a1×b1+c1}`。
在反量化中，用一条指令同时完成 `val×scale + (-zero×scale)` 的两个通道。

### 代码模式

```cuda
typedef __NATIVE_VECTOR__(2, float) v2f;

// 反量化核心：value = quant_val × scale + (-zero × scale)
// packed FMA: {a0×b0+c0, a1×b1+c1}
v2f a0, scale, scale_zero;
scale = {s, s};           // 两个通道使用相同 scale
scale_zero = {z, z};      // 两个通道使用相同 -zero×scale

a0 = __builtin_mxc_pk_fma_f32(a0, scale, scale_zero);
// 等价于:
//   a0.x = a0.x × s + z   = quant_val × scale + (-zero × scale)
//   a0.y = a0.y × s + z
```

### 适用场景

- 量化权重的反量化计算
- 任何需要同时执行 2 路 FP32 FMA 的场景
- 与 `CVT_B0TOF32` 配合使用，形成高效的 4-bit 反量化流水线

---

## Technique 5: C500 专用 FP32→BF16 自定义转换

### 原理

C500 的标准 `__float2bfloat16` 可能不是最优的。hgemm_gptq 使用自定义的
`f32x2_cvt_bf16x2` 函数，利用 `__builtin_mxc_ubfe`（位域提取）和
`__builtin_mxc_byte_perm`（字节重排）实现更高效的 FP32→BF16 转换。

### 代码模式

```cuda
constexpr static uint32_t seil = 0x03020706u;

__device__ __forceinline__ void f32x2_cvt_bf16x2(uint32_t& dst, float src[2]) {
  uint32_t tmp[2];
  // 提取 FP32 的符号位，用于舍入
  tmp[0] = __builtin_mxc_ubfe(*(reinterpret_cast<uint32_t*>(src)), 16, 1);
  tmp[0] = tmp[0] + *reinterpret_cast<uint32_t*>(src);  // 舍入
  tmp[0] = (uint32_t)0x7fff + tmp[0];                   // 截断
  tmp[1] = __builtin_mxc_ubfe(*(reinterpret_cast<uint32_t*>(src + 1)), 16, 1);
  tmp[1] = tmp[1] + *(reinterpret_cast<uint32_t*>(src + 1));
  tmp[1] = (uint32_t)0x7fff + tmp[1];
  // 字节重排：将两个 BF16 打包到一个 uint32_t
  dst = __builtin_mxc_byte_perm(tmp[0], tmp[1], seil);
}
```

### 适用场景

- BF16 模式下的反量化输出转换
- 任何需要 FP32→BF16 高吞吐转换的场景
- 比 `__float2bfloat16_rn` 更适合 C500 微架构

---

## Technique 6: C500 专用 Barrier 指令替代 __syncthreads

### 原理

C500 提供专用的 barrier 指令，比标准 `__syncthreads()` 更轻量。
`barrier_bsm` 专门针对共享内存同步优化，延迟更低。

### 代码模式

```cuda
// 在 Hgemm_common.cuh 中定义
#define barrier __builtin_mxc_barrier_inst
#define barrier_all __builtin_mxc_barrier_ex(0)
#define barrier_bsm __builtin_mxc_barrier_ex(1)   // 共享内存 barrier
#define barrier_inst __builtin_mxc_barrier_ex(2)

// 使用示例
loading_manager.sts_a();    // A 写入共享内存
barrier_bsm;                // 等待所有线程完成写入
loading_manager.lds_a(0);   // 从共享内存读取 A
```

### 适用场景

- 所有需要共享内存同步的 C500 kernel
- 替代 `__syncthreads()` 以获得更低的同步延迟

---

## Technique 7: C500 专用 GVM→BSM 异步拷贝指令

### 原理

C500 提供 `__builtin_mxc_ldg_b32_bsm_predicator` 等内置函数，
可直接从全局内存异步拷贝到共享内存，减少寄存器中转开销。

### 代码模式

```cuda
// 全局内存 → 共享内存 异步拷贝
#define ldg_b32_bsm_async(saddr, base, pred, ret0_en) \
  __builtin_mxc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0, \
                                       ret0_en, true, false, true, pred, 1, MACA_ICMP_EQ);

#define ldg_b64_bsm_async(saddr, base, pred, ret0_en) \
  __builtin_mxc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0, \
                                       ret0_en, true, false, true, pred, 1, MACA_ICMP_EQ);

#define ldg_b128_bsm_async(saddr, base, pred, ret0_en) \
  __builtin_mxc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, \
                                        ret0_en, true, false, true, pred, 1, MACA_ICMP_EQ);

// 到达计数器
#define arrive_gvmcnt(num) __builtin_mxc_arrive(64 + num)
#define arrive_bsmcnt(num) __builtin_mxc_arrive(4096 + 128 * num)
#define arrive_gvm_bsmcnt(gvm, bsm) __builtin_mxc_arrive(4096 | (128 * bsm) | 64 | gvm)
```

### 适用场景

- 大数据量的全局内存→共享内存拷贝
- 需要计算与加载重叠的软件流水线
- 替代 `ldg → reg → sts` 两步拷贝

---

## Technique 8: 反量化与矩阵乘法融合（Fused Dequant+GEMM）

### 原理

量化推理的传统流程是先反量化权重到 FP16/BF16，再做 GEMM。
这会导致：(1) 额外的显存分配；(2) 额外的全局内存读写；(3) 反量化结果无法被 L2 缓存有效复用。
融合方案在寄存器中完成反量化，反量化结果直接送入 MMA 指令，零额外显存开销。

### 代码模式

```cuda
// 关键数据流：全局内存 → 寄存器反量化 → 寄存器 MMA → 累加器

// Step 1: 加载 packed 量化权重到寄存器
PackType local_b[N_ITERS];       // packed uint32_t
PackType local_b_cache[N_ITERS]; // 双缓冲
ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), addr, pred, true);

// Step 2: 交换双缓冲
swap_b_cache(i);  // local_b[i] = local_b_cache[i]

// Step 3: 在寄存器中反量化
scalar_t local_dequanted_b[N_ITERS][8];  // 反量化结果在寄存器中
v2f local_scales[N_ITERS];               // 预打包的 scale
v2f local_zeros[N_ITERS];                // 预打包的 -zero×scale
dequant_gptq_4bits(local_b[0], local_dequanted_b[0],
                    local_scales[0], local_zeros[0]);

// Step 4: 反量化结果直接送入 MMA，不经全局内存
mma_16x16x16<scalar_t>(local_a[mdx][0],
                        *((PackTypeInt2*)local_dequanted_b[i]),
                        *((PackTypeInt4*)output[mdx][i]));
```

### 收益

- 零额外显存分配（反量化结果全在寄存器中）
- 零额外全局内存读写（反量化→MMA 全在寄存器完成）
- 反量化延迟被软件流水线隐藏

### 适用场景

- GPTQ/AWQ/FP8 等量化推理
- 任何需要先解压/反量化再计算的场景

---

## Technique 9: 双缓冲软件流水线（Double Buffering）

### 原理

使用 `local_b` 和 `local_b_cache` 两组寄存器缓冲区，在反量化当前段 B 的同时
预取下一段 B 到 cache buffer，实现计算与加载重叠。

### 代码模式

```cuda
PackType local_b[N_ITERS];       // 当前正在使用的 buffer
PackType local_b_cache[N_ITERS]; // 正在加载的 cache buffer

// on_dequant 内部的双缓冲流水线
__device__ __forceinline__ void on_dequant_niters2(int k_idx) {
  // 使用当前 buffer 并开始反量化
  swap_b_cache(0);     // local_b[0] = local_b_cache[0]
  dequant(0);          // 反量化 local_b[0] → local_dequanted_b[0]

  // A 写入共享内存（与反量化重叠）
  sts_a();

  // 交换下一个 buffer
  swap_b_cache(1);     // local_b[1] = local_b_cache[1]

  // 预取下一段 B（与当前计算重叠）
  if constexpr (!KTAIL) {
    next_k_pre();
    ldg_b(k_idx + 1);   // 加载下一段 B → local_b_cache
    ldg_a(k_idx + 1);   // 加载下一段 A → temp_a
  }

  barrier_bsm;
  lds_a(0);            // 从共享内存读取 A
  dequant(1);          // 反量化 local_b[1]
}
```

### A 矩阵也使用双缓冲

```
全局内存 → temp_a (寄存器) → sts_a → smem_a → lds_a → local_a
           ↑ 当前段加载          ↑ 当前段写入    ↑ 下一段读取
                    ↑ 下段预加载到 temp_a
```

### 适用场景

- 所有需要 K 维度迭代的 GEMM kernel
- 任何需要隐藏全局内存延迟的计算密集型 kernel

---

## Technique 10: 编译期常量与模板特化消除运行时分支

### 原理

将 `BLOCKS_M`, `BLOCKS_N`, `BLOCKS_K`, `HAS_ZP`, `HAS_M_PRED`, `HAS_NK_PRED`,
`N_ITERS`, `k_iterations` 等参数作为编译期常量（模板参数或 constexpr），
编译器可以完全展开循环、消除死代码、移除运行时分支。

### 代码模式

```cuda
// 模板参数控制编译期特化
template <typename scalar_t, const vllm::ScalarTypeId w_type_id,
          int THREADS, int BLOCKS_M, int BLOCKS_N, int BLOCKS_K,
          bool HAS_ACT_ORDER, bool HAS_ZP, bool HAS_M_PRED, bool HAS_NK_PRED>
__global__ void hgemm_gptq(...) {
  // 编译期常量
  constexpr int VPT = 16 / sizeof(InputT);
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int N_ITERS = TILE_N / (WAVES_PER_BLOCK * SLOT);

  // 编译期分支消除
  if constexpr (HAS_M_PRED && HAS_NK_PRED) {
    bool pred = reading_m < m && reading_k < k;
    ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred, true);
  } else if constexpr (HAS_M_PRED) {
    bool pred = reading_m < m;
    ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred, true);
  } else {
    // 无谓词版本：编译器直接生成无条件加载
    ldg_b32_reg_noasync(temp_a[i], gvm_addr, true, true);
  }

  // 编译期循环展开
  #pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) { ... }  // k_iterations 是 constexpr
}
```

### 适用场景

- 所有 C500 kernel（模板特化是 C500 高性能编码的基本范式）
- 特别适合量化类型（kU4/kU4B8/kU8/kU8B128）的编译期分派

---

## Technique 11: 共享内存 Padding 避免 Bank Conflict

### 原理

C500 共享内存有 32 个 bank，每个 bank 4 字节宽。
当同一 warp 内多个线程访问同一 bank 的不同地址时，发生 bank conflict。
通过在 SLICE_K 维度添加 padding（PAD_SLICE_K = SLICE_K + 8），
使得相邻行的起始地址偏移不是 32 的倍数，消除 bank conflict。

### 代码模式

```cuda
constexpr static int SLICE_K = 32;
constexpr static int PAD_SLICE_K = 40;  // 32 + 8 padding

// A 矩阵在共享内存中的布局
// [BLOCKS_M × SLICE_M][PAD_SLICE_K] 而非 [BLOCKS_M × SLICE_M][SLICE_K]
__device__ __forceinline__ void sts_a() {
  FragA* to_bsm_a_ptr = (FragA*)smem_base;
  int t = tv.tid;
  #pragma unroll LOADING_A_LOOP
  for (int i = 0; i < LOADING_A_LOOP; i++) {
    int reading_m = t / (SLICE_K / FragACount);
    int reading_k = t % (SLICE_K / FragACount);
    // 关键：使用 PAD_SLICE_K 计算偏移，而非 SLICE_K
    int bsm_offset = reading_m * (PAD_SLICE_K / FragACount) + reading_k;
    *(to_bsm_a_ptr + bsm_offset) = temp_a[i];
    t += THREADS;
  }
}
```

### even_blocks_k 变体的 Double SLICE_K

```cuda
// even 变体中，一次加载 SLICE_K×2=64 个 K 元素
constexpr static int DOUBLE_SLICE_K = SLICE_K * 2;
constexpr static int DOUBLE_PAD_SLICE_K = SLICE_K * 2 + sizeof(PackTypeInt4) / sizeof(scalar_t);
// = 64 + 8 = 72
```

### 适用场景

- 所有使用共享内存存储 2D tile 的 kernel
- 特别是 GEMM 中 A/B 矩阵的共享内存布局

---

## Technique 12: K 优先 Tile 调度实现 Scales/Zeros 复用

### 原理

Tile 调度按 K 维度优先遍历（先沿 K 走完，再沿 N 移动）。
同一 N 列、不同 K 行的 tile 共享相同的 scales 和 zeros（只要在同一个 quant_group 内）。
K 优先遍历使得 scales/zeros 可以在多个 K 迭代中复用，减少全局内存加载次数。

### 代码模式

```cuda
struct TileManager {
  __device__ __forceinline__ void init(int m, int n, int k, int bidx, int iters) {
    int tile_idx = iters * bidx;
    int tiles_n = div_ceil(n, TILE_N);
    int tiles_k = div_ceil(k, TILE_K);
    int tile_col = tile_idx / tiles_k;  // N 维度索引 = 总索引 / K 方向 tile 数
    int tile_start_row = tile_idx - tile_col * tiles_k;  // K 维度索引
    tile_start_col = tile_col;
  }

  __device__ __forceinline__ void next_tile() {
    // K 优先：先增加 K 索引，K 到头后增加 N 索引
    tile_start_col =
        tile_start_row + 1 == tiles_k ? tile_start_col + 1 : tile_start_col;
    tile_start_row = tile_start_row + 1 == tiles_k ? 0 : tile_start_row + 1;
  }
};
```

### 写回时机

```cuda
// 只在 K 维度最后一个 tile 或最后一个 iter 时写回
__device__ __host__ __forceinline__ bool need_save_data() {
  if (global_pred && my_iters == 1) return true;
  if (global_pred && tile_start_row + 1 == tiles_k) return true;
  return false;
}
```

### 适用场景

- 量化 GEMM kernel（GPTQ/AWQ/FP8），scales/zeros 有复用机会
- 任何按 K 维度归约的 GEMM

---

## Technique 13: BF16 高精度路径——FP32 累加 + 后处理 Reduce

### 原理

C500 没有 BF16 的 `atomicAdd` 指令，不能直接在 BF16 输出上做原子累加。
也不能用 FP16 `atomicAdd`（会丢失 BF16 的指数范围）。
解决方案：在 FP32 临时缓冲区中累加，最后用一个轻量 reduce kernel 转换为 BF16。

### 代码模式

```cuda
// Kernel 中：FP32 atomicAdd 写入临时缓冲区
#ifdef BF16_HIGH_PRECISION
  if constexpr (std::is_same_v<scalar_t, __maca_bfloat16>) {
    atomicAdd(C_temp + offset, v);  // FP32 atomicAdd
  }
#endif

// Launcher 中：前置清零 + 后置 reduce
template <...>
bool launch_gemm_gptq_kernel(...) {
  // Step 1: 清零 FP32 临时缓冲区
  if constexpr (std::is_same_v<scalar_t, __maca_bfloat16>) {
    clean_zero<512, 4><<<clean_blocks, 512, 0, stream>>>((float*)C_temp, num_elem);
  }

  // Step 2: 启动主 kernel（FP32 atomicAdd 到 C_temp）
  hgemm_gptq<<<...>>>(..., C, C_temp, ...);

  // Step 3: FP32 → BF16 reduce
  if constexpr (std::is_same_v<scalar_t, __maca_bfloat16>) {
    all_reduce<512, 4, false><<<reduce_blocks, 512, 0, stream>>>(
        (float*)C_temp, (maca_bfloat16*)C, num_elem);
  }
}

// Reduce kernel: FP32 累加值 → BF16 输出
template <const int THREADS, const int PACK_NUM, const bool USE_C = false>
__global__ void all_reduce(float* in_data, maca_bfloat16* out_data, size_t num_elem) {
  float temp_in_fp32[PACK_NUM];
  maca_bfloat16 temp_out_bf16[PACK_NUM];
  ldg_b128_reg_noasync(*((b128VecType*)temp_in_fp32), ...);
  for (int i = 0; i < PACK_NUM; i++) {
    temp_out_bf16[i] = __float2bfloat16(temp_in_fp32[i]);
  }
  *((b64VecType*)(out_data + idx)) = *((b64VecType*)temp_out_bf16);
}
```

### 适用场景

- C500 上任何需要 BF16 输出但无法直接 atomicAdd 的 kernel
- MoE 推理中多个 block 写同一输出位置

---

## Technique 14: 按 PEU 数量调度 Block 避免 AtomicAdd

### 原理

C500 有 416 个 PEU（13 AP × 4 PEU × 8 DPC）。当 tiles 数量 ≥ PEU 数量时，
可以给每个 N 列分配独立的 block，从而不需要 atomicAdd（每个输出位置只被一个 block 写入）。
只有当 tiles 不足 PEU 时，才需要多个 block 写同一输出位置（使用 atomicAdd）。

### 代码模式

```cuda
template <...>
bool launch_gemm_gptq_kernel(...) {
  int tiles_n = div_ceil(n, TILE_N);
  int tiles_k = div_ceil(k, TILE_K);
  int total_tiles = tiles_n * tiles_k;

  // 判断：tiles 数量充足时，使用 no-atomic 路径
  if (tiles_n * std::max(chunks, 1) >= PEUS) {
    return launch_gemm_gptq_no_atomic_kernel<...>(...);
  }

  // 否则：需要 atomic 路径
  int blocks = PEUS;
  int iters = div_ceil(total_tiles, PEUS);
  // 调整 iters 使其满足 quant_group 对齐
  if (TILE_K < quant_group) {
    iters = div_ceil(iters, quant_group / TILE_K) * quant_group / TILE_K;
    blocks = div_ceil(total_tiles, iters);
  }
  // 尾部优化：减少空跑的 block
  while (iters * blocks - total_tiles >= iters) blocks -= 1;
}
```

### no-atomic 路径的 Grid 配置

```cuda
// 每个 N 列分配独立的 block，gridDim.z = tiles_n
// 无需 atomicAdd，直接写入 C[offset]
hgemm_gptq<<<dim3(1, chunks, tiles_n), THREADS, 0, stream>>>(...);
```

### 适用场景

- 所有 GEMM kernel 的 block 调度策略
- C500 上需要避免 atomicAdd 的场景

---

## Technique 15: 编译期 BLOCKS_K 奇偶分派不同 K 遍历策略

### 原理

当 `BLOCKS_K` 为偶数时，可以一次加载 `SLICE_K×2=64` 个 K 元素到共享内存，
减少 barrier 次数和 A 矩阵加载次数。奇数时只能逐 SLICE_K=32 加载。
两种策略在代码中分别实现为 `__hgemm_singular_blocks_k` 和 `__hgemm_even_blocks_k`。

### 代码模式

```cuda
// 奇数 BLOCKS_K：逐 SLICE_K 加载
namespace __hgemm_singular_blocks_k {
  constexpr static int FragACount = 2;     // 每次 A 加载 2 个 half
  using FragA = PackType;                   // uint32_t = 2×half
  constexpr static int PAD_SLICE_K = 40;    // SLICE_K(32) + 8 padding

  // next_k: 简单移动到下一个 SLICE_K
  __device__ void next_k() {
    bsm_a_ptr = smem_base + slot_tid * (PAD_SLICE_K / ...) + slot_idx;
  }
}

// 偶数 BLOCKS_K：一次加载两个 SLICE_K
namespace __hgemm_even_blocks_k {
  constexpr static int FragACount = 4;      // 每次 A 加载 4 个 half
  using FragA = PackTypeInt2;               // b64VecType = 4×half
  constexpr static int DOUBLE_SLICE_K = 64;
  constexpr static int DOUBLE_PAD_SLICE_K = 72;  // 64 + 8

  // 两个 K 子地址：k0 和 k1
  __device__ void next_k0() {
    bsm_a_ptr = smem_base + slot_tid * (DOUBLE_PAD_SLICE_K / ...) + slot_idx;
  }
  __device__ void next_k1() {
    bsm_a_ptr = smem_base + slot_tid * (DOUBLE_PAD_SLICE_K / ...) + slot_idx + WAVE_SLOTS;
  }

  // 主循环：每次处理 2 个 SLICE_K，减少 barrier
  while (max_iters > 0) {
    on_dequant<0, false>(k_idx);  // K 子段 0
    // ... matmul ...
    on_dequant<1, false>(k_idx);  // K 子段 1
    // ... matmul ...
    k_idx += 2;
  }
}
```

### Launcher 中自动选择

```cuda
if constexpr (BLOCKS_K % 2 == 1) {
  __hgemm_singular_blocks_k::hgemm_gptq<...><<<grid, block>>>(...);
} else {
  __hgemm_even_blocks_k::hgemm_gptq<...><<<grid, block>>>(...);
}
```

### 适用场景

- 所有 GEMM kernel 中 K 维度分块策略
- 偶数 SLICE_K 时减少同步开销

---

## Technique 16: C500 专用向量类型定义

### 原理

C500 的 MACA SDK 提供专有向量类型，用于高效的数据打包和向量化操作。
`b128VecType` (= `__NATIVE_VECTOR__(4, uint32_t)`) 对应 128-bit 寄存器，
可直接用于 `ldg_b128`、`sts`、MMA 等操作。

### 代码模式

```cuda
// Hgemm_common.cuh 中的类型定义
using b32VecType = uint32_t;                          // 32-bit
using b64VecType = __NATIVE_VECTOR__(2, uint32_t);    // 64-bit = 2×uint32
using b128VecType = __NATIVE_VECTOR__(4, uint32_t);   // 128-bit = 4×uint32
using b128VecType_i = __NATIVE_VECTOR__(4, int32_t);  // 128-bit signed
using Float4VecType = __NATIVE_VECTOR__(4, float);    // 4×FP32

// 在 hgemm_gptq.h 中使用
using PackTypeInt4 = b128VecType;   // 128-bit 打包，用于 A 矩阵和 C 矩阵
using PackTypeInt2 = b64VecType;    // 64-bit 打包，用于 MMA 的 A/B fragment
using PackType = uint32_t;          // 32-bit，用于量化权重 B 的单个 packed 值
using v2f = __NATIVE_VECTOR__(2, float);  // 2×FP32，用于 packed FMA

// 零初始化
float zeros[4] = {0.0, 0.0, 0.0, 0.0};
*((b128VecType*)(&data[idx])) = *((b128VecType*)zeros);  // 128-bit 零写入
```

### 适用场景

- 所有 C500 kernel 的向量化数据操作
- 替代标准 CUDA 的 `float4`、`uint4` 等类型

---

## Technique 17: 条件性 Scales/Zeros 预计算减少重复计算

### 原理

不同量化类型的 zero-point 处理方式不同。kU4B8 和 kU8B128 的零点是固定的（8 和 128），
无需从全局内存加载，可直接在 `pack_scales()` 中计算。kU4 需要从全局内存加载零点
并解压为 FP32。通过编译期特化避免不必要的全局内存访问。

### 代码模式

```cuda
__device__ __forceinline__ void pack_scales() {
  if constexpr (w_type_id == vllm::kU4B8.id()) {
    // 零点固定为 8，无需加载
    for (int i = 0; i < N_ITERS; i++) {
      float s = local_dequanted_b[0][i];
      float z = -8 * s;                    // 固定零点
      local_scales[i] = {s, s};
      local_zeros[i] = {z, z};
    }
  } else if constexpr (w_type_id == vllm::kU4.id()) {
    // 从共享内存读取解压后的零点
    for (int i = 0; i < N_ITERS; i++) {
      float s = __bfloat162float(local_dequanted_b[0][i]);
      float z = *(bsm_zeros_ptr + i);      // 从 smem 读取
      z = z * s;                            // -zero × scale
      local_scales[i] = {s, s};
      local_zeros[i] = {z, z};
    }
  } else if constexpr (w_type_id == vllm::kU8B128.id()) {
    // 零点固定为 128
    for (int i = 0; i < N_ITERS; i++) {
      float s = local_dequanted_b[0][i];
      float z = -128 * s;
      local_scales[i] = {s, s};
      local_zeros[i] = {z, z};
    }
  }
}
```

### 适用场景

- 多种量化类型共存的 kernel
- 任何需要编译期消除可选数据加载的场景

---

## Technique 18: Zero-Point 预解压到 FP32 共享内存

### 原理

GPTQ 4-bit 的零点也是 4-bit packed 的，需要在反量化时与量化值一起使用。
与其在每次反量化时都解压零点，不如在 tile 初始化时一次性解压到共享内存，
后续每次 K 迭代只需从共享内存读取已解压的 FP32 零点。

### 代码模式

```cuda
// 加载零点到寄存器
__device__ __forceinline__ void ldg_zp() {
  if constexpr (w_type_id == vllm::kU4.id()) {
    FragZeroLoading* gvm_addr = (FragZeroLoading*)zeros_loading + tv.tid;
    ldg_b32_reg_noasync(*((PackType*)&temp_zeros), gvm_addr, pred, true);
  }
}

// 解压并写入共享内存（每个 tid 解压 8 个零点）
__device__ __forceinline__ void sts_zeros() {
  if constexpr (w_type_id == vllm::kU4.id()) {
    if (pred) {
      float temp[8];
      decompress_zero_4bits(temp_zeros, temp);  // 4-bit packed → 8 个 FP32
      float* zeros_bsm = (float*)(smem_base + 0x3400) + tv.tid * 8;
      for (int i = 0; i < 8; i++) *(zeros_bsm + i) = temp[i];
    }
  }
}

// 预解压函数
__device__ __forceinline__ void decompress_zero_4bits(const PackType& zp, float (&out)[8]) {
  v2f a0;
  int p0 = zp & 0x0f0f0f0f;
  CVT_B0TOF32(p0, a0.x);  out[0] = -a0.x;
  CVT_B2TOF32(p0, a0.y);  out[1] = -a0.y;
  CVT_B1TOF32(p0, a0.x);  out[4] = -a0.x;
  CVT_B3TOF32(p0, a0.y);  out[5] = -a0.y;
  p0 = (zp >> 4) & 0x0f0f0f0f;
  CVT_B0TOF32(p0, a0.x);  out[2] = -a0.x;
  CVT_B2TOF32(p0, a0.y);  out[3] = -a0.y;
  CVT_B1TOF32(p0, a0.x);  out[6] = -a0.x;
  CVT_B3TOF32(p0, a0.y);  out[7] = -a0.y;
}
```

### 适用场景

- GPTQ 4-bit 量化 kernel
- 任何需要将 packed sub-byte 数据预解压到共享内存的场景

---

## Technique 19: M 维度预取实现 MMA 与 LDS 重叠

### 原理

在 M 维度内循环中，`lds_a(m_idx+1)` 和 `matmul(m_idx)` 交替执行，
实现"当前 M 块做 MMA"与"下一 M 块从共享内存读 A"的指令级并行。

### 代码模式

```cuda
int m_idx = 0;
if constexpr (BLOCKS_M > 1) {
  #pragma unroll BLOCKS_M - 1
  for (; m_idx < BLOCKS_M - 1; m_idx++) {
    loading_manager.lds_a(m_idx + 1);   // 预取下一 M 块的 A
    loading_manager.matmul(m_idx);       // 当前 M 块的 MMA
  }
}
// 最后一个 M 块：无预取
loading_manager.matmul(m_idx);
```

### 适用场景

- BLOCKS_M > 1 的 GEMM kernel
- 任何 M 维度有多块的 tile 计算

---

## Technique 20: 16KB 共享内存预算控制

### 原理

C500 每 SM 有 64KB 共享内存，但为了一个 SM 上运行多个 block（提高占用率），
每个 block 的共享内存使用需要控制。hgemm_gptq 精确控制在 16KB (0x4000)，
允许一个 SM 上运行 4 个 block，最大化吞吐。

### 代码模式

```cuda
__shared__ uint8_t smem_base[0x4000];  // 16KB

// 共享内存布局：
// 0x0000 ~ 0x1FFF: A 矩阵 (8KB = BLOCKS_M×SLICE_M × PAD_SLICE_K × sizeof(half))
// 0x2000 ~ 0x2FFF: Scales (4KB，实际使用 ~512B)
// 0x3000 ~ 0x33FF: Zeros (1KB，实际使用 ~256B)
// 0x3400 ~ 0x3FFF: 预留

// 验证:
// A: 1×16 × 40 × 2B = 1280B ≈ 1.25KB (singular, BLOCKS_M=1)
// A: 4×16 × 40 × 2B = 5120B ≈ 5KB (BLOCKS_M=4)
// Scales: 64 × 4B = 256B
// Zeros: 32 × 4B = 128B (8 × 32 / 8 × 4B)
```

### 适用场景

- 所有需要精确控制共享内存使用的 C500 kernel
- 特别是需要高占用率的 GEMM kernel

---

## Quick Reference: C500 专用内置函数速查表

| 内置函数 | 功能 | 替代的标准 CUDA |
|---------|------|----------------|
| `__builtin_mxc_mma_16x16x16f16` | FP16 MMA 16×16×16 | `wmma::mma_sync` |
| `__builtin_mxc_mma_16x16x16bf16` | BF16 MMA 16×16×16 | `wmma::mma_sync` |
| `__builtin_mxc_ldg_b32_predicator` | 带谓词 32-bit 加载 | `if(pred) dst=*ptr` |
| `__builtin_mxc_ldg_b64_predicator` | 带谓词 64-bit 加载 | `if(pred) dst=*ptr` |
| `__builtin_mxc_ldg_b128_predicator` | 带谓词 128-bit 加载 | `if(pred) dst=*ptr` |
| `__builtin_mxc_ldg_b32_bsm_predicator` | GVM→BSM 异步拷贝 | `cp.async` |
| `__builtin_mxc_b0_cast_to_f32` | byte0→FP32 | 移位+int2float |
| `__builtin_mxc_b1_cast_to_f32` | byte1→FP32 | 移位+int2float |
| `__builtin_mxc_b2_cast_to_f32` | byte2→FP32 | 移位+int2float |
| `__builtin_mxc_b3_cast_to_f32` | byte3→FP32 | 移位+int2float |
| `__builtin_mxc_pk_fma_f32` | packed 2×FP32 FMA | 2×`a*b+c` |
| `__builtin_mxc_ubfe` | 位域提取 | 移位+掩码 |
| `__builtin_mxc_byte_perm` | 字节重排 | 多次移位+或 |
| `__builtin_mxc_barrier_ex(1)` | 共享内存 barrier | `__syncthreads()` |
| `__builtin_mxc_arrive` | 到达计数器 | `__bar.sync` |
| `__NATIVE_VECTOR__(N, T)` | 专用向量类型 | `float4`/`uint4` |
