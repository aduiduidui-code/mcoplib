#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef __shfl_down_sync_16
#define __shfl_down_sync_16(mask, val, offset) __shfl_down_sync(mask, val, offset, 16)
#endif

// ============================================================================
// CUDA Kernel (Metax C500 Optimized)
// ============================================================================
template <typename scalar_t, int VEC_SIZE = 8>
__global__ void fused_gemma_rmsnorm_rope_neox_kernel(
    scalar_t* __restrict__ qkv,
    scalar_t const* __restrict__ weight,
    scalar_t const* __restrict__ cos_sin_cache,
    int64_t const* __restrict__ positions,
    float const eps,
    int64_t const qkv_stride_batch,
    int32_t const q_size,
    int32_t const num_heads,
    int32_t const num_kv_heads,
    int32_t const head_dim) 
{
    // 线程块与任务映射
    // Grid = (num_tokens * (num_heads + num_kv_heads))
    // Block = head_dim / VEC_SIZE (例如: 128 / 8 = 16 线程)
    
    int total_active_heads = num_heads + num_kv_heads;
    int task_id = blockIdx.x;
    
    int token_idx = task_id / total_active_heads;
    int head_idx = task_id % total_active_heads;
    
    int64_t position = positions[token_idx];

    // 路由分发：计算当前指针在 QKV 物理内存中的偏移
    scalar_t* head_ptr;
    if (head_idx < num_heads) {
        // 这是 Q head
        head_ptr = qkv + token_idx * qkv_stride_batch + head_idx * head_dim;
    } else {
        // 这是 K head，跳过完整的 Q_size
        int k_head_idx = head_idx - num_heads;
        head_ptr = qkv + token_idx * qkv_stride_batch + q_size + k_head_idx * head_dim;
    }

    int tid = threadIdx.x;
    int vec_idx = tid * VEC_SIZE;

    // Metax SREG 优化：缓存权重与输入到物理寄存器
    float reg_input[VEC_SIZE];
    float reg_weight[VEC_SIZE];
    float ss = 0.0f;

    // 128-bit 向量化加载数据与权重
    if (vec_idx < head_dim) {
        scalar_t local_x[VEC_SIZE];
        scalar_t local_w[VEC_SIZE];
        *(float4*)local_x = *(float4*)(head_ptr + vec_idx);
        *(float4*)local_w = *(float4*)(weight + vec_idx); // Gemma Q, K 共用同一个 RMSNorm weight

        #pragma unroll VEC_SIZE
        for (int i = 0; i < VEC_SIZE; i++) {
            float x = static_cast<float>(local_x[i]);
            float w = static_cast<float>(local_w[i]);
            reg_input[i] = x;
            reg_weight[i] = w;
            ss += x * x;
        }
    }

    // SIMD-16 Warp Reduction (符合 Metax 优化指南)
    #pragma unroll
    for (int i = 8; i > 0; i >>= 1) {
        ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
    }
    
    // Broadcast 结果
    __shared__ float s_rms;
    if (tid == 0) {
        // 使用原生 rsqrtf 等价于 Metax 的快速 SFU 计算
        s_rms = rsqrtf(ss / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();
    float rms = s_rms;

    // 共享内存作为 Neox 风格互相找 Partner 的交换站
    // 最大支持 head_dim=256
    __shared__ float smem_normed[256]; 

    if (vec_idx < head_dim) {
        #pragma unroll VEC_SIZE
        for (int i = 0; i < VEC_SIZE; i++) {
            // Gemma 特有公式: (1.0 + weight)
            float normed = reg_input[i] * rms * (1.0f + reg_weight[i]);
            smem_normed[vec_idx + i] = normed;
        }
    }
    __syncthreads();

    // 加载 RoPE 参数并计算旋转 (Neox Style)
    if (vec_idx < head_dim) {
        int half_d = head_dim / 2;
        
        // 针对 Neox 布局计算 Cache 读取基址
        // VEC_SIZE 保证了单个 float4 一定全落在左半边(cos)或右半边(sin)的对应区间
        int logical_idx = (vec_idx < half_d) ? vec_idx : (vec_idx - half_d);
        int cos_base = position * head_dim + logical_idx;
        int sin_base = cos_base + half_d;

        scalar_t local_cos[VEC_SIZE];
        scalar_t local_sin[VEC_SIZE];
        *(float4*)local_cos = *(float4*)(cos_sin_cache + cos_base);
        *(float4*)local_sin = *(float4*)(cos_sin_cache + sin_base);

        scalar_t out[VEC_SIZE];

        #pragma unroll VEC_SIZE
        for (int i = 0; i < VEC_SIZE; i++) {
            int current_idx = vec_idx + i;
            
            // 跨线程找 Neox Partner
            int partner_idx = (current_idx < half_d) ? (current_idx + half_d) : (current_idx - half_d);
            
            float self_val = smem_normed[current_idx];
            float partner_val = smem_normed[partner_idx];

            float cos_v = static_cast<float>(local_cos[i]);
            float sin_v = static_cast<float>(local_sin[i]);

            float rotated_val;
            // Neox 旋转公式
            // x'_i = x_i * cos - x_{i+d/2} * sin
            // x'_{i+d/2} = x_i * sin + x_{i+d/2} * cos
            if (current_idx < half_d) {
                rotated_val = self_val * cos_v - partner_val * sin_v;
            } else {
                rotated_val = partner_val * sin_v + self_val * cos_v;
            }
            
            out[i] = static_cast<scalar_t>(rotated_val);
        }
        
        // 128-bit 合并写回原 QKV 显存地址 (In-place)
        *(float4*)(head_ptr + vec_idx) = *(float4*)out;
    }
}

// ============================================================================
// Dispatch Function / PyBind Interface
// 这里的参数列表完美匹配 python 代码的需求
// ============================================================================
void gemma_fused_rmsnorm_rope(
    at::Tensor& qkv,
    at::Tensor const& weight,
    at::Tensor const& positions,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_dim,
    double eps,
    at::Tensor const& cos_sin_cache) 
{
    // 强制安全检查
    TORCH_CHECK(qkv.is_cuda(), "qkv must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be a CUDA tensor");
    
    TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
    TORCH_CHECK(head_dim % 8 == 0, "head_dim must be a multiple of 8 for 128-bit vectorization");
    TORCH_CHECK(head_dim <= 256, "head_dim must be <= 256 due to shared memory constraints");

    const c10::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int32_t num_tokens = qkv.size(0);
    int32_t num_heads = q_size / head_dim;
    int32_t num_kv_heads = kv_size / head_dim;
    int64_t qkv_stride_batch = qkv.stride(0);

    // 核心设计：每个 Task 负责 1个 Token 下的 1个 Head (涵盖 Q 和 K，过滤掉 V)
    int total_tasks = num_tokens * (num_heads + num_kv_heads);
    
    // Grid 和 Block 大小分配
    dim3 grid(total_tasks);
    // Block 大小: 例如 head_dim=128, vec_size=8, 则需要 16 线程。
    dim3 block(head_dim / 8); 

    if (qkv.dtype() == at::ScalarType::BFloat16) {
        fused_gemma_rmsnorm_rope_neox_kernel<__nv_bfloat16, 8><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(cos_sin_cache.data_ptr()),
            reinterpret_cast<int64_t const*>(positions.data_ptr()),
            static_cast<float>(eps),
            qkv_stride_batch, static_cast<int32_t>(q_size),
            num_heads, num_kv_heads, static_cast<int32_t>(head_dim)
        );
    } else if (qkv.dtype() == at::ScalarType::Half) {
        fused_gemma_rmsnorm_rope_neox_kernel<__half, 8><<<grid, block, 0, stream>>>(
            reinterpret_cast<__half*>(qkv.data_ptr()),
            reinterpret_cast<__half const*>(weight.data_ptr()),
            reinterpret_cast<__half const*>(cos_sin_cache.data_ptr()),
            reinterpret_cast<int64_t const*>(positions.data_ptr()),
            static_cast<float>(eps),
            qkv_stride_batch, static_cast<int32_t>(q_size),
            num_heads, num_kv_heads, static_cast<int32_t>(head_dim)
        );
    } else {
        TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
    }
}