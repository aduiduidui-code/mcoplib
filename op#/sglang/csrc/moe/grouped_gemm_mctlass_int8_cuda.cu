#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include "utils.h"
#include "mctlass/mctlass_ex.h"

#include "mctlass/half.h"
#include "mctlass/layout/matrix.h"
#include "mctlass/epilogue/thread/scale_type.h"

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = maca_bfloat16;
using ElementCompute = float;
using ElementOutput = ElementC;
using LayoutA = mctlass::layout::RowMajor;
using LayoutB = mctlass::layout::ColumnMajor;
using LayoutC = mctlass::layout::RowMajor;

using ArchTag = mctlass::arch::Sm80;

mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
        mctlass::epilogue::thread::ScaleType::ScaleAvBv;
        // mctlass::epilogue::thread::ScaleType::ScaleAsBv;
        // mctlass::epilogue::thread::ScaleType::ScaleAsBs;
        // mctlass::epilogue::thread::ScaleType::ScaleAvBs;

using mctlassContiguousGroupedGemmOp = mctlassContiguousGroupedGemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementCompute,
    ArchTag,
    scale_type
>;

int32_t get_block_size_m(int32_t batch_size, int32_t m, int32_t n, int32_t k)
{
    mctlassContiguousGroupedGemmOp mctlass_op;
    int blocksizeM = mctlass_op.get_blocksize_m(batch_size, m, n, k);
    return blocksizeM;
}

void grouped_gemm_mctlass_kernel_int8(
    torch::Tensor const& a, torch::Tensor const& b, torch::Tensor& c, int32_t batch_size, int32_t m, int32_t n, int32_t k,
    torch::Tensor const& seg_indptr, torch::Tensor const& weight_indices, torch::Tensor const& m_num_tiles_indptr, torch::Tensor const& scale_a, torch::Tensor const& scale_b)
{
    auto a_ptr = static_cast<ElementA*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB*>(b.data_ptr());
    auto c_ptr = static_cast<ElementC*>(c.data_ptr());
    auto seg_indptr_ptr = seg_indptr.data_ptr<int32_t>();
    auto weight_indices_ptr = weight_indices.data_ptr<int32_t>();
    auto m_num_tiles_indptr_ptr = m_num_tiles_indptr.data_ptr<int32_t>();
    auto scale_a_ptr = static_cast<ElementCompute const*>(scale_a.data_ptr());
    auto scale_b_ptr = static_cast<ElementCompute const*>(scale_b.data_ptr());

    int lda = k;
    int ldb = k;
    int ldc = n;
    mctlassContiguousGroupedGemmOp mctlass_op;
    typename mctlassContiguousGroupedGemmOp::Arguments arguments{
        mctlass::gemm::GemmUniversalMode::kGroupedContiguous,
        {scale_a_ptr, scale_b_ptr},
        {a_ptr, b_ptr, c_ptr, c_ptr, seg_indptr_ptr, weight_indices_ptr, m_num_tiles_indptr_ptr,
         batch_size, m, n, k, lda, ldb, ldc, ldc}
    };
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    mctlass_op(arguments, NULL, stream);
}

