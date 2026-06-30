// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# pragma once

#include "mctlass/mctlass.h"
#include "mctlass/gemm/device/gemm.h"
#include "mctlass/util/host_tensor.h"
#include "mctlass/util/reference/host/gemm.h"
#include "mctlass/util/reference/host/tensor_compare.h"
#include "mctlass/util/reference/host/tensor_copy.h"
#include "mctlass/util/reference/host/tensor_fill.h"
#include "mctlass/util/tensor_view_io.h"

#include "mctlass/gemm/gemm.h"
#include "mctlass/gemm/kernel/gemm_grouped.h"
#include "mctlass/gemm/kernel/default_gemm_grouped.h"
#include "mctlass/gemm/device/gemm_grouped.h"

#include "mctlass/gemm/kernel/maca_gemm_grouped.h"
#include "mctlass/gemm/kernel/maca_default_gemm_grouped.h"
#include "mctlass/gemm/device/maca_gemm_grouped.h"
#include "mctlass/gemm/device/maca_masked_grouped_gemm_adapter.h"

#include "mctlass/gemm/kernel/maca_gemm_masked_grouped.h"
#include "mctlass/gemm/kernel/maca_default_gemm_masked_grouped.h"
#include "mctlass/epilogue/thread/maca_linear_combination_scale.h"

using ColumnMajor = mctlass::layout::ColumnMajor;
using RowMajor = mctlass::layout::RowMajor;

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = maca_bfloat16;
using ElementAccumulator = int32_t;
using ElementCompute = float;
using ElementOutput = ElementC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = mctlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = mctlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock = mctlass::gemm::GemmShape<128, 128, 256>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = mctlass::gemm::GemmShape<64, 64, 256>;
// This code section describes the size of MMA op
using ShapeMMAOp = mctlass::gemm::GemmShape<16, 16, 16>;

mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scaleType =
        // mctlass::epilogue::thread::ScaleType::ScaleAvBvBias; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAsBvBias; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAsBsBias; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAvBsBias; // pass
        mctlass::epilogue::thread::ScaleType::ScaleAvBv; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAsBv; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAsBs; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAvBs; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleBiasKind::NoScaleAsBs;  // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAvBvBiasAzp; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAsBvBiasAzp; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAsBsBiasAzp; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAvBsBiasAzp; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAvBvBiasAzpPerTorken; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAsBvBiasAzpPerTorken; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAsBsBiasAzpPerTorken; // pass
        // mctlass::epilogue::thread::ScaleType::ScaleAvBsBiasAzpPerTorken; // pass

using GemmKernel = typename mctlass::gemm::kernel::MacaDefaultMaskedGroupedGemm<
  ElementA,
  RowMajor,
  mctlass::ComplexTransform::kNone,
  16,
  ElementB,
  ColumnMajor,
  mctlass::ComplexTransform::kNone,
  16,
  ElementOutput,
  RowMajor,
  ElementAccumulator,
  mctlass::arch::OpClassTensorOp,
  mctlass::arch::Sm80,
  ShapeMMAThreadBlock,
  ShapeMMAWarp,
  ShapeMMAOp,
  mctlass::epilogue::thread::MacaLinearCombinationScale<
      ElementAccumulator,
      4,
      ElementCompute,
      ElementOutput,
      scaleType>,
  mctlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
  4
  >::GemmKernel;

using mctlassGemm = mctlass::gemm::device::MacaMaskedGroupedGemmAdapter<GemmKernel>;
using Gemm = mctlassGemm;