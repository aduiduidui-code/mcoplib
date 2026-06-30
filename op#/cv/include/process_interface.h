#ifndef PROCESS_INTERFACE
#define PROCESS_INTERFACE
#include <stdint.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

void fused_postprocess_yuv420(const float* __restrict inputData, uint8_t* __restrict yuv420Data, int width, int height, float scale, cudaStream_t stream);
void fused_postprocess_nv12(const float* __restrict intputData, uint8_t* __restrict nv12Data, int width, int height, float scale, cudaStream_t stream);
void fused_preprocess_yuv420(const uint8_t* __restrict yuv420Data, float* __restrict outputData, int width, int height, float scale, cudaStream_t stream);
void fused_preprocess_nv12(const uint8_t* __restrict nv12Data, float* __restrict outputData, int width, int height, float scale, cudaStream_t stream);
#endif//PROCESS_INTERFACE