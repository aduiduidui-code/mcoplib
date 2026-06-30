#include "utils.h"
#include "split.h"
template<typename T>
__global__ void split2images(const T* input, T* output0, T* output1, int width, int height, int input_stride, int output_stride) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elems = width * height;
    if(index >= num_elems) return;
    int h, w;
    h = index / width;
    w = index % width;
    T reg_input[2];
    const T* ptr_input = input + h * input_stride + w*2;
    copy<2*sizeof(T)>((void*)ptr_input, (void *)reg_input);
    int out_index = h * output_stride + w;
    T* ptr_output0 = output0 + out_index;
    T* ptr_output1 = output1 + out_index;
    ptr_output0[0] = reg_input[0];
    ptr_output1[0] = reg_input[1];
}

template<typename T>
__global__ void split3images(const T *input, T* output0, T * output1, T* output2, int width, int height, int input_stride, int output_stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elems = width * height;
    if(index >= num_elems) return;
    int h, w;
    h = index / width;
    w = index % width;
    T reg_input[3];
    const T* ptr_input = input + h * input_stride + w*3;
    #pragma unroll 3
    for(int i = 0; i < 3; i++) {
        reg_input[i] = ptr_input[i];
    }
    int out_index = h * output_stride + w;
    T* ptr_output0 = output0 + out_index;
    T* ptr_output1 = output1 + out_index;
    T* ptr_output2 = output2 + out_index;
    ptr_output0[0] = reg_input[0];
    ptr_output1[0] = reg_input[1];
    ptr_output2[0] = reg_input[2];
}

template<typename T>
__global__ void split4images(const T* input, T* output0, T* output1, T* output2, T* output3, int width, int height, int input_stride, int output_stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elems = width * height;
    if(index >= num_elems) return;
    int h, w;
    h = index / width;
    w = index % width;
    T reg_input[4];
    const T* ptr_input = input + h * input_stride + w*4;
    copy<4*sizeof(T)>((void*)ptr_input, (void *)reg_input);
    
    int out_index = h * output_stride + w;
    T* ptr_output0 = output0 + out_index;
    T* ptr_output1 = output1 + out_index;
    T* ptr_output2 = output2 + out_index;
    T* ptr_output3 = output3 + out_index;
    ptr_output0[0] = reg_input[0];
    ptr_output1[0] = reg_input[1];
    ptr_output2[0] = reg_input[2];
    ptr_output3[0] = reg_input[3];
}

template<typename T>
void Split2Op(const T* __restrict input, T* __restrict output0, T* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int num_elems = width * height;
    constexpr int N = 16 / sizeof(T);
    int gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    split2images<T><<<gridsize, num_threads, 0, stream>>>(input, output0, output1, width, height, input_stride, output_stride);
}

template<typename T>
void Split3Op(const T* __restrict input, T* __restrict output0, T* __restrict output1, T* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int num_elems = width * height;
    constexpr int N = 16 / sizeof(T);
    int gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    split3images<T><<<gridsize, num_threads, 0, stream>>>(input, output0, output1, output2, width, height, input_stride, output_stride);
}

template<typename T>
void Split4Op(const T* __restrict input, T* __restrict output0, T* __restrict output1, T* __restrict output2, T* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream)
{
    constexpr int num_threads = 512;
    int num_elems = width * height;
    constexpr int N = 16 / sizeof(T);
    int gridsize = (num_elems + num_threads * N - 1) / (num_threads * N);
    split4images<T><<<gridsize, num_threads, 0, stream>>>(input, output0, output1, output2, output3, width, height, input_stride, output_stride);
}

template void Split2Op<uchar>(const uchar* __restrict input, uchar* __restrict output0, uchar* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split2Op<ushort>(const ushort* __restrict input, ushort* __restrict output0, ushort* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split2Op<int>(const int* __restrict input, int* __restrict output0, int* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split2Op<double>(const double* __restrict input, double* __restrict output0, double* __restrict output1, int width, int height, int input_stride, int output_stride, cudaStream_t stream);

template void Split3Op<uchar>(const uchar* __restrict input, uchar* __restrict output0, uchar* __restrict output1,uchar* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split3Op<ushort>(const ushort* __restrict input, ushort* __restrict output0, ushort* __restrict output1, ushort* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split3Op<int>(const int* __restrict input, int* __restrict output0, int* __restrict output1, int* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split3Op<double>(const double* __restrict input, double* __restrict output0, double* __restrict output1, double* __restrict output2, int width, int height, int input_stride, int output_stride, cudaStream_t stream);

template void Split4Op<uchar>(const uchar* __restrict input, uchar* __restrict output0, uchar* __restrict output1,uchar* __restrict output2,uchar* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split4Op<ushort>(const ushort* __restrict input, ushort* __restrict output0, ushort* __restrict output1, ushort* __restrict output2,ushort* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split4Op<int>(const int* __restrict input, int* __restrict output0, int* __restrict output1, int* __restrict output2,int* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
template void Split4Op<double>(const double* __restrict input, double* __restrict output0, double* __restrict output1, double* __restrict output2,double* __restrict output3, int width, int height, int input_stride, int output_stride, cudaStream_t stream);
