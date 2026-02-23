#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dotKernel(const float* a, const float* b, float* partial, int n);

extern "C" float dotProductCUDA(const float* a, const float* b, int n) {
    float* d_a;
    float* d_b;
    float* d_partial;
    float  result = 0.0f;

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_partial, blocks * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    dotKernel<<<blocks, threads>>>(d_a, d_b, d_partial, n);
    cudaDeviceSynchronize();

    float* h_partial = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocks; ++i)
        result += h_partial[i];

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial);
    free(h_partial);

    return result;
}

__global__ void dotKernel(const float* a, const float* b, float* partial, int n) {
    __shared__ float cache[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;
    float temp = 0.0f;

    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    for (int i = blockDim.x / 2; i != 0; i /= 2) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
    }

    if (cacheIdx == 0)
        partial[blockIdx.x] = cache[0];
}
