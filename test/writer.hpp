#pragma once


template <typename T>
__global__ void writer(T *p, size_t n) {
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        p[i] = T(clock64());
    }
}