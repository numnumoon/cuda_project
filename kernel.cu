#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int N) {
    int i = threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

void launchAddKernel(int *d_a, int *d_b, int *d_c, int N) {
    add<<<1, N>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
}
