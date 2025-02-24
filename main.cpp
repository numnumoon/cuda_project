#include <iostream>
#include <cuda_runtime.h>

void launchAddKernel(int *d_a, int *d_b, int *d_c, int N);

int main() {
    const int N = 100;
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    size_t free_memory;
    size_t total_memory;
    cudaError_t cuda_memory_info = cudaMemGetInfo(&free_memory, &total_memory);

    if (cuda_memory_info != cudaSuccess){
        std::cerr << "CUDAメモリ情報の取得に失敗しました: " << cudaGetErrorString(cuda_memory_info) << std::endl;
        return -1;       
    }

     // メモリ量をMB単位で表示
    std::cout << "空きメモリ量: " << free_memory / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "総メモリ量: " << total_memory / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
 

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    launchAddKernel(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
