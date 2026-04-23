// busy.cu : kernel deliberadamente costoso para mantener la GPU ocupada
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
__global__ void busy_kernel ( float *x , int N )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = x [i];
        // Carga computacional : muchas operaciones transcendentes
        // ( suficiente para que se note en nvidia - smi )
        for (int k = 0; k < 20000; ++ k) {
            v = sinf(v) + cosf(v) ;
        }
        x[i] = v ;
    }
}

int main ()
    {
    const int N = 1 << 24;
    float * d = nullptr;
    cudaMalloc(( void **) &d, N * sizeof(float));
    dim3 block(256);
    dim3 grid(( N + block.x - 1) / block.x );
    busy_kernel<<<grid, block>>>(d, N);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}

// nvcc -O3 -lineinfo busy.cu -o busy