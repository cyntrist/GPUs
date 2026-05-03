#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int ndev = 0;
    cudaGetDeviceCount(& ndev) ;
    for (int d = 0; d < ndev; ++d) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, d);
    printf("Device %d: %s\n", d, p.name );
    printf("Compute capability: %d.%d\n", p.major, p.minor);
    printf("Número de SMs: %u\n", p.multiProcessorCount);
    printf("Warp size: %d\n", p.warpSize);
    printf("Memoria global total: %zu\n", p.totalGlobalMem);
    printf("SharedMemPerBlock: %zu\n", p.sharedMemPerBlock);
    printf("RegsPerBlock: %d\n", p.regsPerBlock);
    printf("MaxThreadsPerMultiProcesor: %d\n", p.maxThreadsPerBlock) ;
    printf("MaxThreadsPerMultiProcesor: %d\n", p.maxThreadsPerMultiProcessor);

    printf("Límites de grid x: %d\n", p.maxGridSize[0]) ;
    printf("Límites de grid y: %d\n", p.maxGridSize[1]) ;
    printf("Límites de grid z: %d\n", p.maxGridSize[2]) ;
    printf("Tamaño de la cache L2: %d\n", p.l2CacheSize) ;
    printf("Ancho de banda global en bits: %d\n", p.memoryBusWidth);
    }
}

// nvcc -O2 query.cu -o query