// transpose_v0_skeleton.cu
// Esqueleto inicial para la versión CUDA v0
// El alumno debe completar:
//  - Definición del kernel
//  - Reservas en GPU
//  - Transferencias H2D / D2H
//  - Lanzamiento y temporización del kernel

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

static struct timeval tv0;
double getMicroSeconds()
{
    double t;
    gettimeofday(&tv0, (struct timezone*)0);
    t = ((tv0.tv_usec) + (tv0.tv_sec) * 1000000.0);
    return t;
}

static inline void cudaCheck(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR (%s): %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

void init_seed()
{
    int seedi = 1;
    FILE *fd = fopen("/dev/urandom", "r");
    fread(&seedi, sizeof(int), 1, fd);
    fclose(fd);
    srand(seedi);
}

float **getmemory2D(int nx, int ny)
{
    float **buffer = (float**)malloc((size_t)nx * sizeof(float*));
    if (!buffer) return NULL;

    buffer[0] = (float*)malloc((size_t)nx * (size_t)ny * sizeof(float));
    if (!buffer[0]) {
        free(buffer);
        return NULL;
    }

    for (int i = 1; i < nx; i++)
        buffer[i] = buffer[i - 1] + ny;

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            buffer[i][j] = 0.0f;

    return buffer;
}

float *getmemory1D(int n)
{
    float *buffer = (float*)malloc((size_t)n * sizeof(float));
    if (!buffer) return NULL;

    for (int i = 0; i < n; i++)
        buffer[i] = 0.0f;

    return buffer;
}

void init2Drand(float **buffer, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            buffer[i][j] = 500.0f * ((float)rand() / (float)RAND_MAX) - 500.0f;
}

/* CPU reference */
void transpose1D_cpu(const float *in, float *out, int n)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            out[(size_t)j * n + i] = in[(size_t)i * n + j];
}

int check(const float *GPU, const float *CPU, int n)
{
    for (int i = 0; i < n; i++)
        if (GPU[i] != CPU[i]) return 1;
    return 0;
}

/* ================= CUDA v0 ================= */

#define NTHREADS1D 256

__global__ void transpose_device_v0(const float *in, float *out, int n)
{
    // TODO
}

/* ================= MAIN ================= */

int main(int argc, char **argv)
{
    int n;

    if (argc == 2) n = atoi(argv[1]);
    else {
        n = 8192;
        printf("./exec n (by default n=%i)\n", n);
    }

    init_seed();

    float **array2D       = getmemory2D(n, n);
    float **array2D_trans = getmemory2D(n, n);
    float *array1D_trans_GPU = getmemory1D(n * n);

    if (!array2D || !array2D_trans || !array1D_trans_GPU) return 1;

    float *array1D       = array2D[0];
    float *array1D_trans = array2D_trans[0];

    init2Drand(array2D, n);

    /* CPU reference */
    double bytes = 2.0 * (double)n * (double)n * (double)sizeof(float);

    double t0 = getMicroSeconds();
    transpose1D_cpu(array1D, array1D_trans, n);
    double t1 = getMicroSeconds();
    double secCPU = (t1 - t0) / 1e6;

    printf("Transpose CPU: %f MB/s\n",
           (bytes / secCPU) / 1024.0 / 1024.0);

    /* ===== CUDA PART (to be completed) ===== */

    float *d_in  = NULL;
    float *d_out = NULL;

    // TODO: cudaMalloc
    // TODO: cudaMemcpy H2D

    dim3 dimBlock(NTHREADS1D);
    dim3 dimGrid((n + NTHREADS1D - 1) / NTHREADS1D);

    double tKernel0 = 0.0, tKernel1 = 0.0;

    // TODO: launch kernel
    // TODO: synchronize
    // TODO: measure kernel time

    // TODO: cudaMemcpy D2H

    // TODO: compute and print kernel bandwidth

    if (check(array1D_trans_GPU, array1D_trans, n * n))
        printf("Transpose CPU-GPU differs!!\n");
    else
        printf("Check OK\n");

    // TODO: cudaFree

    free(array2D[0]);       free(array2D);
    free(array2D_trans[0]); free(array2D_trans);
    free(array1D_trans_GPU);

    return 0;
}

