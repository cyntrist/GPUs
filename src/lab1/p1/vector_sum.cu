#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <unistd.h>

int n_iterations = 10;
bool write_to_csv = false;
const char *filename = "sum_benchmarks.csv";
bool pinned = false;

constexpr size_t sizes[]{
	1 << 10, // 1KiB
	1 << 12, // 4KiB
	1 << 14, // 16KiB
	1 << 16, // 64KiB
	1 << 20, // 1MiB
	1 << 24, // 16MiB
	1 << 26	 // 64MiB
};

__global__ void sum_kernel(const float *A,
						   const float *B,
						   float *C,
						   size_t N)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] + B[i];
}

__host__ void sum_cpu(const float *A,
					  const float *B,
					  float *C,
					  size_t N)
{
	for (size_t i = 0; i < N; i++)
		C[i] = A[i] + B[i];
}

void sum_benchmark(size_t N, FILE *csv)
{
	// size inicialization
	size_t bytes = N * sizeof(float);

	// host inizialization
	float *h_a;
	float *h_b;
	float *h_c;

	if (pinned)
	{
		cudaMallocHost(&h_a, bytes);
		cudaMallocHost(&h_b, bytes);
		cudaMallocHost(&h_c, bytes);
	}
	else
	{
		h_a = (float *)malloc(bytes);
		h_b = (float *)malloc(bytes);
		h_c = (float *)malloc(bytes);
	}

	for (size_t i = 0; i < N; i++)
	{
		h_a[i] = 1.0f;
		h_b[i] = 9.0f;
	}

	// gpu inizialization
	float *d_a;
	float *d_b;
	float *d_c;

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	int threads = 256;
	int blocks = (N + threads - 1) / threads;

	// time inicialization
	double cpu_ms = 0.0; // doble precision

	float gpu_ms = 0.0f;
	float kernel_ms = 0.0f;
	float total_ms = 0.0f;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// warmup
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
	sum_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (size_t j = 0; j < n_iterations; ++j)
	{
		cpu_ms = 0.0;
		gpu_ms = 0.0f;
		kernel_ms = 0.0f;
		total_ms = 0.0f;

		// cpu benchmark
		std::chrono::time_point t0 = std::chrono::high_resolution_clock::now();
		sum_cpu(h_a, h_b, h_c, N);
		std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();
		cpu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

		// gpu benchmark
		// gpu h->d + kernel + gpu d->h
		cudaEventRecord(start);
		cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
		sum_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
		cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&gpu_ms, start, stop);
		total_ms += gpu_ms;

		// kernel only
		cudaEventRecord(start);
		sum_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_ms, start, stop);
		kernel_ms += gpu_ms;

		printf("CPU: %.5f ms\n", cpu_ms);
		printf("GPU kernel: %.5f ms\n", kernel_ms);
		printf("GPU total : %.5f ms\n\n", total_ms);

		if (write_to_csv)
			fprintf(csv, "%zu,%.5f,%.5f,%.5f\n",
					N, cpu_ms, kernel_ms, total_ms);
	}

	if (pinned)
	{
		cudaFreeHost(h_a);
		cudaFreeHost(h_b);
		cudaFreeHost(h_c);
	}
	else
	{
		free(h_a);
		free(h_b);
		free(h_c);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main(int argc, char **argv)
{
	int M = sizeof(sizes) / sizeof(size_t);

	int opt;
	while ((opt = getopt(argc, argv, "hwo:n:")) != -1)
	{
		switch (opt)
		{
		case 'w':
			write_to_csv = true;
			break;
		case 'p':
			pinned = true;
			break;
		case 'o':
			write_to_csv = true;
			filename = optarg;
			break;
		case 'n':
			n_iterations = atoi(optarg);
			break;
		case 'h':
			printf("./vector_sum [-n n_iterations] [-p] [-w] / [-o filename]\n");
			return 0;
		}
	}

	FILE *csv = nullptr;
	if (write_to_csv)
	{
		csv = fopen(filename, "w");
		fprintf(csv, "N,cpu_ms,gpu_kernel_ms,gpu_total_ms\n");
	}

	for (size_t i = 0; i < M; ++i)
	{
		printf("+++\nN = %zu\n+++\n\n", sizes[i]);
		sum_benchmark(sizes[i], csv);
	}

	if (write_to_csv)
		fclose(csv);
}

// nvcc -O2 vector_sum.cu -o vector_sum
