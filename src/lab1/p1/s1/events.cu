#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

int n_iterations = 10;
bool write_to_csv = false;
const char *filename = "benchmarks.csv";

constexpr size_t sizes[]{
	1 << 10, // 1KiB
	1 << 12, // 4KiB
	1 << 14, // 16KiB
	1 << 16, // 64KiB
	1 << 20, // 1MiB
	1 << 24, // 16MiB
	1 << 26, // 64MiB
	1 << 28, // 256MiB
};

constexpr float sizes_10[]{
	// imagino en el enunciado que se refiere en base 2 pero he probado por si acaso con amos
	1e3,   // 1KB
	4e3,   // 4KB
	16e3,  // 16KB
	64e3,  // 64KB
	1e6,   // 1MB
	16e6,  // 16MB
	63e6,  // 64MB
	256e6, // 256MB
};

void event(float *a, float *b, size_t bytes,
		   cudaEvent_t start, cudaEvent_t stop,
		   cudaMemcpyKind kind, bool pinned, FILE *csv)
{
	const char *from;
	const char *to;
	const char *mode;
	const char *mem_type = pinned ? "pinned" : "pageable";
	float ms = 0.0f; // tiempo en ms entre ellos

	switch (kind)
	{
	case cudaMemcpyHostToDevice:
		from = "HOST";
		to = "DEVICE";
		mode = "H2D";
		break;
	case cudaMemcpyDeviceToHost:
		from = "DEVICE";
		to = "HOST";
		mode = "D2H";
		break;
	default: // DD
		from = "DEVICE";
		to = "DEVICE";
		mode = "D2D";
		mem_type = "gpu-only";
		break;
	}

	// warmup
	cudaMemcpy(b, a, bytes, kind);
	cudaDeviceSynchronize();
	cudaEventRecord(start); // record start
	cudaMemcpy(b, a, bytes, kind);
	cudaEventRecord(stop);					// record stop
	cudaEventSynchronize(stop);				// esperar hasta que acabe
	cudaEventElapsedTime(&ms, start, stop); // calculo de los ms

	// calculos de bandwidth
	double gb = (double)bytes / 1e9; // sistema internacional el GB
	double bw = gb / (ms / 1e3);
	// escrituras
	printf("%s to %s: %f ms (%.5f GB/s) \n", from, to, ms, bw);
	if (write_to_csv)
		fprintf(csv, "%zu,%s,%s,%.5f,%.5f\n", bytes, mem_type, mode, bw, ms);
}

void events(size_t bytes, cudaEvent_t start, cudaEvent_t stop, bool pinned, FILE *csv)
{
	// dos buffers en host con malloc
	float *h_a;
	float *h_b;
	if (pinned)
	{
		cudaMallocHost(&h_a, bytes);
		cudaMallocHost(&h_b, bytes);
	}
	else
	{
		h_a = (float *)malloc(bytes);
		h_b = (float *)malloc(bytes);
	}

	memset(h_a, 1, bytes);
	memset(h_b, 2, bytes);

	// dos buffers en device con cudamalloc
	float *d_a = nullptr,
		  *d_b = nullptr;

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);

	cudaMemset(d_a, 0, bytes);
	cudaMemset(d_b, 0, bytes);

	for (int i = 0; i < n_iterations; ++i)
	{
		event(h_a, d_a, bytes, start, stop, cudaMemcpyHostToDevice, pinned, csv); // el warmup se hace dentro
		event(d_b, h_b, bytes, start, stop, cudaMemcpyDeviceToHost, pinned, csv);
	}

	cudaFree(d_a);
	cudaFree(d_b);

	if (pinned)
	{
		cudaFreeHost(h_a);
		cudaFreeHost(h_b);
	}
	else
	{
		free(h_a);
		free(h_b);
	}
}

void events_device(size_t bytes, cudaEvent_t start, cudaEvent_t stop, FILE *csv)
{
	// dos buffers en device con cudamalloc
	float *d_a = nullptr,
		  *d_b = nullptr;

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);

	cudaMemset(d_a, 0, bytes);
	cudaMemset(d_b, 0, bytes);

	for (int i = 0; i < n_iterations; ++i)
	{
		event(d_a, d_b, bytes, start, stop, cudaMemcpyDeviceToDevice, false, csv);
		event(d_b, d_a, bytes, start, stop, cudaMemcpyDeviceToDevice, false, csv);
	}

	cudaFree(d_a);
	cudaFree(d_b);
}

void print_size(size_t bytes)
{
	const char *unit;
	size_t value;

	bool bi = true;
	if (bytes % 10 == 0)
		bi = false;

	if (bytes >= (bi ? (1 << 20) : 1e6))
	{
		value = bytes / (bi ? (1 << 20) : 1e6);
		unit = (bi ? "MiB" : "MB");
	}
	else if (bytes >= (bi ? (1 << 10) : 1e3))
	{
		value = bytes / (bi ? (1 << 10) : 1e3);
		unit = (bi ? "KiB" : "KB");
	}
	else
	{
		value = bytes;
		unit = "B";
	}
	printf("EVENTS SIZE %zd %s", value, unit);
}

int main(int argc, char *argv[])
{
	size_t M = sizeof(sizes) / sizeof(sizes[0]); // numero de tamaños para el benchmark

	// inicializacion de eventos de cuda
	cudaEvent_t start; // evento antes
	cudaEvent_t stop;  // evento despues
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// parseo de argumentos
	int opt;
	while ((opt = getopt(argc, argv, "hwo:n:")) != -1)
	{
		switch (opt)
		{
		case 'h':
			printf("Uso: ./events [-n n_iteraciones] [-w] / [-o archivo]\n");
			break;
		case 'w':
			write_to_csv = true;
			break;
		case 'o':
			write_to_csv = true;
			filename = optarg;
			break;
		case 'n':
			n_iterations = atoi(optarg);
			break;
		}
	}

	// inicializacion de csv
	FILE *csv = nullptr;
	if (write_to_csv == true)
	{
		write_to_csv = true;
		csv = fopen(filename, "w");
		fprintf(csv, "size_bytes,mem_mode,trans_kind,bandwidth_GBs,time_ms\n");
	}

	// bucle pageable
	for (int i = 0; i < M; ++i)
	{
		print_size(sizes[i]);
		printf(" (PAGEABLE)\n");
		events(sizes[i], start, stop, false, csv);
		printf("\n");
	}
	printf(">>>>>>>>>>><<<<<<<<<<<\n\n");
	// bucle pinned
	for (int i = 0; i < M; ++i)
	{
		print_size(sizes[i]);
		printf(" (PINNED)\n");
		events(sizes[i], start, stop, true, csv);
		printf("\n");
	}
	printf(">>>>>>>>>>><<<<<<<<<<<\n\n");
	// bucle d->d
	for (int i = 0; i < M; ++i)
	{
		print_size(sizes[i]);
		printf(" (D->D)\n");
		events_device(sizes[i], start, stop, csv);
		printf("\n");
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (write_to_csv)
		fclose(csv);
	return 0;
}

// nvcc -O2 events.cu -o events