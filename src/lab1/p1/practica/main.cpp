#include "png_io.h"
#include "routinesCPU.h"
#include "routinesGPU.h"
#include "routinesGPUoptimized.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Time */
#include <sys/resource.h>
#include <sys/time.h>

#define WARMUP_ITERS 3
static struct timeval tv0;

/* tiempo en microsegundos */
double get_time()
{
	double t;
	gettimeofday(&tv0, (struct timezone *)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec) * 1000000);

	return (t);
}

static void append_benchmark_csv(const char *mode, int width, int height,
								 double total_ms, double kernel_ms, double transfer_ms)
{
	FILE *csv = fopen("benchmark_results.csv", "a+");
	long csv_size;

	if (csv == NULL)
	{
		perror("benchmark_results.csv");
		exit(-1);
	}

	fseek(csv, 0, SEEK_END);
	csv_size = ftell(csv);

	if (csv_size == 0)
		fprintf(csv, "mode,resolution,totaltime,kerneltime,transfertime\n");

	if (kernel_ms > 0.0 || transfer_ms > 0.0)
	{
		fprintf(csv, "%s,%dx%d,%.5f,%.5f,%.5f\n",
				mode, width, height, total_ms, kernel_ms, transfer_ms);
	}
	else
	{
		fprintf(csv, "%s,%dx%d,%.5f,,\n",
				mode, width, height, total_ms);
	}

	fclose(csv);
}

int main(int argc, char **argv)
{
	uint8_t *imtmp, *im;
	int width, height;
	int iterations = 1;

	/* hough prueba angulos entre 0 y 179 grados */
	float sin_table[180], cos_table[180];
	int nlines = 0;
	int x1[10], x2[10], y1[10], y2[10];
	int l;
	double t0, t1;
	gpu_benchmark gpu_timing = {0.0f, 0.0f, 0.0f};

	/* Only accept a concrete number of arguments */
	if (argc != 3 && argc != 4)
	{
		printf("./exec image.png [c/g/o] [iterations]\n");
		exit(-1);
	}

	if (argc == 4)
	{
		iterations = atoi(argv[3]);
		if (iterations < 1)
		{
			printf("Iterations must be >= 1.\n");
			exit(-1);
		}
	}

	/* Read images */
	/*
	 * se carga la imagen en color para poder dibujar el resultado al final
	 * despues se crea una copia en escala de grises para el procesamiento
	 */
	imtmp = read_png_fileRGB(argv[1], &width, &height);
	im = image_RGB2BW(imtmp, height, width);

	/* estas tablas evitan recalcular seno y coseno en cada voto de hough */
	init_cos_sin_table(sin_table, cos_table, 180);

	// Create temporal buffers
	/* buffers temporales que usa la version cpu en cada fase del pipeline */
	uint8_t *imEdge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float *NR = (float *)malloc(sizeof(float) * width * height);
	float *G = (float *)malloc(sizeof(float) * width * height);
	float *phi = (float *)malloc(sizeof(float) * width * height);
	float *Gx = (float *)malloc(sizeof(float) * width * height);
	float *Gy = (float *)malloc(sizeof(float) * width * height);
	uint8_t *pedge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	// Create the accumulators
	/*
	 * acumulador de hough
	 * cada celda guarda cuantos pixeles de borde apoyan una recta concreta
	 */
	float hough_h = ((sqrt(2.0) * (float)(height > width ? height : width)) / 2.0);
	int accu_height = hough_h * 2.0; // -rho -> +rho
	int accu_width = 180;
	uint32_t *accum = (uint32_t *)malloc(accu_width * accu_height * sizeof(uint32_t));

	double total_sum = 0.0;
	double kernel_sum = 0.0;
	double transfer_sum = 0.0;
	/* el segundo argumento decide si se ejecuta cpu o gpu */
	switch (argv[2][0])
	{
	case 'c':
		/* warmup */
		for (int i = 0; i < WARMUP_ITERS; i++)
		{
			nlines = 0;
			lane_assist_CPU(im, height, width,
							imEdge, NR, G, phi, Gx, Gy, pedge,
							sin_table, cos_table,
							accum, accu_height, accu_width,
							x1, y1, x2, y2, &nlines);
		}

		for (int iter = 0; iter < iterations; iter++)
		{
			double cpu_total_ms;
			nlines = 0;
			t0 = get_time();
			lane_assist_CPU(im, height, width,
							imEdge, NR, G, phi, Gx, Gy, pedge,
							sin_table, cos_table,
							accum, accu_height, accu_width,
							x1, y1, x2, y2, &nlines);
			t1 = get_time();
			cpu_total_ms = (t1 - t0) / 1000.0; // de micro a mili
			printf("CPU Exection time %.5f ms.\n", cpu_total_ms);
			append_benchmark_csv("cpu", width, height, cpu_total_ms, 0.0, 0.0);
			if (iterations > 1)
				total_sum += cpu_total_ms;
		}
		if (iterations > 1)
			printf("\nCPU Average time: %.5f ms\n\n", total_sum / iterations);
		break;

	case 'g':
		/* warmup */
		for (int i = 0; i < WARMUP_ITERS; i++)
		{
			nlines = 0;
			lane_assist_GPU(im, height, width,
							x1, y1, x2, y2, &nlines, NULL);
		}
		for (int iter = 0; iter < iterations; iter++)
		{
			nlines = 0;
			lane_assist_GPU(im, height, width,
							x1, y1, x2, y2, &nlines, &gpu_timing);
			printf("GPU Exection Total time: %.5f ms.\n", gpu_timing.total_ms);
			printf("GPU Kernels time: %.5f ms.\n", gpu_timing.kernels_ms);
			printf("GPU H<->D Transfer time: %.5f ms.\n", gpu_timing.transfers_ms);
			append_benchmark_csv("gpu", width, height,
								 gpu_timing.total_ms, gpu_timing.kernels_ms, gpu_timing.transfers_ms);

			if (iterations > 1)
			{
				total_sum += gpu_timing.total_ms;
				kernel_sum += gpu_timing.kernels_ms;
				transfer_sum += gpu_timing.transfers_ms;
			}
		}
		if (iterations > 1)
		{
			printf("\nGPU Average Total time: %.5f ms\n", total_sum / iterations);
			printf("GPU Average Kernel time: %.5f ms\n", kernel_sum / iterations);
			printf("GPU Average Transfer time: %.5f ms\n\n", transfer_sum / iterations);
		}
		break;

	case 'o':
		/* warmup */
		for (int i = 0; i < WARMUP_ITERS; i++)
		{
			nlines = 0;
			lane_assist_GPU_opt(im, height, width,
								x1, y1, x2, y2, &nlines, NULL);
		}
		for (int iter = 0; iter < iterations; iter++)
		{
			nlines = 0;
			lane_assist_GPU_opt(im, height, width,
								x1, y1, x2, y2, &nlines, &gpu_timing);
			printf("GPU Exection Total time: %.5f ms.\n", gpu_timing.total_ms);
			printf("GPU Kernels time: %.5f ms.\n", gpu_timing.kernels_ms);
			printf("GPU H<->D Transfer time: %.5f ms.\n", gpu_timing.transfers_ms);
			append_benchmark_csv("gpu_opt", width, height,
								 gpu_timing.total_ms, gpu_timing.kernels_ms, gpu_timing.transfers_ms);
			if (iterations > 1)
			{
				total_sum += gpu_timing.total_ms;
				kernel_sum += gpu_timing.kernels_ms;
				transfer_sum += gpu_timing.transfers_ms;
			}
		}
		if (iterations > 1)
		{
			printf("\nGPU Optimized Average Total time: %.5f ms\n", total_sum / iterations);
			printf("GPU Optimized Average Kernel time: %.5f ms\n", kernel_sum / iterations);
			printf("GPU Optimized Average Transfer time: %.5f ms\n\n", transfer_sum / iterations);
		}
		break;

	default:
		printf("Not Implemented yet!!\n");
	}

	/* se muestran y se dibujan los segmentos recuperados */
	for (int l = 0; l < nlines; l++)
		printf("(x1,y1)=(%d,%d) (x2,y2)=(%d,%d)\n", x1[l], y1[l], x2[l], y2[l]);

	draw_lines(imtmp, width, height, x1, y1, x2, y2, nlines);

	/* la imagen final se escribe en disco */
	write_png_fileRGB("out.png", imtmp, width, height);
}
