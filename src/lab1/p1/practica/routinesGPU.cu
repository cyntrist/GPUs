#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "routinesCPU.h"
#include "routinesGPU.h"

#define PI_F 3.141593f

/*
 * version en cuda del detector de lineas de carril
 *
 * idea general del pipeline
 * - primero se parte de una imagen en escala de grises
 * - despues se aplica canny, que no "detecta lineas", sino bordes
 *   un borde es una zona donde la intensidad cambia bruscamente
 * - una vez tenemos bordes, se aplica hough
 *   hough no trabaja mirando trozos de recta directamente
 *   lo que hace es transformar cada pixel de borde en votos para muchas
 *   rectas posibles, si muchos pixeles votan por la misma recta,
 *   esa recta probablemente existe en la imagen
 * - por ultimo se convierten las rectas detectadas a segmentos visibles
 *   para poder dibujarlos sobre la imagen original
 *
 * la gpu acelera especialmente las fases donde hay que hacer casi lo mismo
 * para todos los pixeles: suavizado, gradiente, supresion de no maximos,
 * historesis y acumulacion de votos en hough
 *
 * la cpu al acabar se encarga de leer el acumulador de hough ya calculado y extraer
 * segmentos de recta que luego se dibujaran
 *
 * el acumulador guarda votos en coordenadas polares
 * - theta representa la inclinacion de la recta
 * - rho representa la distancia de la recta al centro de la imagen
 *
 * cuando una celda del acumulador tiene muchos votos, significa que
 * muchos pixeles de borde son compatibles con esa misma recta
 */

 
static void getlines_gpu_host(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height,
							  float *sin_table, float *cos_table,
							  int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta;
	uint32_t max;

	for (rho = 0; rho < accu_height; rho++)
	{
		for (theta = 0; theta < accu_width; theta++)
		{
			/* solo nos interesan celdas con suficientes votos */
			if (accumulators[(rho * accu_width) + theta] >= (uint32_t)threshold)
			{
				/*
				 * buscamos un maximo local en una ventana 9x9
				 * esto evita devolver muchas veces la misma recta con
				 * pequenas variaciones de rho y theta
				 */
				max = accumulators[(rho * accu_width) + theta];
				for (int ii = -4; ii <= 4; ii++)
				{
					for (int jj = -4; jj <= 4; jj++)
					{
						if ((ii + rho >= 0 && ii + rho < accu_height) && (jj + theta >= 0 && jj + theta < accu_width))
						{
							if (accumulators[((rho + ii) * accu_width) + (theta + jj)] > max)
								max = accumulators[((rho + ii) * accu_width) + (theta + jj)];
						}
					}
				}

				if (max == accumulators[(rho * accu_width) + theta])
				{
					/*
					 * ahora convertimos la recta polar a dos puntos del plano
					 * esos dos puntos nos permiten dibujar un segmento
					 */
					int x1_local = 0;
					int y1_local = 0;
					int x2_local = 0;
					int y2_local = 0;

					/*
					 * cuando theta esta cerca de vertical usamos la ecuacion
					 * despejada para y, cuando esta cerca de horizontal usamos
					 * la ecuacion despejada para x
					 *
					 * se hace asi para evitar divisiones inestables y para
					 * obtener segmentos razonables dentro de la imagen
					 */
					if (theta >= 45 && theta <= 135)
					{
						if (theta > 90)
						{
							/* caso inclinado hacia la derecha en la mitad inferior */
							x1_local = width / 2;
							y1_local = ((float)(rho - (accu_height / 2)) - ((x1_local - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2_local = width;
							y2_local = ((float)(rho - (accu_height / 2)) - ((x2_local - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
						}
						else
						{
							/* caso inclinado hacia la izquierda en la mitad inferior */
							x1_local = 0;
							y1_local = ((float)(rho - (accu_height / 2)) - ((x1_local - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2_local = width * 2 / 5;
							y2_local = ((float)(rho - (accu_height / 2)) - ((x2_local - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
						}
					}
					else
					{
						/* caso mas cercano a horizontal, despejamos x a partir de y */
						y1_local = 0;
						x1_local = ((float)(rho - (accu_height / 2)) - ((y1_local - (height / 2)) * sin_table[theta])) / cos_table[theta] + (width / 2);
						y2_local = height;
						x2_local = ((float)(rho - (accu_height / 2)) - ((y2_local - (height / 2)) * sin_table[theta])) / cos_table[theta] + (width / 2);
					}

					/* guardamos el segmento recuperado para dibujarlo mas tarde */
					x1_lines[*lines] = x1_local;
					y1_lines[*lines] = y1_local;
					x2_lines[*lines] = x2_local;
					y2_lines[*lines] = y2_local;
					(*lines)++;
				}
			}
		}
	}
}

/* pone el acumulador de hough a cero antes de empezar a votar */
__global__ void init_accum(uint32_t *accum, int size)
{
	/* cada hilo se encarga de una posicion del vector */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
		accum[idx] = 0;
}

/*
 * primer paso de canny, reducir ruido
 *
 * si intentamos detectar bordes sobre una imagen ruidosa, apareceran
 * muchisimos falsos positivos, por eso antes se suaviza la imagen con
 * un filtro 5x5 parecido a una gaussiana
 */
__global__ void reduce_noise_kernel(uint8_t *im, float *NR, int width, int height)
{
	/* x es la columna e y es la fila del pixel que procesa este hilo */
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	/* los bordes de la imagen no tienen vecinos suficientes para el filtro 5x5 */
	if (y < 2 || y >= height - 2 || x < 2 || x >= width - 2)
		return;

	/* combinacion ponderada de los 25 vecinos del entorno */
	NR[y * width + x] =
		(2.0f * im[(y - 2) * width + (x - 2)] + 4.0f * im[(y - 2) * width + (x - 1)] + 5.0f * im[(y - 2) * width + x] + 4.0f * im[(y - 2) * width + (x + 1)] + 2.0f * im[(y - 2) * width + (x + 2)] +
		 4.0f * im[(y - 1) * width + (x - 2)] + 9.0f * im[(y - 1) * width + (x - 1)] + 12.0f * im[(y - 1) * width + x] + 9.0f * im[(y - 1) * width + (x + 1)] + 4.0f * im[(y - 1) * width + (x + 2)] +
		 5.0f * im[y * width + (x - 2)] + 12.0f * im[y * width + (x - 1)] + 15.0f * im[y * width + x] + 12.0f * im[y * width + (x + 1)] + 5.0f * im[y * width + (x + 2)] +
		 4.0f * im[(y + 1) * width + (x - 2)] + 9.0f * im[(y + 1) * width + (x - 1)] + 12.0f * im[(y + 1) * width + x] + 9.0f * im[(y + 1) * width + (x + 1)] + 4.0f * im[(y + 1) * width + (x + 2)] +
		 2.0f * im[(y + 2) * width + (x - 2)] + 4.0f * im[(y + 2) * width + (x - 1)] + 5.0f * im[(y + 2) * width + x] + 4.0f * im[(y + 2) * width + (x + 1)] + 2.0f * im[(y + 2) * width + (x + 2)]) /
		159.0f;
}

/*
 * segundo paso de canny, medir donde cambia la intensidad como gradiente
 *
 * gx mide cuanto cambia la imagen en horizontal
 * gy mide cuanto cambia la imagen en vertical
 * con ambos valores calculamos
 * - g: fuerza del borde
 * - phi: direccion aproximada del borde
 *
 * despues la direccion se simplifica a solo cuatro casos, 0, 45, 90 y 135
 */
__global__ void gradient_kernel(float *NR, float *G, float *phi, float *Gx, float *Gy, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float angle;

	if (y < 2 || y >= height - 2 || x < 2 || x >= width - 2)
		return;

	/* derivada horizontal, resalta cambios izquierda derecha */
	Gx[y * width + x] =
		(1.0f * NR[(y - 2) * width + (x - 2)] + 2.0f * NR[(y - 2) * width + (x - 1)] - 2.0f * NR[(y - 2) * width + (x + 1)] - 1.0f * NR[(y - 2) * width + (x + 2)] +
		 4.0f * NR[(y - 1) * width + (x - 2)] + 8.0f * NR[(y - 1) * width + (x - 1)] - 8.0f * NR[(y - 1) * width + (x + 1)] - 4.0f * NR[(y - 1) * width + (x + 2)] +
		 6.0f * NR[y * width + (x - 2)] + 12.0f * NR[y * width + (x - 1)] - 12.0f * NR[y * width + (x + 1)] - 6.0f * NR[y * width + (x + 2)] +
		 4.0f * NR[(y + 1) * width + (x - 2)] + 8.0f * NR[(y + 1) * width + (x - 1)] - 8.0f * NR[(y + 1) * width + (x + 1)] - 4.0f * NR[(y + 1) * width + (x + 2)] +
		 1.0f * NR[(y + 2) * width + (x - 2)] + 2.0f * NR[(y + 2) * width + (x - 1)] - 2.0f * NR[(y + 2) * width + (x + 1)] - 1.0f * NR[(y + 2) * width + (x + 2)]);

	/* derivada vertical, resalta cambios arriba abajo */
	Gy[y * width + x] =
		(-1.0f * NR[(y - 2) * width + (x - 2)] - 4.0f * NR[(y - 2) * width + (x - 1)] - 6.0f * NR[(y - 2) * width + x] - 4.0f * NR[(y - 2) * width + (x + 1)] - 1.0f * NR[(y - 2) * width + (x + 2)] -
		 2.0f * NR[(y - 1) * width + (x - 2)] - 8.0f * NR[(y - 1) * width + (x - 1)] - 12.0f * NR[(y - 1) * width + x] - 8.0f * NR[(y - 1) * width + (x + 1)] - 2.0f * NR[(y - 1) * width + (x + 2)] +
		 2.0f * NR[(y + 1) * width + (x - 2)] + 8.0f * NR[(y + 1) * width + (x - 1)] + 12.0f * NR[(y + 1) * width + x] + 8.0f * NR[(y + 1) * width + (x + 1)] + 2.0f * NR[(y + 1) * width + (x + 2)] +
		 1.0f * NR[(y + 2) * width + (x - 2)] + 4.0f * NR[(y + 2) * width + (x - 1)] + 6.0f * NR[(y + 2) * width + x] + 4.0f * NR[(y + 2) * width + (x + 1)] + 1.0f * NR[(y + 2) * width + (x + 2)]);

	/* magnitud del gradiente, cuanto de fuerte es el borde */
	G[y * width + x] = sqrtf(Gx[y * width + x] * Gx[y * width + x] + Gy[y * width + x] * Gy[y * width + x]);

	/* angulo del gradiente, hacia donde cambia mas la intensidad */
	angle = atan2f(fabsf(Gy[y * width + x]), fabsf(Gx[y * width + x]));

	/* cuantizacion a cuatro direcciones para simplificar la comparacion vecina */
	if (fabsf(angle) <= PI_F / 8.0f)
		phi[y * width + x] = 0.0f;
	else if (fabsf(angle) <= 3.0f * (PI_F / 8.0f))
		phi[y * width + x] = 45.0f;
	else if (fabsf(angle) <= 5.0f * (PI_F / 8.0f))
		phi[y * width + x] = 90.0f;
	else if (fabsf(angle) <= 7.0f * (PI_F / 8.0f))
		phi[y * width + x] = 135.0f;
	else
		phi[y * width + x] = 0.0f;
}

/*
 * tercer paso de canny, supresion de no maximos
 *
 * esta fase adelgaza los bordes
 * - mira la direccion del borde
 * - compara el pixel actual con sus dos vecinos en esa direccion
 * - solo conserva el pixel si es el mas fuerte
 */
__global__ void edge_kernel(float *G, float *phi, uint8_t *pedge, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx;

	if (y >= height || x >= width)
		return;

	idx = y * width + x;

	/* por defecto asumimos que el pixel no sobrevive */
	pedge[idx] = 0;

	/* evitamos accesos fuera de rango cuando comparemos con vecinos */
	if (y < 3 || y >= height - 3 || x < 3 || x >= width - 3)
		return;

	/* borde casi horizontal, comparamos izquierda y derecha */
	if (phi[idx] == 0.0f)
	{
		if (G[idx] > G[idx + 1] && G[idx] > G[idx - 1])
			pedge[idx] = 1;
	}
	/* diagonal principal */
	else if (phi[idx] == 45.0f)
	{
		if (G[idx] > G[(y + 1) * width + (x + 1)] && G[idx] > G[(y - 1) * width + (x - 1)])
			pedge[idx] = 1;
	}
	/* borde casi vertical, comparamos arriba y abajo */
	else if (phi[idx] == 90.0f)
	{
		if (G[idx] > G[(y + 1) * width + x] && G[idx] > G[(y - 1) * width + x])
			pedge[idx] = 1;
	}
	/* diagonal secundaria */
	else if (phi[idx] == 135.0f)
	{
		if (G[idx] > G[(y + 1) * width + (x - 1)] && G[idx] > G[(y - 1) * width + (x + 1)])
			pedge[idx] = 1;
	}
}

/*
 * cuarto paso de canny, umbralizacion con historesis
 *
 * que bordes se conservan de verdad
 * - si un borde es muy fuerte, se acepta directamente
 * - si es intermedio, solo se acepta si esta conectado a un borde fuerte
 * - si es debil y aislado, se descarta
 */
__global__ void hysteresis_kernel(float *G, uint8_t *pedge, uint8_t *image_out, float level, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx;

	/* umbral bajo y umbral alto derivados del mismo parametro de nivel */
	float lowthres = level / 2.0f;
	float hithres = 2.0f * level;

	if (y >= height || x >= width)
		return;

	idx = y * width + x;

	/* salida a negro por defecto */
	image_out[idx] = 0;

	if (y < 3 || y >= height - 3 || x < 3 || x >= width - 3)
		return;

	/* borde claramente fuerte, se mantiene */
	if (G[idx] > hithres && pedge[idx])
	{
		image_out[idx] = 255;
		return;
	}

	/*
	 * borde intermedio, solo se salva si algun vecino cercano ya supera
	 * el umbral alto, asi preservamos continuidad en los contornos
	 */
	if (pedge[idx] && G[idx] >= lowthres && G[idx] < hithres)
	{
		for (int ii = -1; ii <= 1; ii++)
			for (int jj = -1; jj <= 1; jj++)
				if (G[(y + ii) * width + (x + jj)] > hithres)
					image_out[idx] = 255;
	}
}

/*
 * transformada de hough
 *
 * para cada borde que detecta canny, 
 * hough dice que rectas podrian pasar por él
 *
 * para cada pixel de borde
 * - probamos muchos angulos theta
 * - calculamos el rho correspondiente para cada angulo
 * - sumamos un voto en la celda (rho, theta)
 *
 * si muchos pixeles distintos votan a la misma celda, esa celda describe
 * una recta coherente presente en la imagen
 */
__global__ void hough_kernel(uint8_t *im, int width, int height,
							 uint32_t *accum, int accu_width, int accu_height,
							 float *sin_table, float *cos_table, float hough_h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	/* usamos el centro de la imagen como origen para rho */
	float center_x = width / 2.0f;
	float center_y = height / 2.0f;

	if (y >= height || x >= width)
		return;

	/* solo votan los pixeles que canny ha marcado como borde fuerte */
	if (im[y * width + x] > 250)
	{
		/* un mismo pixel puede pertenecer a muchas rectas posibles */
		for (int theta = 0; theta < accu_width; theta++)
		{
			/* ecuacion polar de la recta respecto al centro de la imagen */
			float rho = ((float)x - center_x) * cos_table[theta] + ((float)y - center_y) * sin_table[theta];

			/* desplazamos rho para poder usarlo como indice no negativo */
			int rho_idx = (int)roundf(rho + hough_h);

			/*
			 * atomicadd es necesaria porque muchos hilos distintos pueden
			 * intentar votar a la misma celda del acumulador al mismo tiempo
			 */
			if (rho_idx >= 0 && rho_idx < accu_height)
				atomicAdd(&accum[rho_idx * accu_width + theta], 1u);
		}
	}
}


/*
 * funcion principal de la version gpu
 * - reserva memoria en la gpu
 * - copia la imagen y las tablas seno coseno
 * - lanza los kernels de canny y hough
 * - recupera el acumulador
 * - reconstruye segmentos en cpu
 * - libera memoria
 */
void lane_assist_GPU(uint8_t *im, int height, int width,
					 int *x1, int *y1, int *x2, int *y2, int *nlines, gpu_benchmark *timings)
{
	/* punteros a memoria de dispositivo */
	uint8_t *d_im;
	uint8_t *d_imEdge, *d_pedge;
	uint32_t *d_accum;
	float *d_NR, *d_G, *d_phi, *d_Gx, *d_Gy;
	float *d_sin, *d_cos;

	/* tablas trigonometricas reutilizadas por hough para no recalcular seno y coseno */
	float sin_table[180], cos_table[180];
	init_cos_sin_table(sin_table, cos_table, 180);

	/* tamano del eje rho del acumulador de hough */
	float hough_h = ((sqrt(2.0) * (float)(height > width ? height : width)) / 2.0);
	int accu_height = hough_h * 2.0;
	int accu_width = 180;

	/* copia en cpu del acumulador, se usa al final para extraer lineas */
	uint32_t *accum = (uint32_t *)malloc(accu_height * accu_width * sizeof(uint32_t));

	/* tamanos auxiliares para reservar y copiar memoria */
	size_t img_size = width * height * sizeof(uint8_t);
	size_t accum_size = accu_height * accu_width * sizeof(uint32_t);
	size_t float_img_size = width * height * sizeof(float);

	cudaEvent_t total_start, total_stop;
	cudaEvent_t kernels_start, kernels_stop;
	cudaEvent_t h2d_start, h2d_stop;
	cudaEvent_t d2h_start, d2h_stop;

	cudaEventCreate(&total_start);
	cudaEventCreate(&total_stop);
	cudaEventCreate(&kernels_start);
	cudaEventCreate(&kernels_stop);
	cudaEventCreate(&h2d_start);
	cudaEventCreate(&h2d_stop);
	cudaEventCreate(&d2h_start);
	cudaEventCreate(&d2h_stop);

	/* reserva de todos los buffers que viven en la gpu */
	cudaMalloc(&d_im, img_size);
	cudaMalloc(&d_imEdge, img_size);
	cudaMalloc(&d_pedge, img_size);
	cudaMalloc(&d_accum, accum_size);
	cudaMalloc(&d_NR, float_img_size);
	cudaMalloc(&d_G, float_img_size);
	cudaMalloc(&d_phi, float_img_size);
	cudaMalloc(&d_Gx, float_img_size);
	cudaMalloc(&d_Gy, float_img_size);
	cudaMalloc(&d_sin, 180 * sizeof(float));
	cudaMalloc(&d_cos, 180 * sizeof(float));

	/* subimos a la gpu la imagen de entrada y las tablas trigonometricas */
	cudaEventRecord(total_start);
	cudaEventRecord(h2d_start);
	cudaMemcpy(d_im, im, img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sin, sin_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cos, cos_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(h2d_stop);

	/* configuracion para inicializar el acumulador linealmente */
	int threads = 256;
	int blocks = (accu_width * accu_height + threads - 1) / threads;

	/* configuracion para kernels que recorren la imagen */
	dim3 block(16, 16);
	dim3 grid((width + 15) / 16, (height + 15) / 16);
	
	cudaEventRecord(kernels_start);
	init_accum<<<blocks, threads>>>(d_accum, accu_width * accu_height);


	/*
	 * CANNY
	 * 1 reduccion de ruido
	 * 2 gradiente
	 * 3 edge
	 * 4 hysteresis
	 * HOUGH
	 * 5 transformada
	 */
	reduce_noise_kernel<<<grid, block>>>(d_im, d_NR, width, height);
	gradient_kernel<<<grid, block>>>(d_NR, d_G, d_phi, d_Gx, d_Gy, width, height);
	edge_kernel<<<grid, block>>>(d_G, d_phi, d_pedge, width, height);
	hysteresis_kernel<<<grid, block>>>(d_G, d_pedge, d_imEdge, 1000.0f, width, height);
	hough_kernel<<<grid, block>>>(d_imEdge,
								  width, height,
								  d_accum,
								  accu_width, accu_height,
								  d_sin, d_cos,
								  hough_h);
	cudaEventRecord(kernels_stop);
	cudaEventSynchronize(kernels_stop);	

	/* volvemos a cpu solo con el acumulador, que es lo necesario para extraer lineas */
	cudaEventRecord(d2h_start);
	cudaMemcpy(accum, d_accum, accum_size, cudaMemcpyDeviceToHost);
	cudaEventRecord(d2h_stop);
	cudaEventRecord(total_stop);
	cudaEventSynchronize(total_stop);

	int threshold;

	/* umbral empirico, crece con el tamano dominante de la imagen */
	if (width > height)
		threshold = width / 6;
	else
		threshold = height / 6;

	/* reconstruimos segmentos visibles a partir de los maximos del acumulador */
	getlines_gpu_host(threshold, accum, accu_width, accu_height,
					  width, height,
					  sin_table, cos_table,
					  x1, y1, x2, y2, nlines);

	if (timings != NULL)
	{
		float total_ms = 0.0f;
		float kernels_ms = 0.0f;
		float h2d_ms = 0.0f;
		float d2h_ms = 0.0f;

		cudaEventElapsedTime(&total_ms, total_start, total_stop);
		cudaEventElapsedTime(&kernels_ms, kernels_start, kernels_stop);
		cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);
		cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);

		timings->total_ms = total_ms;
		timings->kernels_ms = kernels_ms;
		timings->transfers_ms = h2d_ms + d2h_ms;
	}

	/* limpieza final de memoria de dispositivo */
	cudaFree(d_im);
	cudaFree(d_imEdge);
	cudaFree(d_pedge);
	cudaFree(d_accum);
	cudaFree(d_NR);
	cudaFree(d_G);
	cudaFree(d_phi);
	cudaFree(d_Gx);
	cudaFree(d_Gy);
	cudaFree(d_sin);
	cudaFree(d_cos);

	cudaEventDestroy(total_start);
	cudaEventDestroy(total_stop);
	cudaEventDestroy(kernels_start);
	cudaEventDestroy(kernels_stop);
	cudaEventDestroy(h2d_start);
	cudaEventDestroy(h2d_stop);
	cudaEventDestroy(d2h_start);
	cudaEventDestroy(d2h_stop);

	/* limpieza de la copia host del acumulador */
	free(accum);
}
