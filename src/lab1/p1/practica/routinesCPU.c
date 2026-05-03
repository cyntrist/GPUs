#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "routinesCPU.h"

#define DEG2RAD 0.017453f

/*
 * version en cpu del detector de lineas de carril
 *
 * el flujo es el mismo que en la version gpu
 * primero se detectan bordes con canny
 * despues se buscan rectas con hough
 * al final se convierten esas rectas en segmentos dibujables
 */

/*
 * canny no detecta lineas completas
 * detecta bordes, es decir, zonas donde la intensidad cambia mucho
 *
 * para hacerlo sigue cuatro etapas
 * 1 suavizado para reducir ruido
 * 2 calculo del gradiente para medir cambios
 * 3 supresion de no maximos para afinar bordes
 * 4 historesis para decidir que bordes sobreviven
 */
void canny(uint8_t *im, uint8_t *image_out,
		   float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
		   float level,
		   int height, int width)
{
	int i, j;
	int ii, jj;
	float PI = 3.141593;

	float lowthres, hithres;

	for (i = 2; i < height - 2; i++)
		for (j = 2; j < width - 2; j++)
		{
			/* suavizado 5x5 para eliminar ruido antes de buscar bordes */
			NR[i * width + j] =
				(2.0 * im[(i - 2) * width + (j - 2)] + 4.0 * im[(i - 2) * width + (j - 1)] + 5.0 * im[(i - 2) * width + (j)] + 4.0 * im[(i - 2) * width + (j + 1)] + 2.0 * im[(i - 2) * width + (j + 2)] + 4.0 * im[(i - 1) * width + (j - 2)] + 9.0 * im[(i - 1) * width + (j - 1)] + 12.0 * im[(i - 1) * width + (j)] + 9.0 * im[(i - 1) * width + (j + 1)] + 4.0 * im[(i - 1) * width + (j + 2)] + 5.0 * im[(i)*width + (j - 2)] + 12.0 * im[(i)*width + (j - 1)] + 15.0 * im[(i)*width + (j)] + 12.0 * im[(i)*width + (j + 1)] + 5.0 * im[(i)*width + (j + 2)] + 4.0 * im[(i + 1) * width + (j - 2)] + 9.0 * im[(i + 1) * width + (j - 1)] + 12.0 * im[(i + 1) * width + (j)] + 9.0 * im[(i + 1) * width + (j + 1)] + 4.0 * im[(i + 1) * width + (j + 2)] + 2.0 * im[(i + 2) * width + (j - 2)] + 4.0 * im[(i + 2) * width + (j - 1)] + 5.0 * im[(i + 2) * width + (j)] + 4.0 * im[(i + 2) * width + (j + 1)] + 2.0 * im[(i + 2) * width + (j + 2)]) / 159.0;
		}

	for (i = 2; i < height - 2; i++)
		for (j = 2; j < width - 2; j++)
		{
			// Intensity gradient of the image
			Gx[i * width + j] =
				(1.0 * NR[(i - 2) * width + (j - 2)] + 2.0 * NR[(i - 2) * width + (j - 1)] + (-2.0) * NR[(i - 2) * width + (j + 1)] + (-1.0) * NR[(i - 2) * width + (j + 2)] + 4.0 * NR[(i - 1) * width + (j - 2)] + 8.0 * NR[(i - 1) * width + (j - 1)] + (-8.0) * NR[(i - 1) * width + (j + 1)] + (-4.0) * NR[(i - 1) * width + (j + 2)] + 6.0 * NR[(i)*width + (j - 2)] + 12.0 * NR[(i)*width + (j - 1)] + (-12.0) * NR[(i)*width + (j + 1)] + (-6.0) * NR[(i)*width + (j + 2)] + 4.0 * NR[(i + 1) * width + (j - 2)] + 8.0 * NR[(i + 1) * width + (j - 1)] + (-8.0) * NR[(i + 1) * width + (j + 1)] + (-4.0) * NR[(i + 1) * width + (j + 2)] + 1.0 * NR[(i + 2) * width + (j - 2)] + 2.0 * NR[(i + 2) * width + (j - 1)] + (-2.0) * NR[(i + 2) * width + (j + 1)] + (-1.0) * NR[(i + 2) * width + (j + 2)]);

			Gy[i * width + j] =
				((-1.0) * NR[(i - 2) * width + (j - 2)] + (-4.0) * NR[(i - 2) * width + (j - 1)] + (-6.0) * NR[(i - 2) * width + (j)] + (-4.0) * NR[(i - 2) * width + (j + 1)] + (-1.0) * NR[(i - 2) * width + (j + 2)] + (-2.0) * NR[(i - 1) * width + (j - 2)] + (-8.0) * NR[(i - 1) * width + (j - 1)] + (-12.0) * NR[(i - 1) * width + (j)] + (-8.0) * NR[(i - 1) * width + (j + 1)] + (-2.0) * NR[(i - 1) * width + (j + 2)] + 2.0 * NR[(i + 1) * width + (j - 2)] + 8.0 * NR[(i + 1) * width + (j - 1)] + 12.0 * NR[(i + 1) * width + (j)] + 8.0 * NR[(i + 1) * width + (j + 1)] + 2.0 * NR[(i + 1) * width + (j + 2)] + 1.0 * NR[(i + 2) * width + (j - 2)] + 4.0 * NR[(i + 2) * width + (j - 1)] + 6.0 * NR[(i + 2) * width + (j)] + 4.0 * NR[(i + 2) * width + (j + 1)] + 1.0 * NR[(i + 2) * width + (j + 2)]);

			/* g indica la fuerza del borde y phi su direccion aproximada */
			G[i * width + j] = sqrtf((Gx[i * width + j] * Gx[i * width + j]) + (Gy[i * width + j] * Gy[i * width + j]));
			phi[i * width + j] = atan2f(fabs(Gy[i * width + j]), fabs(Gx[i * width + j]));

			/* reducimos la direccion a cuatro casos para comparar vecinos */
			if (fabs(phi[i * width + j]) <= PI / 8)
				phi[i * width + j] = 0;
			else if (fabs(phi[i * width + j]) <= 3 * (PI / 8))
				phi[i * width + j] = 45;
			else if (fabs(phi[i * width + j]) <= 5 * (PI / 8))
				phi[i * width + j] = 90;
			else if (fabs(phi[i * width + j]) <= 7 * (PI / 8))
				phi[i * width + j] = 135;
			else
				phi[i * width + j] = 0;
		}

	// Edge
	/*
	 * si el pixel no es maximo respecto a sus vecinos en la direccion del borde
	 * se descarta para adelgazar el contorno
	 */
	for (i = 3; i < height - 3; i++)
		for (j = 3; j < width - 3; j++)
		{
			pedge[i * width + j] = 0;
			if (phi[i * width + j] == 0)
			{
				/* se compara izquierda y derecha */
				if (G[i * width + j] > G[i * width + j + 1] && G[i * width + j] > G[i * width + j - 1])
					pedge[i * width + j] = 1;
			}
			else if (phi[i * width + j] == 45)
			{
				/* se compara en diagonal principal */
				if (G[i * width + j] > G[(i + 1) * width + j + 1] && G[i * width + j] > G[(i - 1) * width + j - 1])
					pedge[i * width + j] = 1;
			}
			else if (phi[i * width + j] == 90)
			{
				/* se compara arriba y abajo */
				if (G[i * width + j] > G[(i + 1) * width + j] && G[i * width + j] > G[(i - 1) * width + j])
					pedge[i * width + j] = 1;
			}
			else if (phi[i * width + j] == 135)
			{
				/* se compara en diagonal secundaria */
				if (G[i * width + j] > G[(i + 1) * width + j - 1] && G[i * width + j] > G[(i - 1) * width + j + 1])
					pedge[i * width + j] = 1;
			}
		}

	// Hysteresis Thresholding
	/*
	 * los bordes muy fuertes se aceptan
	 * los intermedios solo se aceptan si estan unidos a uno fuerte
	 */
	lowthres = level / 2;
	hithres = 2 * (level);

	for (i = 3; i < height - 3; i++)
		for (j = 3; j < width - 3; j++)
		{
			image_out[i * width + j] = 0;
			if (G[i * width + j] > hithres && pedge[i * width + j])
				image_out[i * width + j] = 255;
			else if (pedge[i * width + j] && G[i * width + j] >= lowthres && G[i * width + j] < hithres)
				// check neighbours 3x3
				/* se revisa el vecindario inmediato para mantener continuidad */
				for (ii = -1; ii <= 1; ii++)
					for (jj = -1; jj <= 1; jj++)
						if (G[(i + ii) * width + j + jj] > hithres)
							image_out[i * width + j] = 255;
		}
}

/*
 * lee el acumulador de hough y recupera rectas candidatas
 *
 * una celda con muchos votos significa que muchos pixeles de borde
 * son compatibles con la misma recta
 */
void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height,
			  float *sin_table, float *cos_table,
			  int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta, ii, jj;
	uint32_t max;

	for (rho = 0; rho < accu_height; rho++)
	{
		for (theta = 0; theta < accu_width; theta++)
		{
			/* solo se estudian celdas con suficientes votos */
			if (accumulators[(rho * accu_width) + theta] >= threshold)
			{
				// Is this point a local maxima (9x9)
				/* buscamos un maximo local para no repetir la misma recta */
				max = accumulators[(rho * accu_width) + theta];
				for (int ii = -4; ii <= 4; ii++)
				{
					for (int jj = -4; jj <= 4; jj++)
					{
						if ((ii + rho >= 0 && ii + rho < accu_height) && (jj + theta >= 0 && jj + theta < accu_width))
						{
							if (accumulators[((rho + ii) * accu_width) + (theta + jj)] > max)
							{
								max = accumulators[((rho + ii) * accu_width) + (theta + jj)];
							}
						}
					}
				}

				if (max == accumulators[(rho * accu_width) + theta]) // local maxima
				{
					int x1, y1, x2, y2;
					x1 = y1 = x2 = y2 = 0;

					/*
					 * la recta esta en formato polar
					 * aqui la convertimos a dos puntos del plano para poder dibujarla
					 */
					if (theta >= 45 && theta <= 135)
					{
						if (theta > 90)
						{
							// y = (r - x cos(t)) / sin(t)
							/* si la recta es casi vertical despejamos y */
							x1 = width / 2;
							y1 = ((float)(rho - (accu_height / 2)) - ((x1 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;
							y2 = ((float)(rho - (accu_height / 2)) - ((x2 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
						}
						else
						{
							// y = (r - x cos(t)) / sin(t)
							/* mismo caso con otro tramo util dentro de la imagen */
							x1 = 0;
							y1 = ((float)(rho - (accu_height / 2)) - ((x1 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width * 2 / 5;
							y2 = ((float)(rho - (accu_height / 2)) - ((x2 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
						}
					}
					else
					{
						// x = (r - y sin(t)) / cos(t);
						/* si la recta es mas horizontal despejamos x */
						y1 = 0;
						x1 = ((float)(rho - (accu_height / 2)) - ((y1 - (height / 2)) * sin_table[theta])) / cos_table[theta] + (width / 2);
						y2 = height;
						x2 = ((float)(rho - (accu_height / 2)) - ((y2 - (height / 2)) * sin_table[theta])) / cos_table[theta] + (width / 2);
					}

					/* se guarda el segmento recuperado */
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}

/* rellena tablas de seno y coseno para evitar recalculos repetidos */
void init_cos_sin_table(float *sin_table, float *cos_table, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		sin_table[i] = sinf(i * DEG2RAD);
		cos_table[i] = cosf(i * DEG2RAD);
	}
}

/*
 * transformada de hough
 *
 * cada pixel de borde vota por muchas rectas posibles
 * si muchos pixeles votan la misma celda del acumulador
 * esa celda representa una recta probable en la imagen
 */
void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height,
					float *sin_table, float *cos_table)
{
	int i, j, theta;

	float hough_h = ((sqrt(2.0) * (float)(height > width ? height : width)) / 2.0);

	/* el acumulador se reinicia antes de empezar a votar */
	for (i = 0; i < accu_width * accu_height; i++)
		accumulators[i] = 0;

	float center_x = width / 2.0;
	float center_y = height / 2.0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			/* solo votan los pixeles que canny ha marcado como borde */
			if (im[(i * width) + j] > 250) // Pixel is edge
			{
				for (theta = 0; theta < 180; theta++)
				{
					/* ecuacion polar de la recta respecto al centro de la imagen */
					float rho = (((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
					accumulators[(int)((round(rho + hough_h) * 180.0)) + theta]++;
				}
			}
		}
	}
}

/* convierte rgb a gris usando pesos perceptuales */
uint8_t *image_RGB2BW(uint8_t *image_in, int height, int width)
{
	int i, j;
	uint8_t *imageBW = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float R, B, G;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			R = (float)(image_in[3 * (i * width + j)]);
			G = (float)(image_in[3 * (i * width + j) + 1]);
			B = (float)(image_in[3 * (i * width + j) + 2]);

			imageBW[i * width + j] = (uint8_t)(0.2989 * R + 0.5870 * G + 0.1140 * B);
		}

	return imageBW;
}

/* pinta cada segmento detectado como una linea roja gruesa */
void draw_lines(uint8_t *imgtmp, int width, int height, int *x1, int *y1, int *x2, int *y2, int nlines)
{
	int x, y, wl, l;
	int width_line = 9;

	for (l = 0; l < nlines; l++)
		for (wl = -(width_line >> 1); wl <= (width_line >> 1); wl++)
			for (x = x1[l]; x < x2[l]; x++)
			{
				/* ecuacion de la recta conocida por dos puntos */
				y = (float)(y2[l] - y1[l]) / (x2[l] - x1[l]) * (x - x1[l]) + y1[l]; // Line eq. known two points
				if (x + wl > 0 && x + wl < width && y > 0 && y < height)
				{
					imgtmp[3 * ((y)*width + x + wl)] = 255;
					imgtmp[3 * ((y)*width + x + wl) + 1] = 0;
					imgtmp[3 * ((y)*width + x + wl) + 2] = 0;
				}
			}
}

/*
 * version completa del pipeline en cpu
 *
 * 1 canny obtiene el mapa de bordes
 * 2 hough convierte esos bordes en votos de recta
 * 3 se extraen las rectas con mas apoyo
 */
void lane_assist_CPU(uint8_t *im, int height, int width,
					 uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
					 float *sin_table, float *cos_table,
					 uint32_t *accum, int accu_height, int accu_width,
					 int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	int threshold;

	/* Canny */
	/* primero se buscan bordes fuertes y consistentes */
	canny(im, imEdge,
		  NR, G, phi, Gx, Gy, pedge,
		  1000.0f, // nivel de historesis
		  height, width);

	/* hough transform */
	/* despues se vota en hough para buscar rectas alineadas con esos bordes */
	houghtransform(imEdge, width, height, accum, accu_width, accu_height, sin_table, cos_table);

	/* el umbral se adapta al tamano principal de la imagen */
	if (width > height)
		threshold = width / 6;
	else
		threshold = height / 6;

	/* por ultimo se recuperan segmentos visibles a partir de los maximos */
	getlines(threshold, accum, accu_width, accu_height, width, height,
			 sin_table, cos_table,
			 x1, y1, x2, y2, nlines);
}
