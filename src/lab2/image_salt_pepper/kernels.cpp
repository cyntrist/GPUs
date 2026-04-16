
#include <sycl/sycl.hpp>
#include "math.h"

using namespace sycl;

#define MAX_WINDOW_SIZE 5 * 5

void _bubble_sort(float array[], int size)
{
	int i, j;
	float tmp;

	for (i = 1; i < size; i++)
		for (j = 0; j < size - i; j++)
			if (array[j] > array[j + 1])
			{
				tmp = array[j];
				array[j] = array[j + 1];
				array[j + 1] = tmp;
			}
}

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out,
					   float thredshold, int window_size,
					   int height, int width)
{
	int i, j, ii, jj;

	int ws2 = (window_size - 1) >> 1;

	Q.submit([&](handler &cgh) {
		cgh.parallel_for(sycl::range<2>(width - ws2, height - ws2), [=](id<2> px)
		{
			float median; 
			float window[MAX_WINDOW_SIZE];
			int i = px[0];
			int j = px[1];
			int ii, jj;

			if (i >= ws2 && j >= ws2) 
			{
				for (ii = -ws2; ii <= ws2; ii++) 
				{
					for (jj = -ws2; jj <= ws2; jj++) 
					{
						window[(ii+ws2)*window_size + jj+ws2] = im[(i+ii)*width + j+jj];
					}
				}	
			}

			_bubble_sort(window, window_size * window_size);
			median = window[(window_size * window_size - 1) >> 1]; 

			if (fabsf((median - im[i * width + j]) / median) <= thredshold)
				image_out[i * width + j] = im[i * width + j];
			else
				image_out[i * width + j] = median;

		});
	 }).wait();
}
