	#ifndef ROUTINESGPUOPTIMIZED_H
	#define ROUTINESGPUOPTIMIZED_H

	#include <stdint.h>
	#include "routinesGPU.h"

	void lane_assist_GPU_opt(uint8_t *im, int height, int width,
		int *x1, int *y1, int *x2, int *y2, int *nlines, gpu_benchmark *benchmark);

	#endif
