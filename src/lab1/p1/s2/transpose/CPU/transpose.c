#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>

/* Time */
static struct timeval tv0;
double getMicroSeconds()
{
  double t;
  gettimeofday(&tv0, (struct timezone*)0);
  t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);
  return t;
}

void init_seed()
{
  int seedi = 1;
  FILE *fd = fopen("/dev/urandom", "r");
  fread(&seedi, sizeof(int), 1, fd);
  fclose(fd);
  srand(seedi);
}

void init2Drand(float **buffer, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      buffer[i][j] = 500.0f * ((float)rand() / (float)RAND_MAX) - 500.0f; /* [-500, 500] */
}

float *getmemory1D(int nx)
{
  float *buffer = (float*)malloc((size_t)nx * sizeof(float));
  if (!buffer) {
    fprintf(stderr, "ERROR in memory allocation\n");
    return NULL;
  }
  for (int i = 0; i < nx; i++) buffer[i] = 0.0f;
  return buffer;
}

float **getmemory2D(int nx, int ny)
{
  float **buffer = (float**)malloc((size_t)nx * sizeof(float*));
  if (!buffer) {
    fprintf(stderr, "ERROR in memory allocation\n");
    return NULL;
  }

  buffer[0] = (float*)malloc((size_t)nx * (size_t)ny * sizeof(float));
  if (!buffer[0]) {
    fprintf(stderr, "ERROR in memory allocation\n");
    free(buffer);
    return NULL;
  }

  for (int i = 1; i < nx; i++) buffer[i] = buffer[i-1] + ny;

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      buffer[i][j] = 0.0f;

  return buffer;
}

/********************************************************************************/
/********************************************************************************/

/*
 * Transpose 2D version (correct OpenMP: single region)
 */
void transpose2D(float **in, float **out, int n)
{
  #pragma omp parallel for collapse(2)
  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++)
      out[j][i] = in[i][j];
}

/*
 * Transpose 1D version
 */
void transpose1D(const float *in, float *out, int n)
{
  #pragma omp parallel for collapse(2)
  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++)
      out[(size_t)j * n + i] = in[(size_t)i * n + j];
}

int main(int argc, char **argv)
{
  int n;

  float **array2D, **array2D_trans;
  float *array1D,  *array1D_trans;

  double t0, seconds;
  double bytes;

  if (argc == 2) n = atoi(argv[1]);
  else {
    n = 4096;
    printf("./exec n (by default n=%i)\n", n);
  }

  init_seed();

  array2D       = getmemory2D(n, n);
  array2D_trans = getmemory2D(n, n);
  if (!array2D || !array2D_trans) return 1;

  array1D       = array2D[0];
  array1D_trans = array2D_trans[0];

  init2Drand(array2D, n);

  /* For transpose: read + write => 2 * n*n*sizeof(float) bytes moved (approx) */
  bytes = 2.0 * (double)n * (double)n * (double)sizeof(float);

  /* Transpose 2D version */
  t0 = getMicroSeconds();
  transpose2D(array2D, array2D_trans, n);
  seconds = (getMicroSeconds() - t0) / 1e6;
  printf("Transpose version 2D: %f MB/s  (threads=%d)\n",
         (bytes / seconds) / 1024.0 / 1024.0, omp_get_max_threads());

  /* Transpose 1D version */
  t0 = getMicroSeconds();
  transpose1D(array1D, array1D_trans, n);
  seconds = (getMicroSeconds() - t0) / 1e6;
  printf("Transpose version 1D: %f MB/s  (threads=%d)\n",
         (bytes / seconds) / 1024.0 / 1024.0, omp_get_max_threads());

  free(array2D[0]);       free(array2D);
  free(array2D_trans[0]); free(array2D_trans);

  return 0;
}

