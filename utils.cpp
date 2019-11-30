#include "utils.h"

double* rnd_flt_matrix(int n, int m)
{
	int size = n * m;
	double* A = (double*)malloc(sizeof(double) * size);

	for (int i = 0; i < size; ++i)
		A[i] = (float)rand() / RAND_MAX;

	return A;
}

double* zeros_flt_matrix(int n, int m)
{
	return (double*)calloc(n * m, sizeof(double));
}