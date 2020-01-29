#include "utils.h"
#include "cuda_mat_lib.h"

#define MAT_SIZE 10000

int main(int argc, char const* argv[])
{
	double* A_h, * B_h, * C_d; // matrixes
	double gflops_gpu, gpuTime;
	float elapsed;

	// event variables used to estimate 
	// the execution times on the GPU
	cudaEvent_t start_gpu, stop_gpu;
	cudaError_t cudaStatus;

	// matrices allocation and filling
	A_h = rnd_flt_matrix(MAT_SIZE, MAT_SIZE);
	B_h = rnd_flt_matrix(MAT_SIZE, MAT_SIZE);
	C_d = zeros_flt_matrix(MAT_SIZE, MAT_SIZE);

	// initiates start and end events
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);

	printf("Matrix size; Time; GFlops\n");

	for (int i = 100; i <= MAT_SIZE; i += 100)
	{
		// records the start event
		cudaEventRecord(start_gpu, 0);

		// run product on the GPU @see matmatCuda
		cudaStatus = matmatCuda(MAT_SIZE, MAT_SIZE, MAT_SIZE,
					i, i, i, A_h, B_h, C_d);

		// records the end event
		cudaEventRecord(stop_gpu, 0);
		cudaEventSynchronize(stop_gpu);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "matmatCuda failed!");
			return 1;
		}

		// calculates the time elapsed between the 2 events
		cudaEventElapsedTime(&elapsed, start_gpu, stop_gpu);

		// turns milliseconds into seconds 
		gpuTime = elapsed / 1000.;

		gflops_gpu = ((2 * pow(i, 3)) / gpuTime) / GIGA;

		printf("%6d; %5.2lf; %5.2lf\n", i, gpuTime, gflops_gpu);
	}

	// destroy cuda events
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(stop_gpu);

	// free memory
	free(A_h);
	free(B_h);
	free(C_d);

	return 0;
}

