#ifndef __CUDA_MAT_LIB_H__
#define __CUDA_MAT_LIB_H__

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define THREADS 32

__host__ cudaError_t matmatCuda(int, int, int, int, int, int, double*, double*, double*);
__global__ void productKernel(double*, double*, double*, int, int);
__global__ void adjustMatrix(double*, int, int);

#endif // !__CUDA_MAT_LIB_H__

