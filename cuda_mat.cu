#include "cuda_mat_lib.h"

cudaError_t matmatCuda(int lda, int ldb, int ldc,
		       int n, int m, int p, double* A, double* B, double* C)
{
	double* A_d = NULL;
	double* B_d = NULL;
	double* C_d = NULL;

	// Calculates the number of grid blocks
	unsigned int nblock = (n + THREADS - 1) / THREADS;
	unsigned int mblock = (m + THREADS - 1) / THREADS;
	unsigned int pblock = (p + THREADS - 1) / THREADS;

	// Calculates the smallest multiple matrix
	// of THREADS that may contain matrix A and B
	int nwidth = nblock * THREADS;
	int mwidth = mblock * THREADS;
	int pwidth = pblock * THREADS;

	cudaError_t cudaStatus;

	// number of block for A
	dim3 blocksA;
	blocksA.x = mblock;
	blocksA.y = nblock;

	// number of block for B
	dim3 blocksB;
	blocksB.x = pblock;
	blocksB.y = mblock;

	// number of block for C
	dim3 blocksC;
	blocksC.x = pblock;
	blocksC.y = nblock;

	// number of threads for each block
	dim3 threads;
	threads.x = THREADS;
	threads.y = THREADS;

	// Select GPU (for multi GPU systems)
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "1: cudaSetDevice failed!\n");
		fprintf(stderr, "Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	// Allocate space in GPU memory 
	cudaStatus = cudaMalloc((void**)&A_d, nwidth * mwidth * sizeof(double));

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "2: cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&B_d, mwidth * pwidth * sizeof(double));
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3: cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&C_d, nwidth * pwidth * sizeof(double));

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "4: cudaMalloc failed!\n");
		goto Error;
	}
	// end memory allocation

	// copy matrices from RAM to GPU memory
	cudaStatus = cudaMemcpy2D(A_d, mwidth * sizeof(double),
				  A, lda * sizeof(double),
				  m * sizeof(double), n, cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "5: cudaMemcpy2D failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy2D(B_d, pwidth * sizeof(double),
				  B, ldb * sizeof(double), p * sizeof(double),
				  m, cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "6: cudaMemcpy2D failed!\n");
		goto Error;
	}
	// end of copy

	// fill the excess space with zeros
	if ((n % THREADS) || (m % THREADS) || (p % THREADS)) {
		adjustMatrix <<<blocksA, threads>>> (A_d, n, m);
		adjustMatrix <<<blocksB, threads>>> (B_d, m, p);
	}

	// multiplies the matrices
	productKernel <<<blocksC, threads>>> (A_d, B_d, C_d, mwidth, pwidth);

	// copies the result from the GPU memory to the RAM memory
	cudaStatus = cudaMemcpy2D(C, ldc * sizeof(double),
				  C_d, pwidth * sizeof(double), p * sizeof(double),
				  n, cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "7: cudaMemcpy2D failed!\n");
		goto Error;
	}

	// deallocation of GPU memory
Error:
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	return cudaStatus;
}

// procedure that inserts the value 0 
// to fill in the empty matrix spaces on the frame
__global__ void adjustMatrix(double* M, int n, int m)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x * i + j;

	if (i >= n || j >= m)
		M[offset] = 0.;
}


// executes the product between a row of A and a column of B
__global__ void productKernel(double* A, double* B, double* C, int m, int p)
{
	// block id on the ordinate inside the grid
	int ib = blockIdx.y;
	// block id on the abscissa inside the grid
	int jb = blockIdx.x;
	// thread id on the ordinate within the block
	int it = threadIdx.y;
	// thread id on the abscissa inside the block  
	int jt = threadIdx.x;

	int a, b, c, k;

	// index of the first sub-matrix of A
	int aBegin = m * THREADS * ib;

	// index of the last sub-matrix of A
	int aEnd = aBegin + m - 1;

	// number of columns between a sub-matrix and the next sub-matrix
	int aStep = THREADS;

	// index of the first sub-matrix of B
	int bBegin = THREADS * jb;

	// number of elements between a sub-matrix and the next sub-matrix
	int bStep = THREADS * p;

	// local variable to memorize the result
	double Csub = 0;

	// block shared sub-matrices
	// reduce the number of access to the GPU memory
	__shared__ double As[THREADS][THREADS];
	__shared__ double Bs[THREADS][THREADS];

	// for each sub-matrix 
	for (a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{
		// each thread uploads an element to the shared sub-matrix
		As[it][jt] = A[a + m * it + jt];
		Bs[it][jt] = B[b + p * it + jt];

		// wait for all threads to load an element before starting 
		__syncthreads();

		// calculates the contributions for the loaded submatrix
		for (k = 0; k < THREADS; ++k)
			Csub += As[it][k] * Bs[k][jt];

		// wait for all threads in the block to finish the calculation
		// before you start loading other elements
		__syncthreads();
	}

	// copy Csub in C
	c = p * THREADS * ib + THREADS * jb;
	C[c + p * it + jt] = Csub;
}
