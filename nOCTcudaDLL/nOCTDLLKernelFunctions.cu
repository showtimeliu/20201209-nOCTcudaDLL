#include "nOCTcudaDLLHeader.cuh"

__global__ void convert2Float(short* pnIMAQ, float* pfIMAQ, int nSize) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x; 
	if (tid < nSize)
		pfIMAQ[tid] = (float)pnIMAQ[tid]; 
}