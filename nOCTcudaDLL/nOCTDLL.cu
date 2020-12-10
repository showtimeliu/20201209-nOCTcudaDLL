
#include "nOCTcudaDLLHeader.cuh"

int getDeviceCount(int* nNumberDevices) {
	// check for GPU
	int nDevices = -1; 
	int nRet = cudaGetDeviceCount(&nDevices); 
	if (nRet == cudaSuccess)
		*(nNumberDevices) = nDevices; 
	return nRet; 
}