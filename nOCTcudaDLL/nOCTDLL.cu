
#include "nOCTcudaDLLHeader.cuh"

int getDeviceCount(int* nNumberDevices) {
	// check for GPU
	int nDevices = -1; 
	int nRet = cudaGetDeviceCount(&nDevices); 
	if (nRet == cudaSuccess)
		*(nNumberDevices) = nDevices; 
	return nRet; 
}

int getDeviceName(int nDeviceNumber, char* strDeviceName) {
    // check for GPU
    cudaDeviceProp currentDevice;
    int nRet = cudaGetDeviceProperties(&currentDevice, nDeviceNumber);
    if (nRet == cudaSuccess) {
        sprintf(strDeviceName, "%s (%d SMs, %d b/s, %d t/b, %d t/s, %d shared kB, %d GB)",
            currentDevice.name,
            currentDevice.multiProcessorCount,
            currentDevice.maxBlocksPerMultiProcessor,
            currentDevice.maxThreadsPerBlock,
            currentDevice.maxThreadsPerMultiProcessor,
            currentDevice.sharedMemPerBlock / 1024,
            currentDevice.totalGlobalMem / 1024 / 1024 / 1024);

    }	// if (nRet
    return nRet;
}