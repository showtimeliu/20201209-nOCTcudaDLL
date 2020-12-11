#pragma once

#ifdef OCT_CUDA_EXPORTS
#define OCT_CUDA_API __declspec(dllexport)
#else
#define OCT_CUDA_API __declspec(dllimport)
#endif // NOCT_CUDA_DLL_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cufft.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>

using namespace std; 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

extern "C" OCT_CUDA_API int getDeviceCount(int* nNumberDevices); 
extern "C" OCT_CUDA_API int getDeviceName(int nDeviceNumber, char* strDeviceName); 
extern "C" OCT_CUDA_API int initialize(int nMode, int nRawLineLength, int nRawNumberLines, int nProcessNumberLines, int nProcessedNumberLines); 
extern "C" OCT_CUDA_API int cleanup(); 

extern "C" OCT_CUDA_API int getDataSDOCT(void* pnIMAQ); 
extern "C" OCT_CUDA_API int getDataPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular);

extern "C" OCT_CUDA_API int processSDOCT(); 
extern "C" OCT_CUDA_API int processPSSDOCT(); 

