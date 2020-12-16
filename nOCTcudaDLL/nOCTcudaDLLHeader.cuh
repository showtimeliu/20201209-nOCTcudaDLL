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

extern "C" OCT_CUDA_API int checkStatus(); 
extern "C" OCT_CUDA_API int initialize(int nMode, int nRawLineLength, int nRawNumberLines, int nProcessNumberLines, int nProcessedNumberLines); 
extern "C" OCT_CUDA_API int cleanup(int nMode); 

extern "C" OCT_CUDA_API int getReferenceData(int nMode, short* pnReferenceParallel, short* pnReferencePerpendicular, bool bIsReferenceRecorded);
extern "C" OCT_CUDA_API int getDataSDOCT(void* pnIMAQ); 
extern "C" OCT_CUDA_API int getDataPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular);

extern "C" OCT_CUDA_API int processSDOCT(); 
extern "C" OCT_CUDA_API int processPSSDOCT(); 

extern "C" OCT_CUDA_API int outputCalibrationPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular, float* pfKParallel, float* pfKPerpendicular, int* pnIndexParallel, int* pnIndexPerpendicular); 

int calculateSpectralDomainCalibration(int nMode); // may be eventually included in other functions

// kernel functions
__global__ void convert2Float(short* pnIMAQ, float* pfIMAQ, int nSize);
__global__ void calculateMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength); 
__global__ void subtractMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength); 
__global__ void calculateMask(float* pfMask, int nLength, int nStart, int nStop, int nRound);   // consider moving to host
__global__ void applyMask(cufftComplex* pcMatrix, float* pfMask, int nNumberLines, int nLineLength); 
__global__ void calculatePhase(cufftComplex* pcMatrix, float* pfPhase, int nNumberLines, int nLineLength); 
__global__ void unwrapPhase(float* pfPhase, int nNumberLines, int nLineLength, float fPiEps, float f2Pi); 
__global__ void matchPhase(float* pfPhase, int nNumberLines, int nLineLength, float f2Pi);
__global__ void getPhaseLimits(float* pfPhase, int nNumberLines, int nLineLength, int nLeft, int nRight, float* pfLeft, float* pfRight);
__global__ void calculateK(float* pfPhase, float* pfK, int* pnIndex, int* pnAssigned, int nNumberLines, int nLineLength, float* pfLineParameters, int nLeft, int nRight, float* pfLeft, float* pfRight, int nMode); 
__global__ void cleanIndex(float* pfK, int* pnIndex, int* pnAssigned, int nNumberLines, int nLineLength); 
