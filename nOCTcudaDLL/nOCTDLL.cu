
#include "nOCTcudaDLLHeader.cuh"

static int gnMode = -1;
static int gnRawLineLength;
static int gnRawNumberLines;
static int gnCalibrationNumberLines;
static int gnProcessNumberLines;
static int gnProcessedNumberLines;
static int gnPerpendicular;
static int gnAllocationStatus = 0;
static int gnMidLength;

static float* gpfRawCalibration;
static float* gpfProcessCalibration;
static size_t gnProcessCalibrationPitch;

// reference
static float* gpfReferenceEven;
static float* gpfReferenceOdd;

// fft
static cufftComplex* gpcProcessDepthProfile;
static size_t gnProcessDepthProfilePitch;
static cufftHandle gchForward;

// calibration mask
static int gnCalibrationStart;
static int gnCalibrationStop;
static int gnCalibrationRound;
static float* gpfCalibrationMask;

// reverse fft
static cufftComplex* gpcProcessSpectrum;
static size_t gnProcessSpectrumPitch;
static cufftHandle gchReverse;

// phase
static float* gpfProcessPhase;
static size_t gnProcessPhasePitch;

// unwrap
static float gfPiEps = (float)(acos(-1.0) - 1.0e-30);
static float gf2Pi = (float)(2.0 * acos(-1.0));

// linear fit and interpolation
static float* gpfLeftPhase;
static float* gpfRightPhase;
static float* gpfKLineCoefficients;
static float* gpfProcessK;
static size_t gnKPitch;
static int* gpnProcessIndex;
static size_t gnIndexPitch;
static int* gpnProcessAssigned;
static size_t gnAssignedPitch;
static int gnKMode;
static float* gpfProcessSpectrumK;
static size_t gnSpectrumKPitch;

static float* gpfProcessOCT;
static size_t gnProcessOCTPitch;
static cufftComplex* gpcProcessedOCT;

// dispersion mask
static int gnDispersionStart;
static int gnDispersionStop;
static int gnDispersionRound;
static float* gpfDispersionMask;

// dispersion correction
static cufftComplex* gpcDispersionCorrection;
static cufftHandle gchForwardComplex;
static cufftComplex* gpcProcessKCorrected;
static size_t gnKCorrectedPitch;


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

int initialize(int nMode, int nRawLineLength, int nRawNumberLines, int nProcessNumberLines, int nProcessedNumberLines) {
    
    cleanup();

    // copy parameters to global parameters
    gnMode = nMode;
    gnRawLineLength = nRawLineLength;
    gnRawNumberLines = nRawNumberLines;
    gnProcessNumberLines = nProcessNumberLines;
    gnProcessedNumberLines = nProcessedNumberLines;

    // allocate memory
    switch (nMode) {
    case 0: // SD-OCT
        gnPerpendicular = 0;
        gnCalibrationNumberLines = 1;
        break;
    case 1: // PS SD-OCT
        gnPerpendicular = 1;
        gnCalibrationNumberLines = gnRawNumberLines;
        break;
    case 2: // line field
        gnPerpendicular = 0;
        gnCalibrationNumberLines = 1;
        break;
    case 3: // OFDI
        gnPerpendicular = 0;
        gnCalibrationNumberLines = gnRawNumberLines;
        break;
    case 4: // PS OFDI
        gnPerpendicular = 1;
        gnCalibrationNumberLines = gnRawNumberLines;
        break;
    } // switch (nMode)

    gpuErrchk(cudaMallocHost((void**)&gpfRawCalibration, (gnRawLineLength * gnCalibrationNumberLines) * sizeof(float)));
    gpuErrchk(cudaMallocPitch((void**)&gpfProcessCalibration, &gnProcessCalibrationPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));

    gpuErrchk(cudaMalloc((void**)&gpfReferenceEven, gnRawLineLength * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&gpfReferenceOdd, gnRawLineLength * sizeof(float)));

    gnMidLength = (int)(gnRawLineLength / 2 + 1);
    gpuErrchk(cudaMallocPitch((void**)&gpcProcessDepthProfile, &gnProcessDepthProfilePitch, gnRawLineLength * sizeof(cufftComplex), gnProcessNumberLines >> 1));
    int nRank = 1;
    int pn[] = { gnRawLineLength };
    int nIStride = 1, nOStride = 1;
    int nIDist = gnProcessCalibrationPitch / sizeof(float);
    int nODist = gnProcessDepthProfilePitch / sizeof(cufftComplex);
    int pnINEmbed[] = { 0 };
    int pnONEmbed[] = { 0 };
    int nBatch = gnProcessNumberLines >> 1;
    cufftPlanMany(&gchForward, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_R2C, nBatch);

    gpuErrchk(cudaMalloc((void**)&gpfCalibrationMask, gnRawLineLength * sizeof(float)));

    gpuErrchk(cudaMallocPitch((void**)&gpcProcessSpectrum, &gnProcessSpectrumPitch, gnRawLineLength * sizeof(cufftComplex), gnProcessNumberLines >> 1));
    nIDist = gnProcessDepthProfilePitch / sizeof(cufftComplex);
    nODist = gnProcessSpectrumPitch / sizeof(cufftComplex);
    cufftPlanMany(&gchReverse, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_C2C, nBatch);

    gpuErrchk(cudaMallocPitch((void**)&gpfProcessPhase, &gnProcessPhasePitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));

    cudaMalloc((void**)&gpfLeftPhase, sizeof(float));
    cudaMalloc((void**)&gpfRightPhase, sizeof(float));
    cudaMalloc((void**)&gpfKLineCoefficients, 2 * sizeof(float));
    gpuErrchk(cudaMallocPitch((void**)&gpfProcessK, &gnKPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));
    gpuErrchk(cudaMallocPitch((void**)&gpnProcessIndex, &gnIndexPitch, gnRawLineLength * sizeof(int), gnProcessNumberLines >> 1));
    gpuErrchk(cudaMallocPitch((void**)&gpnProcessAssigned, &gnAssignedPitch, gnRawLineLength * sizeof(int), gnProcessNumberLines >> 1));

    gpuErrchk(cudaMallocPitch((void**)&gpfProcessSpectrumK, &gnSpectrumKPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));

    gpuErrchk(cudaMallocPitch((void**)&gpfProcessOCT, &gnProcessOCTPitch, gnRawLineLength * sizeof(float), gnProcessNumberLines >> 1));
    gpuErrchk(cudaMallocHost((void**)&gpcProcessedOCT, (gnMidLength * gnProcessedNumberLines) * sizeof(cufftComplex)));

    gpuErrchk(cudaMalloc((void**)&gpfDispersionMask, gnRawLineLength * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&gpcDispersionCorrection, gnRawLineLength * sizeof(cufftComplex)));
    gpuErrchk(cudaMallocPitch((void**)&gpcProcessKCorrected, &gnKCorrectedPitch, gnRawLineLength * sizeof(cufftComplex), gnProcessNumberLines >> 1));

    nIDist = gnKCorrectedPitch / sizeof(cufftComplex);
    cufftPlanMany(&gchForwardComplex, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_C2C, nBatch);

    gnAllocationStatus = 1;

    return -1;

} // int initialize

int cleanup() {

    // free memory allocations
    if (gnAllocationStatus == 1) {
        gpuErrchk(cudaFreeHost(gpfRawCalibration));
        gpuErrchk(cudaFree(gpfProcessCalibration));
        gpuErrchk(cudaFree(gpfReferenceEven));
        gpuErrchk(cudaFree(gpfReferenceOdd));
        gpuErrchk(cudaFree(gpcProcessDepthProfile));
        cufftDestroy(gchForward);
        gpuErrchk(cudaFree(gpfCalibrationMask));
        gpuErrchk(cudaFree(gpcProcessSpectrum));
        cufftDestroy(gchReverse);
        gpuErrchk(cudaFree(gpfProcessPhase));
        cudaFree(gpfLeftPhase);
        cudaFree(gpfRightPhase);
        cudaFree(gpfKLineCoefficients);
        cudaFree(gpfProcessK);
        cudaFree(gpnProcessIndex);
        cudaFree(gpnProcessAssigned);
        cudaFree(gpfProcessSpectrumK);
        cudaFree(gpfProcessOCT);
        cudaFreeHost(gpcProcessedOCT);
        gpuErrchk(cudaFree(gpfDispersionMask));
        gpuErrchk(cudaFree(gpcDispersionCorrection));
        cufftDestroy(gchForwardComplex);
        cudaFree(gpcProcessKCorrected);

        gnAllocationStatus = 0;
    }   // if (gnAllocationStatus
    return -1;

}

int getDataSDOCT(void* pnIMAQ) {
    return -1; }

int getDataPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular) {
    return -1; 
}



int processPSSDOCT() {

}