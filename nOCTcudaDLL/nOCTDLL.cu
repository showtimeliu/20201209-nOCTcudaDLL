
#include "nOCTcudaDLLHeader.cuh"

static int gnMode = -1;
static int gnRawLineLength;
static int gnRawNumberLines;            // number of lines in a frame 
// static int gnCalibrationNumberLines;
static int gnProcessNumberLines;        // number of lines in a chunk
static int gnProcessedNumberLines;
static int gnPerpendicular;
static int gnAllocationStatus = 0;
static int gnMidLength;

/* raw spectra arrays */
    // common
// static short* h_gpnRawIMAQ;             // host: raw spectra from camera 
static short* d_gpnRawIMAQ;             // device: raw spectra from camera
static float* d_gpfRawIMAQ;             // device: raw spectra (gpfRawCalibration) 
static float* gpfIMAQPitched;           // device: raw spectra copied to pitched memory (gpfProcessCalibration)
static size_t gnIMAQPitch;              // gnProcessCalibrationPitch
    // PS-SD-OCT
// static short* h_gpnRawIMAQParallel;     // host: raw spectra from camera 
static short* d_gpnRawIMAQParallel;     // device: raw spectra from camera
static float* d_gpfRawIMAQParallel;     // device: raw spectra (gpfRawCalibration) 
// static short* h_gpnRawIMAQPerpendicular;    // host: raw spectra from camera 
static short* d_gpnRawIMAQPerpendicular;    // device: raw spectra from camera
static float* d_gpfRawIMAQPerpendicular;    // device: raw spectra (gpfRawCalibration) 


/* reference */
    // common 
static float* gpfReference; 
    // PS-SD-OCT
static float* gpfReferenceEven;
static float* gpfReferenceOdd;

/* fft */ 
static cufftHandle gchForward;
static cufftComplex* gpcProcessDepthProfile; 
static size_t gnProcessDepthProfilePitch;    


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
    
    cleanup(nMode);

    // copy parameters to global parameters
    gnMode = nMode;
    gnRawLineLength = nRawLineLength;
    gnRawNumberLines = nRawNumberLines;                 // number of lines in a frame
    gnProcessNumberLines = nProcessNumberLines;         // number of lines in a chunk
    gnProcessedNumberLines = nProcessedNumberLines;

    int nActualProcessNumberLines; 

    // allocate memory  

    switch (nMode) {
    case 0: // SD-OCT
        gnPerpendicular = 0;
        // gnCalibrationNumberLines = 1;
        nActualProcessNumberLines = gnProcessNumberLines; 

        // gpuErrchk(cudaMallocHost((void**)&h_gpnRawIMAQ, (gnRawLineLength * gnRawNumberLines) * sizeof(short)));
        gpuErrchk(cudaMalloc((void**)&d_gpnRawIMAQ, (gnRawLineLength * gnRawNumberLines) * sizeof(short)));
        gpuErrchk(cudaMalloc((void**)&d_gpfRawIMAQ, (gnRawLineLength * gnRawNumberLines) * sizeof(float)));

        gpuErrchk(cudaMalloc((void**)&gpfReference, gnRawLineLength * sizeof(float))); 

        break;
    case 1: // PS SD-OCT
        gnPerpendicular = 1;
        // gnCalibrationNumberLines = gnRawNumberLines; // ???
        nActualProcessNumberLines = gnProcessNumberLines >> 1;      // only process odd or even lines

        // gpuErrchk(cudaMallocHost((void**)&h_gpnRawIMAQParallel, (gnRawLineLength * gnRawNumberLines) * sizeof(short)));
        gpuErrchk(cudaMalloc((void**)&d_gpnRawIMAQParallel, (gnRawLineLength * gnRawNumberLines) * sizeof(short)));
        gpuErrchk(cudaMalloc((void**)&d_gpfRawIMAQParallel, (gnRawLineLength * gnRawNumberLines) * sizeof(float)));
        // gpuErrchk(cudaMallocHost((void**)&h_gpnRawIMAQPerpendicular, (gnRawLineLength * gnRawNumberLines) * sizeof(short)));
        gpuErrchk(cudaMalloc((void**)&d_gpnRawIMAQPerpendicular, (gnRawLineLength * gnRawNumberLines) * sizeof(short)));
        gpuErrchk(cudaMalloc((void**)&d_gpfRawIMAQPerpendicular, (gnRawLineLength * gnRawNumberLines) * sizeof(float)));
        
        gpuErrchk(cudaMalloc((void**)&gpfReferenceEven, gnRawLineLength * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&gpfReferenceOdd, gnRawLineLength * sizeof(float)));        

        break;
    case 2: // line field
        gnPerpendicular = 0;
        // gnCalibrationNumberLines = 1;
        break;
    case 3: // OFDI
        gnPerpendicular = 0;
        // gnCalibrationNumberLines = gnRawNumberLines;
        break;
    case 4: // PS OFDI
        gnPerpendicular = 1;
        // gnCalibrationNumberLines = gnRawNumberLines;
        break;
    } // switch (nMode)

    gpuErrchk(cudaMallocPitch((void**)&gpfIMAQPitched, &gnIMAQPitch, gnRawLineLength * sizeof(float), nActualProcessNumberLines));
     
    gnMidLength = (int)(gnRawLineLength / 2 + 1);
    gpuErrchk(cudaMallocPitch((void**)&gpcProcessDepthProfile, &gnProcessDepthProfilePitch, gnRawLineLength * sizeof(cufftComplex), nActualProcessNumberLines));
    int nRank = 1;
    int pn[] = { gnRawLineLength };
    int nIStride = 1, nOStride = 1;
    int nIDist = gnIMAQPitch / sizeof(float);
    int nODist = gnProcessDepthProfilePitch / sizeof(cufftComplex);
    int pnINEmbed[] = { 0 };
    int pnONEmbed[] = { 0 };
    int nBatch = gnProcessNumberLines >> 1;
    cufftPlanMany(&gchForward, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_R2C, nBatch);

    gpuErrchk(cudaMalloc((void**)&gpfCalibrationMask, gnRawLineLength * sizeof(float)));

    gpuErrchk(cudaMallocPitch((void**)&gpcProcessSpectrum, &gnProcessSpectrumPitch, gnRawLineLength * sizeof(cufftComplex), nActualProcessNumberLines));
    nIDist = gnProcessDepthProfilePitch / sizeof(cufftComplex);
    nODist = gnProcessSpectrumPitch / sizeof(cufftComplex);
    cufftPlanMany(&gchReverse, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_C2C, nBatch);

    gpuErrchk(cudaMallocPitch((void**)&gpfProcessPhase, &gnProcessPhasePitch, gnRawLineLength * sizeof(float), nActualProcessNumberLines));

    cudaMalloc((void**)&gpfLeftPhase, sizeof(float));
    cudaMalloc((void**)&gpfRightPhase, sizeof(float));
    cudaMalloc((void**)&gpfKLineCoefficients, 2 * sizeof(float));
    gpuErrchk(cudaMallocPitch((void**)&gpfProcessK, &gnKPitch, gnRawLineLength * sizeof(float), nActualProcessNumberLines));
    gpuErrchk(cudaMallocPitch((void**)&gpnProcessIndex, &gnIndexPitch, gnRawLineLength * sizeof(int), nActualProcessNumberLines));
    gpuErrchk(cudaMallocPitch((void**)&gpnProcessAssigned, &gnAssignedPitch, gnRawLineLength * sizeof(int), nActualProcessNumberLines));

    gpuErrchk(cudaMallocPitch((void**)&gpfProcessSpectrumK, &gnSpectrumKPitch, gnRawLineLength * sizeof(float), nActualProcessNumberLines));

    gpuErrchk(cudaMallocPitch((void**)&gpfProcessOCT, &gnProcessOCTPitch, gnRawLineLength * sizeof(float), nActualProcessNumberLines));
    gpuErrchk(cudaMallocHost((void**)&gpcProcessedOCT, (gnMidLength * gnProcessedNumberLines) * sizeof(cufftComplex)));

    gpuErrchk(cudaMalloc((void**)&gpfDispersionMask, gnRawLineLength * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&gpcDispersionCorrection, gnRawLineLength * sizeof(cufftComplex)));
    gpuErrchk(cudaMallocPitch((void**)&gpcProcessKCorrected, &gnKCorrectedPitch, gnRawLineLength * sizeof(cufftComplex), nActualProcessNumberLines));

    nIDist = gnKCorrectedPitch / sizeof(cufftComplex);
    cufftPlanMany(&gchForwardComplex, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_C2C, nBatch);

    gpuErrchk(cudaDeviceSynchronize());

    gnAllocationStatus = 1;

    return -1;

} // int initialize

int cleanup(int nMode) {

    // free memory allocations
    if (gnAllocationStatus == 1) {
        
        
        switch (nMode)
        {
        case 0: // SD-OCT
            gpuErrchk(cudaFreeHost(h_gpnRawIMAQ));
            gpuErrchk(cudaFree(d_gpnRawIMAQ));
            gpuErrchk(cudaFree(d_gpfRawIMAQ));

            gpuErrchk(cudaFree(gpfReference)); 
            
            break;
        case 1: // PS SD-OCT 
            gpuErrchk(cudaFreeHost(h_gpnRawIMAQParallel));
            gpuErrchk(cudaFree(d_gpnRawIMAQParallel));
            gpuErrchk(cudaFree(d_gpfRawIMAQParallel));
            gpuErrchk(cudaFreeHost(h_gpnRawIMAQPerpendicular));
            gpuErrchk(cudaFree(d_gpnRawIMAQPerpendicular));
            gpuErrchk(cudaFree(d_gpfRawIMAQPerpendicular));

            gpuErrchk(cudaFree(gpfReferenceEven));
            gpuErrchk(cudaFree(gpfReferenceOdd));

            break;
        case 2: // line field
            
            break;
        case 3: // OFDI
            
            break;
        case 4: // PS OFDI
            
            break;
            break;
        }
        
        gpuErrchk(cudaFree(gpfIMAQPitched));   
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

    


    return -1; 
}

int getDataPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular) {
    cudaMemcpy(d_gpnRawIMAQParallel, pnIMAQParallel, (gnRawLineLength * gnRawNumberLines) * sizeof(short), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_gpnRawIMAQPerpendicular, pnIMAQPerpendicular, (gnRawLineLength * gnRawNumberLines) * sizeof(short), cudaMemcpyHostToDevice);

    gpuErrchk(cudaDeviceSynchronize());

    return -1; 
}


int processSDOCT() {
    return -1; 

}


int processPSSDOCT() {

    int nThreadsPerBlock; 
    dim3 d3Threads; 
    dim3 d3Blocks; 

    // convert to float type 
    d3Threads.x = 512;  d3Threads.y = 1;    d3Threads.z = 1; 
    d3Blocks.x = gnRawLineLength * gnRawNumberLines / d3Threads.x; 
    d3Blocks.y = 1;     d3Blocks.z = 1;
    convert2Float << <d3Blocks, d3Threads >> > (d_gpnRawIMAQParallel, d_gpfRawIMAQParallel, gnRawLineLength * gnRawNumberLines); 
    gpuErrchk(cudaPeekAtLastError());
    convert2Float << <d3Blocks, d3Threads >> > (d_gpnRawIMAQPerpendicular, d_gpfRawIMAQPerpendicular, gnRawLineLength * gnRawNumberLines);
    gpuErrchk(cudaPeekAtLastError()); 

    // loop through cameras
    for (int nCam = 0; nCam < 2; nCam++) {

    }




    return -1; 
}
