
#include "nOCTcudaDLLHeader.cuh"

// status and control parameters
static int gnMode = -1; 
static int gnAllocationStatus = 0; 
static bool gbIsReferenceRecorded   = false; 
static bool gbIsReferenceLoaded     = false;
static bool gbIsCalibrationLoaded   = false; 
static bool gbIsDispersionLoaded    = false; 


static int gnRawLineLength;
static int gnRawNumberLines;            // number of lines in a frame 
// static int gnCalibrationNumberLines;
static int gnProcessNumberLines;        // number of lines in a chunk
static int gnProcessedNumberLines;
static int gnPerpendicular;
static int gnMidLength;

/* raw spectra arrays */
    // common
static short* d_gpnRawIMAQ;             // device: raw spectra from camera
static float* d_gpfRawIMAQ;             // device: raw spectra (gpfRawCalibration) 
static float* gpfIMAQPitched;           // device: raw spectra copied to pitched memory (gpfProcessCalibration)
static size_t gnIMAQPitch;              // gnProcessCalibrationPitch
    // PS-SD-OCT
static short* d_gpnRawIMAQParallel;     // device: raw spectra from camera
static float* d_gpfRawIMAQParallel;     // device: raw spectra (gpfRawCalibration) 
static short* d_gpnRawIMAQPerpendicular;    // device: raw spectra from camera
static float* d_gpfRawIMAQPerpendicular;    // device: raw spectra (gpfRawCalibration) 


/* reference */
    // common 
static float* gpfReference; 
    // PS-SD-OCT
static float* gpfReferenceParallel;
static float* gpfReferencePerpendicular;

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

/* linear fit and interpolation */
    // common
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
    // PS-SD-OCT
static float* gpfProcessKParallel; 
static float* gpfProcessKPerpendicular;
static int* gpnProcessIndexParallel;
static int* gpnProcessIndexPerpendicular;
static size_t gnKPitchParallel; 
static size_t gnKPitchPerpendicular; 
static size_t gnIndexPitchParallel; 
static size_t gnIndexPitchPerpendicular; 

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

// static float* gpfProcessOCT;
static size_t gnProcessOCTPitch;
static cufftComplex* gpcProcessedOCT;
    // PS-SD-OCT
static cufftComplex* gpcProcessedOCTParallelOdd; 
static cufftComplex* gpcProcessedOCTParallelEven;
static cufftComplex* gpcProcessedOCTPerpendicularOdd;
static cufftComplex* gpcProcessedOCTPerpendicularEven;



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

int checkStatus() {

    return -1; 
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
        // gnCalibrationNumberLines = gnRawNumberLines; // QUESTION: what is this parameter?
        nActualProcessNumberLines = gnProcessNumberLines >> 1;      // only process every other line in each array

        gpuErrchk(cudaMalloc((void**)&d_gpnRawIMAQParallel, (gnRawLineLength * gnRawNumberLines) * sizeof(short)));
        gpuErrchk(cudaMalloc((void**)&d_gpfRawIMAQParallel, (gnRawLineLength * gnRawNumberLines) * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&d_gpnRawIMAQPerpendicular, (gnRawLineLength * gnRawNumberLines) * sizeof(short)));
        gpuErrchk(cudaMalloc((void**)&d_gpfRawIMAQPerpendicular, (gnRawLineLength * gnRawNumberLines) * sizeof(float)));
        
        gpuErrchk(cudaMalloc((void**)&gpfReferenceParallel, gnRawLineLength * sizeof(float)));
        gpuErrchk(cudaMalloc((void**)&gpfReferencePerpendicular, gnRawLineLength * sizeof(float)));  
        
        gpuErrchk(cudaMallocPitch((void**)&gpfProcessKParallel, &gnKPitchParallel, gnRawLineLength * sizeof(float), nActualProcessNumberLines));
        gpuErrchk(cudaMallocPitch((void**)&gpfProcessKPerpendicular, &gnKPitchPerpendicular, gnRawLineLength * sizeof(float), nActualProcessNumberLines));
        gpuErrchk(cudaMallocPitch((void**)&gpnProcessIndexParallel, &gnIndexPitchParallel, gnRawLineLength * sizeof(int), nActualProcessNumberLines));
        gpuErrchk(cudaMallocPitch((void**)&gpnProcessIndexPerpendicular, &gnIndexPitchPerpendicular, gnRawLineLength * sizeof(int), nActualProcessNumberLines));

        gpuErrchk(cudaMalloc((void**)&gpcProcessedOCTParallelOdd, (gnMidLength * gnProcessedNumberLines) * sizeof(cufftComplex)));
        gpuErrchk(cudaMalloc((void**)&gpcProcessedOCTParallelEven, (gnMidLength * gnProcessedNumberLines) * sizeof(cufftComplex)));
        gpuErrchk(cudaMalloc((void**)&gpcProcessedOCTPerpendicularOdd, (gnMidLength * gnProcessedNumberLines) * sizeof(cufftComplex)));
        gpuErrchk(cudaMalloc((void**)&gpcProcessedOCTPerpendicularEven, (gnMidLength * gnProcessedNumberLines) * sizeof(cufftComplex)));


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

    // gpuErrchk(cudaMallocPitch((void**)&gpfProcessOCT, &gnProcessOCTPitch, gnRawLineLength * sizeof(float), nActualProcessNumberLines));
    gpuErrchk(cudaMallocHost((void**)&gpcProcessedOCT, (gnMidLength * gnProcessedNumberLines) * sizeof(cufftComplex)));

    gpuErrchk(cudaMalloc((void**)&gpfDispersionMask, gnRawLineLength * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&gpcDispersionCorrection, gnRawLineLength * sizeof(cufftComplex)));
    gpuErrchk(cudaMallocPitch((void**)&gpcProcessKCorrected, &gnKCorrectedPitch, gnRawLineLength * sizeof(cufftComplex), nActualProcessNumberLines));

    nIDist = gnKCorrectedPitch / sizeof(cufftComplex);
    cufftPlanMany(&gchForwardComplex, nRank, pn, pnINEmbed, nIStride, nIDist, pnONEmbed, nOStride, nODist, CUFFT_C2C, nBatch);

    gpuErrchk(cudaDeviceSynchronize());     // QUESTION: will cudaDeviceSynchronize slow down the performance?

    gnAllocationStatus = 1;

    return -1;

} // int initialize

int cleanup(int nMode) {

    // free memory allocations
    if (gnAllocationStatus == 1) {
        
        
        switch (nMode)
        {
        case 0: // SD-OCT
            // gpuErrchk(cudaFreeHost(h_gpnRawIMAQ));
            gpuErrchk(cudaFree(d_gpnRawIMAQ));
            gpuErrchk(cudaFree(d_gpfRawIMAQ));

            gpuErrchk(cudaFree(gpfReference)); 
            
            break;
        case 1: // PS SD-OCT 
            gpuErrchk(cudaFree(d_gpnRawIMAQParallel));
            gpuErrchk(cudaFree(d_gpfRawIMAQParallel));
            gpuErrchk(cudaFree(d_gpnRawIMAQPerpendicular));
            gpuErrchk(cudaFree(d_gpfRawIMAQPerpendicular));

            gpuErrchk(cudaFree(gpfReferenceParallel));
            gpuErrchk(cudaFree(gpfReferencePerpendicular));
            gpuErrchk(cudaFree(gpfProcessKParallel));
            gpuErrchk(cudaFree(gpfProcessKPerpendicular));
            gpuErrchk(cudaFree(gpnProcessIndexParallel));
            gpuErrchk(cudaFree(gpnProcessIndexPerpendicular));

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
        // cudaFree(gpfProcessOCT);
        cudaFreeHost(gpcProcessedOCT);
        gpuErrchk(cudaFree(gpfDispersionMask));
        gpuErrchk(cudaFree(gpcDispersionCorrection));
        cufftDestroy(gchForwardComplex);
        cudaFree(gpcProcessKCorrected);

        gnAllocationStatus = 0;
    }   // if (gnAllocationStatus
    return -1;

}

int getReferenceData(int nMode, short* pnReferenceParallel, short* pnReferencePerpendicular, bool bIsReferenceRecorded) {
    // copy parameters to global parameters
    gbIsReferenceRecorded = bIsReferenceRecorded; 

    if (bIsReferenceRecorded == true) {        

        switch (nMode) {
        case 0: // SD-OCT
            // data type conversion (on host)
            float* pfReference;
            pfReference = (float*)malloc(gnRawLineLength * sizeof(float));

            for (int i; i < gnRawLineLength; i++) {
                pfReference[i] = (float)pnReferenceParallel[i];
            }

            // copy data to device
            gpuErrchk(cudaMemcpy(gpfReference, pfReference, gnRawLineLength * sizeof(short), cudaMemcpyHostToDevice));
            gpuErrchk(cudaDeviceSynchronize());

            free(pfReference); 

            gbIsReferenceLoaded = true;

            break;
        case 1: // PS SD-OCT
            // data type conversion (on host)
            float* pfReferenceParallel, * pfReferencePerpendicular;
            pfReferenceParallel = (float*)malloc(gnRawLineLength * sizeof(float));
            pfReferencePerpendicular = (float*)malloc(gnRawLineLength * sizeof(float));

            for (int i; i < gnRawLineLength; i++) {
                pfReferenceParallel[i] = (float)pnReferenceParallel[i];
                pfReferencePerpendicular[i] = (float)pnReferencePerpendicular[i];
            }

            // copy data to device
            gpuErrchk(cudaMemcpy(gpfReferenceParallel, pfReferenceParallel, gnRawLineLength * sizeof(short), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(gpfReferencePerpendicular, pfReferencePerpendicular, gnRawLineLength * sizeof(short), cudaMemcpyHostToDevice));
            gpuErrchk(cudaDeviceSynchronize());

            free(pfReferenceParallel);
            free(pfReferencePerpendicular); 

            gbIsReferenceLoaded = true;

            break;
        case 2: // line field

            break;
        case 3: // OFDI

            break;
        case 4: // PS OFDI

            break;
        } // switch (nMode)

    }

    return -1; 
}

int getCalibrationData() {

    return -1; 
}

int getDataSDOCT(void* pnIMAQ) {

    


    return -1; 
}

int getDataPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular) {
    // copy data to device
    gpuErrchk(cudaMemcpy(d_gpnRawIMAQParallel, pnIMAQParallel, (gnRawLineLength * gnRawNumberLines) * sizeof(short), cudaMemcpyHostToDevice)); 
    gpuErrchk(cudaMemcpy(d_gpnRawIMAQPerpendicular, pnIMAQPerpendicular, (gnRawLineLength * gnRawNumberLines) * sizeof(short), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    int nThreadsPerBlock;
    dim3 d3Threads;
    dim3 d3Blocks;

    // convert to float type 
    d3Threads.x = 512;  d3Threads.y = 1;    d3Threads.z = 1;
    d3Blocks.x = (gnRawLineLength * gnRawNumberLines - 1) / d3Threads.x + 1;
    d3Blocks.y = 1;     d3Blocks.z = 1;
    convert2Float << <d3Blocks, d3Threads >> > (d_gpnRawIMAQParallel, d_gpfRawIMAQParallel, gnRawLineLength * gnRawNumberLines);
    gpuErrchk(cudaPeekAtLastError());
    convert2Float << <d3Blocks, d3Threads >> > (d_gpnRawIMAQPerpendicular, d_gpfRawIMAQPerpendicular, gnRawLineLength * gnRawNumberLines);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaDeviceSynchronize());

    return -1; 
}

int calculateSpectralDomainCalibration(int nMode) { // can be used in both SD-OCT and PS SD-OCT, not in DLL
    int nThreadsPerBlock;
    dim3 d3Threads;
    dim3 d3Blocks;

    int nActualProcessNumberLines;

    switch (nMode) {
    case 0: // SD-OCT
        nActualProcessNumberLines = gnProcessNumberLines;
        break; 
    case 1: // PS SD-OCT
        nActualProcessNumberLines = gnProcessNumberLines >> 1; 
        break; 
    }

    /********** calibration ************/
    /* forward fft */
    gpuErrchk(cudaMemset2D(gpcProcessDepthProfile, gnProcessDepthProfilePitch, 0.0, gnProcessDepthProfilePitch, nActualProcessNumberLines));
    cufftExecR2C(gchForward, gpfIMAQPitched, gpcProcessDepthProfile);

    /* mask */
    // calculate mask: QUESTION can be done in CPU in the initialize function? (small data size, avoid warp divergence) 
    nThreadsPerBlock = 512;
    calculateMask << <gnRawLineLength / nThreadsPerBlock, nThreadsPerBlock >> > (gpfCalibrationMask, gnRawLineLength, 50, 100, 16);     // grab these numbers from C# UI

    // apply mask
    d3Threads.x = 32;
    d3Threads.y = 1024 / d3Threads.x;
    d3Threads.z = 1;
    d3Blocks.x = gnProcessNumberLines / d3Threads.x;
    d3Blocks.y = 1;
    d3Blocks.z = 1;
    applyMask << <d3Blocks, d3Threads >> > (gpcProcessDepthProfile, gpfCalibrationMask, nActualProcessNumberLines, gnRawLineLength);
    gpuErrchk(cudaPeekAtLastError());

    /* reverse fft */
    cufftExecC2C(gchReverse, gpcProcessDepthProfile, gpcProcessSpectrum, CUFFT_INVERSE);

    /* calculate phase */
    d3Threads.x = 32;
    d3Threads.y = 1024 / d3Threads.x;
    d3Threads.z = 1;
    d3Blocks.x = gnRawLineLength / d3Threads.x;
    d3Blocks.y = nActualProcessNumberLines / d3Threads.y;
    d3Blocks.z = 1;
    calculatePhase << <d3Blocks, d3Threads >> > (gpcProcessSpectrum, gpfProcessPhase, nActualProcessNumberLines, gnRawLineLength);
    gpuErrchk(cudaPeekAtLastError());

    d3Threads.x = 256;
    d3Threads.y = 1024 / d3Threads.x;
    d3Threads.z = 1;
    d3Blocks.x = nActualProcessNumberLines / d3Threads.y;
    d3Blocks.y = 1;
    d3Blocks.z = 1;
    unwrapPhase << <d3Blocks, d3Threads >> > (gpfProcessPhase, nActualProcessNumberLines, gnRawLineLength, gfPiEps, gf2Pi);

    d3Threads.x = 256;
    d3Threads.y = 1024 / d3Threads.x;
    d3Threads.z = 1;
    d3Blocks.x = nActualProcessNumberLines / d3Threads.y;
    d3Blocks.y = 1;
    d3Blocks.z = 1;
    matchPhase << <d3Blocks, d3Threads >> > (gpfProcessPhase, nActualProcessNumberLines, gnRawLineLength, gf2Pi);

    nThreadsPerBlock = 32;
    getPhaseLimits << <2, nThreadsPerBlock >> > (gpfProcessPhase, nActualProcessNumberLines, gnRawLineLength, 32, gnRawLineLength - 32, gpfLeftPhase, gpfRightPhase);

    gnKMode = 1;
    d3Threads.x = 128;
    d3Threads.y = 1024 / d3Threads.x;
    d3Threads.z = 1;
    d3Blocks.x = nActualProcessNumberLines / d3Threads.y;
    d3Blocks.y = 1;
    d3Blocks.z = 1;
    cudaMemset2D(gpnProcessAssigned, gnAssignedPitch, 0, gnRawLineLength * sizeof(int), nActualProcessNumberLines);
    calculateK << <d3Blocks, d3Threads >> > (gpfProcessPhase, gpfProcessK, gpnProcessAssigned, gpnProcessIndex, nActualProcessNumberLines, gnRawLineLength, \
        gpfKLineCoefficients, 32, gnRawLineLength - 32, gpfLeftPhase, gpfRightPhase, gnKMode);

    d3Threads.x = 128;
    d3Threads.y = 1024 / d3Threads.x;
    d3Threads.z = 1;
    d3Blocks.x = nActualProcessNumberLines / d3Threads.y;
    d3Blocks.y = 1;
    d3Blocks.z = 1;
    cleanIndex << <d3Blocks, d3Threads >> > (gpfProcessK, gpnProcessIndex, gpnProcessAssigned, nActualProcessNumberLines, gnRawLineLength);

    gpuErrchk(cudaDeviceSynchronize());

    return -1; 
} 

int outputCalibrationPSSDOCT(void* pnIMAQParallel, void* pnIMAQPerpendicular, float* pfKParallel, float* pfKPerpendicular, int* pnIndexParallel, int* pnIndexPerpendicular) {    
    // output pfK, pnIndex and return to C# and save
    // get a frame of data
    getDataPSSDOCT(pnIMAQParallel, pnIMAQPerpendicular); 

    int nThreadsPerBlock;
    dim3 d3Threads;
    dim3 d3Blocks;    

    int nActualProcessNumberLines = gnProcessNumberLines >> 1;  // half of a chunk 

    // allocate host arrays
    pfKParallel = (float*)malloc(gnRawLineLength * nActualProcessNumberLines * sizeof(float)); 
    pfKPerpendicular = (float*)malloc(gnRawLineLength * nActualProcessNumberLines * sizeof(float)); 
    pnIndexParallel = (int*)malloc(gnRawLineLength * nActualProcessNumberLines * sizeof(int));
    pnIndexPerpendicular = (int*)malloc(gnRawLineLength * nActualProcessNumberLines * sizeof(int));

    // loop through cameras 
    for (int nCam = 0; nCam < 2; nCam++) {  // nCam = 0: parallel camera; nCam = 1: perpendicular camera
        /* copy data */
        switch (nCam) {
        case 0: // parallel camera
            gpuErrchk(cudaMemcpy2D(gpfIMAQPitched, gnIMAQPitch, d_gpfRawIMAQParallel, gnIMAQPitch, gnIMAQPitch, nActualProcessNumberLines, cudaMemcpyDeviceToDevice));
            break; 
        case 1: // perpendicular camera
            gpuErrchk(cudaMemcpy2D(gpfIMAQPitched, gnIMAQPitch, d_gpfRawIMAQPerpendicular, gnIMAQPitch, gnIMAQPitch, nActualProcessNumberLines, cudaMemcpyDeviceToDevice));
            break;
        } 
        gpuErrchk(cudaDeviceSynchronize());

        /* reference */
        if (gbIsReferenceRecorded == false) { // no reference data recorded
            // calculate reference 
            d3Threads.x = 128;
            d3Threads.y = 1024 / d3Threads.x;
            d3Threads.z = 1;
            d3Blocks.x = gnProcessNumberLines / d3Threads.x;
            d3Blocks.y = 1;
            d3Blocks.z = 1;

            // different cameras
            switch (nCam) {
            case 0: // parallel camera
                calculateMean << <d3Blocks, d3Threads >> > (gpfIMAQPitched, gpfReferenceParallel, nActualProcessNumberLines, gnRawLineLength);
                break;
            case 1: // perpendicular camera
                calculateMean << <d3Blocks, d3Threads >> > (gpfIMAQPitched, gpfReferencePerpendicular, nActualProcessNumberLines, gnRawLineLength);
                break;
            }
            gpuErrchk(cudaPeekAtLastError());
        } // if (gbIsReferenceRecorded == false)

        // subtract reference 
        d3Threads.x = 32;
        d3Threads.y = 1024 / d3Threads.x;
        d3Threads.z = 1;
        d3Blocks.x = gnProcessNumberLines / d3Threads.x;
        d3Blocks.y = 1;
        d3Blocks.z = 1;
        // different cameras
        switch (nCam) {
        case 0: // parallel camera
            subtractMean << <d3Blocks, d3Threads >> > (gpfIMAQPitched, gpfReferenceParallel, nActualProcessNumberLines, gnRawLineLength);
            break;
        case 1: // perpendicular camera
            subtractMean << <d3Blocks, d3Threads >> > (gpfIMAQPitched, gpfReferencePerpendicular, nActualProcessNumberLines, gnRawLineLength);
            break;
        }
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaDeviceSynchronize());

        /* calibration */
        calculateSpectralDomainCalibration(1);  // nMode = 1: PS SD-OCT
            // gpfProcessK, gpnProcessIndex, gpnProcessAssigned

        // output calibration parameters
        switch (nCam) {
        case 0: // parallel camera
            gpuErrchk(cudaMemcpy2D(gpfProcessKParallel, gnKPitchParallel, gpfProcessK, gnKPitch, gnKPitch, nActualProcessNumberLines, cudaMemcpyDeviceToDevice));   // QUESTION: ok if only copy one line?
            gpuErrchk(cudaMemcpy2D(gpnProcessIndexParallel, gnIndexPitchParallel, gpnProcessIndex, gnIndexPitch, gnIndexPitch, nActualProcessNumberLines, cudaMemcpyDeviceToDevice));
            break; 
        case 1: // perpendicular camera
            gpuErrchk(cudaMemcpy2D(gpfProcessKPerpendicular, gnKPitchPerpendicular, gpfProcessK, gnKPitch, gnKPitch, nActualProcessNumberLines, cudaMemcpyDeviceToDevice));   // QUESTION: ok if only copy one line?
            gpuErrchk(cudaMemcpy2D(gpnProcessIndexPerpendicular, gnIndexPitchPerpendicular, gpnProcessIndex, gnIndexPitch, gnIndexPitch, nActualProcessNumberLines, cudaMemcpyDeviceToDevice));
        } // switch (nCam)
        gpuErrchk(cudaDeviceSynchronize());

    } // for (int nCam = 0; nCam < 2; nCam++) 

    // copy data to host for output 
    gpuErrchk(cudaMemcpy(pfKParallel, gpfProcessKParallel, gnRawLineLength * nActualProcessNumberLines * sizeof(float), cudaMemcpyDeviceToHost)); 
    gpuErrchk(cudaMemcpy(pfKPerpendicular, gpfProcessKPerpendicular, gnRawLineLength * nActualProcessNumberLines * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(pnIndexParallel, gpnProcessIndexParallel, gnRawLineLength * nActualProcessNumberLines * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(pnIndexPerpendicular, gpnProcessIndexPerpendicular, gnRawLineLength * nActualProcessNumberLines * sizeof(int), cudaMemcpyDeviceToHost)); 

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

    // loop through cameras
    for (int nCam = 0; nCam < 2; nCam++) {  // nCam = 0: parallel camera; nCam = 1: perpendicular camera
        int nNumberLinesPerChunk = gnProcessNumberLines;    // value set in C# UI
        int nNumberChunks = (gnRawNumberLines - 1) / gnProcessNumberLines + 1;  // QUESTION: need to double check. why previous method?

        // loop through chunks
        for (int nChunk = 0; nChunk < nNumberChunks; nChunk++) { 

            // loop through even and odd lines, respectively
            int nSrcPtrOffset = nChunk * (gnRawLineLength * nNumberLinesPerChunk); 
            for (int nOddEven = 0; nOddEven < 2; nOddEven++) { // nOddEven = 0: process even lines; nOddEven = 1; process odd lines
                // copy a chunk: in each data array (on device now), copy every other line 
                switch (nCam) {
                case 0: // parallel camera
                    switch (nOddEven) {
                    case 0: // even lines
                        gpuErrchk(cudaMemcpy2D(gpfIMAQPitched, gnIMAQPitch, d_gpfRawIMAQParallel + nSrcPtrOffset, 2 * gnIMAQPitch, gnIMAQPitch, nNumberLinesPerChunk >> 1, cudaMemcpyDeviceToDevice));
                        break;
                    case 1: // odd lines
                        gpuErrchk(cudaMemcpy2D(gpfIMAQPitched, gnIMAQPitch, d_gpfRawIMAQParallel + gnRawLineLength + nSrcPtrOffset, 2 * gnIMAQPitch, gnIMAQPitch, nNumberLinesPerChunk >> 1, cudaMemcpyDeviceToDevice));
                        break;
                    } // switch (nOddEven)
                    break;
                case 1: // perpendicular camera
                    switch (nOddEven) {
                    case 0: // even lines 
                        gpuErrchk(cudaMemcpy2D(gpfIMAQPitched, gnIMAQPitch, d_gpfRawIMAQPerpendicular + nSrcPtrOffset, 2 * gnIMAQPitch, gnIMAQPitch, nNumberLinesPerChunk >> 1, cudaMemcpyDeviceToDevice));
                        break;
                    case 1: // odd lines
                        gpuErrchk(cudaMemcpy2D(gpfIMAQPitched, gnIMAQPitch, d_gpfRawIMAQPerpendicular + gnRawLineLength + nSrcPtrOffset, 2 * gnIMAQPitch, gnIMAQPitch, nNumberLinesPerChunk >> 1, cudaMemcpyDeviceToDevice));
                        break;
                    } // switch (nOddEven)
                    break; 
                }
                gpuErrchk(cudaDeviceSynchronize()); 

                /* reference */ 
                if (gbIsReferenceRecorded == false) { // no reference data recorded
                    // calculate reference 
                    d3Threads.x = 128;
                    d3Threads.y = 1024 / d3Threads.x;
                    d3Threads.z = 1;
                    d3Blocks.x = nNumberLinesPerChunk / d3Threads.x;
                    d3Blocks.y = 1;
                    d3Blocks.z = 1;

                        // different cameras
                    switch (nCam) {
                    case 0: // parallel camera
                        calculateMean << <d3Blocks, d3Threads >> > (gpfIMAQPitched, gpfReferenceParallel, nNumberLinesPerChunk >> 1, gnRawLineLength);
                        break;
                    case 1: // perpendicular camera
                        calculateMean << <d3Blocks, d3Threads >> > (gpfIMAQPitched, gpfReferencePerpendicular, nNumberLinesPerChunk >> 1, gnRawLineLength);
                        break; 
                    }
                    gpuErrchk(cudaPeekAtLastError()); 
                } // if (gbIsReferenceRecorded == false)
                
                // subtract reference 
                d3Threads.x = 32;
                d3Threads.y = 1024 / d3Threads.x;
                d3Threads.z = 1;
                d3Blocks.x = nNumberLinesPerChunk / d3Threads.x;
                d3Blocks.y = 1;
                d3Blocks.z = 1;
                    // different cameras
                switch (nCam) {
                case 0: // parallel camera
                    subtractMean << <d3Blocks, d3Threads >> > (gpfIMAQPitched, gpfReferenceParallel, nNumberLinesPerChunk >> 1, gnRawLineLength);
                    break;
                case 1: // perpendicular camera
                    subtractMean << <d3Blocks, d3Threads >> > (gpfIMAQPitched, gpfReferencePerpendicular, nNumberLinesPerChunk >> 1, gnRawLineLength);
                    break;
                }
                // now gpfIMAQPitched is the result of reference subtraction  (fringes), SAME AS gpfProcessOCT

                gpuErrchk(cudaPeekAtLastError()); 
                gpuErrchk(cudaDeviceSynchronize()); 

                /* calibration */
                if (gbIsCalibrationLoaded == false) {
                    calculateSpectralDomainCalibration(1);  // nMode = 1: PS SD-OCT
                        // gpfProcessK, gpnProcessIndex, gpnProcessAssigned
                } 

                /* interpolation */
                d3Threads.x = 32;
                d3Threads.y = 1024 / d3Threads.x;
                d3Threads.z = 1;
                d3Blocks.x = gnRawLineLength / d3Threads.x;
                d3Blocks.y = (nNumberLinesPerChunk >> 1) / d3Threads.y;
                d3Blocks.z = 1;
                interpCubicSpline << <d3Blocks, d3Threads >> > (gpfProcessK, gpnProcessIndex, gpfIMAQPitched, gpfProcessSpectrumK, nNumberLinesPerChunk >> 1, gnRawLineLength);
                gpuErrchk(cudaDeviceSynchronize());

                /* forward fft */
                gpuErrchk(cudaMemset2D(gpcProcessDepthProfile, gnProcessDepthProfilePitch, 0.0, gnProcessDepthProfilePitch, nNumberLinesPerChunk >> 1));
                gpuErrchk(cudaDeviceSynchronize());

                cufftExecR2C(gchForward, gpfProcessSpectrumK, gpcProcessDepthProfile);

                /* mask */
                    // calculate mask
                nThreadsPerBlock = 512;
                calculateMask << <gnRawLineLength / nThreadsPerBlock, nThreadsPerBlock >> > (gpfDispersionMask, gnRawLineLength, 50, 100, 16); // need to get values from C# UI

                    // apply mask
                d3Threads.x = 32;
                d3Threads.y = 1024 / d3Threads.x;
                d3Threads.z = 1;
                d3Blocks.x = nNumberLinesPerChunk / d3Threads.x;
                d3Blocks.y = 1;
                d3Blocks.z = 1;
                applyMask << <d3Blocks, d3Threads >> > (gpcProcessDepthProfile, gpfDispersionMask, nNumberLinesPerChunk >> 1, gnRawLineLength);
                gpuErrchk(cudaPeekAtLastError());
                
                /* reverse fft */
                cufftExecC2C(gchReverse, gpcProcessDepthProfile, gpcProcessSpectrum, CUFFT_INVERSE);

                /* calculate phase */
                d3Threads.x = 32;
                d3Threads.y = 1024 / d3Threads.x;
                d3Threads.z = 1;
                d3Blocks.x = gnRawLineLength / d3Threads.x;
                d3Blocks.y = (nNumberLinesPerChunk >> 1) / d3Threads.y;
                d3Blocks.z = 1;
                calculatePhase << <d3Blocks, d3Threads >> > (gpcProcessSpectrum, gpfProcessPhase, nNumberLinesPerChunk >> 1, gnRawLineLength);
                gpuErrchk(cudaPeekAtLastError());
                
                /* dispersion correction */
                    // calculate dispersion correction
                nThreadsPerBlock = 512;
                calculateDispersionCorrection << <gnRawLineLength / nThreadsPerBlock, nThreadsPerBlock >> > (gpfProcessPhase, gpcDispersionCorrection);

                    // apply correction
                d3Threads.x = 32;
                d3Threads.y = 1024 / d3Threads.x;
                d3Threads.z = 1;
                d3Blocks.x = nNumberLinesPerChunk / d3Threads.x;
                d3Blocks.y = 1;
                d3Blocks.z = 1;
                applyDispersionCorrection << <d3Blocks, d3Threads >> > (gpfProcessSpectrumK, gpcDispersionCorrection, gpcProcessKCorrected, nNumberLinesPerChunk >> 1, gnRawLineLength);
                gpuErrchk(cudaPeekAtLastError());
                
                /* forward fft */
                cufftExecC2C(gchForwardComplex, gpcProcessKCorrected, gpcProcessDepthProfile, CUFFT_FORWARD);
                gpuErrchk(cudaDeviceSynchronize());

                // copy to results array
                switch (nCam) {
                case 0: // parallel camera
                    switch (nOddEven) {
                    case 0: // even lines

                        break; 
                    case 1: // odd lines

                        break; 
                    }
                    break; 
                case 1: // perpendicular camera
                    switch (nOddEven) {
                    case 0: // even lines

                        break;
                    case 1: // odd lines

                        break;
                    }
                    break; 
                }


            } // for (int nOddEven = 0; nOddEven < 2; nOddEven++)
             


        } // for (int nChunk = 0; nChunk < nNumberChunks; nChunk++)


    } // for (int nCam = 0; nCam < 2; nCam++)




    return -1; 
}
