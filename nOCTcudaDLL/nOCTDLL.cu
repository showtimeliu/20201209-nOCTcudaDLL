
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
