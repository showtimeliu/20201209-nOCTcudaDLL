#include "nOCTcudaDLLHeader.cuh"

__global__ void convert2Float(short* pnIMAQ, float* pfIMAQ, int nSize) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x; 
	if (tid < nSize)
		pfIMAQ[tid] = (float)pnIMAQ[tid]; 
}

__global__ void calculateMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength) {
    __shared__ float pfSum[1024];

    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fSum = 0.0;
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        fSum += pfMatrix[nPosition];
        nPosition += nLineLength;
    }   // for (int nLine
    pfSum[threadIdx.x * blockDim.y + threadIdx.y] = fSum;

    __syncthreads();
    if (threadIdx.y == 0) {
        fSum = 0;
        nPosition = threadIdx.x * blockDim.y;
        for (nLine = 0; nLine < blockDim.y; nLine++) {
            fSum += pfSum[nPosition];
            nPosition++;
        }   // for (nLine
        pfMean[nPoint] = fSum / nNumberLines;
    }
}   // void calculateMean

__global__ void subtractMean(float* pfMatrix, float* pfMean, int nNumberLines, int nLineLength) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fMean = pfMean[nPoint];
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        pfMatrix[nPosition] -= fMean;
        nPosition += nLineLength;
    }   // for (int nLine
}   // void subtractMean

__global__ void calculateMask(float* pfMask, int nLength, int nStart, int nStop, int nRound) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    pfMask[nPoint] = 0.0;
    if (nPoint < nLength) {
        if (nPoint >= nStart - nRound)
            if (nPoint < nStart)
                pfMask[nPoint] = sin(0.5 * nPoint);
            else
                if (nPoint < nStop)
                    pfMask[nPoint] = 1.0;
                else
                    if (nPoint < nStop + nRound)
                        pfMask[nPoint] = sin(0.5 * nPoint);
    }   // if (nPoint
}   // void calculateMask

__global__ void applyMask(cufftComplex* pcMatrix, float* pfMask, int nNumberLines, int nLineLength) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    float fMask = pfMask[nPoint];
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        pcMatrix[nPosition].x *= fMask;
        pcMatrix[nPosition].y *= fMask;
        nPosition += nLineLength;
    }   // for (int nLine
}   // void subtractMean


__global__ void calculatePhase(cufftComplex* pcMatrix, float* pfPhase, int nNumberLines, int nLineLength) {
    int nPosition = (blockIdx.y * blockDim.y + threadIdx.y) * nLineLength + (blockIdx.x * blockDim.x + threadIdx.x);
    pfPhase[nPosition] = atan2(pcMatrix[nPosition].y, pcMatrix[nPosition].x);
}   // void calculatePhase


__global__ void unwrapPhase(float* pfPhase, int nNumberLines, int nLineLength, float fPiEps, float f2Pi) {
    __shared__ float pfUnwrappedEnds[2048];
    __shared__ int pn2pi[1024];

    int nLineNumber = blockIdx.x * blockDim.y + threadIdx.y;
    int nNumberPoints = nLineLength / blockDim.x;
    int nStartPoint = nLineNumber * nLineLength + threadIdx.x * nNumberPoints;
    int nStopPoint = nStartPoint + nNumberPoints;

    pfUnwrappedEnds[2 * (threadIdx.y * blockDim.x + threadIdx.x)] = pfPhase[nStartPoint];
    int nPoint = nStartPoint;
    float fOldPhase = pfPhase[nPoint];
    float fNewPhase;
    float fDeltaPhase;
    int n2Pi = 0;
    nPoint++;
    while (nPoint < nStopPoint) {
        fNewPhase = pfPhase[nPoint];
        fDeltaPhase = fNewPhase - fOldPhase;
        fOldPhase = fNewPhase;

        if (fDeltaPhase < -fPiEps)
            n2Pi++;
        if (fDeltaPhase > fPiEps)
            n2Pi--;

        pfPhase[nPoint] = fNewPhase + n2Pi * f2Pi;
        nPoint++;
    }   // while (nPoint
    nPoint--;
    pfUnwrappedEnds[2 * (threadIdx.y * blockDim.x + threadIdx.x) + 1] = pfPhase[nPoint];

    __syncthreads();

    if (threadIdx.x == 0) {
        int nSection = threadIdx.y * blockDim.x;
        int nEnd = 2 * nSection + 1;
        int nStart = nEnd + 1;
        pn2pi[nSection] = 0;
        for (nPoint = 1; nPoint < blockDim.y; nPoint++) {
            fDeltaPhase = pfUnwrappedEnds[nStart] - pfUnwrappedEnds[nEnd];
            pn2pi[nSection + 1] = pn2pi[nSection];
            nStart += 2;
            nEnd += 2;
            nSection++;
            if (fDeltaPhase < -fPiEps)
                pn2pi[nSection]++;
            if (fDeltaPhase > fPiEps)
                pn2pi[nSection]--;
        }   // for (nPoint
    }   // if (threadIdx.x

    __syncthreads();

    fDeltaPhase = f2Pi * (pn2pi[threadIdx.y * blockDim.x + threadIdx.x]);
    nPoint = nStartPoint + 1;
    while (nPoint < nStopPoint) {
        pfPhase[nPoint] += fDeltaPhase;
        nPoint++;
    }
}   // void unwrapPhase


__global__ void matchPhase(float* pfPhase, int nNumberLines, int nLineLength, float f2Pi) {
    __shared__ float pfOffset[1024];

    int nLineNumber = blockIdx.x * blockDim.y + threadIdx.y;
    int nNumberPoints = nLineLength / blockDim.x;
    int nStartPoint = nLineNumber * nLineLength + threadIdx.x * nNumberPoints;
    int nStopPoint = nStartPoint + nNumberPoints;

    if (threadIdx.x == 0)
        pfOffset[threadIdx.y] = f2Pi * roundf(pfPhase[nLineNumber * nLineLength + (nLineLength >> 1)] / f2Pi);

    __syncthreads();

    float fOffset = pfOffset[threadIdx.y];
    for (int nPoint = nStartPoint; nPoint < nStopPoint; nPoint++)
        pfPhase[nPoint] -= fOffset;
}   // void matchPhase


__global__ void getPhaseLimits(float* pfPhase, int nNumberLines, int nLineLength, int nLeft, int nRight, float* pfLeft, float* pfRight) {
    __shared__ float pfSum[1024];

    int nLinesInSection = nNumberLines / blockDim.x;
    int nStartingLine = threadIdx.x * nLinesInSection;
    int nPoint = nLeft;
    if (blockIdx.x == 1)
        nPoint = nRight;
    nPoint += nStartingLine * nLineLength;
    int nLine;

    float fSum = 0.0;
    for (nLine = 0; nLine < nLinesInSection; nLine++) {
        fSum += pfPhase[nPoint];
        nPoint += nLineLength;
    }   // for (int nLine
    pfSum[threadIdx.x] = fSum;

    __syncthreads();

    if (threadIdx.x == 0) {
        fSum = 0;
        for (nLine = 0; nLine < blockDim.x; nLine++)
            fSum += pfSum[nLine];
        if (blockIdx.x == 0)
            *pfLeft = fSum / nNumberLines;
        else
            *pfRight = fSum / nNumberLines;
    }   // if (threadIdx.x
}


__global__ void calculateK(float* pfPhase, float* pfK, int* pnIndex, int* pnAssigned, int nNumberLines, int nLineLength, float* pfLineParameters, int nLeft, int nRight, float* pfLeft, float* pfRight, int nMode) {

    // calculate slope and offset
    switch (nMode) {
    case 1:
        pfLineParameters[0] = (pfRight[0] - pfLeft[0]) / ((float)(nRight - nLeft));
        pfLineParameters[1] = -((nLineLength >> 1) + nLineLength) * pfLineParameters[0];
        break;
    case 2:
        break;
    }   // switch (nMode

    float fSlope = pfLineParameters[0];
    float fOffset = pfLineParameters[1];

    int nLine = blockIdx.x * blockDim.y + threadIdx.y;
    int nNumberPoints = nLineLength / blockDim.x;
    int nOffset1 = threadIdx.x * nNumberPoints;
    int nOffset2 = nLine * nLineLength;
    int nIndex = nOffset2 + nOffset1;
    int nX;
    for (int nPoint = 0; nPoint < nNumberPoints; nPoint++) {
        pfK[nIndex] = (pfPhase[nIndex] - fOffset) / fSlope;
        nX = ceilf(pfK[nIndex]) - nLineLength + nOffset1;
        if ((nX >= 0) && (nX < nLineLength)) {
            pnIndex[nOffset2 + nX] = nIndex - nOffset2;
            pnAssigned[nOffset2 + nX] = 1;
        }   //  if ((nX
        nIndex++;
    }   // for (int nPoint
}   // void calculateK


__global__ void cleanIndex(float* pfK, int* pnIndex, int* pnAssigned, int nNumberLines, int nLineLength) {
    int nLine = blockIdx.x * blockDim.y + threadIdx.y;
    int nNumberPoints = nLineLength / blockDim.x;
    int nLineOffset = nLine * nLineLength;
    int nPointOffset = threadIdx.x * nNumberPoints;

    // find first non-assigned element
    bool bKeepSearching = true;
    int nCurrentPoint = nLineOffset + nPointOffset;
    int nEndOfSection = nCurrentPoint + nNumberPoints;
    bKeepSearching = (pnAssigned[nCurrentPoint] == 0);
    while (bKeepSearching) {
        nCurrentPoint++;
        if (nCurrentPoint < nEndOfSection)
            bKeepSearching = (pnAssigned[nCurrentPoint] == 0);
        else
            bKeepSearching = false;
    }

    if (nCurrentPoint != nEndOfSection) {
        // if (thread == 0) track backwards
        if (threadIdx.x == 0) {
            int nBackwardPoint = nCurrentPoint - 1;
            while ((pfK[nBackwardPoint] > nLineLength) && (nBackwardPoint > (nLineOffset + 1)))
                nBackwardPoint--;
            int nSearchK = nBackwardPoint - 1 - nLineOffset;
            for (int nPoint = nLineOffset; nPoint < nCurrentPoint; nPoint++) {
                pnAssigned[nPoint] = 1;
                pnIndex[nPoint] = nSearchK;
            }   // for (int nPoint
        }   // if (threadIdx.x

        // once complete, track forward
        int nEndOfLine = nLineOffset + nLineLength;
        int nLastIndex = pnIndex[nCurrentPoint];
        bKeepSearching = true;
        while (bKeepSearching) {
            if (nCurrentPoint < nEndOfSection) {
                if (pnAssigned[nCurrentPoint] == 0)
                    pnIndex[nCurrentPoint] = nLastIndex;
                else {
                    nLastIndex = pnIndex[nCurrentPoint];
                    if (nLastIndex > nEndOfLine - 3) {
                        nLastIndex = nEndOfLine - 3;
                        pnIndex[nCurrentPoint] = nLastIndex;
                    }
                }
                nCurrentPoint++;
            }
            else {
                if (nCurrentPoint < nEndOfLine)
                    if (pnAssigned[nCurrentPoint] = 0) {
                        pnIndex[nCurrentPoint] = nLastIndex;
                        nCurrentPoint++;
                    }
                    else
                        bKeepSearching = false;
                else
                    bKeepSearching = false;
            }   // if (nCurrentPoint
        }   // while (bKeepSearching
    }   // if (nCurrentPoint

}


__global__ void interpCubicSpline(float* pfK, int* pnIndex, float* pfSpectrum, float* pfInterpSpectrum, int nNumberLines, int nLineLength) {
    int nPosition = (blockIdx.y * blockDim.y + threadIdx.y) * nLineLength + (blockIdx.x * blockDim.x + threadIdx.x);
    int nIndex = pnIndex[nPosition];
    float fk1_1 = pfK[nIndex];
    float fk1_2 = fk1_1 * fk1_1;
    float fk1_3 = fk1_2 * fk1_1;
    float fS1 = pfSpectrum[nIndex];
    nIndex++;
    float fk2_1 = pfK[nIndex];
    float fk2_2 = fk2_1 * fk2_1;
    float fk2_3 = fk2_2 * fk2_1;
    float fS2 = pfSpectrum[nIndex];
    nIndex++;
    float fk3_1 = pfK[nIndex];
    float fk3_2 = fk3_1 * fk3_1;
    float fk3_3 = fk3_2 * fk3_1;
    float fS3 = pfSpectrum[nIndex];
    nIndex++;
    float fk4_1 = pfK[nIndex];
    float fk4_2 = fk4_1 * fk4_1;
    float fk4_3 = fk4_2 * fk4_1;
    float fS4 = pfSpectrum[nIndex];

    float f0 = (fk1_3 * fk2_2 * fk3_1 + fk2_3 * fk3_2 * fk4_1 + fk3_3 * fk4_2 * fk1_1 + fk4_3 * fk1_2 * fk2_1) - (fk1_3 * fk4_2 * fk3_1 + fk2_3 * fk1_2 * fk4_1 + fk3_3 * fk2_2 * fk1_1 + fk4_3 * fk3_2 * fk2_1);
    float f1 = (fS1 * fk2_2 * fk3_1 + fS2 * fk3_2 * fk4_1 + fS3 * fk4_2 * fk1_1 + fS4 * fk1_2 * fk2_1) - (fS1 * fk4_2 * fk3_1 + fS2 * fk1_2 * fk4_1 + fS3 * fk2_2 * fk1_1 + fS4 * fk3_2 * fk2_1);
    float f2 = (fk1_3 * fS2 * fk3_1 + fk2_3 * fS3 * fk4_1 + fk3_3 * fS4 * fk1_1 + fk4_3 * fS1 * fk2_1) - (fk1_3 * fS4 * fk3_1 + fk2_3 * fS1 * fk4_1 + fk3_3 * fS2 * fk1_1 + fk4_3 * fS3 * fk2_1);
    float f3 = (fk1_3 * fk2_2 * fS3 + fk2_3 * fk3_2 * fS4 + fk3_3 * fk4_2 * fS1 + fk4_3 * fk1_2 * fS2) - (fk1_3 * fk4_2 * fS3 + fk2_3 * fk1_2 * fS4 + fk3_3 * fk2_2 * fS1 + fk4_3 * fk3_2 * fS2);
    float f4 = (fk1_3 * fk2_2 * fk3_1 * fS4 + fk2_3 * fk3_2 * fk4_1 * fS1 + fk3_3 * fk4_2 * fk1_1 * fS2 + fk4_3 * fk1_2 * fk2_1 * fS3) - (fk1_3 * fk4_2 * fk3_1 * fS2 + fk2_3 * fk1_2 * fk4_1 * fS3 + fk3_3 * fk2_2 * fk1_1 * fS4 + fk4_3 * fk3_2 * fk2_1 * fS1);

    float fK = (blockIdx.x * blockDim.x + threadIdx.x) + nLineLength;
    pfInterpSpectrum[nPosition] = (((f1 / f0) * fK + (f2 / f0)) * fK + (f3 / f0)) * fK + (f4 / f0);
}


__global__ void calculateDispersionCorrection(float* pfPhase, cufftComplex* pcCorrection) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    pcCorrection[nPoint].x = cosf(pfPhase[nPoint]);
    pcCorrection[nPoint].y = -sinf(pfPhase[nPoint]);
}


__global__ void applyDispersionCorrection(float* pfMatrix, cufftComplex* pcCorrection, cufftComplex* pcMatrix, int nNumberLines, int nLineLength) {
    int nPoint = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex cCorrection = pcCorrection[nPoint];
    float fOriginal;
    int nLine;
    int nNumber = nNumberLines / blockDim.y;
    int nPosition = threadIdx.y * nNumber * nLineLength + nPoint;
    for (nLine = 0; nLine < nNumber; nLine++) {
        fOriginal = pfMatrix[nPosition];
        pcMatrix[nPosition].x = fOriginal * cCorrection.x;
        pcMatrix[nPosition].y = fOriginal * cCorrection.y;
        nPosition += nLineLength;
    }   // for (int nLine
}   // void subtractMean

__global__ void combineCamera() {

}

__global__ void separateFFTRealImag(cufftComplex* pcMatrix, float* pfReal, float* pfImaginary, int nWidth, int nHeight) {
    int id = (blockIdx.y * blockDim.y + threadIdx.y) * (blockDim.x * gridDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
    if (id < (nWidth * nHeight)) {
        pfReal[id] = pcMatrix[id].x;
        pfImaginary[id] = pcMatrix[id].y;
    }
}
