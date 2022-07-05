/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"

/*
 * Device code
 */
void __global__ init(double const * const A,
                         double * const B,
                         int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] = A[i];
    }
}


/* CUBLAS code 
static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}
*/
/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *A, *B;
    mxGPUArray *C;
    const double *d_A, *d_B;
    const double **d_Aarr, **d_Barr;
    double *d_C;
    double const alpha = 1, beta = 0;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* we expect 2 double (later uint16) gpu Arrays and 2 uint64 offset arrays */

    if (nrhs!=2) {
        mexErrMsgIdAndTxt("parallel:gpu:pagemtimes:WrongNumberOfInputs", "Expected 2 inputs.");
    } else if( !mxIsGPUArray(prhs[0]) || !mxIsGPUArray(prhs[1]) ) {
        mexErrMsgIdAndTxt("parallel:gpu:pagemtimes:WrongInputType", "Expected arguments 1 and 2 to be gpuArray types.");
    }

    // get the actual matrix gpu reference
    A = mxGPUCreateFromMxArray(prhs[0]); // first matrix 
    B = mxGPUCreateFromMxArray(prhs[1]); // second matrix

    const mwSize nDimsA = mxGPUGetNumberOfDimensions(A);
    const mwSize * dimsA = mxGPUGetDimensions(A);
    const mwSize nDimsB = mxGPUGetNumberOfDimensions(B);
    const mwSize * dimsB = mxGPUGetDimensions(B);
    /* Throw an error if the input is not a GPU array. */
    if(nDimsA < 2){
        mxGPUDestroyGPUArray(A); // cleanup
        mxGPUDestroyGPUArray(B); // cleanup
        mexErrMsgIdAndTxt("parallel:gpu:pagemtimes:EmptyInput", "Expected input 1 to have 2 or more dimensions.");
    } else if(nDimsB < 2){
        mxGPUDestroyGPUArray(A); // cleanup
        mxGPUDestroyGPUArray(B); // cleanup
        mexErrMsgIdAndTxt("parallel:gpu:pagemtimes:EmptyInput", "Expected input 2 to have 2 or more dimensions.");
    } else if(dimsA[1]!= dimsB[0]){
        mxGPUDestroyGPUArray(A); // cleanup
        mxGPUDestroyGPUArray(B); // cleanup
        mexErrMsgIdAndTxt("parallel:gpu:pagemtimes:MatrixMultiplicationDimensions", "The size of A in dimension 2 must match the size of B in dimension 1.");
    }

    // matrix array size
    const int M = dimsA[0], K = dimsA[1], N = dimsB[1];

    /*
     * Verify that A really is a double array before extracting the pointer.
     % I think this is unnecessary?
     */
    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS || mxGPUGetClassID(B) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt("parallel:gpu:pagemtimes:WrongInputType", "Expected underlying type to be double.");
        mxGPUDestroyGPUArray(A); // cleanup
        mxGPUDestroyGPUArray(B); // cleanup
    }
    d_A = (const double *)(mxGPUGetDataReadOnly(A));
    d_B = (const double *)(mxGPUGetDataReadOnly(B));
    
    /* 
     * Inputs 3 and 4 are the strides for A and B: we trust this blindly for now
     * Input 5 is the number of dimensions.
    */
    

    // get the output matrix size
    const mwSize nDimsC = max(nDimsA, nDimsB); // number of output dimensions
    mwSize * dimsC = (mwSize *)mxMalloc(nDimsC * sizeof(nDimsC)); // array for each dimension size
    dimsC[0] = M;
    dimsC[1] = N;
    for(int d = 2; d < nDimsC; ++d) // for dims 3+
        dimsC[d] = max((d < nDimsA ? dimsA[d] : 1), (d < nDimsB ? dimsB[d] : 1)); // new size is max of either size

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    C = mxGPUCreateGPUArray(nDimsC,
                            dimsC,
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    const size_t Csz = (size_t) mxGPUGetNumberOfElements(C);
    d_C = (double *)(mxGPUGetData(C)); // point to device data
    
    // make sure that the number of output strides matches the size of the data that we computed
    const size_t L = Csz / M / N; // number of strides we need to find

    /* we need to generate a set of pointers that point to the 
       location of the data for each stride, 
       while broadcasting over dimensions */
    d_Aarr = (const double **) mxMalloc(L * sizeof(d_Aarr));
    for(int i = 0; i < L; ++i){
        size_t szA = 1, szC = 1; // size so far
        d_Aarr[i] = d_A; // initial pointer locations
        for(int d = 2; d < nDimsA; ++d){ // for each upper dim
            const size_t ind = (dimsA[d] == 1) ? 0 : ((i / szC) % dimsC[d]); // index for this dim
            d_Aarr[i] += (ind*M*K*szA); // increment pointer
            szA *= dimsA[d]; // increment stride size
            szC *= dimsC[d]; // increment stride size
        }
    }

    d_Barr = (const double **) mxMalloc(L * sizeof(d_Barr));
    for(int i = 0; i < L; ++i){
        size_t szB = 1, szC = 1; // size so far
        d_Barr[i] = d_B; // initial pointer locations
        for(int d = 2; d < nDimsB; ++d){ // for each upper dim
            const size_t ind = (dimsB[d] == 1) ? 0 : ((i / szC) % dimsC[d]); // index for this dim
            d_Barr[i] += (ind*K*N*szB); // increment pointer
            szB *= dimsB[d]; // increment stride size
            szC *= dimsC[d]; // increment stride size
        }
    }

    /* code to call CUBLAS */
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasOperation_t trans = CUBLAS_OP_N; // ONE OF CUBLAS_OP_{N,T,C} for none/trans/ctrans
    stat = cublasCreate(&handle);
    if (stat == CUBLAS_STATUS_SUCCESS) { // we succeeded: call gemm for basic matrix x matrx multiply
        for(int i = 0; i < L; ++i) // for each output index
            stat = cublasDgemm(handle, trans, trans, M,N,K, &alpha,
                           d_Aarr[i], M, d_Barr[i], K, &beta, d_C + i*M*N, M); // call directly
    }

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(C);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    cublasDestroy(handle); // cleanup
    mxGPUDestroyGPUArray(A); // cleanup
    mxGPUDestroyGPUArray(B); // cleanup
    mxGPUDestroyGPUArray(C); // cleanup

    if(stat != CUBLAS_STATUS_SUCCESS){
        mexErrMsgIdAndTxt("parallel:gpu:pagemtimes:failure", "Failed to call the CUDA kernels");
    }
}
