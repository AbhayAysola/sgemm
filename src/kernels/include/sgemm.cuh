#ifndef __CUDAHEADER_CUH__
#define __CUDAHEADER_CUH__

const int BLOCK_SIZE = 32;
__global__ void sgemm_kernel(float *M_d, float *N_d, float *P_d, int m, int n,
                             int k, float alpha, float beta);
#endif
