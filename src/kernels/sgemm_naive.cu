#include "./include/sgemm.cuh"

// multiplies two matrices M (mxn) and N (nxk) and stores the result in P (mxk)
// one thread per output element
__global__ void sgemm_kernel(float *M, float *N, float *P, int m, int n, int k,
                             float alpha, float beta) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row < m and col < k) {
    float p_value = 0;
    for (int i = 0; i < n; i++) {
      p_value += M[row * n + i] * N[i * k + col];
    }
    P[row * k + col] *= beta;
    P[row * k + col] += alpha * p_value;
  }
}
