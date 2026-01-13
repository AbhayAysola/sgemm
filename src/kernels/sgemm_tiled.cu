#include "./include/sgemm.cuh"

// multiplies two matrices M (mxn) and N (nxk) and stores the result in P (mxk)
// one thread per output element
__global__ void sgemm_kernel(float *M, float *N, float *P, int m, int n, int k,
                             float alpha, float beta) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockDim.y * by + ty;
  int col = blockDim.x * bx + tx;

  __shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];
  float p_value = 0;
  int tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int tile = 0; tile < tiles; tile++) {
    // loading
    int m_col = tile * BLOCK_SIZE + tx;
    int n_row = tile * BLOCK_SIZE + ty;
    if (m_col < n and row < m)
      Ms[ty][tx] = M[row * n + m_col];
    else
      Ms[ty][tx] = 0;

    if (n_row < n and col < k)
      Ns[ty][tx] = N[n_row * k + col];
    else
      Ns[ty][tx] = 0;

    __syncthreads();
    // computing
    for (int i = 0; i < BLOCK_SIZE; i++) {
      p_value += Ms[ty][i] * Ns[i][tx];
    }
    __syncthreads();
  }
  if (row < m and col < k)
    P[row * k + col] = p_value;
}
