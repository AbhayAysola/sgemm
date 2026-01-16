#include "./include/sgemm.cuh"

const int BLOCK_SIZE = 64;
const int TILE_SIZE = 4;

__host__ __device__ int cdiv(int a, int b) { return (a + b - 1) / b; }

// multiplies two matrices M (mxn) and N (nxk) and stores the result in P (mxk)
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

  float M_regs[TILE_SIZE];
  float N_regs[TILE_SIZE];
  float results_regs[TILE_SIZE][TILE_SIZE] = {0.0f};

  int tiles = cdiv(n, BLOCK_SIZE);
  for (int tile = 0; tile < tiles; tile++) {
    // cooperative load
    for (int i = 0; i < TILE_SIZE; i++) {
      for (int j = 0; j < TILE_SIZE; j++) {
        int s_row = ty * TILE_SIZE + i;
        int s_col = tx * TILE_SIZE + j;

        int m_row = row * TILE_SIZE + i;
        int m_col = tile * BLOCK_SIZE + tx * TILE_SIZE + j;
        if (m_row < m and m_col < n)
          Ms[s_col][s_row] = M[m_row * n + m_col];
        else
          Ms[s_col][s_row] = 0.0f;

        int n_row = tile * BLOCK_SIZE + ty * TILE_SIZE + i;
        int n_col = col * TILE_SIZE + j;
        if (n_row < n and n_col < k)
          Ns[s_row][s_col] = N[n_row * k + n_col];
        else
          Ns[s_row][s_col] = 0.0f;
      }
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
      for (int i = 0; i < TILE_SIZE; i++) {
        M_regs[i] = Ms[k][ty * TILE_SIZE + i];
        N_regs[i] = Ns[k][tx * TILE_SIZE + i];
      }

      for (int res_i = 0; res_i < TILE_SIZE; res_i++) {
        for (int res_j = 0; res_j < TILE_SIZE; res_j++) {
          results_regs[res_i][res_j] += M_regs[res_i] * N_regs[res_j];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < TILE_SIZE; i++) {
    for (int j = 0; j < TILE_SIZE; j++) {
      int p_row = row * TILE_SIZE + i;
      int p_col = col * TILE_SIZE + j;
      if (p_row < m and p_col < k)
        P[p_row * k + p_col] = results_regs[i][j];
    }
  }
}

void sgemm(float *M_d, float *N_d, float *P_d, int m, int n, int k, float alpha,
           float beta) {
  dim3 grid_dim(cdiv(k, BLOCK_SIZE), cdiv(m, BLOCK_SIZE), 1);
  dim3 block_dim(BLOCK_SIZE / TILE_SIZE, BLOCK_SIZE / TILE_SIZE, 1);

  sgemm_kernel<<<grid_dim, block_dim>>>(M_d, N_d, P_d, m, n, k, 1.0, 0.0);
}
