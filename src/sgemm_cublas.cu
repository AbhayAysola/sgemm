#include "cublas_v2.h"
#include <iostream>

void sgemm(float *M, float *N, float *P, int m, int n, int k) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, k, m, n, &alpha, N, k, M, n,
              &beta, P, k);
  cublasDestroy(handle);
}

void init_matrix(float *M, int m, int n) {
  for (int i = 0; i < m * n; i++)
    M[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Usage: " << std::endl;
    std::cout << argv[0] << " M N K" << std::endl;
    return 1;
  }
  int m, n, k;
  m = atoi(argv[1]);
  n = atoi(argv[2]);
  k = atoi(argv[3]);

  float *M_h = (float *)malloc(sizeof(float) * m * n);
  float *N_h = (float *)malloc(sizeof(float) * n * k);
  float *P_h = (float *)malloc(sizeof(float) * m * k);

  init_matrix(M_h, m, n);
  init_matrix(N_h, n, k);

  float *M_d;
  float *N_d;
  float *P_d;

  cudaMalloc((void **)&M_d, sizeof(float) * m * n);
  cudaMalloc((void **)&N_d, sizeof(float) * n * k);
  cudaMalloc((void **)&P_d, sizeof(float) * m * k);

  cudaMemcpy(M_d, M_h, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N_h, sizeof(float) * n * k, cudaMemcpyHostToDevice);

  const int iterations = 50;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  for (int i = 0; i < iterations; i++)
    sgemm(M_d, N_d, P_d, m, n, k);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms;
  cudaEventElapsedTime(&ms, start, end);
  float avg_time_ms = ms / iterations;

  double flops = 2.0 * m * n * k;
  double gflops = (flops * 1e-9) / (avg_time_ms * 1e-3);

  std::cout << m << ',' << n << ',' << k << ',' << avg_time_ms << ',' << gflops
            << std::endl;

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
  free(M_h);
  free(N_h);
  free(P_h);

  return 0;
}
