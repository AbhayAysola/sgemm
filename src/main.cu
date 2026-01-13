#include "./kernels/include/sgemm.cuh"
#include <iostream>

void init_matrix(float *M, int m, int n) {
  for (int i = 0; i < m * n; i++)
    M[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

float epsilon = 1e-4;

bool verify_cpu(float *M, float *N, float *P, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      double sum = 0.0f;
      for (int x = 0; x < n; x++) {
        sum += M[i * n + x] * N[x * k + j];
      }
      double absolute_diff = std::fabs((double)P[i * k + j] - sum);
      double relative_epsilon =
          epsilon * std::max((double)std::fabs(P[i * k + j]), std::fabs(sum));
      if (absolute_diff > epsilon && absolute_diff > relative_epsilon) {
        return false;
      }
    }
  }
  return true;
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
  bool verify = false;
  if (argc == 5) {
    std::string verify_flag = argv[4];
    if (verify_flag == "-v" or verify_flag == "--verify")
      verify = true;
  }

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

  dim3 grid_dim(cdiv(k, BLOCK_SIZE), cdiv(m, BLOCK_SIZE), 1);
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

  // warmup
  // TODO: increase number of warmups?
  sgemm_kernel<<<grid_dim, block_dim>>>(M_d, N_d, P_d, m, n, k, 1.0, 0.0);
  cudaDeviceSynchronize();

  cudaMemcpy(P_h, P_d, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
  if (verify) {
    if (!verify_cpu(M_h, N_h, P_h, m, n, k)) {
      std::cout << "Incorrect kernel!" << std::endl;
      return 1;
    }
  }
  const int iterations = 50;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);
  for (int i = 0; i < iterations; i++)
    sgemm_kernel<<<grid_dim, block_dim>>>(M_d, N_d, P_d, m, n, k, 1.0, 0.0);
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
