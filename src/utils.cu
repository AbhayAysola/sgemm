#include <iterator>
void init_matrix(float *M, int m, int n) {
  for (int i = 0; i < m * n; i++)
    M[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void flush_cache() {
  int dev_id;
  int m_12_size;
  void *buffer;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&m_12_size, cudaDevAttrL2CacheSize, dev_id);
  if (m_12_size > 0) {
    cudaMalloc(&buffer, static_cast<std::size_t>(m_12_size));
    int *m_12_buffer = reinterpret_cast<int *>(buffer);
    cudaMemsetAsync(m_12_buffer, 0, static_cast<std::size_t>(m_12_size));
    cudaFree(m_12_buffer);
  }
}
