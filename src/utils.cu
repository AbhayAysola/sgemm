void init_matrix(float *M, int m, int n) {
  for (int i = 0; i < m * n; i++)
    M[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void flush_cache() {}
