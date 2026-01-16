import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob

def benchmark(path, sizes, output_path):
    results = []
    print(f"{'Size (N)':<10} | {'Time (ms)':<10} | {'GFLOPS':<10}")
    print("-" * 35)

    for n in sizes:
        # Run the compiled CUDA binary
        cmd = [path, str(n), str(n), str(n)]
        output = subprocess.check_output(cmd).decode("utf-8").strip()

        # Parse CSV output
        m, n, k, time_ms, gflops = map(float, output.split(','))
        results.append({"Size": int(n), "Time_ms": time_ms, "GFLOPS": gflops})
        print(f"{int(n):<10} | {time_ms:<10.4f} | {gflops:<10.2f}")

    df = pd.DataFrame(results)
    df.to_csv(output_path)
    return df

CACHE_DIR = "./.cache"
os.makedirs(CACHE_DIR)
BUILD_DIR = "./build"

CUBLAS_BINARY_PATH = BUILD_DIR + "/sgemm_cublas"
CUBLAS_OUTPUT_PATH = CACHE_DIR + "/sgemm_cublas.csv"
SIZES = [128, 512, 1024, 2048, 4096]

for kernel in sys.argv[1:]:
    if kernel == "cublas":
        cublas_df = benchmark(CUBLAS_BINARY_PATH, SIZES, CUBLAS_OUTPUT_PATH)
        continue
    benchmark(BUILD_DIR + f"/sgemm_{kernel}", SIZES, CACHE_DIR + f"/sgemm_{kernel}.csv")

if not os.path.exists(CUBLAS_OUTPUT_PATH):
    cublas_df = benchmark(CUBLAS_BINARY_PATH, SIZES, CUBLAS_OUTPUT_PATH)
else:
    cublas_df = pd.read_csv(CUBLAS_OUTPUT_PATH)


csv_files = glob.glob(os.path.join(CACHE_DIR, "*.csv"))
dfs = {} 
for f in csv_files:
    name = os.path.basename(f)
    df = pd.read_csv(f)
    if (name != CUBLAS_OUTPUT_PATH):
        dfs[name.replace(".csv", "")] = df

plt.figure(figsize=(15, 6))
# Plot GFLOPS (Higher is better)
plt.subplot(1, 2, 1)
for kernel, df_kernel in dfs.items():
    plt.plot(df_kernel['Size'], df_kernel['GFLOPS'], marker='o', label=kernel)

plt.plot(cublas_df['Size'], cublas_df['GFLOPS'], 'k--', marker='x', label='cuBLAS (Baseline)')
plt.title('SGEMM Absolute Performance')
plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('GFLOPS (Higher is better)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(bottom=0)

plt.subplot(1, 2, 2)
for kernel, df_kernel in dfs.items():
    # Calculate speedup relative to cublas (df_kernel / cublas_df)
    relative_perf = df_kernel['GFLOPS'].values / cublas_df['GFLOPS'].values
    plt.plot(df_kernel['Size'], relative_perf*100, marker='o', label=f'{kernel} / cuBLAS')

plt.title('Relative Performance to cuBLAS')
plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('Percentage of cuBLAS Performance')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()
