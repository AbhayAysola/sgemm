import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

kernel = input("kernel: ")

BINARY_PATH = f"./build/sgemm_{kernel}"
CUBLAS_BINARY_PATH = "./build/sgemm_cublas"
SIZES = [64, 128, 256, 512, 1024, 2048, 4096]


results = []
print(f"{'Size (N)':<10} | {'Time (ms)':<10} | {'GFLOPS':<10}")
print("-" * 35)

for n in SIZES:
    # Run the compiled CUDA binary
    cmd = [BINARY_PATH, str(n), str(n), str(n)]
    output = subprocess.check_output(cmd).decode("utf-8").strip()
    
    # Parse CSV output
    m, n_val, k, time_ms, gflops = map(float, output.split(','))
    results.append({"Size": int(n_val), "Time_ms": time_ms, "GFLOPS": gflops})
    print(f"{int(n_val):<10} | {time_ms:<10.4f} | {gflops:<10.2f}")

df = pd.DataFrame(results)

cmd = [CUBLAS_BINARY_PATH, str(4096), str(4096), str(4096)]
output = subprocess.check_output(cmd).decode("utf-8").strip()
m, n_val, k, time_ms, gflops = map(float, output.split(','))
print(f"CuBLAS | {time_ms:<10.4f} | {gflops:<10.2f}")
print(f"{100*results[-1]["GFLOPS"]/gflops}")

plt.figure(figsize=(10, 5))

# Plot GFLOPS (Higher is better)
plt.subplot(1, 2, 1)
plt.plot(df['Size'], df['GFLOPS'], marker='o', color='royalblue', linewidth=2)
plt.title('SGEMM Performance')
plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('GFLOPS')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('sgemm_benchmark.png')
plt.show()
