# Reference benchmark results

This directory retains the benchmark figures and raw measurements shared with the
companion VibeGEMM implementation.

Both datasets use FP16 square GEMM with `M = N = K = 8192` and report average
latency plus derived TFLOPS for cuBLAS and successive VibeGEMM kernel versions.

| Dataset | Versions | Peak | cuBLAS | Peak / cuBLAS |
|---|---:|---:|---:|---:|
| NVIDIA A100 | v0-v11 | 220.55 TFLOPS | 226.66 TFLOPS | 97.3% |
| NVIDIA H100 NVL | v0-v20 | 460.83 TFLOPS | 451.30 TFLOPS | 102.1% |

The `.txt` files are the source measurements for the `.png` figures. These are
retained historical results; the reconstructed placeholder pipeline does not claim
to reproduce them until real CUDA validation is connected.

