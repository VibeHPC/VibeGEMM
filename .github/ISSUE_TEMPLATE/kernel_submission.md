---
name: GEMM Kernel Submission
about: Submit a high-performance GEMM kernel
title: "[SUBMISSION] Kernel Name"
labels: submission
---

## Kernel Name
e.g., vibegemm_M1024_N1024_K1024_FP16FP16FP32FP16_H100_submission

## Author / Organization
e.g., VibeHPC

## GPU
e.g., H100, A100

## Precision
e.g., A/B = FP16, accumulate = FP32, C = FP16

## GEMM Shape
Format: M×N×K  
e.g., 1024×1024×1024

## Performance
- Latency:
- TFLOPS:
- % of cuBLAS:

## Source Code
Submission_Template: https://github.com/VibeHPC/VibeGEMM/blob/main/.github/ISSUE_TEMPLATE/GEMM_M1024_N1024_K1024_Submission_Template.cu

Put the link where the submitting code is stored: https://github.com/...

## Compile Command
```bash
e.g., nvcc -O3 -std=c++17 -arch=sm_80 GEMM_M1024_N1024_K1024_Submission_Template.cu -lcublas -o gemm_submission
```

## Run Command
```bash
./gemm_submission
```

## Notes
(optional)
