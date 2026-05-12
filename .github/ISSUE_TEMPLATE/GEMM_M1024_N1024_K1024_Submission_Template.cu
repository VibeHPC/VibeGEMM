/*****************************************************************************************
 * GEMM Submission Template
 *
 * Purpose:
 *   Compare a submitted/custom GEMM kernel against cuBLAS on:
 *     1) correctness: max absolute/relative error and pass/fail
 *     2) performance: average latency and TFLOPS
 *
 * Matrix layout:
 *   A uses row-major layout:    A[M,K], index A[row*K + k]
 *   B uses column-major layout: B[K,N], index B[k + col*K]
 *   C uses column-major layout: C[M,N], index C[row + col*M]
 *   C = A * B
 *
 * Note:
 *   This mixed layout is chosen to match the cuBLAS call in run_cublas_gemm():
 *   cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, ...).
 *
 * Data type:
 *   FP16 input/output, FP32 accumulation in cuBLAS and in the example custom kernel.
 *
 * Build example:
 *   nvcc -O3 -std=c++17 -arch=sm_80 GEMM_M1024_N1024_K1024_Submission_Template.cu -lcublas -o gemm_submission
 *   nvcc -O3 -std=c++17 -arch=sm_90a GEMM_M1024_N1024_K1024_Submission_Template.cu -lcublas -o gemm_submission
 *
 * Run examples:
 *   ./gemm_submission
 *   ./gemm_submission 1024 1024 1024
 *   ./gemm_submission 1024 1024 1024 10 30
 *
 * Where to put submitted code:
 *   Search for "SUBMITTER SECTION" below. Replace launch_custom_gemm() and simple_tiled_gemm_kernel()
 *   with your optimized implementation. Keep the function signature unchanged.
 *****************************************************************************************/

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ========================================================================================
// Error checking helpers
// ========================================================================================

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(err__));                                        \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                                  \
    do {                                                                                    \
        cublasStatus_t status__ = (call);                                                    \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                             \
            std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__,     \
                         static_cast<int>(status__));                                        \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (0)

static inline void check_last_cuda_error(const char* name) {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    (void)name;
}

// ========================================================================================
// Utility functions
// ========================================================================================

static inline double gemm_tflops(int M, int N, int K, double milliseconds) {
    // GEMM has approximately 2*M*N*K floating-point operations.
    return (2.0 * static_cast<double>(M) * N * K) / (milliseconds * 1.0e9);
}

static void fill_random_fp16(std::vector<half>& x, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (half& v : x) {
        v = __float2half(dist(rng));
    }
}

struct GpuTimer {
    cudaEvent_t start{};
    cudaEvent_t stop{};

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin() { CUDA_CHECK(cudaEventRecord(start)); }

    float end_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

struct CheckResult {
    bool passed = false;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    size_t max_abs_index = 0;
    size_t max_rel_index = 0;
    size_t mismatch_count = 0;
};

static CheckResult check_correctness(const half* d_ref,
                                     const half* d_test,
                                     size_t elements,
                                     float atol = 1e-2f, // absolute tolerance
                                     float rtol = 1e-2f, // relative tolerance
                                     size_t print_first_n = 5) {
    std::vector<half> h_ref(elements);
    std::vector<half> h_test(elements);

    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, elements * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_test.data(), d_test, elements * sizeof(half), cudaMemcpyDeviceToHost));

    CheckResult result;
    result.passed = true;

    size_t printed = 0;
    for (size_t i = 0; i < elements; ++i) {
        float ref = __half2float(h_ref[i]);
        float got = __half2float(h_test[i]);
        float abs_error = std::abs(ref - got);
        float rel_error = abs_error / std::max(std::abs(ref), 1e-6f);

        if (abs_error > result.max_abs_error) {
            result.max_abs_error = abs_error;
            result.max_abs_index = i;
        }
        if (rel_error > result.max_rel_error) {
            result.max_rel_error = rel_error;
            result.max_rel_index = i;
        }

        bool ok = abs_error <= (atol + rtol * std::abs(ref));
        if (!ok) {
            result.passed = false;
            result.mismatch_count++;
            if (printed < print_first_n) {
                std::printf("    mismatch[%zu]: ref=% .6f  test=% .6f  abs=% .6e  rel=% .6e\n",
                            i, ref, got, abs_error, rel_error);
                printed++;
            }
        }
    }

    return result;
}

// ========================================================================================
// cuBLAS reference implementation
// ========================================================================================

static void run_cublas_gemm(cublasHandle_t handle,
                            const half* d_A,
                            const half* d_B,
                            half* d_C,
                            int M,
                            int N,
                            int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS assumes column-major matrices. In this template:
    //   A is stored as row-major A[M,K]. The same memory can be viewed by cuBLAS
    //   as a column-major matrix with shape K x M, so CUBLAS_OP_T gives A[M,K].
    //   B is stored as column-major B[K,N].
    //   C is stored as column-major C[M,N].
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              M,                 // rows of C
                              N,                 // columns of C
                              K,                 // reduction dimension
                              &alpha,
                              d_A,
                              CUDA_R_16F,
                              K,                 // leading dimension of A viewed as K x M
                              d_B,
                              CUDA_R_16F,
                              K,                 // leading dimension of B
                              &beta,
                              d_C,
                              CUDA_R_16F,
                              M,                 // leading dimension of C
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

// ========================================================================================
// SUBMITTER SECTION: replace this simple kernel with an optimized implementation
// ========================================================================================

#ifndef CUSTOM_GEMM_TILE
#define CUSTOM_GEMM_TILE 16
#endif

__global__ void simple_tiled_gemm_kernel(const half* __restrict__ A,
                                         const half* __restrict__ B,
                                         half* __restrict__ C,
                                         int M,
                                         int N,
                                         int K) {
    constexpr int TILE = CUSTOM_GEMM_TILE;

    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        int a_col = k0 + tx;
        int b_row = k0 + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[static_cast<size_t>(row) * K + a_col]
                                            : __float2half(0.0f);
        // B is column-major: B[k, col] is stored at B[k + col*K].
        Bs[ty][tx] = (b_row < K && col < N) ? B[static_cast<size_t>(b_row) + static_cast<size_t>(col) * K]
                                            : __float2half(0.0f);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += __half2float(As[ty][kk]) * __half2float(Bs[kk][tx]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        // C is column-major: C[row, col] is stored at C[row + col*M].
        C[static_cast<size_t>(row) + static_cast<size_t>(col) * M] = __float2half(acc);
    }
}

// Keep this signature unchanged. The benchmark calls this function directly.
static void launch_custom_gemm(const half* d_A,
                               const half* d_B,
                               half* d_C,
                               int M,
                               int N,
                               int K) {
    constexpr int TILE = CUSTOM_GEMM_TILE;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    simple_tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

// ========================================================================================
// Benchmark wrapper
// ========================================================================================

enum class BackendKind { Cublas, Custom };

static float benchmark_backend(BackendKind backend,
                               cublasHandle_t handle,
                               const half* d_A,
                               const half* d_B,
                               half* d_C,
                               int M,
                               int N,
                               int K,
                               int warmup_iters,
                               int benchmark_iters) {
    CUDA_CHECK(cudaMemset(d_C, 0, static_cast<size_t>(M) * N * sizeof(half)));

    for (int i = 0; i < warmup_iters; ++i) {
        if (backend == BackendKind::Cublas) {
            run_cublas_gemm(handle, d_A, d_B, d_C, M, N, K);
        } else {
            launch_custom_gemm(d_A, d_B, d_C, M, N, K);
        }
    }
    check_last_cuda_error("warmup");

    GpuTimer timer;
    timer.begin();
    for (int i = 0; i < benchmark_iters; ++i) {
        if (backend == BackendKind::Cublas) {
            run_cublas_gemm(handle, d_A, d_B, d_C, M, N, K);
        } else {
            launch_custom_gemm(d_A, d_B, d_C, M, N, K);
        }
    }
    float total_ms = timer.end_ms();
    check_last_cuda_error("benchmark");

    return total_ms / static_cast<float>(benchmark_iters);
}

static void print_usage(const char* program) {
    std::printf("Usage:\n");
    std::printf("  %s [M N K] [warmup_iters benchmark_iters]\n", program);
    std::printf("\nDefaults:\n");
    std::printf("  M=N=K=1024, warmup_iters=10, benchmark_iters=30\n");
}

int main(int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int warmup_iters = 10;
    int benchmark_iters = 30;

    if (argc != 1 && argc != 4 && argc != 6) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc == 6) {
        warmup_iters = std::atoi(argv[4]);
        benchmark_iters = std::atoi(argv[5]);
    }
    if (M <= 0 || N <= 0 || K <= 0 || warmup_iters < 0 || benchmark_iters <= 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);
    std::printf("Problem: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Layout : A row-major, B column-major, C column-major; C[%d,%d] = A[%d,%d] * B[%d,%d]\n",
                M, N, M, K, K, N);
    std::printf("Iters  : warmup=%d, benchmark=%d\n\n", warmup_iters, benchmark_iters);

    const size_t elems_A = static_cast<size_t>(M) * K;
    const size_t elems_B = static_cast<size_t>(K) * N;
    const size_t elems_C = static_cast<size_t>(M) * N;

    std::vector<half> h_A(elems_A);
    std::vector<half> h_B(elems_B);
    fill_random_fp16(h_A, 42);
    fill_random_fp16(h_B, 77);
    // h_A is interpreted as row-major A[M,K].
    // h_B is interpreted as column-major B[K,N].

    half* d_A = nullptr;
    half* d_B = nullptr;
    half* d_C_cublas = nullptr;
    half* d_C_custom = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, elems_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, elems_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, elems_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_custom, elems_C * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), elems_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), elems_B * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    // Generate the reference result once with cuBLAS.
    CUDA_CHECK(cudaMemset(d_C_cublas, 0, elems_C * sizeof(half)));
    run_cublas_gemm(handle, d_A, d_B, d_C_cublas, M, N, K);
    check_last_cuda_error("cuBLAS reference");

    // Generate the custom result once for correctness checking.
    CUDA_CHECK(cudaMemset(d_C_custom, 0, elems_C * sizeof(half)));
    launch_custom_gemm(d_A, d_B, d_C_custom, M, N, K);
    check_last_cuda_error("custom correctness run");

    std::printf("Correctness compared with cuBLAS:\n");
    CheckResult check = check_correctness(d_C_cublas, d_C_custom, elems_C);
    std::printf("  max absolute error = %.6e at index %zu\n", check.max_abs_error, check.max_abs_index);
    std::printf("  max relative error = %.6e at index %zu\n", check.max_rel_error, check.max_rel_index);
    std::printf("  mismatches         = %zu\n", check.mismatch_count);
    std::printf("  result             = %s\n\n", check.passed ? "PASS" : "FAIL");

    float cublas_ms = benchmark_backend(BackendKind::Cublas,
                                        handle,
                                        d_A,
                                        d_B,
                                        d_C_cublas,
                                        M,
                                        N,
                                        K,
                                        warmup_iters,
                                        benchmark_iters);
    float custom_ms = benchmark_backend(BackendKind::Custom,
                                        handle,
                                        d_A,
                                        d_B,
                                        d_C_custom,
                                        M,
                                        N,
                                        K,
                                        warmup_iters,
                                        benchmark_iters);

    double cublas_tflops = gemm_tflops(M, N, K, cublas_ms);
    double custom_tflops = gemm_tflops(M, N, K, custom_ms);
    double speedup_vs_cublas = cublas_ms / custom_ms;

    std::printf("Performance:\n");
    std::printf("  %-16s %12s %12s\n", "Backend", "Latency(ms)", "TFLOPS");
    std::printf("  %-16s %12.4f %12.3f\n", "cuBLAS", cublas_ms, cublas_tflops);
    std::printf("  %-16s %12.4f %12.3f\n", "Custom", custom_ms, custom_tflops);
    std::printf("\n");
    std::printf("Custom/cuBLAS TFLOPS ratio = %.4f\n", custom_tflops / cublas_tflops);
    std::printf("Custom speedup vs cuBLAS   = %.4fx\n", speedup_vs_cublas);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_cublas));
    CUDA_CHECK(cudaFree(d_C_custom));

    return check.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
