// Compile Command:
// nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --expt-extended-lambda vibegemm_M256_N256_K256_FP16FP16FP32FP16_H100_submission.cu -lcublas -lcuda -o vibegemm_M256_N256_K256_FP16FP16FP32FP16_H100
//
// Run Command:
// ./vibegemm_M256_N256_K256_FP16FP16FP32FP16_H100

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ========================================================================================
// Error checking helpers
// ========================================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status__ = (call);                                     \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                              \
            std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, \
                         __LINE__, static_cast<int>(status__));               \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

static inline void check_last_cuda_error(const char *name) {
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

static void fill_random_fp16(std::vector<half> &x, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (half &v : x) {
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

static CheckResult check_correctness(const half *d_ref, const half *d_test,
                                     size_t elements,
                                     float atol = 1e-2f,  // absolute tolerance
                                     float rtol = 1e-2f,  // relative tolerance
                                     size_t print_first_n = 5) {
    std::vector<half> h_ref(elements);
    std::vector<half> h_test(elements);

    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, elements * sizeof(half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_test.data(), d_test, elements * sizeof(half),
                          cudaMemcpyDeviceToHost));

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
                std::printf(
                    "    mismatch[%zu]: ref=% .6f  test=% .6f  abs=% .6e  "
                    "rel=% .6e\n",
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

static void run_cublas_gemm(cublasHandle_t handle, const half *d_A,
                            const half *d_B, half *d_C, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS assumes column-major matrices. In this template:
    //   A is stored as row-major A[M,K]. The same memory can be viewed by
    //   cuBLAS as a column-major matrix with shape K x M, so CUBLAS_OP_T gives
    //   A[M,K]. B is stored as column-major B[K,N]. C is stored as column-major
    //   C[M,N].
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              M,  // rows of C
                              N,  // columns of C
                              K,  // reduction dimension
                              &alpha, d_A, CUDA_R_16F,
                              K,  // leading dimension of A viewed as K x M
                              d_B, CUDA_R_16F,
                              K,  // leading dimension of B
                              &beta, d_C, CUDA_R_16F,
                              M,  // leading dimension of C
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// ========================================================================================
// SUBMITTER SECTION: replace this simple kernel with an optimized
// implementation
// ========================================================================================

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ===================================
// 1. Atom-Level
// ===================================

__device__ __forceinline__ uint64_t encodeMatrixDescriptor(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ __forceinline__ uint64_t makeSmemDescriptor(const half *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= encodeMatrixDescriptor(addr);
    desc |= encodeMatrixDescriptor((uint64_t)16) << 16;
    desc |= encodeMatrixDescriptor((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

__device__ __forceinline__ void wgmmaFence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmmaCommit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wgmmaWait() {
    static_assert(N >= 0 && N <= 7,
                  "WGMMA wait groups must be between 0 and 7.");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1, int TransA = 0,
          int TransB = 0>
__device__ __forceinline__ void wgmmaM64N64K16(float d[4][8], const half *sA,
                                               const half *sB) {
    uint64_t desc_a = makeSmemDescriptor(&sA[0]);
    uint64_t desc_b = makeSmemDescriptor(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

// ===================================
// 2. Tile-Level
// ===================================

template <int M, int N, int K, int WGMMA_M, int WGMMA_N, int WGMMA_K,
          int THREADS>
struct BlockShape {
    static constexpr int TileM = M;
    static constexpr int TileN = N;
    static constexpr int TileK = K;
    static constexpr int NumThreads = THREADS;

    static constexpr int WgmmaM = WGMMA_M;
    static constexpr int WgmmaN = WGMMA_N;
    static constexpr int WgmmaK = WGMMA_K;

    static constexpr int NumWgmmaM = TileM / WgmmaM;
    static constexpr int NumWgmmaN = TileN / WgmmaN;
    static constexpr int NumItersK = TileK / WgmmaK;

    static constexpr int SmemElemsA = TileM * TileK;
    static constexpr int SmemElemsB = TileK * TileN;
    static constexpr int StageElems = SmemElemsA + SmemElemsB;
};

__device__ __forceinline__ void mapThreadToEpilogue(int tid, int wgM, int wgN,
                                                    int w, int &row, int &col) {
    const int lane = tid % 32;
    const int warp = tid / 32;
    row = wgM * 64 + warp * 16 + lane / 4;
    col = wgN * 64 + 16 * w + 2 * (tid % 4);
}

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockM, int BlockK>
__host__ CUtensorMap *createTmaMapDevice(const half *src, int blocksHeight,
                                         int blocksWidth) {
    CUtensorMap hostMap;
    uint64_t shape[5] = {(uint64_t)BlockK * blocksWidth,
                         (uint64_t)BlockM * blocksHeight, 1, 1, 1};
    uint64_t stride[5] = {sizeof(half), sizeof(half) * BlockK * blocksWidth, 0,
                          0, 0};
    uint32_t smemShape[5] = {uint32_t(BlockK), uint32_t(BlockM), 1, 1, 1};
    uint32_t smemStride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &hostMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void *)src, shape,
        stride + 1, smemShape, smemStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(result == CUDA_SUCCESS);

    CUtensorMap *devMap;
    cudaMalloc(&devMap, sizeof(CUtensorMap));
    cudaMemcpy(devMap, &hostMap, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return devMap;
}

template <typename Shape>
struct alignas(128) SingleBufferSmem {
    half A[Shape::SmemElemsA];
    half B[Shape::SmemElemsB];
};

__device__ __forceinline__ void tmaFetchAsync(
    half *dstA, half *dstB, const CUtensorMap *mapA, const CUtensorMap *mapB,
    int globalBlockIdxM, int globalBlockIdxN, int globalBlockKStep, int TileM,
    int TileN, int TileK, Barrier &barA, Barrier &barB) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        dstA, mapA, globalBlockKStep * TileK, globalBlockIdxM * TileM, barA);
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        dstB, mapB, globalBlockKStep * TileK, globalBlockIdxN * TileN, barB);
}

// ===================================
// 4. Kernel-Level
// ===================================

template <typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads)
    matmulKernel3(int M, int N, int K, half *C,
                  const CUtensorMap *__restrict__ tmaA,
                  const CUtensorMap *__restrict__ tmaB) {
    __shared__ SingleBufferSmem<Shape> smem;
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier barA, barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    half *sA = smem.A;
    half *sB = smem.B;

    float d[Shape::NumWgmmaM][Shape::WgmmaN / 16][8];

    static_assert(sizeof(d) * Shape::NumThreads ==
                  Shape::TileM * Shape::TileN * sizeof(float));

    const int numBlocksK = K / Shape::TileK;
    const int bIdxN = blockIdx.x % (N / Shape::TileN);
    const int bIdxM = blockIdx.x / (N / Shape::TileN);

    Barrier::arrival_token tokenA, tokenB;

    for (int kStep = 0; kStep < numBlocksK; ++kStep) {
        if (threadIdx.x == 0) {
            tmaFetchAsync(sA, sB, tmaA, tmaB, bIdxM, bIdxN, kStep, Shape::TileM,
                          Shape::TileN, Shape::TileK, barA, barB);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(smem.A));
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(smem.B));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        wgmmaFence();

        if (kStep == 0) {
#pragma unroll
            for (int mi = 0; mi < Shape::NumWgmmaM; ++mi) {
                half *ptrA = &sA[0];
                wgmmaM64N64K16<0, 1, 1, 0, 0>(d[mi], &ptrA[0], &sB[0]);
                wgmmaM64N64K16<1, 1, 1, 0, 0>(d[mi], &ptrA[Shape::WgmmaK],
                                              &sB[Shape::WgmmaK]);
                wgmmaM64N64K16<1, 1, 1, 0, 0>(d[mi], &ptrA[2 * Shape::WgmmaK],
                                              &sB[2 * Shape::WgmmaK]);
                wgmmaM64N64K16<1, 1, 1, 0, 0>(d[mi], &ptrA[3 * Shape::WgmmaK],
                                              &sB[3 * Shape::WgmmaK]);
            }
        } else {
#pragma unroll
            for (int mi = 0; mi < Shape::NumWgmmaM; ++mi) {
                half *ptrA = &sA[0];
                wgmmaM64N64K16<1, 1, 1, 0, 0>(d[mi], &ptrA[0], &sB[0]);
                wgmmaM64N64K16<1, 1, 1, 0, 0>(d[mi], &ptrA[Shape::WgmmaK],
                                              &sB[Shape::WgmmaK]);
                wgmmaM64N64K16<1, 1, 1, 0, 0>(d[mi], &ptrA[2 * Shape::WgmmaK],
                                              &sB[2 * Shape::WgmmaK]);
                wgmmaM64N64K16<1, 1, 1, 0, 0>(d[mi], &ptrA[3 * Shape::WgmmaK],
                                              &sB[3 * Shape::WgmmaK]);
            }
        }

        wgmmaCommit();
        wgmmaWait<0>();
    }

    half *blockC = C + (bIdxN * Shape::TileN * M) + (bIdxM * Shape::TileM);
    for (int mi = 0; mi < Shape::NumWgmmaM; ++mi) {
        for (int ni = 0; ni < Shape::NumWgmmaN; ++ni) {
            for (int w = 0; w < Shape::WgmmaN / 16; ++w) {
                int row, col;
                mapThreadToEpilogue(threadIdx.x, mi, ni, w, row, col);

                float *frag = &d[mi][w][0];
                auto storeC = [&](int rOff, int cOff, float val) {
                    blockC[(col + cOff) * M + (row + rOff)] = __float2half(val);
                };

                storeC(0, 0, frag[0]);
                storeC(0, 1, frag[1]);
                storeC(8, 0, frag[2]);
                storeC(8, 1, frag[3]);
                storeC(0, 8, frag[4]);
                storeC(0, 9, frag[5]);
                storeC(8, 8, frag[6]);
                storeC(8, 9, frag[7]);
            }
        }
    }
}

// ===================================
// 5. Device-Level
// ===================================

static CUtensorMap *globalTmaMapABase = nullptr;
static CUtensorMap *globalTmaMapBBase = nullptr;
static int prevMBase = 0, prevNBase = 0, prevKBase = 0;

// Keep this signature unchanged. The benchmark calls this function directly.
static void launch_custom_gemm(const half *d_A, const half *d_B, half *d_C,
                               int M, int N, int K) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 64;
    constexpr int WGMMA_K = 16;
    constexpr int NUM_THREADS = 128;
    using Shape =
        BlockShape<BM, BN, BK, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS>;

    if (!globalTmaMapABase || M != prevMBase || N != prevNBase ||
        K != prevKBase) {
        if (globalTmaMapABase) cudaFree(globalTmaMapABase);
        if (globalTmaMapBBase) cudaFree(globalTmaMapBBase);
        globalTmaMapABase = createTmaMapDevice<Shape::TileM, Shape::TileK>(
            d_A, M / Shape::TileM, K / Shape::TileK);
        globalTmaMapBBase = createTmaMapDevice<Shape::TileN, Shape::TileK>(
            d_B, N / Shape::TileN, K / Shape::TileK);
        prevMBase = M;
        prevNBase = N;
        prevKBase = K;
    }

    dim3 grid((M / Shape::TileM) * (N / Shape::TileN));
    dim3 block(Shape::NumThreads);
    matmulKernel3<Shape>
        <<<grid, block>>>(M, N, K, d_C, globalTmaMapABase, globalTmaMapBBase);
}

// ========================================================================================
// Benchmark wrapper
// ========================================================================================

enum class BackendKind { Cublas, Custom };

static float benchmark_backend(BackendKind backend, cublasHandle_t handle,
                               const half *d_A, const half *d_B, half *d_C,
                               int M, int N, int K, int warmup_iters,
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

static void print_usage(const char *program) {
    std::printf("Usage:\n");
    std::printf("  %s [M N K] [warmup_iters benchmark_iters]\n", program);
    std::printf("\nDefaults:\n");
    std::printf("  M=N=K=256, warmup_iters=10, benchmark_iters=30\n");
}

int main(int argc, char **argv) {
    int M = 256;
    int N = 256;
    int K = 256;
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
    if (M <= 0 || N <= 0 || K <= 0 || warmup_iters < 0 ||
        benchmark_iters <= 0) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);
    std::printf("Problem: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf(
        "Layout : A row-major, B column-major, C column-major; C[%d,%d] = "
        "A[%d,%d] * B[%d,%d]\n",
        M, N, M, K, K, N);
    std::printf("Iters  : warmup=%d, benchmark=%d\n\n", warmup_iters,
                benchmark_iters);

    const size_t elems_A = static_cast<size_t>(M) * K;
    const size_t elems_B = static_cast<size_t>(K) * N;
    const size_t elems_C = static_cast<size_t>(M) * N;

    std::vector<half> h_A(elems_A);
    std::vector<half> h_B(elems_B);
    fill_random_fp16(h_A, 42);
    fill_random_fp16(h_B, 77);
    // h_A is interpreted as row-major A[M,K].
    // h_B is interpreted as column-major B[K,N].

    half *d_A = nullptr;
    half *d_B = nullptr;
    half *d_C_cublas = nullptr;
    half *d_C_custom = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, elems_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, elems_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, elems_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_custom, elems_C * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), elems_A * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), elems_B * sizeof(half),
                          cudaMemcpyHostToDevice));

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
    std::printf("  max absolute error = %.6e at index %zu\n",
                check.max_abs_error, check.max_abs_index);
    std::printf("  max relative error = %.6e at index %zu\n",
                check.max_rel_error, check.max_rel_index);
    std::printf("  mismatches         = %zu\n", check.mismatch_count);
    std::printf("  result             = %s\n\n",
                check.passed ? "PASS" : "FAIL");

    float cublas_ms =
        benchmark_backend(BackendKind::Cublas, handle, d_A, d_B, d_C_cublas, M,
                          N, K, warmup_iters, benchmark_iters);
    float custom_ms =
        benchmark_backend(BackendKind::Custom, handle, d_A, d_B, d_C_custom, M,
                          N, K, warmup_iters, benchmark_iters);

    double cublas_tflops = gemm_tflops(M, N, K, cublas_ms);
    double custom_tflops = gemm_tflops(M, N, K, custom_ms);
    double speedup_vs_cublas = cublas_ms / custom_ms;

    std::printf("Performance:\n");
    std::printf("  %-16s %12s %12s\n", "Backend", "Latency(ms)", "TFLOPS");
    std::printf("  %-16s %12.4f %12.3f\n", "cuBLAS", cublas_ms, cublas_tflops);
    std::printf("  %-16s %12.4f %12.3f\n", "Custom", custom_ms, custom_tflops);
    std::printf("\n");
    std::printf("Custom/cuBLAS TFLOPS ratio = %.4f\n",
                custom_tflops / cublas_tflops);
    std::printf("Custom speedup vs cuBLAS   = %.4fx\n", speedup_vs_cublas);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_cublas));
    CUDA_CHECK(cudaFree(d_C_custom));

    return check.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
