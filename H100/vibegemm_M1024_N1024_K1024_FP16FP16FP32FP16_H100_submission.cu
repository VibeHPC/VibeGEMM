// Compile Command:
// nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --expt-extended-lambda vibegemm_M1024_N1024_K1024_FP16FP16FP32FP16_H100_submission.cu -lcublas -lcuda -o vibegemm_M1024_N1024_K1024_FP16FP16FP32FP16_H100
//
// Run Command:
// ./vibegemm_M1024_N1024_K1024_FP16FP16FP32FP16_H100

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

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

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

// ===================================
// 1. Atom-Level
// ===================================

__device__ static inline uint64_t encodeMatrixDescriptor(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t makeSmemDescriptor(half *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= encodeMatrixDescriptor(addr);
    desc |= encodeMatrixDescriptor((uint64_t)16) << 16;
    desc |= encodeMatrixDescriptor((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

__device__ void wgmmaFence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void wgmmaCommit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void wgmmaWait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmmaM64N80K16(float d[5][8], half *sA,
                                               half *sB) {
    uint64_t desc_a = makeSmemDescriptor(&sA[0]);
    uint64_t desc_b = makeSmemDescriptor(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n80k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39},  "
        " %40,"
        " %41,"
        " %42,    %43,  %44,  %45,  %46;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmmaM64N128K16(float d[8][8], half *sA,
                                                half *sB) {
    uint64_t desc_a = makeSmemDescriptor(&sA[0]);
    uint64_t desc_b = makeSmemDescriptor(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
          "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
          "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
          "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
          "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
          "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA,
          int TransB>
__device__ __forceinline__ void wgmma(float d[WGMMA_N / 16][8], half *sA,
                                      half *sB) {
    static_assert(WGMMA_N == 80 || WGMMA_N == 128,
                  "Supported WGMMA_N: 80, 128");
    if constexpr (WGMMA_N == 80)
        wgmmaM64N80K16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if constexpr (WGMMA_N == 128)
        wgmmaM64N128K16<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
}

template <uint32_t RegCount>
__device__ void warpgroupRegAlloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void warpgroupRegDealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ static __forceinline__ void initBarrier(uint64_t *bar,
                                                   int thread_count,
                                                   int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
                 "r"(thread_count + transaction_count));
}

__device__ static __forceinline__ void expectBytes(uint64_t *bar,
                                                   uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            bar_ptr),
        "r"(bytes));
}

__device__ static __forceinline__ void wait(uint64_t *bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(mbar_ptr),
        "r"(kPhaseBit));
}

__device__ static __forceinline__ void arrive(uint64_t *bar,
                                              uint32_t count = 1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 :
                 : "r"(mbar_ptr), "r"(count)
                 : "memory");
}

__device__ void arriveCluster(uint64_t *bar, uint32_t cta_id,
                              uint32_t count = 1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(count));
}

// ===================================
// 2. Tile-Level
// ===================================

constexpr int SPACE_LEN = 512;

template <int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
    int it;
    int *space;

    __device__ __forceinline__ Schedule(int M, int N, int block, int *_space) {
        it = 0;
        space = _space;
    }

    __device__ __forceinline__ bool next(int &block_m, int &block_n) {
        if (it >= SPACE_LEN) return false;
        int now = space[it];
        if (now == -1) return false;
        block_m = now >> 16;
        block_n = (now & ((1 << 16) - 1));
        ++it;
        return true;
    }
};

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajorSize, int BlockMinorSize, bool swizzle = true,
          bool padding = false>
__host__ static inline CUtensorMap createTmaMapDevice(const half *gmem_ptr,
                                                      int global_height,
                                                      int global_width) {
    CUtensorMap tma_map;
    void *gmem_address = const_cast<half *>(gmem_ptr);
    static_assert(BlockMinorSize >= 64);
    assert(global_width % 64 == 0);
    uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height,
                                   (uint64_t)global_width / 64, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(half) * global_width,
                                    64 * sizeof(half), 0, 0, 0};
    uint32_t smem_box_shape[5] = {padding ? 72 : 64, uint32_t(BlockMajorSize),
                                  uint32_t(BlockMinorSize / 64), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, gmem_address,
        gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return tma_map;
}

__device__ static inline void storeAsync(void const *dst_tma_map, half *src,
                                         int global_col_idx,
                                         int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :
        : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(global_row_idx),
          "r"(global_col_idx / 64)
        : "memory");
}

__device__ static inline void tmaFetchAsync(half *dst,
                                            void const *const src_tma_map,
                                            uint64_t *bar, int global_col_idx,
                                            int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0),
          "r"(global_row_idx), "r"(global_col_idx / 64)
        : "memory");
}

__device__ static inline void tmaFetchAsyncMulticast(
    half *dst, void const *const src_tma_map, uint64_t *bar, int global_col_idx,
    int global_row_idx, uint16_t clusterMask) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0),
          "r"(global_row_idx), "r"(global_col_idx / 64), "h"(clusterMask)
        : "memory");
}

template <int BM, int BN, int BK, int QSIZE>
struct SingleBufferSmem {
    alignas(128) half A[BM * BK * QSIZE];
    alignas(128) half B[BK * BN * QSIZE];
    alignas(128) half sC[BN * (BM + 8)];
    alignas(8) uint64_t full[QSIZE], empty[QSIZE];
    int space[SPACE_LEN];
};

// ===================================
// 4. Kernel-Level
// ===================================

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM,
          int CLUSTER_M, int CLUSTER_N>
__global__ __launch_bounds__(NUM_THREADS) void __cluster_dims__(
    CLUSTER_M *CLUSTER_N, 1, 1)
    gemm(int M, int N, int K, half *C,
         const __grid_constant__ CUtensorMap tensorMapA,
         const __grid_constant__ CUtensorMap tensorMapB,
         const __grid_constant__ CUtensorMap tensorMapC, int *dspace) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;
    constexpr int B_WG_M_PADDED = B_WG_M + 8;
    (void)C;

    extern __shared__ __align__(128) uint8_t smem[];
    SingleBufferSmem<BM, BN, BK, QSIZE> &s =
        *reinterpret_cast<SingleBufferSmem<BM, BN, BK, QSIZE> *>(smem);
    half *sA = s.A;
    half *sB = s.B;
    half *sC = s.sC;
    uint64_t *full = s.full;
    uint64_t *empty = s.empty;
    int *my_space = s.space;

    uint32_t rank;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(rank) :);

    if (threadIdx.x < SPACE_LEN) {
        my_space[threadIdx.x] = dspace[rank * SPACE_LEN + threadIdx.x];
    }

    const int numBlocksK = K / BK;
    int wgIdx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int qIdx = 0; qIdx < QSIZE; ++qIdx) {
            initBarrier(&full[qIdx], 0, 1);
            initBarrier(&empty[qIdx], 0, num_consumers * CLUSTERS);
        }
    }
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    uint32_t rankM = rank / CLUSTER_N;
    uint32_t rankN = rank % CLUSTER_N;

    if (wgIdx == 0) {
        warpgroupRegDealloc<32>();
        if (tid == 0) {
            int p = 0;
            int qIdx = 0;
            uint32_t colMask = 0;
            for (int i = 0; i < CLUSTER_M; ++i) {
                colMask |= (1 << (i * CLUSTER_N));
            }
            Schedule<NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N,
                     16 / CLUSTER_M, 8 / CLUSTER_N>
                schedule(M, N, rank, my_space);

            int bIdxM, bIdxN;
            while (schedule.next(bIdxM, bIdxN)) {
                bIdxN = bIdxN * CLUSTER_N + rankN;
                bIdxM = bIdxM * CLUSTER_M + rankM;
                for (int kStep = 0; kStep < numBlocksK; ++kStep, ++qIdx) {
                    if (qIdx == QSIZE) {
                        qIdx = 0;
                        p ^= 1;
                    }
                    wait(&empty[qIdx], p);
                    expectBytes(&full[qIdx],
                                (BK * BM + BK * BN) * sizeof(half));

                    if constexpr (CLUSTER_M > 1) {
                        if (rankM == 0)
                            tmaFetchAsyncMulticast(
                                &sB[qIdx * BK * BN], &tensorMapB, &full[qIdx],
                                kStep * BK, bIdxN * BN, colMask << rankN);
                    } else {
                        tmaFetchAsync(&sB[qIdx * BK * BN], &tensorMapB,
                                      &full[qIdx], kStep * BK, bIdxN * BN);
                    }

                    if constexpr (CLUSTER_N > 1) {
                        uint32_t mask = ((1 << CLUSTER_N) - 1)
                                        << (rankM * CLUSTER_N);
                        if (rankN == 0)
                            tmaFetchAsyncMulticast(
                                &sA[qIdx * BK * BM], &tensorMapA, &full[qIdx],
                                kStep * BK, bIdxM * BM, mask);
                    } else {
                        tmaFetchAsync(&sA[qIdx * BK * BM], &tensorMapA,
                                      &full[qIdx], kStep * BK, bIdxM * BM);
                    }
                }
            }
        }
    } else {
        warpgroupRegAlloc<256>();
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
        --wgIdx;
        for (int qIdx = 0; qIdx < QSIZE; ++qIdx) {
            if (tid < CLUSTERS) arriveCluster(&empty[qIdx], tid);
        }
        int p = 0;
        int qIdx = 0;
        int lane = tid % 32, warp = tid / 32;
        half *blockSC = sC + wgIdx * B_WG_M_PADDED * BN;
        uint32_t tidOffset = warp * 16 + (lane % 8) * B_WG_M_PADDED;
        tidOffset += (lane / 16) * B_WG_M_PADDED * 8 + (lane & 8);
        uint32_t baseAddr =
            static_cast<uint32_t>(__cvta_generic_to_shared(blockSC)) +
            tidOffset * sizeof(half);
        int bIdxM, bIdxN;
        Schedule<NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N,
                 16 / CLUSTER_M, 8 / CLUSTER_N>
            schedule(M, N, rank, my_space);

        while (schedule.next(bIdxM, bIdxN)) {
            bIdxN = bIdxN * CLUSTER_N + rankN;
            bIdxM = bIdxM * CLUSTER_M + rankM;

            {
                wait(&full[qIdx], p);
                wgmmaFence();
#pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half *wgmmaSA =
                        sA + qIdx * BK * BM +
                        64 * (mIt + wgIdx * (B_WG_M / WGMMA_M)) * WGMMA_M;
                    half *wgmmaSB = sB + qIdx * BK * BN;
                    {
                        wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[mIt], &wgmmaSA[0],
                                                      &wgmmaSB[0]);
#pragma unroll
                        for (int kIt = 1; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(
                                d[mIt], &wgmmaSA[kIt * WGMMA_K],
                                &wgmmaSB[kIt * WGMMA_K]);
                        }
                        wgmmaSA += 64 * BM;
                        wgmmaSB += 64 * BN;
                    }
#pragma unroll
                    for (int bk = 64; bk < BK; bk += 64) {
#pragma unroll
                        for (int kIt = 0; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(
                                d[mIt], &wgmmaSA[kIt * WGMMA_K],
                                &wgmmaSB[kIt * WGMMA_K]);
                        }
                        wgmmaSA += 64 * BM;
                        wgmmaSB += 64 * BN;
                    }
                }
                wgmmaCommit();
                wgmmaWait<0>();
                if (tid < CLUSTERS) arriveCluster(&empty[qIdx], tid);
                ++qIdx;
                if (qIdx == QSIZE) {
                    qIdx = 0;
                    p ^= 1;
                }
            }

            for (int kStep = 1; kStep < numBlocksK; ++kStep) {
                wait(&full[qIdx], p);
                wgmmaFence();
#pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half *wgmmaSA =
                        sA + qIdx * BK * BM +
                        64 * (mIt + wgIdx * (B_WG_M / WGMMA_M)) * WGMMA_M;
                    half *wgmmaSB = sB + qIdx * BK * BN;
#pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
#pragma unroll
                        for (int kIt = 0; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(
                                d[mIt], &wgmmaSA[kIt * WGMMA_K],
                                &wgmmaSB[kIt * WGMMA_K]);
                        }
                        wgmmaSA += 64 * BM;
                        wgmmaSB += 64 * BN;
                    }
                }
                wgmmaCommit();
                wgmmaWait<0>();
                if (tid < CLUSTERS) arriveCluster(&empty[qIdx], tid);
                ++qIdx;
                if (qIdx == QSIZE) {
                    qIdx = 0;
                    p ^= 1;
                }
            }

            asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");

            half d_half[8];
            int *data_ptr = (int *)d_half;
#pragma unroll
            for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
#pragma unroll
                for (int w = 0; w < WGMMA_N; w += 16) {
                    uint32_t addr =
                        baseAddr +
                        (w * B_WG_M_PADDED + mIt * WGMMA_M) * sizeof(half);
#pragma unroll
                    for (int k = 0; k < 8; k++)
                        d_half[k] = (half)(d[mIt][w / 16][k]);
                    asm volatile(
                        "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 "
                        "[%0], {%1, %2, %3, %4};"
                        :
                        : "r"(addr), "r"(data_ptr[0]), "r"(data_ptr[1]),
                          "r"(data_ptr[2]), "r"(data_ptr[3]));
                }
            }
            asm volatile("bar.sync %0, 128;\n" : : "r"(wgIdx + 2) : "memory");
            if (tid == 0) {
                storeAsync(&tensorMapC, blockSC, bIdxM * BM + wgIdx * B_WG_M,
                           bIdxN * BN);
                asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
            }
        }
    }
}

// ===================================
// 5. Device-Level
// ===================================

void rot(int n, int &x, int &y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            x = n - 1 - x;
            y = n - 1 - y;
        }
        int t = x;
        x = y;
        y = t;
    }
}

void d2xy(int n, int d, int &x, int &y) {
    int rx, ry, s, t = d;
    x = y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        x += s * rx;
        y += s * ry;
        t /= 4;
    }
}

void createHilbert(int M, int N, int CORES, int *space) {
    int mm = (M > N) ? M : N;
    int dim = (mm <= 1)
                  ? 1
                  : (1 << (32 - __builtin_clz(static_cast<unsigned>(mm - 1))));
    int core = 0, loc = 0;
    std::vector<std::string> v(dim, std::string(dim, '.'));
    memset(space, -1, sizeof(int) * CORES * SPACE_LEN);
    int total = 0;
    for (int i = 0; i < dim * dim; ++i) {
        int x, y;
        d2xy(dim, i, x, y);
        if (x < M && y < N) {
            assert(loc < SPACE_LEN);
            v[x][y] = '*';
            ++total;
            space[core * SPACE_LEN + loc] = ((x << 16) | y);
            ++core;
            if (core == CORES) {
                core = 0;
                loc++;
            }
        }
    }
    assert(total == M * N);
}

static CUtensorMap globalTmaMapABase;
static CUtensorMap globalTmaMapBBase;
static CUtensorMap globalTmaMapCBase;
static int prevMBase = 0, prevNBase = 0, prevKBase = 0;
static int *globalDspaceBase = nullptr;

static void launch_custom_gemm(const half *d_A, const half *d_B, half *d_C,
                               int M, int N, int K) {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 256;
    constexpr int QSIZE = 8;

    constexpr int CLUSTER_M = 2;
    constexpr int CLUSTER_N = 1;

    constexpr int NUM_SM = 128;

    int num_clusters = NUM_SM / (CLUSTER_M * CLUSTER_N);

    if (prevMBase != M || prevNBase != N || prevKBase != K) {
        constexpr int num_consumers = (NUM_THREADS / 128) - 1;
        static_assert(BM % num_consumers == 0);
        globalTmaMapABase = createTmaMapDevice<BM, BK>(d_A, M, K);
        globalTmaMapBBase = createTmaMapDevice<BN, BK>(d_B, N, K);
        assert(N % 64 == 0 && M % 64 == 0);
        assert(N % 64 == 0 && M % 64 == 0);
        globalTmaMapCBase =
            createTmaMapDevice<BN, BM / num_consumers, false, true>(d_C, N, M);
        prevMBase = M;
        prevNBase = N;
        prevKBase = K;

        if (globalDspaceBase != nullptr) CUDA_CHECK(cudaFree(globalDspaceBase));
        int *space = (int *)malloc(sizeof(int) * num_clusters * SPACE_LEN);
        createHilbert(CEIL_DIV(M, BM * CLUSTER_M), CEIL_DIV(N, BN * CLUSTER_N),
                      num_clusters, space);
        CUDA_CHECK(cudaMalloc((void **)&globalDspaceBase,
                             sizeof(int) * num_clusters * SPACE_LEN));
        CUDA_CHECK(cudaMemcpy(globalDspaceBase, space,
                             sizeof(int) * num_clusters * SPACE_LEN,
                             cudaMemcpyHostToDevice));
        free(space);
    }

    assert(M == prevMBase && N == prevNBase && K == prevKBase);

    auto *kernel =
        gemm<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M, CLUSTER_N>;
    size_t sMemSize = sizeof(SingleBufferSmem<BM, BN, BK, QSIZE>);

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    kernel<<<NUM_SM, NUM_THREADS, sMemSize>>>(
        M, N, K, d_C, globalTmaMapABase, globalTmaMapBBase, globalTmaMapCBase,
        globalDspaceBase);
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
    std::printf("  M=N=K=1024, warmup_iters=10, benchmark_iters=30\n");
}

int main(int argc, char **argv) {
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
