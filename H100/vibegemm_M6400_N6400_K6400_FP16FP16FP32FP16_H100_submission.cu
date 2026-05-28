// Compile Command:
// nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --expt-extended-lambda vibegemm_M6400_N6400_K6400_FP16FP16FP32FP16_H100_submission.cu -lcublas -lcuda -o vibegemm_M6400_N6400_K6400_FP16FP16FP32FP16_H100
//
// Run Command:
// ./vibegemm_M6400_N6400_K6400_FP16FP16FP32FP16_H100

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

namespace cde = cuda::device::experimental;

// ===================================
// 1. Atom-Level
// ===================================

constexpr int CLUSTER_M = 2;
constexpr int CLUSTER_N = 1;
constexpr int CLUSTER_SIZE = CLUSTER_M * CLUSTER_N;

__device__ __forceinline__ uint32_t smemPtrU32(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t encodeMatrixDescriptor(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ __forceinline__ uint64_t makeSmemDescriptor(half *ptr) {
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

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroupRegAlloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(RegCount)
                 : "memory");
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroupRegDealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(RegCount)
                 : "memory");
}

__device__ __forceinline__ void initBarrier(uint64_t *bar, int threadCount,
                                            int transactionCount) {
    uint32_t bar_ptr = smemPtrU32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
                 "r"(threadCount + transactionCount));
}

__device__ __forceinline__ void expectBytes(uint64_t *bar, uint32_t bytes) {
    uint32_t bar_ptr = smemPtrU32(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :
                 : "r"(bar_ptr), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void waitBarrier(uint64_t *bar, int phaseBit) {
    uint32_t bar_ptr = smemPtrU32(bar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :
        : "r"(bar_ptr), "r"(phaseBit)
        : "memory");
}

__device__ __forceinline__ void arriveCluster(uint64_t *bar, uint32_t ctaId,
                                              uint32_t count = 1) {
    uint32_t smem_addr = smemPtrU32(bar);
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(ctaId), "r"(count));
}

template <int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA,
          int TransB>
__device__ __forceinline__ void wgmma(float d[][8], half *sA, half *sB) {
    uint64_t desc_a = makeSmemDescriptor(sA);
    uint64_t desc_b = makeSmemDescriptor(sB);

    if constexpr (WGMMA_N == 256) {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
            "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
            "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
            "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,"
            "%110,%111,"
            "%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%"
            "125,%126,%127}, "
            "%128,%129,%130,%131,%132,%133,%134;"
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
              "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
              "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
              "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
              "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
              "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
              "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
              "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
              "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
              "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
              "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
              "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
              "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
              "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
              "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
              "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
              "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]),
              "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
            : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
              "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)), "n"(int32_t(TransA)),
              "n"(int32_t(TransB)));
    }
}

// ===================================
// 2. Tile-Level
// ===================================

template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(128) half A[BM * BK * QSIZE];
    alignas(128) half B[BK * BN * QSIZE];
    alignas(128) half C[BN * BM];
    alignas(8) uint64_t full[QSIZE], empty[QSIZE];
    alignas(16) int space[128];
};

struct HilbertSchedule {
    const int *space;
    int it;

    __device__ __forceinline__ HilbertSchedule(const int *_space)
        : space(_space), it(0) {}

    __device__ __forceinline__ bool next(int &block_m, int &block_n) {
        if (it >= 128) return false;
        int value = space[it++];
        if (value < 0) return false;
        block_m = (value >> 16) & 0xFFFF;
        block_n = value & 0xFFFF;
        return true;
    }
};

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajorSize, int BlockMinorSize, bool swizzle = true>
__host__ CUtensorMap createTmaMapDevice(const half *src, int globalHeight,
                                        int globalWidth) {
    CUtensorMap hostMap;
    static_assert(BlockMinorSize >= 64);
    assert(globalWidth % 64 == 0);

    uint64_t shape[5] = {64, (uint64_t)globalHeight, (uint64_t)globalWidth / 64,
                         1, 1};
    uint64_t stride[5] = {sizeof(half) * globalWidth, 64 * sizeof(half), 0, 0,
                          0};
    uint32_t smemShape[5] = {64, uint32_t(BlockMajorSize),
                             uint32_t(BlockMinorSize / 64), 1, 1};
    uint32_t smemStride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &hostMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, (void *)src, shape,
        stride, smemShape, smemStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(result == CUDA_SUCCESS);

    return hostMap;
}

__device__ __forceinline__ void tmaFetchAsync(half *dst, void const *map,
                                              uint64_t *bar, int globalColIdx,
                                              int globalRowIdx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(map);
    uint32_t bar_ptr = smemPtrU32(bar);
    uint32_t dst_ptr = smemPtrU32(dst);

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(bar_ptr), "n"(0), "r"(globalRowIdx),
          "r"(globalColIdx / 64)
        : "memory");
}

__device__ __forceinline__ void tmaFetchAsyncMulticast(
    half *dst, void const *map, uint64_t *bar, int globalColIdx,
    int globalRowIdx, uint16_t clusterMask) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(map);
    uint32_t bar_ptr = smemPtrU32(bar);
    uint32_t dst_ptr = smemPtrU32(dst);

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(bar_ptr), "n"(0), "r"(globalRowIdx),
          "r"(globalColIdx / 64), "h"(clusterMask)
        : "memory");
}

__device__ __forceinline__ void tmaStoreAsync(void const *map, half *src,
                                              int globalColIdx,
                                              int globalRowIdx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(map);
    uint32_t src_ptr = smemPtrU32(src);

    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :
        : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(globalRowIdx),
          "r"(globalColIdx / 64)
        : "memory");
}

// ===================================
// 4. Kernel-Level
// ===================================

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM,
          int CLUSTER_M_, int CLUSTER_N_>
__global__ void __launch_bounds__(NUM_THREADS)
    __cluster_dims__(CLUSTER_M_ *CLUSTER_N_, 1, 1)
        matmulKernel3Pipeline(int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmaC,
                              const __grid_constant__ CUtensorMap tmaA,
                              const __grid_constant__ CUtensorMap tmaB,
                              const int *__restrict__ dspace) {
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_K = 16;
    constexpr int WGMMA_N = BN;
    constexpr int NumConsumers = (NUM_THREADS / 128) - 1;
    constexpr int BWgM = BM / NumConsumers;
    constexpr int CLUSTERS = CLUSTER_M_ * CLUSTER_N_;

    assert((M / BM) % CLUSTER_M_ == 0);
    assert((N / BN) % CLUSTER_N_ == 0);

    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM, BN, BK, QSIZE> &s =
        *reinterpret_cast<SMem<BM, BN, BK, QSIZE> *>(smem);

    half *sA = s.A;
    half *sB = s.B;
    half *sC = s.C;
    uint64_t *full = s.full;
    uint64_t *empty = s.empty;

    uint32_t cluster_id;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_id) :);

    const int numBlocksK = K / BK;
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            initBarrier(&full[i], 0, 1);
            initBarrier(&empty[i], 0, NumConsumers * CLUSTERS);
        }
    }

    if (threadIdx.x < 128) {
        s.space[threadIdx.x] = dspace[cluster_id * 128 + threadIdx.x];
    }

    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    HilbertSchedule schedule(s.space);

    uint32_t rank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    uint32_t rank_m = rank / CLUSTER_N_;
    uint32_t rank_n = rank % CLUSTER_N_;

    if (wg_idx == 0) {
        constexpr int NumRegs = (NumConsumers <= 2 ? 24 : 32);
        warpgroupRegDealloc<NumRegs>();

        if (tid == 0) {
            int p = 0;
            int qidx = 0;
            uint16_t b_mask = 0;

            if constexpr (CLUSTER_M_ > 1) {
#pragma unroll
                for (int rm = 0; rm < CLUSTER_M_; ++rm) {
                    b_mask |= (uint16_t)(1u << (rm * CLUSTER_N_ + rank_n));
                }
            }

            int num_block_m, num_block_n;
            while (schedule.next(num_block_m, num_block_n)) {
                num_block_n = num_block_n * CLUSTER_N_ + rank_n;
                num_block_m = num_block_m * CLUSTER_M_ + rank_m;

                for (int kStep = 0; kStep < numBlocksK; ++kStep, ++qidx) {
                    if (qidx == QSIZE) {
                        qidx = 0;
                        p ^= 1;
                    }

                    waitBarrier(&empty[qidx], p);
                    expectBytes(&full[qidx],
                                (BK * BN + BK * BM) * sizeof(half));

                    if constexpr (CLUSTER_N_ > 1) {
                        uint32_t mask = ((1u << CLUSTER_N_) - 1u)
                                        << (rank_m * CLUSTER_N_);
                        if (rank_n == 0) {
                            tmaFetchAsyncMulticast(
                                &sA[qidx * BK * BM], &tmaA, &full[qidx],
                                kStep * BK, num_block_m * BM, (uint16_t)mask);
                        }
                    } else {
                        tmaFetchAsync(&sA[qidx * BK * BM], &tmaA, &full[qidx],
                                      kStep * BK, num_block_m * BM);
                    }

                    if constexpr (CLUSTER_M_ > 1) {
                        if (rank_m == 0) {
                            tmaFetchAsyncMulticast(&sB[qidx * BK * BN], &tmaB,
                                                   &full[qidx], kStep * BK,
                                                   num_block_n * BN, b_mask);
                        }
                    } else {
                        tmaFetchAsync(&sB[qidx * BK * BN], &tmaB, &full[qidx],
                                      kStep * BK, num_block_n * BN);
                    }
                }
            }
        }
    } else {
        warpgroupRegAlloc<240>();

        float d[BWgM / WGMMA_M][WGMMA_N / 16][8];

        --wg_idx;

        for (int qidx = 0; qidx < QSIZE; ++qidx) {
            if (tid < CLUSTERS) arriveCluster(&empty[qidx], tid);
        }

        int p = 0;
        int qidx = 0;
        int num_block_m, num_block_n;

        while (schedule.next(num_block_m, num_block_n)) {
            num_block_n = num_block_n * CLUSTER_N_ + rank_n;
            num_block_m = num_block_m * CLUSTER_M_ + rank_m;

            {
                if (qidx == QSIZE) {
                    qidx = 0;
                    p ^= 1;
                }

                waitBarrier(&full[qidx], p);
                wgmmaFence();

#pragma unroll
                for (int mi = 0; mi < BWgM / WGMMA_M; ++mi) {
                    half *ptrA = sA + qidx * BK * BM +
                                 64 * (mi + wg_idx * BWgM / WGMMA_M) * WGMMA_M;
                    half *ptrB = sB + qidx * BK * BN;

                    wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[mi], &ptrA[0], &ptrB[0]);

#pragma unroll
                    for (int ki = 1; ki < 64 / WGMMA_K; ++ki) {
                        wgmma<WGMMA_N, 1, 1, 1, 0, 0>(
                            d[mi], &ptrA[ki * WGMMA_K], &ptrB[ki * WGMMA_K]);
                    }

                    ptrA += 64 * BM;
                    ptrB += 64 * BN;

#pragma unroll
                    for (int bk = 64; bk < BK; bk += 64) {
#pragma unroll
                        for (int ki = 0; ki < 64 / WGMMA_K; ++ki) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[mi],
                                                          &ptrA[ki * WGMMA_K],
                                                          &ptrB[ki * WGMMA_K]);
                        }
                        ptrA += 64 * BM;
                        ptrB += 64 * BN;
                    }
                }

                wgmmaCommit();
                wgmmaWait<0>();

                if (tid < CLUSTERS) arriveCluster(&empty[qidx], tid);

                ++qidx;
            }

            for (int kStep = 1; kStep < numBlocksK; ++kStep, ++qidx) {
                if (qidx == QSIZE) {
                    qidx = 0;
                    p ^= 1;
                }

                waitBarrier(&full[qidx], p);
                wgmmaFence();

#pragma unroll
                for (int mi = 0; mi < BWgM / WGMMA_M; ++mi) {
                    half *ptrA = sA + qidx * BK * BM +
                                 64 * (mi + wg_idx * BWgM / WGMMA_M) * WGMMA_M;
                    half *ptrB = sB + qidx * BK * BN;

#pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
#pragma unroll
                        for (int ki = 0; ki < 64 / WGMMA_K; ++ki) {
                            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[mi],
                                                          &ptrA[ki * WGMMA_K],
                                                          &ptrB[ki * WGMMA_K]);
                        }
                        ptrA += 64 * BM;
                        ptrB += 64 * BN;
                    }
                }

                wgmmaCommit();
                wgmmaWait<0>();

                if (tid < CLUSTERS) arriveCluster(&empty[qidx], tid);
            }

            asm volatile("cp.async.bulk.wait_group 0;");

            int lane = tid % 32;
            int warp = tid / 32;
            int row = warp * 16 + lane / 4;
            int col_base = 2 * (tid % 4);

            half *block_sC = sC + wg_idx * BWgM * BN;

#pragma unroll
            for (int mi = 0; mi < BWgM / 64; ++mi) {
                int row_offset = mi * 64;

#pragma unroll
                for (int ni = 0; ni < BN / 16; ++ni) {
                    int col_offset = ni * 16;

#define ST(r_off, c_off, val)                                                \
    block_sC[(col_offset + (c_off)) * BWgM + (row_offset + row + (r_off))] = \
        __float2half(val)

                    ST(0, col_base, d[mi][ni][0]);
                    ST(8, col_base, d[mi][ni][2]);
                    ST(0, col_base + 1, d[mi][ni][1]);
                    ST(8, col_base + 1, d[mi][ni][3]);

                    ST(0, col_base + 8, d[mi][ni][4]);
                    ST(8, col_base + 8, d[mi][ni][6]);
                    ST(0, col_base + 9, d[mi][ni][5]);
                    ST(8, col_base + 9, d[mi][ni][7]);

#undef ST
                }
            }

            asm volatile("bar.sync 10, 256;\n");

            if (threadIdx.x == 128) {
                tmaStoreAsync(&tmaC, sC, num_block_m * BM, num_block_n * BN);
                asm volatile("cp.async.bulk.commit_group;");
            }
        }

        warpgroupRegDealloc<240>();
    }
}

// ===================================
// 5. Device-Level
// ===================================

static int *_dspace = nullptr;
static int _space_m_tiles = 0;
static int _space_n_tiles = 0;
static int _space_num_clusters = 0;

static CUtensorMap tmaA;
static CUtensorMap tmaB;
static CUtensorMap tmaC;

static int _prev_m = 0;
static int _prev_n = 0;
static int _prev_k = 0;

static inline void rotHilbert(int n, int &x, int &y, int rx, int ry) {
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

static inline void d2xyHilbert(int n, int d, int &x, int &y) {
    int t = d;
    x = y = 0;

    for (int s = 1; s < n; s <<= 1) {
        int rx = 1 & (t >> 1);
        int ry = 1 & (t ^ rx);
        rotHilbert(s, x, y, rx, ry);
        x += s * rx;
        y += s * ry;
        t >>= 2;
    }
}

static inline int nextPow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

static int *generateHilbertHostData(int mTiles, int nTiles, int numClusters) {
    const int depth = 128;
    const int totalTasks = mTiles * nTiles;

    const int side = nextPow2((mTiles > nTiles) ? mTiles : nTiles);
    const int maxD = side * side;

    int *hostSpace = (int *)malloc((size_t)numClusters * depth * sizeof(int));

    for (int i = 0; i < numClusters * depth; ++i) hostSpace[i] = -1;

    int taskId = 0;

    for (int d = 0; d < maxD && taskId < totalTasks; ++d) {
        int x, y;
        d2xyHilbert(side, d, x, y);

        if (y < mTiles && x < nTiles) {
            int block_m = y;
            int block_n = x;

            int cluster = taskId % numClusters;
            int pos = taskId / numClusters;

            if (pos < depth) {
                hostSpace[cluster * depth + pos] =
                    (block_m << 16) | (block_n & 0xFFFF);
            }

            ++taskId;
        }
    }

    return hostSpace;
}

static void launch_custom_gemm(const half *d_A, const half *d_B, half *d_C,
                               int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int QSIZE = 3;
    constexpr int NUM_THREADS = 384;
    constexpr int NUM_SM = 128;
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

    if (_prev_m != M || _prev_n != N || _prev_k != K) {
        tmaA = createTmaMapDevice<BM, BK>(d_A, M, K);
        tmaB = createTmaMapDevice<BN, BK>(d_B, N, K);
        tmaC = createTmaMapDevice<BN, BM, false>(d_C, N, M);

        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
    }

    const int NUM_CLUSTERS = NUM_SM / CLUSTERS;
    const int m_tiles = CEIL_DIV(M, BM * CLUSTER_M);
    const int n_tiles = CEIL_DIV(N, BN * CLUSTER_N);

    const bool need_regen =
        (_dspace == nullptr) || (_space_m_tiles != m_tiles) ||
        (_space_n_tiles != n_tiles) || (_space_num_clusters != NUM_CLUSTERS);

    if (need_regen) {
        if (_dspace) {
            cudaFree(_dspace);
            _dspace = nullptr;
        }

        int *hostSpace =
            generateHilbertHostData(m_tiles, n_tiles, NUM_CLUSTERS);

        cudaMalloc(&_dspace, (size_t)NUM_CLUSTERS * 128 * sizeof(int));
        cudaMemcpy(_dspace, hostSpace, (size_t)NUM_CLUSTERS * 128 * sizeof(int),
                   cudaMemcpyHostToDevice);

        free(hostSpace);

        _space_m_tiles = m_tiles;
        _space_n_tiles = n_tiles;
        _space_num_clusters = NUM_CLUSTERS;
    }

    size_t sharedBytes = sizeof(SMem<BM, BN, BK, QSIZE>);
    auto kernel = &matmulKernel3Pipeline<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM,
                                         CLUSTER_M, CLUSTER_N>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sharedBytes);

    kernel<<<NUM_SM, NUM_THREADS, sharedBytes>>>(M, N, K, tmaC, tmaA, tmaB,
                                                 _dspace);
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
    std::printf("  M=N=K=6400, warmup_iters=10, benchmark_iters=30\n");
}

int main(int argc, char **argv) {
    int M = 6400;
    int N = 6400;
    int K = 6400;
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
