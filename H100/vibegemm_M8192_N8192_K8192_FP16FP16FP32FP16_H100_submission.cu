// Compile Command:
// nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --expt-extended-lambda vibegemm_M8192_N8192_K8192_FP16FP16FP32FP16_H100_submission.cu -lcublas -lcuda -o vibegemm_M8192_N8192_K8192_FP16FP16FP32FP16_H100
//
// Run Command:
// ./vibegemm_M8192_N8192_K8192_FP16FP16FP32FP16_H100

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

__device__ __forceinline__ uint32_t smemPtrU32(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrierInit(uint64_t *bar, int threadCount, int transactionCount) {
    uint32_t barPtr = smemPtrU32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barPtr),
                 "r"(threadCount + transactionCount));
}

__device__ __forceinline__ void mbarrierInit(uint64_t *bar, uint32_t count) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(ptr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrierArrive(uint64_t *bar) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" : : "r"(ptr) : "memory");
}

__device__ __forceinline__ void mbarrierExpectTx(uint64_t *bar, uint32_t bytes) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 :
                 : "r"(ptr), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void mbarrierWait(uint64_t *bar, int phaseBit) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra.uni DONE;\n"
                 "bra.uni LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n"
                 :
                 : "r"(ptr), "r"(phaseBit)
                 : "memory");
}

__device__ __forceinline__ void arriveCluster(uint64_t *bar, uint32_t ctaId, uint32_t count = 1) {
    uint32_t smemAddr = smemPtrU32(bar);
    asm volatile("{\n\t"
                 ".reg .b32 remAddr32;\n\t"
                 "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                 "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                 "}"
                 :
                 : "r"(smemAddr), "r"(ctaId), "r"(count));
}

__device__ __forceinline__ void loadAsyncMulticast(half *dst, void const *const srcTma,
                                                   uint64_t *bar, int globalColIdx,
                                                   int globalRowIdx, uint16_t clusterMask) {
    uint64_t tmaPtr = reinterpret_cast<uint64_t>(srcTma);
    uint32_t mbarPtr = smemPtrU32(bar);
    uint32_t dstPtr = smemPtrU32(dst);
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes."
                 "multicast::cluster "
                 "[%0], [%1, {%3, %4, %5}], [%2], %6;"
                 :
                 : "r"(dstPtr), "l"(tmaPtr), "r"(mbarPtr), "n"(0), "r"(globalRowIdx),
                   "r"(globalColIdx / 64), "h"(clusterMask)
                 : "memory");
}

__device__ __forceinline__ void loadAsync(half *dst, void const *const srcTma, uint64_t *bar,
                                          int globalColIdx, int globalRowIdx) {
    uint64_t tmaPtr = reinterpret_cast<uint64_t>(srcTma);
    uint32_t mbarPtr = smemPtrU32(bar);
    uint32_t dstPtr = smemPtrU32(dst);
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%3, %4, %5}], [%2];"
                 :
                 : "r"(dstPtr), "l"(tmaPtr), "r"(mbarPtr), "n"(0), "r"(globalRowIdx),
                   "r"(globalColIdx / 64)
                 : "memory");
}

__device__ __forceinline__ void storeAsync(void const *dstTmaMap, half *src, int globalColIdx,
                                           int globalRowIdx) {
    uint64_t tmaPtr = reinterpret_cast<uint64_t>(dstTmaMap);
    uint32_t srcPtr = smemPtrU32(src);
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group "
                 "[%0, {%2, %3, %4}], [%1];"
                 :
                 : "l"(tmaPtr), "r"(srcPtr), "n"(0), "r"(globalRowIdx), "r"(globalColIdx / 64)
                 : "memory");
}

__device__ __forceinline__ uint64_t matrixDescriptorEncode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ __forceinline__ uint64_t makeSmemDesc(half *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= matrixDescriptorEncode(addr);
    desc |= matrixDescriptorEncode((uint64_t)16) << 16;
    desc |= matrixDescriptorEncode((uint64_t)1024) << 32;
    desc |= 1llu << 62;
    return desc;
}

__device__ __forceinline__ void warpgroupArrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroupCommitBatch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N> __device__ __forceinline__ void warpgroupWait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmmaM64N256K16(float d[][8], half *sA, half *sB) {
    uint64_t descA = makeSmemDesc(sA);
    uint64_t descB = makeSmemDesc(sB);
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%"
        "19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%"
        "36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%"
        "53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%"
        "70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%"
        "87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%"
        "103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%"
        "117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, "
        "%128,%129,%130,%131,%132,%133,%134;"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
          "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
          "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]), "+f"(d[8][0]), "+f"(d[8][1]),
          "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
          "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]),
          "+f"(d[9][6]), "+f"(d[9][7]), "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]),
          "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
          "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]),
          "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]), "+f"(d[12][0]), "+f"(d[12][1]),
          "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]),
          "+f"(d[12][7]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
          "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]), "+f"(d[14][0]),
          "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]),
          "+f"(d[14][6]), "+f"(d[14][7]), "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]),
          "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(descA), "l"(descB), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
          "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <uint32_t RegCount> __device__ __forceinline__ void warpgroupRegAlloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) : "memory");
}

template <uint32_t RegCount> __device__ __forceinline__ void warpgroupRegDealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) : "memory");
}

// ===================================
// 2. Tile-Level
// ===================================

template <int BM, int BN, int BK, int QSIZE, int WGMMA_M, int WGMMA_N, int WGMMA_K, int THREADS,
          int CLUSTER_M, int CLUSTER_N>
struct BlockShape {
    static constexpr int TileM = BM;
    static constexpr int TileN = BN;
    static constexpr int TileK = BK;
    static constexpr int QueueSize = QSIZE;
    static constexpr int WgmmaM = WGMMA_M;
    static constexpr int WgmmaN = WGMMA_N;
    static constexpr int WgmmaK = WGMMA_K;
    static constexpr int NumThreads = THREADS;
    static constexpr int ClusterM = CLUSTER_M;
    static constexpr int ClusterN = CLUSTER_N;
    static constexpr int ClusterSize = CLUSTER_M * CLUSTER_N;
};

template <int BM, int BN, int BK, int QSIZE> struct SharedStorage {
    alignas(128) half A[BM * BK * QSIZE];
    alignas(128) half B[BK * BN * QSIZE];
    alignas(128) half C[BN * BM];
    alignas(8) uint64_t full[QSIZE];
    alignas(8) uint64_t empty[QSIZE];
};

template <int BM, int BN, int TM = 16, int TN = 8> struct Schedule {
    int block, it;
    int totalBlocksM, totalBlocksN;
    int numSm;

    __device__ __forceinline__ Schedule(int M, int N, int _block, int _numSm)
        : block(_block), it(0), numSm(_numSm) {
        totalBlocksM = (M + BM - 1) / BM;
        totalBlocksN = (N + BN - 1) / BN;
    }

    __device__ __forceinline__ bool next(int &blockM, int &blockN) {
        constexpr int KTM = (TM > 0 ? TM : 1);
        constexpr int KTN = (TN > 0 ? TN : 1);

        int total = totalBlocksM * totalBlocksN;
        if (total <= 0)
            return false;

        int tilesM = (totalBlocksM + KTM - 1) / KTM;
        int tilesN = (totalBlocksN + KTN - 1) / KTN;
        int tilesTotal = tilesM * tilesN;
        int virtualTotal = tilesTotal * KTM * KTN;

        int num = it * numSm + block;
        if (num >= virtualTotal)
            return false;

        while (num < virtualTotal) {
            int curTile = num / (KTM * KTN);
            int curPos = num - curTile * (KTM * KTN);
            int tileM = curTile / tilesN;
            int tileN = curTile - tileM * tilesN;

            int m0 = tileM * KTM;
            int n0 = tileN * KTN;

            int dm = curPos / KTN;
            int dn = curPos - dm * KTN;

            blockM = m0 + dm;
            blockN = n0 + dn;
            ++it;

            if (blockM < totalBlocksM && blockN < totalBlocksN) {
                return true;
            }

            num = it * numSm + block;
        }

        return false;
    }
};

// ===================================
// 3. Collective-Level
// ===================================

template <int BlockMajorSize, int BlockMinorSize, bool Swizzle = true>
__host__ void createTensorMap(CUtensorMap *tmaMap, const half *gmemPtr, int globalHeight,
                              int globalWidth) {
    void *gmemAddress = (void *)gmemPtr;
    uint64_t gmemProbShape[5] = {64, (uint64_t)globalHeight, (uint64_t)(globalWidth / 64), 1, 1};
    uint64_t gmemProbStride[5] = {sizeof(half) * globalWidth, 64 * sizeof(half), 0, 0, 0};
    uint32_t smemBoxShape[5] = {64, (uint32_t)BlockMajorSize, (uint32_t)(BlockMinorSize / 64), 1,
                                1};
    uint32_t smemBoxStride[5] = {1, 1, 1, 1, 1};
    cuTensorMapEncodeTiled(tmaMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, gmemAddress, gmemProbShape,
                           gmemProbStride, smemBoxShape, smemBoxStride,
                           CU_TENSOR_MAP_INTERLEAVE_NONE,
                           Swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <int BlockMajorSize, int BlockMinorSize, bool Swizzle = true>
__host__ CUtensorMap createTensorMapValue(const half *src, int globalHeight, int globalWidth) {
    CUtensorMap tmaHost{};
    createTensorMap<BlockMajorSize, BlockMinorSize, Swizzle>(&tmaHost, src, globalHeight,
                                                             globalWidth);
    return tmaHost;
}

// ===================================
// 4. Kernel-Level
// ===================================

template <typename Shape>
__global__ void __launch_bounds__(Shape::NumThreads) __cluster_dims__(Shape::ClusterSize, 1, 1)
    matmulKernel(int M, int N, int K, const __grid_constant__ CUtensorMap tensorMapC,
                 const __grid_constant__ CUtensorMap tensorMapA,
                 const __grid_constant__ CUtensorMap tensorMapB) {
    constexpr int BM = Shape::TileM;
    constexpr int BN = Shape::TileN;
    constexpr int BK = Shape::TileK;
    constexpr int QSIZE = Shape::QueueSize;
    constexpr int WGMMA_M = Shape::WgmmaM;
    constexpr int WGMMA_N = Shape::WgmmaN;
    constexpr int WGMMA_K = Shape::WgmmaK;
    constexpr int NUM_THREADS = Shape::NumThreads;

    constexpr int numConsumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / numConsumers;

    const int wgIdx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;

    extern __shared__ __align__(128) uint8_t smemRaw[];
    auto &smem = *reinterpret_cast<SharedStorage<BM, BN, BK, QSIZE> *>(smemRaw);
    half *sA = smem.A;
    half *sB = smem.B;
    half *sC = smem.C;
    uint64_t *full = smem.full;
    uint64_t *empty = smem.empty;

    uint32_t rank;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(rank) :);

    const int numBlocksK = K / BK;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < QSIZE; ++i) {
            mbarrierInit(&full[i], 0, 1);
            mbarrierInit(&empty[i], 0, numConsumers * Shape::ClusterSize);
        }
    }

    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    const int clusterId = blockIdx.x / Shape::ClusterSize;
    const int clusterCount = gridDim.x / Shape::ClusterSize;
    Schedule<BM * Shape::ClusterM, BN * Shape::ClusterN, 16 / Shape::ClusterM, 8 / Shape::ClusterN>
        schedule(M, N, clusterId, clusterCount);

    uint32_t ctaRank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(ctaRank) :);
    uint32_t rankM = ctaRank / Shape::ClusterN;
    uint32_t rankN = ctaRank % Shape::ClusterN;

    if (wgIdx == 0) {
        constexpr int numRegs = (numConsumers <= 2 ? 24 : 32);
        warpgroupRegDealloc<numRegs>();

        if (tid == 0) {
            int p = 0, qidx = 0;
            uint32_t colMask = 0;
#pragma unroll
            for (int i = 0; i < Shape::ClusterM; ++i) {
                colMask |= (1 << (i * Shape::ClusterN));
            }

            int numBlockM, numBlockN;
            while (schedule.next(numBlockM, numBlockN)) {
                numBlockN = numBlockN * Shape::ClusterN + rankN;
                numBlockM = numBlockM * Shape::ClusterM + rankM;

                for (int blockKIter = 0; blockKIter < numBlocksK; ++blockKIter, ++qidx) {
                    if (qidx == QSIZE) {
                        qidx = 0;
                        p ^= 1;
                    }
                    mbarrierWait(&empty[qidx], p);

                    mbarrierExpectTx(&full[qidx], (BK * BN + BK * BM) * sizeof(half));

                    if constexpr (Shape::ClusterN > 1) {
                        uint32_t mask = ((1 << Shape::ClusterN) - 1) << (rankM * Shape::ClusterN);
                        if (rankN == 0) {
                            loadAsyncMulticast(&sA[qidx * BK * BM], &tensorMapA, &full[qidx],
                                               blockKIter * BK, numBlockM * BM, mask);
                        }
                    } else {
                        loadAsync(&sA[qidx * BK * BM], &tensorMapA, &full[qidx], blockKIter * BK,
                                  numBlockM * BM);
                    }

                    if constexpr (Shape::ClusterM > 1) {
                        if (rankM == 0) {
                            loadAsyncMulticast(&sB[qidx * BK * BN], &tensorMapB, &full[qidx],
                                               blockKIter * BK, numBlockN * BN, colMask << rankN);
                        }
                    } else {
                        loadAsync(&sB[qidx * BK * BN], &tensorMapB, &full[qidx], blockKIter * BK,
                                  numBlockN * BN);
                    }
                }
            }
        }
    } else {
        warpgroupRegAlloc<240>();

        const int activeConsumerIdx = wgIdx - 1;
        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];

#pragma unroll
        for (int qidx = 0; qidx < QSIZE; ++qidx) {
            if (tid < Shape::ClusterSize)
                arriveCluster(&empty[qidx], tid);
        }

        int p = 0, qidx = 0;
        int numBlockM, numBlockN;

        while (schedule.next(numBlockM, numBlockN)) {
            numBlockN = numBlockN * Shape::ClusterN + rankN;
            numBlockM = numBlockM * Shape::ClusterM + rankM;

            {
                if (qidx == QSIZE) {
                    qidx = 0;
                    p ^= 1;
                }
                mbarrierWait(&full[qidx], p);
                warpgroupArrive();

#pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half *wgmmaSa = sA + qidx * BK * BM +
                                    64 * (mIt + activeConsumerIdx * (B_WG_M / WGMMA_M)) * WGMMA_M;
                    half *wgmmaSb = sB + qidx * BK * BN;
                    {
                        wgmmaM64N256K16<0, 1, 1, 0, 0>(d[mIt], &wgmmaSa[0], &wgmmaSb[0]);
#pragma unroll
                        for (int kIt = 1; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmmaM64N256K16<1, 1, 1, 0, 0>(d[mIt], &wgmmaSa[kIt * WGMMA_K],
                                                           &wgmmaSb[kIt * WGMMA_K]);
                        }
                        wgmmaSa += 64 * BM;
                        wgmmaSb += 64 * BN;
                    }
#pragma unroll
                    for (int bk = 64; bk < BK; bk += 64) {
#pragma unroll
                        for (int kIt = 0; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmmaM64N256K16<1, 1, 1, 0, 0>(d[mIt], &wgmmaSa[kIt * WGMMA_K],
                                                           &wgmmaSb[kIt * WGMMA_K]);
                        }
                        wgmmaSa += 64 * BM;
                        wgmmaSb += 64 * BN;
                    }
                }
                warpgroupCommitBatch();
                warpgroupWait<0>();

                if (tid < Shape::ClusterSize)
                    arriveCluster(&empty[qidx], tid);
                ++qidx;
            }

            for (int blockKIter = 1; blockKIter < numBlocksK; ++blockKIter, ++qidx) {
                if (qidx == QSIZE) {
                    qidx = 0;
                    p ^= 1;
                }
                mbarrierWait(&full[qidx], p);
                warpgroupArrive();

#pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half *wgmmaSa = sA + qidx * BK * BM +
                                    64 * (mIt + activeConsumerIdx * (B_WG_M / WGMMA_M)) * WGMMA_M;
                    half *wgmmaSb = sB + qidx * BK * BN;

#pragma unroll
                    for (int bk = 0; bk < BK; bk += 64) {
#pragma unroll
                        for (int kIt = 0; kIt < 64 / WGMMA_K; ++kIt) {
                            wgmmaM64N256K16<1, 1, 1, 0, 0>(d[mIt], &wgmmaSa[kIt * WGMMA_K],
                                                          &wgmmaSb[kIt * WGMMA_K]);
                        }
                        wgmmaSa += 64 * BM;
                        wgmmaSb += 64 * BN;
                    }
                }
                warpgroupCommitBatch();
                warpgroupWait<0>();

                if (tid < Shape::ClusterSize)
                    arriveCluster(&empty[qidx], tid);
            }

            asm volatile("cp.async.bulk.wait_group 0;");

            const int lane = tid % 32;
            const int warp = tid / 32;
            const int row = warp * 16 + lane / 4;

            half *blockSc = sC + activeConsumerIdx * B_WG_M * BN;

            auto storeC = [&](int i, int j, float val) {
                blockSc[j * B_WG_M + i] = __float2half(val);
            };

#pragma unroll
            for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                int yo = mIt * WGMMA_M;
#pragma unroll
                for (int w = 0; w < WGMMA_N; w += 16) {
                    int col = w + 2 * (tid % 4);

                    storeC(row + yo, col, d[mIt][w / 16][0]);
                    storeC(row + 8 + yo, col, d[mIt][w / 16][2]);

                    storeC(row + yo, col + 1, d[mIt][w / 16][1]);
                    storeC(row + 8 + yo, col + 1, d[mIt][w / 16][3]);

                    storeC(row + yo, col + 8, d[mIt][w / 16][4]);
                    storeC(row + 8 + yo, col + 8, d[mIt][w / 16][6]);

                    storeC(row + yo, col + 9, d[mIt][w / 16][5]);
                    storeC(row + 8 + yo, col + 9, d[mIt][w / 16][7]);
                }
            }

            asm volatile("bar.sync 10, 256;\n");

            if (threadIdx.x == 128) {
                storeAsync(&tensorMapC, &sC[0], numBlockM * BM, numBlockN * BN);
                asm volatile("cp.async.bulk.commit_group;");
            }
        }

        warpgroupRegDealloc<240>();
    }
}

// ===================================
// 5. Device-Level
// ===================================

static CUtensorMap globalTmaMapA;
static CUtensorMap globalTmaMapB;
static CUtensorMap globalTmaMapC;
static bool tmaInitialized = false;
static int prevM = 0;
static int prevN = 0;
static int prevK = 0;

static void launch_custom_gemm(const half *d_A, const half *d_B, half *d_C, int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 256;
    constexpr int WGMMA_K = 16;
    constexpr int QSIZE = 3;
    constexpr int NUM_THREADS = 384;
    constexpr int NUM_SM = 128;
    constexpr int CLUSTER_M = 2;
    constexpr int CLUSTER_N = 1;

    using Shape = BlockShape<BM, BN, BK, QSIZE, WGMMA_M, WGMMA_N, WGMMA_K, NUM_THREADS,
                             CLUSTER_M, CLUSTER_N>;

    if (!tmaInitialized || M != prevM || N != prevN || K != prevK) {
        globalTmaMapA = createTensorMapValue<BM, BK>(d_A, M, K);
        globalTmaMapB = createTensorMapValue<BN, BK>(d_B, N, K);
        globalTmaMapC = createTensorMapValue<BN, BM, false>(d_C, N, M);

        tmaInitialized = true;
        prevM = M;
        prevN = N;
        prevK = K;
    }

    size_t sharedBytes = sizeof(SharedStorage<BM, BN, BK, QSIZE>);
    auto kptr = &matmulKernel<Shape>;

    cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);

    kptr<<<NUM_SM, NUM_THREADS, sharedBytes>>>(M, N, K, globalTmaMapC, globalTmaMapA,
                                               globalTmaMapB);
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
    std::printf("  M=N=K=8192, warmup_iters=10, benchmark_iters=30\n");
}

int main(int argc, char **argv) {
    int M = 8192;
    int N = 8192;
    int K = 8192;
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
