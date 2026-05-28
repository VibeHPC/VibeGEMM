// Compile Command:
// nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr --expt-extended-lambda vibegemm_M3328_N3328_K3328_FP16FP16FP32FP16_H100_submission.cu -lcublas -lcuda -o vibegemm_M3328_N3328_K3328_FP16FP16FP32FP16_H100
//
// Run Command:
// ./vibegemm_M3328_N3328_K3328_FP16FP16FP32FP16_H100

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

constexpr int CLUSTER_M = 2;
constexpr int CLUSTER_N = 1;
constexpr int CLUSTER_SIZE = CLUSTER_M * CLUSTER_N;

__device__ static __forceinline__ uint32_t smemPtrU32(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ static __forceinline__ void initBarrier(uint64_t *bar,
                                                   int thread_count,
                                                   int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
                 "r"(thread_count + transaction_count));
}

__device__ static __forceinline__ void mbarrierArrive(uint64_t *bar) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
                 :
                 : "r"(ptr)
                 : "memory");
}

__device__ static __forceinline__ void mbarrierExpectTx(uint64_t *bar,
                                                        uint32_t bytes) {
    uint32_t ptr = smemPtrU32(bar);
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(ptr), "r"(bytes)
        : "memory");
}

__device__ static __forceinline__ void mbarrierWait(uint64_t *bar,
                                                    int phase_bit) {
    uint32_t ptr = smemPtrU32(bar);
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
        : "r"(ptr), "r"(phase_bit)
        : "memory");
}

__device__ void arriveCluster(uint64_t *bar, uint32_t cta_id,
                              uint32_t count = 1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(count));
}

__device__ static inline uint64_t encodeMatrixDescriptor(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t makeSmemDescriptor(half *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
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
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
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

template <uint32_t RegCount>
__device__ void warpgroupRegAlloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(RegCount)
                 : "memory");
}

template <uint32_t RegCount>
__device__ void warpgroupRegDealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(RegCount)
                 : "memory");
}

__device__ static __forceinline__ void expectBytes(uint64_t *bar,
                                                   uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            bar_ptr),
        "r"(bytes));
}

__device__ static __forceinline__ void fenceProxyAsyncSharedCta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ===================================
// 2. Tile-Level
// ===================================

template <int BM, int BN, int BK, int QSIZE>
struct SingleBufferSmem {
    alignas(128) half A[BM * BK * QSIZE];
    alignas(128) half B[BK * BN * QSIZE];
    alignas(128) half C[BN * (BM + 16)];
    alignas(8) uint64_t full[QSIZE], empty[QSIZE];
    int space[128];
};

struct HilbertSchedule {
    const int *space;
    int it;

    __device__ __forceinline__ HilbertSchedule(const int *_space)
        : space(_space), it(0) {}

    __device__ __forceinline__ bool next(int &block_m, int &block_n) {
        if (it >= 128) return false;
        int v = space[it++];
        if (v < 0) return false;
        block_m = (v >> 16) & 0xFFFF;
        block_n = v & 0xFFFF;
        return true;
    }
};

// ===================================
// 3. Collective-Level
// ===================================

__device__ static inline void tmaFetchAsyncMulticast(
    half *dst, void const *const src_tma, uint64_t *bar, int global_col_idx,
    int global_row_idx, uint16_t cluster_mask) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma);
    uint32_t mbar_ptr = smemPtrU32(bar);
    uint32_t dst_ptr = smemPtrU32(dst);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4, %5}], [%2], "
        "%6;"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0),
          "r"(global_row_idx), "r"(global_col_idx / 64), "h"(cluster_mask)
        : "memory");
}

__device__ __forceinline__ void tmaLoad3d(half *smem_ptr,
                                          const CUtensorMap *map, int coord_k64,
                                          int coord_mn, uint64_t *bar) {
    uint32_t smem_u32 = smemPtrU32(smem_ptr);
    uint32_t bar_u32 = smemPtrU32(bar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :
        : "r"(smem_u32), "l"(map), "r"(bar_u32), "n"(0), "r"(coord_mn),
          "r"(coord_k64)
        : "memory");
}

template <int BlockMajorSize, int BlockMinorSize, bool swizzle = true,
          bool padded = false>
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

    uint32_t smem_box_shape[5] = {padded ? 72u : 64u, uint32_t(BlockMajorSize),
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

__device__ static inline void tmaFetchAsync(half *dst, void const *src_tma_map,
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

// ===================================
// 4. Kernel-Level
// ===================================

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM,
          int CLUSTER_M_, int CLUSTER_N_>
__global__ __launch_bounds__(NUM_THREADS) void __cluster_dims__(
    CLUSTER_M_ *CLUSTER_N_, 1, 1)
    gemm(int M, int N, int K, const __grid_constant__ CUtensorMap tensorMapC,
         const __grid_constant__ CUtensorMap tensorMapA,
         const __grid_constant__ CUtensorMap tensorMapB,
         const int *__restrict__ dspace) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;
    constexpr int CLUSTERS = CLUSTER_M_ * CLUSTER_N_;

    assert((M / BM) % CLUSTER_M_ == 0);
    assert((N / BN) % CLUSTER_N_ == 0);

    extern __shared__ __align__(128) uint8_t smem[];
    SingleBufferSmem<BM, BN, BK, QSIZE> &s =
        *reinterpret_cast<SingleBufferSmem<BM, BN, BK, QSIZE> *>(smem);
    half *sA = s.A, *sB = s.B, *sC = s.C;
    uint64_t *full = s.full, *empty = s.empty;

    uint32_t cluster_id;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_id) :);

    const int numBlocksK = K / BK;
    int wgIdx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            initBarrier(&full[i], 0, 1);
            initBarrier(&empty[i], 0, num_consumers * CLUSTERS);
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
    uint32_t rankM = rank / CLUSTER_N_;
    uint32_t rankN = rank % CLUSTER_N_;

    if (wgIdx == 0) {
        constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
        warpgroupRegDealloc<num_regs>();

        if (tid == 0) {
            int p = 0;
            int qIdx = 0;

            uint16_t bMask = 0;
            if constexpr (CLUSTER_M_ > 1) {
#pragma unroll
                for (int rm = 0; rm < CLUSTER_M_; ++rm) {
                    bMask |= (uint16_t)(1u << (rm * CLUSTER_N_ + rankN));
                }
            }

            int bIdxM, bIdxN;
            while (schedule.next(bIdxM, bIdxN)) {
                bIdxN = bIdxN * CLUSTER_N_ + rankN;
                bIdxM = bIdxM * CLUSTER_M_ + rankM;

                for (int kStep = 0; kStep < numBlocksK; ++kStep, ++qIdx) {
                    if (qIdx == QSIZE) {
                        qIdx = 0;
                        p ^= 1;
                    }
                    mbarrierWait(&empty[qIdx], p);

                    expectBytes(&full[qIdx],
                                (BK * BN + BK * BM) * sizeof(half));

                    if constexpr (CLUSTER_N_ > 1) {
                        uint32_t mask = ((1u << CLUSTER_N_) - 1u)
                                        << (rankM * CLUSTER_N_);
                        if (rankN == 0) {
                            tmaFetchAsyncMulticast(
                                &sA[qIdx * BK * BM], &tensorMapA, &full[qIdx],
                                kStep * BK, bIdxM * BM, (uint16_t)mask);
                        }
                    } else {
                        tmaFetchAsync(&sA[qIdx * BK * BM], &tensorMapA,
                                      &full[qIdx], kStep * BK, bIdxM * BM);
                    }

                    if constexpr (CLUSTER_M_ > 1) {
                        if (rankM == 0) {
                            tmaFetchAsyncMulticast(
                                &sB[qIdx * BK * BN], &tensorMapB, &full[qIdx],
                                kStep * BK, bIdxN * BN, bMask);
                        }
                    } else {
                        tmaFetchAsync(&sB[qIdx * BK * BN], &tensorMapB,
                                      &full[qIdx], kStep * BK, bIdxN * BN);
                    }
                }
            }
        }
    } else {
        warpgroupRegAlloc<240>();

        float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];

        --wgIdx;
        for (int qIdx = 0; qIdx < QSIZE; ++qIdx) {
            if (tid < CLUSTERS) arriveCluster(&empty[qIdx], tid);
        }

        int p = 0;
        int qIdx = 0;
        int bIdxM, bIdxN;

        while (schedule.next(bIdxM, bIdxN)) {
            bIdxN = bIdxN * CLUSTER_N_ + rankN;
            bIdxM = bIdxM * CLUSTER_M_ + rankM;

            {
                if (qIdx == QSIZE) {
                    qIdx = 0;
                    p ^= 1;
                }
                mbarrierWait(&full[qIdx], p);

                wgmmaFence();
#pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half *wgmmaSA =
                        sA + qIdx * BK * BM +
                        64 * (mIt + wgIdx * B_WG_M / WGMMA_M) * WGMMA_M;
                    half *wgmmaSB = sB + qIdx * BK * BN;

                    wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[mIt], &wgmmaSA[0],
                                                  &wgmmaSB[0]);
#pragma unroll
                    for (int kIt = 1; kIt < 64 / WGMMA_K; ++kIt) {
                        wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[mIt],
                                                      &wgmmaSA[kIt * WGMMA_K],
                                                      &wgmmaSB[kIt * WGMMA_K]);
                    }
                    wgmmaSA += 64 * BM;
                    wgmmaSB += 64 * BN;

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
            }

            for (int kStep = 1; kStep < numBlocksK; ++kStep, ++qIdx) {
                if (qIdx == QSIZE) {
                    qIdx = 0;
                    p ^= 1;
                }
                mbarrierWait(&full[qIdx], p);

                wgmmaFence();
#pragma unroll
                for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
                    half *wgmmaSA =
                        sA + qIdx * BK * BM +
                        64 * (mIt + wgIdx * B_WG_M / WGMMA_M) * WGMMA_M;
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
            }

            asm volatile("cp.async.bulk.wait_group 0;");

            constexpr int B_WG_M_PADDED = B_WG_M + 8;
            int lane = tid % 32;
            int warp = tid / 32;

            half *blockSC = sC + wgIdx * B_WG_M_PADDED * BN;

            uint32_t tidOffset = warp * 16 + (lane % 8) * B_WG_M_PADDED;
            tidOffset += (lane / 16) * B_WG_M_PADDED * 8 + (lane & 8);
            uint32_t baseAddr =
                static_cast<uint32_t>(__cvta_generic_to_shared(blockSC)) +
                tidOffset * sizeof(half);

#pragma unroll
            for (int mIt = 0; mIt < B_WG_M / WGMMA_M; ++mIt) {
#pragma unroll
                for (int w = 0; w < WGMMA_N; w += 16) {
                    uint32_t addr =
                        baseAddr +
                        (w * B_WG_M_PADDED + mIt * WGMMA_M) * sizeof(half);

                    half d_half[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        d_half[k] = (half)(d[mIt][w / 16][k]);
                    }
                    uint32_t *data_ptr = reinterpret_cast<uint32_t *>(d_half);

                    asm volatile(
                        "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 "
                        "[%0], {%1, %2, %3, %4};"
                        :
                        : "r"(addr), "r"(data_ptr[0]), "r"(data_ptr[1]),
                          "r"(data_ptr[2]), "r"(data_ptr[3]));
                }
            }

            asm volatile("bar.sync %0, 128;\n" ::"r"(wgIdx + 2) : "memory");

            if (tid == 0) {
                fenceProxyAsyncSharedCta();
                storeAsync(&tensorMapC, blockSC, bIdxM * BM + wgIdx * B_WG_M,
                           bIdxN * BN);
                asm volatile("cp.async.bulk.commit_group;");
            }
        }

        warpgroupRegDealloc<240>();
    }
}

// ===================================
// 5. Device-Level
// ===================================

static int *globalDspaceBase = nullptr;
static int spaceMTilesBase = 0;
static int spaceNTilesBase = 0;
static int spaceNumClustersBase = 0;

static CUtensorMap globalTmaMapABase;
static CUtensorMap globalTmaMapBBase;
static CUtensorMap globalTmaMapCBase;
static int prevMBase = 0, prevNBase = 0, prevKBase = 0;

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

static int *generateHilbertHostData(int m_tiles, int n_tiles,
                                    int num_clusters) {
    const int depth = 128;
    const int total_tasks = m_tiles * n_tiles;

    const int side = nextPow2((m_tiles > n_tiles) ? m_tiles : n_tiles);
    const int max_d = side * side;

    int *h = (int *)malloc((size_t)num_clusters * depth * sizeof(int));
    for (int i = 0; i < num_clusters * depth; ++i) h[i] = -1;

    int task_id = 0;
    for (int d = 0; d < max_d && task_id < total_tasks; ++d) {
        int x, y;
        d2xyHilbert(side, d, x, y);
        if (y < m_tiles && x < n_tiles) {
            int block_m = y;
            int block_n = x;

            int cluster = task_id % num_clusters;
            int pos = task_id / num_clusters;
            if (pos < depth) {
                h[cluster * depth + pos] = (block_m << 16) | (block_n & 0xFFFF);
            }
            ++task_id;
        }
    }
    return h;
}

static void launch_custom_gemm(const half *d_A, const half *d_B, half *d_C,
                               int M, int N, int K) {
    constexpr int BM = 128, BN = 256, BK = 64, QSIZE = 3, NUM_THREADS = 384,
                  NUM_SM = 128;
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;

    if (prevMBase != M || prevNBase != N || prevKBase != K) {
        globalTmaMapABase = createTmaMapDevice<BM, BK>(d_A, M, K);
        globalTmaMapBBase = createTmaMapDevice<BN, BK>(d_B, N, K);
        globalTmaMapCBase =
            createTmaMapDevice<BN, BM / num_consumers, false, true>(d_C, N, M);
        prevMBase = M;
        prevNBase = N;
        prevKBase = K;
    }

    const int NUM_CLUSTERS = NUM_SM / CLUSTERS;
    const int m_tiles = CEIL_DIV(M, BM * CLUSTER_M);
    const int n_tiles = CEIL_DIV(N, BN * CLUSTER_N);

    const bool need_regen =
        (globalDspaceBase == nullptr) || (spaceMTilesBase != m_tiles) ||
        (spaceNTilesBase != n_tiles) || (spaceNumClustersBase != NUM_CLUSTERS);

    if (need_regen) {
        if (globalDspaceBase) {
            CUDA_CHECK(cudaFree(globalDspaceBase));
            globalDspaceBase = nullptr;
        }

        int *h_space = generateHilbertHostData(m_tiles, n_tiles, NUM_CLUSTERS);
        CUDA_CHECK(cudaMalloc((void **)&globalDspaceBase,
                              (size_t)NUM_CLUSTERS * 128 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(globalDspaceBase, h_space,
                              (size_t)NUM_CLUSTERS * 128 * sizeof(int),
                              cudaMemcpyHostToDevice));
        free(h_space);

        spaceMTilesBase = m_tiles;
        spaceNTilesBase = n_tiles;
        spaceNumClustersBase = NUM_CLUSTERS;
    }

    size_t sharedBytes = sizeof(SingleBufferSmem<BM, BN, BK, QSIZE>);
    auto kptr =
        &gemm<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M, CLUSTER_N>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes));

    kptr<<<NUM_SM, NUM_THREADS, sharedBytes>>>(
        M, N, K, globalTmaMapCBase, globalTmaMapABase, globalTmaMapBBase,
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
    std::printf("  M=N=K=3328, warmup_iters=10, benchmark_iters=30\n");
}

int main(int argc, char **argv) {
    int M = 3328;
    int N = 3328;
    int K = 3328;
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
