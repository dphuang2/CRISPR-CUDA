#include "common/fmt.hpp"
#include "common/utils.hpp"
#include <string>
#include <iostream>
#include <set>
#include <assert.h>
#include <fstream>

using namespace std;

/*
 *these files should exist from the docker image and uploaded build folder
 */
#define GENOME_FILE_PATH "/data/hg38.txt"
#define GUIDES_FILE_PATH "./src/guides.txt"
/*
 *akin to string.format in Python
 */
#define PRINT(...) LOG(info, string(fmt::format(__VA_ARGS__)))
#define NUM_GUIDES 46
#define GUIDE_SIZE 20
/*
 *Add another byte to account for the null byte that is appended during
 *ifstream.getline
 */
#define GUIDE_BUFFER_SIZE (GUIDE_SIZE + 1)
#define SIZE_OF_GUIDES 966 // NUM_GUIDES * GUIDE_BUFFER_SIZE
/*
 *This is the common edit distance standard for the sgRNA guide
 */
#define EDIT_DISTANCE_THRESHOLD 4
/*
 *This is used for predicting the maximum number of matches we might possibly
 *have
 */
#define MATCHES_PER_GUIDE 8000


#define TILE_WIDTH 512
#define TILE_WIDTH_SHUF 256
#define WARP_SIZE 32
#define NUM_WARPS (TILE_WIDTH / WARP_SIZE)

#define OUTPUT_PER_THREAD_REG 8
#define OUTPUT_PER_THREAD_SHUF 4
#define OUTPUT_PER_THREAD_SHARED 8

// ceil((WARP_SIZE * OUTPUT_PER_THREAD + GUIDE_SIZE - 1) / WARP_SIZE) = 9
#define LOCAL_REGISTER_SIZE (OUTPUT_PER_THREAD_SHUF + 1)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 1024
#define BQ_ELEM_PER_THREAD (BQ_CAPACITY / TILE_WIDTH)

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 32
#define WQ_ELEM_PER_THREAD (WQ_CAPACITY / WARP_SIZE)


#define CUDA_CHECK(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line) {
    if (result != cudaSuccess) {
        PRINT("{}@{}: CUDA Runtime Error: {}", file, line, cudaGetErrorString(result));
        exit(-1);
    }
}

typedef set<tuple<int, int64_t> > results_t;

/***************************************************************
  I/O HELPER FUNCTIONS
***************************************************************/

char * read_genome(string filename, int64_t * genome_length) {
    /*
     * ios::binary means to read file as is
     * ios::ate means start file pointer at the end so we can get file size
     */
    ifstream file(filename, ios::binary | ios::ate);

    if (!file) {
        PRINT("Could not open file: {}", filename);
        PRINT("Error code: {}", strerror(errno));
    }

    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    *genome_length = size;

    char * buffer = (char * ) malloc(size);
    if (file.read(buffer, size)) {
        return buffer;
    }

    PRINT("Unable to read {}", filename);
    exit(EXIT_FAILURE);
}

char * read_guides(string filename, int * num_guides) {
    char * guides;
    ifstream file(filename, ios::binary);

    if (!file) {
        PRINT("Could not open file: {}", filename);
        PRINT("Error code: {}", strerror(errno));
        exit(EXIT_FAILURE);
    }

    // Count number of lines
    string line;
    for (*num_guides = 0; getline(file, line); (*num_guides)++);
    // Move file pointer back to beginning
    file.clear(); // https://stackoverflow.com/questions/5343173/returning-to-beginning-of-file-after-getline
    file.seekg(0, file.beg);

    guides = (char *) malloc(*num_guides * GUIDE_BUFFER_SIZE);

    for(int line_number = 0; line_number < *num_guides; line_number++) {
        file.getline(&guides[line_number * GUIDE_BUFFER_SIZE],
                GUIDE_BUFFER_SIZE);
    }

    return guides;
}

void assert_results_equal(results_t cpuResults, int64_t * gpuResults, int gpuNumResults) {
    /*
     *For comparing the gpu results and the cpuResults
     */
    string failMessage = "Results were not equal, sad D:";
    string successMessage = "Results were equal, yay!";
    results_t seen;
    if (gpuNumResults != cpuResults.size()) {
        PRINT("{}", failMessage);
        return;
    }
    for (int i = 0; i < gpuNumResults; i++) {
        int gpuResultsIdx = i * 2;
        int guideIdx = (int) gpuResults[gpuResultsIdx];
        int64_t genomeIdx = gpuResults[gpuResultsIdx + 1];
        tuple<int, int64_t> match = make_tuple(guideIdx, genomeIdx);
        if (!cpuResults.count(match) || seen.count(match)) {
            PRINT("{}", failMessage);
            return;
        } else {
            seen.insert(match);
        }
    }
    PRINT("{}", successMessage);
}

void print_results(int64_t * result, int64_t result_size) {
    for (int i = 0; i < result_size; i++) {
        int64_t result_idx = i * 2;
        int64_t guide_idx = result[result_idx];
        int64_t sequence_idx = result[result_idx + 1];
        PRINT("guide_idx: {}; sequence_idx: {}", guide_idx, sequence_idx);
    }
}

void estimate_total_time(float msec, int64_t genome_length, int64_t genome_length_test) {
    float multiplier = float(genome_length) / genome_length_test;
    msec *= multiplier;
    float sec = msec / 100; // milliseconds to seconds
    float min = sec / 60; // seconds to minutes
    float hr = min / 60; // minutes to hours
    PRINT("Estimated total time: {} hours ({} minutes)", hr, min);

}

/***************************************************************
  NAIVE CPU IMPLEMENTATION
***************************************************************/

results_t naive_cpu_guide_matching(char * genome, int64_t genome_length, char * guides, int num_guides) {
    results_t results;
    char * guide;
    int mismatches;
    int64_t i;
    int j, k;

    for (i = 0; i < genome_length; i++) {
        for (j = 0; j < num_guides; j++) {
            guide = guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE; k++)
                mismatches += genome[i + k] != guide[k];

            if (mismatches <= EDIT_DISTANCE_THRESHOLD)
                results.insert(make_tuple(j, i));
        }
    }
    return results;
}

/***************************************************************
  GPU HELPER FUNCTIONS
***************************************************************/

__device__ bool insert_into_queue(
        int64_t * queue,
        int64_t capacity,
        int * counter,
        int guide_idx,
        int64_t sequence_idx
        )
{
    int results_idx = atomicAdd(counter, 1);
    if (results_idx < capacity) {
        int queue_idx = results_idx * 2;
        queue[queue_idx] = (int64_t) guide_idx;
        queue[queue_idx + 1] = sequence_idx;
        return true;
    }
    return false;
}

/***************************************************************
  NAIVE GPU IMPLEMENTATION
***************************************************************/

__global__ void naive_gpu_kernel(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    char * guide;
    int64_t i;
    int mismatches;
    int j, k;

    for (i = tid; i < genome_length - GUIDE_SIZE - 1; i += gridDim.x * blockDim.x) {
        for (j = 0; j < num_guides; j++) {
            guide = guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE; k++)
                mismatches += genome[i + k] != guide[k];

            if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                insert_into_queue(results, sizeOfResults, numResults, j, i);
            }
        }
    }
}

/***************************************************************
  GPU-AWARE IMPLEMENTATION

  Possible optimizations:
  - Constant memory for the guides
  - Shared memory for the genome
  - Warp-queue like structure for the results to avoid atomic operation
  - Striding for data coalescing (each thread processes a portion of the genome)
  - Thread coarsening to take advantage of registers
  - Pre-process the guides into hashes, use rolling hash on genome.
***************************************************************/

/***************************************************************
  REGISTER TILING GPU IMPLEMENTATION
***************************************************************/

__global__ void register_tiling_gpu_kernel(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    char * guide;
    int64_t i;
    int mismatches;
    int j, k;
    char rc[GUIDE_SIZE];

    for (i = tid; i < genome_length - GUIDE_SIZE - 1; i += gridDim.x * blockDim.x) {
        for (j = 0; j < GUIDE_SIZE; j++)
            rc[j] = genome[i + j];

        for (j = 0; j < num_guides; j++) {
            guide = guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE; k++)
                mismatches += rc[k] != guide[k];

            if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                insert_into_queue(results, sizeOfResults, numResults, j, i);
            }
        }
    }
}

/***************************************************************
  COARSENED REGISTER TILING GPU IMPLEMENTATION
***************************************************************/

__global__ void coarsened_register_tiling_gpu_kernel(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    char * guide;
    int64_t i;
    int mismatches;
    int n, j, k;
    char rc[GUIDE_SIZE + OUTPUT_PER_THREAD_REG];

    for (i = OUTPUT_PER_THREAD_REG * tid; i < genome_length - GUIDE_SIZE - 1 - OUTPUT_PER_THREAD_REG;
            i += OUTPUT_PER_THREAD_REG * gridDim.x * blockDim.x) {

        for (j = 0; j < GUIDE_SIZE + OUTPUT_PER_THREAD_REG; j++)
            rc[j] = genome[i + j];

        for (n = 0; n < OUTPUT_PER_THREAD_REG; n++) {
            for (j = 0; j < num_guides; j++) {
                guide = guides + (j * GUIDE_BUFFER_SIZE);

                mismatches = 0;
                for (k = 0; k < GUIDE_SIZE; k++)
                    mismatches += rc[k + n] != guide[k];

                if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                    insert_into_queue(results, sizeOfResults, numResults, j, i + n);
                }
            }
        }
    }
}

/***************************************************************
  COARSENED REGISTER TILING + SHARED MEMORY GUIDES GPU IMPLEMENTATION
***************************************************************/

__global__ void coarsened_register_tiling_shared_guides_gpu_kernel(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tx = threadIdx.x;
    int64_t tid = blockDim.x * blockIdx.x + tx;
    char * guide;
    int64_t i;
    int mismatches;
    int n, j, k;
    char rc[GUIDE_SIZE + OUTPUT_PER_THREAD_REG];

    __shared__ char shared_guides[SIZE_OF_GUIDES];
    for (i = tx; i < SIZE_OF_GUIDES; i += blockDim.x)
        shared_guides[i] = guides[i];
    __syncthreads();

    for (i = OUTPUT_PER_THREAD_REG * tid; i < genome_length - GUIDE_SIZE - 1 - OUTPUT_PER_THREAD_REG;
            i += OUTPUT_PER_THREAD_REG * gridDim.x * blockDim.x) {

        for (j = 0; j < GUIDE_SIZE + OUTPUT_PER_THREAD_REG; j++)
            rc[j] = genome[i + j];

        for (n = 0; n < OUTPUT_PER_THREAD_REG; n++) {
            for (j = 0; j < num_guides; j++) {
                guide = shared_guides + (j * GUIDE_BUFFER_SIZE);

                mismatches = 0;
                for (k = 0; k < GUIDE_SIZE; k++)
                    mismatches += rc[k + n] != guide[k];

                if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                    insert_into_queue(results, sizeOfResults, numResults, j, i + n);
                }
            }
        }
    }
}

/***************************************************************
  COARSENED REGISTER TILING W/ SHUFFLE + SHARED MEMORY GUIDES GPU IMPLEMENTATION
***************************************************************/

__global__ void coarsened_register_tiling_shuffle_shared_guides_gpu_kernel(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tx = threadIdx.x;
    char * guide;
    int mismatches;
    int64_t k;
    int i, j, n;

    __shared__ char shared_guides[SIZE_OF_GUIDES];
    for(i = tx; i < SIZE_OF_GUIDES; i += blockDim.x)
        shared_guides[i] = guides[i];
    __syncthreads();

    // Declare local register cache
    char rc[LOCAL_REGISTER_SIZE];

    // ID of thread within its own warp
    int64_t local_id = tx % WARP_SIZE;

    // The first index of output element computed by this warp.
    int64_t start_of_warp = (blockIdx.x * blockDim.x + WARP_SIZE * (tx / WARP_SIZE)) * OUTPUT_PER_THREAD_SHUF; 

    // The Id of the thread in the scope of the grid.
    int64_t global_id = start_of_warp + local_id;

    if (global_id >= genome_length)
        return;

#pragma unroll
    for (i = 0 ; i < LOCAL_REGISTER_SIZE; i++)
    {
        int64_t genome_idx = global_id + WARP_SIZE * i;
        if (genome_idx >= genome_length) {
            continue;
        }
        rc[i] = genome[genome_idx];
    }

    bool warp_has_inactive = genome_length - start_of_warp < WARP_SIZE;

    int num_inactive_threads = warp_has_inactive ? start_of_warp + WARP_SIZE - genome_length : 0;

    unsigned mask = (0xffffffff) >> (num_inactive_threads);

    for (j = 0; j < num_guides; j++) {
        guide = shared_guides + (j * GUIDE_BUFFER_SIZE);

#pragma unroll
        for (n = 0; n < OUTPUT_PER_THREAD_SHUF; n++) {
            char to_share = rc[n];
            mismatches = 0;

#pragma unroll
            for (k = 0; k < GUIDE_SIZE; k++) {
                // "& (WARP_SIZE - 1)" is equal to modulo WARP_SIZE
                mismatches += __shfl_sync(mask, to_share, (local_id + k) & (WARP_SIZE - 1)) != guide[k];
                to_share += (k == local_id) * (rc[n + 1] - rc[n]);
            }

            if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                insert_into_queue(results, sizeOfResults, numResults, j, global_id + n * WARP_SIZE);
            }
        }
    }
}

/***************************************************************
  COARSENED REGISTER TILING + WARP QUEUE + SHARED MEMORY GUIDES GPU IMPLEMENTATION
***************************************************************/

__global__ void coarsened_register_tiling_shared_guides_wq_gpu_kernel(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tx = threadIdx.x;
    int64_t tid = blockDim.x * blockIdx.x + tx;
    char * guide;
    int64_t i;
    int mismatches;
    int n, j, k;
    int warpIdx = threadIdx.x / WARP_SIZE;
    int queueIdx;
    char rc[GUIDE_SIZE + OUTPUT_PER_THREAD_REG];

    __shared__ char shared_guides[SIZE_OF_GUIDES];

    __shared__ int64_t blockQueue[BQ_CAPACITY * 2];
    __shared__ int64_t warpQueue[NUM_WARPS][WQ_CAPACITY * 2];

    __shared__ int blockQueueSize[1];
    __shared__ int warpQueueSize[NUM_WARPS];
    __shared__ int blockQueueOffset[NUM_WARPS];
    __shared__ int globalQueueOffset[1];

    // Reset shared memory cause its set for some reason
    if (tx == 0) {
        blockQueueSize[0] = 0;
        globalQueueOffset[0] = 0;
    }

    if (tx < NUM_WARPS) {
        warpQueueSize[tx] = 0;
        blockQueueOffset[tx] = 0;
    }

    for (i = tx; i < SIZE_OF_GUIDES; i += blockDim.x)
        shared_guides[i] = guides[i];
    __syncthreads();

    for (i = OUTPUT_PER_THREAD_REG * tid; i < genome_length - GUIDE_SIZE - 1 - OUTPUT_PER_THREAD_REG;
            i += OUTPUT_PER_THREAD_REG * gridDim.x * blockDim.x) {

        for (j = 0; j < GUIDE_SIZE + OUTPUT_PER_THREAD_REG; j++)
            rc[j] = genome[i + j];

        for (n = 0; n < OUTPUT_PER_THREAD_REG; n++) {
            for (j = 0; j < num_guides; j++) {
                guide = shared_guides + (j * GUIDE_BUFFER_SIZE);

                mismatches = 0;
                for (k = 0; k < GUIDE_SIZE; k++)
                    mismatches += rc[k + n] != guide[k];

                if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                    // Insert into warp queue
                    if (!insert_into_queue(warpQueue[warpIdx], WQ_CAPACITY,
                                &warpQueueSize[warpIdx], j, i + n)) {
                        // Insert into block queue
                        if(!insert_into_queue(blockQueue, BQ_CAPACITY,
                                    blockQueueSize, j, i + n)) {
                            // Insert into global queue
                            insert_into_queue(results, sizeOfResults,
                                    numResults, j, i + n);
                        }
                    }
                }
            }
        }
    }

      __syncthreads();

      // Perform scan on first thread
      if (threadIdx.x == 0 ) {
          for (i = 1; i < NUM_WARPS; i++) 
              blockQueueOffset[i] = blockQueueOffset[i - 1] + warpQueueSize[i - 1];
          blockQueueSize[0] = blockQueueOffset[NUM_WARPS - 1] + warpQueueSize[NUM_WARPS - 1];
          globalQueueOffset[0] = atomicAdd(numResults, blockQueueSize[0]);
      }

      // warp queue -> block queue data transfer
      int warpThreadIdx = threadIdx.x % WARP_SIZE;
      int blockQueueIdx, warpQueueIdx;
      for (i = warpThreadIdx; i < WQ_CAPACITY; i += WQ_ELEM_PER_THREAD) {
          if (i < warpQueueSize[warpIdx]) {
              blockQueueIdx = 2 * (blockQueueOffset[warpIdx] + i);
              warpQueueIdx = 2 * i;
              
              blockQueue[blockQueueIdx] = warpQueue[warpIdx][warpQueueIdx];
              blockQueue[blockQueueIdx + 1] = warpQueue[warpIdx][warpQueueIdx + 1];
          }
      }

      __syncthreads();

      // block queue -> global queue data transfer
      queueIdx = threadIdx.x * BQ_ELEM_PER_THREAD;
      int globalQueueIdx;
      for (i = 0; i < BQ_ELEM_PER_THREAD; i++) {
          if (queueIdx + i < blockQueueSize[0]) {
              globalQueueIdx = 2 * (globalQueueOffset[0] + queueIdx + i);
              blockQueueIdx = 2 * (queueIdx + i);
              results[globalQueueIdx] = blockQueue[blockQueueIdx];
              results[globalQueueIdx + 1] = blockQueue[blockQueueIdx + 1];
          }
      }
}

/***************************************************************
  SHARED MEMORY & CONSTANT MEMORY GPU IMPLEMENTATION
***************************************************************/

__constant__ char constant_guides[SIZE_OF_GUIDES];
__global__ void shared_constant_memory_gpu_kernel(
        char * genome,
        int64_t genome_length,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tx = threadIdx.x;
    int64_t tid = blockDim.x * blockIdx.x + tx;
    char * guide;
    int64_t i, b_start;
    int mismatches;
    int j, k;

    __shared__ char shared_genome[TILE_WIDTH * 2];

    for (i = tid; i < genome_length - GUIDE_SIZE - 1;
            i += gridDim.x * blockDim.x) {

        b_start = (i / blockDim.x) * blockDim.x;

        if (b_start + tx < genome_length)
            shared_genome[tx] = genome[b_start + tx];
        if (b_start + tx + TILE_WIDTH < genome_length)
            shared_genome[tx + TILE_WIDTH] = genome[b_start + tx + TILE_WIDTH];
        __syncthreads();

        for (j = 0; j < num_guides; j++) {
            guide = constant_guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE; k++)
                mismatches += shared_genome[tx + k] != guide[k];

            if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                insert_into_queue(results, sizeOfResults, numResults, j, i);
            }
        }

        __syncthreads();
    }
}

/***************************************************************
  SHARED MEMORY GPU IMPLEMENTATION
***************************************************************/

__global__ void shared_memory_gpu_kernel(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tx = threadIdx.x;
    int64_t tid = blockDim.x * blockIdx.x + tx;
    char * guide;
    int64_t i, b_start;
    int mismatches;
    int j, k;

    __shared__ char shared_genome[TILE_WIDTH * 2];
    __shared__ char shared_guides[SIZE_OF_GUIDES];

    for (i = tx; i < SIZE_OF_GUIDES; i += blockDim.x)
        shared_guides[i] = guides[i];

    for (i = tid; i < genome_length - GUIDE_SIZE - 1;
            i += gridDim.x * blockDim.x) {

        b_start = (i / blockDim.x) * blockDim.x + tx;

        if (b_start < genome_length)
            shared_genome[tx] = genome[b_start];
        if (b_start + TILE_WIDTH < genome_length)
            shared_genome[tx + TILE_WIDTH] = genome[b_start + TILE_WIDTH];

        __syncthreads();

        for (j = 0; j < num_guides; j++) {
            guide = shared_guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE; k++)
                mismatches += shared_genome[tx + k] != guide[k];

            if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                insert_into_queue(results, sizeOfResults, numResults, j, i);
            }
        }

        __syncthreads();
    }
}

/***************************************************************
  COARSENED REGISTER TILING SHARED MEMORY GPU IMPLEMENTATION
***************************************************************/

__global__ void coarsened_register_shared_mem_gpu_kernel(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * results,
        int * numResults,
        int64_t sizeOfResults)
{
    int64_t tx = threadIdx.x;
    int64_t tid = blockDim.x * blockIdx.x + tx;
    char * guide;
    int64_t i, b_start;
    int mismatches;
    int j, k, n;
    char rc[GUIDE_SIZE + OUTPUT_PER_THREAD_SHARED];

    __shared__ char shared_genome[TILE_WIDTH * (OUTPUT_PER_THREAD_SHARED + 1)];
    __shared__ char shared_guides[SIZE_OF_GUIDES];

    for (i = tx; i < SIZE_OF_GUIDES; i++)
        shared_guides[i] = guides[i];

    for (i = tid * OUTPUT_PER_THREAD_SHARED; i < genome_length - GUIDE_SIZE - 1 - OUTPUT_PER_THREAD_SHARED;
            i += OUTPUT_PER_THREAD_SHARED * gridDim.x * blockDim.x) {

        b_start = (i / (blockDim.x * OUTPUT_PER_THREAD_SHARED)) * blockDim.x * OUTPUT_PER_THREAD_SHARED + tx;

        for (n = 0; n < OUTPUT_PER_THREAD_SHARED + 1; n++) {
            if (b_start + TILE_WIDTH * n < genome_length)
                shared_genome[tx + TILE_WIDTH * n] = genome[b_start +
                    TILE_WIDTH * n];
        }
        __syncthreads();

        for (n = 0; n < GUIDE_SIZE + OUTPUT_PER_THREAD_SHARED; n++)
            rc[n] = shared_genome[OUTPUT_PER_THREAD_SHARED * tx + n];

        for (n = 0; n < OUTPUT_PER_THREAD_SHARED; n++) {
            for (j = 0; j < num_guides; j++) {
                guide = shared_guides + (j * GUIDE_BUFFER_SIZE);
                mismatches = 0;
                for (k = 0; k < GUIDE_SIZE; k++)
                    mismatches += rc[n + k] != guide[k];

                if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                    insert_into_queue(results, sizeOfResults, numResults, j, i + n);
                }
            }
        }

        __syncthreads();
    }
}

/***************************************************************
  GPU KERNEL LAUNCHER
***************************************************************/

enum methods {
    naive,
    s_memory,
    s_c_memory,
    coarsened_s_r_memory,
    register_tiling,
    coarsened_register_tiling,
    coarsened_register_tiling_s_g,
    coarsened_register_tiling_s_g_wq,
    coarsened_register_tiling_s_g_shuffle
};
static const string method_strings[] = {
    "Naive",
    "Shared Genome & Guides",
    "Shared Genome + Constant Guides",
    "Coarsened Shared Genome & Guides + Register Cache",
    "Register cache",
    "Coarsened Register Cache",
    "Coarsened Register Cache + Shared Guides",
    "Coarsened Register Cache + Shared Guides + Warp Queue",
    "Coarsened Register Cache with shuffle + Shared Guides"
};
void gpu_guide_matching(
        char * genome,
        int64_t genome_length,
        char * guides,
        int num_guides,
        int64_t * hostResults,
        int * hostNumResults,
        int64_t sizeOfResults,
        methods method)
{
    PRINT("genome_test_length: {}", genome_length);
    int64_t * deviceResults;
    char * deviceGenome;
    char * deviceGuides;
    int * deviceNumResults;

    timer_start("Allocating GPU Memory");
    CUDA_CHECK(cudaMalloc((void **) &deviceResults, sizeOfResults));
    CUDA_CHECK(cudaMalloc((void **) &deviceGenome, genome_length));
    CUDA_CHECK(cudaMalloc((void **) &deviceGuides, num_guides * GUIDE_BUFFER_SIZE));
    CUDA_CHECK(cudaMalloc((void **) &deviceNumResults, sizeof(int)));
    timer_stop();

    timer_start("Copying output from host to device");
    CUDA_CHECK(cudaMemcpy(deviceGenome, genome, genome_length, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceGuides, guides, num_guides * GUIDE_BUFFER_SIZE, cudaMemcpyHostToDevice));
    timer_stop();

    dim3 dimGridNaive(ceil((genome_length) / double(TILE_WIDTH)));
    dim3 dimBlockNaive(TILE_WIDTH);

    dim3 dimGridCoarsenedReg(ceil(genome_length / double(TILE_WIDTH * OUTPUT_PER_THREAD_REG)));
    dim3 dimGridCoarsenedShared(ceil(genome_length / double(TILE_WIDTH * OUTPUT_PER_THREAD_SHARED)));

    dim3 dimBlockSHUF(TILE_WIDTH_SHUF);
    dim3 dimGridSHUF(ceil(genome_length / double(TILE_WIDTH_SHUF * OUTPUT_PER_THREAD_SHUF)));

    timer_start("Executing kernel w/ " + method_strings[method]);
    switch (method) {
        case naive:
            naive_gpu_kernel<<<dimGridNaive, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case s_c_memory:
            CUDA_CHECK(cudaMemcpyToSymbol(constant_guides, deviceGuides, SIZE_OF_GUIDES));
            shared_constant_memory_gpu_kernel<<<dimGridNaive, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case s_memory:
            shared_memory_gpu_kernel<<<dimGridNaive, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case coarsened_s_r_memory:
            coarsened_register_shared_mem_gpu_kernel<<<dimGridCoarsenedShared, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case register_tiling:
            register_tiling_gpu_kernel<<<dimGridNaive, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case coarsened_register_tiling:
            coarsened_register_tiling_gpu_kernel<<<dimGridCoarsenedReg, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case coarsened_register_tiling_s_g:
            coarsened_register_tiling_shared_guides_gpu_kernel<<<dimGridCoarsenedReg, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case coarsened_register_tiling_s_g_wq:
            coarsened_register_tiling_shared_guides_wq_gpu_kernel<<<dimGridCoarsenedReg, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case coarsened_register_tiling_s_g_shuffle:
            coarsened_register_tiling_shuffle_shared_guides_gpu_kernel<<<dimGridSHUF, dimBlockSHUF>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
    }
    timer_stop();

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    timer_start("Copying memory from device to host");
    CUDA_CHECK(cudaMemcpy(hostNumResults, deviceNumResults, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostResults, deviceResults, sizeOfResults, cudaMemcpyDeviceToHost));
    timer_stop();

    CUDA_CHECK(cudaFree(deviceResults));
    CUDA_CHECK(cudaFree(deviceGenome));
    CUDA_CHECK(cudaFree(deviceGuides));
    CUDA_CHECK(cudaFree(deviceNumResults));
}

/***************************************************************
  MAIN FUNCTION
***************************************************************/

int main(int argc, char ** argv) {
    if (argc < 2) {
        PRINT("Please provide GPU method");
        exit(1);
    }

    int method = stoi(string(argv[1]));

    /*
     *Read the genome and guides into memory
     */
    int num_guides;
    int64_t genome_length;
    char * genome = read_genome(GENOME_FILE_PATH, &genome_length);
    char * guides = read_guides(GUIDES_FILE_PATH, &num_guides);
    float msec;

    PRINT("Genome length: {} nucleotides (bytes)", genome_length);
    PRINT("Number of guides: {}", num_guides);

    assert(NUM_GUIDES == num_guides);

    /*
     *Instantiate our results variables
     */
    results_t results_truth;
    int64_t sizeOfResults = num_guides * MATCHES_PER_GUIDE * sizeof(int64_t);
    int64_t * hostResults = (int64_t *) malloc(sizeOfResults);
    int hostNumResults;

    int64_t genome_length_test = 10000000;

    timer_start("Naive CPU");
    results_truth = naive_cpu_guide_matching(genome, genome_length_test, guides, num_guides);
    msec = timer_stop();
    estimate_total_time(msec, genome_length, genome_length_test);
    PRINT("Ground truth results size: {}", results_truth.size());

    int64_t step = 100000000;
    string divider = "============================";
    for (genome_length_test = step; genome_length_test <= genome_length; genome_length_test += step) {
        gpu_guide_matching(
                genome,
                genome_length_test,
                guides,
                num_guides,
                hostResults,
                &hostNumResults,
                sizeOfResults,
                static_cast<methods>(method));
        assert_results_equal(results_truth, hostResults, hostNumResults);
        PRINT("{}hostNumResults: {}{}", divider, hostNumResults, divider);
    }

    /*
     *Free up any dynamic memory
     */
    free(genome);
    free(guides);
    free(hostResults);
    return EXIT_SUCCESS;
}
