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
#define COARSE_FAC 40
#define COARSENED_TILE_WIDTH (TILE_WIDTH * COARSE_FAC)
#define CONSTANT_MEMORY_SIZE (NUM_GUIDES * GUIDE_BUFFER_SIZE)

#define CUDA_CHECK(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line) {
    if (result != cudaSuccess) {
        PRINT("{}@{}: CUDA Runtime Error: {}", file, line, cudaGetErrorString(result));
        exit(-1);
    }
}

typedef set<tuple<int, uint64_t> > results_t;

/***************************************************************
  I/O HELPER FUNCTIONS
***************************************************************/
char * read_genome(string filename, uint64_t * genome_length) {
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

    char * buffer = new char [size];
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

void assert_results_equal(results_t cpuResults, uint64_t * gpuResults, int gpuNumResults) {
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
        uint64_t genomeIdx = gpuResults[gpuResultsIdx + 1];
        tuple<int, uint64_t> match = make_tuple(guideIdx, genomeIdx);
        if (!cpuResults.count(match) || seen.count(match)) {
            PRINT("{}", failMessage);
            return;
        } else {
            seen.insert(match);
        }
    }
    PRINT("{}", successMessage);
}

void estimate_total_time(float msec, uint64_t genome_length, uint64_t genome_length_test) {
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

results_t naive_cpu_guide_matching(char * genome, uint64_t genome_length, char * guides, int num_guides) {
    results_t results;
    char * guide;
    int mismatches;
    uint64_t i;
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
  NAIVE GPU IMPLEMENTATION
***************************************************************/

__global__ void naive_gpu_kernel(
        char * genome,
        uint64_t genome_length,
        char * guides,
        int num_guides,
        uint64_t * results,
        int * numResults,
        uint64_t sizeOfResults)
{
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    int numResultsLocal;
    int resultsIdx;
    char * guide;
    uint64_t i;
    int mismatches;
    int j, k;

    for (i = tid; i < genome_length - GUIDE_SIZE - 1; i += gridDim.x * blockDim.x) {
        for (j = 0; j < num_guides; j++) {
            guide = guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE; k++)
                mismatches += genome[i + k] != guide[k];

            if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                numResultsLocal = atomicAdd(numResults, 1);
                if (numResultsLocal < sizeOfResults) {
                    resultsIdx = numResultsLocal * 2;
                    results[resultsIdx] = (uint64_t) j;
                    results[resultsIdx + 1] = i;
                }
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
  SHARED MEMORY GPU IMPLEMENTATION
***************************************************************/

__global__ void shared_memory_gpu_kernel(
        char * genome,
        uint64_t genome_length,
        char * guides,
        int num_guides,
        uint64_t * results,
        int * numResults,
        uint64_t sizeOfResults)
{
    uint64_t tx = threadIdx.x;
    uint64_t tid = blockDim.x * blockIdx.x + tx;
    int numResultsLocal;
    int resultsIdx;
    char * guide;
    uint64_t i, b_start;
    int mismatches;
    int j, k, t_start;

    __shared__ char shared_genome[TILE_WIDTH * 2];

    for (i = tid; i < genome_length - GUIDE_SIZE - 1; i += gridDim.x * blockDim.x) {

        b_start = (i / blockDim.x) * blockDim.x;
        t_start = 2 * tx;

        if (b_start + t_start < genome_length)
            shared_genome[t_start] = genome[b_start + t_start];
        if (b_start + t_start + 1 < genome_length)
            shared_genome[t_start + 1] = genome[b_start + t_start + 1];
        __syncthreads();

        for (j = 0; j < num_guides; j++) {
            guide = guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE; k++)
                mismatches += shared_genome[tx + k] != guide[k];

            if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                numResultsLocal = atomicAdd(numResults, 1);
                if (numResultsLocal < sizeOfResults) {
                    resultsIdx = numResultsLocal * 2;
                    results[resultsIdx] = (uint64_t) j;
                    results[resultsIdx + 1] = i;
                }
            }
        }

        __syncthreads();
    }
}

/***************************************************************
  SHARED MEMORY & CONSTANT MEMORY GPU IMPLEMENTATION
***************************************************************/

__constant__ char constant_guides[CONSTANT_MEMORY_SIZE];
__global__ void shared_constant_memory_gpu_kernel(
        char * genome,
        uint64_t genome_length,
        char * guides,
        int num_guides,
        uint64_t * results,
        int * numResults,
        uint64_t sizeOfResults)
{
    uint64_t tx = threadIdx.x;
    uint64_t tid = blockDim.x * blockIdx.x + tx;
    int numResultsLocal;
    int resultsIdx;
    char * guide;
    uint64_t i, b_start;
    int mismatches;
    int j, k, t_start;

    __shared__ char shared_genome[TILE_WIDTH * 2];

    for (i = tid; i < genome_length - GUIDE_SIZE - 1; i += gridDim.x * blockDim.x) {

        b_start = (i / blockDim.x) * blockDim.x;
        t_start = 2 * tx;

        if (b_start + t_start < genome_length)
            shared_genome[t_start] = genome[b_start + t_start];
        if (b_start + t_start + 1 < genome_length)
            shared_genome[t_start + 1] = genome[b_start + t_start + 1];
        __syncthreads();

        for (j = 0; j < num_guides; j++) {
            guide = constant_guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE; k++)
                mismatches += shared_genome[tx + k] != guide[k];

            if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                numResultsLocal = atomicAdd(numResults, 1);
                if (numResultsLocal < sizeOfResults) {
                    resultsIdx = numResultsLocal * 2;
                    results[resultsIdx] = (uint64_t) j;
                    results[resultsIdx + 1] = i;
                }
            }
        }

        __syncthreads();
    }
}

/***************************************************************
  COARSENED SHARED MEMORY & CONSTANT MEMORY GPU IMPLEMENTATION
***************************************************************/

__global__ void coarsened_shared_constant_memory_gpu_kernel(
        char * genome,
        uint64_t genome_length,
        char * guides,
        int num_guides,
        uint64_t * results,
        int * numResults,
        uint64_t sizeOfResults)
{
    uint64_t tx = threadIdx.x;
    uint64_t tid = blockDim.x * blockIdx.x + tx;
    int numResultsLocal;
    int resultsIdx;
    char * guide;
    uint64_t i, b_start;
    int mismatches;
    int j, k, n, t_start;
    char rc[GUIDE_SIZE + COARSE_FAC];

    __shared__ char shared_genome[TILE_WIDTH * (COARSE_FAC + 1)];

    for (i = tid * COARSE_FAC; i < genome_length - GUIDE_SIZE - 1; i += COARSE_FAC * gridDim.x * blockDim.x) {

        b_start = (i / (blockDim.x * COARSE_FAC)) * blockDim.x * COARSE_FAC;
        t_start = (COARSE_FAC + 1) * tx;

        for (n = 0; n < COARSE_FAC + 1; n++) {
            if (b_start + t_start + n < genome_length)
                shared_genome[t_start + n] = genome[b_start + t_start + n];
        }
        __syncthreads();

        for (n = 0; n < GUIDE_SIZE + COARSE_FAC; n++)
            rc[n] = shared_genome[COARSE_FAC * tx + n];

        for (n = 0; n < COARSE_FAC; n++) {
            for (j = 0; j < num_guides; j++) {
                guide = constant_guides + (j * GUIDE_BUFFER_SIZE);
                mismatches = 0;
                for (k = 0; k < GUIDE_SIZE; k++)
                    mismatches += rc[n + k] != guide[k];

                if (mismatches <= EDIT_DISTANCE_THRESHOLD) {
                    numResultsLocal = atomicAdd(numResults, 1);
                    if (numResultsLocal < sizeOfResults) {
                        resultsIdx = numResultsLocal * 2;
                        results[resultsIdx] = (uint64_t) j;
                        results[resultsIdx + 1] = i + n;
                    }
                }
            }
        }

        __syncthreads();
    }
}

/***************************************************************
  GPU KERNEL LAUNCHER
***************************************************************/

enum methods {naive, s_memory, s_c_memory, coarsened_s_c_memory};
void gpu_guide_matching(
        char * genome,
        uint64_t genome_length,
        char * guides,
        int num_guides,
        uint64_t * hostResults,
        int * hostNumResults,
        uint64_t sizeOfResults,
        methods method)
{
    uint64_t * deviceResults;
    char * deviceGenome;
    char * deviceGuides;
    int * deviceNumResults;

    CUDA_CHECK(cudaMalloc((void **) &deviceResults, sizeOfResults));
    CUDA_CHECK(cudaMalloc((void **) &deviceGenome, genome_length));
    CUDA_CHECK(cudaMalloc((void **) &deviceGuides, num_guides * GUIDE_BUFFER_SIZE));
    CUDA_CHECK(cudaMalloc((void **) &deviceNumResults, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(deviceGenome, genome, genome_length, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceGuides, guides, num_guides * GUIDE_BUFFER_SIZE, cudaMemcpyHostToDevice));

    dim3 dimGridNaive(ceil((genome_length) / double(TILE_WIDTH)));
    dim3 dimBlockNaive(TILE_WIDTH);
    dim3 dimGridCoarsened(ceil(genome_length / double(TILE_WIDTH * COARSE_FAC)));
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
        case s_c_memory:
            CUDA_CHECK(cudaMemcpyToSymbol(constant_guides, deviceGuides, CONSTANT_MEMORY_SIZE));
            shared_constant_memory_gpu_kernel<<<dimGridNaive, dimBlockNaive>>>(
                    deviceGenome,
                    genome_length,
                    deviceGuides,
                    num_guides,
                    deviceResults,
                    deviceNumResults,
                    sizeOfResults
                    );
            break;
        case coarsened_s_c_memory:
            CUDA_CHECK(cudaMemcpyToSymbol(constant_guides, deviceGuides, CONSTANT_MEMORY_SIZE));
            coarsened_shared_constant_memory_gpu_kernel<<<dimGridCoarsened, dimBlockNaive>>>(
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

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hostNumResults, deviceNumResults, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostResults, deviceResults, sizeOfResults, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(deviceResults));
    CUDA_CHECK(cudaFree(deviceGenome));
    CUDA_CHECK(cudaFree(deviceGuides));
}

/***************************************************************
  MAIN FUNCTION
***************************************************************/

int main(int argc, char ** argv) {
    /*
     *Read the genome and guides into memory
     */
    int num_guides;
    uint64_t genome_length;
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
    uint64_t sizeOfResults = num_guides * MATCHES_PER_GUIDE * sizeof(uint64_t);
    uint64_t * hostResults = (uint64_t *) malloc(sizeOfResults);
    int hostNumResults;

    // For testing smaller lengths
    uint64_t genome_length_test = genome_length;

    /*
     *timer_start("Naive CPU");
     *results_truth = naive_cpu_guide_matching(genome, genome_length_test, guides, num_guides);
     *msec = timer_stop();
     *estimate_total_time(msec, genome_length, genome_length_test);
     *PRINT("Ground truth results size: {}", results_truth.size());
     */

    methods method;

    timer_start("Naive GPU");
    method = naive;
    gpu_guide_matching(
            genome,
            genome_length_test,
            guides,
            num_guides,
            hostResults,
            &hostNumResults,
            sizeOfResults,
            method);
    msec = timer_stop();
    estimate_total_time(msec, genome_length, genome_length_test);
    PRINT("Naive GPU results size: {}", hostNumResults);
    assert_results_equal(results_truth, hostResults, hostNumResults);

    timer_start("shared memory gpu");
    method = s_memory;
    gpu_guide_matching(
            genome,
            genome_length_test,
            guides,
            num_guides,
            hostResults,
            &hostNumResults,
            sizeOfResults,
            method);
    msec = timer_stop();
    estimate_total_time(msec, genome_length, genome_length_test);
    PRINT("Shared Memory GPU results size: {}", hostNumResults);
    assert_results_equal(results_truth, hostResults, hostNumResults);

    timer_start("Shared + Constant Memory GPU");
    method = s_c_memory;
    gpu_guide_matching(
            genome,
            genome_length_test,
            guides,
            num_guides,
            hostResults,
            &hostNumResults,
            sizeOfResults,
            s_c_memory);
    msec = timer_stop();
    estimate_total_time(msec, genome_length, genome_length_test);
    PRINT("Shared + Constant Memory GPU results size: {}", hostNumResults);
    assert_results_equal(results_truth, hostResults, hostNumResults);

    timer_start("Coarsened Shared + Constant Memory GPU");
    method = s_c_memory;
    gpu_guide_matching(
            genome,
            genome_length_test,
            guides,
            num_guides,
            hostResults,
            &hostNumResults,
            sizeOfResults,
            coarsened_s_c_memory);
    msec = timer_stop();
    estimate_total_time(msec, genome_length, genome_length_test);
    PRINT("Coarsened Shared + Constant Memory GPU results size: {}", hostNumResults);
    assert_results_equal(results_truth, hostResults, hostNumResults);

    /*
     *Free up any dynamic memory
     */
    delete[] genome;
    free(guides);
    free(hostResults);
    return EXIT_SUCCESS;
}
