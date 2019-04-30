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
#define GENOME_FILE_PATH "/data/hg38.twobit"
#define GUIDES_FILE_PATH "./src/sequences.twobit"
/*
 *akin to string.format in Python
 */
#define PRINT(...) LOG(info, string(fmt::format(__VA_ARGS__)))
/*
 *20 nucleotides (typical guide length) / 4 nucleotides per byte = 5
 */
#define GUIDE_SIZE_NUCLEOTIDES 20
#define NUCLEOTIDES_PER_BYTE 4
#define GUIDE_SIZE (GUIDE_SIZE_NUCLEOTIDES / NUCLEOTIDES_PER_BYTE)
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
 * The mapping of bits to A, C, T, or G based on the to_2bit.py
 * script in the scripts folder.
 *
 * The mapping is:
 *     T: 0
 *     C: 1
 *     A: 2
 *     G: 3
 */
#define T_IN_BITS 0b00
#define C_IN_BITS 0b01
#define A_IN_BITS 0b10
#define G_IN_BITS 0b11
/*
 *Since we don't want to test naive on entire genome cause its takes too long,
 *we will limit our testing region to this constant
 */
#define GENOME_TEST_LENGTH_BYTES (1000000 * 1)
/*
 *This is used for predicting the maximum number of matches we might possibly
 *have
 */
#define MATCHES_PER_GUIDE 8000

#define TILE_WIDTH 1024

#define CUDA_CHECK(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line) {
    if (result != cudaSuccess) {
        PRINT("{}@{}: CUDA Runtime Error: {}", file, line, cudaGetErrorString(result));
        exit(-1);
    }
}

typedef set<tuple<int, uint64_t> > results_t;

typedef struct four_nt {
    /*
     * Since we can represent four nucleotides (nt) (A, C, T, or G) as 2 bit
     * values (0b00, 0b01, 0b10, 0b11), we can use one byte to represent 4 nt.
     * This struct encapsulates four nt in a single byte and has some helper
     * functions for printing.
     */
    unsigned char one: 2;
    unsigned char two: 2;
    unsigned char three: 2;
    unsigned char four: 2;

    char bits_to_char(unsigned char bits) {
        switch(bits) {
            case T_IN_BITS:
                return 'T';
            case C_IN_BITS:
                return 'C';
            case A_IN_BITS:
                return 'A';
            case G_IN_BITS:
                return 'G';
        }
        return -1;
    }

    char to_char(int position) {
        assert(position >= 0 && position <= 3);
        switch(position) {
            case 0:
                return bits_to_char(one);
            case 1:
                return bits_to_char(two);
            case 2:
                return bits_to_char(three);
            case 3:
                return bits_to_char(four);
        }
        return -1;
    }

} four_nt;


/***************************************************************
  CPU/GPU HELPER FUNCTIONS
***************************************************************/

__device__ __host__ char get_nucleotide_from_index(char * sequence, uint64_t index) {
    char byte = sequence[index / NUCLEOTIDES_PER_BYTE];
    byte >>= 2 * (NUCLEOTIDES_PER_BYTE - (index % NUCLEOTIDES_PER_BYTE) - 1);
    byte &= 0b11;
    return byte;
}

/***************************************************************
  NAIVE CPU IMPLEMENTATION
    
Pseudocode for the naive algorithm:

for genome_index in xrange(len(genome) - 19):
    for guide in guides:
        mismatches = 0
        for base_index in xrange(20):
             if genome[genome_index + base_index] != guide[base_index]:
                 mismatches += 1
        if mismatches <= 4:
            results.insert((guide, genome_index))
***************************************************************/

results_t naive_cpu_guide_matching(char * genome, uint64_t genome_length_bytes, char * guides, int num_guides) {
    results_t results;
    char genome_nucleotide;
    char guide_nucleotide;
    char * guide;
    int mismatches;
    uint64_t i;
    int j, k;

    for (i = 0; i <= (genome_length_bytes * NUCLEOTIDES_PER_BYTE) - GUIDE_SIZE_NUCLEOTIDES + 1;
            i++) {
        for (j = 0; j < num_guides; j++) {
            guide = guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE_NUCLEOTIDES; k++) {
                genome_nucleotide = get_nucleotide_from_index(genome, i + k);
                guide_nucleotide = get_nucleotide_from_index(guide, k);
                mismatches += genome_nucleotide != guide_nucleotide;
            }

            if (mismatches <= EDIT_DISTANCE_THRESHOLD)
                results.insert(make_tuple(j, i));
        }
    }
    return results;
}

/***************************************************************
  NAIVE GPU IMPLEMENTATION
***************************************************************/

__global__ void naive_gpu_guide_matching_kernel(
        char * genome,
        uint64_t genome_length_bytes,
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
    char genome_nucleotide;
    char guide_nucleotide;

    for (i = tid; i <= (genome_length_bytes * NUCLEOTIDES_PER_BYTE) - GUIDE_SIZE_NUCLEOTIDES + 1;
            i += gridDim.x * blockDim.x) {
        for (j = 0; j < num_guides; j++) {
            guide = guides + (j * GUIDE_BUFFER_SIZE);

            mismatches = 0;
            for (k = 0; k < GUIDE_SIZE_NUCLEOTIDES; k++) {
                genome_nucleotide = get_nucleotide_from_index(genome, i + k);
                guide_nucleotide = get_nucleotide_from_index(guide, k);
                mismatches += genome_nucleotide != guide_nucleotide;
            }

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
  GPU KERNEL LAUNCHER
***************************************************************/

enum methods {naive, thread_coarsening};
void gpu_guide_matching(
        char * genome,
        uint64_t genome_length_bytes,
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
    CUDA_CHECK(cudaMalloc((void **) &deviceGenome, genome_length_bytes));
    CUDA_CHECK(cudaMalloc((void **) &deviceGuides, num_guides * GUIDE_BUFFER_SIZE));
    CUDA_CHECK(cudaMalloc((void **) &deviceNumResults, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(deviceGenome, genome, genome_length_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceGuides, guides, num_guides * GUIDE_BUFFER_SIZE, cudaMemcpyHostToDevice));

    switch (method) {
        case naive: 
            dim3 dimGrid(ceil((genome_length_bytes * NUCLEOTIDES_PER_BYTE) / double(TILE_WIDTH)));
            dim3 dimBlock(TILE_WIDTH);
            naive_gpu_guide_matching_kernel<<<dimGrid, dimBlock>>>(
                    deviceGenome,
                    genome_length_bytes,
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
  GPU-AWARE IMPLEMENTATION
  
  Possible optimizations:
  - Warp-queue like structure for the results to avoid atomic operation
  - Striding for data coalescing (each thread processes a portion of the genome)
  - Shared memory for the guides 
  - Shared memory for the genome
  - Thread coarsening to take advantage of registers
  - Pre-process the guides into hashes, use rolling hash on genome.
***************************************************************/

/***************************************************************
  I/O HELPER FUNCTIONS
***************************************************************/
four_nt * read_genome(string filename, uint64_t * genome_length_bytes) {
    /*
     * Used for reading our file into a byte buffer casted as our four_nt struct.
     */

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
    *genome_length_bytes = size;

    /*
     * I would love to use vector<four_nt> here but if we want to use the read
     * function, we must pass in a buffer of type char *. I thought about
     * trying to modify the .data() pointer of a vector but that is a
     * read-only pointer and so I am left to casting a char* as four_nt *
     */
    char * buffer = new char [size];
    if (file.read(buffer, size)) {
        return (four_nt *) buffer;
    }

    PRINT("Unable to read {}", filename);
    exit(EXIT_FAILURE);
}

four_nt * read_guides(string filename, int * num_guides) {

    // Assert that these are equal just in case four_nt struct gets changed
    assert(sizeof(four_nt) == sizeof(char));
    
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

    PRINT("Number of guides to process: {}", *num_guides);

    guides = (char *) malloc(*num_guides * GUIDE_BUFFER_SIZE);

    for(int line_number = 0; line_number < *num_guides; line_number++) {
        file.getline(&guides[line_number * GUIDE_BUFFER_SIZE],
                GUIDE_BUFFER_SIZE);
    }

    PRINT("Successfully read all guides");
    return (four_nt *) guides;
}

void print_sequence(four_nt * sequence) {
    for (int i = 0; i < GUIDE_SIZE; i++) {
        four_nt sequence_byte = sequence[i];
        char one = sequence_byte.to_char(0);
        char two = sequence_byte.to_char(1);
        char three = sequence_byte.to_char(2);
        char four = sequence_byte.to_char(3);
        /*
         *We are printing in reverse order because our machine is little-endian
         */
        printf("%c%c%c%c", four, three ,two, one);
    }
    printf("\n");
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
        if (!cpuResults.count(match) && !seen.count(match)) {
            PRINT("{}", failMessage);
            return;
        } else {
            seen.insert(match);
        }
    }
    PRINT("{}", successMessage);
}

int main(int argc, char ** argv) {
    /*
     *Read the genome and guides into memory
     */
    int num_guides;
    uint64_t genome_length_bytes;
    four_nt * genome = read_genome(GENOME_FILE_PATH, &genome_length_bytes);
    four_nt * guides = read_guides(GUIDES_FILE_PATH, &num_guides);

    for (int i = 0; i < num_guides; i++) {
        four_nt * guide = &guides[i * GUIDE_BUFFER_SIZE];
        PRINT("--------guides[{}]--------", i);
        print_sequence(guide);
    }

    PRINT("Genome length: {} bytes", genome_length_bytes);

    /*
     *Instantiate our results variables
     */
    results_t results_truth;
    uint64_t sizeOfResults = num_guides * MATCHES_PER_GUIDE * sizeof(uint64_t);
    uint64_t * hostResults = (uint64_t *) malloc(sizeOfResults);
    int hostNumResults;

    timer_start("Naive CPU");
    results_truth = naive_cpu_guide_matching((char *) genome, GENOME_TEST_LENGTH_BYTES, (char *) guides, num_guides);
    timer_stop();

    PRINT("Ground truth results size: {}", results_truth.size());

    timer_start("Naive GPU");
    methods method = naive;
    gpu_guide_matching(
            (char *) genome,
            GENOME_TEST_LENGTH_BYTES,
            (char *) guides,
            num_guides,
            hostResults,
            &hostNumResults,
            sizeOfResults,
            method);
    timer_stop();

    PRINT("Naive GPU results size: {}", hostNumResults);
    assert_results_equal(results_truth, hostResults, hostNumResults);


    /*
     *Free up any dynamic memory
     */
    delete[] genome;
    free(guides);
    free(hostResults);
    return EXIT_SUCCESS;
}
