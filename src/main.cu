#include "common/fmt.hpp"
#include "common/utils.hpp"
#include <string>
#include <iostream>
#include <fstream>

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
#define GUIDE_SIZE_BYTES (GUIDE_SIZE_NUCLEOTIDES / NUCLEOTIDES_PER_BYTE)
/*
 *Add another byte to account for the null byte that is appended during getline
 */
#define GUIDE_BUFFER_SIZE GUIDE_SIZE_BYTES + 1

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
#define T_TO_BITS 0b00
#define C_TO_BITS 0b01
#define A_TO_BITS 0b10
#define G_TO_BITS 0b11

/*
 *Since we don't want to test naive on entire genome cause its takes too long,
 *we will limit our testing region to this constant
 */
#define GENOME_TEST_LENGTH 5000000

using namespace std;

typedef struct four_nt {
    /*
     * Since we can represent four nucleotides (nt) (A, C, T, or G) as 2 bit
     * values (0b00, 0b01, 0b10, 0b11), we can use one byte to represent 4 nt.
     * This struct encapsulates four nt in a single byte and has some helper
     * functions for comparisons. 
     */
    unsigned char one: 2;
    unsigned char two: 2;
    unsigned char three: 2;
    unsigned char four: 2;

    int hamming_distance(four_nt other) {
        return !(one == other.one) +
            !(two == other.two) +
            !(three == other.three) +
            !(four == other.four);
    }

    char bits_to_char(unsigned char bits) {
        switch(bits) {
            case T_TO_BITS:
                return 'T';
            case C_TO_BITS:
                return 'C';
            case A_TO_BITS:
                return 'A';
            case G_TO_BITS:
                return 'G';
        }
        PRINT("Invalid bits, bits must be in [0, 3]");
        return -1;
    }

    char to_char(int position) {
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
        PRINT("Invalid position, position must be in [0, 3]");
        return -1;
    }

} four_nt;

four_nt * read_genome(string filename, unsigned long long * genome_length) {
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
    *genome_length = size;

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

vector<four_nt *> read_guides(string filename) {

    vector<four_nt *> guides;
    ifstream file(filename, ios::binary);

    if (!file) {
        PRINT("Could not open file: {}", filename);
        PRINT("Error code: {}", strerror(errno));
        exit(EXIT_FAILURE);
    }

    char * line = new char[GUIDE_BUFFER_SIZE];
    file.getline(line, GUIDE_BUFFER_SIZE);
    for(;!file.eof(); file.getline(line, GUIDE_BUFFER_SIZE)) {
        guides.push_back((four_nt *) line);
        line = new char[GUIDE_BUFFER_SIZE];
    }
    return guides;
}

void print_sequence(four_nt * sequence) {
    for (int i = 0; i < GUIDE_SIZE_BYTES; i++) {
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

vector<tuple<int, unsigned long long> > naive_cpu_guide_matching(four_nt * genome, unsigned long long genome_length, vector<four_nt *> guides) {
    /*
     *Pseudocode for the naive algorithm:
     *
     *for genome_index in xrange(len(genome) - 19):
     *    for guide in guides:
     *        mismatches = 0
     *        for base_index in xrange(20):
     *             if genome[genome_index + base_index] != guide[base_index]:
     *                 mismatches += 1
     *        if mismatches <= 4:
     *            results.append((guide, genome_index))
     */
    vector<tuple<int, unsigned long long> > results;
    
    for (unsigned long long i = 0; i <= GENOME_TEST_LENGTH - GUIDE_SIZE_BYTES;
            i++) {
        for (int j = 0; j < guides.size(); j++) {
            four_nt * guide = guides[j];
            four_nt * genome_base = &genome[i];

            int mismatches = 0;
            for (int k = 0; k < GUIDE_SIZE_BYTES; k++)
                mismatches += genome_base[k].hamming_distance(guide[k]);

            if (mismatches <= EDIT_DISTANCE_THRESHOLD)
                results.push_back(make_tuple(j, i));
        }
    }
    return results;
}

int main(int argc, char ** argv) {
    /*
     *Read the genome and guides into memory
     */
    unsigned long long genome_length;
    four_nt * genome = read_genome(GENOME_FILE_PATH, &genome_length);
    vector<four_nt *> guides = read_guides(GUIDES_FILE_PATH);

    for (int i = 0; i < guides.size(); i++) {
        four_nt * guide = guides[i];
        PRINT("--------guides[{}]--------", i);
        for (int j = 0; j < 5; j++) {
            PRINT("{}{}{}{}", 
                    guide[j].to_char(0),
                    guide[j].to_char(1),
                    guide[j].to_char(2),
                    guide[j].to_char(3)
                    );
        }
    }

    PRINT("Genome length: {} bytes", genome_length);

    timer_start("Naive CPU");
    naive_cpu_guide_matching(genome, genome_length, guides);
    timer_stop();

    /*
     *Free up any dynamic memory
     */
    delete[] genome;
    for (int i = 0; i < guides.size(); i++)
        delete guides[i];
    guides.clear();
    return EXIT_SUCCESS;
}
