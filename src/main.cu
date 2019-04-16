#include "common/fmt.hpp"
#include "common/utils.hpp"
#include <string>
#include <iostream>
#include <fstream>

// this file should exist from the docker image
#define GENOME_FILE_PATH "/data/hg38.twobit"
#define GUIDES_FILE_PATH "./src/sequences.twobit"
// akin to string.format in Python
#define PRINT(...) LOG(info, string(fmt::format(__VA_ARGS__)))

using namespace std;

typedef struct four_bases {
    /*
     *Since we can represent four bases (A, C, T, or G) as 2 bit values
     *(0b00, 0b01, 0b10, 0b11), we can use one byte to represent 4
     *bases. This struct encapsulates four bases in a single byte
     *and has some helper functions for comparisons.
     */
    unsigned char one: 2;
    unsigned char two: 2;
    unsigned char three: 2;
    unsigned char four: 2;


    /*
     *four_bases (char byte) {
     *    one = (byte >> 6); 
     *    two = (byte >> 4) & 0x3; 
     *    three = (byte >> 2) & 0x3; 
     *    four = byte & 0x3; 
     *}
     */

    int hamming_distance(four_bases other) {
        return !(one == other.one) +
            !(two == other.two) +
            !(three == other.three) +
            !(four == other.four);
    }

} four_bases;

four_bases * read_genome(string filename) {
    /*
     *Used for reading our file into a byte buffer. Functions returns a vector<char>
     *type so we can just the nice properties of our vector library but still
     *being able to access the data through the vector.data() function
     */

    /*
     *ios::binary means to read file as is
     *ios::ate means start file pointer at the end so we can get file size
     */
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) {
        PRINT("Could not open file: {}", filename);
        PRINT("Error code: {}", strerror(errno));
    }
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    char * buffer = new char [size];
    if (file.read(buffer, size))
        return (four_bases *) buffer;
    PRINT("Unable to read {}", filename);
    exit(EXIT_FAILURE);
}

vector<char *> read_guides(string filename) {
    vector<vector<four_bases>> guides;
    ifstream file(filename, ios::binary);

    while(getline(file, line)) {

    }

    return guides;
} 

int main(int argc, char ** argv) {
    // Read the data
    four_bases * genome = read_genome(GENOME_FILE_PATH);
    vector<vector<four_bases>> guides = read_guides(GUIDES_FILE_PATH);

    four_bases n_ = genome[0];
    for (int i = 0; i < 20; i++) {
        four_bases n = genome[i];
        PRINT("difference with last four nucleotide: {}", n.hamming_distance(n_));
        PRINT("genome[{}].one: {}", i, n.one);
        PRINT("genome[{}].two: {}", i, n.two);
        PRINT("genome[{}].three: {}", i, n.three);
        PRINT("genome[{}].four: {}", i, n.four);
        n_ = n;
    }

    delete[] genome;
    return EXIT_SUCCESS;
}
