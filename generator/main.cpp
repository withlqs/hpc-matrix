#include <stdio.h>
#include <stdlib.h>
const int BUFFER_SIZE = 24*1024*1024;

double buffer[BUFFER_SIZE];

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("number of arguments error.\n");
        exit(-1);
    }

    long long size;
    sscanf(argv[1], "%lld", &size);

    size *= 24;
    size *= size;
    FILE *fp = fopen(argv[2], "wb");
    long long seed;
    sscanf(argv[3], "%lld", &seed);
    srand(seed);

    double value;
    
    for (long long i = 0; i < size/BUFFER_SIZE; ++i) {
        for (long long j = 0; j < BUFFER_SIZE; ++j) {
            buffer[j] = ((double)rand())/RAND_MAX+((double)rand());
            //printf("%f ", buffer[j]);
        }
        fwrite(&buffer, sizeof(buffer), 1, fp);
    }

    long long remain;
    if ((remain = size%BUFFER_SIZE) != 0) {
        for (long long j = 0; j < remain; ++j) {
            buffer[j] = ((double)rand())/RAND_MAX+((double)rand());
            //printf("%f ", buffer[j]);
        }
        fwrite(&buffer, sizeof(double)*remain, 1, fp);
    }

    fclose(fp);
    return 0;
}

        
