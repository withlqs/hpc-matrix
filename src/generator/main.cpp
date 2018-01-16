#include <cstdio>
#include <cstdlib>
#include <common.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("[error] number of arguments error.\n");
        exit(-1);
    }

    ull size;
    sscanf(argv[1], "%llu", &size);
    size *= size;

    FILE *fp = fopen(argv[2], "wb");
    long long seed;
    sscanf(argv[3], "%lld", &seed);
    srand(seed);

    double *buffer = (double *) aligned_alloc(64, size*sizeof(double));
    
    for (long long i = 0; i < size; ++i) {
        buffer[i] = ((double)rand())/RAND_MAX+((double)rand());
    }

    fwrite(buffer, sizeof(double)*size, 1, fp);
    fclose(fp);
    return 0;
}

        
