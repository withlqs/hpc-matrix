#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned long long ull;


ull root(ull num) {
    ull v= sqrt(num);
    if ((v-1)*(v-1) == num) {
        return v-1;
    }
    if (v*v == num) {
        return v;
    }
    if ((v+1)*(v+1) == num) {
        return v+1;
    }
    printf("error: can not sqrt size.\n");
    exit(-1);
    return 0;
}


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("number of arguments error.\n");
        exit(-1);
    }
    ull file_size;
    FILE *fp = fopen(argv[1], "rb");
    if (fp == NULL) {
        printf("error while opening file.\n");
        exit(-1);
    }
    fseek(fp, 0L, SEEK_END);
    file_size = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    double value;
    ull mat_size = root(file_size/sizeof(double));
    for (int i = 0; i < mat_size; ++i) {
        for (int j = 0; j < mat_size; ++j) {
            fread(&value, sizeof(double), 1, fp);
            printf("%f ", value);
        }
        printf("\n");
    }
    fclose(fp);
    printf("%llu\n", mat_size);
    return 0;
}