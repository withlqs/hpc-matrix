#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <utils.h>


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
    ull mat_size = int_root(file_size/sizeof(double));
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
