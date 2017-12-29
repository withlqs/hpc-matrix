#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <utils.h>

ull int_root(ull num) {
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

ull load_mat(const char* file, double* &data) {
    FILE *fp = fopen(file, "rb");
    fseek(fp, 0L, SEEK_END);
    ull file_size = ftell(fp);
    rewind(fp);
    ull mat_size = int_root(file_size/sizeof(double));
    data = (double *)aligned_alloc(64, sizeof(double)*mat_size*mat_size);
    fread(data, sizeof(double)*mat_size*mat_size, 1, fp);
    printf("[info] load %s as a %llux%llu matrix finished\n", file, mat_size, mat_size);
    fclose(fp);
    return mat_size;
}
