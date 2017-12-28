#include <stdio.h>
#include <stdlib.h>
#include <utils/utils.h>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

using std::swap;

typedef unsigned long long ull;


ull load_mat(const char* file, double* &data) {
    FILE *fp = fopen(file, "rb");
    fseek(fp, 0L, SEEK_END);
    ull file_size = ftell(fp);
    rewind(fp);
    ull mat_size = int_root(file_size/sizeof(double));
    //printf("info: loading %s as a matrix, size: %llux%llu\n", file, mat_size, mat_size);
    data = (double *)aligned_alloc(32, sizeof(double)*mat_size*mat_size);
    fread(data, sizeof(double)*mat_size*mat_size, 1, fp);
    printf("info: load %s as a matrix finished\n", file);
    fclose(fp);
    return mat_size;
}

void transpose(double* data, ull mat_size) {
    int i, j;
    #pragma omp parallel for private(i, j) shared(mat_size, data)
    for (i = 0; i < mat_size; ++i) {
        #pragma omp parallel for private(j) shared(i, mat_size, data)
        for (j = i+1; j < mat_size; ++j) {
            swap(data[i*mat_size+j], data[j*mat_size+i]);
        }
    }
}

double* multiply(double *a, double *b, ull mat_size) {
    double *c;
    c = (double *)aligned_alloc(32, sizeof(double)*mat_size*mat_size);

    __m256d av, bv, cv, dv;
    double *d;
    int i, j, k;
    #pragma omp parallel for private(i, j, k, av, bv, cv, dv, d) shared(mat_size, a, b, c)
    for (i = 0; i < mat_size; ++i) {
        //#pragma omp parallel for private(j, k, av, bv, cv, dv, d) shared(i, mat_size, a, b, c)
        for (j = 0; j < mat_size; ++j) {
            //#pragma omp parallel for private(k, av, bv, cv, dv, d) shared(i, j, mat_size, a, b, c)
            for (k = 0; k < mat_size; k += 4) {
                av = _mm256_load_pd(a+i*mat_size+k);
                bv = _mm256_load_pd(b+j*mat_size+k);
                cv = _mm256_mul_pd(av, bv);
                dv = _mm256_hadd_pd(cv, cv);
                d = (double *) &dv;
                c[i*mat_size+j] = d[0]+d[2];
            }

        }
    }
    return c;
}

void save_mat(double *data, const char* file, ull mat_size) {
    FILE *fp = fopen(file, "wb");
    fwrite(data, sizeof(double)*mat_size*mat_size, 1, fp);
    fclose(fp);
}


int main(int argc, char **argv) {
    if (argc != 4) {
        printf("error: numebr of arguments is %d\n", argc);
        exit(-1);
    }
    double *data_a;
    double *data_b;
    double *data_c;
    show_time(load_mat(argv[1], data_a), "load a");
    ull mat_size;
    show_time(mat_size = load_mat(argv[2], data_b), "load b");
    show_time(transpose(data_b, mat_size), "transpose");
    show_time(data_c = multiply(data_a, data_b, mat_size), "multiply");
    show_time(save_mat(data_c, argv[3], mat_size), "save mat");
}
