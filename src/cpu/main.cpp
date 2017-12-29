#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

using std::swap;

typedef unsigned long long ull;

void transpose(double* data, ull mat_size) {
    int i, j;
    #pragma omp parallel for private(i, j, mat_size, data)
    for (i = 0; i < mat_size; ++i) {
        //#pragma omp parallel for private(j, i, mat_size, data)
        for (j = i+1; j < mat_size; ++j) {
            swap(data[i*mat_size+j], data[j*mat_size+i]);
        }
    }
}

double* multiply(double *a, double *b, ull mat_size) {
    double *c = (double *)aligned_alloc(64, sizeof(double)*mat_size*mat_size);

    for (int i = 0; i < mat_size; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < mat_size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < mat_size; k += 4) {
                __m256d av, bv, cv, dv;
                double *d;
                av = _mm256_load_pd(a+i*mat_size+k);
                bv = _mm256_load_pd(b+j*mat_size+k);
                cv = _mm256_mul_pd(av, bv);
                dv = _mm256_hadd_pd(cv, cv);
                d = (double *) &dv;
                sum += d[0]+d[2];
            }
            c[i*mat_size+j] = sum;
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
        printf("[error] numebr of arguments is %d\n", argc);
        exit(-1);
    }
    double *data_a;
    double *data_b;
    double *data_c;
    show_time(load_mat(argv[1], data_a), [info] load A mat);
    ull mat_size;
    show_time(mat_size = load_mat(argv[2], data_b), [info] load B mat);
    show_time(transpose(data_b, mat_size), [info] transpose B mat);
    show_time(data_c = multiply(data_a, data_b, mat_size), [info] mutiply);
    show_time(save_mat(data_c, argv[3], mat_size), [info] save C mat);
}
