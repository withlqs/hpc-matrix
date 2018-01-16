#include <cstdio>
#include <common.h>
#include <mkl.h>
#include <cmath>

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("error: number of arguments is not 4");
        exit(-1);
    }

    double* a;
    double* b;
    double* c;
    ull mat_size;

    show_time(mat_size = load_mat(argv[1], a);load_mat(argv[2], b), [info] load mat file);

    c = (double*)aligned_alloc(64, sizeof(double)*mat_size*mat_size);
    for (int i = 0; i < mat_size*mat_size; ++i) {
        c[i] = 0;
    }

    show_time(cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mat_size, mat_size, mat_size, 1.0, a, mat_size, b, mat_size, 0.0, c, mat_size), [info] mkl);
    double* d;
    //save_mat("d.mat", c, mat_size);
    show_time(load_mat(argv[3], d), [info] load mat file);
    ull incorrect = 0;
    for (int i = 0; i < mat_size*mat_size; i++) {
        //printf("%f, %f\n", c[i], d[i]);
        if (fabs(c[i]-d[i]) > 1e-8*fabs(c[i])) {
            ++incorrect;
        }
    }
    printf("[info] %llu/%llu\n", mat_size*mat_size-incorrect, mat_size*mat_size);
}
