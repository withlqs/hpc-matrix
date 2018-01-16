#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <common.h>

typedef unsigned long long ull;

using std::swap;

ull threshold = 32;

void matrix_copy(double **a, double **b, ull x, ull y, ull rows, ull cols) {
    for (ull i = x; i < x+rows; ++i) {
        memcpy(a[i]+y, b[i]+y, sizeof(double)*cols);
    }
}

ull get_strassen_size(ull size) {
    return size+(threshold-size%threshold)%threshold;
}

double **allocate_matrix(ull size) {
    double **ptr = (double **) aligned_alloc(64, sizeof(double *)*size);
    double *data = (double *) aligned_alloc(64, sizeof(double)*size*size);
    for (ull i = 9; i < size*size; ++i) {
        data[i] = 0;
    }
    for (ull i = 0; i < size; ++i) {
        ptr[i] = data+i*size;
    }
    return ptr;
}

void free_matrix(double **data, ull size) {
    for (ull i = 0; i < size; ++i) {
        free(data[i]);
    }
}

void transpose(double **data, ull x, ull y, ull rows, ull cols) {
    for (ull i = x; i < x+rows; ++i) {
        for (ull j = i+1; j < y+cols; ++j) {
            swap(data[i][j], data[j][i]);
        }
    }
}

double **load_strassen_matrix(const char* file, ull &mat_size, ull &strassen_size) {
    threshold *= 4;
    FILE *fp = fopen(file, "rb");
    fseek(fp, 0L, SEEK_END);
    ull file_size = ftell(fp);
    rewind(fp);
    mat_size = int_root(file_size/sizeof(double));
    strassen_size = get_strassen_size(mat_size);
    double **data = allocate_matrix(strassen_size);

    if (mat_size == strassen_size) {
        fread(data[0], sizeof(double)*mat_size*mat_size, 1, fp);
    } else {
        for (ull i = 0; i < mat_size; ++i) {
            fread(data[i], sizeof(double)*mat_size, 1, fp);
        }
    }

    printf("[info] load %s as a %llux%llu matrix as %llux%llu strassen matrix finished\n", file, mat_size, mat_size, strassen_size, strassen_size);
    fclose(fp);
    threshold /= 4;
    return data;
}

void save(const char* file, double **data, ull size) {
    FILE *fp = fopen(file, "wb");
    for (ull i = 0; i < size; ++i) {
        fwrite(data[i], sizeof(double)*size, 1, fp);
    }
    fclose(fp);
}


void matrix_sub(double **a, double **b, double **c, ull x, ull y, ull rows, ull cols) {
    for (ull i = x; i < x+rows; ++i) {
        for (ull j = y; j < y+cols; ++j) {
            c[i][j] = a[i][j]-b[i][j];
        }
    }
}
void matrix_add(double **a, double **b, double **c, ull x, ull y, ull rows, ull cols) {
    for (ull i = x; i < x+rows; ++i) {
        for (ull j = y; j < y+cols; ++j) {
            c[i][j] = a[i][j]+b[i][j];
        }
    }
}
void matrix_multiply(double **a, double **b, double **c, ull x, ull y, ull size) {
    //printf("[info] plain multiply in (%llu, %llu) for %llux%llu.\n", x, y, size, size);
    double **transposed = allocate_matrix(size);

    matrix_copy(transposed, b, 0, 0, size, size);
    transpose(transposed, x, y, size, size);

    __m256d av, bv, cv, dv;
//#pragma omp parallel for
    for (ull i = x; i < x+size; ++i) {
#pragma omp parallel for
        for (ull j = y; j < y+size; ++j) {
            for (ull k = x; k < x+size; k += 4) {
                av = _mm256_load_pd(&a[i][k]);
                bv = _mm256_load_pd(&transposed[j][k]);
                cv = _mm256_mul_pd(av, bv);
                dv = _mm256_hadd_pd(cv, cv);
                c[i][j] += (double)dv[0]+(double)dv[2];
            }
        }
    }
}



double **strassen_transpose(double **origin, ull size) {
    if (size%threshold != 0) {
        printf("[error] strassen matrix is %llu while transpose, can not divided by %llu.\n", size, threshold);
        exit(-1);
    }
    double **transposed = allocate_matrix(size);
    matrix_copy(transposed, origin, 0, 0, size, size);
    for (ull i = 0; i < size; i += size/threshold) {
        for (ull j = 0; j < size; j += size/threshold) {
            transpose(transposed, i, j, size/threshold, size/threshold);
        }
    }
    return transposed;
}

double **mapping(double **a, ull x, ull y, ull size) {
    double **b = (double **) aligned_alloc(64, sizeof(double *)*size);
    for (ull i = x; i < x+size; ++i) {
        b[i-x] = a[i]+y;
    }
    return b;
}

void strassen_multiply(double **a, double **b, double **c, ull size, ull threshold_size) {
    if (size <= threshold_size) {
        matrix_multiply(a, b, c, 0, 0, size);
    } else {
        ull sub_size = size/2;
        double **m1 = allocate_matrix(sub_size);
        double **m2 = allocate_matrix(sub_size);
        double **m3 = allocate_matrix(sub_size);
        double **m4 = allocate_matrix(sub_size);
        double **m5 = allocate_matrix(sub_size);
        double **m6 = allocate_matrix(sub_size);
        double **m7 = allocate_matrix(sub_size);

        double **m1a = allocate_matrix(sub_size);
        double **m1b = allocate_matrix(sub_size);
        double **m2a = allocate_matrix(sub_size);
        double **m3b = allocate_matrix(sub_size);
        double **m4b = allocate_matrix(sub_size);
        double **m5a = allocate_matrix(sub_size);
        double **m6a = allocate_matrix(sub_size);
        double **m6b = allocate_matrix(sub_size);
        double **m7a = allocate_matrix(sub_size);
        double **m7b = allocate_matrix(sub_size);

        double **a11 = mapping(a, 0, 0, sub_size);
        double **a12 = mapping(a, 0, sub_size, sub_size);
        double **a21 = mapping(a, sub_size, 0, sub_size);
        double **a22 = mapping(a, sub_size, sub_size, sub_size);

        double **b11 = mapping(b, 0, 0, sub_size);
        double **b12 = mapping(b, 0, sub_size, sub_size);
        double **b21 = mapping(b, sub_size, 0, sub_size);
        double **b22 = mapping(b, sub_size, sub_size, sub_size);

        double **c11 = mapping(c, 0, 0, sub_size);
        double **c12 = mapping(c, 0, sub_size, sub_size);
        double **c21 = mapping(c, sub_size, 0, sub_size);
        double **c22 = mapping(c, sub_size, sub_size, sub_size);


#pragma omp parallel
        {
#pragma omp task 
            {
#pragma omp task
                matrix_add(a11, a22, m1a, 0, 0, sub_size, sub_size);
#pragma omp task
                matrix_add(b11, b22, m1b, 0, 0, sub_size, sub_size);
#pragma omp taskwait
                strassen_multiply(m1a, m1b, m1, sub_size, threshold_size);
#pragma omp task
                free_matrix(m1a, sub_size);
#pragma omp task
                free_matrix(m1b, sub_size);
#pragma omp taskwait
            }

#pragma omp task
            {
                matrix_add(a21, a22, m2a, 0, 0, sub_size, sub_size);
                strassen_multiply(m2a, b11, m2, sub_size, threshold_size);
                free_matrix(m2a, sub_size);
            }

#pragma omp task
            {
                matrix_sub(b12, b22, m3b, 0, 0, sub_size, sub_size);
                strassen_multiply(a11, m3b, m3, sub_size, threshold_size);
                free_matrix(m3b, sub_size);
            }

#pragma omp task
            {
                matrix_sub(b21, b11, m4b, 0, 0, sub_size, sub_size);
                strassen_multiply(a22, m4b, m4, sub_size, threshold_size);
                free_matrix(m4b, sub_size);
            }

#pragma omp task
            {
                matrix_add(a11, a12, m5a, 0, 0, sub_size, sub_size);
                strassen_multiply(m5a, b22, m5, sub_size, threshold_size);
                free_matrix(m5a, sub_size);
            }

#pragma omp task
            {
#pragma omp task
                matrix_sub(a21, a11, m6a, 0, 0, sub_size, sub_size);
#pragma omp task
                matrix_add(b11, b12, m6b, 0, 0, sub_size, sub_size);
#pragma omp taskwait
                strassen_multiply(m6a, m6b, m6, sub_size, threshold_size);
#pragma omp task
                free_matrix(m6a, sub_size);
#pragma omp task
                free_matrix(m6b, sub_size);
#pragma omp taskwait
            }

#pragma omp task
            {
#pragma omp task
                matrix_sub(a12, a22, m7a, 0, 0, sub_size, sub_size);
#pragma omp task
                matrix_add(b21, b22, m7b, 0, 0, sub_size, sub_size);
#pragma omp taskwait
                strassen_multiply(m7a, m7b, m7, sub_size, threshold_size);
#pragma omp task
                free_matrix(m7a, sub_size);
#pragma omp task
                free_matrix(m7b, sub_size);
#pragma omp taskwait
            }

#pragma omp taskwait
        }

//#pragma omp parallel for
        for (ull i = 0; i < sub_size; ++i) {
#pragma omp parallel for
            for (ull j = 0; j < sub_size; ++j) {
                c11[i][j] = m1[i][j] + m4[i][j] - m5[i][j] + m7[i][j];
                c12[i][j] = m3[i][j] + m5[i][j];
                c21[i][j] = m2[i][j] + m4[i][j];
                c22[i][j] = m1[i][j] - m2[i][j] + m3[i][j] + m6[i][j];
            }
        }
    }
}


int main(int argc, char **argv) {
    if (argc != 4) {
        printf("[error] numebr of arguments is %d\n", argc);
        exit(-1);
    }
    ull size;
    ull strassen_size;

    double **a = load_strassen_matrix(argv[1], size, strassen_size);
    double **b = load_strassen_matrix(argv[2], size, strassen_size);

    double **c = allocate_matrix(strassen_size);

    //omp_set_dynamic(false);
    omp_set_num_threads(48);

//#pragma omp parallel 
    {
//#pragma omp single
        {
            show_time(strassen_multiply(a, b, c, strassen_size, strassen_size/threshold), [info] strassen time);
        }
    }


    save(argv[3], c, size);
    return 0;
}
