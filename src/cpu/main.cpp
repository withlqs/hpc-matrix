#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <common.h>

typedef unsigned long long ull;

using std::swap;

ull threshold = 16;
ull atom_size = 0;

inline void transpose(double **data, ull x, ull y, ull rows, ull cols) {
    for (ull i = 0; i < rows; ++i) {
        for (ull j = i+1; j < cols; ++j) {
            swap(data[i+x][j+y], data[j+x][i+y]);
        }
    }
}

inline void block_transpose(double **origin, ull size) {
    ull block_size = size/threshold;
    for (ull i = 0; i < size; i += block_size) {
        for (ull j = 0; j < size; j += block_size) {
            transpose(origin, i, j, block_size, block_size);
        }
    }
}

inline ull get_padding_size(ull size) {
    ull padding = threshold*4;
    return size+(padding-size%padding)%padding;
}

inline double **allocate_2d(ull size) {
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

inline void copy_2d(double **a, double **b, ull x, ull y, ull rows, ull cols) {
    for (ull i = x; i < x+rows; ++i) {
        memcpy(a[i]+y, b[i]+y, sizeof(double)*cols);
    }
}

inline void free_2d(double **data) {
    free(data[0]);
    free(data);
}

inline double **mapping_2d(double **a, ull x, ull y, ull size) {
    double **b = (double **) aligned_alloc(64, sizeof(double *)*size);
    for (ull i = x; i < x+size; ++i) {
        b[i-x] = a[i]+y;
    }
    return b;
}

struct matrix {
    bool real_value;
    ull size;
    double **data;
    matrix *m11, *m12, *m21, *m22;

    matrix(double **value_data, ull data_size) {
        size = data_size;
        if (size == atom_size) {
            real_value = true;
            data = allocate_2d(size);
            copy_2d(data, value_data, 0, 0, size, size);
        } else if (size > atom_size) {
            real_value = false;
            ull sub_size = size/2;
            m11 = new matrix(mapping_2d(value_data, 0, 0, sub_size), sub_size);
            m12 = new matrix(mapping_2d(value_data, 0, sub_size, sub_size), sub_size);
            m21 = new matrix(mapping_2d(value_data, sub_size, 0, sub_size), sub_size);
            m22 = new matrix(mapping_2d(value_data, sub_size, sub_size, sub_size), sub_size);
        }
    }

    matrix(matrix *a11, matrix *a12, matrix *a21, matrix *a22, ull value_size) {
        real_value = false;
        size = value_size;
        m11 = a11;
        m12 = a12;
        m21 = a21;
        m22 = a22;
    }

    ~matrix() {
        if (real_value) {
            free_2d(data);
        } else {
            delete m11;
            delete m12;
            delete m21;
            delete m22;
        }
    }
};

inline void block_free(matrix *m) {
    delete m;
}

inline matrix *load_block_matrix(const char* file, ull &mat_size, ull &padding_size, bool trans) {
    FILE *fp = fopen(file, "rb");
    fseek(fp, 0L, SEEK_END);
    ull file_size = ftell(fp);
    rewind(fp);
    mat_size = int_root(file_size/sizeof(double));
    padding_size = get_padding_size(mat_size);
    double **data = allocate_2d(padding_size);
    if (mat_size == padding_size) {
        fread(data[0], sizeof(double)*mat_size*mat_size, 1, fp);
    } else {
        for (ull i = 0; i < mat_size; ++i) {
            fread(data[i], sizeof(double)*mat_size, 1, fp);
        }
    }
    fclose(fp);

    atom_size = padding_size/threshold;

    if (trans) {
        block_transpose(data, padding_size);
    }
    matrix *m = new matrix(data, padding_size);
    free_2d(data);

    printf("[info] load %s as a %llux%llu matrix as %llux%llu padding matrix finished\n", file, mat_size, mat_size, padding_size, padding_size);

    return m;
}


inline void restore_2d(matrix *m, ull x, ull y, double **data) {
    ull size = m->size;
    if (m->real_value) {
        for (ull i = 0; i < size; ++i) {
            for (ull j = 0; j < size; ++j) {
                data[i+x][j+y] = m->data[i][j];
            }
        }
    } else {
        ull sub_size = size/2;
        restore_2d(m->m11, x, y, data);
        restore_2d(m->m12, x, y+sub_size, data);
        restore_2d(m->m21, x+sub_size, y, data);
        restore_2d(m->m22, x+sub_size, y+sub_size, data);
    }
}


inline void save_to_fp(FILE *fp, double **data, ull size) {
    for (ull i = 0; i < size; ++i) {
        fwrite(data[i], sizeof(double)*size, 1, fp);
    }
}

inline void save(const char* file, matrix *m, ull origin_size) {
    FILE *fp = fopen(file, "wb");
    double **data = allocate_2d(m->size);
    restore_2d(m, 0, 0, data);
    save_to_fp(fp, data, origin_size);
    fclose(fp);
}


inline matrix *naive_multiply(matrix *a, matrix *b) {
    ull size = a->size;
    matrix *c = new matrix(allocate_2d(size), size);
    __m256d av, bv, cv, dv;
    for (ull i = 0; i < size; ++i) {
        for (ull j = 0; j < size; ++j) {
            for (ull k = 0; k < size; k += 4) {
                av = _mm256_load_pd(&a->data[i][k]);
                bv = _mm256_load_pd(&b->data[j][k]);
                cv = _mm256_mul_pd(av, bv);
                dv = _mm256_hadd_pd(cv, cv);
                c->data[i][j] += (double)dv[0]+(double)dv[2];
            }
        }
    }
    return c;
}

inline matrix *naive_add(matrix *a, matrix *b) {
    ull size = a->size;
    matrix *c = new matrix(allocate_2d(size), size);

    __m256d av, bv, cv;
    for (ull i = 0; i < size; ++i) {
        for (ull j = 0; j < size; j += 4) {
            av = _mm256_load_pd(&a->data[i][j]);
            bv = _mm256_load_pd(&b->data[i][j]);
            cv = _mm256_add_pd(av, bv);
            c->data[i][j] = (double)cv[0];
            c->data[i][j+1] = (double)cv[1];
            c->data[i][j+2] = (double)cv[2];
            c->data[i][j+3] = (double)cv[3];
        }
    }

    return c;
}

inline matrix *block_add(matrix *a, matrix *b) {
    if (a->size == atom_size) {
        return naive_add(a, b);
    } else {
        ull size = a->size;
        ull sub_size = size/2;

        matrix *a11 = a->m11;
        matrix *a12 = a->m12;
        matrix *a21 = a->m21;
        matrix *a22 = a->m22;

        matrix *b11 = b->m11;
        matrix *b12 = b->m12;
        matrix *b21 = b->m21;
        matrix *b22 = b->m22;

        matrix *c11 = block_add(a11, b11);
        matrix *c12 = block_add(a12, b12);
        matrix *c21 = block_add(a21, b21);
        matrix *c22 = block_add(a22, b22);

        matrix *c = new matrix(c11, c12, c21, c22, size);

        return c;
    }
}

inline matrix *block_multiply(matrix *a, matrix *b) {
    if (a->size <= atom_size) {
        return naive_multiply(a, b);
    } else {
        ull size = a->size;
        ull sub_size = size/2;

        matrix *a11 = a->m11;
        matrix *a12 = a->m12;
        matrix *a21 = a->m21;
        matrix *a22 = a->m22;

        matrix *b11 = b->m11;
        matrix *b12 = b->m12;
        matrix *b21 = b->m21;
        matrix *b22 = b->m22;

        matrix *c11;
        matrix *c12;
        matrix *c21;
        matrix *c22;

#pragma omp task shared(c11)
        {
            matrix *c111 = block_multiply(a11, b11);
            matrix *c112 = block_multiply(a12, b21);
            c11 = block_add(c111, c112);
            block_free(c111);
            block_free(c112);
        }

#pragma omp task shared(c12)
        {
            matrix *c121 = block_multiply(a11, b12);
            matrix *c122 = block_multiply(a12, b22);
            c12 = block_add(c121, c122);
            block_free(c121);
            block_free(c122);
        }

#pragma omp task shared(c21)
        {
            matrix *c211 = block_multiply(a21, b11);
            matrix *c212 = block_multiply(a22, b21);
            c21 = block_add(c211, c212);
            block_free(c211);
            block_free(c212);
        }

#pragma omp task shared(c22)
        {
            matrix *c221 = block_multiply(a21, b12);
            matrix *c222 = block_multiply(a22, b22);
            c22 = block_add(c221, c222);
            block_free(c221);
            block_free(c222);
        }

#pragma omp taskwait

        matrix *c = new matrix(c11, c12, c21, c22, size);

        return c;
    }
}


int main(int argc, char **argv) {
    if (argc != 4) {
        printf("[error] numebr of arguments is %d\n", argc);
        exit(-1);
    }

    ull size;
    ull padding_size;

    matrix *a = load_block_matrix(argv[1], size, padding_size, false);
    matrix *b = load_block_matrix(argv[2], size, padding_size, true);

    matrix *c;

    omp_set_num_threads(24);

#pragma omp parallel
    {  
#pragma omp single
        show_time(c = block_multiply(a, b), [info] block time);
    }

    save(argv[3], c, size);
    return 0;
}
