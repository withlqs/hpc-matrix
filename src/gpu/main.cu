#include <cstdio>
#include <cstdlib>
#include <common.h>
#include <cuda_runtime.h>
#include <algorithm>

#define block_size 32

using std::swap;


ull get_padding_size(ull size) {
    return size+(block_size-size%block_size)%block_size;
}

void transpose(double *data, ull size) {
    for (ull i = 0; i < size; ++i) {
        for (ull j = i+1; j < size; ++j) {
            swap(data[i*size+j], data[j*size+i]);
        }
    }
}

double *load_padding_mat(const char* file, ull &mat_size, ull &padding_size) {
    FILE *fp = fopen(file, "rb");
    fseek(fp, 0L, SEEK_END);
    ull file_size = ftell(fp);
    rewind(fp);
    mat_size = int_root(file_size/sizeof(double));
    padding_size = get_padding_size(mat_size);
    double *data = (double *)aligned_alloc(64, sizeof(double)*padding_size*padding_size);
    for (ull i = 0; i < padding_size*padding_size; ++i) {
        data[i] = 0;
    }
    if (mat_size == padding_size) {
        fread(data, sizeof(double)*mat_size*mat_size, 1, fp);
    } else {
        for (ull i = 0; i < mat_size; ++i) {
            fread(data+i*padding_size, sizeof(double)*mat_size, 1, fp);
        }
    }
    fclose(fp);
    printf("[info] load a %llux%llu matrix as %llux%llu padding matrix finished\n", mat_size, mat_size, padding_size, padding_size);
    return data;
}

__global__ void matrix_multiply(ui size, double *a, double *b, double *c) {

    __shared__ double sub_a[block_size][block_size];
    __shared__ double sub_b[block_size][block_size];
    ui thread_x = threadIdx.x;
    ui thread_y = threadIdx.y;
    ui block_x = blockIdx.x;
    ui block_y = blockIdx.y;

    ui begin = size*block_size*block_y;
    ui end = begin+size;

    double sum = 0;

    for (ui x = begin; x < end; x += block_size) {
        sub_a[thread_y][thread_x] = a[x+thread_y*size+thread_x];
        sub_b[thread_y][thread_x] = b[x+thread_y*size+thread_x];
        __syncthreads();
#pragma unroll
        for (ui k = 0; k < block_size; ++k) {
            sum += sub_a[thread_y][k]*sub_b[thread_x][k];
        }
        __syncthreads();
    }
    c[size*block_size*block_y+size*thread_y+block_x*block_size+thread_x] = sum;
}

void save_mat(const char* file, double* data, ull mat_size, ull padding_size) {
    FILE *fp = fopen(file, "wb");
    if (mat_size == padding_size) {
        fwrite(data, sizeof(double)*mat_size*mat_size, 1, fp);
    } else {
        for (ull i = 0; i < mat_size; ++i) {
            fwrite(data+i*padding_size, sizeof(double)*mat_size, 1, fp);
        }
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("[error] number of arguments is error.\n");
        return 1;
    }

    ull mat_size;
    ull padding_size;

    double *a = load_padding_mat(argv[1], mat_size, padding_size);
    double *b = load_padding_mat(argv[2], mat_size, padding_size);
    transpose(b, padding_size);

    double *device_a;
    double *device_b;
    double *device_c;

    ull mem_size = sizeof(double)*padding_size*padding_size;
    double *c = (double *)aligned_alloc(64, mem_size);

    cudaMalloc(&device_a, mem_size);
    cudaMalloc(&device_b, mem_size);
    cudaMalloc(&device_c, mem_size);

    cudaMemcpy(device_a, a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, mem_size, cudaMemcpyHostToDevice);

    dim3 thread_block(block_size, block_size, 1);
    dim3 thread_grid(padding_size/block_size, padding_size/block_size, 1);

    matrix_multiply<<<thread_grid, thread_block>>>(padding_size, device_a, device_b, device_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, device_c, mem_size, cudaMemcpyDeviceToHost);
    save_mat(argv[3], c, mat_size, padding_size);
    return 0;
}
