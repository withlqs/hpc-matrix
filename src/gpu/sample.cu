#include <stdio.h>

__global__ void vector_add(const int *a, const int *b, int *c) {
    *c = *a + *b;
}

int main(void) {
    const int a = 2, b = 5;
    int c = 0;

    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void **)&dev_a, sizeof(int));
    cudaMalloc((void **)&dev_b, sizeof(int));
    cudaMalloc((void **)&dev_c, sizeof(int));

    cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    vector_add<<<1, 1>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d + %d = %d, Is that right?\n", a, b, c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
