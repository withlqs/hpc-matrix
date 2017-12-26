#include <stdio.h>
#include <queue>

using std::queue;



struct rowfirst_matrix_read {
    FILE *fp;
    long long cache_size;
    queue<double> q;



    matrix(char *file, long long cache_size) {
        fp = fopen(file, "rb");
        if (fp == NULL) {
            printf("error while opening matrix file.");
            exit(-1);
        }
        this->cache_size = cache_size;
    }

    void read(char *file) {
    }

    void wirte(char *file) {
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("number of arguments error.");
    }

    char *m1 = argv[1];
    char *m2 = argv[2];
    char *m3 = argv[3];

    FILE *fp1 = NULL;
    fp = fopen(fp1, "rb");
    if (fseeko(fp, 0 , SEEK_END) != 0) {
        print("error while fseeko.");
        exit(-1);
    }
    long long file_size = ftello(fp);
    fseeko(fp, 0, SEEK_SET);
    printf("file size %zu\n", file_size);
}
