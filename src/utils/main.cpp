#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <utils/utils.h>

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
