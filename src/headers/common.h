#include <ctime>
#include <sys/time.h>

#define show_time(func, str) { \
    timeval t_start, t_end; \
    gettimeofday(&t_start, NULL); \
    func; \
    gettimeofday(&t_end, NULL); \
    ull start =  (ull)t_start.tv_sec*1000000+(ull)t_start.tv_usec; \
    ull end = (ull)t_end.tv_sec*1000000+(ull)t_end.tv_usec; \
    printf(#str" time:%.6f\n", (double)(end-start)/1000000); \
} while (false)

typedef unsigned long long ull;

ull int_root(ull num);
ull load_mat(const char* file, double* &data);
void save_mat(const char* file, double* data, ull mat_size);
