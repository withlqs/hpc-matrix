#include <ctime>
#include <sys/time.h>

#define show_time(func, str) { \
    timeval t_start, t_end; \
    gettimeofday(&t_start, NULL); \
    func; \
    gettimeofday(&t_end, NULL); \
    ull start =  ((ull)t_start.tv_sec)*1000+(ull)t_start.tv_usec/1000; \
    ull end = ((ull)t_end.tv_sec)*1000+(ull)t_end.tv_usec/1000; \
    printf(#str" time:%.3f\n", (double)(end-start)/1000); \
} while (false)

typedef unsigned long long ull;

ull int_root(ull num);
