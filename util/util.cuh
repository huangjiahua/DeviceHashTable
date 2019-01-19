#ifndef DHT_UTIL_CUH
#define DHT_UTIL_CUH
#include <cstdint>

__device__ int datacmp(const unsigned char *l_dat, const unsigned char *r_dat, uint32_t len); 

#endif