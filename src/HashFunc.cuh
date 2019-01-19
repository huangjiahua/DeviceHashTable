#ifndef HASHFUNC_CUH
#define HASHFUNC_CUH

#include <cstdint>

#define	DCHARHASH(h, c)	((h) = 0x63c63cd9*(h) + 0x9c39c33d + (c))

__device__ uint32_t __hash_func1(const void *key, uint32_t len);



#endif // HASHFUNC_CUH