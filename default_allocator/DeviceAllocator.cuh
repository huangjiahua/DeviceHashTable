#ifndef DEVICEALLOCATOR_CUH
#define DEVICEALLOCATOR_CUH
#include <cstdint>

class DeviceAllocator {
public:
    __device__ virtual void *alloc(uint32_t size)=0;
    __device__ virtual void recyc(void *ptr, uint32_t size)=0;

};


#endif // DEVICEALLOCATOR_CUH