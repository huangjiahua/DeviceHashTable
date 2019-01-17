#ifndef DEFAULTALLOCATOR_CUH
#define DEFAULTALLOCATOR_CUH
#include <cstdint>

class DeviceAllocator {
public:
    __device__ virtual void *alloc(uint64_t size)=0;

};

class DefaultAllocator: public DeviceAllocator {

};


#endif // DEFAULTALLOCATOR_CUH