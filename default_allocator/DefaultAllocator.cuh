#ifndef DEFAULTALLOCATOR_CUH
#define DEFAULTALLOCATOR_CUH
#include "DeviceAllocator.cuh"


class DefaultAllocator: public DeviceAllocator {
private:
    int i;
public:
    __device__  void *alloc(uint32_t size) override {
        void *ptr;
        cudaMalloc((void**)&ptr, size);
        return ptr;
    }
    __device__  void recyc(void *ptr, uint32_t size) override {
        cudaFree(ptr);
    }
};


#endif // DEFAULTALLOCATOR_CUH