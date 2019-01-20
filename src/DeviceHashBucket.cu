#include "../include/DeviceHashTable.cuh"
#include <stdio.h>


__device__ 
void 
DeviceHashBucket::init(
      DeviceAllocator *alloc_p, 
      size_type data_size, 
      size_type max_key_size, 
      size_type max_elem_size) {
    _capacity = data_size;
    _size = 0;
    _read_num = 0;
    _max_key_size = max_key_size;
    _max_elem_size = max_elem_size;
    size_type total_size = 0;
    total_size += data_size * sizeof(status_type);
    total_size += data_size * 2 * sizeof(size_type);
    total_size += data_size * (_max_elem_size);

    if (alloc_p != nullptr)
        _ptr = reinterpret_cast<unsigned char*>(alloc_p->alloc(total_size));
    else
        cudaMalloc((void**)&_ptr, total_size);
}


__device__ 
void 
DeviceHashBucket::free(DeviceAllocator *alloc_p) {
    if (alloc_p != nullptr) {
        alloc_p->recyc(_ptr, 0);
    } else {
        cudaFree(_ptr);
    }
}