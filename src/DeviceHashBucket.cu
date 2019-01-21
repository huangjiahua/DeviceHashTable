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
    
    memset(_ptr, 0x00, _capacity * sizeof(size_type));
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

__device__ 
void 
DeviceHashBucket::reallocate(DeviceAllocator *alloc_p) {
    unsigned char *new_ptr;
    size_type curr_size = getTotalSize();
    size_type curr_cap = _capacity;
    size_type new_cap = curr_cap * 2;

    // allocation
    if (alloc_p != nullptr) {
		new_ptr = reinterpret_cast<unsigned char*>(alloc_p->alloc(curr_size * 2));
    } else {
        cudaMalloc((void**)&new_ptr, curr_size * 2);
    }

    // copying

    // TODO: this is silly, it will be changed to memcpy 
    size_type *szs = reinterpret_cast<size_type*>(new_ptr);
    for (size_type i = 0; i < curr_cap; i++) {
        szs[i] = VALID;
    }

    memcpy(new_ptr + new_cap * sizeof(size_type),
           _ptr + curr_cap * sizeof(size_type),
           curr_cap * 2 * sizeof(size_type));

    memcpy(new_ptr + new_cap * 3 * sizeof(size_type),
           _ptr + curr_cap * 3 * sizeof(size_type),
           curr_cap * _max_elem_size);
    

    // chage the capacity
    atomicExch(&_capacity, new_cap);
    atomicExch(&_read_num, 0);
    cudaFree(_ptr);
    _ptr = new_ptr;
}

__device__ 
DeviceHashBucket::size_type *
DeviceHashBucket::getStatusPtr(size_type offset) const {
    return ( reinterpret_cast<size_type*>(_ptr) + offset );
}

__device__ 
DeviceHashBucket::size_type *
DeviceHashBucket::getKeySizePtr(size_type offset) const {
    return ( reinterpret_cast<size_type*>(_ptr) + _capacity + 2 * offset );
}

__device__ 
unsigned char *
DeviceHashBucket::getDataPtr(size_type offset) const {
    return ( _ptr + 3 * sizeof(size_type) * _capacity + offset * _max_elem_size );
}

__device__ 
DeviceHashBucket::size_type 
DeviceHashBucket::getTotalSize() const {
    return ( ( sizeof(size_type) * 3 + _max_elem_size ) * _capacity );
}