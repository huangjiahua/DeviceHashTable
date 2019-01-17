#ifndef DEVICEHASHTABLE_CUH
#define DEVICEHASHTABLE_CUH
#include "../default_allocator/DefaultAllocator.cuh"
#include "../src/HashFunc.cuh"
#include <cstdint>

#define OVERFLOW_COUNT (1000)

enum IstRet {
    SUCCESSUL = 0,
    UNKNOWN,
    OVERFLOWED,
    FULL,
    MODIFIED,
    DUPLICATE
};

struct DeviceDataBlock {
    void *data;
    uint32_t size; 
};

class DeviceHashTable {
public:
    typedef uint32_t size_type;
private:
    unsigned char *_data_ptr, *_overflow_data_ptr;
    size_type *_elem_info_ptr, *_bkt_info_ptr;
    size_type _bkt_cnt, _bkt_elem_cnt, _max_key_size, _max_elem_size;

    __device__ size_type *getBktCntAddr(size_type bkt_no);
    __device__ size_type *getKeySzAddr(size_type bkt_no, size_type dst);
    __device__ unsigned char *getDataAddr(size_type bkt_no, size_type dst);

public:
    __device__ void setup(uint32_t *nums, unsigned char **ptrs); 

    // Lookup function on device
    __device__ uint32_t memorySize() const;
    __device__ uint32_t maxElementCount() const;
    __device__ uint32_t maxKeySize() const;
    __device__ uint32_t maxValueSize() const;
    __device__ uint32_t bucketCount() const;
    __device__ void *bucketInfoAddress() const;
    __device__ void *elementInfoAddress() const;
    __device__ void *dataAddress() const;
    __device__ void *overflowDataAddress() const;

    // Inserting function
    __device__ IstRet insert(const DeviceDataBlock &key, const DeviceDataBlock &value);
    __device__ void find(const DeviceDataBlock &key, DeviceDataBlock &value);

};


__device__ 
void 
DeviceHashTable::setup (uint32_t *nums, unsigned char **ptrs) {
    _bkt_cnt = nums[0];
    _bkt_elem_cnt = nums[1];
    _max_key_size = nums[2];
    _max_elem_size = nums[2] + nums[3];

    _data_ptr = ptrs[2];
    _overflow_data_ptr = ptrs[3];
    _elem_info_ptr = reinterpret_cast<uint32_t *>(ptrs[1]);
    _bkt_info_ptr = reinterpret_cast<uint32_t *>(ptrs[0]);

}

// Lookup functions
__device__ 
uint32_t
DeviceHashTable::memorySize() const {
    return (sizeof(DeviceHashTable) + 
            (_bkt_cnt + 1) * sizeof(uint32_t) + 
            (_max_elem_size + OVERFLOW_COUNT) * 2 * sizeof(uint32_t) + 
            (_max_elem_size + OVERFLOW_COUNT) * (_max_elem_size));
}

__device__ 
uint32_t
DeviceHashTable::maxElementCount() const {
    return ( (_bkt_cnt) * (_bkt_elem_cnt) );
}

__device__ 
uint32_t
DeviceHashTable::maxKeySize() const {
    return (_max_key_size);
}

__device__ 
uint32_t
DeviceHashTable::maxValueSize() const {
    return ( _max_elem_size - _max_key_size );
}

__device__ 
uint32_t
DeviceHashTable::bucketCount() const {
    return (_bkt_cnt);
}


__device__ 
void *
DeviceHashTable::bucketInfoAddress() const {
    return reinterpret_cast<void *>(_bkt_info_ptr);
}

__device__ 
void *
DeviceHashTable::elementInfoAddress() const {
    return reinterpret_cast<void *>(_elem_info_ptr);
}

__device__ 
void *
DeviceHashTable::dataAddress() const {
    return reinterpret_cast<void *>(_data_ptr);
}

__device__ 
void *
DeviceHashTable::overflowDataAddress() const {
    return reinterpret_cast<void *>(_overflow_data_ptr);
}


__device__ 
IstRet 
DeviceHashTable::insert(const DeviceDataBlock &key, const DeviceDataBlock &value) {
    size_type bkt_no = __hash_func1(key.data, key.size) % _bkt_cnt;
    size_type *indicator_p = getBktCntAddr(bkt_no);
    IstRet ret = IstRet::SUCCESSUL;

    size_type dst = atomicAdd(indicator_p, 1);

    if (dst >= _bkt_elem_cnt) {
        bkt_no = _bkt_cnt; // Overflow bucket
        indicator_p = getBktCntAddr(bkt_no);
        dst = atomicAdd(indicator_p, 1);
        if (dst >= OVERFLOW_COUNT) { // the overflow bucket is full
            return IstRet::FULL;
        }
        ret = IstRet::OVERFLOWED;
    }

    size_type *key_sz_p = getKeySzAddr(bkt_no, dst);
    size_type *val_sz_p = key_sz_p + 1;
    unsigned char *key_data_p = getDataAddr(bkt_no, dst);
    unsigned char *val_data_p = key_data_p + _max_key_size;

    *key_sz_p = key.size;
    *val_sz_p = value.size;
    memcpy(key_data_p, key.data, key.size);
    memcpy(val_data_p, value.data, value.size);

    return ret;
}


__device__ 
void 
DeviceHashTable::find(const DeviceDataBlock &key, DeviceDataBlock &value) {
    size_type bkt_no = __hash_func1(key.data, key.size);
    size_type elem_cnt = *getBktCntAddr(bkt_no);
    unsigned char *bkt = getDataAddr(bkt_no, elem_cnt);

    int i = 0;

    for (; i < elem_cnt; i++, bkt += _max_elem_size) 
        if (memcmp(bkt, key.data, key.size) == 0)
            break;
    
    if (i == elem_cnt) { // not in this bucket (might in overflow bucket)
        bkt_no = _bkt_cnt;
        elem_cnt  *getBktCntAddr(bkt_no);
        bkt = getDataAddr(bkt_no);

        i = 0;
        for (; i < elem_cnt; i++, bkt += _max_elem_size)
            if (memcmp(bkt, key.data, key.size) == 0)
                break;

        if (i >= elem_cnt) { // not found
            value->data = nullptr;
            return;
        }
    }

    memcpy(value.data, bkt + _max_key_size, value.size);
}

__device__ 
typename DeviceHashTable::size_type *
DeviceHashTable::getBktCntAddr(size_type bkt_no) {
    return ( _bkt_info_ptr + bkt_no );
}

__device__ 
typename DeviceHashTable::size_type *
getKeySzAddr(size_type bkt_no, size_type dst) {
    return ( _elem_info_ptr + bkt_no * _bkt_elem_cnt * 2 + dst * 2 );
}

__device__ 
unsigned char *
getDataAddr(size_type bkt_no, size_type dst) {
    return ( _data_ptr + (bkt_no * _bkt_elem_cnt + dst) * _max_elem_size );
}


__global__ 
void 
setupKernel(DeviceHashTable *dht, uint32_t *nums, unsigned char **ptrs) {
    dht->setup(nums, ptrs);
}

__host__
void 
DestroyDeviceHashTable(DeviceHashTable *dht) {
    cudaFree(dht);
}

__host__
void 
CreateDeviceHashTable(
      DeviceHashTable *&dht, 
      uint32_t max_elem_cnt, uint32_t bkt_cnt, 
      uint32_t max_key_size, uint32_t max_val_size) {

    uint32_t bkt_elem_cnt = (max_elem_cnt + bkt_cnt - 1) / bkt_cnt;
    max_elem_cnt = bkt_elem_cnt * bkt_cnt;

    uint32_t _mem_size = 0;
    _mem_size += sizeof(DeviceHashTable);
    _mem_size += sizeof(uint32_t) * ( bkt_cnt + max_elem_cnt * 2 );
    _mem_size += sizeof(uint32_t) * ( 1 + OVERFLOW_COUNT * 2 );
    _mem_size += ( max_elem_cnt + OVERFLOW_COUNT ) * (max_key_size + max_val_size);

    cudaMalloc((void**)&dht, _mem_size);
    cudaMemset((void*)dht, 0x00, sizeof(DeviceHashTable) + sizeof(uint32_t) * (bkt_cnt + 1));

    unsigned char *start = reinterpret_cast<unsigned char *>(dht);
    unsigned char *bkt_info_p = start + sizeof(DeviceHashTable);
    unsigned char *elem_info_p = bkt_info_p + (bkt_cnt + 1) * sizeof(uint32_t);
    unsigned char *data_p = elem_info_p + (max_elem_cnt + OVERFLOW_COUNT) * 2 * sizeof(uint32_t);
    unsigned char *overflow_data_p = data_p + max_elem_cnt * (max_key_size + max_val_size);

    // char *bkt_info_p = NULL;
    // char *elem_info_p = NULL;
    // char *data_p = NULL;
    // char *overflow_data_p = NULL;

    uint32_t numbers[4] = {bkt_cnt, bkt_elem_cnt, max_key_size, max_val_size};
    unsigned char *pointers[4] = {bkt_info_p, elem_info_p, data_p, overflow_data_p};

    uint32_t *dev_numbers;
    unsigned char **dev_pointers;

    cudaMalloc((void**)&dev_numbers, 4 * sizeof(uint32_t));
    cudaMalloc((void**)&dev_pointers, 4 * sizeof(unsigned char *));

    cudaMemcpy(dev_numbers, numbers, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pointers, pointers, 4 * sizeof(unsigned char *), cudaMemcpyHostToDevice);

    setupKernel<<<1, 1>>>(dht, dev_numbers, dev_pointers);

    cudaDeviceSynchronize();
    cudaFree(dev_numbers);
    cudaFree(dev_pointers);
}



#endif // DEVICEHASHTABLE_CUH