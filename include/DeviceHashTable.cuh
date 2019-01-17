#ifndef DEVICEHASHTABLE_CUH
#define DEVICEHASHTABLE_CUH
#include "../default_allocator/DefaultAllocator.cuh"
#include <cstdint>

#define OVERFLOW_COUNT (1000)

enum IstRet {
    SUCCESS = 0,
    UNKNOWN
};

struct DeviceDataBlock {
    void *data;
    uint32_t size; 
};

class DeviceHashTable {
public:
    typedef uint32_t size_type;
private:
    char *_data_ptr, *_overflow_data_ptr;
    size_type *_elem_info_ptr, *_bkt_info_ptr;
    size_type _bkt_cnt, _bkt_elem_cnt, _max_key_size, _max_elem_size;

public:
    __device__ void setup(uint32_t *nums, char **ptrs); 

    // Lookup function on device
    __device__ uint32_t memorySize() const;
    __device__ uint32_t maxElementCount() const;
    __device__ uint32_t maxKeySize() const;
    __device__ uint32_t maxValueSize() const;
    __device__ uint32_t bucketCount() const;

    // Inserting function
    __device__ IstRet insert(const DeviceDataBlock &key, const DeviceDataBlock &value);
    __device__ void find(const DeviceDataBlock &key, DeviceDataBlock &value);

};


// Constructor
__device__ void DeviceHashTable::setup (uint32_t *nums, char **ptrs) {
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
	return 0;
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

__global__ void setupKernel(DeviceHashTable *dht, uint32_t *nums, char **ptrs) {
    dht->setup(nums, ptrs);
}

void DestroyDeviceHashTable(DeviceHashTable *dht) {
    cudaFree(dht);
}

void CreateDeviceHashTable(
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

    char *start = reinterpret_cast<char *>(dht);
    char *bkt_info_p = start + sizeof(DeviceHashTable);
    char *elem_info_p = bkt_info_p + (bkt_cnt + 1) * sizeof(uint32_t);
    char *data_p = elem_info_p + (max_elem_cnt + OVERFLOW_COUNT) * 2 * sizeof(uint32_t);
    char *overflow_data_p = data_p + max_elem_cnt * (max_key_size + max_val_size);

    // char *bkt_info_p = NULL;
    // char *elem_info_p = NULL;
    // char *data_p = NULL;
    // char *overflow_data_p = NULL;

    uint32_t numbers[4] = {bkt_cnt, bkt_elem_cnt, max_key_size, max_val_size};
    char *pointers[4] = {bkt_info_p, elem_info_p, data_p, overflow_data_p};

    uint32_t *dev_numbers;
    char **dev_pointers;

    cudaMalloc((void**)&dev_numbers, 4 * sizeof(uint32_t));
    cudaMalloc((void**)&dev_pointers, 4 * sizeof(char *));

    cudaMemcpy(dev_numbers, numbers, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pointers, pointers, 4 * sizeof(char *), cudaMemcpyHostToDevice);

    setupKernel<<<1, 1>>>(dht, dev_numbers, dev_pointers);

    cudaDeviceSynchronize();
    cudaFree(dev_numbers);
    cudaFree(dev_pointers);
}



#endif // DEVICEHASHTABLE_CUH