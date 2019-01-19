#ifndef DEVICEHASHTABLE_CUH
#define DEVICEHASHTABLE_CUH
#include "../default_allocator/DefaultAllocator.cuh"
#include <cstdint>

#define OVERFLOW_COUNT (1000)

#define EMPTY          (0)
#define VALID          (1)
#define OCCUPIED       (2)
#define READING        (3)


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

struct DeviceHashTableInsertBlock {
    unsigned char *key_buf;
	unsigned char *val_buf;
	IstRet *ret_buf;
    uint32_t *key_size_buf;
	uint32_t *val_size_buf;
    uint32_t max_key_size;
	uint32_t max_val_size;
	uint32_t len;
};

struct DeviceHashTableFindBlock {
    unsigned char *key_buf;
	unsigned char *val_buf;
    uint32_t *key_size_buf;
	uint32_t *val_size_buf;
    uint32_t max_key_size;
	uint32_t max_val_size;
	uint32_t len;
};

class DeviceHashTable {
public:
    typedef uint32_t size_type;
    typedef uint32_t  status_type;
private:
    unsigned char *_data_ptr;
    status_type *_data_info_ptr;
    size_type *_elem_info_ptr, *_bkt_info_ptr;
    size_type _bkt_cnt, _bkt_elem_cnt, _max_key_size, _max_elem_size;

    __device__ size_type *getBktCntAddr(size_type bkt_no);
    __device__ size_type *getKeySzAddr(size_type bkt_no, size_type dst);
    __device__ unsigned char *getDataAddr(size_type bkt_no, size_type dst);
    __device__ status_type *getStatusAddr(size_type bkt_no, size_type dst);

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

    // Inserting function
    __device__ IstRet insert(const DeviceDataBlock &key, const DeviceDataBlock &value);
    __device__ void find(const DeviceDataBlock &key, DeviceDataBlock &value);

};

__global__ void setupKernel(DeviceHashTable *dht, uint32_t *nums, unsigned char **ptrs);
__global__ void findKernel(DeviceHashTable *dht, DeviceHashTableFindBlock buf);
__global__ void insertKernel(DeviceHashTable *dht, DeviceHashTableInsertBlock buf);
__global__ void getInfoKernel(DeviceHashTable *dht, uint32_t *output, void **output_ptrs);

__host__ void destroyDeviceHashTable(DeviceHashTable *dht);

__host__ void createDeviceHashTable(DeviceHashTable *&dht, 
                                    uint32_t max_elem_cnt, uint32_t bkt_cnt, 
                                    uint32_t max_key_size, uint32_t max_val_size); 





#endif // DEVICEHASHTABLE_CUH