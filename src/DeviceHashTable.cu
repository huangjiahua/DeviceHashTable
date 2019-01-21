#include "../include/DeviceHashTable.cuh"
#include "../src/HashFunc.cuh"
#include "../util/util.cuh"
#include <stdio.h>


__device__ 
void 
DeviceHashTable::init(const DHTInitBlock &init_blk) {
    _bkts_p = init_blk.bkts_p;
    if (init_blk.alloc_p != nullptr) 
        _alloc_p = init_blk.alloc_p;
    else {
        _alloc_p = nullptr;
    }
    _bkt_num = init_blk.bkt_num;
}

__device__ 
void 
DeviceHashTable::freeBucket(size_type bkt_no) {
    getBucketPtr(bkt_no)->free(_alloc_p);
}

__device__ 
void 
DeviceHashTable::initBucket(size_type bkt_no, const DHTInitBlock &init_blk) {
    getBucketPtr(bkt_no)->init(_alloc_p, init_blk.bkt_size, init_blk.max_key_size, init_blk.max_elem_size);
}

__device__ 
DeviceHashTable::dhb *
DeviceHashTable::getBucketPtr(size_type bkt_no) {
    return (_bkts_p + bkt_no);
}


__device__ 
void 
DeviceHashTable::setup (uint32_t *nums, unsigned char **ptrs) {
    _bkt_cnt = nums[0];
    _bkt_elem_cnt = nums[1];
    _max_key_size = nums[2];
    _max_elem_size = nums[2] + nums[3];

    _data_ptr = ptrs[2];
    _elem_info_ptr = reinterpret_cast<uint32_t *>(ptrs[1]);
    _bkt_info_ptr = reinterpret_cast<uint32_t *>(ptrs[0]);
    _data_info_ptr = reinterpret_cast<uint32_t *>(ptrs[3]);
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
    return (_bkt_num);
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
IstRet 
DeviceHashTable::insert(const DeviceDataBlock &key, const DeviceDataBlock &value) {
    // printf("HELLO, %d\n", *(uint32_t*)(key.data));
    size_type bkt_no = __hash_func1(key.data, key.size) % _bkt_num;
    dhb *bkt_info = getBucketPtr(bkt_no);
    // printf("inst: key: %d, bkt_no: %d, bkt_addr: %d\n", *(uint32_t*)(key.data), bkt_no, (uint32_t)bkt_info);
    status_type *stat_p;
    unsigned char *data_p;
    size_type *size_p;
    IstRet ret = IstRet::SUCCESSUL;

    
    uint32_t dst = atomicAdd(&(bkt_info->_size), 1);


    if (dst > bkt_info->_capacity) {
        while (atomicOr(&bkt_info->_capacity, 0) < dst) {
            for (int i = 0; i < 10000000; i++) ;
        }
    }

    if (dst == bkt_info->_capacity) { 
        ret = IstRet::OVERFLOWED;
        // ready to reallocate
        uint32_t counter = bkt_info->_capacity;
        uint32_t cap = bkt_info->_capacity;
        uint32_t k = 0;

        // The first two steps aim to make sure no threads are acquiring data from the 
        // dynamic memory area, because if they read when the reallocation is processing
        // it will cause serious problems

        // First, wait all writes are done
        // TODO: this is silly, it will be changed
        while (counter != 0) {
            if (atomicCAS(bkt_info->getStatusPtr(k), VALID, OCCUPIED) == VALID)
                counter--;
            k = (k + 1) % cap;
        }

        // Second, wait all reads are gone
        while (atomicCAS(&(bkt_info->_read_num), 0, -99999) != 0)
            ;

        // Third, it is the time to reallocate
        // it will set all status to VALID and set _read_num to 0 again
        // and it will set the _capacity a new value, so that other waiting
        // threads can continue to insert
        bkt_info->reallocate(_alloc_p); 
    }

    // now we can assure the dst < capacity and we do the insert
    if (dst < bkt_info->_capacity) {
        // it can now write
        stat_p = bkt_info->getStatusPtr(dst);
        data_p = bkt_info->getDataPtr(dst);
        size_p = bkt_info->getKeySizePtr(dst);
        uint32_t r;
        if ((r = atomicCAS(stat_p, EMPTY, OCCUPIED)) != EMPTY) {
            return IstRet::UNKNOWN;
        }

        *size_p = key.size;
        *(size_p + 1) = value.size;
        memcpy(data_p, key.data, key.size);
        memcpy(data_p + bkt_info->_max_key_size, value.data, value.size);
        if (atomicCAS(stat_p, OCCUPIED, VALID) != OCCUPIED) {
            return IstRet::UNKNOWN;
        }
    } 
	return ret;
}


__device__ 
void 
DeviceHashTable::find(const DeviceDataBlock &key, DeviceDataBlock &value) {
    size_type bkt_no = __hash_func1(key.data, key.size) % _bkt_num;
    dhb *bkt_p = getBucketPtr(bkt_no);
    // printf("find: key: %d, bkt_no: %d, bkt_addr: %d\n", *(uint32_t*)(key.data), bkt_no, (uint32_t)bkt_p);
    size_type *stat_p, *size_p;
    unsigned char *data_p;
    size_type size;
    int i;
    
    // if the bucket is being reallocated, wait until the process is done
    while (atomicInc(&bkt_p->_read_num, 0) < 0)
        ;
    
    size = bkt_p->_size;

    // now the dynamic zone is safe to read
    for (i = 0; i < size; i++) {
        stat_p = bkt_p->getStatusPtr(i);
        size_p = bkt_p->getKeySizePtr(i);
        data_p = bkt_p->getDataPtr(i);
        uint32_t ret;

        // wait until the data is properly written
        while (atomicInc(stat_p, VALID) < VALID) // add *stat_p if *stat_p >= VALID
            ;
        
        if (key.size == *size_p && datacmp(reinterpret_cast<unsigned char*>(key.data), data_p, key.size) == 0)
            break;
        
        atomicSub(stat_p, 1);
    }

    if (i < size) { // found
        value.size = *(size_p + 1);
        // printf("%d, %d\n", *(uint32_t*)(key.data), *(uint32_t*)(data_p + _max_key_size));
        memcpy(value.data, data_p + bkt_p->_max_key_size, value.size);
        atomicSub(stat_p, 1);
    } else {
        value.size = 0;
    }

    atomicSub(&bkt_p->_read_num, 1);
}

__device__ 
typename DeviceHashTable::size_type *
DeviceHashTable::getBktCntAddr(size_type bkt_no) {
    return ( &getBucketPtr(bkt_no)->_size );
}

__device__ 
typename DeviceHashTable::size_type *
DeviceHashTable::getKeySzAddr(size_type bkt_no, size_type dst) {
    return ( _elem_info_ptr + bkt_no * _bkt_elem_cnt * 2 + dst * 2 );
}

__device__ 
unsigned char *
DeviceHashTable::getDataAddr(size_type bkt_no, size_type dst) {
    return ( _data_ptr + (bkt_no * _bkt_elem_cnt + dst) * _max_elem_size );
}


__device__ 
DeviceHashTable::status_type *
DeviceHashTable::getStatusAddr(size_type bkt_no, size_type dst) {
    return ( _data_info_ptr + (bkt_no * _bkt_elem_cnt + dst) );
}

__device__ 
DeviceAllocator *
DeviceHashTable::getAllocatorPtr() const {
    return _alloc_p;
}




__global__ 
void
initDHTKernel(DeviceHashTable *dht, DHTInitBlock init_blk) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    // I don't know why, but the cudaMalloc in the bucket init function only works if these five lines of
    // nonsense code exists 
    // -----------------------------------
    // if (tid == 0) {
        // void *ptr;
        // int k = cudaMalloc((void**)&ptr, 32);
        // cudaFree(ptr);
        // }
        // -----------------------------------
    
    if (tid == 0) {
        dht->init(init_blk);
    }
    __syncthreads(); // the alloc pointer should be ready
    
    while (tid < init_blk.bkt_num) {
        // init_blk.bkts_p[tid].init(dht->getAllocatorPtr(), init_blk.bkt_size, init_blk.max_key_size, init_blk.max_elem_size);
		dht->initBucket(tid, init_blk);
        tid += stride;
    }
}

__global__
void
freeDHTKernel(DeviceHashTable *dht) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    uint32_t n = dht->bucketCount();

    uint32_t bkt_no = tid;

    while (bkt_no < n) {
        dht->freeBucket(bkt_no);
        bkt_no += stride;
    }
}

__host__
void 
createDeviceHashTable(
      DeviceHashTable *&dht, 
      uint32_t max_elem_cnt, 
      uint32_t bkt_cnt, 
      uint32_t max_key_size, 
      uint32_t max_val_size,
      DeviceAllocator *alloc_p) {
    uint32_t bkt_size = max_elem_cnt / bkt_cnt;

    uint32_t total_size = sizeof(DeviceHashTable) + bkt_cnt * sizeof(DeviceHashBucket);
    HANDLE_ERROR(cudaMalloc((void**)&dht, total_size));

    unsigned char *ptr = reinterpret_cast<unsigned char *>(dht);
    ptr += sizeof(DeviceHashTable);

    DHTInitBlock dib {
        reinterpret_cast<DeviceHashBucket *>(ptr),
        alloc_p,
        bkt_cnt,
        bkt_size,
        max_key_size,
        max_key_size + max_val_size
    };

    initDHTKernel<<<4, 64>>>(dht, dib);
    HANDLE_ERROR(cudaDeviceSynchronize());
}

__host__
void 
destroyDeviceHashTable(DeviceHashTable *dht) {
    freeDHTKernel<<<4, 64>>>(dht);   
    cudaDeviceSynchronize();
    cudaFree(dht);
}

__global__ 
void 
setupKernel(DeviceHashTable *dht, uint32_t *nums, unsigned char **ptrs) {
    dht->setup(nums, ptrs);
}

__global__ 
void 
getInfoKernel(DeviceHashTable *dht, uint32_t *output, void **output_ptrs) {
	output[0] = dht->memorySize();
	output[1] = dht->maxElementCount();
	output[2] = dht->maxKeySize();
	output[3] = dht->maxValueSize();
	output[4] = dht->bucketCount();
	output_ptrs[0] = dht->bucketInfoAddress();
	output_ptrs[1] = dht->elementInfoAddress();
	output_ptrs[2] = dht->dataAddress();
}

__global__
void 
insertKernel(DeviceHashTable *dht, DeviceHashTableInsertBlock buf) {
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t stride = gridDim.x * blockDim.x;
	DeviceDataBlock key_blk, val_blk;
	IstRet ret; 

	while (tid < buf.len) {
		key_blk.data = buf.key_buf + tid * buf.max_key_size;
		key_blk.size = buf.key_size_buf[tid];
		val_blk.data = buf.val_buf + tid * buf.max_val_size;
		val_blk.size = buf.val_size_buf[tid];
		ret = dht->insert(key_blk, val_blk);
		if (buf.ret_buf != nullptr)
			buf.ret_buf[tid] = ret;
		tid += stride;
	}
}

__global__
void 
findKernel(DeviceHashTable *dht, DeviceHashTableFindBlock buf) {
	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t stride = gridDim.x * blockDim.x;
	DeviceDataBlock key_blk, val_blk;


	while (tid < buf.len) {
		key_blk.data = buf.key_buf + tid * buf.max_key_size;
		key_blk.size = buf.key_size_buf[tid];
		val_blk.data = buf.val_buf + tid * buf.max_val_size;
		dht->find(key_blk, val_blk); // value data is already copied to output buffer 
		buf.val_size_buf[tid] = val_blk.size; // if not found, this size is 0, the user shall know.
		tid += stride;
	} 
}