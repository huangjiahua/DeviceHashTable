#include <stdio.h>
#include <iostream>
#include "../include/DeviceHashTable.cuh"

using namespace std;

__global__
void
print_bkt_info(DeviceHashTable *dht, uint32_t num, uint32_t *addr_out) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t i = tid;
    uint32_t stride = blockDim.x * gridDim.x;



    while (i < num) {
      DeviceHashBucket *ptr = dht->getBucketPtr(tid);
      addr_out[tid] = (int)ptr->_ptr;
      i += stride;
    }

	// if (tid == 0) {
	// 	addr_out[496] = (uint32_t)dht;
	// 	addr_out[497] = (uint32_t)dht->getBucketPtr(0);
	// 	addr_out[498] = (uint32_t)dht->getBucketPtr(1);
	// 	addr_out[499] = (uint32_t)dht->getBucketPtr(345)->_ptr;
	// }
}

int main() {
    DeviceHashTable *dht = nullptr;
    uint32_t *dev_dat;
    uint32_t dat[500];
    printf("hello\n");
    createDeviceHashTable(dht, 5000, 500, 4, 4);

    cudaMalloc((void**)&dev_dat, sizeof(uint32_t) * 500);

    print_bkt_info<<<8, 128>>>(dht, 500, dev_dat);

    cudaMemcpy(dat, dev_dat, 500 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(dev_dat);
    for (int i = 0; i < 500; i++)
        cout << i << ". " << dat[i] << endl;

    destroyDeviceHashTable(dht);
    printf("okay\n");
#ifdef NEED_PAUSE
	system("pause");
#endif // NEED_PAUSE
    return 0;
}