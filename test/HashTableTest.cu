#include <iostream>
#include <cuda_runtime.h>
#include "../include/DeviceHashTable.cuh"

using namespace std;

__global__ 
void getInfo(DeviceHashTable *dht, uint32_t *output, void **output_ptrs) {
	output[0] = dht->memorySize();
	output[1] = dht->maxElementCount();
	output[2] = dht->maxKeySize();
	output[3] = dht->maxValueSize();
	output[4] = dht->bucketCount();
	output_ptrs[0] = dht->bucketInfoAddress();
	output_ptrs[1] = dht->elementInfoAddress();
	output_ptrs[2] = dht->dataAddress();
	output_ptrs[3] = dht->overflowDataAddress();
}

int main() {
	cout << "Hello" << endl;

	DeviceHashTable *dht = NULL;
	uint32_t *dev_output;
	uint32_t output[5];
	void **dev_ptrs;
	void *ptrs[4];

	CreateDeviceHashTable(dht, 5000, 500, 4, 4);
	cudaMalloc((void**)&dev_output, sizeof(uint32_t) * 5);
	cudaMalloc((void**)&dev_ptrs, sizeof(void *) * 4);
	
	getInfo<<<1, 1>>>(dht, dev_output, dev_ptrs);

	cudaMemcpy(output, dev_output, sizeof(uint32_t) * 5, cudaMemcpyDeviceToHost);
	cudaMemcpy(ptrs, dev_ptrs, sizeof(void *) * 4, cudaMemcpyDeviceToHost);

	
	for (int i = 0; i < 5; i++)
		cout << output[i] << endl;
	
	cout << (uint64_t)dht << endl;

	for (int i = 0; i < 4; i++)
		cout << (uint64_t)ptrs[i] << endl;

	cudaFree(dev_output);
	DestroyDeviceHashTable(dht);
#ifdef NEED_PAUSE
	system("pause");
#endif // NEED_PAUSE
}