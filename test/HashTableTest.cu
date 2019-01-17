#include <iostream>
#include <cuda_runtime.h>
#include "../include/DeviceHashTable.cuh"

using namespace std;

__global__ 
void getInfo(DeviceHashTable *dht, uint32_t *output) {
	//output[0] = dht->memorySize();
	output[1] = dht->maxElementCount();
	output[2] = dht->maxKeySize();
	output[3] = dht->maxValueSize();
	output[4] = dht->bucketCount();
}

int main() {
	cout << "Hello" << endl;

	DeviceHashTable *dht = NULL;
	uint32_t *dev_output;
	uint32_t output[5];

	CreateDeviceHashTable(dht, 5000, 500, 4, 4);
	cudaMalloc((void**)&dev_output, sizeof(uint32_t) * 5);
	
	getInfo<<<1, 1>>>(dht, dev_output);

	cudaMemcpy(output, dev_output, sizeof(uint32_t) * 5, cudaMemcpyDeviceToHost);

	cudaFree(dev_output);
	DestroyDeviceHashTable(dht);

	for (int i = 0; i < 5; i++)
		cout << output[i] << endl;
#ifdef NEED_PAUSE
	system("pause");
#endif // NEED_PAUSE
}