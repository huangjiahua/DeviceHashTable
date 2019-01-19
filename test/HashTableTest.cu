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
}

__global__
void insertKernel(DeviceHashTable *dht, DeviceHashTableInsertBlock buf) {
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
void findKernel(DeviceHashTable *dht, DeviceHashTableFindBlock buf) {
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

int main() {
	cout << "Hello" << endl;

	DeviceHashTable *dht = NULL;
	uint32_t *dev_output;
	uint32_t output[5];
	void **dev_ptrs;
	void *ptrs[4];

	uint32_t keys[500], values[500];
	uint32_t key_size[500], value_size[500];
	IstRet ret[500];
	uint32_t *dev_keys, *dev_values;
	uint32_t *dev_key_size, *dev_value_size;
	IstRet *dev_ret;

	for (int i = 0; i < 500; i++) {
		keys[i] = i;
		values[i] = i + 1;
		key_size[i] = value_size[i] = sizeof(uint32_t);
	}

	CreateDeviceHashTable(dht, 5000, 500, 4, 4);

	cudaMalloc((void**)&dev_keys, 500 * sizeof(uint32_t));
	cudaMalloc((void**)&dev_values, 500 * sizeof(uint32_t));
	cudaMalloc((void**)&dev_key_size, 500 * sizeof(uint32_t));
	cudaMalloc((void**)&dev_value_size, 500 * sizeof(uint32_t));
	cudaMalloc((void**)&dev_ret, 500 * sizeof(IstRet));


	cudaMemcpy(dev_keys, keys, 500 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, 500 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_key_size, key_size, 500 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_value_size, value_size, 500 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	DeviceHashTableInsertBlock ins{
		reinterpret_cast<unsigned char*>(dev_keys),
		reinterpret_cast<unsigned char*>(dev_values),
		dev_ret,
		dev_key_size,
		dev_value_size,
		sizeof(uint32_t),
		sizeof(uint32_t),
		500
	};

	insertKernel<<<4, 64>>>(dht, ins);

	cudaMemcpy(ret, dev_ret, 500 * sizeof(IstRet), cudaMemcpyDeviceToHost);
	cudaMemset((void*)dev_values, 0x00, 500 * sizeof(uint32_t));
	cudaMemset((void*)dev_value_size, 0x00, 500 * sizeof(uint32_t));


	DeviceHashTableFindBlock fnd {
		reinterpret_cast<unsigned char*>(dev_keys),
		reinterpret_cast<unsigned char*>(dev_values),
		dev_key_size,
		dev_value_size,
		sizeof(uint32_t),
		sizeof(uint32_t),
		498
	};

	 findKernel<<<4, 64>>>(dht, fnd);



	cudaMemcpy(values, dev_values, 500 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(value_size, dev_value_size, 500 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cout << endl;
	for (int i = 0; i < 500; i++) {
		cout << i << " --- " << values[i] << " --- " << value_size[i] << "  ret: " << ret[i] << endl;
	}
	cudaFree(dev_keys);
	cudaFree(dev_values);
	cudaFree(dev_key_size);
	cudaFree(dev_value_size);
	cudaFree(dev_ret);
	DestroyDeviceHashTable(dht);
#ifdef NEED_PAUSE
	system("pause");
#endif // NEED_PAUSE
}

	//  cudaMalloc((void**)&dev_output, sizeof(uint32_t) * 5);
	//  cudaMalloc((void**)&dev_ptrs, sizeof(void *) * 4);
	
	//  getInfo<<<1, 1>>>(dht, dev_output, dev_ptrs);

	//  cudaMemcpy(output, dev_output, sizeof(uint32_t) * 5, cudaMemcpyDeviceToHost);
	//  cudaMemcpy(ptrs, dev_ptrs, sizeof(void *) * 4, cudaMemcpyDeviceToHost);

	
	//  for (int i = 0; i < 5; i++)
	//  	cout << output[i] << endl;

	//  cout << endl;
	
	//  cout << (uint64_t)dht << endl;

	//  for (int i = 0; i < 4; i++)
	//  	cout << (uint64_t)ptrs[i] << endl;

	//  cudaFree(dev_output);