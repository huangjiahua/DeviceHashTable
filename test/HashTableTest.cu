#include <iostream>
#include <cuda_runtime.h>
#include "../include/DeviceHashTable.cuh"

using namespace std;



int main() {
	cout << "Hello" << endl;

	DeviceHashTable *dht = NULL;

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

	createDeviceHashTable(dht, 20, 20, 4, 4);

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
	destroyDeviceHashTable(dht);
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