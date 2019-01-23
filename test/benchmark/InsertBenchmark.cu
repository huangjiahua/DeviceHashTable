#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include "../../include/DeviceHashTable.cuh"

typedef std::chrono::time_point<std::chrono::steady_clock> tp;

const uint32_t TOTAL_NUM = 500000; // 50 million

using namespace std;

void generate_random_data(uint32_t *keys, uint32_t *values, uint32_t n) {
	default_random_engine en(chrono::system_clock::now().time_since_epoch().count());
	uniform_int_distribution<uint32_t> dis(0, n);
	for (uint32_t i = 0; i < n; i++) {
		keys[i] = i;
		values[i] = 10;
	}
}

inline unsigned long long elapsed_time(tp from, tp to) {
	return chrono::duration_cast<chrono::milliseconds>(to - from).count();
}

int main() {
	DeviceHashTable *dht = nullptr;

	cudaEvent_t ev_bf_ins, ev_af_ins;
	cudaEventCreate(&ev_bf_ins);
	cudaEventCreate(&ev_af_ins);

	uint32_t *keys, *values;
	uint32_t *key_size, *value_size;

	uint32_t *dev_keys = nullptr, *dev_values = nullptr;
	uint32_t *dev_key_size = nullptr, *dev_value_size = nullptr;

	keys = new uint32_t[TOTAL_NUM];
	values = new uint32_t[TOTAL_NUM];
	key_size = new uint32_t[TOTAL_NUM];
	value_size = new uint32_t[TOTAL_NUM];

	generate_random_data(keys, values, TOTAL_NUM);
	for (int i = 0; i < TOTAL_NUM; i++)
		key_size[i] = value_size[i] = sizeof(uint32_t);

	cout << "Here it begins: " << "Inserting " << TOTAL_NUM << " data elements." <<endl;
	auto bf_crt_tb = chrono::steady_clock::now();
	cudaEventRecord(ev_bf_ins);
	createDeviceHashTable(dht, TOTAL_NUM*2, TOTAL_NUM/10, sizeof(uint32_t), sizeof(uint32_t));
	HANDLE_ERROR(cudaDeviceSynchronize());
	auto af_crt_tb = chrono::steady_clock::now();

	auto bf_ins = chrono::steady_clock::now();
	cudaMalloc((void**)&dev_keys, TOTAL_NUM * sizeof(uint32_t));
	cudaMalloc((void**)&dev_values, TOTAL_NUM * sizeof(uint32_t));
	cudaMalloc((void**)&dev_key_size, TOTAL_NUM * sizeof(uint32_t));
	cudaMalloc((void**)&dev_value_size, TOTAL_NUM * sizeof(uint32_t));

	cudaMemcpy(dev_keys, keys, TOTAL_NUM * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, TOTAL_NUM * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_key_size, key_size, TOTAL_NUM * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_value_size, value_size, TOTAL_NUM * sizeof(uint32_t), cudaMemcpyHostToDevice);

	DeviceHashTableInsertBlock ins{
		reinterpret_cast<unsigned char*>(dev_keys),
		reinterpret_cast<unsigned char*>(dev_values),
		nullptr,
		dev_key_size,
		dev_value_size,
		sizeof(uint32_t),
		sizeof(uint32_t),
		TOTAL_NUM
	};

	auto bf_gpu_ins = chrono::steady_clock::now();

	insertKernel<<<16, 128>>> (dht, ins);
	cudaDeviceSynchronize();

	cudaEventRecord(ev_af_ins);
	cudaEventSynchronize(ev_af_ins);
	auto af_ins = chrono::steady_clock::now();
	auto af_gpu_ins = af_ins;

	float cd_tm;
	cudaEventElapsedTime(&cd_tm, ev_bf_ins, ev_af_ins);

	cout << "Total inserting time: " << elapsed_time(bf_ins, af_ins) << "  from cuda measurement: " << cd_tm << endl;
	cout << "Copying time:         " << elapsed_time(bf_ins, bf_gpu_ins) << endl;
	cout << "GPU inserting time:   " << elapsed_time(bf_gpu_ins, af_gpu_ins) << endl;
	cout << "Table Creating time:  " << elapsed_time(bf_crt_tb, af_crt_tb) << endl;
	cout << "(ms)" << endl;


	// DeviceHashTableFindBlock fnd{
	// 	reinterpret_cast<unsigned char*>(dev_keys),
	// 	reinterpret_cast<unsigned char*>(dev_values),
	// 	dev_key_size,
	// 	dev_value_size,
	// 	sizeof(uint32_t),
	// 	sizeof(uint32_t),
	// 	498
	// };

	// findKernel << <4, 64 >> > (dht, fnd);



	// cudaMemcpy(values, dev_values, 500 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(value_size, dev_value_size, 500 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	// cout << endl;
	// for (int i = 0; i < 500; i++) {
	// 	cout << i << " --- " << values[i] << " --- " << value_size[i] << "  ret: " << ret[i] << endl;
	// }
	cudaFree(dev_keys);
	cudaFree(dev_values);
	cudaFree(dev_key_size);
	cudaFree(dev_value_size);
	destroyDeviceHashTable(dht);

	delete[] keys;
	delete[] values;
	delete[] key_size;
	delete[] value_size;
	cudaEventDestroy(ev_af_ins);
	cudaEventDestroy(ev_bf_ins);
#ifdef NEED_PAUSE
	system("pause");
#endif // NEED_PAUSE
}