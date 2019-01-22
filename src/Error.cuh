#ifndef ERROR_CUH
#define ERROR_CUH
#include <stdio.h>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		system("pause");
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif