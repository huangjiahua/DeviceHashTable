#include "HashFunc.cuh"

__device__
uint32_t
__hash_func1(const void *key, uint32_t len) {
	const uint8_t *e, *k;
	uint32_t h;
	uint8_t c;

	k = reinterpret_cast<const uint8_t *>(key);
	e = k + len;
	for (h = 0; k != e;) {
		c = *k++;
		if (!c && k > e)
			break;
		DCHARHASH(h, c);
	}
	return (h);
}
