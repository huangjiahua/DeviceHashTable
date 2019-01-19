#include "util.cuh"

__device__
int 
datacmp(const unsigned char *l_dat, const unsigned char *r_dat, uint32_t len) {
    int match = 0;
    uint32_t i = 0;
    uint32_t done = 0;
    while ( i < len && match == 0 && !done ) {
        if ( l_dat[i] != r_dat[i] ) {
            match = i + 1;
            if ( (int)l_dat[i] - (int)r_dat[i] < 0 ) {
                match = 0 - (i + 1);
            }
        }
        i++;
    }
    return match;
}