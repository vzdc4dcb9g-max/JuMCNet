#include <cstdint>
#include <cstddef>
extern "C" void lut_chw(const uint8_t* src, const float* lut, float* dst, size_t n) {
    float* r=dst; float* g=dst+n; float* b=dst+2*n;
    for (size_t i=0;i<n;++i) {
        r[i]=lut[src[3*i]];
        g[i]=lut[256+src[3*i+1]];
        b[i]=lut[512+src[3*i+2]];
    }
}
