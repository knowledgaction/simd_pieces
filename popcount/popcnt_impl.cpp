#include <bits/stdc++.h>
#include <immintrin.h>

#define N (1<<12)

const int K = 2e9 / N;
alignas(32) int a[N];

 // naive implement
int popcnt_naive() {
    int res = 0;
    for (int i = 0; i < N; i++)
        for (int l = 0; l < 32; l++)
            res += (a[i] >> l & 1);
    return res;
}

 // builtin int
int popcnt_i() {
    int res = 0;
    for (int i = 0; i < N; i++)
        res += __builtin_popcount(a[i]);
    return res;
}

 // lut
alignas(64) constexpr auto LUT = [](){
    constexpr size_t SIZE = 1<<16;
    std::array<char, SIZE> counts;
    for (int i = 0; i < SIZE; i++)
        counts[i] = __builtin_popcount(i);
    return counts;
}();

int popcnt_lut() {
    auto b = (unsigned short*) a;
    int res = 0;
    for (int i = 0; i < 2 * N; i++)
        res += LUT[b[i]];
    return res;
}

// simd -- pshufb
typedef __m256i reg;

const reg lookup = _mm256_setr_epi8(
    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
);

const reg low_mask = _mm256_set1_epi8(0x0f);

int hsum(__m256i x) {
    __m128i l = _mm256_extracti128_si256(x, 0);
    __m128i h = _mm256_extracti128_si256(x, 1);
    l = _mm_add_epi32(l, h);
    l = _mm_hadd_epi32(l, l);
    return _mm_extract_epi32(l, 0) + _mm_extract_epi32(l, 1);
}

const int block_size = (255 / 8) * 8;

int popcnt_simd() {
    int k = 0;

    reg t = _mm256_setzero_si256();

    for (; k + block_size < N; k += block_size) {
        reg s = _mm256_setzero_si256();
        
        for (int i = 0; i < block_size; i += 8) {
            reg x = _mm256_load_si256( (reg*) &a[k + i] );
            
            reg l = _mm256_and_si256(x, low_mask);
            reg h = _mm256_and_si256(_mm256_srli_epi16(x, 4), low_mask);

            reg pl = _mm256_shuffle_epi8(lookup, l);
            reg ph = _mm256_shuffle_epi8(lookup, h);

            s = _mm256_add_epi8(s, pl);
            s = _mm256_add_epi8(s, ph);
        }

        t = _mm256_add_epi64(t, _mm256_sad_epu8(s, _mm256_setzero_si256()));
    }

    int res = hsum(t);

    while (k < N)
        res += __builtin_popcount(a[k++]);

    return res;
}

template<class FUNC>
void throughput_est(int* a ,FUNC func){
    int checksum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < K; t++) {
        __sync_synchronize();
        checksum += func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("%.2f GFLOPS\n", 1e-6 * N * K / duration);

    printf("%d\n", checksum);
}

int main() {
    for (int i = 0; i < N; i++)
        a[i] = rand();

    throughput_est(a, popcnt_naive);
    throughput_est(a, popcnt_lut);
    throughput_est(a, popcnt_i);
    throughput_est(a, popcnt_simd);

    return 0;
}