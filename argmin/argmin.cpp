#include <bits/stdc++.h>
#include <immintrin.h>

#define N (1<<16)
const int K = 1e9 / N;
alignas(64) int a[N];

 // std implement
int argmin_std(int *a, int n) {
    int k = std::min_element(a, a + n) - a;
    return k;
}

 // simple implement
int argmin_simple(int *a, int n) {
    int k = 0;
    for (int i = 0; i < n; i++)
        if (a[i] < a[k])
            k = i;
    return k;
}

// simd
typedef __m256i reg;

int argmin_simd(int *a, int n) {
    int min = INT_MAX, idx = 0;
    
    reg p = _mm256_set1_epi32(min);

    for (int i = 0; i < n; i += 32) {
        reg y1 = _mm256_load_si256((reg*) &a[i]);
        reg y2 = _mm256_load_si256((reg*) &a[i + 8]);
        reg y3 = _mm256_load_si256((reg*) &a[i + 16]);
        reg y4 = _mm256_load_si256((reg*) &a[i + 24]);
        y1 = _mm256_min_epi32(y1, y2);
        y3 = _mm256_min_epi32(y3, y4);
        y1 = _mm256_min_epi32(y1, y3);
        reg mask = _mm256_cmpgt_epi32(p, y1);
        if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
            idx = i;
            for (int j = i; j < i + 32; j++)
                min = (a[j] < min ? a[j] : min);
            p = _mm256_set1_epi32(min);
        }
    }

    for (int i = idx; i < idx + 31; i++)
        if (a[i] == min)
            return i;
    
    return idx + 31;
}


template<class FUNC>
void throughput_est(int* a ,FUNC func){
    volatile int k = func();
    
    int m = (0 <= k && k < N ? a[k] : -1);
    printf("%d %d\n", k, m);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < K; i++) {
        __sync_synchronize();
        k = func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("%.2f GFLOPS\n", 1e-6 * N * K / duration);

}

int main() {
    for (int i = 0; i < N; i++)
        a[i] = rand();

    throughput_est(a, [&](){ return argmin_std(a, N); }); // 传递 lambda 函数，该 lambda 函数接受无参数，并调用 argmin_std 函数并传递参数
    throughput_est(a, [&](){ return argmin_simple(a, N); });
    throughput_est(a, [&](){ return argmin_simd(a, N); });
    return 0;
}