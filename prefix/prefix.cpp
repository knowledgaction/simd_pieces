#include <bits/stdc++.h>
#include <immintrin.h>

int n = 4096;

void prefix_std(int *a, int n) {
    std::partial_sum(a, a + n, a);
}
void prefix_scalar(int *a, int n) {
    for (int i = 1; i < n; i++)
        a[i] += a[i - 1];
}
// =========================simd================================
typedef __m256i v8i;
typedef __m128i v4i;

v4i broadcast(int *p) {
    return (v4i) _mm_broadcast_ss((float*) p);
}
void prefix_simd(int *a, int n) {
    for (int i = 0; i < n; i += 8) {
        v8i x = _mm256_load_si256((v8i*) &a[i]);
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
        _mm256_store_si256((v8i*) &a[i], x);
    }
    v4i s = broadcast(&a[3]);
    for (int i = 4; i < n; i += 4) {
        v4i d = broadcast(&a[i + 3]);
        v4i x = _mm_load_si128((v4i*) &a[i]);
        x = _mm_add_epi32(s, x);
        _mm_store_si128((v4i*) &a[i], x);
        s = _mm_add_epi32(s, d);
    }   
}
// =========================simd+block================================
const int B = (1 << 12);
v4i local_prefix(int *a, v4i s) {
    for (int i = 0; i < B; i += 8) {
        v8i x = _mm256_load_si256((v8i*) &a[i]);
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
        _mm256_store_si256((v8i*) &a[i], x);
    }
    for (int i = 0; i < B; i += 4) {
        v4i d = broadcast(&a[i + 3]);
        v4i x = _mm_load_si128((v4i*) &a[i]);
        x = _mm_add_epi32(s, x);
        _mm_store_si128((v4i*) &a[i], x);
        s = _mm_add_epi32(s, d);
    }
    return s;
}

void prefix_simd_b(int *a, int n) {
    v4i s = _mm_setzero_si128();
    for (int i = 0; i < n; i += B)
        s = local_prefix(a + i, s);
}
// =========================simd+singlepass================================
const v8i perm = _mm256_setr_epi32(0, 0, 0, 0, 3, 3, 3, 3);
const v8i mask = _mm256_setr_epi32(0, 0, 0, 0, -1, -1, -1, -1);
const v8i bcast = _mm256_setr_epi32(7, 7, 7, 7, 7, 7, 7, 7);

void prefix_singlepass(int *a, int n) {
    v8i s = _mm256_setzero_si256();

    for (int i = 0; i < n; i += 8) {
        // __builtin_prefetch(&a[i + (1 << 10)]);
        v8i x = _mm256_load_si256((v8i*) &a[i]);

        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));

        v8i y = _mm256_permutevar8x32_epi32(x, perm);
        y = _mm256_and_si256(y, mask);
        x = _mm256_add_epi32(x, y);

        v8i d = _mm256_permutevar8x32_epi32(x, bcast);
        
        x = _mm256_add_epi32(x, s);
        s = _mm256_add_epi32(s, d);
        
        _mm256_store_si256((v8i*) &a[i], x);   
    }
}

template<class FUNC>
void throughput_est(int* a ,FUNC func){
    int k = 2e9 / n;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < k; i++) {
        __sync_synchronize();
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("%.2f GFLOPS\n", 1e-6 * n * k / duration);
}

int main() {
    // auto a = (int*) std::aligned_alloc(64, 4 * n);
    int* a = static_cast<int*>(_aligned_malloc(4 * n, 64));
    for (int i = 0; i < n; i++)
        a[i] = rand() % 100;
    throughput_est(a, [&](){ return prefix_std(a, n); }); 
    throughput_est(a, [&](){ return prefix_scalar(a, n); }); 
    throughput_est(a, [&](){ return prefix_simd(a, n); }); 
    throughput_est(a, [&](){ return prefix_simd_b(a, n); }); 
    throughput_est(a, [&](){ return prefix_singlepass(a, n); }); 
    return 0;
}