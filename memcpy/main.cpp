#include <algorithm>
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <immintrin.h>
#include <cstring>
#include <chrono>

const int n = (1<<13), k = 1e10 / n, t = 32;
alignas(32) char a[t][n], b[t][n];

void naive_memcpy(int idx){
    for (int i = 0; i < k; i++) {
        memset(a[idx], 0, n);
        memcpy(b[idx], a[idx], n);
        for (int j = 0; j < n; j++){
        a[idx][j] += b[idx][j];
        a[idx][j] = b[idx][j];
        }
    }
}



int main(){
    std::vector<std::thread> threadpool;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i =0; i<t; ++i){
        threadpool.push_back(std::thread(naive_memcpy, i));
    };
    for(auto &t : threadpool) t.join();

    auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("%.2f GB/s\n", 1e-9 * n * k * t / elapsed);
	return 0;	
}

