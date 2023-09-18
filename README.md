# simd_pieces
simd toy programs &amp; study notes



## popcount

| 具体实现方法 | 吞吐         |
| ------------ | ------------ |
| naive        | 0.81 GFLOPS  |
| builtin      | 2.54 GFLOPS  |
| LUT          | 4.36 GFLOPS  |
| pshufb(simd) | 14.81 GFLOPS |

## argmin

| 具体实现方法 | 吞吐         |
| ------------ | ------------ |
| std          | 0.74 GFLOPS  |
| scalar       | 2.23 GFLOPS  |
| simd         | 40.00 GFLOPS |





# Other links

* [sslotin/amh-code: Complete implementations from "Algorithms for Modern Hardware" ](https://github.com/sslotin/amh-code)

* [parallel101/simdtutor: x86-64 SIMD矢量优化系列教程](https://github.com/parallel101/simdtutor)
