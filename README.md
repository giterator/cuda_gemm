# Performant CUDA GEMM

Performant implementation of CUDA GEMM kernel written from scratch. The goal was to reproduce Simon Boehm's results and document my learnings: https://giterator.github.io/blog/posts/gemm/

## Results
The kernel was run on a Tesla T4 GPU for a variety of matrix sizes and the performance relative to cuBLAS is shown below:

| Matrix Size | cuBLAS GFLOPs/s | My GFLOPs/s | % of cuBLAS Performance Achieved |
|:-----------:|:----------------------:|:------------------:|:-----------:|
| 256         | 697.015                | 118.654            | 17.02       |
| 512         | 2062.63                | 485.039            | 23.52       |
| 1024        | 2657.68                | 2662.89            | 100.20      |
| 2048        | 4447.76                | 3944.75            | 88.69       |
| 4096        | 4312.22                | 4446.76            | 103.12      |
| 8192        | 4039.70                | 4151.41            | 102.77      |

## Acknowledgement
[Simon Boehm's implementation](https://github.com/siboehm/SGEMM_CUDA) is the primary source of inspiration.
