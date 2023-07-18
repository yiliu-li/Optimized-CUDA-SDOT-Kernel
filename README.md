# Optimized-Cuda-SDOT-Kernel-on-NVIDIA-Turing-GPUs
An optimized CUDA SDOT(Single Floating-Point DOT Product) kernel on NVIDIA Turing GPUs. Better performance than the cuBLAS kernel.
## Description

This code demonstrates a usage of cuBLAS `sdot` function to apply the dot product to vector x and y

Optimization of operations like SDOT in High-Performance Computing (HPC) mainly revolves around reducing data movement. This is achieved on GPU platforms by leveraging parallelism, reusing data at the cache/register level, and utilizing manual prefetching. 

By testing, this GPU-optimized kernel boosts the efficiency of the SDOT operation by minimizing data movement, which runs faster than the cuBLAS SDOT kernel.

Any discussions are welcome, please send them to yiliuli2006@gmail.com




# How to run
- Clone the code.
- Build command in terminal.
    ```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
- Run the executable file.
```bash
  ./cublas_dot_example
```
- Enter kernel number.
- Enter block-size ( 64 - 256 recommended ).

------------

## Kernel List
### Kernel 1
This kernel operate traditionally, 


### Kernel 2
### Kernel 3
### Kernel 4






