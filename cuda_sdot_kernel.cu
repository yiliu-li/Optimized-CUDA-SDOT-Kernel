#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"

using namespace std;
using data_type = float;


__global__ void sdot2_4(data_type *a, data_type *b, data_type *c, int n){

    // Define variables.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    data_type temp;
    temp = 0;
    // Define shared memories.
    __shared__ data_type s_data[1024];
    unsigned int tid = threadIdx.x;
    // Multiplication of data in the index.
    for (int i = index; i < n; i += stride){
        temp += ( a[i] * b[i] );
    }
    // Assign value to shared memory.
    s_data[tid] = temp;
    __syncthreads();
    // Add up products.
    for (int s = blockDim.x / 4; s > 0 ; s >>= 2){
        if ((tid < s)) {
            temp = s_data[tid];
            temp += s_data[tid + s];
            temp += s_data[tid + (s << 1)];
            temp += s_data[tid + (3 * s)];
            s_data[tid] = temp;
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd( c , s_data[0] );
    }
}


__global__ void sdot2_2(data_type *a, data_type *b, data_type *c, int n){

    // Define variables.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    data_type temp;
    temp = 0;
    // Define shared memories.
    __shared__ data_type s_data[1024];
    unsigned int tid = threadIdx.x;
    // Multiplication of data in the index.
    for (int i = index; i < n; i += stride){
        temp += ( a[i] * b[i] );
    }
    // Assign value to shared memory.
    s_data[tid] = temp;
    __syncthreads();
    // Add up products.
    for (int s = blockDim.x / 4; s > 0 ; s >>= 2){
        if ((tid < s)) {
            temp = s_data[tid];
            temp += s_data[tid + s];
            temp += s_data[tid + (s << 1)];
            temp += s_data[tid + (3 * s)];
            s_data[tid] = temp;
        }
        __syncthreads();
    }
    s_data[0] += s_data[1];
    if(tid == 0){
        atomicAdd( c , s_data[0] );
    }
}


__global__ void sdot1(data_type *a, data_type *b, data_type *c, int n){
    // Define variables.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    data_type temp = 0.0;
    __shared__ data_type s_data[1024];
    // __shared__ data_type s_data[(int)gridDim.x];
    unsigned int tid = threadIdx.x;
    // Multiplication of data in an index.
    for (int i = index; i < n; i += stride){
        temp += ( a[i] * b[i] );
    }
    s_data[tid] = temp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0 ; s >>= 1){
        if ((tid < s) && (index + s < n))
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd( c , s_data[0] );
    }
}


int main(int argc, char *argv[]){

    // Define random seed
    srand(0);

    // Parameter assignment.
    int N, nBytes;
    int kernel_num, block_size;
    data_type *A, *B, *C;

    int start_n = 100;
    int step_n = 200;
    int repeat_n = 20;
    int end_n = start_n + (repeat_n * step_n);

    // Set parameter
    printf("Kernel Number (0 for CuBLAS): ");
    scanf("%d", &kernel_num);
    printf("Block Size: ");
    scanf("%d", &block_size);
    block_size = 256;

    for (int repeat_t = start_n; repeat_t <= end_n; repeat_t += step_n){

        // Initialize cuda event
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        // Memory size assignment
        N = repeat_t * repeat_t;
        nBytes = N * sizeof(data_type);

        // Allocate memory of the host to store data.
        A = (data_type*)malloc(nBytes);
        B = (data_type*)malloc(nBytes);
        C = (data_type*)malloc(sizeof(data_type));

        // Assign data to the variable.
        for (int i = 0; i < N; ++i)
        {
            A[i] = (float)((rand()%10000)/100);
            B[i] = (float)((rand()%10000)/100);
        }
        *C = 0;
        printf("------------------------\n");
        printf("Length of Vector = %d\n", N);  

        // Allocate memory of the device to store data.
        data_type *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, nBytes);
        cudaMalloc((void**)&d_B, nBytes);
        cudaMalloc((void**)&d_C, sizeof(data_type));

        // Copy data from host to device.
        cudaMemcpy((void*)d_A, (void*)A, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_B, (void*)B, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_C, (void*)C, sizeof(data_type), cudaMemcpyHostToDevice);

        // Define the configuration.
        dim3 blockSize(block_size);
        dim3 gridSize(min(1024,(N + blockSize.x - 1) / blockSize.x));



        if (kernel_num == 2){
            float isInt = log((float)block_size)/log(4);
            printf("%f", isInt);
            if (isInt == (float)(int)isInt){
                //Start timer.
                cudaEventRecord(start);

                // Run the kernel.
                sdot2_4 <<< gridSize, blockSize >>>(d_A, d_B, d_C, N);

                // End timer, Calculate performance.
                cudaEventRecord(end);
            }
            else{
                //Start timer.
                cudaEventRecord(start);

                // Run the kernel.
                sdot2_2 <<< gridSize, blockSize >>>(d_A, d_B, d_C, N);

                // End timer, Calculate performance.
                cudaEventRecord(end);
            }
        }
        else if(kernel_num == 1){
            //Start timer.
            cudaEventRecord(start);

            // Run the kernel.
            sdot1 <<< gridSize, blockSize >>>(d_A, d_B, d_C, N);

            // End timer, Calculate performance.
            cudaEventRecord(end);
        }
        else if(kernel_num == 0){
            cublasHandle_t handle;
            cublasCreate(&handle); 
            //Start timer.
            cudaEventRecord(start);
            // Run the kernel.    
            cublasSdot(handle, N, d_A, 1, d_B, 1, d_C);
            // End timer, Calculate performance.
            cudaEventRecord(end);
        }




        cudaEventSynchronize(start);
        cudaEventSynchronize(end);
        float elapsedtime = 0.0;
        double flops;
        cudaEventElapsedTime(&elapsedtime, start, end);
        flops = (double)((2 * N) - 1) / (double)elapsedtime;
        flops /= 1000000.0;

        // Copy the result from device to host.
        cudaMemcpy((void*)C, (void*)d_C, sizeof(float), cudaMemcpyDeviceToHost);


        printf("Time spent: %f ms\n", elapsedtime);
        printf("Performance: %f GFLOPS", flops);
        printf("\n========================\n\n");



        // Release memory on device.
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);


        // Release memory on host.
        free(A);
        free(B);
        free(C);
        


}










}