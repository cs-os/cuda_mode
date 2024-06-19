#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>


__global__ void mem_trs_test(int * input){
    int gid = blockIdx.x * blockDim.x + threadIdx.x ;
    printf("tid : %d, gid : %d, value : %d \n", threadIdx.x, gid, input[gid]);
}

int main(){
    int size = 128;
    int byte_size = size * sizeof(int);

    int * h_input;
    h_input = (int*)malloc(byte_size);  // malloc returns void pointer so casting it to integer pointer


    time_t t;

    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++){
        h_input[i] = (int)(rand() & 0xff0);
    }

    int * d_input;

    cudaMalloc((void**)&d_input, byte_size);

    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid(2);

    mem_trs_test <<<grid, block>>> (d_input);
    cudaDeviceSynchronize();
    cudaFree(d_input);
    free(h_input);

    cudaDeviceReset();
    return 0;
}