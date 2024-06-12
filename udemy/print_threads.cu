#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void print_threadIds(){
    printf(
        "threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d, \n ",
        threadIdx.x, threadIdx.y, threadIdx.z
    );
}


int main(){
    dim3 grid(2,2);
    dim3 block(8,8);


    print_threadIds<<<grid, block>>>();
    cudaDeviceSynchronize();


    return 0;
}
