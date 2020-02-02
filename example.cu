/*
 * MiniballCuda
 * main.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2020 Hendrik Schwanekamp
 *
 */

#include <random>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>

#include "MiniballCuda.hpp"

// ----------------
// settings
constexpr int d = 3; // dimensions need to be constexpr
constexpr int n = 6; // n of a single ball can be lower, but never higher than this
int numOfSpheres = 100000; // number of spheres to build
constexpr int iterations = 10; // number of iterations to average timings
// ----------------

__global__ void miniballExampleKernel(int numSpheres, const float* input, float* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= numSpheres)
        return;

    // load data from global memory into local buffer
    int startId = d*n*index; // every thread processes one sphere
    float data[n][d];
    for (int i=0; i<n; ++i)
        for (int j=0; j<d; ++j)
            data[i][j] = input[startId + i*d + j];

    // define the types of iterators through the points and their coordinates
    // ----------------------------------------------------------------------
    typedef const float (*PointIterator)[d];
    typedef const float* CoordIterator;

    // create an instance of MiniballCuda
    // ------------------------------
    typedef MiniballCuda::Miniball <MiniballCuda::CoordAccessor<PointIterator, CoordIterator>,d,n,false> MB;
    MB mb( &data[0], &data[n]);

    // store center and radii in output buffer
    auto center = mb.center();
    for(int i=0; i<d; ++i, ++center)
        output[index*(d+1) +i] = *center;
    output[index*(d+1) +d] = mb.squared_radius();
}

void checkCudaError(cudaError_t code)
{
    if(code != cudaSuccess)
        throw std::runtime_error("Cuda error:" + std::string(cudaGetErrorString(code)));
}

int main()
{
    // get some cuda memory
    float* input;
    float* output;

    checkCudaError( cudaMallocManaged(&input, d*n*numOfSpheres* sizeof(float)) );
    checkCudaError( cudaMallocManaged(&output, (d+1)*numOfSpheres* sizeof(float)) );

    // generate some random input
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_real_distribution<float> dist(0,1);

    for(int i =0; i<d*n*numOfSpheres; i++)
        input[i] = dist(rng);

    // ------------------------------------------------------------
    // now call the cuda kernel
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(input, d*n*numOfSpheres*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(output, (d+1)*numOfSpheres*sizeof(float), device, NULL);

    int blockSize = 256; // experiment with different block sizes depending on your GPU
    int numBlocks = (numOfSpheres + blockSize - 1) / blockSize;

    auto startTime = std::chrono::steady_clock::now();
    for(int j =0; j<iterations; j++)
    {
        miniballExampleKernel<<<numBlocks,blockSize>>>(numOfSpheres,input,output);
        checkCudaError(cudaGetLastError());
        checkCudaError(cudaDeviceSynchronize());
    }
    std::cout << "Duration GPU: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - startTime).count()*1000.0/iterations
              << "ms" << std::endl;

    // free memory
    cudaFree(input);
    cudaFree(output);
}