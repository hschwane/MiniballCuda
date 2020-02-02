# Miniball Cuda
CUDA port of the Miniball code. The original Miniball code was written by Bernd Gaertner and can be downloaded on the [ETHZ Website](https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html).
The algorithm is described in his [Paper](http://www.inf.ethz.ch/personal/gaertner/texts/own_work/esa99_final.pdf).
 
Note that I did not parallelize the Miniball algorithm itself. Instead this implementation provides Miniball as a device function.
This way many balls can be computed at the same time. Depending on your system you will only see an improvement when the number of independent
spheres gets large (>1000). There may also be limitations in terms of flexibility and memory usage with this modified version (see usage).

There is a usage example in `example.cu` together with a minimal CMakeList file to build it. 
There is also a performance comparison available between CPU and GPU in `timing_test.cu`.
Note however that the modified code is slightly slower when run on the CPU then the original implementation.

## building
- include MiniballCuda.h in your code
- enable the cuda compiler option "--expt-relaxed-constexpr"
- compile as usual

## usage
Usage is similar to the original Miniball implementation.
However you can call it from host and device code.
This way every cuda thread can calculate one sphere.
That requires the dimension and maximum number of points per sphere to be known at compile time.
They are passed to the algorithm as template parameters.

For more information see the example code in `example.cu`.
 
There is also an option (template parameter) to switch back to the recursive implementation. 
That can be faster but leads to a stack overflow when using many points per sphere (GPU's have limited stack).

## changes from original Miniball code
The following changes were made to the original miniball code to allow it to run with cuda. 
- make dimension a compile time constant,
- that allows to remove dynamic memory allocation
- make max amount of points per sphere a compile time constant
- add list class that works without dynamic memory
- add iterative implementation for mtf_mp iterative
- remove internal timing code

## note on performance
The algorithm is not very well suited for the GPU and my implementation far from optimal.
On my GPU it can only archive about 18% compute utilization (measured with nsight compute).
If you fork this repository and put some time in optimizing the code I would be very interested in the results.