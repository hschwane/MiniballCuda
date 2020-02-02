cmake_minimum_required(VERSION 3.15)
project(miniball)

enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(miniballTiming
        timing_test.cu
        dummy.cpp)

find_package(OpenMP)
target_link_libraries(miniballTiming OpenMP::OpenMP_CXX)

set_target_properties( miniballTiming PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(miniballTiming PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(miniballTiming PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
#target_compile_options(miniballTiming PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
#target_compile_options(miniballTiming PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)

add_executable(miniballExample
        example.cu
        dummy.cpp)

set_target_properties( miniballExample PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(miniballExample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(miniballExample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
