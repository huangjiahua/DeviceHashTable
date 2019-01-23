# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.10)


project (DeviceHashTable CUDA CXX)

# C4819 will happen on windows due to language encoding issue, it won't affect anything.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd 4819\" " )


# To support device version of cudaMalloc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61" )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo" )




# Include sub-projects.
add_subdirectory (src)

add_subdirectory (test)


