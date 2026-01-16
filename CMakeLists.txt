cmake_minimum_required(VERSION 3.18)
project(CUDA_SGEMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# specify CUDA architectures (change 'native' to a specific SM version if needed, e.g. '80')
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Enable fast math for performance and optimization level O3
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>")

add_library(gpu_config INTERFACE)
add_library(utils OBJECT src/utils.cu)
target_link_libraries(gpu_config INTERFACE cudart)
target_link_libraries(gpu_config INTERFACE utils)

set(CMAKE_CUDA_SEPERABLE_COMPILATION ON)


add_library(common_main OBJECT src/main.cu)
target_link_libraries(common_main PRIVATE gpu_config)



set(KERNELS "naive" "tiled" "register_tiled" "vectorized" "double_buffered")

foreach(kernel IN LISTS KERNELS)
  set(target_name "sgemm_${kernel}")
  add_executable(${target_name}
    $<TARGET_OBJECTS:common_main> 
    "src/kernels/${target_name}.cu"
  )
  target_link_libraries(${target_name} PRIVATE gpu_config)
endforeach()

add_executable(sgemm_cublas
  "src/sgemm_cublas.cu"
)
target_link_libraries(sgemm_cublas PRIVATE gpu_config)
target_link_libraries(sgemm_cublas PRIVATE cublas)
