cmake_minimum_required(VERSION 3.20)
project(nebula_hesai_cuda)

find_package(ament_cmake REQUIRED)
find_package(nebula_hesai_decoders REQUIRED)

# Without CUDA nothing is built, and nebula_hesai decodes on the CPU only
include(CheckLanguage)
check_language(CUDA)
find_package(CUDAToolkit QUIET)
if(NOT CMAKE_CUDA_COMPILER OR NOT CUDAToolkit_FOUND)
  message(WARNING "CUDA not found, nebula_hesai_cuda is not built")
  ament_package()
  return()
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 87 89)
endif()
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ISO C++ (no GNU extensions) also keeps GCC from fusing multiply-adds, as in the other nebula
# packages (autoware_package()); the decoder's tables are computed with the CPU decoder's code.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# autoware_package() would also pass these to nvcc, which does not accept them
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic> $<$<COMPILE_LANGUAGE:CXX>:-Werror>)

add_library(nebula_hesai_cuda SHARED src/hesai_cuda_decoder.cpp src/cuda_scan_builder.cu)

# The point coordinates must come out of the same float operations as on the CPU
set_source_files_properties(src/cuda_scan_builder.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")

target_include_directories(
  nebula_hesai_cuda
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

target_link_libraries(
  nebula_hesai_cuda PUBLIC nebula_hesai_decoders::nebula_hesai_decoders
                           CUDA::cudart)

install(TARGETS nebula_hesai_cuda EXPORT export_nebula_hesai_cuda)
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include/${PROJECT_NAME})

ament_export_targets(export_nebula_hesai_cuda)
ament_export_dependencies(nebula_hesai_decoders CUDAToolkit)

ament_package()
