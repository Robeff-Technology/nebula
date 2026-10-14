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

execute_process(
  COMMAND ${CMAKE_CUDA_COMPILER} --version
  OUTPUT_VARIABLE nvcc_version_output
  ERROR_QUIET)
string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _ "${nvcc_version_output}")
set(nvcc_version "${CMAKE_MATCH_1}")

# The architectures the installed nvcc knows (CUDA 11.5 from Ubuntu 22.04 stops at 87, 89 needs
# 11.8). The last one is built as SASS and PTX, so newer GPUs (e.g. an RTX 50xx, sm_120) get the
# kernel JIT-compiled by the driver; the others as SASS only. --fmad=false is applied when the PTX
# is generated, so the JIT keeps the CPU decoder's float operations.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  if(nvcc_version VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 87-real 89-real 120)
  elseif(nvcc_version VERSION_GREATER_EQUAL 11.8)
    set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 87-real 89)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 87)
  endif()
  message(STATUS "nvcc ${nvcc_version}: CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# nvcc before 11.6 cannot parse GCC 11's <functional> (pulled in by cub): "parameter packs not
# expanded with '...'" in std_function.h. Ubuntu 22.04 ships nvcc 11.5 with GCC 11 as the default,
# and GCC 10 next to it.
if(NOT CMAKE_CUDA_HOST_COMPILER AND nvcc_version VERSION_LESS 11.6)
  find_program(gxx_10 g++-10)
  if(gxx_10)
    set(CMAKE_CUDA_HOST_COMPILER ${gxx_10})
    message(STATUS "nvcc ${nvcc_version}: CUDA host compiler ${gxx_10}")
  endif()
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
