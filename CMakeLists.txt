cmake_minimum_required(VERSION 3.16)

project(amrex_cfd)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_CUSTOM "-O2 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native")

file(CREATE_LINK ${CMAKE_BINARY_DIR}/compile_commands.json
                 ${CMAKE_SOURCE_DIR}/compile_commands.json
                 SYMBOLIC)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(NOT DEFINED AMReX_ROOT)
    message(
        WARNING
            "No AMReX installation specified! Setting defaults. If these defaults \
            do not correspond to AMReX installations on your system, make sure to use \
            the flag -DAMReX_ROOT=/path/to/amrex when calling cmake"
    )
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        set(AMReX_ROOT /home/bendp/bin/amrex-mpi-omp)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(AMReX_ROOT /home/bendp/bin/amrex-basic-debug)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Custom")
        set(AMReX_ROOT /home/bendp/bin/amrex-basic-debug)
    else()
        set(AMReX_ROOT /home/bendp/bin/amrex-omp-debug)
    endif()
endif()

# Finding packages
find_package(AMReX PATHS ${AMReX_ROOT}/lib/cmake/AMReX
                   REQUIRED 1D 2D 3D NO_DEFAULT_PATH)

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    message(STATUS "Detected CUDA support in provided AMReX build")
    enable_language(CUDA)
    # AMReX 21.06+ supports CUDA_ARCHITECTURES
    if(CMAKE_VERSION VERSION_LESS 3.20)
        include(AMReX_SetupCUDA)
    else()
        include(AMReXTargetHelpers)
    endif()

    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(AMReX_MPI STREQUAL "YES")
    message(STATUS "Detected MPI support in provided AMReX build")
    find_package(MPI REQUIRED)
    target_link_libraries(AMReX::amrex_1d INTERFACE ${MPI_LIBRARIES})
    target_link_libraries(AMReX::amrex_2d INTERFACE ${MPI_LIBRARIES})
    target_link_libraries(AMReX::amrex_3d INTERFACE ${MPI_LIBRARIES})
endif()

set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(Source)
add_subdirectory(Exec)
add_subdirectory(Tests)
add_subdirectory(Profiling)