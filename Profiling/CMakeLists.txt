# Profiling/CMakeLists.txt

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

#
# MultiFab adding
#

add_executable(multifab_adding
    EXCLUDE_FROM_ALL
    multifab_adding.cpp
)

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(multifab_adding)
endif()

target_link_libraries(multifab_adding AMReX::amrex_3d)

#
# Explicit scaling
#

add_executable(explicit_scaling
    EXCLUDE_FROM_ALL
    explicit_scaling.cpp
    ${SRC_NO_MAIN}
)

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(explicit_scaling)
endif()

target_link_libraries(explicit_scaling AMReX::amrex_2d)