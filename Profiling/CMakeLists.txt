# Profiling/CMakeLists.txt

add_executable(multifab_adding
    EXCLUDE_FROM_ALL
    multifab_adding.cpp
)

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(multifab_adding)
endif()

target_link_libraries(multifab_adding AMReX::amrex_3d)