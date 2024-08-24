# Exec/Jet/CMakeLists.txt

set(JET_SRC ${SRC}
    Prob.cpp)

include_directories(.
       ${CMAKE_SOURCE_DIR}/Source 
       ${CMAKE_SOURCE_DIR}/Source/Src_K)

# add_executable(jet ${JET_SRC})
add_executable(jet_2d ${JET_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
# setup_target_for_cuda_compilation(jet)
setup_target_for_cuda_compilation(jet_2d)
endif()

# target_link_libraries(jet AMReX::amrex_3d)
target_link_libraries(jet_2d AMReX::amrex_2d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
    ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)