# Exec/Geometry/CMakeLists.txt

set(GEOM_SRC ${SRC}
    Prob.cpp)

include_directories(.
       ${CMAKE_SOURCE_DIR}/Source 
       ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(geom_3d ${GEOM_SRC})
add_executable(geom_2d ${GEOM_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
setup_target_for_cuda_compilation(geom_3d)
setup_target_for_cuda_compilation(geom_2d)
endif()

target_link_libraries(geom_3d AMReX::amrex_3d)
target_link_libraries(geom_2d AMReX::amrex_2d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
    ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)