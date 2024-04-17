# Exec/CylindricalExplosion/CMakeLists.txt

set(CYLINDRICAL_EXPLOSION_SRC ${SRC}
                              Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(cyl_exp ${CYLINDRICAL_EXPLOSION_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(cyl_exp)
endif()

target_link_libraries(cyl_exp AMReX::amrex_3d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_BINARY_DIR}/Exec/CylindricalExplosion/inputs SYMBOLIC)