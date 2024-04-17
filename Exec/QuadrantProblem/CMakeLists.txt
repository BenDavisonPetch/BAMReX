# Exec/CylindricalExplosion/CMakeLists.txt

set(RIEMANN_PROBLEM_2D_SRC ${SRC}
                           Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(riemann_prob2d ${RIEMANN_PROBLEM_2D_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(riemann_prob2d)
endif()

target_link_libraries(riemann_prob2d AMReX::amrex_2d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs-test1
                 ${CMAKE_BINARY_DIR}/Exec/RiemannProblem2D/inputs-test1 SYMBOLIC)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs-test2
                 ${CMAKE_BINARY_DIR}/Exec/RiemannProblem2D/inputs-test2 SYMBOLIC)