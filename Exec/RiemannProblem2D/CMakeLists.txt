# Exec/RiemannProblem2D/CMakeLists.txt

set(RIEMANN_PROBLEM_2D_SRC ${SRC}
                         Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(riemann_prob_2d ${RIEMANN_PROBLEM_2D_SRC})
add_executable(riemann_prob_3d ${RIEMANN_PROBLEM_2D_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(riemann_prob_2d)
    setup_target_for_cuda_compilation(riemann_prob_3d)
endif()

target_link_libraries(riemann_prob_2d AMReX::amrex_2d)
target_link_libraries(riemann_prob_3d AMReX::amrex_3d)

add_custom_target(run_rp2d
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "inputs-*" -exec $<TARGET_FILE:riemann_prob_2d> {} \;
    COMMENT "Running Toro tests"
    DEPENDS
        riemann_prob_2d
    VERBATIM
)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)