# Exec/RiemannProblem/CMakeLists.txt

set(RIEMANN_PROBLEM_SRC ${SRC}
                         Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(riemann_prob ${RIEMANN_PROBLEM_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(riemann_prob)
endif()

target_link_libraries(riemann_prob AMReX::amrex_1d)

add_custom_target(run_rp
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "inputs-*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running Toro tests"
    DEPENDS
        riemann_prob
    VERBATIM
)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)