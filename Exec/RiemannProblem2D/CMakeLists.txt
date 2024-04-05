# Exec/RiemannProblem2D/CMakeLists.txt

set(RIEMANN_PROBLEM_2D_SRC ${SRC}
                         Adv_prob.cpp
                         Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(riemann_prob_2d ${RIEMANN_PROBLEM_2D_SRC})

target_link_libraries(riemann_prob_2d AMReX::amrex_2d)

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