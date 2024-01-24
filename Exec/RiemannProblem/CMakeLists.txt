# Exec/RiemannProblem/CMakeLists.txt

set(RIEMANN_PROBLEM_SRC ${SRC}
                         Adv_prob.cpp
                         Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(riemann_prob ${RIEMANN_PROBLEM_SRC})

target_link_libraries(riemann_prob AMReX::amrex_1d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_BINARY_DIR}/Exec/RiemannProblem/inputs SYMBOLIC)