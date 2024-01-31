# Exec/RiemannProblem/CMakeLists.txt

set(RIEMANN_PROBLEM_SRC ${SRC}
                         Adv_prob.cpp
                         Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(riemann_prob ${RIEMANN_PROBLEM_SRC})

target_link_libraries(riemann_prob AMReX::amrex_2d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs-test1
                 ${CMAKE_BINARY_DIR}/Exec/RiemannProblem/inputs-test1 SYMBOLIC)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs-test2
                 ${CMAKE_BINARY_DIR}/Exec/RiemannProblem/inputs-test2 SYMBOLIC)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs-test3
                 ${CMAKE_BINARY_DIR}/Exec/RiemannProblem/inputs-test3 SYMBOLIC)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs-test4
                 ${CMAKE_BINARY_DIR}/Exec/RiemannProblem/inputs-test4 SYMBOLIC)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs-test5
                 ${CMAKE_BINARY_DIR}/Exec/RiemannProblem/inputs-test5 SYMBOLIC)