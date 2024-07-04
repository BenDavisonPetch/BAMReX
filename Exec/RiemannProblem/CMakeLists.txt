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

add_custom_target(run_lax
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "lax*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running Lax tests"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_lax_order2
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-order2 -iname "lax*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running Lax tests"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_sod
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "test1*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running Sod test"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_sod_order2
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-order2 -iname "test1*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running Sod test, 2nd order methods"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_test2
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "test2*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running test 2"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_test2_order2
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-order2 -iname "test2*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running test 2"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_test3
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "test3*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running test 3"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_test3_order2
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-order2 -iname "test3*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running test 3"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_incshocks
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "incshocks*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running incident shock tests"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_incshocks_order2
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-order2 -iname "incshocks*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running incident shock tests"
    DEPENDS
        riemann_prob
    VERBATIM
)


add_custom_target(run_cfl
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-CFLtest -iname "lax*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running Lax tests"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_cfl_order2
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-CFLtest-order2 -iname "lax*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running Lax tests"
    DEPENDS
        riemann_prob
    VERBATIM
)

add_custom_target(run_contact
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-contact/ -iname "contact*" -exec $<TARGET_FILE:riemann_prob> {} \;
    COMMENT "Running contact tests"
    DEPENDS
        riemann_prob
    VERBATIM
)

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