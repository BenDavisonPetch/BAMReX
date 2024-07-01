# Exec/SinProblem/CMakeLists.txt

set(SIN_PROBLEM_SRC ${SRC}
                    Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(sin_prob ${SIN_PROBLEM_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(sin_prob)
endif()

target_link_libraries(sin_prob AMReX::amrex_1d)

add_custom_target(run_sin
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs/ -iname "eps*" -exec $<TARGET_FILE:sin_prob> {} \;
    COMMENT "Running tests"
    DEPENDS
        sin_prob
    VERBATIM
)

add_custom_target(run_sin_order2
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-order2/ -iname "eps*" -exec $<TARGET_FILE:sin_prob> {} \;
    COMMENT "Running second-order tests"
    DEPENDS
        sin_prob
    VERBATIM
)

add_custom_target(run_sin_order2_EEEK
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-order2-EEEK/ -iname "eps*" -exec $<TARGET_FILE:sin_prob> {} \;
    COMMENT "Running second-order EEEK tests"
    DEPENDS
        sin_prob
    VERBATIM
)

add_custom_target(run_time_order
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/time_order_test/inputs/ -iname "dt*" -exec $<TARGET_FILE:sin_prob> {} \;
    COMMENT "Running time order tests"
    DEPENDS
        sin_prob
    VERBATIM
)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)