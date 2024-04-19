# Exec/Advection/CMakeLists.txt

set(ADVECTION_SRC ${SRC}
                  Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(adv_2d ${ADVECTION_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(adv_2d)
endif()

target_link_libraries(adv_2d AMReX::amrex_2d)

add_custom_target(run_adv
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "inputs-*" -exec $<TARGET_FILE:adv_2d> {} \;
    COMMAND
        bash -c [[for d in output/*/*/ ; do (cd ${d} ; ls -1 plt?????/Header | tee headers.visit); done]]
    VERBATIM
    COMMENT "Running tests"
    DEPENDS
        adv_2d
)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)