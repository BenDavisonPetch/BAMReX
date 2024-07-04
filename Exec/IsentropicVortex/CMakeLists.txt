# Exec/IsentropicVortex/CMakeLists.txt

set(ISENTROPIC_VORTEX_SRC ${SRC}
                          Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(isen_2d ${ISENTROPIC_VORTEX_SRC})
add_executable(isen_3d ${ISENTROPIC_VORTEX_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(isen_2d)
    setup_target_for_cuda_compilation(isen_3d)
endif()

target_link_libraries(isen_2d AMReX::amrex_2d)
target_link_libraries(isen_3d AMReX::amrex_3d)

add_custom_target(run_isen_stat2d
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs_stat -iname "stat2d*" -exec $<TARGET_FILE:isen_2d> {} \;
    COMMAND
        bash -c [[for d in output/*/*/*/ ; do (cd ${d} ; ls -1 plt?????/Header | tee headers.visit); done]]
    VERBATIM
    COMMENT "Running tests"
    DEPENDS
        isen_2d
)

add_custom_target(run_isen_stat2d_mpi
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs_stat -iname "stat2d*" -exec mpirun -np 10 $<TARGET_FILE:isen_2d> {} \;
        COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs_stat_big -iname "stat2d*" -exec mpirun -np 10 $<TARGET_FILE:isen_2d> {} \;
    COMMAND
        bash -c [[for d in output/*/*/*/ ; do (cd ${d} ; ls -1 plt?????/Header | tee headers.visit); done]]
    VERBATIM
    COMMENT "Running tests"
    DEPENDS
        isen_2d
)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)