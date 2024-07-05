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

if (AMReX_MPI STREQUAL "YES")
    add_custom_target(run_adv_mpi
        COMMAND
            find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-exp -iname "eps*" -exec mpirun -np 10 $<TARGET_FILE:adv_2d> {} \;
        COMMAND
            find ${CMAKE_CURRENT_SOURCE_DIR}/inputs-IMEX -iname "eps*" -exec $<TARGET_FILE:adv_2d> {} \;
        VERBATIM
        DEPENDS
            adv_2d
    )
endif()

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)