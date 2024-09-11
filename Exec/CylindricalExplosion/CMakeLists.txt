# Exec/CylindricalExplosion/CMakeLists.txt

set(CYLINDRICAL_EXPLOSION_SRC ${SRC}
                              Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(cyl_exp_ref ${CYLINDRICAL_EXPLOSION_SRC})
add_executable(cyl_exp_2d ${CYLINDRICAL_EXPLOSION_SRC})
add_executable(cyl_exp_3d ${CYLINDRICAL_EXPLOSION_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(cyl_exp_ref)
    setup_target_for_cuda_compilation(cyl_exp_2d)
    setup_target_for_cuda_compilation(cyl_exp_3d)
endif()

target_link_libraries(cyl_exp_ref AMReX::amrex_1d)
target_link_libraries(cyl_exp_2d AMReX::amrex_2d)
target_link_libraries(cyl_exp_3d AMReX::amrex_3d)

# Convenience commands for running tests
add_custom_target(run_ref
    COMMAND
        $<TARGET_FILE:cyl_exp_ref> ${CMAKE_CURRENT_SOURCE_DIR}/inputs-ref/cyl-RCM-5000
    COMMAND
        $<TARGET_FILE:cyl_exp_ref> ${CMAKE_CURRENT_SOURCE_DIR}/inputs-ref/sph-RCM-1000
    COMMENT "Running reference solutions"
    DEPENDS
        cyl_exp_ref
    VERBATIM
)

if(AMReX_MPI STREQUAL "YES")
    add_custom_target(run_geom_src_tests
        COMMAND
            rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output/visc
        COMMAND
            mpirun -np 1 $<TARGET_FILE:cyl_exp_ref> ${CMAKE_CURRENT_SOURCE_DIR}/inputs/visc/MHM-1d
        COMMAND
            mpirun -np 8 $<TARGET_FILE:cyl_exp_2d> ${CMAKE_CURRENT_SOURCE_DIR}/inputs/visc/MHM-2d
        COMMAND
            mpirun -np 8 $<TARGET_FILE:cyl_exp_3d> ${CMAKE_CURRENT_SOURCE_DIR}/inputs/visc/MHM-3d
        DEPENDS
            cyl_exp_ref
            cyl_exp_2d
            cyl_exp_3d
        VERBATIM
    )
endif()