# Exec/GreshoVortex/CMakeLists.txt

set(GRESHO_VORTEX_SRC ${SRC}
                      Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(gresho_2d ${GRESHO_VORTEX_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(gresho_2d)
    setup_target_for_cuda_compilation(isen_3d)
endif()

target_link_libraries(gresho_2d AMReX::amrex_2d)

add_custom_target(run_gresho
    COMMAND
        rm -rf ${CMAKE_CURRENT_BINARY_DIR}/output
    COMMAND
        find ${CMAKE_CURRENT_SOURCE_DIR}/inputs -iname "M*" -exec $<TARGET_FILE:gresho_2d> {} \;
    COMMAND
        bash -c [[for d in output/*/*/*/ ; do (cd ${d} ; ls -1 plt?????/Header | tee headers.visit); done]]
    VERBATIM
    COMMENT "Running tests"
    DEPENDS
        gresho_2d
)

add_custom_target(generate_visit_headers
    COMMAND
        bash -c [[for d in output/*/*/*/ ; do (cd ${d} ; ls -1 plt?????/Header | tee headers.visit); done]]
    VERBATIM
)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)