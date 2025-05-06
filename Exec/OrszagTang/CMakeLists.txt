# Exec/OrszagTang/CMakeLists.txt

set(ORSZAGTANG_SRC ${SRC}
    Prob.cpp)

include_directories(.
       ${CMAKE_SOURCE_DIR}/Source 
       ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(orszagtang ${ORSZAGTANG_SRC})

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
setup_target_for_cuda_compilation(orszagtang)
endif()

target_link_libraries(orszagtang AMReX::amrex_2d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
    ${CMAKE_CURRENT_BINARY_DIR}/inputs SYMBOLIC)

