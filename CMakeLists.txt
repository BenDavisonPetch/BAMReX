cmake_minimum_required(VERSION 3.16)

project(amrex_cfd)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

file(CREATE_LINK ${CMAKE_BINARY_DIR}/compile_commands.json
                 ${CMAKE_SOURCE_DIR}/compile_commands.json
                 SYMBOLIC)

# Finding packages
find_package(AMReX PATHS ${AMReX_ROOT}/lib/cmake/AMReX/AMReXConfig.cmake
                   REQUIRED 1D 2D)
find_package(MPI REQUIRED)

target_link_libraries(AMReX::amrex_1d INTERFACE ${MPI_LIBRARIES})
target_link_libraries(AMReX::amrex_2d INTERFACE ${MPI_LIBRARIES})
target_link_libraries(AMReX::amrex_3d INTERFACE ${MPI_LIBRARIES})

add_subdirectory(Source)
add_subdirectory(Exec)