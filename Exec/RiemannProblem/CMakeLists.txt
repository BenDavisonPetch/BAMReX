# Exec/RiemannProblem/CMakeLists.txt

set(UNIFORM_VELOCITY_SRC ${SRC}
                         Adv_prob.cpp
                         Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(uniform_velocity ${UNIFORM_VELOCITY_SRC})

target_link_libraries(uniform_velocity AMReX::amrex_2d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_BINARY_DIR}/Exec/UniformVelocity/inputs SYMBOLIC)