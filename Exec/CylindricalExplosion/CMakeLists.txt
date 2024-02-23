# Exec/CylindricalExplosion/CMakeLists.txt

set(CYLINDRICAL_EXPLOSION_SRC ${SRC}
                              Adv_prob.cpp
                              Prob.cpp)

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

add_executable(cyl_exp ${CYLINDRICAL_EXPLOSION_SRC})

target_link_libraries(cyl_exp AMReX::amrex_3d)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                 ${CMAKE_BINARY_DIR}/Exec/CylindricalExplosion/inputs SYMBOLIC)