# Exec/CMakeLists.txt

add_subdirectory(RiemannProblem)
add_subdirectory(CylindricalExplosion)
add_subdirectory(RiemannProblem2D)
add_subdirectory(QuadrantProblem)
add_subdirectory(SinProblem)
add_subdirectory(IsentropicVortex)
add_subdirectory(Advection)
add_subdirectory(GreshoVortex)

FILE(CREATE_LINK
    ${CMAKE_SOURCE_DIR}/Tools/gen_visit_headers.sh
    ${CMAKE_CURRENT_BINARY_DIR}/gen_visit_headers.sh
    SYMBOLIC
)