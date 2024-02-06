# Source/CMakeLists.txt

set (SRC_NO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/AmrLevelAdv.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/bc_nullfill.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/LevelBldAdv.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Tagging_params.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/Force.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/SLIC.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/Flux_Helpers.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/MUSCL_Hancock.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/HLLC.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Euler/Euler.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Euler/RiemannSolver.cpp
                 PARENT_SCOPE
)

set (SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
         ${SRC_NO_MAIN}
         PARENT_SCOPE)