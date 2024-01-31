# Source/CMakeLists.txt

set (SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/AmrLevelAdv.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/bc_nullfill.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/LevelBldAdv.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/Tagging_params.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/Force.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/SLIC.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/Euler/Euler.cpp
         PARENT_SCOPE)