# Source/CMakeLists.txt

set (SRC_NO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/AmrLevelAdv.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Adv_prob.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/bc_nullfill.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/LevelBldAdv.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Tagging_params.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/MFIterLoop.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/Force.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/SLIC.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/Flux_Helpers.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/MUSCL_Hancock.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/HLLC.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/IMEX_O1.cpp
                #  ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/IMEX_O1_bp.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/IMEX_Fluxes.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/IMEX_RK.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/ButcherTableau.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Euler/Euler.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Euler/RiemannSolver.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/LinearSolver/MLALaplacianGrad.cpp
)

set (SRC_NO_MAIN ${SRC_NO_MAIN} PARENT_SCOPE)

set (SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
         ${SRC_NO_MAIN}
         PARENT_SCOPE)