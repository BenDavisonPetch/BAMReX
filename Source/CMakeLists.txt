# Source/CMakeLists.txt

set (SRC_NO_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/AmrLevelAdv.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/AmrLevelAdv_advance.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/AmrLevelAdv_LSinit.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/AmrLevelAdv_derive.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Adv_prob.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/LevelBldAdv.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Tagging_params.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/MFIterLoop.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/System/Euler/Euler.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/System/Euler/EulerDerive.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/System/System.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/System/FluidSystem.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/System/PlasmaSystem.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/System/Plasma/PlasmaDerive.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/Force.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/SLIC.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/Flux_Helpers.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/MUSCL_Hancock.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Fluxes/HLLC.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/IMEX_O1.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/IMEX_Fluxes.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/IMEX_RK.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/IMEXSettings.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/IMEX/ButcherTableau.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/LinearSolver/MLALaplacianGrad.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/SourceTerms/GeometricSource.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/RCM/RCM.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/GFM/GFMFlag.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/GFM/RigidBody.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/GFM/Extrapolate.cc
                 ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/Normals.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/PolygonSDF.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Geometry/MakeSDF.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditions/BCs.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditions/bc_nullfill.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditions/bc_jet.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryConditions/bc_ss_jet.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Visc/Visc.cpp
)

set (SRC_NO_MAIN ${SRC_NO_MAIN} PARENT_SCOPE)

set (SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
         ${SRC_NO_MAIN}
         PARENT_SCOPE)