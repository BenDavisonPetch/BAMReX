# Tests/CMakeLists.txt

include_directories(.
                    ${CMAKE_SOURCE_DIR}/Source 
                    ${CMAKE_SOURCE_DIR}/Source/Src_K)

include(FetchContent)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

enable_testing()

add_executable(tests_1d
  test_main.cpp
  ${SRC_NO_MAIN}
  RootFinding01.cpp
  ExactRiemann01.cpp
  RusanovAdv01.cpp
  HLLC01.cpp
  MLABecLapGrad01.cpp
  IMEX_O1_01.cpp
  IMEXButcherTableaux01.cpp
  Limiters01.cpp
  Interp01.cpp
)

add_executable(tests_2d
  test_main.cpp
  ${SRC_NO_MAIN}
  IMEX_O1_01.cpp
  Euler01.cpp
  MLABecLapGrad01.cpp
  GFMFlag01.cpp
  Interp01.cpp
)

add_executable(tests_3d
  test_main.cpp
  ${SRC_NO_MAIN}
  RusanovAdv01.cpp
  IMEX_O1_01.cpp
  Euler01.cpp
  MLABecLapGrad01.cpp
  Update01.cpp
  Interp01.cpp
)

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(tests_1d)
    setup_target_for_cuda_compilation(tests_2d)
    setup_target_for_cuda_compilation(tests_3d)
endif()

target_link_libraries(tests_1d GTest::gtest)
target_link_libraries(tests_1d AMReX::amrex_1d)
gtest_discover_tests(tests_1d)

target_link_libraries(tests_2d GTest::gtest)
target_link_libraries(tests_2d AMReX::amrex_2d)
gtest_discover_tests(tests_2d)

target_link_libraries(tests_3d GTest::gtest)
target_link_libraries(tests_3d AMReX::amrex_3d)
gtest_discover_tests(tests_3d)

## Non-AMReX tests

add_executable(extra_tests
  PseudoRandom01.cpp
)
target_link_libraries(extra_tests GTest::gtest_main)
gtest_discover_tests(extra_tests)