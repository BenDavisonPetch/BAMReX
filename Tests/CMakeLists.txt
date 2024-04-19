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
)

add_executable(tests_2d
  test_main.cpp
  ${SRC_NO_MAIN}
  IMEX_O1_01.cpp
  Euler01.cpp
)

add_executable(tests_3d
  test_main.cpp
  ${SRC_NO_MAIN}
  RusanovAdv01.cpp
  PressureSourceVec01.cpp
  IMEX_O1_01.cpp
  Euler01.cpp
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

# add_executable(root_finding RootFinding01.cpp test_main.cpp)
# target_link_libraries(root_finding GTest::gtest)
# target_link_libraries(root_finding AMReX::amrex_1d)
# gtest_discover_tests(root_finding)

# add_executable(exact_riemann ExactRiemann01.cpp test_main.cpp)
# target_link_libraries(exact_riemann GTest::gtest)
# target_link_libraries(exact_riemann AMReX::amrex_1d)
# gtest_discover_tests(exact_riemann)

# add_executable(rusanov_adv_3d RusanovAdv01.cpp test_main.cpp)
# target_link_libraries(rusanov_adv_3d GTest::gtest)
# target_link_libraries(rusanov_adv_3d testing_base_lib_3d)
# gtest_discover_tests(rusanov_adv_3d)

# add_executable(rusanov_adv_1d RusanovAdv01.cpp test_main.cpp)
# target_link_libraries(rusanov_adv_1d GTest::gtest)
# target_link_libraries(rusanov_adv_1d testing_base_lib_1d)
# gtest_discover_tests(rusanov_adv_1d)

# add_executable(pressure_source_3d PressureSourceVec01.cpp test_main.cpp)
# target_link_libraries(pressure_source_3d GTest::gtest)
# target_link_libraries(pressure_source_3d testing_base_lib_3d)
# gtest_discover_tests(pressure_source_3d)

# add_executable(imex_3d IMEX_O1_01.cpp test_main.cpp)
# target_link_libraries(imex_3d GTest::gtest)
# target_link_libraries(imex_3d testing_base_lib_3d)
# gtest_discover_tests(imex_3d)

# add_executable(imex_2d IMEX_O1_01.cpp test_main.cpp)
# target_link_libraries(imex_2d GTest::gtest)
# target_link_libraries(imex_2d testing_base_lib_2d)
# gtest_discover_tests(imex_2d)