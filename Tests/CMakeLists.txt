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

add_executable(root_finding RootFinding01.cpp ${SRC_NO_MAIN} test_main.cpp)
target_include_directories(root_finding PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(root_finding GTest::gtest)
target_link_libraries(root_finding AMReX::amrex_1d)
gtest_discover_tests(root_finding)

add_executable(exact_riemann ExactRiemann01.cpp ${SRC_NO_MAIN} test_main.cpp)
target_include_directories(exact_riemann PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(exact_riemann GTest::gtest)
target_link_libraries(exact_riemann AMReX::amrex_1d)
gtest_discover_tests(exact_riemann)

add_executable(rusanov_adv_3d RusanovAdv01.cpp ${SRC_NO_MAIN} test_main.cpp)
target_include_directories(rusanov_adv_3d PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(rusanov_adv_3d GTest::gtest)
target_link_libraries(rusanov_adv_3d AMReX::amrex_3d)
gtest_discover_tests(rusanov_adv_3d)

add_executable(rusanov_adv_1d RusanovAdv01.cpp ${SRC_NO_MAIN} test_main.cpp)
target_include_directories(rusanov_adv_1d PRIVATE ${CMAKE_SOURCE_DIR}/Source)
target_link_libraries(rusanov_adv_1d GTest::gtest)
target_link_libraries(rusanov_adv_1d AMReX::amrex_1d)
gtest_discover_tests(rusanov_adv_1d)