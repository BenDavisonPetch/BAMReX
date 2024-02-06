#include <AMReX.H>

#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    testing::InitGoogleTest(&argc,argv);
    int outcome = RUN_ALL_TESTS();
    amrex::Finalize();
    return outcome;
}