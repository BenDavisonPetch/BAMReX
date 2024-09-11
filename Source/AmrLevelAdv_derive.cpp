#include "AmrLevelAdv_derive.H"
#include <AMReX_ParallelDescriptor.H>

using namespace amrex;

void derive_rank(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                 int /*ncomp*/, const amrex::FArrayBox & /*datafab*/,
                 const amrex::Geometry & /*geomdata*/, amrex::Real /*time*/,
                 const int * /*bcrec*/, int /*level*/)
{
    const int   rank = ParallelDescriptor::MyProc();
    const auto &der  = derfab.array();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { der(i, j, k, dcomp) = rank; });
}