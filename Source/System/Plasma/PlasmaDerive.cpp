#include "PlasmaDerive.H"
#include "AmrLevelAdv.H"
#include "System/Parm.H"
#include "System/Plasma/Plasma_K.H"
#include "System/PlasmaIndex.H"

using namespace amrex;

void mhd_derpres(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                 int /*ncomp*/, const amrex::FArrayBox &datafab,
                 const amrex::Geometry & /*geomdata*/, amrex::Real /*time*/,
                 const int * /*bcrec*/, int /*level*/)
{
    AMREX_ASSERT(datafab.nComp() == PUInd::size);
    const Parm *d_parm = AmrLevelAdv::d_parm;
    const auto &p      = derfab.array();
    const auto &U      = datafab.const_array();
    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const auto &Q     = mhd_ctop_noB(i, j, k, U, *d_parm);
                    p(i, j, k, dcomp) = Q[PQInd::P];
                });
}

void mhd_dertemp(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                 int /*ncomp*/, const amrex::FArrayBox &datafab,
                 const amrex::Geometry & /*geomdata*/, amrex::Real /*time*/,
                 const int * /*bcrec*/, int /*level*/)
{
    AMREX_ASSERT(datafab.nComp() == PUInd::size);
    const Parm *d_parm = AmrLevelAdv::d_parm;
    const auto &T      = derfab.array();
    const auto &U      = datafab.const_array();
    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const auto &Q = mhd_ctop_noB(i, j, k, U, *d_parm);
                    T(i, j, k, dcomp)
                        = Q[PQInd::P]
                          / (U(i, j, k, PUInd::RHO) * d_parm->R_spec);
                });
}