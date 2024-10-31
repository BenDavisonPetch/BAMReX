#include "EulerDerive.H"
#include "AmrLevelAdv.H"
#include "IMEX_K/euler_K.H"
#include "System/Index.H"
#include "System/Parm.H"

using namespace amrex;

void euler_derpres(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                   int /*ncomp*/, const amrex::FArrayBox &datafab,
                   const amrex::Geometry & /*geomdata*/, amrex::Real /*time*/,
                   const int * /*bcrec*/, int /*level*/)
{
    AMREX_ASSERT(datafab.nComp() == UInd::size);
    const Parm *d_parm = AmrLevelAdv::d_parm;
    const auto &p      = derfab.array();
    const auto &U      = datafab.const_array();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { p(i, j, k, dcomp) = pressure(i, j, k, U, *d_parm); });
}

void euler_dervel(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                  int /*ncomp*/, const amrex::FArrayBox &datafab,
                  const amrex::Geometry & /*geomdata*/, amrex::Real /*time*/,
                  const int * /*bcrec*/, int /*level*/)
{
    AMREX_ASSERT(datafab.nComp() == 2);
    const auto &u   = derfab.array();
    const auto &src = datafab.const_array();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { u(i, j, k, dcomp) = src(i, j, k, 1) / src(i, j, k, 0); });
}

void euler_dertemp(const amrex::Box &bx, amrex::FArrayBox &derfab, int dcomp,
                   int /*ncomp*/, const amrex::FArrayBox &datafab,
                   const amrex::Geometry & /*geomdata*/, amrex::Real /*time*/,
                   const int * /*bcrec*/, int /*level*/)
{
    AMREX_ASSERT(datafab.nComp() == UInd::size);
    const Parm *d_parm = AmrLevelAdv::d_parm;
    const auto &T      = derfab.array();
    const auto &U      = datafab.const_array();
    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    T(i, j, k, dcomp)
                        = pressure(i, j, k, U, *d_parm)
                          / (U(i, j, k, UInd::RHO) * d_parm->R_spec);
                });
}