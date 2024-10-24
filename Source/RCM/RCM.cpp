#include "RCM.H"
#include "AmrLevelAdv.H"
#include "System/Euler/Euler.H"
#include "System/Euler/RiemannSolver.H"
#include <AMReX_FArrayBox.H>

using namespace amrex;

AMREX_GPU_HOST
void advance_RCM(amrex::Real dt, amrex::Real dx, amrex::Real random,
                 amrex::Real adia, amrex::Real eps, const amrex::Box &bx,
                 const amrex::FArrayBox &statein, amrex::FArrayBox &stateout)
{
    AMREX_ASSERT(statein.nComp() == 3);  // only works for 1D!
    AMREX_ASSERT(stateout.nComp() == 3); // only works for 1D!
    AMREX_ASSERT(random >= 0);
    AMREX_ASSERT(random <= 1);
    AMREX_ASSERT(AMREX_SPACEDIM == 1);
#if AMREX_SPACEDIM == 1

    Real S = (random > 0.5) ? (random - 1) * dx / dt : random * dx / dt;

    const auto &in  = statein.const_array();
    const auto &out = stateout.array();

    const int i_off = (random > 0.5) ? 1 : 0;

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const auto &primv_L
                        = primv_from_consv(in, adia, eps, i - 1 + i_off, j, k);
                    const auto &primv_R
                        = primv_from_consv(in, adia, eps, i + i_off, j, k);
                    EulerExactRiemannSolver<0, 1> rp_solver(primv_L, primv_R,
                                                            adia, adia, eps);
                    rp_solver.compute_soln();
                    rp_solver.sample(S, out, i, j, k);
                });

#else
    amrex::ignore_unused(dt, dx, random, adia, eps, bx, statein, stateout);
#endif
}