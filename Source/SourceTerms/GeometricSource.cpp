#include "GeometricSource.H"
#include "AmrLevelAdv.H"
#include <AMReX_Extension.H>

AMREX_GPU_HOST
void advance_geometric(amrex::Real dt, amrex::Real dr, amrex::Real r_lo,
                       int alpha, const amrex::Box &bx,
                       const amrex::FArrayBox &statein,
                       amrex::FArrayBox       &stateout)
{
    using namespace amrex;
    constexpr int NCOMP = 3;
    AMREX_ASSERT(statein.nComp() == NCOMP);
    AMREX_ASSERT(stateout.nComp() == NCOMP);

    amrex::FArrayBox tmpfab(bx, NCOMP * 4, The_Async_Arena());

    const auto &in  = statein.const_array();
    const auto &out = stateout.array();

    const auto &k1 = tmpfab.array(0, 3);
    const auto &k2 = tmpfab.array(NCOMP, 3);
    const auto &k3 = tmpfab.array(2 * NCOMP, 3);
    const auto &k4 = tmpfab.array(3 * NCOMP, 3);

    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_prob_parm->epsilon;

    // Compute K1
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (i + 0.5) * dr + r_lo;
                    const auto &k1arr
                        = geometric_source(r, alpha, adia, eps, out, i);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k1(i, j, k, n) = dt * k1arr[n];
                });

    // Compute K2
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + 0.5 * k1(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (i + 0.5) * dr + r_lo;
                    const auto &k2arr
                        = geometric_source(r, alpha, adia, eps, out, i);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k2(i, j, k, n) = dt * k2arr[n];
                });

    // Compute K3
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + 0.5 * k2(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (i + 0.5) * dr + r_lo;
                    const auto &k3arr
                        = geometric_source(r, alpha, adia, eps, out, i);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k3(i, j, k, n) = dt * k3arr[n];
                });

    // Compute K4
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + k3(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r = (i + 0.5) * dr + r_lo;
                    const auto &k4arr
                        = geometric_source(r, alpha, adia, eps, out, i);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k4(i, j, k, n) = dt * k4arr[n];
                });

    // Compute Un+1
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    out(i, j, k, n) = in(i, j, k, n)
                                      + (k1(i, j, k, n) + 2 * k2(i, j, k, n)
                                         + 2 * k3(i, j, k, n) + k4(i, j, k, n))
                                            / 6;
                });
}

template <int axis_loc>
AMREX_GPU_HOST void advance_geometric_2d(amrex::Real dt, amrex::Real dr,
                                         amrex::Real r_lo, int alpha,
                                         const amrex::Box       &bx,
                                         const amrex::FArrayBox &statein,
                                         amrex::FArrayBox       &stateout)
{
    using namespace amrex;
    constexpr int NCOMP = 4;
    AMREX_ASSERT(statein.nComp() == NCOMP);
    AMREX_ASSERT(stateout.nComp() == NCOMP);

    amrex::FArrayBox tmpfab(bx, NCOMP * 4, The_Async_Arena());

    const auto &in  = statein.const_array();
    const auto &out = stateout.array();

    const auto &k1 = tmpfab.array(0, NCOMP);
    const auto &k2 = tmpfab.array(NCOMP, NCOMP);
    const auto &k3 = tmpfab.array(2 * NCOMP, NCOMP);
    const auto &k4 = tmpfab.array(3 * NCOMP, NCOMP);

    const Real adia = AmrLevelAdv::h_prob_parm->adiabatic;
    const Real eps  = AmrLevelAdv::h_prob_parm->epsilon;

    // Compute K1
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r     = (axis_loc == 0) ? (i + 0.5) * dr + r_lo
                                                        : (j + 0.5) * dr + r_lo;
                    const auto &k1arr = geometric_source<axis_loc>(
                        r, alpha, adia, eps, out, i, j);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k1(i, j, k, n) = dt * k1arr[n];
                });

    // Compute K2
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + 0.5 * k1(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r     = (axis_loc == 0) ? (i + 0.5) * dr + r_lo
                                                        : (j + 0.5) * dr + r_lo;
                    const auto &k2arr = geometric_source<axis_loc>(
                        r, alpha, adia, eps, out, i, j);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k2(i, j, k, n) = dt * k2arr[n];
                });

    // Compute K3
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + 0.5 * k2(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r     = (axis_loc == 0) ? (i + 0.5) * dr + r_lo
                                                        : (j + 0.5) * dr + r_lo;
                    const auto &k3arr = geometric_source<axis_loc>(
                        r, alpha, adia, eps, out, i, j);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k3(i, j, k, n) = dt * k3arr[n];
                });

    // Compute K4
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                { out(i, j, k, n) = in(i, j, k, n) + k3(i, j, k, n); });

    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const Real  r     = (axis_loc == 0) ? (i + 0.5) * dr + r_lo
                                                        : (j + 0.5) * dr + r_lo;
                    const auto &k4arr = geometric_source<axis_loc>(
                        r, alpha, adia, eps, out, i, j);
                    AMREX_PRAGMA_SIMD
                    for (int n = 0; n < NCOMP; ++n)
                        k4(i, j, k, n) = dt * k4arr[n];
                });

    // Compute Un+1
    ParallelFor(bx, NCOMP,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    out(i, j, k, n) = in(i, j, k, n)
                                      + (k1(i, j, k, n) + 2 * k2(i, j, k, n)
                                         + 2 * k3(i, j, k, n) + k4(i, j, k, n))
                                            / 6;
                });
}

// template instantiation
template AMREX_GPU_HOST void advance_geometric_2d<0>(amrex::Real, amrex::Real,
                                                     amrex::Real, int,
                                                     const amrex::Box &,
                                                     const amrex::FArrayBox &,
                                                     amrex::FArrayBox &);

template AMREX_GPU_HOST void advance_geometric_2d<1>(amrex::Real, amrex::Real,
                                                     amrex::Real, int,
                                                     const amrex::Box &,
                                                     const amrex::FArrayBox &,
                                                     amrex::FArrayBox &);