/**
 * Normal vector computation for the supported geometry types. Normal vectors
 * always point into the fluid. Since LS > 0 inside body, normal is -grad(LS)
 */

#include "Normals.H"

#include "AMReX_MultiFab.H"
#include <AMReX_MFIter.H>

using namespace amrex;

void fill_ls_normals(const amrex::Geometry &geom, amrex::MultiFab &LS)
{
    AMREX_ASSERT(LS.nComp() == AMREX_SPACEDIM + 1);

    const auto &dx = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(LS, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx = mfi.growntilebox(LS.nGrowVect() - 1);
            // const auto &bx = mfi.tilebox();
            const auto &ls = LS.array(mfi);
            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        const int i_off = (d == 0) ? 1 : 0;
                        const int j_off = (d == 1) ? 1 : 0;
                        const int k_off = (d == 2) ? 1 : 0;
                        ls(i, j, k, 1 + d)
                            = ((std::fabs(ls(i + i_off, j + j_off, k + k_off))
                                < std::fabs(
                                    ls(i - i_off, j - j_off, k - k_off)))
                                   ? ls(i + i_off, j + j_off, k + k_off)
                                         - ls(i, j, k)
                                   : ls(i, j, k)
                                         - ls(i - i_off, j - j_off, k - k_off))
                              / dx[d];
                    }
                    // normalise just in case (should be 1 but who knows)
                    const Real magnitude
                        = sqrt(AMREX_D_TERM(ls(i, j, k, 1) * ls(i, j, k, 1),
                                            +ls(i, j, k, 2) * ls(i, j, k, 2),
                                            +ls(i, j, k, 3) * ls(i, j, k, 3)));
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        ls(i, j, k, 1 + d) /= -magnitude;
                });
        }
    }
}