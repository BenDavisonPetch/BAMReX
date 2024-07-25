#include "GFMFlag.H"
#include <AMReX_MFIter.H>

using namespace amrex;

AMREX_FORCE_INLINE AMREX_GPU_DEVICE bool
on_interface(const Array4<const Real> &ls, int i, int j, int k)
{
    return AMREX_D_TERM((ls(i, j, k) * ls(i + 1, j, k) <= 0
                         || ls(i, j, k) * ls(i - 1, j, k) <= 0),
                        || (ls(i, j, k) * ls(i, j + 1, k) <= 0
                            || ls(i, j, k) * ls(i, j - 1, k) <= 0),
                        || (ls(i, j, k) * ls(i, j, k + 1) <= 0
                            || ls(i, j, k) * ls(i, j, k - 1) <= 0));
}

AMREX_FORCE_INLINE AMREX_GPU_DEVICE bool
close_to_interface(const Array4<const Real> &ls, int i, int j, int k)
{
    bool close = false;
    for (int dist = 1; dist <= GFM_GROW; ++dist)
    {
        close
            = close
              || AMREX_D_TERM((ls(i, j, k) * ls(i + dist, j, k) <= 0)
                                  || (ls(i, j, k) * ls(i - dist, j, k) <= 0),
                              || (ls(i, j, k) * ls(i, j + dist, k) <= 0)
                                  || (ls(i, j, k) * ls(i, j - dist, k) <= 0),
                              || (ls(i, j, k) * ls(i, j, k + dist) <= 0)
                                  || (ls(i, j, k) * ls(i, j, k - dist) <= 0));
    }
    return close;
}

AMREX_GPU_HOST
void build_gfm_flags(amrex::iMultiFab &flags, const amrex::MultiFab &LS)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(flags, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto &bx      = mfi.tilebox();
            const auto &ls      = LS.const_array(mfi);
            const auto &flagarr = flags.array(mfi);

            ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            flagarr(i, j, k) = 0;
                            if (ls(i, j, k) >= 0) // body
                                flagarr(i, j, k) |= GFMFlag::ghost;
                            if (on_interface(ls, i, j, k))
                                flagarr(i, j, k) |= GFMFlag::interface;
                            if (ls(i, j, k) >= 0
                                && close_to_interface(ls, i, j, k))
                                flagarr(i, j, k) |= GFMFlag::fill;
                        });
        }
    }
}