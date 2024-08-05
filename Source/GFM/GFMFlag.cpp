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
close_to_interface(const Array4<const Real> &ls, int i, int j, int k,
                   const GFMFillInfo &fill_info)
{
    bool close = false;

    if (!fill_info.fill_diagonals)
    {
        for (int dist = -fill_info.stencil; dist <= fill_info.stencil; ++dist)
        {
            close
                = close
                  || AMREX_D_TERM((ls(i, j, k) * ls(i + dist, j, k) <= 0),
                                  || (ls(i, j, k) * ls(i, j + dist, k) <= 0),
                                  || (ls(i, j, k) * ls(i, j, k + dist) <= 0));
        }
    }
    else
    {
        for (int ii = -fill_info.stencil; ii <= fill_info.stencil; ++ii)
        {
            const int jrange
                = (AMREX_SPACEDIM >= 2) ? fill_info.stencil - std::abs(ii) : 0;
            for (int jj = -jrange; jj <= jrange; ++jj)
            {
                const int krange
                    = (AMREX_SPACEDIM == 3) ? jrange - std::abs(jj) : 0;
                for (int kk = -krange; kk <= krange; ++kk)
                {
                    close = close
                            || ls(i, j, k) * ls(i + ii, j + jj, k + kk) <= 0;
                }
            }
        }
    }
    return close;
}

AMREX_GPU_HOST
void build_gfm_flags(amrex::iMultiFab &flags, const amrex::MultiFab &LS,
                     const GFMFillInfo &fill_info)
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
                                && close_to_interface(ls, i, j, k, fill_info))
                                flagarr(i, j, k) |= GFMFlag::fill;
                        });
        }
    }
}