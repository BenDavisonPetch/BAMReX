#ifndef Interp_K_H_
#define Interp_K_H_

#include <AMReX_Array4.H>
#include <AMReX_RealVect.H>

#include <AMReX_Print.H>

/**
 * \brief Bi(tri)-linear interpolation of \c src at \c p onto \c dst(i,j,k)
 * assumes that the origin is at index (0,0)
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void bilinear_interp(const amrex::RealVect                  &p,
                     const amrex::Array4<const amrex::Real> &src,
                     const amrex::Array4<amrex::Real> &dst, int i_dst,
                     int j_dst, int k_dst,
                     const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> &dx,
                     int                                                 ncomp)
{
    using namespace amrex;
    // coordinates of desired point in index space
    RealVect ijk(AMREX_D_DECL(p[0] / dx[0] - 0.5, p[1] / dx[1] - 0.5,
                              p[2] / dx[2] - 0.5));

    IntVect ijk000(AMREX_D_DECL(ijk[0], ijk[1], ijk[2]));
    IntVect ijk100(AMREX_D_DECL(ceil(ijk[0]), ijk[1], ijk[2]));

    IntVect ijk010(AMREX_D_DECL(ijk[0], ceil(ijk[1]), ijk[2]));
    IntVect ijk110(AMREX_D_DECL(ceil(ijk[0]), ceil(ijk[1]), ijk[2]));

    IntVect ijk001(AMREX_D_DECL(ijk[0], ijk[1], ceil(ijk[2])));
    IntVect ijk101(AMREX_D_DECL(ceil(ijk[0]), ijk[1], ceil(ijk[2])));
    IntVect ijk011(AMREX_D_DECL(ijk[0], ceil(ijk[1]), ceil(ijk[2])));
    IntVect ijk111(AMREX_D_DECL(ceil(ijk[0]), ceil(ijk[1]), ceil(ijk[2])));

    // translate desired point so that all the values are in interval [0,1)
    ijk -= ijk000;
#if AMREX_SPACEDIM == 1
    XDim3 Dx({ ijk[0], 0, 0 });
#elif AMREX_SPACEDIM == 2
    XDim3 Dx({ ijk[0], ijk[1], 0 });
#elif AMREX_SPACEDIM == 3
    XDim3 Dx({ ijk[0], ijk[1], ijk[2] });
#endif

    AMREX_PRAGMA_SIMD
    for (int n = 0; n < ncomp; ++n)
    {
        const Real c0 = src(ijk000, n);
        const Real c1 = src(ijk100, n) - src(ijk000, n);
        const Real c2 = src(ijk010, n) - src(ijk000, n);
        const Real c3 = src(ijk001, n) - src(ijk000, n);
        const Real c4 = src(ijk110, n) - src(ijk010, n) - src(ijk100, n)
                        + src(ijk000, n);
        const Real c5 = src(ijk011, n) - src(ijk001, n) - src(ijk010, n)
                        + src(ijk000, n);
        const Real c6 = src(ijk101, n) - src(ijk001, n) - src(ijk100, n)
                        + src(ijk000, n);
        const Real c7 = src(ijk111, n) - src(ijk011, n) - src(ijk101, n)
                        - src(ijk110, n) + src(ijk100, n) + src(ijk001, n)
                        + src(ijk010) - src(ijk000);

        dst(i_dst, j_dst, k_dst, n)
            = c0 + c1 * Dx.x + c2 * Dx.y + c3 * Dx.z + c4 * Dx.x * Dx.y
              + c5 * Dx.y * Dx.z + c6 * Dx.z * Dx.x + c7 * Dx.x * Dx.y * Dx.z;
    }
}

#endif