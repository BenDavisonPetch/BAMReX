/**
 * Takes care of all ext_dir and custom boundary conditions
 */

#ifndef bc_extfill_H_
#define bc_extfill_H_

#include "Euler/Euler.H"
#include "Euler/NComp.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_StateDescriptor.H>

/**
 * \tparam N how many components the function will be used to fill
 */
template <unsigned int N>
struct ExtFill
{
    ExtFill(const amrex::GpuArray<amrex::Real, N>& fill_vals)
        : m_fill(fill_vals)
    {
    }

    AMREX_GPU_DEVICE
    void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &dest, int dcomp,
                    int numcomp, amrex::GeometryData const &geom, amrex::Real /*time*/,
                    const amrex::BCRec * bcr, int bcomp,
                    int /*orig_comp*/) const
    {
        using namespace amrex;
        // const auto &dx      = geom.CellSize();
        // const auto &prob_lo = geom.ProbLo();

#if AMREX_SPACEDIM == 1
        const int  i = iv[0];
        const int  j = 0;
        const int  k = 0;
        // const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        // const Real y = 0;
        // const Real z = 0;
#endif
#if AMREX_SPACEDIM == 2
        const int  i = iv[0];
        const int  j = iv[1];
        const int  k = 0;
        // const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        // const Real y = (j + 0.5) * dx[1] + prob_lo[1] - m_c.y;
        // const Real z = 0;
#endif
#if AMREX_SPACEDIM == 3
        const int  i = iv[0];
        const int  j = iv[1];
        const int  k = iv[2];
        // const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        // const Real y = (j + 0.5) * dx[1] + prob_lo[1] - m_c.y;
        // const Real z = (k + 0.5) * dx[2] + prob_lo[2] - m_c.z;
#endif

        const auto &domain_lo = geom.Domain().loVect3d();
        const auto &domain_hi = geom.Domain().hiVect3d();
        const int   ilo       = domain_lo[0];
        const int   jlo       = domain_lo[1];
        const int   klo       = domain_lo[2];
        const int   ihi       = domain_hi[0];
        const int   jhi       = domain_hi[1];
        const int   khi       = domain_hi[2];

        for (int n = dcomp; n < dcomp + numcomp; ++n)
        {
            const BCRec& bc = bcr[bcomp+n-dcomp];

            if (i < ilo)
            {
                if (bc.lo(0) == BCType::ext_dir)
                    dest(i, j, k, n) = m_fill[n];
            }
            else if (i > ihi)
            {
                if (bc.hi(0) == BCType::ext_dir)
                    dest(i, j, k, n) = m_fill[n];
            }

            if (j < jlo)
            {
                if (bc.lo(1) == BCType::ext_dir)
                    dest(i, j, k, n) = m_fill[n];
            }
            else if (j > jhi)
            {
                if (bc.hi(1) == BCType::ext_dir)
                    dest(i, j, k, n) = m_fill[n];
            }

            if (k < klo)
            {
                if (bc.lo(2) == BCType::ext_dir)
                    dest(i, j, k, n) = m_fill[n];
            }
            else if (k > khi)
            {
                if (bc.hi(2) == BCType::ext_dir)
                    dest(i, j, k, n) = m_fill[n];
            }
        }
    }

  protected:
    amrex::GpuArray<amrex::Real, N> m_fill;
};

template <unsigned int N>
amrex::BndryFuncFabDefault
make_extfill(const amrex::GpuArray<amrex::Real, N>& fill_val)
{
    using namespace amrex;
    GpuBndryFuncFab<ExtFill<N>> gpu_bndry_func(ExtFill<N>{fill_val});
    return gpu_bndry_func;
}

#endif