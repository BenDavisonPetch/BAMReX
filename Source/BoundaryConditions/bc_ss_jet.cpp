/**
 * Boundary conditions for a co-axial jet
 */
#include "BoundaryConditions/bc_ss_jet.H"
#include "System/Euler/Euler.H"
#include "System/Euler/NComp.H"
#include "System/Euler/RH.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;
using namespace bamrex;

/**
 * Fills boundaries for the supersonic jet problem. For the inflow, sets all
 * variables according to Rankine Hugoniot conditions, using the mach number
 * relative to the given ambient values
 *
 * All boundaries other than the inflow are transmissive (foextrap) boundaries.
 *
 * wall specifies which wall the jet is on: 0 : x_lo, 1: x_hi, 2: y_lo, 3:
 * y_hi, 4: z_lo, 5: z_hi
 */
struct SuperSonicJetFill
{
    AMREX_GPU_HOST
    SuperSonicJetFill(const amrex::RealArray &center, amrex::Real radius,
                      int wall, amrex::Real M, amrex::Real rho_amb,
                      amrex::Real p_amb)
        : m_c(makeXDim3(center))
        , m_r(radius)
        , m_p_amb(p_amb)
        , m_wall(wall)
    {
        const Parm *hparm      = AmrLevelAdv::h_parm;
        const auto  Q_inflo_1D = post_shock(M, rho_amb, p_amb, *hparm);
        const Real  v          = Q_inflo_1D[1] * ((wall % 2 == 0) ? 1 : -1);
        const GpuArray<Real, EULER_NCOMP> Q_inflo{
            Q_inflo_1D[0],
            AMREX_D_DECL((wall == 0 || wall == 1) ? v : 0,
                         (wall == 2 || wall == 3) ? v : 0,
                         (wall == 4 || wall == 5) ? v : 0),
            Q_inflo_1D[2]
        };
        m_U_inflo
            = consv_from_primv(Q_inflo, hparm->adiabatic, hparm->epsilon);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(const IntVect &iv, Array4<Real> const &dest, int dcomp,
               int numcomp, GeometryData const &geom, Real /*time*/,
               const BCRec *bcr, int bcomp, int /*orig_comp*/) const
    {
        const auto &dx      = geom.CellSize();
        const auto &prob_lo = geom.ProbLo();

        AMREX_ASSERT(numcomp == EULER_NCOMP);
        AMREX_ASSERT(dcomp == 0);

#if AMREX_SPACEDIM == 1
        const int  i = iv[0];
        const int  j = 0;
        const int  k = 0;
        const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        const Real y = 0;
        const Real z = 0;
#endif
#if AMREX_SPACEDIM == 2
        const int  i = iv[0];
        const int  j = iv[1];
        const int  k = 0;
        const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        const Real y = (j + 0.5) * dx[1] + prob_lo[1] - m_c.y;
        const Real z = 0;
#endif
#if AMREX_SPACEDIM == 3
        const int  i = iv[0];
        const int  j = iv[1];
        const int  k = iv[2];
        const Real x = (i + 0.5) * dx[0] + prob_lo[0] - m_c.x;
        const Real y = (j + 0.5) * dx[1] + prob_lo[1] - m_c.y;
        const Real z = (k + 0.5) * dx[2] + prob_lo[2] - m_c.z;
#endif

        const auto &domain_lo = geom.Domain().loVect3d();
        const auto &domain_hi = geom.Domain().hiVect3d();
        const int   ilo       = domain_lo[0];
        const int   jlo       = domain_lo[1];
        const int   klo       = domain_lo[2];
        const int   ihi       = domain_hi[0];
        const int   jhi       = domain_hi[1];
        const int   khi       = domain_hi[2];

        // bcrec of density component. If it is not user_1, we shouldn't do
        // anything here
        const BCRec &bc = bcr[bcomp - dcomp];
        if (i < ilo && bc.lo(0) != BCType::user_1)
            return;
        if (j < jlo && bc.lo(1) != BCType::user_1)
            return;
        if (k < klo && bc.lo(2) != BCType::user_1)
            return;
        if (i > ihi && bc.hi(0) != BCType::user_1)
            return;
        if (j > jhi && bc.hi(1) != BCType::user_1)
            return;
        if (k > khi && bc.hi(2) != BCType::user_1)
            return;

        [[maybe_unused]] const int ireal
            = (i < ilo) ? ilo : ((i > ihi) ? ihi : i);
        [[maybe_unused]] const int jreal
            = (j < jlo) ? jlo : ((j > jhi) ? jhi : j);
        [[maybe_unused]] const int kreal
            = (k < klo) ? klo : ((k > khi) ? khi : k);
        const IntVect ivreal(AMREX_D_DECL(ireal, jreal, kreal));

        bool jet_wall
            = ((i < ilo && m_wall == 0) || (i > ihi && m_wall == 1)
               || (j < jlo && m_wall == 2) || (j > jhi && m_wall == 3)
               || (k < klo && m_wall == 4) || (k > khi && m_wall == 5));

        Real r = 0;
        switch (m_wall)
        {
        case 0:
        case 1: r = sqrt(y * y + z * z); break;
        case 2:
        case 3: r = sqrt(x * x + z * z); break;
        case 4:
        case 5: r = sqrt(x * x + y * y); break;
        }

        if (r > m_r || !jet_wall)
        {
            // transmissive
            for (int n = dcomp; n < dcomp + numcomp; ++n)
                dest(i, j, k, n) = dest(ivreal, n);
        }
        else // jet
        {
            // supersonic inflow
            for (int n = dcomp; n < dcomp + numcomp; ++n)
                dest(i, j, k, n) = m_U_inflo[n];
        }
    }

  protected:
    const XDim3 m_c; // center
    const Real  m_r; // jet radius
    const Real  m_p_amb;
    const int m_wall; // 0 : x_lo, 1: x_hi, 2: y_lo, 3: y_hi, 4: z_lo, 5: z_hi
    GpuArray<Real, EULER_NCOMP> m_U_inflo;
};

amrex::BndryFuncFabDefault
make_ss_jetfill_bcfunc(const amrex::RealArray &center, amrex::Real radius,
                       int wall, amrex::Real M, amrex::Real rho_amb,
                       amrex::Real p_amb)
{
    GpuBndryFuncFab<SuperSonicJetFill> gpu_bndry_func(
        SuperSonicJetFill{ center, radius, wall, M, rho_amb, p_amb });
    return gpu_bndry_func;
}