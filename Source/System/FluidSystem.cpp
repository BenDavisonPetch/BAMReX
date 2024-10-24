#include "FluidSystem.H"
#include "GFM/RigidBody.H"
#include "System/Euler/Euler.H"
#include "Visc/Visc_K.H"

using namespace amrex;

void FluidSystem::ReadParameters(Parm &parm, amrex::ParmParse &pp,
                                 amrex::ParmParse &pp_limiting)
{
    set_parm_defaults(parm);

    pp.get("adiabatic", parm.adiabatic);
    pp.query("epsilon", parm.epsilon);
    pp.query("eos_mu", parm.eos_mu);
    pp.query("Pr", parm.Pr);
    pp.query("beta", parm.beta);
    pp.query("eta_prime", parm.eta_prime);
    pp.query("T_prime", parm.T_prime);
    pp.query("S_eta", parm.S_eta);
    pp.query("Re", parm.Re);

    std::string              density_limiter, energy_limiter;
    std::vector<std::string> mom_limiters(3);
    pp_limiting.query("limiter", parm.limiter);
    pp_limiting.query("density", density_limiter);
    pp_limiting.queryarr("mom", mom_limiters);
    pp_limiting.query("energy", energy_limiter);

    if (!density_limiter.empty())
    {
        parm.limiter_indices[UInd::RHO] = VariableIndex(density_limiter);
        parm.limit_on_primv[UInd::RHO]  = VariableIsPrimitive(density_limiter);
    }
    if (!energy_limiter.empty())
    {
        parm.limiter_indices[UInd::E] = VariableIndex(energy_limiter);
        parm.limit_on_primv[UInd::E]  = VariableIsPrimitive(energy_limiter);
    }
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        if (!mom_limiters[i].empty())
        {
            parm.limiter_indices[1 + i] = VariableIndex(mom_limiters[i]);
            parm.limit_on_primv[1 + i]  = VariableIsPrimitive(mom_limiters[i]);
        }

    init_parm(parm);
}

void FluidSystem::set_parm_defaults(Parm &parm)
{
    parm.adiabatic = -1;       // Must be set in AmrLevelAdv::read_params()
    parm.eos_mu    = 28.96;    // air
    parm.Pr        = 0.71;     // air
    parm.beta      = 1.5;      // air
    parm.eta_prime = 1.716e-5; // air
    parm.T_prime   = 273;      // air
    parm.S_eta     = 111;      // air

    parm.epsilon = 1;
    parm.Re      = 1;
    parm.limiter = Limiter::van_leer;
    for (int i = 0; i < MAX_NCOMP; ++i)
    {
        parm.limiter_indices[i] = UInd::E;
        parm.limit_on_primv[i]  = false;
    }
    parm.need_primitive_for_limiting = false;
}

void FluidSystem::init_parm(Parm &parm)
{
    //! Calculate derived variables
    // adiabatic precomputation
    const Real adiabatic = parm.adiabatic;
    parm.adia[0]         = adiabatic;
    parm.adia[1]         = adiabatic + 1;
    parm.adia[2]         = adiabatic - 1;
    parm.adia[3]         = (Real)1 / (adiabatic + 1);
    parm.adia[4]         = (Real)1 / (adiabatic - 1);
    parm.adia[5]         = (adiabatic + 1) / (2 * adiabatic);
    parm.adia[6]         = (adiabatic - 1) / (2 * adiabatic);
    parm.adia[7]         = adiabatic / (adiabatic + 1);
    parm.adia[8]         = adiabatic / (adiabatic - 1);

    constexpr Real Ru = 8.3145; // J/mol/K
    // eos_mu is in g/mol
    parm.R_spec = Ru / (parm.eos_mu * 1e-3); // J/kg/K
    parm.cv     = parm.R_spec * parm.adia[4];
    parm.cp     = adiabatic * parm.cv;
    parm.Reinv  = (Real)1 / parm.Re;

    parm.need_primitive_for_limiting = false;
    for (int n = 0; n < MAX_NCOMP; ++n)
    {
        parm.need_primitive_for_limiting
            = parm.need_primitive_for_limiting || parm.limit_on_primv[n];
    }
}

AMREX_GPU_HOST
void FluidSystem::FillFluxes(int dir, amrex::Real time, const amrex::Box &bx,
                             const amrex::Array4<amrex::Real>       &flux,
                             const amrex::Array4<const amrex::Real> &U,
                             const Parm *h_parm, const Parm * /*d_parm*/) const
{
    const Real adia = h_parm->adiabatic;
    const Real eps  = h_parm->epsilon;
    if (dir == 0)
        compute_flux_function<0>(time, bx, flux, U, adia, eps);
#if AMREX_SPACEDIM > 1
    else if (dir == 1)
        compute_flux_function<1>(time, bx, flux, U, adia, eps);
#if AMREX_SPACEDIM > 2
    else if (dir == 2)
        compute_flux_function<2>(time, bx, flux, U, adia, eps);
#endif
#endif
}

AMREX_GPU_HOST
void FluidSystem::UtoQ(const amrex::Box                       &bx,
                       const amrex::Array4<amrex::Real>       &Q,
                       const amrex::Array4<const amrex::Real> &U,
                       const Parm * /*h_parm*/, const Parm *d_parm) const
{
    AMREX_ASSERT(U.nComp() == UInd::size);
    AMREX_ASSERT(Q.nComp() == QInd::size);
    ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const auto Qloc = primv_from_consv(
                        U, d_parm->adiabatic, d_parm->epsilon, i, j, k);
                    for (int n = 0; n < EULER_NCOMP; ++n)
                    {
                        Q(i, j, k, n) = Qloc[n];
                    }
                });
}

AMREX_GPU_HOST
void FluidSystem::UtoExtQ(const amrex::Box                       &bx,
                          const amrex::Array4<amrex::Real>       &Q,
                          const amrex::Array4<const amrex::Real> &U,
                          const Parm * /*h_parm*/, const Parm *d_parm) const
{
    AMREX_ASSERT(U.nComp() == UInd::size);
    AMREX_ASSERT(Q.nComp() == QInd::vsize);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { visc_ctop(i, j, k, Q, U, *d_parm); });
}

AMREX_GPU_HOST
amrex::Real FluidSystem::MaxWaveSpeed(const amrex::Real      time,
                                      const amrex::MultiFab &U,
                                      const Parm            *h_parm,
                                      const Parm * /*d_parm*/) const
{
    return max_wave_speed(time, U, h_parm->adiabatic, h_parm->epsilon);
}

AMREX_GPU_HOST
void FluidSystem::FillRBGhostCells(
    const amrex::Geometry &geom, amrex::MultiFab &U, const amrex::MultiFab &LS,
    const amrex::iMultiFab          &gfm_flags,
    RigidBodyBCType::rigidbodybctype rb_bc) const
{
    fill_ghost_rb(geom, U, LS, gfm_flags, rb_bc);
}