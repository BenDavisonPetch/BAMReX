#include "PlasmaSystem.H"
#include "GFM/RigidBody.H"
#include "System/Plasma/Plasma_K.H"
#include "Visc/Visc_K.H"

using namespace amrex;

void PlasmaSystem::ReadParameters(Parm &parm, amrex::ParmParse &pp,
                                  amrex::ParmParse &pp_limiting)
{
    BL_PROFILE("PlasmaSystem::ReadParameters()");

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
    std::vector<std::string> B_limiters(3);
    pp_limiting.query("limiter", parm.limiter);
    pp_limiting.query("density", density_limiter);
    pp_limiting.queryarr("mom", mom_limiters);
    pp_limiting.query("energy", energy_limiter);
    pp_limiting.queryarr("B", B_limiters);

    if (!density_limiter.empty())
    {
        parm.limiter_indices[PUInd::RHO] = VariableIndex(density_limiter);
        parm.limit_on_primv[PUInd::RHO] = VariableIsPrimitive(density_limiter);
    }
    if (!energy_limiter.empty())
    {
        parm.limiter_indices[PUInd::E] = VariableIndex(energy_limiter);
        parm.limit_on_primv[PUInd::E]  = VariableIsPrimitive(energy_limiter);
    }
    for (int i = 0; i < 3; ++i)
        if (!mom_limiters[i].empty())
        {
            parm.limiter_indices[PUInd::Mdir(i)]
                = VariableIndex(mom_limiters[i]);
            parm.limit_on_primv[PUInd::Mdir(i)]
                = VariableIsPrimitive(mom_limiters[i]);
        }
    for (int i = 0; i < 3; ++i)
        if (!B_limiters[i].empty())
        {
            parm.limiter_indices[PUInd::Bdir(i)]
                = VariableIndex(B_limiters[i]);
            parm.limit_on_primv[PUInd::Bdir(i)]
                = VariableIsPrimitive(B_limiters[i]);
        }

    init_parm(parm);
}

void PlasmaSystem::set_parm_defaults(Parm &parm)
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
        parm.limiter_indices[i] = PUInd::E;
        parm.limit_on_primv[i]  = false;
    }
    parm.need_primitive_for_limiting = false;
}

void PlasmaSystem::init_parm(Parm &parm)
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

AMREX_GPU_HOST amrex::Vector<std::string> PlasmaSystem::VariableNames() const
{
    return amrex::Vector<std::string>{ "density", "mom_x", "mom_y", "mom_z",
                                       "energy",  "B_x",   "B_y",   "B_z" };
}

AMREX_GPU_HOST
void PlasmaSystem::FillFluxes(int dir, amrex::Real time, const amrex::Box &bx,
                              const amrex::Array4<amrex::Real>       &flux,
                              const amrex::Array4<const amrex::Real> &U,
                              const Parm *h_parm, const Parm *d_parm) const
{
    switch (dir)
    {
    case 0: FillFluxes<0>(time, bx, flux, U, h_parm, d_parm); break;
    case 1: FillFluxes<1>(time, bx, flux, U, h_parm, d_parm); break;
    case 2: FillFluxes<2>(time, bx, flux, U, h_parm, d_parm); break;
    }
}

template <int dir>
AMREX_GPU_HOST void
PlasmaSystem::FillFluxes(amrex::Real /*time*/, const amrex::Box &bx,
                         const amrex::Array4<amrex::Real>       &flux,
                         const amrex::Array4<const amrex::Real> &U,
                         const Parm * /*h_parm*/, const Parm *d_parm) const
{
    BL_PROFILE("PlasmaSystem::FillFluxes()");
    AMREX_ASSERT(U.nComp() == PUInd::size);
    AMREX_ASSERT(flux.nComp() == PUInd::size);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { mhd_flux<dir>(i, j, k, flux, U, *d_parm); });
}

AMREX_GPU_HOST
void PlasmaSystem::UtoQ(const amrex::Box                       &bx,
                        const amrex::Array4<amrex::Real>       &Q,
                        const amrex::Array4<const amrex::Real> &U,
                        const Parm * /*h_parm*/, const Parm *d_parm) const
{
    BL_PROFILE("PlasmaSystem::UtoQ()");
    AMREX_ASSERT(U.nComp() == PUInd::size);
    AMREX_ASSERT(Q.nComp() == PQInd::size);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { mhd_ctop(i, j, k, Q, U, *d_parm); });
}

AMREX_GPU_HOST
void PlasmaSystem::UtoExtQ(const amrex::Box                       &bx,
                           const amrex::Array4<amrex::Real>       &Q,
                           const amrex::Array4<const amrex::Real> &U,
                           const Parm *h_parm, const Parm *d_parm) const
{
    BL_PROFILE("PlasmaSystem::UtoExtQ()");
    ignore_unused(bx, Q, U, h_parm, d_parm);
    Abort("UtoExtQ not implemented for PlasmaSystem");
}

AMREX_GPU_HOST
amrex::Real PlasmaSystem::ExplicitTimeStepDimSplit(const amrex::Real /*time*/,
                                                   const amrex::Geometry &geom,
                                                   const amrex::MultiFab &U,
                                                   const Parm * /*h_parm*/,
                                                   const Parm *d_parm) const
{
    BL_PROFILE("PlasmaSystem::ExplicitTimeStepDimSplit()");
    const auto &dx     = geom.CellSizeArray();
    const auto &ma     = U.const_arrays();
    auto        min_dt = ParReduce(
               TypeList<ReduceOpMin>{}, TypeList<Real>{}, U,
               IntVect(0), // no ghost cells
               [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                   noexcept -> GpuTuple<Real>
               {
            const Array4<const Real> &Uarr = ma[box_no];
            return mhd_exp_timestep_dim_split(i, j, k, Uarr, dx, *d_parm);
        });

    BL_PROFILE_VAR("PlasmaSystem::ExplicitTimeStepDimSplit::ReduceRealMin",
                   preduce);
    ParallelDescriptor::ReduceRealMin(min_dt);
    BL_PROFILE_VAR_STOP(preduce);
    return min_dt;
}

AMREX_GPU_HOST
void PlasmaSystem::FillRBGhostCells(
    const amrex::Geometry &geom, amrex::MultiFab &U, const amrex::MultiFab &LS,
    const amrex::iMultiFab          &gfm_flags,
    RigidBodyBCType::rigidbodybctype rb_bc) const
{
    BL_PROFILE("PlasmaSystem::FillRBGhostCells()");

    ignore_unused(geom, U, LS, gfm_flags, rb_bc);
    Abort("Ghost cell filling not implemented for plasmas!");
}

//
// Template instantiation
//

template AMREX_GPU_HOST void
PlasmaSystem::FillFluxes<0>(amrex::Real, const amrex::Box &,
                            const amrex::Array4<amrex::Real> &,
                            const amrex::Array4<const amrex::Real> &,
                            const Parm *, const Parm *) const;

template AMREX_GPU_HOST void
PlasmaSystem::FillFluxes<1>(amrex::Real, const amrex::Box &,
                            const amrex::Array4<amrex::Real> &,
                            const amrex::Array4<const amrex::Real> &,
                            const Parm *, const Parm *) const;

template AMREX_GPU_HOST void
PlasmaSystem::FillFluxes<2>(amrex::Real, const amrex::Box &,
                            const amrex::Array4<amrex::Real> &,
                            const amrex::Array4<const amrex::Real> &,
                            const Parm *, const Parm *) const;