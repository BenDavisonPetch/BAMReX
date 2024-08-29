#include "Parm.H"
#include "Fluxes/Limiters.H"

using namespace amrex;

void Parm::set_defaults()
{
    adiabatic = -1;       // Must be set in AmrLevelAdv::read_params()
    eos_mu    = 28.96;    // air
    Pr        = 0.71;     // air
    beta      = 1.5;      // air
    eta_prime = 1.716e-5; // air
    T_prime   = 273;      // air
    S_eta     = 111;      // air

    epsilon = 1;
    Re      = 1;
    limiter = van_leer_limiter;
    for (int i = 0; i < EULER_NCOMP; ++i)
        limiter_indices[i] = 1 + AMREX_SPACEDIM;
}

void Parm::init()
{
    //! Calculate derived variables
    // adiabatic precomputation
    adia[0] = adiabatic;
    adia[1] = adiabatic + 1;
    adia[2] = adiabatic - 1;
    adia[3] = (Real)1 / (adiabatic + 1);
    adia[4] = (Real)1 / (adiabatic - 1);
    adia[5] = (adiabatic + 1) / (2 * adiabatic);
    adia[6] = (adiabatic - 1) / (2 * adiabatic);
    adia[7] = adiabatic / (adiabatic + 1);
    adia[8] = adiabatic / (adiabatic - 1);

    constexpr Real Ru = 8.3145; // J/mol/K
    // eos_mu is in g/mol
    R_spec = Ru / (eos_mu * 1e-3); // J/kg/K
    cv     = R_spec * adia[4];
    cp     = adiabatic * cv;
    Reinv  = (Real)1 / Re;
}