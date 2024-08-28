#include "Parm.H"
#include "Fluxes/Limiters.H"

void Parm::set_defaults()
{
    adiabatic = -1; // Must be set in AmrLevelAdv::read_params()
    epsilon   = 1;
    limiter   = van_leer_limiter;
    for (int i = 0; i < EULER_NCOMP; ++i)
        limiter_indices[i] = 1 + AMREX_SPACEDIM;
}

void Parm::init()
{
    // Calculate derived variables
}