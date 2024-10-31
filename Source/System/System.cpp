#include "System.H"
#include "FluidSystem.H"
#include "Fluxes/Limiters.H"
#include "PlasmaSystem.H"
#include <AMReX_Print.H>

using namespace amrex;

System::~System() {}

System *System::NewSystem(ParmParse &pp)
{
    enum SystemType a_sys;
    pp.get_enum_case_insensitive("system", a_sys);
    System *sys;
    switch (a_sys)
    {
    case SystemType::fluid:
        sys         = new FluidSystem;
        sys->system = SystemType::fluid;
        break;
    case SystemType::plasma:
        sys         = new PlasmaSystem;
        sys->system = SystemType::plasma;
        break;
    default: sys = nullptr;
    }
    return sys;
}