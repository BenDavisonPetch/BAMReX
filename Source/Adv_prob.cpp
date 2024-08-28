#include <AMReX_REAL.H>

extern "C"
{
    void amrex_probinit(const int * /*init*/, const int * /*name*/,
                        const int * /*namelen*/,
                        const amrex::Real * /*problo*/,
                        const amrex::Real * /*probhi*/)
    {
    }
}
