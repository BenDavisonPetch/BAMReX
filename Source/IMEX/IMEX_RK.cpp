#include "IMEX_RK.H"

void advance_imex_rk(int level, amrex::IntVect &crse_ratio,
                     const amrex::Real time, const amrex::Geometry &geom,
                     const amrex::MultiFab &statein, amrex::MultiFab &stateout, amrex::MultiFab& pressure,
                     amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> &fluxes,
                     const amrex::Real                              dt,
                     const amrex::Vector<amrex::BCRec>             &domainbcs,
                     const IMEXSettings settings, int verbose,
                     int bottom_verbose)
{
    // if pressure not defined call init_pressure function
    advance_imex_o1(time, geom, statein, stateout, pressure, fluxes, dt, domainbcs, settings);

    // conservative update
    // don forget to testtt
}