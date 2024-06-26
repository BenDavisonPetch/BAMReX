#ifndef AMREX_MLABECLAP_K_H_
#define AMREX_MLABECLAP_K_H_
#include <AMReX_Config.H>

#include <AMReX_FArrayBox.H>

#if (AMREX_SPACEDIM == 1)
#include "MLALapGrad_1D_K.H"
#elif (AMREX_SPACEDIM == 2)
#include "MLALapGrad_2D_K.H"
#else
#include "MLALapGrad_3D_K.H"
#endif

#endif
