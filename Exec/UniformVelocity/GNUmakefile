# This should point to the location of the amrex source code.
AMREX_HOME = ../../../amrex

USE_EB =FALSE
PRECISION  = DOUBLE
PROFILE    = FALSE

# Debug mode compiles without optimisation, so a debugger returns
# accurate line information, without optimising out variables.
# Additionally, additional bounds checking is implemented in AMReX's
# debug mode
DEBUG      = TRUE
#DEBUG      = FALSE

# Make sure AMReX doesn't complain about there being no probin file
AMREX_NO_PROBINIT = TRUE


# Dimensionality is specified at compile time
DIM        = 2
#DIM       = 3

# Compiler can be specified here
COMP	   = gnu

# MPI and OpenMP compatibility must be included at compile time.  OpenMP will also require appropriate pragmas.
USE_MPI    = TRUE
USE_OMP    = FALSE

Bpack   := ./Make.package 
Blocs   := . 

include ../Make.Adv
