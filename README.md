# Overview

The code in `./Source` is independent of initial conditions and problem-specific parameters.
Initial conditions are set in `./Exec/ProblemDir/Prob.H` and `./Exec/ProblemDir/Prob.cpp`.

Problem specific parameters and numerical method choice are specified in the input files.

# Building and Running
```shell
$ mkdir build
$ cd build
$ cmake .. -DAMReX_ROOT=/path/to/cmake/built/amrex
$ cd Exec/RiemannProblem
$ make -j
$ make run_lax #OR
$ ./riemann_prob inputs/desired-input-file
```