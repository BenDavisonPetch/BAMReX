# Overview

The code in `./Source` is independent of initial conditions and problem-specific parameters.
Initial conditions are set in `./Exec/ProblemDir/Prob.H` and `./Exec/ProblemDir/Prob.cpp`.

Problem specific parameters are read from the inputs file in `./Exec/ProblemDir/Prob.H` and `./Exec/ProblemDir/Adv_prob.cpp`. (The header file sets the default and the cpp file overwrites with file input). **These parameters are not currently in use.**

# Building
```shell
$ mkdir build
$ cd build
$ cmake .. -DAMReX_ROOT=/path/to/cmake/built/amrex
$ cd Exec/UniformVelocity
$ make -j
$ ./uniform_velocity inputs
```
## Visualisation
```shell
$ amrvis2d -a output/advection1/advect_plt*
```