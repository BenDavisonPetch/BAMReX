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

# Gallery
![A supersonic jet.](https://github.com/BenDavisonPetch/BAMReX/blob/main/Doc/img/ss_jet.gif?raw=true)

![The shock-cylinder interaction test using the MUSCL-Hancock method. The bottom half of the image is an experimental schlieren image, and the top half is a computational schlieren image obtained from simulations results.](https://github.com/BenDavisonPetch/BAMReX/blob/main/Doc/img/shockcyl_mhm-1.png?raw=true)

![A comparison of the novel PEEK Implicit-Explicit (IMEX) method with the MUSCL-Hancock method, showing the pressure in the shock-cylinder interaction test.](https://github.com/BenDavisonPetch/BAMReX/blob/main/Doc/img/shockcyl_p_mhm_imexnoeb-1.png?raw=true)

![The strong scaling of the MUSCL-Hancock method implementation on CSD3.](https://github.com/BenDavisonPetch/BAMReX/blob/main/Doc/img/mhm_multires-1.png?raw=true)

![The strong scaling of the novel PESK Implicit-Explicit (IMEX) method on CSD3.](https://github.com/BenDavisonPetch/BAMReX/blob/main/Doc/img/pesk_multires-1.png?raw=true)

![The Sod shock-tube test, with various methods.](https://github.com/BenDavisonPetch/BAMReX/blob/main/Doc/img/POSTER_sod_order2-1.png?raw=true)