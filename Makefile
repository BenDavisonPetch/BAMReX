AMREX_DEBUG_ROOT=~/bin/amrex-basic-debug
AMREX_MPI_DEBUG_ROOT=~/bin/amrex-mpi-debug
AMREX_RELEASE_ROOT=~/bin/amrex-mpi
CMAKE=cmake

debug:
	rm -rf build/CMakeCache.txt
	$(CMAKE) -S . -B build -DCMAKE_BUILD_TYPE=Custom -DAMReX_ROOT=${AMREX_DEBUG_ROOT}

debug-debug:
	rm -rf build/CMakeCache.txt
	$(CMAKE) -S . -B build -DCMAKE_BUILD_TYPE=Debug -DAMReX_ROOT=${AMREX_DEBUG_ROOT}

debug-mpi:
	rm -rf build/CMakeCache.txt
	$(CMAKE) -S . -B build -DCMAKE_BUILD_TYPE=Custom -DAMReX_ROOT=${AMREX_MPI_DEBUG_ROOT}

release:
	rm -rf build/CMakeCache.txt
	$(CMAKE) -S . -B build -DCMAKE_BUILD_TYPE=Release -DAMReX_ROOT=${AMREX_RELEASE_ROOT}