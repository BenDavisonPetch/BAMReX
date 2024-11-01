BUILD_DIR=build
AMREX_DEBUG_ROOT=~/bin/amrex-basic-debug
AMREX_MPI_DEBUG_ROOT=~/bin/amrex-mpi-debug
AMREX_RELEASE_ROOT=~/bin/amrex-mpi
AMREX_EB_DEBUG_ROOT=~/bin/amrex-EB-debug
AMREX_MPI_EB_DEBUG_ROOT=~/bin/amrex-EB-mpi-debug
AMREX_EB_RELEASE_ROOT=~/bin/amrex-EB-mpi
AMREX_GPU_DEBUG_ROOT=~/bin/amrex-gpu-debug
AMREX_GPU_RELEASE_ROOT=~/bin/amrex-gpu-release
CMAKE=cmake

debug:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Custom -DAMReX_ROOT=${AMREX_DEBUG_ROOT}

debug-debug:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DAMReX_ROOT=${AMREX_DEBUG_ROOT}

debug-mpi:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Custom -DAMReX_ROOT=${AMREX_MPI_DEBUG_ROOT}

release:
	rm -rf build/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DAMReX_ROOT=${AMREX_RELEASE_ROOT}

debug-eb:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Custom -DAMReX_ROOT=${AMREX_EB_DEBUG_ROOT}

debug-debug-eb:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DAMReX_ROOT=${AMREX_EB_DEBUG_ROOT}

debug-mpi-eb:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Custom -DAMReX_ROOT=${AMREX_MPI_EB_DEBUG_ROOT}

release-eb:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DAMReX_ROOT=${AMREX_EB_RELEASE_ROOT}

debug-gpu:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Debug -DAMReX_ROOT=${AMREX_GPU_DEBUG_ROOT}

release-gpu:
	rm -rf ${BUILD_DIR}/CMakeCache.txt
	$(CMAKE) -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DAMReX_ROOT=${AMREX_GPU_RELEASE_ROOT}