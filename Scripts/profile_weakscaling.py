from BAMReXTools.profiling import *
import matplotlib.pyplot as plt
import BAMReXTools.plot_defaults
import glob

PROFILE_LOG_DIR = "/home/bendp/practicals/remote_results/cerberus3-multires/"
NAME_FMT = "M1-{METHOD}-{RESOLUTION}.{MPI_NUM_RANKS}.1.?"
METHODS = ["MHM", "PESK-MUSCL-SSP222"]
# RESOLUTIONS = [256, 512, 1024]
# MPI_RANKS = [2, 8, 32]
RESOLUTIONS = [128, 256]
MPI_RANKS = [1, 8]

fastest_runs = {}
fastest_data = {}

for method in METHODS:
    fastest_runs[method] = {}
    fastest_data[method] = {}
    for res in RESOLUTIONS:
        fastest_runs[method][res] = {}
        fastest_data[method][res] = {}
        for mpi_num_ranks in MPI_RANKS:
            infile_pattern = PROFILE_LOG_DIR + NAME_FMT.format(METHOD=method, MPI_NUM_RANKS=mpi_num_ranks, RESOLUTION=res)
            infiles = glob.glob(infile_pattern)
            if (len(infiles) == 0):
                print(f"WARNING: Cannot find input files {infile_pattern}\n")
                continue
            profiledata = [parse_output(infile) for infile in infiles]
            for data in profiledata:
                assert(data.mpi_num_ranks == mpi_num_ranks)
            iminrun = np.argmin([data.run_time for data in profiledata])
            fastest_runs[method][res][mpi_num_ranks] = profiledata[iminrun].run_time
            fastest_data[method][res][mpi_num_ranks] = profiledata[iminrun]

fig, ax = plt.subplots(1, 1, figsize=(3.5,3))
for method in METHODS:
    runtimes = [fastest_runs[method][RESOLUTIONS[i]][MPI_RANKS[i]] for i in range(len(MPI_RANKS))]
    ax.plot(MPI_RANKS, runtimes, label=method)

ax.set_xlabel("MPI ranks")
ax.set_ylabel("Run time / s")
ax.legend()

fig.tight_layout(pad=0.8)

plt.show()