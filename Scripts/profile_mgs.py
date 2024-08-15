from BAMReXTools.profiling import *
import matplotlib.pyplot as plt
import BAMReXTools.plot_defaults
import glob
import matplotlib.ticker as ticker

#
# MHM
#
PROFILE_LOG_DIR = "/home/bendp/practicals/remote_results/cerberus3-mgs-mhm/"
NAME_FMT = "M1-MHM-{MAX_GRID_SIZE}-1024.32.1.?"
MAX_GRID_SIZES = [8, 16, 32, 64, 128]

fastest_runs_mhm = []
for mgs in MAX_GRID_SIZES:
    infile_pattern = PROFILE_LOG_DIR + NAME_FMT.format(MAX_GRID_SIZE=mgs)
    infiles = glob.glob(infile_pattern)
    if (len(infiles) == 0):
        print(f"WARNING: Cannot find input files {infile_pattern}\n")
        continue
    profiledata = [parse_output(infile) for infile in infiles]
    for data in profiledata:
        assert(data.mpi_num_ranks == 32)
    iminrun = np.argmin([data.run_time for data in profiledata])
    fastest_runs_mhm.append(profiledata[iminrun])

fig, ax = plt.subplots(1, 1, figsize=(3.5,3))

runtimes_mhm = [run.run_time for run in fastest_runs_mhm]
ax.plot(MAX_GRID_SIZES, runtimes_mhm, label="MHM")

#
# IMEX
#
PROFILE_LOG_DIR = "/home/bendp/practicals/remote_results/cerberus2-mgs-pesk/"
NAME_FMT = "M1-PESK-MUSCL-SSP222-{MAX_GRID_SIZE}-1024.32.1.?"
MAX_GRID_SIZES = [8, 16, 32, 64, 128]

fastest_runs_pesk = []
for mgs in MAX_GRID_SIZES:
    infile_pattern = PROFILE_LOG_DIR + NAME_FMT.format(MAX_GRID_SIZE=mgs)
    infiles = glob.glob(infile_pattern)
    if (len(infiles) == 0):
        print(f"WARNING: Cannot find input files {infile_pattern}\n")
        continue
    profiledata = [parse_output(infile) for infile in infiles]
    for data in profiledata:
        assert(data.mpi_num_ranks == 32)
    iminrun = np.argmin([data.run_time for data in profiledata])
    fastest_runs_pesk.append(profiledata[iminrun])

runtimes_pesk = [run.run_time for run in fastest_runs_pesk]
ax.plot(MAX_GRID_SIZES, runtimes_pesk, label="PESK-SSP222")

ax.set_xlabel("Max grid size")
ax.set_ylabel("Run time / s")

ax.set_yscale("log")
ax.set_ylim(ax.get_ylim()[0], 1100)

ax.legend()

fig.tight_layout(pad = 0.8)

fig.savefig("outputs/profiling/mgs.pdf", bbox_inches="tight")

plt.show()
