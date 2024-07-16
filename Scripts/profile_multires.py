from BAMReXTools.profiling import *
import matplotlib.pyplot as plt
import BAMReXTools.plot_defaults
import glob

PROFILE_LOG_DIR = "/home/bendp/practicals/remote_results/cerberus3-multires/"
NAME_FMT = "M1-{METHOD}-{RESOLUTION}.{MPI_NUM_RANKS}.1.?"
METHODS = ["MHM", "PESK-MUSCL-SSP222"]
RESOLUTIONS = [128, 256, 512, 1024]
MPI_RANKS = [1,2,4,8,16,32]

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

fig, ax = plt.subplots(2, 2, figsize=(7,6))
for method in METHODS:
    for res in RESOLUTIONS:
        if len(fastest_runs[method][res]) == 0: continue
        nranks = fastest_runs[method][res].keys()
        runtimes = [fastest_runs[method][res][nmpi] for nmpi in nranks]
        speedups = [fastest_runs[method][res][1]/fastest_runs[method][res][nmpi] for nmpi in nranks]
        efficiencies = [speedups[i]/nmpi for i, nmpi in enumerate(nranks)]
        x = [res*res/nmpi for nmpi in nranks]
        # x = [res/nmpi for nmpi in nranks]
        # x = nranks

        N_timesteps = [fastest_data[method][res][nmpi].incl_data["n_calls"]["Amr::coarseTimeStep()"] for nmpi in nranks]
        print(f"Res = {res}, Ntimesteps = {N_timesteps}")
        # x = [res*res*N_timesteps[i]/nmpi for i, nmpi in enumerate(nranks)]

        ax[0,0].plot(nranks, runtimes, label=f"{method} {res}")
        ax[0,1].plot(nranks, speedups, label=f"{method} {res}")
        # ax[0,1].plot(nranks, efficiencies, label=f"{method} {res}")
        ax[1,0].plot(x, runtimes, label=f"{method} {res}")
        ax[1,1].plot(x, speedups, label=f"{method} {res}")
        # ax[1,1].plot(x, efficiencies, label=f"{method} {res}")


ax[0, 0].set_xlabel("MPI ranks")
ax[0, 1].set_xlabel("MPI ranks")
ax[0, 0].set_ylabel("Total run time / s")
ax[0, 1].set_ylabel("Speedup")

# ax[1, 0].set_xlabel("N cells * N time steps / MPI ranks")
# ax[1, 1].set_xlabel("N cells * N time steps / MPI ranks")
ax[1, 0].set_xlabel("N cells / MPI ranks")
ax[1, 1].set_xlabel("N cells / MPI ranks")
ax[1, 0].set_ylabel("Total run time / s")
ax[1, 1].set_ylabel("Speedup")

ax[0,0].set_xscale("log")
ax[0,0].set_yscale("log")
ax[0,1].set_xscale("log")
ax[0,1].set_yscale("log")
ax[1,0].set_xscale("log")
ax[1,0].set_yscale("log")
ax[1,1].set_xscale("log")
ax[1,1].set_yscale("log")

ax[0,1].legend(fontsize=6)

xlim = ax[1,1].get_xlim()
ylim = ax[1,1].get_ylim()
# grid_xs = [1e3 * (10**(0.5*i)) for i in range(12)]
# y_int = ylim[0]

grid_xs = [res*res for res in RESOLUTIONS]
y_int = 1

# for getting second point
x_off = 1000
for grid_x in grid_xs:
    if grid_x < xlim[1]:
        p1 = [grid_x,y_int]
        p2 = [grid_x - x_off, grid_x*y_int/(grid_x-x_off)]
    else:
        p1 = [xlim[1], grid_x*y_int/(xlim[1])]
        p2 = [xlim[1]-x_off, grid_x*y_int/(xlim[1]-x_off)]
    ax[1,1].axline(p1, p2,ls=":",color="grey",zorder=-10,alpha=0.5)

# ax[1,1].axline([65536, 1], [65536, 2], ls="-.", color="r", zorder=-5, alpha=0.5)
# ax[1,1].axline([16384, 1], [16384, 2], ls="-.", color="b", zorder=-5, alpha=0.5)

ax[1,1].set_xlim(xlim)
ax[1,1].set_ylim(ylim)

fig.tight_layout(pad=0.8)

plt.show()