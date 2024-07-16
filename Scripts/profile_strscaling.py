from BAMReXTools.profiling import *
import matplotlib.pyplot as plt
import BAMReXTools.plot_defaults
import glob

PROFILE_LOG_DIR = "/home/bendp/practicals/remote_results/cerberus3-profiling/"
NAME_FMT = "M1-{METHOD}-1024.{MPI_NUM_RANKS}.1.?"
METHODS = ["MHM", "PESK-MUSCL-SSP222", "FI-MUSCL-SSP222"]
# MPI_RANKS = [1,2,4,6,8,12,16,20,24,28,32]
MPI_RANKS = [1,2,4,8,16,32]

fastest_runs = {}

total_cputime = 0

for method in METHODS:
    fastest_runs[method] = {}
    for mpi_num_ranks in MPI_RANKS:
        infile_pattern = PROFILE_LOG_DIR + NAME_FMT.format(METHOD=method, MPI_NUM_RANKS=mpi_num_ranks)
        infiles = glob.glob(infile_pattern)
        if (len(infiles) == 0):
            print(f"WARNING: Cannot find input files {infile_pattern}\n")
            continue
        profiledata = [parse_output(infile) for infile in infiles]
        for data in profiledata:
            assert(data.mpi_num_ranks == mpi_num_ranks)
        runtimes = [data.run_time for data in profiledata]
        total_cputime += sum(runtimes) * mpi_num_ranks
        print(f"CPU time for {mpi_num_ranks} = {sum(runtimes) * mpi_num_ranks}")
        iminrun = np.argmin(runtimes)
        if max(runtimes) - min(runtimes) > runtimes[iminrun]*0.05:
            print(f"WARNING: HIGH VARIATION IN RUN TIMES FOR {infile_pattern}:")
            print(runtimes)
            print()
        fastest_runs[method][mpi_num_ranks] = profiledata[iminrun]

print(f"Total CPU time: {total_cputime}")

#
# RUN TIME VS MPI NUM RANKS
#
fig, ax = plt.subplots(1, 2, figsize=(7,3))

for method in METHODS:
    runtimes = [fastest_runs[method][nmpi].run_time for nmpi in MPI_RANKS]
    speedups = [fastest_runs[method][1].run_time/fastest_runs[method][nmpi].run_time for nmpi in MPI_RANKS]
    ax[0].plot(MPI_RANKS, runtimes, label=method)
    ax[1].plot(MPI_RANKS, speedups, label=method)

ax[0].set_xlabel("MPI ranks")
ax[1].set_xlabel("MPI ranks")
ax[0].set_ylabel("Total run time / s")
ax[1].set_ylabel("Speedup")

ax[0].set_xscale("log")
ax[0].set_yscale("log")

# ax[0].legend(fontsize=6)
ax[1].legend(fontsize=6)

ax[1].axline((1,1),slope=1,color="k",linestyle="-.", label="Ideal")


fig.tight_layout(pad=0.8)
fig.savefig("outputs/total_runtime.pdf", bbox_inches="tight")

#
# RUN TIME BREAKDOWN
#

GROUP_OTHER = True

# Uses exclusive times
mhm_profiling_sections = {"Flux computation" : ["compute_HLLC_flux_LR()", "reconstruct_and_half_time_step()", "compute_MUSCL_hancock_flux()", "AmrLevelAdv::flux_computation"],
                          "Time step computation" : ["max_wave_speed()", "AmrLevelAdv::computeNewDt()", "AmrLevelAdv::estTimeStep()", "wave_wave_speed::ReduceRealMax"],
                          "Boundary filling" : ["FabArray::setDomainBndry()", "FillBoundary_nowait()", "FabArray::FillBoundary()", "StateData::FillBoundary(geom)", "AmrLevel::FillPatch()"],
                          "Copies" : ["amrex::Copy()", "FabArray::ParallelCopy_nowait()", "FabArray::mult()", "AmrLevel::FillPatch()", "FillPatchIterator::FillFromLevel0()", "FillPatchSingleLevel", "FabArray::ParallelCopy()", "AmrLevel::FillPatcherFill()"],
                          "Barriers" : ["FillBoundary_finish()", "FabArray::ParallelCopy_finish()"]
                          }
# Uses inclusive times (bc the linear solver output is a nightmare)
imex_profiling_sections = {"Explicit update" : ["advance_MUSCL_rusanov_adv()"],
                           "Linear solve" : ["details::solve_pressure()"],
                           "RK operations" : ["conservative_update(geom, statein, fluxes, stateout1, stateout2, dt1, dt2, N)", "sum_fluxes()", "conservative_update(geom, statein, fluxes, stateout, dt)"],
                           "Solver construction" : ["details::setup_pressure_op()"],
                           "Coefficient computation" : ["update_energy_flux()", "compute_ke()", "pressure_source_vector()", "compute_velocity()", "update_momentum_flux()", "compute_pressure()", "compute_enthalpy()"],
                           "Time step computation" : ["AmrLevelAdv::computeNewDt()"]
                          }
# imex_profiling_sections = {"Explicit update" : ["advance_MUSCL_rusanov_adv()"],
#                            "advance_MUSCL_rusanov_adv::flux_calculation" : ["advance_MUSCL_rusanov_adv::flux_calculation"],
#                            "advance_MUSCL_rusanov_adv::physical_fluxes" : ["advance_MUSCL_rusanov_adv::physical_fluxes"],
#                            "advance_MUSCL_rusanov_adv::reconstruction" : ["advance_MUSCL_rusanov_adv::reconstruction"],
#                            "advance_MUSCL_rusanov_adv::update" : ["advance_MUSCL_rusanov_adv::update"],
#                            "advance_MUSCL_rusanov_adv::flux_copy" : ["advance_MUSCL_rusanov_adv::flux_copy"]
#                           }
profiling_sections = {"MHM" : mhm_profiling_sections,
                      "PESK-MUSCL-SSP222" : imex_profiling_sections,
                      "FI-MUSCL-SSP222" : imex_profiling_sections}
use_excl = {"MHM" : True,
            "PESK-MUSCL-SSP222" : False,
            "FI-MUSCL-SSP222" : False}

fig, ax = plt.subplots(3, 2, figsize=(7, 9), sharey="col")

for imethod, method in enumerate(METHODS):
    print()
    print("===========")
    print(method)
    print("===========")
    covered_functions = set([func for functions in profiling_sections[method].values() for func in functions])

    section_runtimes = {section : {} for section in profiling_sections[method].keys()}
    # "Other" includes all functions that appear in exclusive timing
    # grid but not in covered_functions
    if GROUP_OTHER or not use_excl[method]:
        section_runtimes["Other"] = {}

    for nmpi in MPI_RANKS:
        data = fastest_runs[method][nmpi].excl_data if use_excl[method] else fastest_runs[method][nmpi].incl_data
        for section, functions in profiling_sections[method].items():
            for func in functions:
                if func not in data.index:
                    print(f"WARNING: {func} not found for {nmpi} MPI ranks")
            runtime = sum([data["avg"][func] for func in functions if func in data.index])
            section_runtimes[section][nmpi] = runtime
        if use_excl[method]:
            if GROUP_OTHER:
                section_runtimes["Other"][nmpi] = sum([data["avg"][func] for func in data["name"] if func not in covered_functions])
            else:
                for func in data["name"]:
                    if func not in covered_functions:
                        if func not in section_runtimes.keys():
                            section_runtimes[func] = {}
                        section_runtimes[func][nmpi] = data["avg"][func]
        else:
            section_runtimes["Other"][nmpi] = fastest_runs[method][nmpi].avg_proc_rt - sum([section_runtimes[func][nmpi] for func in profiling_sections[method].keys()])

    for section, runtimes in section_runtimes.items():
        ax[imethod, 0].plot(runtimes.keys(), runtimes.values(), label=section)
        if runtimes[1] == 0:
            ax[imethod, 1].plot([], [], label=section)
            continue
        if 1 in runtimes.keys():
            speedups = [runtimes[1]/runtime for runtime in runtimes.values()]
            ax[imethod, 1].plot(runtimes.keys(), speedups, label=section)

    ax[imethod, 0].set_xlabel("MPI ranks")
    ax[imethod, 0].set_ylabel("Run time / s")
    ax[imethod, 1].set_xlabel("MPI ranks")
    ax[imethod, 1].set_ylabel("Speedup")
    ax[imethod, 0].set_xscale("log")
    ax[imethod, 0].set_yscale("log")
    # ax[imethod, 0].legend(fontsize=6)
    ax[imethod, 1].legend(loc="upper left", fontsize=6)

    ax[imethod, 1].axline((1,1),slope=1,color="k",linestyle="-.", label="Ideal")

text_ys = [1, 0.668, 0.343]
for imethod, method in enumerate(METHODS):
    fig.text(0.5, text_ys[imethod], "\\textbf{" + method + "}", horizontalalignment="center")
fig.tight_layout(pad=0.8)

fig.savefig("outputs/runtime_breakdown.pdf", bbox_inches="tight")

#
# EXPLICIT-ONLY CODE
#

fig, ax = plt.subplots(1, 2, figsize=(7,3))

EXP_ONLY_PROFILE_LOGS="/home/bendp/practicals/remote_results/cerberus3-explicit-scaling/input.{MPI_NUM_RANKS}.1.?"

fastest_exp_runs = {}
for mpi_num_ranks in MPI_RANKS:
    infile_pattern = EXP_ONLY_PROFILE_LOGS.format(MPI_NUM_RANKS=mpi_num_ranks)
    infiles = glob.glob(infile_pattern)
    if (len(infiles) == 0):
        print(f"WARNING: Cannot find input files {infile_pattern}\n")
        continue
    for infile in infiles:
        print("Parsing " + infile)
        parse_output(infile)
    profiledata = [parse_output(infile) for infile in infiles]
    for data in profiledata:
        assert(data.mpi_num_ranks == mpi_num_ranks)
    iminrun = np.argmin([data.run_time for data in profiledata])
    fastest_exp_runs[mpi_num_ranks] = profiledata[iminrun]


mhm_times = [fastest_exp_runs[nmpi].incl_data["avg"]["mhm_timestep"] for nmpi in MPI_RANKS]
rus_times = [fastest_exp_runs[nmpi].incl_data["avg"]["rus_timestep"] for nmpi in MPI_RANKS]
ax[0].plot(MPI_RANKS,
           mhm_times,
           label="MHM time step")
ax[0].plot(MPI_RANKS,
           rus_times,
           label="MUSCL-Rusanov time step")
mhm_speedup = [mhm_times[0] / runtime for runtime in mhm_times]
rus_speedup = [rus_times[0] / runtime for runtime in rus_times]

ax[1].plot(MPI_RANKS,
           mhm_speedup,
           label="MHM time step")
ax[1].plot(MPI_RANKS,
           rus_speedup,
           label="MUSCL-Rusanov time step")

ax[0].set_xlabel("MPI ranks")
ax[0].set_ylabel("Run time / s")
ax[1].set_xlabel("MPI ranks")
ax[1].set_ylabel("Speedup")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
# ax[imethod, 0].legend(fontsize=6)
ax[1].legend(loc="upper left", fontsize=6)

ax[1].axline((1,1),slope=1,color="k",linestyle="-.", label="Ideal")

fig.tight_layout(pad=0.8)

fig.savefig("outputs/runtime_exponly.pdf")

plt.show()