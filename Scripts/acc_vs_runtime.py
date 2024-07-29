from BAMReXTools.profiling import *
from BAMReXTools.plot_util import *
import matplotlib.pyplot as plt
import BAMReXTools.plot_defaults
import glob

# structure works as runs[nprocs][method] = [res1,res2,...]
#  a method key of "IMEX" gets replaced by all the IMEX schemes
#  a method key of "exp" gets replaced by all the explicit
runs = {8 : {"IMEX" : [128]},
        32 : {"IMEX" : [256]},
        128 : {"IMEX" : [512]},
        256 : {"exp" : [128]},
        512 : {"IMEX" : [1024]},
        1024 : {"exp" : [256, 512, 1024]}}

adiabatic = 1.4

test_names = ["M1", "M1e-2"]

log_dir = "/home/bendp/practicals/remote_results/csd3-accres-logs-extra"
output_dir = "/home/bendp/practicals/remote_results/csd3-accres-output-extra/"

output_fmt = output_dir+"{method}/{testcase}/{res:04d}/plt*"
exact_fmt = output_dir+"{method}/{testcase}/{res:04d}/plt00000"
log_file_fmt = log_dir+"/{testcase}-{method}-{res}.{_N_MPI_TASKS}.0"

imex_methods = ["PESK-MUSCL-SSP222", "PEEK-MUSCL-SSP222",
               "FI-MUSCL-SSP222", "PESK-SP111",
               "PEEK-SP111", "FI-SP111"]
exp_methods = ["MHM", "HLLC"]
all_methods = imex_methods + exp_methods

class RunData:
    """
    error_norms should be
    ["N", "dx", "density L1", "density L2", "density Linf",
                               "u L1", "u L2", "u Linf", "v L1", "v L2", "v Linf",
                               "w L1", "w L2", "w Linf", "pressure L1", "pressure L2",
                               "pressure Linf"]
    """
    def __init__(self, error_norms, profile_data):
        assert(len(error_norms) == 17)
        self.N = error_norms[0]
        self.dx = error_norms[1]
        self.density_L1 = error_norms[2]
        self.density_L2 = error_norms[3]
        self.density_Linf = error_norms[4]
        self.u_L1 = error_norms[5]
        self.u_L2 = error_norms[6]
        self.u_Linf = error_norms[7]
        self.v_L1 = error_norms[8]
        self.v_L2 = error_norms[9]
        self.v_Linf = error_norms[10]
        self.w_L1 = error_norms[11]
        self.w_L2 = error_norms[12]
        self.w_Linf = error_norms[13]
        self.pressure_L1 = error_norms[14]
        self.pressure_L2 = error_norms[15]
        self.pressure_Linf = error_norms[16]
        self.profile_data = profile_data
        
# formatted as results[testcase][method] = [RunData,...]
results = {}
## init results
for testcase in test_names:
    results[testcase] = {}
    for method in all_methods:
        results[testcase][method] = []

#
# Process data
for testcase in test_names:
    for nmpi in runs.keys():
        for method_type in runs[nmpi].keys():
            methods = imex_methods if method_type == "IMEX" else exp_methods
            for method in methods:
                for resolution in runs[nmpi][method_type]:
                    # Get error data
                    output_glob = output_fmt.format(method=method, testcase=testcase,res=resolution)
                    plotfiles = glob.glob(output_glob)
                    if (len(plotfiles) == 0):
                        print(f"Cannot find plotfiles {output_glob}!")
                        continue
                        # raise ValueError(f"Cannot find plotfiles {output_glob}")
                    plotfiles = sorted([plotfile for plotfile in plotfiles if ".old." not in plotfile])
                    final_output = plotfiles[-1]
                    exact_plotfile = exact_fmt.format(method=method, testcase=testcase,res=resolution)
                    
                    error_norms = compute_error_norms(final_output, exact_plotfile, adiabatic, method_type == "IMEX", True)

                    if (error_norms[2] == 0):
                        print(f"Error norms are suspiciously zero for {final_output}")
                        exit()

                    # Get profiling data
                    logfile = log_file_fmt.format(method=method, testcase=testcase, res=resolution, _N_MPI_TASKS=nmpi)
                    try:
                        profile_data = parse_output(logfile)
                    except FileNotFoundError:
                        print(f"WARNING: Found output data for {method}-{resolution}.{nmpi} but could not find log file {logfile}!")
                        continue
                    assert(profile_data.mpi_num_ranks == nmpi)

                    # add results
                    results[testcase][method].append(RunData(error_norms, profile_data))

#
# PLOT
#
plot_combos = [{m : test_names for m in ["FI-SP111", "PESK-SP111", "PEEK-SP111"]},
               {m : test_names for m in ["FI-MUSCL-SSP222", "PESK-MUSCL-SSP222", "PEEK-MUSCL-SSP222"]},
               {m : test_names for m in ["FI-SP111", "FI-MUSCL-SSP222"]},
               {m : test_names for m in ["PESK-SP111", "PESK-MUSCL-SSP222"]},
               {m : test_names for m in ["PEEK-SP111", "PEEK-MUSCL-SSP222"]},
               {m : test_names for m in ["MHM", "HLLC"]},
               {"MHM" : test_names, "PEEK-MUSCL-SSP222" : ["M1"]}]
for plot_combo in plot_combos:
    fig, ax = plt.subplots(1, 1, figsize=(7*0.75,4*0.75))

    for method in plot_combo.keys():
        for testcase in plot_combo[method]:
            label = f"{method}, {testcase}"
            profile_datas = [res.profile_data for res in results[testcase][method]]
            run_times = [data.run_time for data in profile_datas]
            errors = [res.pressure_L1 for res in results[testcase][method]]
            ax.plot(errors, run_times, label=label)

    ax.set_xlabel(r"$L_{rel}^1(p)$")
    ax.set_ylabel("Run time / s")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")

plt.show()