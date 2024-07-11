import numpy as np
from pandas import DataFrame
import re

class ProfileData:
    def __init__(self, run_time, mpi_num_ranks, min_proc_rt, avg_proc_rt, max_proc_rt, excl_data, incl_data):
        self.run_time = run_time
        self.mpi_num_ranks = mpi_num_ranks
        self.min_proc_rt = min_proc_rt
        self.avg_proc_rt = avg_proc_rt
        self.max_proc_rt = max_proc_rt
        self.excl_data = excl_data
        self.incl_data = incl_data

def parse_output(output_file):
    lines = open(output_file).readlines()
    if (lines[1].startswith("MPI initialized with ")):
        mpi_num_ranks = int(re.search(r"\d+", lines[1]).group())
    else:
        mpi_num_ranks = 1
    lines = strip_output(lines)
    assert(lines[0].startswith("Run time = "))
    run_time = float(lines[0][len("Run time = "):])
    min_proc_rt, avg_proc_rt, max_proc_rt = [float(x) for x in lines[1][len("TinyProfiler total time across processes [min...avg...max]: "):].split("...")]
    hline_indices = lines_starting_with("-------------------------------", lines)
    excl_data = construct_df(lines[hline_indices[1]+1:hline_indices[2]])
    incl_data = construct_df(lines[hline_indices[4]+1:hline_indices[5]])
    return ProfileData(run_time, mpi_num_ranks, min_proc_rt, avg_proc_rt, max_proc_rt, excl_data, incl_data)

def construct_df(lines):
    lines = [line.split() for line in lines]

    # Fix the unnecessary splitting of function names with whitespace
    for i in range(len(lines)):
        if len(lines[i]) > 6:
            lines[i] = [" ".join(lines[i][:-5])] + lines[i][-5:]
    
    lines = [[line[0],int(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5][:-1])] for line in lines]
    function_names = [line[0] for line in lines]

    return DataFrame(lines, columns=["name","n_calls","min","avg","max","max%"], index=function_names)

def lines_starting_with(phrase, lines):
    indices = [i for i, line in enumerate(lines) if line.startswith(phrase)]
    return indices

def strip_output(lines):
    lines = [line.strip() for line in lines if not line.isspace()]
    rt_indices = lines_starting_with("Run time", lines)
    assert(len(rt_indices) == 1)
    lines = lines[rt_indices[0]:]
    hline_indices = lines_starting_with("-------------------------------", lines)
    assert(len(hline_indices) >= 6)
    lines = lines[:hline_indices[5]+1]
    return lines


if __name__ == "__main__":
    output_file = "/home/bendp/practicals/remote_results/cerberus3-profiling/M1-PESK-MUSCL-SSP222-1024.28.1.2"
    print(parse_output(output_file))

