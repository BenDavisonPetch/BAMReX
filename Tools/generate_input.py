"""
Generates input files for BAMReX
"""

import argparse
import os

path = os.path.dirname(os.path.realpath(__file__))

name_format = r"{testcase}-{num_method}-{resolution}"

parser = argparse.ArgumentParser(prog = "BAMReX input generator",
                                 description = "Generates input files for BAMReX")
parser.add_argument("-m", "--num_methods", type=str, nargs="+", required=True, help="Numerical methods to generate inputs for")
parser.add_argument("-r", "--resolutions", type=int, nargs="+", required=True, help="Resolutions to generate inputs for")
parser.add_argument("--cfl", type=float, nargs="+", required=True, help="CFLs to use. If fewer values are supplied than the number of numerical methods, the last CFL is used for the remaining methods")
parser.add_argument("-n", "--name", type=str, required=True, help="Test case name")
parser.add_argument("-i", "--problem_input", required=True, type=argparse.FileType("r"), help="Additional text to give the input file. Should contain geometry settings, problem-specific entries, and generally anything not included in the template")
parser.add_argument("-T", "--template", required=False, default=path+"/input_templates/master_template", type=argparse.FileType("r"), help="Template input file to use (optional)")
parser.add_argument("-o", "--output", required=True, type=str, help="Output directory")

args = parser.parse_args()

assert(len(args.cfl) >= 1)
assert(len(args.num_methods) >= 1)
assert(len(args.resolutions) >= 1)

for num_method in args.num_methods:
    if not os.path.isfile(path + "/input_templates/num_method_inputs/" + num_method):
        print("Cannot find input template for numerical method \"" + num_method + "\"")
        exit()

template_text = args.template.read()
problem_input = args.problem_input.read()
input_file_dir = os.path.realpath(args.output)

for i, num_method in enumerate(args.num_methods):
    if i >= len(args.cfl):
        cfl = args.cfl[-1]
    else:
        cfl = args.cfl[i]
    
    with open(path + "/input_templates/num_method_inputs/" + num_method) as nm_file:
        num_method_settings = nm_file.read()
    
    for resolution in args.resolutions:
        resolution_padded = f"{resolution:04}"
        input_file_text = template_text.format(
            problem_input = problem_input,
            resolution = resolution,
            cfl = cfl,
            num_method_settings = num_method_settings,
            num_method = num_method,
            testcase = args.name,
            resolution_padded = resolution_padded
            )
        input_file_name = name_format.format(
            testcase = args.name,
            num_method = num_method, 
            resolution = resolution_padded
            )
        input_file_path = input_file_dir + "/" + input_file_name
        with open(input_file_path, "w") as input_file:
            input_file.write(input_file_text)