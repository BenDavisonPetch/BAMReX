#!/usr/bin/python

"""
Converts curve file output (usually generated by VisIt) into a
format suitable for use with the polygon_union geometry
"""

import argparse
import os

parser = argparse.ArgumentParser(prog="Curve to polygon converter",
                                 description="Converts curve file output (usually generated by VisIt) into a format suitable for use with the polygon_union geometry for BAMReX")
parser.add_argument("input", metavar="input", type=argparse.FileType("r"))
parser.add_argument("output", metavar="output", type=str)

args = parser.parse_args()

input = args.input.read()
if (input[0] == "#"):
    input = input.split("#")[1:]
elif (input[0] == "%"):
    input = input.split("%")[1:]
else:
    raise RuntimeError("Malformed input: file should start with \'#\' or \'%\'")

if (not all([curve.startswith(" curve") for curve in input])):
    raise RuntimeError("Malformed input: each comment should mark a new curve and read \'# curveN\' or \'% curveN\'")

NPoly = len(input)
print(f"Found {NPoly} curves")
input = [curve.split("\n")[1:] for curve in input]
input = [[entry for entry in curve if entry] for curve in input]

lines = [f"ls.polygons = {NPoly}\n"]
for n in range(NPoly):
   NVert = len(input[n])
   lines.append(f"polygon_{n+1}.vertices = {NVert}\n")
   lines.append(f"polygon_{n+1}.fluid_inside = false\n")
   for v in range(NVert):
        lines.append(f"polygon_{n+1}.vertex_{v+1} = {input[n][v]}\n")

with open(args.output, "w") as outfile:
    outfile.writelines(lines)

