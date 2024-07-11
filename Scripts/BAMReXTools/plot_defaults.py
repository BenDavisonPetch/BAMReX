import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

mpl.rcParams["font.family"] = "serif"
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams["text.usetex"] = True
mpl.rcParams["lines.linewidth"] = 1
mpl.rcParams["lines.markersize"] = 3.5
mpl.rcParams["xtick.direction"] = "out"
mpl.rcParams["ytick.direction"] = "out"
mpl.rcParams["xtick.major.size"] = 2
mpl.rcParams["ytick.major.size"] = 2
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["legend.fontsize"] = "small"
mpl.rcParams["axes.titlesize"] = "medium"

fmts =     ["go", "bD", "r+", "k^", "yD", "ms", "cp", "gd", "bv", "r1"]
colors = ["g", "b", "r", "k", "y", "m", "c", "g", "b", "r"]
markers = ["o", "D", "+", "^", "D", "s", "p", "d", "v", "1"]
marker_sizes = [3, 2.5, 4, 4, 4, 4, 3, 3, 3, 3]
marker_scaling = 1.5
default_cycler = (cycler(color=colors) +
                  cycler(marker=markers) + 
                  cycler(markersize=[marker_size*marker_scaling for marker_size in marker_sizes]))

mpl.rcParams["axes.prop_cycle"] = default_cycler
mpl.rcParams["markers.fillstyle"] = "none"
mpl.rcParams["lines.linestyle"] = "none"
mpl.rcParams["lines.markeredgewidth"] = 0.3*marker_scaling