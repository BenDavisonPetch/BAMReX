import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import glob

yt.set_log_level("error")

def get_conservative_field_names(dim, has_imex_pressure=False):
    fields = ["x", "density", "mom_x", "energy"]
    if has_imex_pressure: fields.append("IMEX_pressure")
    if dim >= 2:
        fields.append("y")
        fields.append("mom_y")
    if dim >= 3:
        fields.append("z")
        fields.append("mom_z")
    return fields

def get_ray(adiabatic, epsilon, rot, dim, ds, has_imex_pressure=False, ray_start=None, ray_finish=None):
    fields = get_conservative_field_names(dim, has_imex_pressure)
    if not ray_start and not ray_finish:
        if dim == 1:
            ray = ds.ray([ds.domain_left_edge[0].value,0,0], [ds.domain_right_edge[0].value,0,0]).to_dataframe(fields)
        else:
            if rot != 90:
                rot = np.deg2rad(rot)
                ray = ds.ray([0, 0.5 - 0.5*np.tan(rot), 0], [1, 0.5 + 0.5*np.tan(rot), 0]).to_dataframe(fields)
            else:
                rot = np.deg2rad(rot)
                ray = ds.ray([0.5, 0, 0], [0.5, 1, 0]).to_dataframe(fields)
    else:
        assert(ray_start and ray_finish)
        ray = ds.ray(ray_start, ray_finish).to_dataframe(fields)
    size = np.array(ray["x"]).size
    if (dim < 2):
        ray["mom_y"] = np.zeros(size)
        ray["y"] = np.zeros(size)
    if (dim < 3):
       ray["mom_z"] = np.zeros(size)
       ray["z"] = np.zeros(size)

    if not ray_start and not ray_finish:
        if (dim == 1):
            ray["d"] = ray["x"]
        else:
            ray["d"] = (ray["x"] - 0.5)*np.cos(rot) + (ray["y"] - 0.5)*np.sin(rot) + 0.5
    else:
        if dim == 1:
            ray["d"] = ray["x"] - ray_start[0]
        if dim == 2:
            ray["d"] = np.sqrt((ray["x"] - ray_start[0])**2 + (ray["y"] - ray_start[1])**2)
        if dim == 3:
            ray["d"] = np.sqrt((ray["x"] - ray_start[0])**2 + (ray["y"] - ray_start[1])**2 + (ray["z"] - ray_start[2])**2)
    ray["mom"] = np.sqrt(ray["mom_x"] * ray["mom_x"] + ray["mom_y"] * ray["mom_y"] + ray["mom_z"] * ray["mom_z"])
    ray["vel"] = ray["mom"] / ray["density"] * np.sign(ray["mom_x"])
    ray["int_energy"] = (ray["energy"] - 0.5 * epsilon * ray["mom"] * ray["mom"] / ray["density"]) / ray["density"]
    ray["pressure"] = (adiabatic - 1) * ray["density"] * ray["int_energy"]
    return ray


def plot_four(ray, ax, fmt, label, plot_imex_pressure=False, **kwargs):    
    ax[0,0].plot(ray["d"],ray["density"],fmt,label=label, **kwargs)
    ax[0,1].plot(ray["d"],ray["vel"],fmt,label=label, **kwargs)
    # ax[0,1].plot(ray["d"],ray["mom_x"],fmt,label=label, **kwargs)
    if not plot_imex_pressure:
        ax[1,0].plot(ray["d"],ray["pressure"],fmt,label=label, **kwargs)
    else:
        ax[1,0].plot(ray["d"],ray["IMEX_pressure"],fmt,label=label, **kwargs)
    ax[1,1].plot(ray["d"],ray["int_energy"],fmt,label=label, **kwargs)

# Takes an approximate solution and an exact solution with different resolutions and return an array of errors
def get_error(approx_soln, approx_x, exact_soln, exact_x):
    assert(approx_soln.size == approx_x.size)
    assert(exact_soln.size == exact_x.size)
    exact_interp = np.interp(approx_x, exact_x, exact_soln, np.NAN, np.NAN)
    return approx_soln - exact_interp

def get_dx(soln):
    return soln["x"][1] - soln["x"][0]

def error_norm(error, dx, ord):
    if (ord == np.inf):
        return np.abs(error).max()
    return np.power(np.power(np.abs(error), ord).sum() * dx.prod(), 1/ord)

def add_inset(x1,x2,y1,y2,pos,ax):
    axins = ax.inset_axes(
            pos,
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    lines = ax.lines
    for line in lines:
        axins.plot(line.get_xdata(), line.get_ydata(), c=line.get_color(), marker=line.get_marker(), markersize=line.get_markersize(), linestyle=line.get_linestyle(), fillstyle=line.get_fillstyle(), markeredgewidth=line.get_markeredgewidth())
    ax.indicate_inset_zoom(axins, edgecolor="black")

"""
Takes a yt box/region/data container and figures out the dimension of the problem
"""
def get_dim(data):
    assert(data.shape[0] > 1)
    if data.shape[1] == 1:
        assert(data.shape[2] == 1)
        return 1
    if data.shape[2] == 1:
        return 2
    return 3

"""
Takes a dataframe from and adds the primitive variables in
"""
def compute_primitives(data, dim, adiabatic):
    shape = data["x"].shape
    if dim < 2:
        data["y"] = np.zeros(shape)
        data["mom_y"] = np.zeros(shape)
    if dim < 3:
        data["z"] = np.zeros(shape)
        data["mom_z"] = np.zeros(shape)
    data["u"] = data["mom_x"] / data["density"]
    data["v"] = data["mom_y"] / data["density"]
    data["w"] = data["mom_z"] / data["density"]
    data["u_sq"] = data["u"] * data["u"] + data["v"] * data["v"] + data["w"] * data["w"]
    data["int_energy"] = (data["energy"] - 0.5 * data["density"] * data["u_sq"]) / data["density"]
    data["pressure"] = (adiabatic - 1) * (data["energy"] - 0.5 * data["density"] * data["u_sq"])

"""
Takes a yt data object and converts it to a DataFrame.
The fact I have to write this myself is very annoying, but the
to_dataframe() method provided by YT doesn't work for data in >1D...
"""
def to_dataframe(data, has_imex_pressure=False):
    dim = get_dim(data)
    fields = get_conservative_field_names(dim, has_imex_pressure)
    iterables = [range(0, data.shape[0]), range(0, data.shape[1]), range(0, data.shape[2])]
    index = pd.MultiIndex.from_product(iterables, names=["i","j","k"])
    entries = {}
    for field in fields:
        entries[field] = data[field].to_ndarray().flatten()
    return pd.DataFrame(entries, index=index)

"""
Takes the path to a plotfile and returns a DataFrame with primitive variables computed and its dimension
"""
def open_plotfile(plotfile, adiabatic, has_imex_pressure=False):
    ds = yt.load(plotfile)
    data = ds.covering_grid(level=ds.max_level, left_edge=ds.domain_left_edge,dims=ds.domain_dimensions*ds.relative_refinement(0,ds.max_level))
    dim = get_dim(data)
    data = to_dataframe(data, has_imex_pressure)
    compute_primitives(data, dim, adiabatic)
    return data, dim

"""
Computes the error norms for density, velocity and pressure using the plotfiles
provided and returns them in array form along with the cell spacing:

[N, dx, density L1, density L2, density Linf, u L1, u L2, u Linf, v L1, v L2, v Linf,
 w L1, w L2, w Linf, pressure L1, pressure L2, pressure Linf]
"""
def compute_error_norms(numerical_plotfile, exact_plotfile, adiabatic, use_imex_pressure=False, normalize=False):
    # Load files
    num_data, dim = open_plotfile(numerical_plotfile, adiabatic, use_imex_pressure)
    exact_data, exact_dim = open_plotfile(exact_plotfile, adiabatic, use_imex_pressure)
    if not num_data.shape == exact_data.shape:
        print(f"Error! num_data.shape = {num_data.shape}\texact_data.shape = {exact_data.shape}")
    assert(num_data.shape == exact_data.shape)
    assert(dim == exact_dim)

    # Get N (TOTAL NUMBER OF CELLS)
    N = num_data.shape[0]
    # Get dx (remember that the covered grid has constant cell spacing)
    dx = [num_data["x"][1,0,0]-num_data["x"][0,0,0]]
    if (dim >= 2):
        dx.append(num_data["y"][0,1,0]-num_data["y"][0,0,0])
    if (dim == 3):
        dx.append(num_data["z"][0,0,1]-num_data["z"][0,0,0])
    dx = np.array(dx)

    # Now we can start computing errors

    ## The row returned should match:
    ## ["N", "dx", "density L1", "density L2", "density Linf", "u L1", "u L2", "u Linf",
    ##  "v L1", "v L2", "v Linf", "w L1", "w L2", "w Linf",
    ##  "pressure L1", "pressure L2", "pressure Linf"]

    ## Fields to calculate errors for.
    ## Note that even in 1D v and w have been populated with zeros
    if use_imex_pressure:
        fields = ["density", "u", "v", "w", "IMEX_pressure"]
    else:
        fields = ["density", "u", "v", "w", "pressure"]
    ## This is the row we'll add to the dataframe passed to this function
    row = [N, dx[0]]
    for field in fields:
        error = num_data[field] - exact_data[field]
        if normalize:
            exdata = exact_data[field]
            if dim == 1 and field in ["v", "w"]:
                row.append(0)
                row.append(0)
                row.append(0)
            elif dim == 2 and field == "w":
                row.append(0)
                row.append(0)
                row.append(0)
            else:
                row.append(error_norm(error, dx, 1) / error_norm(exdata, dx, 1))
                row.append(error_norm(error, dx, 2) / error_norm(exdata, dx, 2))
                row.append(error_norm(error, dx, np.inf) / error_norm(exdata, dx, np.inf))
        else:
            row.append(error_norm(error, dx, 1))
            row.append(error_norm(error, dx, 2))
            row.append(error_norm(error, dx, np.inf))
    return row

"""
Parameters
----------
test_dir : test case directory with a glob, for instance \"./build/Exec/IsentropicVortex/output/2d/*\"
    where inside that directory is subdirectories \"100\", \"200\", etc
adiabatic : the adiabatic index
exact_soln_plotfile : the name of the plotfile with the exact solution for each test. Should have the
    same resolution as the test itself
"""
def get_convergence_properties(test_dir, adiabatic, exact_soln_plotfile = "pltEXACT_SOLN", use_imex_pressure=False):
    SUBRUN_DIRS = sorted(glob.glob(test_dir))
    if (len(SUBRUN_DIRS) == 0):
        print (f"Cannot find results with pattern {test_dir}")
        return
    SUBRUN_FINAL_OUTPUTS = [sorted(glob.glob(SUBRUN_DIR+"/plt[!E]*"))[-1] for SUBRUN_DIR in SUBRUN_DIRS]
    EXACT_SOLUTIONS = [SUBRUN_DIR + "/" + exact_soln_plotfile for SUBRUN_DIR in SUBRUN_DIRS]
    error_norms = []
    for i in range(len(SUBRUN_FINAL_OUTPUTS)):
        error_norms.append(compute_error_norms(SUBRUN_FINAL_OUTPUTS[i], EXACT_SOLUTIONS[i], adiabatic, use_imex_pressure))
    df = pd.DataFrame(columns=["N", "dx", "density L1", "density L2", "density Linf",
                               "u L1", "u L2", "u Linf", "v L1", "v L2", "v Linf",
                               "w L1", "w L2", "w Linf", "pressure L1", "pressure L2",
                               "pressure Linf"], data=error_norms)

    fields = ["density", "u", "v", "w", "pressure"]
    for ifield, field in enumerate(fields):
        df.insert(6 * ifield + 2, field + " L1 order", np.log(df[field + " L1"] / np.roll(df[field + " L1"],1)) / np.log(df["dx"] / np.roll(df["dx"], 1)))
        df.insert(6 * ifield + 4, field + " L2 order", np.log(df[field + " L2"] / np.roll(df[field + " L2"], 1)) / np.log(df["dx"] / np.roll(df["dx"], 1)))
        df.insert(6 * ifield + 6, field + " Linf order", np.log(df[field + " Linf"] / np.roll(df[field + " Linf"], 1)) / np.log(df["dx"] / np.roll(df["dx"], 1)))

    return df