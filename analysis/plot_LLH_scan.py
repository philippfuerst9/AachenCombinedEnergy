import pickle
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook 
#import matplotlib.colors as colors 
import os
from scipy.interpolate import griddata as scipygrid


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_dir", type=str, required = True, 
                       help    = "Directory where FitRes pickles are saved, e.g. /data/user/pfuerst/DiffuseExtensions/fitdata/AstroScan_truncated_fine")
    
    parser.add_argument("--name", type = str, required = True,
                       help   = "this name + grid bins x grid bins will be the plot name.")
    
    parser.add_argument("--plot_dir", type = str, required = False, 
                       default = "/home/pfuerst/master_thesis/plots/NNMFit/LLHscans",
                       help    = "directory where plots are saved")  
    
    parser.add_argument("--n_grid", type = int, default = 1000,
                        help   = "number of bins for interpolation grid. \
                                  Phi will have 2 more for internal consistency checks")

    args = parser.parse_args()
    return args

def scipy_grid_builder(x, y, values, n_grid_x = 1000, n_grid_y = 1002, max_deltaLLH = 30):
    """interpolates linearly between values on an x,y grid and evaluates it on n_grid_x x n_grid_y points.
    No extrapolation beyond input data
    max_deltaLLH can be set to set a maximum value above which the grid is filled with NaN.
    """
    x_grid = np.linspace(x.min(), x.max(), n_grid_x)
    y_grid   = np.linspace(y.min(),   y.max(),   n_grid_y)

    point_tuples = [[x[i], y[i]] for i in range(len(x))]

    ts_gridded = scipygrid(point_tuples, values, (x_grid[None,:], y_grid[:,None]),
                           method = "linear", rescale = False)

    #kill extreme outliers
    ts_gridded[ts_gridded > max_deltaLLH] = np.NaN
    ts_gridded = np.ma.masked_where(np.isnan(ts_gridded),ts_gridded)
    
    return x_grid, y_grid, ts_gridded
    
if __name__ == '__main__':
    
    args        = parse_arguments()  
    plot_name   = args.name
    pickle_path = args.fit_dir
    plot_dir    = args.plot_dir
    n_grid      = args.n_grid
    
    
    files = os.listdir(pickle_path)
    
    fit_files = []
    best_fit_file = None
    for file in files:
        if file.endswith(".pickle"):
            if file.startswith("FitRes"):
                fit_files.append(os.path.join(pickle_path,file))
            elif file.startswith("Freefit"):
                best_fit_file = (os.path.join(pickle_path,file))

    gammas = []
    phis   = []
    llhs   = []

    for f in fit_files:
        current_fit   = pd.read_pickle(f)
        current_gamma = current_fit["fixed-parameters"]["gamma_astro"]
        current_phi   = current_fit["fixed-parameters"]["astro_norm"]
        current_llh   = current_fit["fit-result"][0][1]

        gammas.append(current_gamma)
        phis.append(current_phi)
        llhs.append(current_llh)

    gammas = np.array(gammas)
    phis   = np.array(phis)
    llhs   = np.array(llhs)
    llhs   = 2 * llhs
    #llhs = -2 * llhs    #usually plot -2deltaLLH so you can instantly read off sigmas
    
    if best_fit_file is not None:
        best_fit = pd.read_pickle(best_fit_file)
        best_gamma = best_fit["fit-result"][1]["gamma_astro"]
        best_phi   = best_fit["fit-result"][1]["astro_norm"]
        best_llh   = best_fit["fit-result"][0][1]
    elif best_fit_file is None:
        best_fit_file = (os.path.join("/data/user/pfuerst/DiffuseExtensions/fitdata/AstroScan/Freefit.pickle"))
        best_fit = pd.read_pickle(best_fit_file)
        best_gamma = best_fit["fit-result"][1]["gamma_astro"]
        best_phi   = best_fit["fit-result"][1]["astro_norm"]
        best_llh   = best_fit["fit-result"][0][1]
        print("No best fit file found in scan! Best fit point will be picked from first run.")
        
    #grid
    #gammas_grid = np.linspace(gammas.min(), gammas.max(), n_grid)
    #phis_grid   = np.linspace(phis.min(),   phis.max(),   n_grid)

    #point_tuples = [[gammas[i], phis[i]] for i in range(len(gammas))]
    gammas_grid, phis_grid, ts_gridded = scipy_grid_builder(gammas, phis, llhs, n_grid_x = n_grid, n_grid_y = n_grid+2)
    #ts_gridded = scipygrid(point_tuples, llhs, (gammas_grid[None,:], phis_grid[:,None]),
                           #method = "linear", rescale = False)

    #kill extreme outliers
    #ts_gridded[ts_gridded > 30] = np.NaN
    #ts_gridded = np.ma.masked_where(np.isnan(ts_gridded),ts_gridded)
    
    
    from scipy import stats
    levels = stats.chi2.ppf(1-2*stats.norm.sf([1,2,3], 0, 1), 2)   #the list is the sigma levels you want to plot
    #sf is survivor-function = 1-cumulative, i.e. the gaussian tail
    #1-2*sf is the area inside 1 2 3 5 sigma region
    #integrate chi2 ndof = 2 to include up to this region to get the contourplot number
    #this works because we can use Wilk's theorem to assume the LLH space to be chi2 distributed.
    
    size = 12
    tick_size = size -2
    fig = plt.figure(dpi=250)
    mesh = plt.pcolormesh(gammas_grid, phis_grid, ts_gridded, 
                       #norm = colors.LogNorm(vmin = ts_gridded.min(), vmax = ts_gridded.max()),
                      cmap = "Blues_r")
    cbar = fig.colorbar(mesh)
    #cbar2 = fig.colorbar(c2,ax = ax2)
    cbar.set_label('$-2\Delta \mathrm{LLH}$',size=size,rotation=90)
    cbar.ax.tick_params(labelsize=tick_size)
    plt.plot(2.0,1.0, linestyle="None", marker="x",color="black", markersize = 10, label="Input Parameters")

    plt.plot(best_gamma, best_phi, linestyle="None", marker="*", color="darkgoldenrod", markersize = 10,label= "Asimov best-fit")


    contour = plt.contour(gammas_grid, phis_grid, ts_gridded, levels, colors="white")
    fmt = {}
    strs = ['1 $\sigma$', '2 $\sigma$', '3 $\sigma$']
    for l, s in zip(contour.levels, strs):
        fmt[l] = s
    plt.clabel(contour, levels, colors = "white", fmt = fmt)
    #contour_1sig = np.transpose(contour.collections[0].get_paths()[0].vertices)
    #contour_2sig = np.transpose(contour.collections[1].get_paths()[0].vertices)
    plt.legend(loc="lower right", fontsize = size)
    plt.xlabel("Spectral Index $\gamma$", fontsize = size)
    plt.ylabel("$\Phi^{astroph.}_{@100 \mathrm{TeV}} $ $/$ $ 10^{-18} \mathrm{GeV}^{-1} \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}$", fontsize = size) # phi-->0 is excluded
    plt.xticks(fontsize = tick_size)
    plt.yticks(fontsize = tick_size)
    
    full_name = plot_name+"_grid_"+str(n_grid)+".png"
    plt.savefig(os.path.join(plot_dir, full_name))
    print("fig saved at "+str(os.path.join(plot_dir, full_name)))
    