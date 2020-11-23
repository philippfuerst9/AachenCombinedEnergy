#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/icetray-start
#METAPROJECT /home/pfuerst/i3_software/combo/build


import numpy as np
#import Segmented_Muon_Energy_jstettner as sme
from icecube import dataio, dataclasses
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import matplotlib.cbook as cbook #?
import matplotlib.colors as colors #?
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats
import imp
sme  = imp.load_source('stettner_module', '/home/pfuerst/master_thesis/software/reco_analysis/Segmented_Muon_Energy_jstettner.py')

def create_log_bin_edges(Real_Min, Real_Max, nbins):
    return np.logspace(np.log10(Real_Min), np.log10(Real_Max), nbins+2)

def create_linear_bin_edges(Real_Min, Real_Max, nbins):
    return np.linspace(Real_Min, Real_Max, nbins+2) #now nbins is number of bins, not number of edges.

def get_log_bin_centers(log_bin_edges):
    return 10**(np.log10(log_bin_edges)[:-1]+np.diff(np.log10(log_bin_edges))/2.)

def get_linear_bin_centers(linear_bin_edges):
    return linear_bin_edges[:-1]+0.5*np.diff(linear_bin_edges)

def get_overall_min_max(*args):
    """Gets overall min and max from several arrays.
    """
    mins = []
    maxs = []
    for arg in args:
        currentmin = min(arg)
        currentmax = max(arg)
        mins.append(currentmin)
        maxs.append(currentmax)
    themin = min(mins)
    themax = max(maxs)
    return themin, themax

def get_log_sigma_quantile(upper, lower):
    """takes +/- 0.34 quantiles
    returns diff in log10
    """
    log_upper = np.log10(upper)
    log_lower = np.log10(lower)
    return log_upper-log_lower

def EnergyResolutionLogGaussY(E_true, E_guess, E_bins_true, E_bins_reco):
    '''
    INPUT:
    MC True Energies, 1darray, Reconstructed Energies, 1darray, 
    Bin edges for both axes, should be np.logspace(minE, maxE)
    
    OUTPUT:
    Lists containing the results of the gaussian fits:
    List of means, stds, norms(absolute scale), covariance matrices
    XvalList = bin centers of the former Reconstructed Energies 
    YvalList = total event counts in the Bins used for 1 fit, i.e. in one Y-slice.
    TrueEnergies = the MC Truth bin center of the Y-slice
    '''
    
    histogram, E_xedges, E_yedges = np.histogram2d(E_true, E_guess, bins = (E_bins_true, E_bins_reco))
    FitGauss = lambda x, loc, scale, norm : norm*scipy.stats.norm.pdf(x,loc,scale)
    means = []
    bin_stds = []
    norms  = []
    errlist = []
    XvalList = []
    YvalList = []
    TrueEnergies = get_log_bin_centers(E_xedges)        #bin centers on the MC truth energy axis.
    for i in xrange(np.shape(histogram)[0]): 
        
        Hist_Slice = histogram[i][:]
        xvals = np.log10(get_log_bin_centers(E_yedges)) #go to logspace for even spacing
        yvals = Hist_Slice                              #maybe np.log10(Hist_Slice)
        
        pcov_failed = np.empty((3,3))                   #empty pcov matrix to append
        pcov_failed[:] = np.NaN                         #should the fit fail
        
        if sum(i>0 for i in Hist_Slice)>3:              #we fit a 3 param function so really should have 
                                                        #4 data points to say something.
            try:
                ppar, pcov = curve_fit(FitGauss, xvals, yvals)
                means.append(ppar[0])
                bin_stds.append(ppar[1])
                norms.append(ppar[2])
                errlist.append(pcov)

            except:
                means.append(np.NaN)
                bin_stds.append(np.NaN)
                norms.append(np.NaN)
                errlist.append(pcov_failed)

        else:
            means.append(np.NaN)
            bin_stds.append(np.NaN)
            norms.append(np.NaN)
            errlist.append(pcov_failed)
            
        XvalList.append(xvals)
        YvalList.append(yvals)
        

    return np.array(means), np.array(bin_stds),np.array(norms), np.array(errlist), np.array(XvalList), np.array(YvalList), TrueEnergies


def gauss_fit_along_y_hist2d(x_list, y_list, x_bin_edges, y_bin_edges, x_space = "linear", y_space = "log"):
    '''
    INPUT:
    xlist: 1darray, ylist: 1darray, 
    x_bins, y_bins: hist bin edges, if in logspace, set x_space/ y_space to "log"
    
    OUTPUT:
    returns 1d arrays of means, stds, norms, error-matrices, 
    x-values for every sliced 1d hist, y-values for every sliced 1d hist and
    the original 2dhist x bin centers. 
    
    !!! ATTENTION !!!
    if y_space == "log", the output means and stds will live in log10-space.
    calculate 10**means to get real numbers.
    '''
    
    FitGauss = lambda x, loc, scale, norm : norm*scipy.stats.norm.pdf(x,loc,scale)
    histogram, x_edges, y_edges = np.histogram2d(x_list, y_list, bins = (x_bin_edges,y_bin_edges))
    
    means = []
    stds  = []
    norms = []
    errs  = []
    slice_x_vals_list = []
    slice_y_vals_list = []

    if   x_space == "linear":
        x_bin_centers = get_linear_bin_centers(x_edges)
        
    elif x_space == "log":
        x_bin_centers = get_log_bin_centers(x_edges)
        
    else:
        raise ValueError("x_space must be 'linear' or 'log'")

    if   y_space == "linear":
        y_bin_centers = get_linear_bin_centers(y_edges)
        
    elif y_space == "log":
        y_bin_centers = get_log_bin_centers(y_edges)
        
    else:
        raise ValueError("x_space must be 'linear' or 'log'")


    for i in xrange(np.shape(histogram)[0]): 
        
        hist_slice = histogram[i][:]
        if y_space == "linear":
            slice_x_vals = y_bin_centers 
        elif y_space == "log":
            slice_x_vals = np.log10(y_bin_centers) #even spacing in logspace
            
            
        slice_y_vals = hist_slice                       #maybe np.log10(Hist_Slice)
        
        pcov_failed = np.empty((3,3))                   #empty pcov matrix to append
        pcov_failed[:] = np.NaN                         #should the fit fail
        
        if sum(i>0 for i in hist_slice)>3:              #we fit a 3 param function so really should have 
                                                        #4 data points to say something.
            try:
                ppar, pcov = curve_fit(FitGauss, slice_x_vals, slice_y_vals)
                means.append(ppar[0])
                stds.append(ppar[1])
                norms.append(ppar[2])
                errs.append(pcov)

            except:
                means.append(np.NaN)
                stds.append(np.NaN)
                norms.append(np.NaN)
                errs.append(pcov_failed)

        else:
            means.append(np.NaN)
            stds.append(np.NaN)
            norms.append(np.NaN)
            errs.append(pcov_failed)
            
        slice_x_vals_list.append(slice_x_vals)
        slice_y_vals_list.append(slice_y_vals)

    out_dict = {
    "means"            : np.array(means),
    "stds"             : np.array(stds),
    "norms"            : np.array(norms),
    "covs"             : np.array(errs),
    "slice_x_val_list" : np.array(slice_x_vals_list),
    "slice_y_val_list" : np.array(slice_y_vals_list),
    "orig_x_bin_centers"    : np.array(x_bin_centers),
    "orig_hist2d"      : histogram
    }
        
    return out_dict

"""
def median_and_sigma_y(x_list, y_list, x_bin_edges, y_bin_edges, x_space = "linear", y_space = "log"):
    '''
    Calculate median and one sigma environments for 2d histogram. 
    '''
    histogram, x_edges, y_edges = np.histogram2d(x_list, y_list, bins = (x_bin_edges,y_bin_edges))

    out_dict = {
    "median"              :  ,
    "median_minus_sigma"  :  ,
    "median_plus_sigma"   :  ,
    "slice_x_val_list"    :  ,
    "slice_y_val_list"    :  ,
    "orig_x_bin_centers"  :  ,
    "orig_hist2d"         :  
    }
        
    return out_dict
"""
   
def Hist2d_PercentResiduals(TrueE, RecoE, Ebins = 20, Rbins = 20):
    absolute_diff = (np.log10(TrueE)-np.log10(RecoE)) #= log10(TrueE/RecoE), i.e = 0 if TrueE = RecoE, =1 if RecoE = 10% of TrueE
    #relative_diff = absolute_diff/TrueE
    lower_E = min(TrueE)
    higher_E= max(TrueE)
    lower_Residue = min(absolute_diff)
    higher_Residue = max(absolute_diff)
    hist = np.histogram2d(TrueE, absolute_diff, bins = (create_log_bin_edges(lower_E, higher_E, Ebins), np.linspace(lower_Residue, higher_Residue, num=Rbins)))
    return hist


def median_and_quantile_along_hist2d_y_and_x(x_array, y_array, 
                                       x_bin_edges, y_bin_edges,
                                       quantile_around_median = 0.341):
    '''
    y_median and quantiles: median and quantiles of x-slices, to plot against x bin centers
    x_median and quantiles are opposite.
    '''

    #if x_array.shape != y_array.shape:
    #    raise InputError("x_array and y_array must have same shape!")

    data = pd.DataFrame(data = {"x":x_array, "y":y_array})
    data = data.dropna()
    
    histogram, x_edges, y_edges = np.histogram2d(x_array, y_array, bins = (x_bin_edges,y_bin_edges))
    
    def upper_quantile(array):
        return np.percentile(array, q = 50+100*quantile_around_median)
    
    def lower_quantile(array):
        return np.percentile(array, q = 50-100*quantile_around_median)
    
    y_median,  bin_edges, binnumber = scipy.stats.binned_statistic(data["x"],data["y"],
                                                             statistic = 'median',           bins = x_bin_edges)  
    y_upper_q, bin_edges, binnumber = scipy.stats.binned_statistic(data["x"],data["y"],
                                                                 statistic = upper_quantile, bins = x_bin_edges)
    y_lower_q, bin_edges, binnumber = scipy.stats.binned_statistic(data["x"],data["y"],
                                                                 statistic = lower_quantile, bins = x_bin_edges)
    
    x_median,  bin_edges, binnumber = scipy.stats.binned_statistic(data["y"],data["x"],
                                                             statistic = 'median',           bins = y_bin_edges)  
    x_upper_q, bin_edges, binnumber = scipy.stats.binned_statistic(data["y"],data["x"],
                                                                 statistic = upper_quantile, bins = y_bin_edges)
    x_lower_q, bin_edges, binnumber = scipy.stats.binned_statistic(data["y"],data["x"],
                                                                 statistic = lower_quantile, bins = y_bin_edges)
    
    outdict = {"y_median":   y_median,
               "y_upper_q":  y_upper_q,
               "y_lower_q":  y_lower_q,
               "x_median":   x_median,
               "x_upper_q":  x_upper_q,
               "x_lower_q":  x_lower_q,
               "orig_hist2d": histogram
    }
    
    return outdict






################################ plotting functionality ##############################################

def recoplotter_vs_parameter(reco_dict_list, reco_names_list, reco_main_colors, reco_second_colors, rel_dev_bins,
                             par_bins, xlabel, ylabel):
    '''
    reco dict list:
    
    '''
    size = 12

    alpha_bckgrd = 1
    fig, (axes) = plt.subplots(len(reco_dict_list)+1,1, figsize = (12,5+2.5*len(reco_dict_list)), sharex=True) #adaptive figsize 5*len(recos)


    for idx_reco, current_dict in enumerate(reco_dict_list):
        c_color = reco_main_colors[idx_reco]
        c_color2= reco_second_colors[idx_reco]
        c_name = reco_names_list[idx_reco]

        c = axes[idx_reco].pcolormesh(par_bins, rel_dev_bins, current_dict["orig_hist2d"].T, cmap="Greys",
                            alpha=alpha_bckgrd, norm=colors.LogNorm(vmin = 1, vmax = np.max(current_dict["orig_hist2d"])) )
        cbar = fig.colorbar(c,ax = axes[idx_reco])
        cbar.set_label('# Events, {}x{} bins'.format(len(par_bins), len(rel_dev_bins)),size=size,rotation=90)
        sigma_plus  = 10**(current_dict["means"]+current_dict["stds"])
        sigma_minus = 10**(current_dict["means"]-current_dict["stds"])


        axes[idx_reco].step(par_bins[:-1], sigma_plus, color=c_color2, where="post", label="std.dev")
        axes[idx_reco].hlines(sigma_plus[-1], par_bins[-2], par_bins[-1], color=c_color2)

        axes[idx_reco].step(par_bins[:-1], sigma_minus, color=c_color2, where="post")
        axes[idx_reco].hlines(sigma_minus[-1], par_bins[-2], par_bins[-1], color=c_color2)

        axes[idx_reco].step(par_bins[:-1], 10**current_dict["means"], color=c_color, where="post", label="mean", linewidth=1.1)
        axes[idx_reco].hlines(10**current_dict["means"][-1], par_bins[-2], par_bins[-1], color=c_color, linewidth=1.1)

        axes[idx_reco].set_yscale("log")
        axes[idx_reco].text(0.05, 0.95, c_name, transform=axes[idx_reco].transAxes, fontsize=size,
                verticalalignment='top', color=c_color)
        axes[idx_reco].legend(loc="lower right")                                                                                                                              
        plt.setp(axes[idx_reco].get_yticklabels(), fontsize=size)
        plt.setp(axes[idx_reco].get_xticklabels(), fontsize=size)


        axes[-1].hlines(10**current_dict["means"][-1], par_bins[-2], par_bins[-1], color=c_color)
        axes[-1].step(par_bins[:-1], 10**current_dict["means"], color=c_color, where="post",
                 label=c_name)

    axes[-1].set_yscale("log")
    axes[-2].set_ylabel(ylabel, fontsize = size)                                                                                                                                 
    axes[-1].set_xlabel(xlabel, fontsize = size)
    axes[-1].grid()
    axes[-1].legend()
    plt.setp(axes[-1].get_yticklabels(), fontsize=size)
    plt.setp(axes[-1].get_xticklabels(), fontsize=size)

    plt.tight_layout()
    return plt.gcf()

def recoplotter_linear_vs_parameter(reco_dict_list, reco_names_list, reco_main_colors, reco_second_colors, rel_dev_bins,
                             par_bins, xlabel, ylabel):
    '''
    bins = bin edges arrays
    '''
    size = 12

    alpha_bckgrd = 1
    fig, (axes) = plt.subplots(len(reco_dict_list)+2,1, figsize = (12,5+2.5*len(reco_dict_list))) #adaptive figsize 5*len(recos)


    for idx_reco, current_dict in enumerate(reco_dict_list):
        c_color = reco_main_colors[idx_reco]
        c_color2= reco_second_colors[idx_reco]
        c_name = reco_names_list[idx_reco]

        c = axes[idx_reco].pcolormesh(par_bins, rel_dev_bins, current_dict["orig_hist2d"].T, cmap="Greys",
                            alpha=alpha_bckgrd, norm=colors.LogNorm(vmin = 1, vmax = np.max(current_dict["orig_hist2d"])) )
        cbar = fig.colorbar(c,ax = axes[idx_reco])
        cbar.set_label('# Events'.format(len(par_bins)-1, len(rel_dev_bins)-1),size=size,rotation=90)
        sigma_plus  = (current_dict["means"]+current_dict["stds"])
        sigma_minus = (current_dict["means"]-current_dict["stds"])


        axes[idx_reco].step(par_bins[:-1], sigma_plus, color=c_color2, where="post", label="std.dev")
        axes[idx_reco].hlines(sigma_plus[-1], par_bins[-2], par_bins[-1], color=c_color2)

        axes[idx_reco].step(par_bins[:-1], sigma_minus, color=c_color2, where="post")
        axes[idx_reco].hlines(sigma_minus[-1], par_bins[-2], par_bins[-1], color=c_color2)

        axes[idx_reco].step(par_bins[:-1], current_dict["means"], color=c_color, where="post", label="mean", linewidth=1.1)
        axes[idx_reco].hlines(current_dict["means"][-1], par_bins[-2], par_bins[-1], color=c_color, linewidth=1.1)

        axes[idx_reco].set_yscale("linear")
        axes[idx_reco].text(0.05, 0.95, c_name, transform=axes[idx_reco].transAxes, fontsize=size,
                verticalalignment='top', color=c_color)
        axes[idx_reco].legend(loc="lower right")                                                                                                                              
        plt.setp(axes[idx_reco].get_yticklabels(), fontsize=size)
        plt.setp(axes[idx_reco].get_xticklabels(), fontsize=size)
        axes[idx_reco].set_ylabel(ylabel, fontsize = size)                                                                                                                                 

        axes[-2].hlines(current_dict["stds"][-1], par_bins[-2], par_bins[-1], color=c_color2)
        axes[-2].step(par_bins[:-1], current_dict["stds"], color=c_color2, where="post",
                 label=c_name+" resolution")

        
        axes[-1].hlines(current_dict["means"][-1], par_bins[-2], par_bins[-1], color=c_color)
        axes[-1].step(par_bins[:-1], current_dict["means"], color=c_color, where="post",
                 label=c_name)

    axes[-2].set_ylabel(ylabel, fontsize = size)  
    axes[-2].grid()
    axes[-2].legend()

        
    axes[-1].set_yscale("linear")
    axes[-1].set_ylabel(ylabel, fontsize = size)                                                                                                                                 
    axes[-1].set_xlabel(xlabel, fontsize = size)
    axes[-1].grid()
    axes[-1].legend()
    plt.setp(axes[-1].get_yticklabels(), fontsize=size)
    plt.setp(axes[-1].get_xticklabels(), fontsize=size)

    plt.tight_layout()
    return plt.gcf()


def reco_medians_vs_parameter(reco_dict_list, reco_names_list, reco_main_colors, reco_second_colors, rel_dev_bins,
                             par_bins, xlabel, ylabel, ylabel_2, log_x = False):
    size = 14

    alpha_bckgrd = 1
    fig, (axes) = plt.subplots(len(reco_dict_list)+2,1, figsize = (12,6+3*len(reco_dict_list))) #adaptive figsize 5*len(recos)


    for idx_reco, current_dict in enumerate(reco_dict_list):
        c_color = reco_main_colors[idx_reco]
        c_color2= reco_second_colors[idx_reco]
        c_name = reco_names_list[idx_reco]

        c = axes[idx_reco].pcolormesh(par_bins, rel_dev_bins, current_dict["orig_hist2d"].T, cmap="Greys",
                            alpha=alpha_bckgrd, norm=colors.LogNorm(vmin = 1, vmax = np.max(current_dict["orig_hist2d"])) )
        cbar = fig.colorbar(c,ax = axes[idx_reco])
        cbar.set_label('# Events'.format(len(par_bins)-1, len(rel_dev_bins)-1),size=size,rotation=90)
        sigma_plus  = (current_dict["y_upper_q"])
        sigma_minus = (current_dict["y_lower_q"])


        axes[idx_reco].step(par_bins[:-1], sigma_plus, color=c_color2, where="post", label="1 $\sigma$ quantile")
        axes[idx_reco].hlines(sigma_plus[-1], par_bins[-2], par_bins[-1], color=c_color2)

        axes[idx_reco].step(par_bins[:-1], sigma_minus, color=c_color2, where="post")
        axes[idx_reco].hlines(sigma_minus[-1], par_bins[-2], par_bins[-1], color=c_color2)

        axes[idx_reco].step(par_bins[:-1], current_dict["y_median"], color=c_color, where="post", label="median", linewidth=1.1)
        axes[idx_reco].hlines(current_dict["y_median"][-1], par_bins[-2], par_bins[-1], color=c_color, linewidth=1.1)

        axes[idx_reco].set_yscale("linear")
        axes[idx_reco].text(0.05, 0.95, c_name, transform=axes[idx_reco].transAxes, fontsize=size,
                verticalalignment='top', color=c_color)
        axes[idx_reco].legend(loc="lower right", fontsize = size)                                                                                                                              
        plt.setp(axes[idx_reco].get_yticklabels(), fontsize=size)
        plt.setp(axes[idx_reco].get_xticklabels(), fontsize=size)
        axes[idx_reco].set_ylabel(ylabel, fontsize = size)                                                                                                                                 

        current_1sigma = current_dict["y_upper_q"] - current_dict["y_lower_q"]
        axes[-1].hlines(current_1sigma[-1], par_bins[-2], par_bins[-1], color=c_color2)
        axes[-1].step(par_bins[:-1], current_1sigma, color=c_color2, where="post",
                 label=c_name+" 1 $\sigma$ resolution")

        
        axes[-2].hlines(current_dict["y_median"][-1], par_bins[-2], par_bins[-1], color=c_color)
        axes[-2].step(par_bins[:-1], current_dict["y_median"], color=c_color, where="post",
                 label=c_name+" median")

    if log_x == True:
        for axis in axes:
            axis.set_xscale("log") 

    axes[-2].set_ylabel(ylabel, fontsize = size)  
    axes[-2].grid()
    axes[-2].legend(fontsize = size)
    axes[-2].set_yscale("linear")
    plt.setp(axes[-2].get_yticklabels(), fontsize=size)
    plt.setp(axes[-2].get_xticklabels(), fontsize=size)
    
    axes[-1].set_ylabel(ylabel_2, fontsize = size)                                       
    axes[-1].set_xlabel(xlabel, fontsize = size)
    axes[-1].grid()
    axes[-1].legend(fontsize = size)
    plt.setp(axes[-1].get_yticklabels(), fontsize=size)
    plt.setp(axes[-1].get_xticklabels(), fontsize=size)
    
    #plt.tight_layout()
    return plt.gcf()

