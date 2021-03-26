"""
Collection of analysis classes and math functions to analyze 1D parameter asimov scans
"""

# imports
import glob
import imp
import os

import numpy as np
import pandas as pd
import pickle
from scipy import stats
from scipy.interpolate import griddata as scipygrid
from scipy.interpolate import UnivariateSpline
import scipy

plot_config = {}
plot_config["labels"] = {
    "barr_y"     : r"Barr $Y$",
    "barr_z"     : r"Barr $Z$",
    "muon_norm"  : r"Muon Template",
    "ice_scat"   : r"Ice Scattering",
    "dom_eff"    : r"DOM Efficiency",
    "barr_w"     : r"Barr $W$",
    "barr_h"     : r"Barr $H$",
    "conv_norm"  : r"$\Phi_{\text{Conventional}}$",
    "CR_grad"    : r"CR Model Interp.",
    "gamma_astro": r"$\gamma_{\text{Astro}}$",
    "astro_norm" : r"$\Phi_{\text{Astro}}$",
    "prompt_norm": r"$\Phi_{\text{Prompt}}$",
    "delta_gamma": r"CR $\Delta \gamma$",
    "ice_holep0" : r"Ice Hole $p_0$",
    "ice_abs"    : r"Ice Absorption"
}

plot_config["colors"] = {
    "barr_y"     : 'brown',
    "barr_z"     : 'indianred',
    "barr_w"     : 'firebrick',
    "barr_h"     : 'darkred',
    
    "muon_norm"  : 'green',
    "conv_norm"  : '#7f7f7f',
    "astro_norm" : 'blue',
    "prompt_norm": '#17becf',
    
    "CR_grad"    : '#bcbd22',
    "gamma_astro": 'blue',
    "delta_gamma": '#ff7f0e',
    
    "ice_holep0" : 'cornflowerblue',
    "ice_abs"    : 'royalblue',
    "ice_scat"   : 'darkblue',
    "dom_eff"    : 'blue'
}

plot_config["linestyles"] = {
    "barr_y"     : '-.',
    "barr_z"     : '-',
    "barr_w"     : '--',
    "barr_h"     : ':',
    
    "muon_norm"  : ':',
    "conv_norm"  : '-',
    "astro_norm" : '--',
    "prompt_norm": '-.',
    
    "CR_grad"    : '-.',
    "gamma_astro": '-',
    "delta_gamma": '-.',
    
    "ice_holep0" : '-',
    "ice_abs"    : '--',
    "ice_scat"   : '-.',
    "dom_eff"    : ':'
}



class LLHScan_1D(object):
    """class to load LLH values for a scan with just one free systematic parameter.
    """
    def __init__(self, path, get_best_LLHs = -1.0):

        self.path = os.path.join(path)
        self.scan_gamma_files = sorted(glob.glob(os.path.join(self.path, "*FitRes*gamma_astro*")))
        self.scan_phi_files = sorted(glob.glob(os.path.join(self.path, "*FitRes*astro_norm*")))
        self.best_fit = None

        self.best_gamma = None
        self.best_phi = None
        if get_best_LLHs == -1.0:
            try:
                self.best_fit = pd.read_pickle(os.path.join(self.path, "Freefit.pickle"))
                self.best_gamma = self.best_fit["fit-result"][1]["gamma_astro"]
                self.best_phi = self.best_fit["fit-result"][1]["astro_norm"]
                self.min_LLH = float(self.best_fit["fit-result"][0][1])
                print("best-fit LLH val {}".format(self.min_LLH))
            except:
                print("no best fit file found. Assuming best LLH = 0.0!")
                self.min_LLH = 0.0
        elif get_best_LLHs != -1.0:
            print("Best LLH set manually to {}".format(get_best_LLHs))
            self.min_LLH = get_best_LLHs
            
    def random_file(self):
        """returns one pickle fit file."""
        return pd.read_pickle(os.path.join(path, self.scan_gamma_files[0]))
    
    def fit_results(self):
        """loops over the fit files and loads the calculated llh values at the fixed phi(gamma) position.
        """
        gammas_scan_points = []
        phis_scan_points = []
        gamma_llhs = []
        phi_llhs = []
        fitted_phis = []
        fitted_gammas = []
        fitted_pars_fix_gamma = []
        fitted_pars_fix_phi = []

        if len(self.scan_gamma_files)>0 and len(self.scan_phi_files)>0:
            for g in self.scan_gamma_files:
                current_fit = pd.read_pickle(g)
                current_gamma = current_fit["fixed-parameters"]["gamma_astro"]
                current_llh   = 2 * (float(current_fit["fit-result"][0][1]) - self.min_LLH)
                current_fit_phi = current_fit["fit-result"][1]["astro_norm"]
                current_fitted_pars = current_fit["fit-result"][1] 
                
                gammas_scan_points.append(current_gamma)
                gamma_llhs.append(current_llh)
                fitted_phis.append(current_fit_phi)
                fitted_pars_fix_gamma.append(current_fitted_pars)
                
            for p in self.scan_phi_files:
                current_fit = pd.read_pickle(p)
                current_phi = current_fit["fixed-parameters"]["astro_norm"]
                current_llh   = 2 * (float(current_fit["fit-result"][0][1]) - self.min_LLH)
                current_fit_gamma = current_fit["fit-result"][1]["gamma_astro"]
                current_fitted_pars = current_fit["fit-result"][1] 
                
                phis_scan_points.append(current_phi)
                phi_llhs.append(current_llh)  
                fitted_gammas.append(current_fit_gamma)
                fitted_pars_fix_phi.append(current_fitted_pars)

            # sort gamma scans for nice plotting
            gamma_llhs = np.array(gamma_llhs)
            gamma_llhs = gamma_llhs[np.argsort(np.array(gammas_scan_points))]
            
            fitted_phis = np.array(fitted_phis)
            fitted_phis = fitted_phis[np.argsort(np.array(gammas_scan_points))]
            
            fitted_pars_fix_gamma = np.array(fitted_pars_fix_gamma)
            fitted_pars_fix_gamma = fitted_pars_fix_gamma[np.argsort(np.array(gammas_scan_points))]
            
            gammas_scan_points = np.sort(np.array(gammas_scan_points))

            # sort phi scans for plotting
            phi_llhs = np.array(phi_llhs)
            phi_llhs = phi_llhs[np.argsort(np.array(phis_scan_points))]
            
            fitted_gammas = np.array(fitted_gammas)
            fitted_gammas = fitted_gammas[np.argsort(np.array(phis_scan_points))]
            
            fitted_pars_fix_phi = np.array(fitted_pars_fix_phi)
            fitted_pars_fix_phi = fitted_pars_fix_phi[np.argsort(np.array(phis_scan_points))]
            
            phis_scan_points = np.sort(np.array(phis_scan_points))

            gamma_dict = {"scan_points": np.array(gammas_scan_points),
                         "2DLLH": np.array(gamma_llhs),
                         "fitted_phis": np.array(fitted_phis),
                         "fitted_pars": fitted_pars_fix_gamma
                         }
                            
            phi_dict = {"scan_points": np.array(phis_scan_points),
                       "2DLLH":np.array(phi_llhs),
                       "fitted_gammas": np.array(fitted_gammas),
                       "fitted_pars": fitted_pars_fix_phi
                       }
            
            return_dict = {"gamma" : gamma_dict,
                           "phi" : phi_dict
                          }

            return return_dict
        else:
            print("No fit files detected in {}".format(self.path))
            return None

class Param_Scan_1d(object):
    """class to load 1d-fit scans.
    load with:
    path: path to dir containing scans
    params: list of strings that should be contained in the .pickle names, e.g. "barr_w", "gamma_astro"
    """
    def __init__(self, path, params):
        self.path = os.path.join(path)
        self.params = params
        self.path_lists = {}
        for param in self.params:
            print(param)
            self.path_lists[param] = sorted(glob.glob(os.path.join(self.path, "*FitRes*_{}_*".format(param))))
            
            if len(self.path_lists[param]) == 0:
                print("No fit files found for parameter{}".format(param))
        try:
            print("found best fit file")
            self.best_fit = pd.read_pickle(os.path.join(self.path, "Freefit.pickle"))
            try:
                self.best_gamma = self.best_fit["fit-result"][1]["gamma_astro"]
                self.best_phi = self.best_fit["fit-result"][1]["astro_norm"]
            except:
                print("gamma and phi not found for bestfit. Are they fixed?")
            self.min_LLH = float(self.best_fit["fit-result"][0][1])
            print("best-fit LLH val {}".format(self.min_LLH))
        except:
            print("no best fit file found. Assuming best LLH = 0.0!")
            self.min_LLH = 0.0
   


    def get_random_file(self, param):
        """returns one pickle fit file."""
        return pd.read_pickle(os.path.join(self.path, self.path_lists[param][0]))

    def get_fit_results(self, param):
        """loops over the fit files and loads the calculated llh values at the fixed param position.
        """     
        results = pd.DataFrame(columns = ["fixed_{}".format(param), "2DLLH", "fitted_pars"])
        
        #read all the fits in path list
        for i, file in enumerate(self.path_lists[param]):
            fit = pd.read_pickle(file)
            fixed_param = fit["fixed-parameters"][param]
            dllh = 2*float(fit["fit-result"][0][1] - self.min_LLH)
            #dllh = float(fit["fit-result"][0][1])
            fitted_pars = fit["fit-result"][1]
            row = [fixed_param, dllh, fitted_pars]
            results.loc[i] = row
            
        #sort results since the file path list is not always correctly sorted (naming scheme)
        results = results.sort_values("fixed_{}".format(param), ascending = True)
        return results

    def get_fitted_pars(self, results):
        """takes a list of fitted parameter dictionaries 
        and returns a dict of lists
        """
        ld = results["fitted_pars"]
        dl = {k: [dic[k] for dic in ld] for k in ld[0]}
        return dl
        
def get_results(path, systematics, get_best_LLHs = -1.0):
    """builds a dictionary with keys [systematics], 
    each containing the fit results for this systematic
    """
    fit_results = {}
    for systematic in systematics:
        print("reading {} scan".format(systematic))
        scan = LLHScan_1D(os.path.join(path, systematic), get_best_LLHs = get_best_LLHs)
        print(scan)
        results = scan.fit_results()
        fit_results[systematic] = results
    return fit_results

def read_random_fitfile(directory):
    """reads an arbitrary fit result pickle file from directory."""
    files = os.listdir(directory)
    return pd.read_pickle(os.path.join(directory, files[0])) 


def find_local_maximum(x,y,factor = 1.1):
    """returns index and y-values of every point in x \
    bigger than its two adjacent values.
    factor: 
        factor to multiply the comparison points by. 
    returns: 
        max_indices: indices of maxima
        max_pos: x-values of maxima
    """
    if factor < 1:
        raise ValueError("factor cannot be <1 but was set to {}".format(factor))
    if len(x) != len(y):
        raise ValueError("lists x and y must have the same lengths.")
    test_for_max = y[1:-1]
    left_vals = y[0:-2]*factor
    right_vals = y[2:]*factor
    x = np.array(x)
    mask = np.zeros(len(y))
    mask[1:-1] = np.greater(test_for_max, left_vals) & np.greater(test_for_max, right_vals)
    #print(np.where(mask))
    #print(x[np.where(mask)])
    max_indices= np.where(mask)
    max_pos = x[max_indices]
    #return list(max_indices), list(max_pos)
    return list(max_indices[0])

def clean_interp(x_values, y_values, bad_indices = None, k=3, n=1000, s = None):
    if bad_indices != None:
        x_smooth = np.delete(x_values,bad_indices)
        y_smooth = np.delete(y_values,bad_indices)
    elif bad_indices == None:
        x_smooth = x_values
        y_smooth = y_values
    f = UnivariateSpline(x_smooth, y_smooth, s=s, k=k)
    x_fine = np.linspace(min(x_values), max(x_values),n)
    y_fine = f(x_fine)
    return x_fine, y_fine

#def fix_index(x_values, y_values, index, k=3):
#    """takes x and y as 1d-arrays of same len and an index to fix. 
#    returns a new y array where the specified index value is inferred by a Univariate SPline
#    """
#    f = UnivariateSpline(x_values, y_values, s=0, k=k)
#    y_max = f (x_values[index])
#    new_y_values = y_values
#    new_y_values[index] = y_max
#    return new_y_values

def delta_ratio(x_initial, y_numerator, y_denominator, y_position):
    """Function to calculate the widths of two distributions around a minimum.
    x_initial : array of x values
    y_numerator : array of y values of numerator distribution corresponding to x_initial
    y_denominator : array of y values of denominator distribution
    y_position : position on y-axis where the distribution widths are calculated
    
    returns:
    delta_num : width of numerator distr. at y_position
    delta_denom : width of denominator distr. at y_position.
    """
    yreduced_num = np.array(y_numerator) - y_position
    freduced_num = UnivariateSpline(x_initial, yreduced_num, s=0)
    delta_num = freduced_num.roots()[1]- freduced_num.roots()[0]
    
    yreduced_denom = np.array(y_denominator) - y_position
    freduced_denom = UnivariateSpline(x_initial, yreduced_denom, s=0)
    delta_denom = freduced_denom.roots()[1]- freduced_denom.roots()[0]
    return delta_num, delta_denom

def get_x_from_y(y_position, x_values, y_values):
    """estimates x values for certain y value in 1-D distribution.
    input: y_position: the y_value for which to find roots
            x_values, y_values: 1d arrays describing the distribution
    returns: all x values where this distribution reaches the value y_position.
    """
    y_reduced = y_values - y_position
    #print(np.shape(y_reduced))
    return UnivariateSpline(x_values, y_reduced, s=0).roots()


def sigma_syst_base_ratio(sigma_tot, sigma_base):
    return np.sqrt((sigma_tot/sigma_base)**2 -1)

def get_chisq_val_for_sigma(sigma = 1, k = 1):
    """gets the value of a Chi-Square distribtuion with k degrees of freedom
    so that the probability to draw a Chi-Square smaller than this value corresponds to sigma.
    i.e. 1sigma --> 68% prb to draw Chi-Square less than value.
    """
    value = stats.chi2.ppf(1-2*stats.norm.sf(sigma, 0, 1), k) 
    return value

def get_sigma_for_chisq_val(chisq_val, k=1):
    """get the width in gaussian sigmas corresponding to a chisq value in a 
    chisq distribution with ndof = k.
    """
    return stats.norm.isf(stats.chi2(k).sf(dLLH*2.)*0.5 )

def dl_from_ld(ld):
    """ 
    builds a dict of lists from list of dicts. 
    All dicts need to have the same keys.
    """
    keys = ld[0].keys()
    for key in keys:
        if not all(key in dic for dic in ld):
            raise ValueError("dict entries in list have different keys! Keys in first dict: {}".format(keys))
        
    dl = {k: [dic[k] for dic in ld] for k in ld[0]}
    return dl

def ld_from_dl(dl):
    """ 
    builds a list of dicts from dict of lists. 
    All lists need to have the same len.
    """

    ld = [dict(zip(dl,t)) for t in zip(*dl.values())]
    return ld

def interp(x,y):
    """
    interpolates (1D) for two arrays x,y of same len
    returns 1000 y-points in the input x range
    """
    interpolated = scipy.interpolate.interp1d(x,y,bounds_error = True)
    xfine = np.linspace(min(x), max(x), 1000)
    yfine = interpolated(xfine)
    #axis.plot(xfine, yfine, color = c, linewidth = 0.5)
    return xfine, yfine

def get_x_from_y(x_fine, y_fine, y_to_find):
    """ Takes (sufficiently fine) x and y=f(x) 1d arrays and finds the closest matches to y_to_find.
    """
    yreduced = np.array(y_fine) - y_to_find
    freduced = scipy.interpolate.UnivariateSpline(x_fine, yreduced, s=0)
    return freduced.roots()



# Some error and plotting functionality

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
def rolling_window_2d(a, shape):  # rolling window for 2D array
    shape = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# custom step function to include the last bin
def full_step(x_bins,y, label=None, color="black", alpha = 1, ax = None, linestyle = "-"):
    if ax !=None:
        ax.step(x_bins[:-1],y, color=color, label=label, where = "post", alpha = alpha, linestyle = linestyle)
        ax.hlines(y[-1], x_bins[-2], x_bins[-1], color = color, alpha = alpha, linestyle = linestyle) 
    else:
        plt.step(x_bins[:-1],y, color=color, label=label, where = "post", alpha = alpha, linestyle = linestyle)
        plt.hlines(y[-1], x_bins[-2], x_bins[-1], color = color, alpha = alpha, linestyle = linestyle)

def poisson_hist(samples, bins, weights=None, normed=False):
    """wrapper around numpy hsitogram to yield poisson errors per bin.
    """
    # set weights if not give
    if weights is None:
        weights = np.ones_like(samples)
    hist, bins = np.histogram(samples, bins, weights=weights)
    if normed:
        norm = 1./np.diff(bins)/hist.sum()
        hist = hist*norm
    # calculate error
    idxs = np.digitize(samples, bins)
    yerror = np.zeros_like(hist)
    for i in range(len(hist)):
        yerror[i] = np.sqrt(np.sum((weights[idxs==(i+1)])**2))
    if normed:
        yerror = yerror*norm
    print("{:,}".format(len(samples)))
    return hist, yerror

def plot_hist(ax, hist, bins, yerror, **kwargs):
    l = ax.errorbar(np.mean(rolling_window(bins, 2), axis=1), hist, yerr=yerror,
                    drawstyle="steps-mid", capsize=4.0, capthick=2,
                    **kwargs)
    return l
def plot_ratio_single_err(ax, hist, bins, yerror, hist_baseline, **kwargs):
    yerror_ratio = yerror/hist_baseline
    l = ax.errorbar(np.mean(rolling_window(bins, 2), axis=1), hist/hist_baseline, yerr=yerror_ratio,
                    drawstyle="steps-mid", capsize=4.0, capthick=2,
                    **kwargs)
    return l

def get_ratio_error(hist, hist_baseline, sigma_hist, sigma_baseline):
    return np.sqrt(sigma_baseline**2 / hist_baseline**2 + hist**2 / hist_baseline**4 * sigma_hist**2)

def plot_ratio_double_err(ax, hist, hist_baseline, sigma_hist, sigma_baseline, bins, **kwargs):
    yerror = get_ratio_error(hist, hist_baseline, sigma_hist, sigma_baseline)
    l = ax.errorbar(np.mean(rolling_window(bins, 2), axis=1), hist/hist_baseline, yerr=yerror,
                    drawstyle="steps-mid", capsize=2.0, capthick=1,
                    **kwargs)
    return l