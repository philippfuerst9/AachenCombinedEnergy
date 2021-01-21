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

class LLHScan_1D(object):
    """class to load LLH values for a scan with just one free systematic parameter.
    """
    def __init__(self, path):

        self.path = os.path.join(path)
        self.scan_gamma_files = sorted(glob.glob(os.path.join(self.path, "*FitRes*gamma_astro*")))
        self.scan_phi_files = sorted(glob.glob(os.path.join(self.path, "*FitRes*astro_norm*")))
        self.best_fit = None
        self.min_LLH = 0.0
        self.best_gamma = None
        self.best_phi = None
        try:
            self.best_fit = pd.read_pickle(os.path.join(self.path, "Freefit.pickle"))
            self.best_gamma = self.best_fit["fit-result"][1]["gamma_astro"]
            self.best_phi = self.best_fit["fit-result"][1]["astro_norm"]
            self.min_LLH = float(self.best_fit["fit-result"][0][1])
            print("best-fit LLH val {}".format(self.min_LLH))
        except:
            print("no best fit file found. Assuming best LLH = 0.0!")

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

def get_results(path, systematics):
    """builds a dictionary with keys [systematics], 
    each containing the fit results for this systematic
    """
    fit_results = {}
    for systematic in systematics:
        print("reading {} scan".format(systematic))
        scan = LLHScan_1D(os.path.join(path, systematic))
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
    print(np.where(mask))
    print(x[np.where(mask)])
    max_indices= np.where(mask)
    max_pos = x[max_indices]
    return max_indices, max_pos

def fix_index(x_values, y_values, index, k=3):
    """takes x and y as 1d-arrays of same len and an index to fix. 
    returns a new y array where the specified index value is inferred by a Univariate SPline
    """
    f = UnivariateSpline(x_values, y_reduced, s=0, k=k)
    y_max = f (x_values[index])
    new_y_values = y_values
    new_y_values[index] = y_max
    return new_y_values

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

def get_x_from_y(y_position, x_values, y_values, x_center = 0):
    """estimates x values for certain y value in 1-D distribution.
    returns: all x values where this distribution reaches the value y_position.
    """
    y_reduced = np.array(y_values) - y_position
    f_reduced = UnivariateSpline(x_values, y_reduced, s=0)
    deltas = []
    for root in f_reduced.roots():
        deltas.append(root-x_center)
    return deltas

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