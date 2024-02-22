# -----------------------------------------------------------------------------
# Solar_FLux_Analytical.py
# Authors: Stephan Meighen-Berger
# Estimates the solar atmospheric neutrino flux using an analytical approach
# -----------------------------------------------------------------------------
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# -----------------------------------------------------------------------------
# Input values
# Primary flux
e_grid = np.logspace(0., 10., 101)
primary_flux_norm = 1.
gamma = 2.
primary_flux = primary_flux_norm * e_grid**(-gamma)
# Interaction lengths
interaction_length_p = 1.
interaction_length_pi = 0.5
# Z kernels
Z_pp = 0.5
Z_ppi = 0.75
Z_pipi = 0.25
# Beta functions
beta_pi = 1. / 130.
beta_mu = 1. / 1000.
# Lambdas
Lambda_p = interaction_length_p / (1. - Z_pp)
Lambda_pi = interaction_length_pi / (1. - Z_pipi)
# Energy transfer constants
a_mu = 0.9
a_nu = 0.5
# Lazy constants
c_pi_int = (primary_flux_norm * 
    Z_ppi / (1. - Z_pp) *
    Lambda_pi / (Lambda_p - Lambda_pi)
)
print("C_pi_int: %.2f" %(c_pi_int))
c_pi_dec = (
    Z_ppi / interaction_length_p
)
print("C_pi_dec: %.2f" %(c_pi_dec))
# -----------------------------------------------------------------------------
# Analytical flux equations
def dphidX_int(X, phi, E):
    """ Differential equation for the pi interaction assumption

    Parameters
    ----------
    X : float
        The cascade depth
    phi : float
        The current flux
    E : float
        The energy of interest

    Returns
    -------
    float
        The resulting differential flux value
    """
    loss = -phi / (X * E *beta_mu)
    gain = (
        c_pi_int * (E/a_mu)**(-(gamma + 1.)) / (X * beta_pi) *
        (np.exp(-X/Lambda_pi) - np.exp(-X/Lambda_p))
    )
    return (loss + gain)

def dphidX_dec(X, phi, E):
    """ Differential equation for the pi decay assumption

    Parameters
    ----------
    X : float
        The cascade depth
    phi : float
        The current flux
    E : float
        The energy of interest

    Returns
    -------
    float
        The resulting differential flux value
    """
    loss = -phi / (X * E *beta_mu)
    gain = (
        c_pi_dec * (E/a_mu)**(-(gamma)) * primary_flux_norm * E**(-(gamma)) *
        (np.exp(-X/Lambda_p))
    )
    return (loss + gain)
# -----------------------------------------------------------------------------
# Analytical solution
def phi_mu_dec(X, E):
    """ Analytical solution for the pi decay assumption

    Parameters
    ----------
    X : float
        The cascade depth
    E : np.array
        The energy of interest

    Returns
    -------
    np.array
        The resulting differential flux value
    """
    prefac = (
        c_pi_dec * beta_mu * a_mu**(gamma) * primary_flux_norm * E**(-gamma)
    )
    var = X * np.exp(-X/Lambda_p) * E**(-gamma + 1.)
    return prefac * var
# -----------------------------------------------------------------------------
# Solving
x_points = np.array([10., 20.]) # np.linspace(0., 100, 101)
sol_dec = odeint(dphidX_dec, 0., x_points, args=(1e4, ))
print(sol_dec)
print(phi_mu_dec(x_points, 1e4))
# -----------------------------------------------------------------------------
# Plotting options
std_fig_size = 6.
std_fontsize = 20.
std_lw = 3.
std_hlength = 2.
# -----------------------------------------------------------------------------
# Plotting
fig, ax = plt.subplots(figsize=(std_fig_size, std_fig_size * 6. / 8.))

ax.tick_params(axis = 'both', which = 'major', labelsize=std_fontsize, direction='in')
ax.tick_params(axis = 'both', which = 'minor', labelsize=std_fontsize, direction='in')
ax.set_xlabel(r'Injection Energy [GeV]', fontsize=std_fontsize)
ax.set_ylabel(r'Norm', fontsize=std_fontsize)
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig("C:\\Users\\steph\\Desktop\\compare.pdf",
            bbox_inches='tight')
# -----------------------------------------------------------------------------