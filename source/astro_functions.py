#!/usr/bin/env python3
"""
File: astro_functions.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Functions to compute astrophysical quantities.


Functions:
    - compute_z_from_mag: Compute the redshift z of a galaxy with apparent magnitude mag_app and absolute magnitude mag_abs.
    - compute_z_func: Return dummy function for newton's method computation of z.
    - compute_mag_abs: Compute the absolute magnitude for a given apparent magnitude and redshift.
    - K_correction_func: NOT IMPLEMENTED
    - setup_cosmology: Setup the assumed cosmology.


TODO:
    - Write K-corrections for the different channels.
    - Central absolute magnitude values assume a fixed redshift. Fix this?
    - Is Newton's method a stupid way to calculate the redshift? Probably.
"""
from astropy import constants
from astropy import cosmology
from astropy import units as u
from scipy.optimize import newton
import numpy as np


def setup_cosmology(H0=67.4, omegam0=0.32, omegade0=0.68):
    """Setup the assumed cosmology."""
    cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=omegam0)

    return cosmo


def K_correction_func(z):
    return 0


def compute_mag_abs(z, mag_app, cosmo, k_correction=K_correction_func):
    """Compute the absolute magnitude for a given apparent magnitude and redshift.


    Input:
        - z: redshift
        - mag_app: apparent magnitude
        - cosmo: astropy cosmology
        - k_correction: function of z calculating the k-correction
    """
    # Eq form given in Eales 93 after eq (2)
    return mag_app - 5 * np.log10(cosmo.luminosity_distance(z).to(u.pc).value / 10) - k_correction(z)


def compute_z_func(z, mag_abs, mag_app, cosmo):
    """Return dummy function for newton's method computation of z."""

    return mag_abs - compute_mag_abs(z, mag_app, cosmo)


def compute_z_from_mag(mag_abs, mag_app, z_guess, cosmo):
    """Compute the redshift z of a galaxy with apparent magnitude mag_app and absolute magnitude mag_abs.


    Input:
        - mag_abs: absolute magnitude
        - mag_app: apparent magnitude
        - z_guess: initial guess of z, redshift from mag_abs value should be good I think
        - cosmo: astropy cosmology

    Return:
        - z_newton: The calculated redshift for the function
    """
    # Define constants
    c = constants.c
    H0 = cosmo.H0
    omega0 = cosmo.Om0

    # Choose K-correction
    K = lambda z: 0

    # Look for zero of magnitude, luminosity distance equation
    try:
        z_newton = newton(compute_z_func, z_guess, args=(mag_abs, mag_app, cosmo))
    except RuntimeError:
        print(mag_abs, mag_app, z_guess)
        z_newton = None

    return z_newton


def double_power_law(L, A, Lstar, gamma1, gamma2, z=[], L_multiplier=1, evolution_multiplier=1):
    """Compute value of broken double power law.

    Can give a muliplier on L and on the whole function to test different evolution models.
    """

    return A / (((L * L_multiplier) / Lstar)**gamma1 + ((L * L_multiplier) / Lstar)**gamma2) * evolution_multiplier


def LADE(L, z, *, A, gamma1, gamma2, Lstar, zc, p1, p2, d, no_k=False):
    """Compute LF with LADE evolution."""
    k = (1 + zc)**p1 + (1 + zc)**p2
    if no_k is True:
        k = 1

    lade_vals_at_z = []
    for zz in z:
        eta1 = 1 / k * (((1 + zc) / (1 + zz))**p1 + ((1 + zc) / (1 + zz))**p2)
        etad = 10**(d * (1 + zz))

        lade_vals_at_z.append(np.array([L, double_power_law(L, A, Lstar, gamma1, gamma2, z=zz, L_multiplier=eta1, evolution_multiplier=etad)]).T)
        # print(lade_vals_at_z)

    return lade_vals_at_z


def IR_evol(L, z, *, A, gamma1, gamma2, zref, Lstar, k1, k2, k3, limits=None):
    """Compute LF with IR evolution."""
    lade_vals_at_z = []
    for zz in z:
        eps = np.log10((1 + zz) / (1 + zref))
        L_mult = 10**-(k1 * eps + k2 * eps**2 + k3 * eps**3)

        if limits is None:
            lade_vals_at_z.append(np.array([L, double_power_law(L, A, Lstar, gamma1, gamma2, z=zz, L_multiplier=L_mult)]).T)
        else:
            for l in limits:
                low_index = np.argwhere(L > l[0]) - 1
                high_index = np.argwhere(L > l[1])
                lade_vals_at_z.append(np.array([L[low_index:high_index], double_power_law(L, A, Lstar, gamma1, gamma2, z=zz, L_multiplier=L_mult)][low_index:high_index]).T)
        # print(lade_vals_at_z)

    return lade_vals_at_z
