#!/usr/bin/env python3
"""
File: vmax.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Functions to compute V_max for given objects
"""
import astropy.table as table
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import cosmology, constants
from astropy import units as u
import numpy as np
from scipy.integrate import dblquad
from source.astro_functions import compute_z_from_mag, setup_cosmology
import source.utils as utils


def calc_v_max(z_min, z_max, l_min, l_max, coverage, cosmo):
    """Compute V_max for the given object.

    Inputs:
        - z_min: minimum redshift for the object or redshift bin
        - z_max: maximum redshift for the object or redshift bin
        - l_min: minimum luminosity for the object or luminosity bin [erg/s]
        - l_max: maximum luminosity for the object or luminosity bin [erg/s]
        - coverage: coverage function f(l, z) of redshift and luminosity [deg^2]
        - cosmo: astropy cosmology

    Returns:
        - V_max [Mpc^3 erg/s]
    """
    deg2_to_sr = (1 * u.deg * u.deg).to(u.sr).value

    def integrand(l, z):
        return coverage(l, z) * deg2_to_sr * cosmo.differential_comoving_volume(z).value

    vmax = dblquad(integrand, z_min, z_max, lambda z: l_min, lambda z: l_max)

    # return vmax[0].decompose(bases=[u.Mpc])
    return vmax[0]


def compute_lf_vmax_per_z_bin(luminosities, v_max, lum_bins):
    """Compute value of luminosity function for each lum_bin."""
    lf = []
    for i in range(1, len(lum_bins)):
        in_bin = (lum_bins[i - 1] < luminosities) & (luminosities < lum_bins[i])
        lf.append(np.sum(1 / v_max[in_bin]))

    return np.array(lf)


def compute_binned_vmax_values(l, z, l_bins, z_bins, cosmo, coverage=(lambda l, z: 5.6), bin_l_bounds=True, bin_z_bounds=True):
    """Compute the LF values for all redshift bins."""
    z_insert_indices = np.searchsorted(z_bins, z)
    l_insert_indices = np.searchsorted(l_bins, l)

    if bin_z_bounds is False:
        z, zmin, zmax = z
    else:
        zmin = z_bins
        zmax = z_bins

    if bin_l_bounds is False:
        l, lmin, lmax = l
    else:
        lmin = l_bins
        lmax = l_bins
    # print(zmin)
    # print(z_bins)
    vmaxes = [calc_v_max(zmin[i - 1], zmax[i], lmin[j - 1], lmax[j], coverage, cosmo) for i, j in zip(z_insert_indices, l_insert_indices)]

    return np.array(vmaxes)


def compute_lf_values(l, z, vmax, z_bins, l_bins):
    """Compute LF values for all redshift bins."""
    lf_values = []
    z_insert_indices = np.searchsorted(z_bins, z)

    for zbin in range(1, len(z_bins)):
        objs_in_zbin = (z_insert_indices == zbin)
        bin_l = l[objs_in_zbin]
        v = vmax[objs_in_zbin]
        phi = compute_lf_vmax_per_z_bin(bin_l, v, l_bins)
        lf_values.append(phi)

    return np.array(lf_values)


def plot_lf_vmax(lf_values, redshift_bins, lum_bins, title='', outfile='./lf.png'):
    """Plot 1/V_max LF."""
    num_subplots = len(lf_values)
    ncols = np.ceil(np.sqrt(num_subplots))
    nrows = num_subplots // ncols + num_subplots % ncols

    fig, axs = utils.plot_setup(int(nrows), int(ncols), d=num_subplots)
    fig.subplots_adjust(hspace=0, wspace=0)

    axflat = axs.flatten()

    for i, lf_vals in enumerate(lf_values):
        label = rf'{round(redshift_bins[i], 2)} $<$ z $\leq$ {round(redshift_bins[i+1], 2)}'
        bin_centers = (lum_bins[:-1] + lum_bins[1:]) / 2
        axflat[i].plot(bin_centers, bin_centers * lf_vals, label=label, ls='', marker='o', color='xkcd:tangerine')
        axflat[i].set_xscale('log')
        axflat[i].set_xlim(left=lum_bins[0], right=lum_bins[-1])
        axflat[i].set_ylim(top=8e-4, bottom=1e-8)
        axflat[i].set_yscale('log')
        axflat[i].set_xticklabels(np.log10(axflat[i].get_xticks()).astype(int))
        # ylabs = np.logspace(-8, -3, num=6)
        # axflat[i].set_yticks(ylabs)
        # axflat[i].set_yticklabels(np.log10(ylabs)[:-1].astype(int))
        # axflat[i].set_yticklabels(axflat[i].get_yticklabels()[:-1])
        axflat[i].tick_params(axis='both', which='both', left=True, right=True, top=True, bottom=True, direction='in')

        axflat[i].legend(loc='upper right', handlelength=0, handletextpad=0, markerscale=0, fontsize='x-large', frameon=False)
        # axflat[i].set_yticklabels(axflat[i].get_yticklabels()[:-1])

    ax = fig.add_subplot(111, frameon=False)
    ax.set_xlabel(r'$\log$(L / erg s$^{-1}$)', labelpad=20)
    ax.set_ylabel(r'd$\Phi$/d$\log$L [Mpc$^{-3}$]', labelpad=35)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    fig.suptitle(title, fontsize=24, y=0.93)

    fig.savefig(outfile)

    return fig, axs, ax


def l_z_histo(l, z, l_bins, z_bins, band='band', unit=''):
    """Create a histogram of object l and z."""
    fig, ax = plt.subplots()
    histo, _, _, im = ax.hist2d(z, l, bins=[z_bins, l_bins])
    ax.set_yscale('log')
    ax.set_xlabel('Redshifts')
    ax.set_ylabel(rf'{band} luminosity [{unit}]')
    fig.colorbar(im)
    ax.set_title(rf'Histogram of luminosity \& redshift for {len(l)} {band} sources')

    return fig, ax
