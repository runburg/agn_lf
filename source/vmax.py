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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astropy.io import fits
from astropy import cosmology, constants
from astropy import units as u
from scipy.integrate import dblquad
from source.astro_functions import compute_z_from_mag, setup_cosmology
import source.utils as utils
from scipy.interpolate import interp1d


def calc_v_max(z_min, z_max, l_min, l_max, coverage, cosmo, use_dblquad=True):
    """Compute V_max for the given object.

    Inputs:
        - z_min: minimum redshift for the object or redshift bin
        - z_max: maximum redshift for the object or redshift bin
        - l_min: minimum luminosity for the object or luminosity bin [erg/s]
        - l_max: maximum luminosity for the object or luminosity bin [erg/s]
        - coverage: coverage function f(l, z) of redshift and luminosity [deg^2]
        - cosmo: astropy cosmology
        - use_dblquad: use quadrature to evaluate double integral rather than simps rule

    Returns:
        - V_max [Mpc^3 erg/s]
    """
    deg2_to_sr = (1 * u.deg * u.deg).to(u.sr).value

    if np.any(z_min < 0):
        print("BAD ZMIN")

    def integrand(l, z):
        cov = coverage(l, z)
        cov[cov < 0] = 1e-10 
        # print(cov)
        return cov * deg2_to_sr * cosmo.differential_comoving_volume(z).value * l / l

    if use_dblquad is True:
        vmax = dblquad(integrand, z_min, z_max, lambda z: l_min, lambda z: l_max)
        return vmax[0]

    z = np.linspace(z_min, z_max, num=25)
    l = np.logspace(np.log10(l_min), np.log10(l_max), num=20)[:, np.newaxis, :]

    integrand_vals = integrand(l, z[np.newaxis, :, :])

    if np.any(integrand_vals) < 0:
        print("negative coverage/comoving vol detected", np.sum(integrand_vals < 0))
    vmax = np.trapz(np.trapz(integrand_vals, l, axis=0), z, axis=0)

    if np.any(vmax < 0):
        print("negative vmax detected")
        print(np.sum(vmax < 0), "negative values")
        print("bad z bounds", np.sum(z_max - z_min < 0))
    return vmax


def compute_lf_vmax_per_z_bin(luminosities, v_max, lum_bins, poisson_error=False):
    """Compute value of luminosity function for each lum_bin."""
    lf = []
    errors = []

    def poiereror(n, S=1):
        return (n+1) * (1 - 1 / (9 * (n+1)) + S / (3 * np.sqrt(n+1)))**3

    for i in range(1, len(lum_bins)):
        in_bin = (lum_bins[i - 1] < luminosities) & (luminosities < lum_bins[i])
        lf.append(np.sum(1 / v_max[in_bin], where=(v_max[in_bin]>0)&(~np.isnan(v_max[in_bin]))))
        if poisson_error is False:
            errors.append(np.sqrt(np.sum((1 / v_max[in_bin])**2), where=v_max[in_bin]>0))
        else:
            upper_error = poiereror(np.sum(in_bin))
            lower_error = poiereror(np.sum(in_bin) - 1)
            errors.append([upper_error, lower_error])

    return np.array(lf), np.array(errors)


def compute_binned_vmax_values(l, z, l_bins, z_bins, cosmo, coverage=(lambda l, z: 8), bin_l_bounds=True, bin_z_bounds=True):
    """Compute the LF values for all redshift bins."""
    if bin_z_bounds is False:
        z, zlower, zupper = z

    if bin_l_bounds is False:
        l, llower, lupper = l

    z_insert_indices = np.searchsorted(z_bins, z)
    # print(z_insert_indices)
    l_insert_indices = np.searchsorted(l_bins, l)

    zmin = z_bins[z_insert_indices - 1]
    zmax = z_bins[z_insert_indices]

    lmin = l_bins[l_insert_indices - 1]
    lmax = l_bins[l_insert_indices]

    if bin_z_bounds is False:
        zmaxbin = np.minimum(zmax, zupper)
        zminbin = np.maximum(zmin, zlower)

        good_indices = (zmaxbin > zminbin)
        zmax[good_indices] = zmaxbin[good_indices][:]
        zmin[good_indices] = zminbin[good_indices][:]

        # print(zmin)
    # print(z_bins)
    # vmaxes = [calc_v_max(zzmin, zzmax, llmin, llmax, coverage, cosmo, use_dblquad=False) for i, j, zzmin, zzmax, llmin, llmax in zip(z_insert_indices, l_insert_indices, zmin, zmax, lmin, lmax)]
    vmaxes = calc_v_max(zmin, zmax, lmin, lmax, coverage, cosmo, use_dblquad=False)

    return np.array(vmaxes)


def compute_lf_values(l, z, vmax, z_bins, l_bins, poisson_error=False):
    """Compute LF values for all redshift bins."""
    lf_values = []
    lf_errors = []
    z_insert_indices = np.searchsorted(z_bins, z)

    for zbin in range(1, len(z_bins)):
        objs_in_zbin = (z_insert_indices == zbin)
        bin_l = l[objs_in_zbin]
        v = vmax[objs_in_zbin]
        phi, error_phi = compute_lf_vmax_per_z_bin(bin_l, v, l_bins, poisson_error=poisson_error)
        lf_values.append(phi)
        lf_errors.append(error_phi)

    return np.array(lf_values), np.array(lf_errors)


def compute_zmax(l, z, cosmo, flux_limit, zspacing=0.1, jack_version=False, output=True):
    """Return the maximum z such that the object's flux would be above the limit."""
    import astropy.units as u

    if jack_version is False:
        from astropy.cosmology import z_at_value

        def func(lum):
            return (np.sqrt(lum / (4 * np.pi * flux_limit)) * u.cm).to(u.Mpc)

        zmin = z_at_value(cosmo.luminosity_distance, func(l.min()), zmin=1e-16, zmax=8)
        zmax = z_at_value(cosmo.luminosity_distance, func(l.max()), zmax=50)
        zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 50)

        lgrid = cosmo.luminosity_distance(zgrid)

        zmaxes = np.interp(func(l).value, lgrid.value, zgrid)
        if np.any(zmaxes < z) and output is True:
            print("bad zmax inference", np.sum(zmaxes < z))
            print("this is probably due to an issue with the flux limit")
            print("zmax/zmin will default to zbin range")

        return zmaxes

    zmaxes = []

    zspacing_orig = zspacing
    for ll, zz in zip(l, z):
        diff = 0.1
        ztrial = zz
        fine = False
        zspacing = zspacing_orig
        while diff > 0:
            ztrial += zspacing
            diff = ll / (4 * np.pi * cosmo.luminosity_distance(ztrial).to(u.cm).value**2) - flux_limit
            if diff < 0 and fine is False:
                fine = True
                diff = 0.1
                ztrial -= zspacing
                zspacing /= 10
        zmaxes.append(ztrial)

    return np.array(zmaxes)


def coverage_correction(full_fluxes, selected_fluxes):
    """Compute the correction to the coverage at each flux.

    Returns an interpolated function."""
    from astropy.stats import histogram
    from scipy.interpolate import interp1d

    # full_binned, bin_vals = histogram(full_fluxes, bins='blocks')
    # selected_binned, bin_edges = histogram(selected_fluxes, bins=bin_vals)

    bins = np.logspace(np.log10(full_fluxes.min())*0.99999999, np.log10(full_fluxes.max())*1.00000001, num=20)
    full_binned, bin_vals = np.histogram(full_fluxes, bins=bins)
    selected_binned, bin_edges = histogram(selected_fluxes, bins=bins)

    if np.any(selected_binned) == 0:
        print(np.sum(selected_binned == 0), 'zeros in coverage correction')

    corrections = selected_binned / full_binned
    bin_centers = (bin_vals[1:] + bin_vals[:-1]) / 2
    # print('corrections', corrections)

    return interp1d(bin_centers, corrections, fill_value=1, bounds_error=False, kind='nearest')


def coverage_function(data, wcs, xlength, ylength, detector_area, photon_energy, truncate_radec=1):
    """Calculate the coverage as a function of flux."""

    x = np.arange(xlength)
    y = np.arange(ylength)
    X, Y = np.meshgrid(x, y)
    ra, dec = wcs.wcs_pix2world(X, Y, 0)

    if truncate_radec != 1:
        truncate_radec = (ra[:-1, :-1] > truncate_radec[0]) & (ra[:-1, :-1] < truncate_radec[1]) & (dec[:-1, :-1] > truncate_radec[2]) & (dec[:-1, :-1] < truncate_radec[3])

    areas = np.abs((ra[:-1, :-1] - ra[:-1, 1:]) * (dec[1:, :-1] - dec[:-1, :-1])) * truncate_radec

    fluxes = 1/(data[:-1, :-1] * truncate_radec).flatten()
    fluxes[fluxes < 0] = 0
    good_indices = (~np.isinf(fluxes) & (fluxes > 0))
    areas = areas.flatten()[good_indices]
    fluxes = fluxes[good_indices] / detector_area * photon_energy

    sort_indices = np.argsort(fluxes)
    fluxes = fluxes[sort_indices]
    areas = np.cumsum(areas[sort_indices])

    cov_func = interp1d(fluxes, areas, fill_value='extrapolate')

    print('The total coverage is', areas[-1], 'square degrees.')

    return cov_func


def plot_lf_vmax(lf_values, lf_errors, redshift_bins, lum_bins, this_label='This analysis', title='', lum_limits=[], compare_to_others={}, other_runs={}, outfile='./lf.png', lum_sublabel='', others_limits={}, x_ray_correction={}):
    """Plot 1/V_max LF."""
    num_subplots = len(lf_values)
    ncols = np.ceil(np.sqrt(num_subplots))
    nrows = num_subplots // ncols + num_subplots % ncols

    fig, axs = utils.plot_setup(int(nrows), int(ncols), d=num_subplots, set_global_params=True)
    fig.subplots_adjust(hspace=0, wspace=0)

    axflat = axs.flatten()

    vmax_legend = []
    lit_legend = []

    bin_centers = (lum_bins[:-1] + lum_bins[1:]) / 2

    lum_errors = (lum_bins[1:] - lum_bins[:-1]) / 2

    for i, lf_vals in enumerate(lf_values):
        label = rf'{round(redshift_bins[i], 2)} $<$ z $\leq$ {round(redshift_bins[i+1], 2)}'

        lum_lim_index = 0
        if len(lum_limits) > 0:
            # axflat[i].axvline(lum_limits[i], color='xkcd:pale peach', lw=2)
            lum_lim_index = np.argmax(lum_limits[i] < bin_centers)

        axflat[i].errorbar(bin_centers[lum_lim_index:], np.log(10) * bin_centers[lum_lim_index:] * lf_vals[lum_lim_index:], xerr=lum_errors[lum_lim_index:], yerr=bin_centers[lum_lim_index:]*lf_errors[i][:, lum_lim_index:]*np.log(10), label=label, ls='', marker='o', color='xkcd:orangered', capsize=3, zorder=100)

        axflat[i].set_xlim(left=lum_bins[0], right=lum_bins[-1])
        axflat[i].set_xscale('log')
        axflat[i].set_xticklabels(np.log10(axflat[i].get_xticks()).astype(int))

        axflat[i].set_ylim(top=1e-3, bottom=2e-8)
        axflat[i].set_yscale('log')

        axflat[i].tick_params(axis='both', which='both', left=True, right=True, top=True, bottom=True, direction='in', labelsize=12)
        handles, labels = axflat[i].get_legend_handles_labels()
        axflat[i].legend(handles=handles[0], labels=labels, loc='upper right', handlelength=0, handletextpad=0, markerscale=0, fontsize='x-large', frameon=False)

    ax = fig.add_subplot(111, frameon=False)
    ax.set_xlabel(rf'$\log_{{10}}$($L{lum_sublabel}$ / erg s$^{{-1}}$)', labelpad=20, fontsize=20)
    ax.set_ylabel(rf'd$\Phi$/d$\log_{{10}}$ $L{lum_sublabel}$ [Mpc$^{{-3}}$]', labelpad=35, fontsize=20)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    colors = (color for color in ['xkcd:petrol', 'xkcd:royal', 'xkcd:red wine', 'xkcd:lilac', 'xkcd:cerulean', 'xkcd:aquamarine', 'xkcd:wine', 'xkcd:coral', 'xkcd:seafoam green'])
    offset = (off for off in [1.05, 0.95, 1.1, 0.9, 1.25, 0.975])
    hatches = (h for h in ['///', '\\\\\\'])
    hatches = (h for h in ['|||', '---'])

    for other in other_runs:
        lf_values = other_runs[other][0]
        lf_errors = other_runs[other][1]

        color = next(colors)
        offs = next(offset)
        for i, lf_vals in enumerate(lf_values):
            lum_lim_index = 0
            if len(lum_limits) > 0:
                # axflat[i].axvline(lum_limits[i], color='xkcd:pale peach', lw=2)
                lum_lim_index = np.argmax(lum_limits[i] < bin_centers)

            axflat[i].errorbar(bin_centers[lum_lim_index:] * offs, np.log(10) * bin_centers[lum_lim_index:] * lf_vals[lum_lim_index:], xerr=lum_errors[lum_lim_index:], yerr=bin_centers[lum_lim_index:]*lf_errors[i][:, lum_lim_index:]*np.log(10), label=other, ls='', marker='p', color=color, capsize=3, markersize=8, markerfacecolor='xkcd:white')

        vmax_legend.append(Line2D([0], [0], marker='p', color=color, ls='', markersize=8, label=other, markerfacecolor='xkcd:white'))

    colors = (color for color in ['xkcd:silver', 'xkcd:dusty lavender'])
    if len(compare_to_others) > 0:
        vmax_legend.append(Line2D([0], [0], marker='o', color='xkcd:orangered', ls='', markersize=10, label=this_label))
        for author in compare_to_others:
            color = next(colors)
            hatch = next(hatches)
            try:
                lims = others_limits[author]
            except KeyError:
                lims = None
                print('No limits for', author)
            if len(compare_to_others[author][0]) == 3:
                marker = ''
                ls = '-'
                alpha = 0.7
            else:
                marker = 'o'
                ls = ''
                alpha = 1
            # lit_legend.append(Line2D([0], [0], marker=marker, color=color, lw=3, markersize=10, label=author, ls=ls))
            lit_legend.append(Patch(edgecolor=color, facecolor='none', hatch=hatch, label=author))
            offs = next(offset)
            for i, vals in enumerate(compare_to_others[author]):
                if len(vals) == 3:
                    upper_index = None
                    lower_index = None
                    # axflat[i].plot(vals[0][lower_index:upper_index, 0], vals[0][lower_index:upper_index, 1], ls='--', marker=marker, color=color, alpha=0.3)
                    axflat[i].fill_between(vals[1][lower_index:upper_index, 0], vals[1][lower_index:upper_index, 1], vals[2][lower_index:upper_index, 1], ls='--', facecolor='none', edgecolor=utils.lighten_color(color, amount=0.3), alpha=0.1, hatch=hatch)
                    # axflat[i].fill_between(vals[1][lower_index:upper_index, 0], vals[1][lower_index:upper_index, 1], vals[2][lower_index:upper_index, 1], ls='--', facecolor='none', edgecolor=color, alpha=0.1, hatch=hatch)
                    if lims is not None:
                        lower_index = np.argmax(lims[i][0] < vals[0][:, 0]) - 1
                        upper_index = np.argmax(lims[i][1] < vals[0][:, 0])
                        if upper_index == 0: upper_index = None
                        # if lower_index < 0: lower_index = 0
                        # print(lower_index)
                        # print(lims[i][0], lower_index, upper_index, vals[0][:, 0])
                        # axflat[i].plot(vals[0][lower_index:upper_index, 0], vals[0][lower_index:upper_index, 1], ls=ls, marker=marker, color=color, alpha=0.5)
                        axflat[i].fill_between(vals[1][lower_index:upper_index, 0], vals[1][lower_index:upper_index, 1], vals[2][lower_index:upper_index, 1], ls=ls, facecolor='none', edgecolor=color, alpha=0.4, hatch=hatch)
                else:
                    axflat[i].plot(vals[:, 0] + offs, vals[:, 1], ls=ls, marker=marker, color=color)
        # ax.legend(handles=big_legend, bbox_to_anchor=(1.01, 0.5), loc='center left')
    lil_legend = ax.legend(handles=vmax_legend, loc='lower left')
    ax.add_artist(lil_legend)
    ax.legend(handles=lit_legend, loc='lower left', bbox_to_anchor=(0.67, 0))

    fig.suptitle(title, fontsize=24, y=0.93)

    fig.savefig(outfile)

    return fig, axs, ax


def l_z_histo(l, z, l_bins, z_bins, band='band', unit=''):
    """Create a histogram of object l and z."""
    from matplotlib import cm

    fig, ax = plt.subplots()
    histo, _, _, im = ax.hist2d(z, l, bins=[z_bins, l_bins], cmap=cm.cividis)
    ax.set_yscale('log')
    ax.set_xlabel('Redshifts')
    ax.set_ylabel(rf'$L_{{{band}}}$ [{unit}]')
    fig.colorbar(im)
    # ax.set_title(rf'Histogram of luminosity \& redshift for {len(l)} {band} sources')

    return fig, ax, histo


def incompleteness_histo(full_sample_fluxes, agn_fluxes, flux_bins):
    """Histogram the full sample and agn_fluxes to investigate incompleteness."""
    fig, ax = utils.plot_setup(1, 1, set_global_params=True)

    # from astropy.visualization import hist
    colors = ['xkcd:very light pink', 'xkcd:pastel pink']
    labels = ['Full sample', 'Selected AGN']
    ax.hist((full_sample_fluxes, agn_fluxes), bins=flux_bins, histtype='bar', stacked=True, color=colors, label=labels)

    ax.set_xlabel(r"Flux [erg s$^{-1}$ cm$^{-2}$]")
    ax.set_xscale('log')
    ax.set_facecolor('xkcd:dark purple')
    ax.set_title("Histo for incompleteness")
    ax.legend(loc='upper right')

    return fig, ax


def exposure_plot(wcs, data, survey='', band='', outfile='', truncate_radec=None):
    """Plot the exposure map for the given survey and band."""
    fig = plt.figure()

    ax = fig.add_subplot(111, projection=wcs)
    ax.imshow(data, cmap=plt.cm.cividis)
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    if truncate_radec is not None:
        ax.plot([truncate_radec[0]]*2+[truncate_radec[1]]*2+[truncate_radec[0]], [truncate_radec[2], truncate_radec[3], truncate_radec[3], truncate_radec[2], truncate_radec[2]], transform=ax.get_transform('world'))

    lon = ax.coords[0]
    print(lon)
    lat = ax.coords[1]
    lon.set_major_formatter('d')
    lat.set_major_formatter('d')
    # ax.set_title(f'{survey} exposure map for {band} in ICRS coordinates')
    if len(outfile) < 1:
        fig.savefig('./output/exposure_map.png')
    else: 
        fig.savefig(outfile, bbox_inches='tight')

    return fig, ax


def cov_func_plot(cov_func, min_logflux, max_logflux, truncate_radec=None, comparison_data=(), survey='', band='', outfile=''):
    """Plot coverage as a function of flux."""
    fig, ax = utils.plot_setup(1, 1, set_global_params=True)

    fluxes = np.logspace(min_logflux, max_logflux, num=100)
    ax.plot(fluxes, cov_func(fluxes), label="This analysis", color='xkcd:raspberry')
    print('Total coverage for:', cov_func(fluxes[-1]), 'square degrees.')

    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(left=min_logflux/100, right=max_logflux*10)
    ax.set_ylim(bottom=1e-2)

    if truncate_radec is not None:
        ax.axvline(truncate_radec[0])
        ax.axvline(truncate_radec[1])
        ax.axhline(truncate_radec[2])
        ax.axhline(truncate_radec[3])

    colors = iter(['xkcd:blurple', 'xkcd:periwinkle', 'xkcd:bubblegum'])
    for label, data in comparison_data:
        ax.plot(data[:, 0], data[:, 1], label=label, color=next(colors))

    # ax.set_title(f'{survey} flux coverage for {band} band')
    ax.set_xlabel(r'Flux [erg cm$^{-2}$ s$^{-1}$]')
    ax.set_ylabel(r'Coverage [deg$^2$]')
    
    if len(outfile) < 1:
        fig.savefig('./output/coverage_flux_plot.png')
    else:
        fig.savefig(outfile, bbox_inches='tight')

    return fig, ax


def fracerr_poisson(n):
    # sigma_upper
    n_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    # from tables 1+2 of Gehrels 1986
    cl_upper = np.array([1.841, 3.300, 4.638, 5.918, 7.163, 8.382, 9.584, 10.77, 11.95, 13.11, 14.27, 15.42, 16.56, 17.70, 18.83, 19.96])
    cl_lower = np.array([0.0, 0.173, 0.708, 1.367, 2.086, 2.840, 3.620, 4.419, 5.232, 6.057, 6.891, 7.734, 8.585, 9.441, 10.30, 11.17])
    if n == 0:
        frac_err_upper = 0
        frac_err_lower = 0
    elif (n <= 15):
        hit = (n_values == n)
        # frac_err i..e dN/N
        frac_err_upper = (cl_upper[hit] - n) / n
        frac_err_lower = (n - cl_lower[hit]) / n
    else:
        # this is good for the lower limit but somewhat underestimates the upper limit (but not too significant at these n)
        frac_err_upper = np.sqrt(n) / n
        frac_err_lower = np.sqrt(n) / n

    # print(n, frac_err_lower, frac_err_upper)
    return [frac_err_lower, frac_err_upper]


def poisson_errors(l, z, l_bins, z_bins):
    """Compute the poisson errors for the binned data."""
    upper_error = np.zeros((len(z_bins), len(l_bins)))
    lower_error = np.zeros((len(z_bins), len(l_bins)))

    counts, *_ = np.histogram2d(z, l, bins=[z_bins, l_bins])

    for i in range(len(z_bins) - 1):
        for j in range(len(l_bins) - 1):
            lower_error[i, j], upper_error[i, j] = fracerr_poisson(counts[i, j])

    return [lower_error, upper_error]
