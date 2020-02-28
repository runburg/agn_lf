#!/usr/bin/env python3
"""
File: lf_vmax.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Compute a luminosity function using the 1/V_max method.

These functions calculate the V_max for a given object, manage the V_max values for a catalog, and plot the luminosity function of a catalog using the 1/V_max method.

Functions:
    - add_V_max_to_table: Add column of V max values to table
    - calc_V_max: Calculate the value of V max for a given object
    - plot_lf_V_max: Plot the luminosity function using the 1/V_max method
    - v_max_counts: Get the counts for the LF

TODO:
    - v_max_counts needs selection functions/weights for errors
    - v_max_counts needs a better way to calculate magnitude errors
    - calc_V_max needs a selection function to account for incompleteness in data
"""
import astropy.table as table
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import cosmology, constants
from astropy import units as u
# from scipy.optimize import newton
import numpy as np
from scipy.integrate import quad
from source.astro_functions import compute_z_from_mag, setup_cosmology


def add_V_max_to_table(data_table, band_mag, band_mag_err, band_mag_abs, z_columns, cosmo=None):
    """Add column of V max values to table.


    For a given band with symmetric errors, add a column to the table with the computed V_max value.

    Input:
        - data_table: astropy Table
        - band_mag: string of band column name in table or index of column in table
        - band_mag_err: string of band error column name or index of column
        - band_mag_abs: string of absolute magnitude column name or index of columns
        - z_columns: tuple of strings or indices of redshift columns *in order of preference*
        - cosmo: astropy cosmology. If it is None (default), will assume a FlatLambdaCDM

    Returns:
        - Name of new column in table
    """
    # Check cosmology
    if cosmo is None:
        cosmo = setup_cosmology()

    v_max_values = np.zeros(len(data_table)) * u.Mpc**3
    for i, row in enumerate(data_table):
        # Define variables for readability and check for nans!
        object_id = row['Object_1']

        mag_app = row[band_mag]
        if np.isnan(mag_app):
            print(f"Mag is nan for Object {object_id}")
            continue

        mag_app_err = row[band_mag_err]
        if np.isnan(mag_app_err):
            print(f"Mag error is nan for Object {object_id} ")
            continue

        mag_abs = row[band_mag_abs]

        # Choose the preferred value of redshift
        z = None
        for z_col in z_columns:
            if not np.isnan(row[z_col]):
                z = row[z_col]
                break
        else:
            print(f"No valid redshift provided in any of {z_columns}!")
            continue

        # Get redshift bounds for the object
        # z_upper = compute_z_from_mag(mag_abs, mag_app + mag_app_err, z, cosmo)
        z_upper = max(data_table[z_columns[-1]])
        if z_upper is None:
            print(f"Error calculating z upper for Object {object_id}")
            continue

        # z_lower = compute_z_from_mag(mag_abs, mag_app - mag_app_err, z, cosmo)
        z_lower = 0
        if z_lower is None:
            print(f"Error calculating z lower bound for Object {object_id}")
            continue

        # Compute the V_max value
        vm = calc_V_max(z_lower, z_upper, cosmo, np.pi * 3.5**2 * u.deg * u.deg)

        # Append to list of vmax values
        v_max_values[i] = vm

    # Add new column to table
    data_table[band_mag + '_Vmax'] = v_max_values

    return band_mag + '_Vmax'


@u.quantity_input(survey_area=u.deg**2)
def calc_V_max(z_min, z_max, cosmo, survey_area):
    """Calculate the V_max for a given object.


    Input:
        - z_min: Minimum value of redshift for a given object
        - z_max: Maximum value of redshift for a given object
        - cosmo: astropy cosmology
        - survey_area: Area of the survey from which the object originates [units=deg^2]

    Returns:
        - V_max: The maximum volume the object could have taken up in the survey [units=Mpc^3]
    """
    # Define constants
    c = constants.c.to(u.Mpc / u.s)
    H0 = cosmo.H0
    omega_m = cosmo.Om0
    omega_lambda = cosmo.Ode0
    survey_area = survey_area.to(u.sr).value

    # Bastardization of Kulkarni 2019 and Cool 2012, units are Mpc^3
    V_max = c / H0 * u.Mpc**2 * survey_area * quad(lambda z: cosmo.luminosity_distance(z).to(u.Mpc).value**2 / ((1 + z)**2 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)), z_min, z_max)
    return V_max[0].decompose(bases=[u.Mpc])


def plot_lf_V_max(central_points, mag_errs, counts, counts_err, save_file=None):
    """Plot the luminosity function using the 1/V_max method.


    Inputs:
       - save_file: If not None, save plot to file.
       - central_points: Central values of the bins
       - mag_errs: Errors in the location of the central points (i.e. Variability in the location of the central point of the bin
       - counts: Counts in the bins
       - counts_err: Poissonian errors of the counts


    Returns:
        - fig, ax: Figure and axes of plot
   """
    # Create plot
    fig, ax = plt.subplots()
    errbar = ax.errorbar(central_points, counts, yerr=counts_err, fmt='o')
    ax.set_ylabel(r"$\log_{10}(\phi/\mathrm{Mpc^3mag^{-1}})$")
    ax.set_yscale('log')
    ax.set_xlabel("Magnitude [mag]")
    ax.set_xlim((max(central_points) + 1, min(central_points) - 1))

    # Save fig if requested
    if save_file is not None:
        fig.savefig(save_file)

    return fig, ax


def example_selection_func(magnitudes):
    """Compute the selection function for the given magnitudes.

    Surveys are incomplete for various reason. This function serves to correct for that incompleteness by calculating corrections to the luminosity function as a function of magnitude.


    Inputs:
        - magnitudes: the array of object magnitudes being used to construct the selection function

    Returns:
        - An array of correction factors
    """

    return 5e16 * np.exp(2.2 * magnitudes) + 1


def v_max_counts(data_table, bins, band, selection_function=None, vmax_band=None, band_err=None):
    """Return the counts for the 1/V_max luminosity function.

    The counts are for the bins given. The error in the counts is assumed to be Poissonian and form is given in Cool 2012.


    Inputs:
        - data_table: The table with the V_max counts and absolute magnitudes
        - band: Column name of band to use magnitudes of
        - vmax_band: Column name of vmax for given band. Default value is: band.strip('_abs')+'_Vmax'
        - bins: Number of bins for luminosity function. If an integer, use equally spaced bins with number = bins. If a list, use bins as histogram bins. Default value is: 10
        - selection_function: Weights from selection function. Default value is: 1
        - band_err: Column name of errors in magnitudes

    Returns:
        - bins: central bin values for plotting
        - st_dev_bin_mag: standard deviation of the magnitudes in a given bin
        - counts: the number of objects in each bin
        - count_err: the Poissonian error in of each count
    """
    # Set values based on function arguments
    if vmax_band is None:
        vmax_band = band.strip('_abs') + '_Vmax'
    if isinstance(bins, int):
        bins = np.linspace(min(data_table[band]), -5, num=bins)
    if selection_function is None:
        selection_function = 1

    # Initialize luminosity counts
    counts = []
    counts_err = []
    avg_mag_err = []
    central_points = (bins[:-1] + bins[1:]) / 2

    # Sort the magnitudes into their appropriate bins
    # NEEDS WORK: could use existing histogram, but also should incorporate error
    for i in range(1, len(bins)):
        # Find the objects that are in the bin
        has_vmax = data_table[vmax_band].data != 0
        # print(has_vmax)
        not_nan = ~np.isnan(data_table[band].data)
        in_bin = (bins[i - 1] < data_table[band].data) & (data_table[band].data < bins[i])
        # print(np.sum(in_bin))
        passing_indices = in_bin & not_nan & has_vmax

        # Add counts for the bin
        counts.append(1 / (bins[i] - bins[i - 1]) * np.sum(selection_function[passing_indices] / data_table[vmax_band][passing_indices]))
        counts_err.append(1 / (bins[i] - bins[i - 1]) * np.sqrt(np.sum(selection_function[passing_indices]**2 / data_table[vmax_band][passing_indices]**2)))
        avg_mag_err.append(np.std(data_table[band][passing_indices]))

    return central_points, avg_mag_err, counts, counts_err


if __name__ == "__main__":
    from table_functions import add_abs_mag_and_err_to_table, setup_cosmology
    catalog = fits.open("data/Anna_Nick_Tractor_photometry_samplev9_MASTER.fits", memmap=True)
    ct = table.Table(catalog[1].data)
    flagged_agn = ct[np.where(ct['agn_flag'] == 1)]

    cosmo = setup_cosmology()
    band = 'Mag_U_CFHT'
    survey_area = np.pi * 3.5**2
    band_err = 'Mag_err_U_CFHT'
    band_mag_abs, _, _ = add_abs_mag_and_err_to_table(flagged_agn, band, band_err, survey_area, ('zSpec', 'z_eazy'), cosmo=cosmo)
    add_V_max_to_table(flagged_agn, band, band_err, band_mag_abs, ('zSpec', 'z_eazy'), cosmo=cosmo)

    fig, ax = plot_lf_V_max(flagged_agn, band)
    plt.show()
