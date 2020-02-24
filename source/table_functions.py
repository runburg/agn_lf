#!/usr/bin/env python3
"""
File: table_functions.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Generic functions to manage catalog data table.


Functions:
    - add_abs_mag_and_err_to_table: Add columns of absolute magnitude and corresponding error to table.


TODO:
    - Central absolute magnitude values assume a fixed redshift. Fix this?
"""
from astropy import cosmology, constants
from astropy import units as u
from scipy.optimize import newton
import numpy as np
from scipy.integrate import quad 
from source.astro_functions import compute_mag_abs, setup_cosmology


def add_abs_mag_and_err_to_table(data_table, band_mag, band_mag_err, z_columns, cosmo=None):
    """Add columns of absolute magnitude and corresponding error to table.


    For a given band with symmetric errors, add a column to the table with the computed mag_abs values.

    Input:
        - data_table: astropy Table
        - band_mag: string of band column name in table or index of column in table
        - band_mag_err: string of band error column name or index of column
        - z_columns: tuple of strings or indices of redshift columns *in order of preference*
        - cosmo: astropy cosmology. If it is None (default), will assume a FlatLambdaCDM

    Returns:
        - Names of new columns (central, low, high)
    """
    # Check cosmology
    if cosmo is None:
        cosmo = setup_cosmology()

    mag_abs_values = np.zeros(len(data_table))
    mag_abs_values[:] = np.nan
    mag_abs_err_high_values = mag_abs_values[:]
    mag_abs_err_low_values = mag_abs_values[:]
    for i, row in enumerate(data_table):
        # Define variables for readibility and check for nans!
        object_id = row['Object_1']

        mag_app = row[band_mag]
        if np.isnan(mag_app):
            print(f"Mag is nan for Object {object_id}")
            continue

        mag_app_err = row[band_mag_err]
        if np.isnan(mag_app_err):
            print(f"Mag error is nan for Object {object_id} ")
            continue

        # Choose the preferred value of redshift
        z = None
        for z_col in z_columns:
            if not np.isnan(row[z_col]):
                z = row[z_col]
                break
        else:
            print(f"No valid redshift provided in any of {z_columns}!")
            continue

        # Calculate absolute magnitude based on z value
        mag_abs_values[i] = compute_mag_abs(z, mag_app, cosmo)
        mag_abs_err_low_values[i] = compute_mag_abs(z, mag_app - mag_app_err, cosmo)
        mag_abs_err_high_values[i] = compute_mag_abs(z, mag_app + mag_app_err, cosmo)

    # Add new column to table
    mid_column_name = band_mag + '_abs'
    low_column_name = mid_column_name + '_err_low'
    high_column_name = mid_column_name + '_err_high'
    data_table[mid_column_name] = mag_abs_values * u.mag
    data_table[low_column_name] = mag_abs_err_low_values * u.mag
    data_table[high_column_name] = mag_abs_err_high_values * u.mag

    return mid_column_name, low_column_name, high_column_name


