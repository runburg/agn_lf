#!/usr/bin/env python3
"""
File: selection_criteria.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Selection criteria for AGN at different wavelengths

Functions:
    - select_ir: Return selection based on IRAC bands


TODO:
    - X-ray: donley
    - ir: stern
"""
import numpy as np
import matplotlib.pyplot as plt
import source.lf_vmax as vmax
import source.astro_functions as af


def select_ir(data_table, flux3_6, flux4_5, flux5_8, flux8_0, selection_cuts='lacy05'):
    """Select AGN using IRAC criteria.

    Different color cuts have been proposed to select AGN in IR bands.
    Lacy et al. 2007, Donley et al. 2012, and Stern 2005 are implemented.


    Inputs:
        - data_table: Table containing objects for selecting
        - flux3_6: Name of column with 3.6 um flux
        - flux4_5: Name of column with 4.5 um flux
        - flux5_8: Name of column with 5.8 um flux
        - flux8_0: Name of column with 8.0 um flux
        - selection_cuts: The selection cuts to use. Options are 'lacy05', 'stern05', 'donley12' [default value = 'lacy05']

    Returns:
        - Indices of IR selected AGN
    """
    # Grab the data from the table
    flux3_6 = data_table[flux3_6]
    flux4_5 = data_table[flux4_5]
    flux5_8 = data_table[flux5_8]
    flux8_0 = data_table[flux8_0]

    # log10 ratio of 8.0 and 4.5 fluxes
    lf80_45 = np.log10(flux8_0 / flux4_5)
    lf58_36 = np.log10(flux5_8 / flux3_6)

    if selection_cuts == 'lacy05':
        agn_candidates = (lf80_45 > -0.2) & (lf58_36 > -0.1) & (lf80_45 < (0.8 * lf58_36 + 0.5))
    elif selection_cuts == 'stern05':
        pass
    elif selection_cuts == 'donley12':
        agn_candidates = (lf80_45 >= 0.15) & (lf58_36 >= 0.08) & (lf80_45 >= 1.21 * lf58_36 - 0.27) & (lf80_45 >= 1.21 * lf58_36 + 0.27) & (flux8_0 > flux5_8 > flux4_5 > flux3_6)
        # Donley also gives criteria for large z but they are not implemented yet

    return agn_candidates

