#!/usr/bin/env python3
"""
File: lf_xi2.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Compute the luminosity function fitting to a power law by minimizing x1^2.


Functions:
    - double_power_law: The function to fit against

TODO:
    -
"""
import numpy as np
import matplotlib.pyplot as plt
import source.lf_vmax as vmax
import source.astro_functions as af


def double_power_law(params, m):
    """The function to fit magnitude data.

    Functional form and notation as given in Kulkarni.


    Inputs:
        - m: Magnitude of object
        - params: (phi_star, m_star, alpha, beta):
            - phi_star: Amplitude of LF
            - m_star: Break magnitude (where the break happens)
            - alpha: Faint-end slope
            - beta: Near-end slope

    Returns:
        - Value of luminosity function
    """
    phi_star, m_star, alpha, beta = params

    return phi_star / (10**(0.4 * (alpha + 1) * (m - m_star)) + 10**(0.4 * (beta + 1) * (m - m_star)))


def odr_fit(x_data, x_data_err, y_data, y_data_err, fitfunc=double_power_law, initial_guesses=(100, -20, 0, -0.5)):
    """Fit data to a function using ODR fitting.


    Inputs:
        - x_data: Data along x-axis
        - x_data_err: Error of data along x-axis
        - y_data: Data along y-axis
        - y_data_err: Error in data along y-axis
        - fitfunc: Function to fit data to [default value = double_power_law]

    Returns:
        - ODR output object
    """
    from scipy import odr

    # Instantiate classes for ODR run
    fit_func = odr.Model(fitfunc)
    mydata = odr.RealData(x_data, y_data, sx=x_data_err, sy=y_data_err)

    # Instantiate ODR
    myodr = odr.ODR(mydata, fit_func, beta0=initial_guesses)

    # Get output
    output = myodr.run()
    output.pprint()

    return output
