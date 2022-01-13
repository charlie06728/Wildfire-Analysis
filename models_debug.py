"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains functions for debugging the mathematical models in models.py.
This file (and its contents) is for the purpose of debugging only.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
from models import fit_quadratic, fit_periodic, generate_quadratic_data, generate_periodic_data
from typing import Optional, List
import plotly.graph_objects as go


################################################################################
# Visualization data and fit (for debugging only)
################################################################################
# This section contains functions for visualizing how good the optimized coefficients
# generated from the model-fitting functions fit on the data. This section is for
# debugging only.

def graph_quadratic_fit(a_true: float, b_true: float, c_true: float) -> None:
    """Generate quadratic data using the three input coefficients. Calculate
    optimized coefficients by using quadratic_fit() on the generated data, and
    generate quadratic line based on optimized coefficients. Plot the data and
    the quadratic line.

    # plot the scattered graph of y = x^2 + 2 and the optimized quadratic line
    for the graph.
    >>> graph_quadratic_fit(2, 0, 2)
    """
    data = generate_quadratic_data(a_true, b_true, c_true)
    a, b, c, _ = fit_quadratic(data[0], data[1])
    calc_data = generate_quadratic_data(a, b, c)

    # plotting
    fig = go.Figure(data=go.Scatter(x=data[0], y=data[1], mode='markers'))
    fig.add_trace(go.Scatter(x=calc_data[0], y=calc_data[1], mode='lines', name='lines'))
    fig.show()


def graph_periodic_fit(a_true: float, b_true: float, c_true: float, m_true: float, n_true: float, p_true: float,
                       d_true: float, e_true: float, initial_guess: Optional[List] = None) -> None:
    """Generate periodic data using the five input coefficients. Calculate
    optimized coefficients by using periodic_fit() on the generated data, and
    generate periodic line based on optimized coefficients. Plot the data and
    the periodic line.

    Use of initial_guess see fit_periodic().

    # plot the scattered graph of y = -2 * (cos(-3 * x)) - 0.3 * x + 10 and the
    optimized periodic line for the graph.
    >>> graph_periodic_fit(-2, -3, 0, -0.3, 10, [-2, -3, 1, 1, 1])
    """
    data = generate_periodic_data(a_true, b_true, c_true, d_true, e_true)
    a, b, c, m, n, p, d, e, _ = fit_periodic(data[0], data[1], initial_guess=initial_guess)
    calc_data = generate_periodic_data(a, b, c, d, e)

    # plotting
    fig = go.Figure(data=go.Scatter(x=data[0], y=data[1], mode='markers'))
    fig.add_trace(go.Scatter(x=calc_data[0], y=calc_data[1], mode='lines', name='lines'))
    fig.show()


if __name__ == '__main__':
    ...
