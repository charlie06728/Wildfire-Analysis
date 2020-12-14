"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains the mathematical models used to fit the data.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
from typing import List, Tuple
import math
import numpy as np
import scipy.optimize


################################################################################
# Models
################################################################################
# This section contains all the mathematical models that can be used to fit data.
# The models include linear, quadratic, inverse, and periodic. See fit_linear,
# fit_quadratic, fit_inverse, and fit_periodic for usage details.

def fit_exponential(data_x: List[float], data_y: List[float]) -> Tuple[float, float, float, float]:
    """Return the optimized values of a, b in the equation y = a * (b^x) + c for fitting
    the provided data, and the root mean squared error of the model.
    data_x is the list of x coordinates of the data points and data_y is the
    corresponding list of y coordinates of the data points.

    Note: this function uses scipy.optimize.curve_fit() to find the optimized values of a, b.

    Preconditions:
        - len(data_x) == len(data_y)

    Sample usage refer to test_linear_model().
    """
    optimized_parameters = scipy.optimize.curve_fit(exponential, data_x, data_y, maxfev=5000)
    a, b, c = optimized_parameters[0]
    prediction_data = [exponential(x_i, a, b, c) for x_i in data_x]
    rmse = calculate_rmse(data_y, prediction_data)
    return (a, b, c, rmse)


def fit_quadratic(data_x: List[float], data_y: List[float]) -> Tuple[float, float, float, float]:
    """Return the optimized values of a, b, c in the quadratic line equation
    y = a * x^2 + b * x + c for fitting the provided data, and the root mean squared error of the model.
    data_x is the list of x coordinates of the data points and data_y is the corresponding list
    of y coordinates of the data points.

    Note: this function uses scipy.optimize.curve_fit() to find the optimized values of a, b, c.

    Preconditions:
        - len(data_x) == len(data_y)

    Sample usage refer to test_quadratic_model().
    """
    optimized_parameters = scipy.optimize.curve_fit(quadratic, data_x, data_y)
    a, b, c = optimized_parameters[0]
    prediction_data = [quadratic(x_i, a, b, c) for x_i in data_x]
    rmse = calculate_rmse(data_y, prediction_data)
    return (a, b, c, rmse)


def fit_inverse(data_x: List[float], data_y: List[float]) -> Tuple[float, float, float]:
    """Return the optimized values of a, b in the equation y = a / x + b for fitting
    the provided data, and the root mean squared error of the model.
    data_x is the list of x coordinates of the data points and data_y is the corresponding
    list of y coordinates of the data points.

    Note: this function uses scipy.optimize.curve_fit() to find the optimized values of a, b.

    Preconditions:
        - len(data_x) == len(data_y)

    Sample usage refer to test_inverse_model().
    """
    optimized_parameters = scipy.optimize.curve_fit(inverse, data_x, data_y)
    a, b = optimized_parameters[0]
    prediction_data = [inverse(x_i, a, b) for x_i in data_x]
    rmse = calculate_rmse(data_y, prediction_data)
    return (a, b, rmse)


def fit_logarithm(data_x: List[float], data_y: List[float]) -> Tuple[float, float, float]:
    """Return the optimized values of a, b in the equation y = a * log(x) + b for fitting
    the provided data, and the root mean squared error of the model.
    data_x is the list of x coordinates of the data points and data_y is the corresponding
    list of y coordinates of the data points.

    Note: this function uses scipy.optimize.curve_fit() to find the optimized values of a, b.

    Preconditions:
        - len(data_x) == len(data_y)

    Sample usage refer to test_inverse_model().
    """
    optimized_parameters = scipy.optimize.curve_fit(logarithm, data_x, data_y)
    a, b = optimized_parameters[0]
    prediction_data = [logarithm(x_i, a, b) for x_i in data_x]
    rmse = calculate_rmse(data_y, prediction_data)
    return (a, b, rmse)


def fit_periodic(data_x: List[float], data_y: List[float],
                 initial_guess: Tuple[float, float, float, float, float, float, float, float],
                 test_range: Tuple[float, float]) \
        -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Return the optimized values of a, b, c, d, e in the periodic line equation
    y = a * (cos(b * (x - c))) + m * (cos(n * (x - p))) + d * x + e for fitting the provided data, and
    the root mean squared error of the model.(note: the term d * x is used for compensating
    increasing/decreasing trend of the overall data).

    data_x is the list of x coordinates of the data points and data_y is the corresponding
    list of y coordinates of the data points.

    initial_guess is an optional list for setting the initial values of the coefficients
    a, b, c, d, e prior to being optimized. Pass in initial_guess if the optimized coefficients
    does not appear to be fitting the data optimally (scipy.optimize.curve_fit() may be stuck at
    local minimums when trying to minimize the loss function).

    Note: this function uses scipy.optimize.curve_fit() to find the optimized values of a, b, c.

    Preconditions:
        - len(data_x) == len(data_y)

    Sample usage refer to test_periodic_model().
    """
    a0 = initial_guess[0]
    b0 = initial_guess[1]
    c0 = initial_guess[2]
    m0 = initial_guess[3]
    n0 = initial_guess[4]
    p0 = initial_guess[5]
    d0 = initial_guess[6]
    e0 = initial_guess[7]
    begin = test_range[0]
    end = test_range[1]
    a_possible = list(np.linspace(begin * a0, end * a0, 100))
    b_possible = list(np.linspace(begin * b0, end * b0, 100))
    c_possible = list(np.linspace(begin * c0, end * c0, 100))
    m_possible = list(np.linspace(begin * m0, end * m0, 100))
    n_possible = list(np.linspace(begin * n0, end * n0, 100))
    p_possible = list(np.linspace(begin * p0, end * p0, 100))
    d_possible = list(np.linspace(begin * d0, end * d0, 100))
    e_possible = list(np.linspace(begin * e0, end * e0, 100))
    best_so_far = (a0, b0, c0, m0, n0, p0, d0, e0, math.inf)
    for _ in range(50):
        for a in a_possible:
            b = best_so_far[1]
            c = best_so_far[2]
            m = best_so_far[3]
            n = best_so_far[4]
            p = best_so_far[5]
            d = best_so_far[6]
            e = best_so_far[7]
            prediction_data = [periodic(x, a, b, c, m, n, p, d, e) for x in data_x]
            rmse = calculate_rmse(data_y, prediction_data)
            if rmse < best_so_far[8]:
                best_so_far = (a, b, c, m, n, p, d, e, rmse)
        for b in b_possible:
            a = best_so_far[0]
            c = best_so_far[2]
            m = best_so_far[3]
            n = best_so_far[4]
            p = best_so_far[5]
            d = best_so_far[6]
            e = best_so_far[7]
            prediction_data = [periodic(x, a, b, c, m, n, p, d, e) for x in data_x]
            rmse = calculate_rmse(data_y, prediction_data)
            if rmse < best_so_far[8]:
                best_so_far = (a, b, c, m, n, p, d, e, rmse)
        for c in c_possible:
            a = best_so_far[0]
            b = best_so_far[1]
            m = best_so_far[3]
            n = best_so_far[4]
            p = best_so_far[5]
            d = best_so_far[6]
            e = best_so_far[7]
            prediction_data = [periodic(x, a, b, c, m, n, p, d, e) for x in data_x]
            rmse = calculate_rmse(data_y, prediction_data)
            if rmse < best_so_far[8]:
                best_so_far = (a, b, c, m, n, p, d, e, rmse)
        for m in m_possible:
            a = best_so_far[0]
            b = best_so_far[1]
            c = best_so_far[2]
            n = best_so_far[4]
            p = best_so_far[5]
            d = best_so_far[6]
            e = best_so_far[7]
            prediction_data = [periodic(x, a, b, c, m, n, p, d, e) for x in data_x]
            rmse = calculate_rmse(data_y, prediction_data)
            if rmse < best_so_far[8]:
                best_so_far = (a, b, c, m, n, p, d, e, rmse)
        for n in n_possible:
            a = best_so_far[0]
            b = best_so_far[1]
            c = best_so_far[2]
            m = best_so_far[3]
            p = best_so_far[5]
            d = best_so_far[6]
            e = best_so_far[7]
            prediction_data = [periodic(x, a, b, c, m, n, p, d, e) for x in data_x]
            rmse = calculate_rmse(data_y, prediction_data)
            if rmse < best_so_far[8]:
                best_so_far = (a, b, c, m, n, p, d, e, rmse)
        for p in p_possible:
            a = best_so_far[0]
            b = best_so_far[1]
            c = best_so_far[2]
            m = best_so_far[3]
            n = best_so_far[4]
            d = best_so_far[6]
            e = best_so_far[7]
            prediction_data = [periodic(x, a, b, c, m, n, p, d, e) for x in data_x]
            rmse = calculate_rmse(data_y, prediction_data)
            if rmse < best_so_far[8]:
                best_so_far = (a, b, c, m, n, p, d, e, rmse)
        for e in e_possible:
            a = best_so_far[0]
            b = best_so_far[1]
            c = best_so_far[2]
            m = best_so_far[3]
            n = best_so_far[4]
            p = best_so_far[5]
            d = best_so_far[6]
            prediction_data = [periodic(x, a, b, c, m, n, p, d, e) for x in data_x]
            rmse = calculate_rmse(data_y, prediction_data)
            if rmse < best_so_far[8]:
                best_so_far = (a, b, c, m, n, p, d, e, rmse)
        for d in d_possible:
            a = best_so_far[0]
            b = best_so_far[1]
            c = best_so_far[2]
            m = best_so_far[3]
            n = best_so_far[4]
            p = best_so_far[5]
            e = best_so_far[7]
            prediction_data = [periodic(x, a, b, c, m, n, p, d, e) for x in data_x]
            rmse = calculate_rmse(data_y, prediction_data)
            if rmse < best_so_far[8]:
                best_so_far = (a, b, c, m, n, p, d, e, rmse)

    return best_so_far


def exponential(x: float, a: float, b: float, c: float) -> float:
    """Return the y value according to the equation y = a * (b^x) + c.

    Preconditions:
        - b > 0
    """
    return a * (b ** x) + c


def quadratic(x: float, a: float, b: float, c: float) -> float:
    """Return the y value according to the equation y = a * x^2 + b * x + c."""
    return a * x ** 2 + b * x + c


def inverse(x: float, a: float, b: float) -> float:
    """Return the y value according to the equation y = a / x + b.

    Precondition:
        - x != 0
    """
    return a / x + b


def logarithm(x: float, a: float, b: float) -> float:
    """Return the y value according to the equation y = a * log(x) + b.

    Precondition:
        - x > 0
    """
    return a * np.log(x) + b


def periodic(x: float, a: float, b: float, c: float, m: float, n: float, p: float, d: float, e: float) -> float:
    """Return the y value according to the equation y = a * (cos(b * (x - c))) + d * x + e"""
    return a * np.cos(b * (x - c)) + m * np.cos(n * (x - p)) + d * x + e


def calculate_rmse(real_data: List[float], prediction_data: List[float]) -> float:
    """Return the root mean squared error of a model.

    Precondition:
      - len(real_data) == len(prediction_data)
    """
    n = len(real_data)
    square = [(real_data[i] - prediction_data[i]) ** 2 for i in range(n)]
    return (sum(square) / len(square)) ** 0.5


################################################################################
# Model fitting tests
################################################################################
# This section include unit tests for the model-fitting functions. Check out the
# tests for sample usages of the model-fitting functions.

def test_fit_exponential() -> None:
    """Generate 2d data in the form y = a * (b ** x) + c, according to chosen coefficients.
    Test if the optimized coefficients calculated by fit_linear() using the generated
    data is close the the chosen coefficients."""

    # test 1, generated data in the form: y = 0.2 * (2 ** x) - 1:
    a_true, b_true, c_true = 0.2, 2, -1
    data = generate_exponential_data(a_true, b_true, c_true)
    a, b, c, _ = fit_exponential(data[0], data[1])
    assert math.isclose(a, a_true, abs_tol=1e-05) and math.isclose(b, b_true, abs_tol=1e-05) \
           and math.isclose(c, c_true, abs_tol=1e-05)

    # test 2, generated data in the form: -6 * (0.5 ** x) + 3:
    a_true, b_true, c_true = -6, 0.5, 3
    data = generate_exponential_data(a_true, b_true, c_true)
    a, b, c, _ = fit_exponential(data[0], data[1])
    assert math.isclose(a, a_true, abs_tol=1e-05) and math.isclose(b, b_true, abs_tol=1e-05) \
           and math.isclose(c, c_true, abs_tol=1e-05)


def test_fit_quadratic() -> None:
    """Generate 2d data in the form y = a * x^2 + b * x + c, according to chosen coefficients.
    Test if the optimized coefficients calculated by fit_quadratic() using the generated
    data is close the the chosen coefficients."""

    # test 1, generated data in the form: y = 0.2 * x^2 + 7:
    a_true, b_true, c_true = 0.2, 0, 7
    data = generate_quadratic_data(a_true, b_true, c_true)
    a, b, c, _ = fit_quadratic(data[0], data[1])
    assert math.isclose(a, a_true, abs_tol=1e-05) and math.isclose(b, b_true, abs_tol=1e-05) \
           and math.isclose(c, c_true, abs_tol=1e-05)

    # test 2, generated data in the form: y = -2 * x^2 - 6 * x - 9:
    a_true, b_true, c_true = -2, -6, -9
    data = generate_quadratic_data(a_true, b_true, c_true)
    a, b, c, _ = fit_quadratic(data[0], data[1])
    assert math.isclose(a, a_true, abs_tol=1e-05) and math.isclose(b, b_true, abs_tol=1e-05) \
           and math.isclose(c, c_true, abs_tol=1e-05)


def test_fit_inverse() -> None:
    """Generate 2d data in the form y = a / x + b, according to chosen coefficients.
    Test if the optimized coefficients calculated by fit_inverse() using the generated
    data is close the the chosen coefficients."""

    # test 1, generated data in the form: y = 0.2 / x - 1:
    a_true, b_true = 0.2, -1
    data = generate_inverse_data(a_true, b_true)
    a, b, _ = fit_inverse(data[0], data[1])
    assert math.isclose(a, a_true, abs_tol=1e-05) and math.isclose(b, b_true, abs_tol=1e-05)

    # test 2, generated data in the form: y = -6 / x + 8:
    a_true, b_true = -6, 8
    data = generate_inverse_data(a_true, b_true)
    a, b, _ = fit_inverse(data[0], data[1])
    assert math.isclose(a, a_true, abs_tol=1e-05) and math.isclose(b, b_true, abs_tol=1e-05)


def test_fit_logarithm() -> None:
    """Generate 2d data in the form y = a * log(x) + b, according to chosen coefficients.
    Test if the optimized coefficients calculated by fit_logarithm() using the generated
    data is close the the chosen coefficients."""

    # test 1, generated data in the form: y = 0.2 * log(x) - 1:
    a_true, b_true = 0.2, -1
    data = generate_logarithm_data(a_true, b_true)
    a, b, _ = fit_logarithm(data[0], data[1])
    assert math.isclose(a, a_true, abs_tol=1e-05) and math.isclose(b, b_true, abs_tol=1e-05)

    # test 2, generated data in the form: y = -3 * log(x) + 4:
    a_true, b_true = -3, 4
    data = generate_logarithm_data(a_true, b_true)
    a, b, _ = fit_logarithm(data[0], data[1])
    assert math.isclose(a, a_true, abs_tol=1e-05) and math.isclose(b, b_true, abs_tol=1e-05)


def test_fit_periodic() -> None:
    """Generate 2d data in the form y = a * (cos(b * (x - c))) + m * (cos(n * (x - p))) + d * x + e, according
    to chosen coefficients; then, generate the same type of data according to the optimized
    coefficients calculated by fit_periodic() using the already generated data. Test if both
    data are close to each other.

    (We are not testing if the optimized coefficients are close to the chosen coefficients
    because the fit is periodic, meaning the optimized coefficients and the chosen coefficients
    could be different but still generate the same data)
    """

    # test 1, generated data in the form: y = 10 * (cos(2 * (x - 1))) + 3 * (cos(0.8 * (x - 2))) + 0.02 * x + 7:
    a_true, b_true, c_true, m_true, n_true, p_true, d_true, e_true = 10, 2, 1, 3, 0.8, 2, 0.02, 7
    data = generate_periodic_data(a_true, b_true, c_true, m_true, n_true, p_true, d_true, e_true)
    a, b, c, m, n, p, d, e, _ = fit_periodic(data[0], data[1],
                                             initial_guess=(9.8, 2.1, 1.1, 3.02, 0.88, 2, 0.021, 7.1),
                                             test_range=(0.7, 1.3))
    calculated_data = generate_periodic_data(a, b, c, m, n, p, d, e)
    assert all(math.isclose(data[1][i], calculated_data[1][i], abs_tol=1e-05)
               for i in range(len(data[1])))

    # test 2, generated data in the form: y = -2 * (cos(-3 * x)) + 0.6 * (cos(2 * (x - 0.5))) - 0.3 * x + 10:
    a_true, b_true, c_true, m_true, n_true, p_true, d_true, e_true = -2, -3, 0, 0.6, 2, 0.5, -0.3, 10
    data = generate_periodic_data(a_true, b_true, c_true, m_true, n_true, p_true, d_true, e_true)
    a, b, c, m, n, p, d, e, _ = fit_periodic(data[0], data[1],
                                             initial_guess=(-2.1, -2.98, 0.1, 0.62, 2.2, 0.53, -0.31, 9.9),
                                             test_range=(0.7, 1.3))
    calculated_data = generate_periodic_data(a, b, c, m, n, p, d, e)
    assert all(math.isclose(data[1][i], calculated_data[1][i], abs_tol=1e-05)
               for i in range(len(data[1])))


################################################################################
# Data generation functions for tests
################################################################################


def generate_exponential_data(a: float, b: float, c: float) -> Tuple[List[float], List[float]]:
    """Return a tuple where the first element is a list of x coordinates [0, 1, 2,...,50]
    and the second element is a list of corresponding y coordinates such that
    y = a * (b ** x) + c."""
    data_x = list(range(0, 51))
    data_y = [exponential(x, a, b, c) for x in range(0, 51)]
    return (data_x, data_y)


def generate_quadratic_data(a: float, b: float, c: float) -> Tuple[List[float], List[float]]:
    """Return a tuple where the first element is a list of x coordinates [0, 1, 2,...,50]
    and the second element is a list of corresponding y coordinates such that
    y = a * x^2 + b * x + c."""
    data_x = list(range(0, 51))
    data_y = [quadratic(x, a, b, c) for x in range(0, 51)]
    return (data_x, data_y)


def generate_inverse_data(a: float, b: float) -> Tuple[List[float], List[float]]:
    """Return a tuple where the first element is a list of x coordinates [1, 2, 3,...,50]
    and the second element is a list of corresponding y coordinates such that
    y = a / x + b."""
    data_x = list(range(1, 51))
    data_y = [inverse(x, a, b) for x in range(1, 51)]
    return (data_x, data_y)


def generate_logarithm_data(a: float, b: float) -> Tuple[List[float], List[float]]:
    """Return a tuple where the first element is a list of x coordinates [1, 2, 3,...,50]
    and the second element is a list of corresponding y coordinates such that
    y = a / x + b."""
    data_x = list(range(1, 51))
    data_y = [inverse(x, a, b) for x in range(1, 51)]
    return (data_x, data_y)


def generate_periodic_data(a: float, b: float, c: float, m: float, n: float, p: float, d: float, e: float) \
        -> Tuple[List[float], List[float]]:
    """Return a tuple where the first element is a list of x coordinates [0, 1, 2,...,50]
    and the second element is a list of corresponding y coordinates such that
    y = a * (cos(b * (x - c))) + m * (cos(n * (x - p))) + d * x + e"""
    data_x = list(range(0, 51))
    data_y = [periodic(x, a, b, c, m, n, p, d, e) for x in range(0, 51)]
    return (data_x, data_y)


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['scipy.optimize', 'numpy', 'math', 'python_ta.contracts'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()

    import pytest
    pytest.main(['models.py'])
