"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains the analysis of models.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
from data_prep import Temperature, Precipitation, Wildfire
import figure
import models
import numpy as np
from typing import List, Tuple
import datetime


class StateDataAnalysis:
    """The analysis of a state's data.

    Instance Attributes:
      - name: the name of the state, represented by two capital letters
      - begin: the year when the analysis begins
      - end: the year when the analysis ends
      - temp_max:
      - temp_min:
      - temp_mean:
      - prcp:
      - fire_freq:
      - fire_size:
    """
    name: str
    begin: int
    end: int
    temp_max: List[float]
    temp_min: List[float]
    temp_mean: List[float]
    prcp: List[float]
    fire_freq: List[int]
    fire_size: List[float]

    def __init__(self, name: str, begin: int, end: int, climate_file: str, wildfire_file: str) -> None:
        """Initialize a new state data for analysis."""
        self.name = name
        self.begin = begin
        self.end = end
        temp = Temperature(name, climate_file, 'DATE', 'TAVG', 'TMAX', 'TMIN')
        prcp = Precipitation(name, climate_file, 'DATE', 'PRCP')
        wildfire = Wildfire(name, wildfire_file, 'FIRE_YEAR', 'DISCOVERY_DOY', 'FIRE_SIZE')
        self.temp_max = temp.max_temperature(begin, end)
        self.temp_min = temp.min_temperature(begin, end)
        self.temp_mean = temp.mean_temperature(begin, end)
        self.prcp = prcp.total_precipitation(begin, end)
        self.fire_freq = wildfire.number_of_fires_in_months(begin, end)
        self.fire_size = wildfire.mean_size_of_fires_in_months(begin, end)

    def temp_time(self, max_guess: tuple, min_guess: tuple, mean_guess: tuple, test_range: tuple) -> List[tuple]:
        """Use periodic model to analise the relation between the temperature and wildfire frequency of the state."""
        time_lag_list = generate_time_lag_list(self.begin, self.end)
        sub_title = ' temp - time of ' + self.name + ' (periodic model)'
        max_result = analysis_periodic('max' + sub_title, 'days', time_lag_list,
                                       'temperature(Fahrenheit)', self.temp_max, max_guess, test_range)
        min_result = analysis_periodic('min' + sub_title, 'days', time_lag_list,
                                       'temperature(Fahrenheit)', self.temp_min, min_guess, test_range)
        mean_result = analysis_periodic('mean' + sub_title, 'days', time_lag_list,
                                        'temperature(Fahrenheit)', self.temp_mean, mean_guess, test_range)
        return [max_result, min_result, mean_result]

    def prcp_time(self, guess: tuple, test_range: tuple) -> tuple:
        """Use periodic model to analise the relation between the precipitation and wildfire frequency of the state."""
        time_lag_list = generate_time_lag_list(self.begin, self.end)
        title = 'prcp - time of ' + self.name + ' (periodic model)'
        result = analysis_periodic(title, 'days', time_lag_list, 'precipitation(mm)', self.prcp, guess, test_range)
        return result

    def temp_freq_exponential(self) -> List[tuple]:
        """Use exponential model to analise the relation between the temperature and wildfire frequency of the state."""
        sub_title = ' temp-wildfire frequency of ' + self.name + ' (exponential model)'
        max_result = analysis_exponential('max' + sub_title,
                                          'temp(Fahrenheit)', self.temp_max, 'frequency', self.fire_freq)
        min_result = analysis_exponential('min' + sub_title, 'temp(Fahrenheit)',
                                          self.temp_min, 'frequency', self.fire_freq)
        mean_result = analysis_exponential('mean' + sub_title, 'temp(Fahrenheit)',
                                           self.temp_mean, 'frequency', self.fire_freq)
        return [max_result, min_result, mean_result]

    def temp_freq_quadratic(self) -> List[tuple]:
        """Use quadratic model to analise the relation between the temperature and wildfire frequency of the state."""
        sub_title = ' temp-wildfire frequency of ' + self.name + ' (quadratic model)'
        max_result = analysis_quadratic('max' + sub_title,
                                        'temp(Fahrenheit)', self.temp_max, 'frequency', self.fire_freq)
        min_result = analysis_quadratic('min' + sub_title,
                                        'temp(Fahrenheit)', self.temp_min, 'frequency', self.fire_freq)
        mean_result = analysis_quadratic('mean' + sub_title,
                                         'temp(Fahrenheit)', self.temp_mean, 'frequency', self.fire_freq)
        return [max_result, min_result, mean_result]

    def temp_size_exponential(self) -> List[tuple]:
        """Use exponential model to analise the relation between the temperature and wildfire size of the state."""
        sub_title = ' temp-wildfire size of ' + self.name + ' (exponential model)'
        max_result = analysis_exponential('max' + sub_title,
                                          'temp(Fahrenheit)', self.temp_max, 'size(acre)', self.fire_size)
        min_result = analysis_exponential('min' + sub_title,
                                          'temp(Fahrenheit)', self.temp_min, 'size(acre)', self.fire_size)
        mean_result = analysis_exponential('mean' + sub_title,
                                           'temp(Fahrenheit)', self.temp_mean, 'size(acre)', self.fire_size)
        return [max_result, min_result, mean_result]

    def temp_size_quadratic(self) -> List[tuple]:
        """Use quadratic model to analise the relation between the temperature and wildfire size of the state."""
        sub_title = ' temp-wildfire size of ' + self.name + ' (quadratic model)'
        max_result = analysis_quadratic('max' + sub_title,
                                        'temp(Fahrenheit)', self.temp_max, 'size(acre)', self.fire_size)
        min_result = analysis_quadratic('min' + sub_title,
                                        'temp(Fahrenheit)', self.temp_min, 'size(acre)', self.fire_size)
        mean_result = analysis_quadratic('mean' + sub_title,
                                         'temp(Fahrenheit)', self.temp_mean, 'size(acre)', self.fire_size)
        return [max_result, min_result, mean_result]

    def prcp_freq_inverse(self) -> tuple:
        """Use inverse model to analise the relation between the precipitation and wildfire frequency of the state"""
        title = 'prcp-wildfire frequency of ' + self.name + ' (inverse model)'
        result = analysis_inverse(title, 'prcp(mm)', self.prcp, 'frequency', self.fire_freq)
        return result

    def prcp_freq_logarithm(self) -> tuple:
        """Use logarithm model to analise the relation between the precipitation and wildfire frequency of the state"""
        title = 'prcp-wildfire frequency of ' + self.name + ' (logarithm model)'
        result = analysis_logarithm(title, 'prcp(mm)', self.prcp, 'frequency', self.fire_freq)
        return result

    def prcp_size_inverse(self) -> tuple:
        """Use inverse model to analise the relation between the precipitation and wildfire size of the state"""
        title = 'prcp-wildfire size of ' + self.name + ' (inverse model)'
        result = analysis_inverse(title, 'prcp(mm)', self.prcp, 'size(acre)', self.fire_size)
        return result

    def prcp_size_logarithm(self) -> tuple:
        """Use logarithm model to analise the relation between the precipitation and wildfire size of the state"""
        title = 'prcp-wildfire size of ' + self.name + ' (logarithm model)'
        result = analysis_logarithm(title, 'prcp(mm)', self.prcp, 'size(acre)', self.fire_size)
        return result

    def analise_wildfire(self) -> None:
        """Analise all data of the state."""
        self.temp_freq_exponential()
        self.temp_freq_quadratic()
        self.temp_size_exponential()
        self.temp_size_quadratic()
        self.prcp_freq_inverse()
        self.prcp_freq_logarithm()
        self.prcp_size_inverse()
        self.prcp_size_logarithm()

    def predict_temp(self, begin: int, end: int, max_guess: tuple, min_guess: tuple, mean_guess: tuple) -> tuple:
        """Prediction the temperature of the state from the year of begin to the year of the end."""
        ta1, tb1, tc1, tm1, tn1, tp1, td1, te1 = max_guess
        ta2, tb2, tc2, tm2, tn2, tp2, td2, te2 = min_guess
        ta3, tb3, tc3, tm3, tn3, tp3, td3, te3 = mean_guess
        date = generate_date_list(begin, end)
        time_lag = calculate_time_lag_list(self.begin, begin, end)
        temp_predict1 = [models.periodic(x, ta1, tb1, tc1, tm1, tn1, tp1, td1, te1) for x in time_lag]
        temp_predict2 = [models.periodic(x, ta2, tb2, tc2, tm2, tn2, tp2, td2, te2) for x in time_lag]
        temp_predict3 = [models.periodic(x, ta3, tb3, tc3, tm3, tn3, tp3, td3, te3) for x in time_lag]
        figure.figure_line('Prediction: temperature (max mean min)', 'date', date, 'temp(Fahrenheit)', temp_predict1)
        figure.figure_line('Prediction: temperature (max mean min)', 'date', date, 'temp(Fahrenheit)', temp_predict2)
        figure.figure_line('Prediction: temperature (max mean min)', 'date', date, 'temp(Fahrenheit)', temp_predict3)
        return (temp_predict1, temp_predict2, temp_predict3)

    def predict_prcp(self, begin: int, end: int, guess: tuple) -> list:
        """Prediction the precipitation of the state from the year of begin to the year of the end."""
        ta, tb, tc, tm, tn, tp, td, te = guess
        date = generate_date_list(begin, end)
        time_lag = calculate_time_lag_list(self.begin, begin, end)
        prcp_predict = [models.periodic(x, ta, tb, tc, tm, tn, tp, td, te) for x in time_lag]
        figure.figure_line('Prediction: precipitation', 'date', date, 'precipitation(mm)', prcp_predict)
        return prcp_predict

    def predict_freq_temp_exp(self, begin: int, end: int, max_guess: tuple, min_guess: tuple, mean_guess: tuple) \
            -> None:
        """Use exponential module to predict wildfire frequency of the state till from the year of begin to the
        year of end,based on the prediction of temperature."""
        temp_predict1, temp_predict2, temp_predict3 = self.predict_temp(begin, end, max_guess, min_guess, mean_guess)
        f_max, f_min, f_mean = self.temp_freq_exponential()
        fa1, fb1, fc1 = f_max
        fa2, fb2, fc2 = f_min
        fa3, fb3, fc3 = f_mean
        date = generate_date_list(begin, end)
        freq_predict1 = [models.exponential(x, fa1, fb1, fc1) for x in temp_predict1]
        freq_predict2 = [models.exponential(x, fa2, fb2, fc2) for x in temp_predict2]
        freq_predict3 = [models.exponential(x, fa3, fb3, fc3) for x in temp_predict3]
        figure.figure_line('Prediction: wildfire frequency based on the max temperature',
                           'date', date, 'frequency', freq_predict1)
        figure.figure_line('Prediction: wildfire frequency based on the min temperature',
                           'date', date, 'frequency', freq_predict2)
        figure.figure_line('Prediction: wildfire frequency based on the mean temperature',
                           'date', date, 'frequency', freq_predict3)

    def predict_freq_temp_qua(self, begin: int, end: int, max_guess: tuple, min_guess: tuple, mean_guess: tuple) \
            -> None:
        """Use quadratic module to predict wildfire frequency of the state till from the year of begin to
        the year of end,based on the prediction of temperature."""
        temp_predict1, temp_predict2, temp_predict3 = self.predict_temp(begin, end, max_guess, min_guess, mean_guess)
        f_max, f_min, f_mean = self.temp_freq_quadratic()
        fa1, fb1, fc1 = f_max
        fa2, fb2, fc2 = f_min
        fa3, fb3, fc3 = f_mean
        date = generate_date_list(begin, end)
        freq_predict1 = [models.quadratic(x, fa1, fb1, fc1) for x in temp_predict1]
        freq_predict2 = [models.quadratic(x, fa2, fb2, fc2) for x in temp_predict2]
        freq_predict3 = [models.quadratic(x, fa3, fb3, fc3) for x in temp_predict3]
        figure.figure_line('Prediction: wildfire frequency based on the max temperature',
                           'date', date, 'frequency', freq_predict1)
        figure.figure_line('Prediction: wildfire frequency based on the min temperature',
                           'date', date, 'frequency', freq_predict2)
        figure.figure_line('Prediction: wildfire frequency based on the mean temperature',
                           'date', date, 'frequency', freq_predict3)

    def predict_freq_prcp_inv(self, begin: int, end: int, guess: tuple) \
            -> None:
        """Use inverse module to predict wildfire frequency of the state till from the year of begin to
        the year of end,based on the prediction of precipitation."""
        fa, fb = self.prcp_freq_inverse()
        date = generate_date_list(begin, end)
        prcp_predict = self.predict_prcp(begin, end, guess)
        freq_predict = [models.inverse(x, fa, fb) for x in prcp_predict]
        figure.figure_line('Prediction: wildfire frequency based on the precipitation',
                           'date', date, 'frequency', freq_predict)

    def predict_freq_prcp_log(self, begin: int, end: int, guess: tuple) \
            -> None:
        """Use logarithm module to predict wildfire frequency of the state till from the year of begin to
        the year of end,based on the prediction of precipitation."""
        fa, fb = self.prcp_freq_logarithm()
        date = generate_date_list(begin, end)
        prcp_predict = self.predict_prcp(begin, end, guess)
        freq_predict = [models.logarithm(x, fa, fb) for x in prcp_predict]
        figure.figure_line('Prediction: wildfire frequency based on the precipitation',
                           'date', date, 'frequency', freq_predict)


def analysis_exponential(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> tuple:
    """Modeling between x and y using exponential model."""
    a, b, c, rmse = models.fit_exponential(x, y)
    print(title + ': RMSE=%f, y = %f * (%f^x) + %f' % (rmse, a, b, c))
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.exponential(x_i, a, b, c) for x_i in new_x]
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b, c)


def analysis_quadratic(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> tuple:
    """Modeling between x and y using quadratic model."""
    a, b, c, rmse = models.fit_quadratic(x, y)
    print(title + ': RMSE=%f, y = %f * x^2 + %f * x + %f' % (rmse, a, b, c))
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.quadratic(x_i, a, b, c) for x_i in new_x]
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b, c)


def analysis_inverse(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> tuple:
    """Modeling between x and y using inverse model."""
    a, b, rmse = models.fit_inverse(x, y)
    print(title + ': RMSE=%f, y = %f / x + %f' % (rmse, a, b))
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.inverse(x_i, a, b) for x_i in new_x]
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b)


def analysis_logarithm(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> tuple:
    """Modeling between x and y using inverse model."""
    a, b, rmse = models.fit_logarithm(x, y)
    print(title + ': RMSE=%f, y = %f * log(x) + %f' % (rmse, a, b))
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.logarithm(x_i, a, b) for x_i in new_x]
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b)


def analysis_periodic(title: str, x_label: str, x: List[float], y_label: str, y: List[float],
                      initial_guess: Tuple[float, float, float, float, float, float, float, float],
                      test_range: Tuple[float, float]) -> tuple:
    """Modeling between x and y using periodic model."""
    a, b, c, m, n, p, d, e, rmse = models.fit_periodic(x, y, initial_guess, test_range)
    print(title + ': RMSE=%f, y = %f * (cos(%f * (x - %f))) + (%f) * (cos(%f * (x - %f))) + %f * x + %f'
          % (rmse, a, b, c, m, n, p, d, e))
    new_x = list(np.linspace(min(x), max(x), 5000))
    prediction_y = [models.periodic(x_i, a, b, c, m, n, p, d, e) for x_i in new_x]
    figure.double_figure_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b, c, m, n, p, d, e)


def generate_date_list(begin: int, end: int) -> List[datetime.date]:
    """Return a list of months from the year of begin to the year of end."""
    date_list_so_far = []
    for year in range(begin, end + 1):
        for month in range(1, 13):
            date = datetime.date(year, month, 1)
            date_list_so_far.append(date)
    return date_list_so_far


def generate_time_lag_list(begin: int, end: int) -> List[int]:
    """Return a list of time lag between the first day of every month and the first day of the year of begin,
     till the year of end."""
    begin_date = datetime.date(begin, 1, 1)
    time_lag_list_so_far = []
    for year in range(begin, end + 1):
        for month in range(1, 13):
            time_lag = (datetime.date(year, month, 1) - begin_date).days
            time_lag_list_so_far.append(time_lag)
    return time_lag_list_so_far


def calculate_time_lag_list(base: int, begin: int, end: int) -> List[int]:
    """Return a list of time lag between the first day of every month and the first day of the year of base,
     from the year of begin to the year of end."""
    base_date = datetime.date(base, 1, 1)
    time_lag_list_so_far = []
    for year in range(begin, end + 1):
        for month in range(1, 13):
            time_lag = (datetime.date(year, month, 1) - base_date).days
            time_lag_list_so_far.append(time_lag)
    return time_lag_list_so_far


if __name__ == '__main__':
    ...
