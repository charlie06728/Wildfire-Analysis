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
from typing import List
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
        wildfire = Wildfire(name, wildfire_file, 'STATE', 'FIRE_YEAR', 'DISCOVERY_DOY', 'FIRE_SIZE')
        self.temp_max = temp.max_months(begin, end)
        self.temp_min = temp.min_months(begin, end)
        self.temp_mean = temp.mean_months(begin, end)
        self.prcp = prcp.total_months(begin, end)
        self.fire_freq = wildfire.frequency_months(begin, end)
        self.fire_size = wildfire.mean_size_months(begin, end)

    def temp_time(self) -> None:
        """Use periodic model to analise the relation between the temperature and wildfire frequency of the state."""
        time_lag_list = generate_time_lag_list(self.begin, self.end)
        sub_title = ' temp - time of ' + self.name + ' (periodic model)'
        analysis_periodic('max' + sub_title, 'days', time_lag_list, 'temperature(Fahrenheit)', self.temp_max)
        analysis_periodic('min' + sub_title, 'days', time_lag_list, 'temperature(Fahrenheit)', self.temp_min)
        analysis_periodic('mean' + sub_title, 'days', time_lag_list, 'temperature(Fahrenheit)', self.temp_mean)

    def prcp_time(self) -> None:
        """Use periodic model to analise the relation between the precipitation and wildfire frequency of the state."""
        time_lag_list = generate_time_lag_list(self.begin, self.end)
        title = 'prcp - time of ' + self.name + ' (periodic model)'
        analysis_periodic(title, 'days', time_lag_list, 'precipitation(mm)', self.prcp)

    def temp_freq_exponential(self) -> None:
        """Use exponential model to analise the relation between the temperature and wildfire frequency of the state."""
        sub_title = ' temp-wildfire frequency of ' + self.name + ' (exponential model)'
        analysis_exponential('max' + sub_title, 'temp(Fahrenheit)', self.temp_max, 'frequency', self.fire_freq)
        analysis_exponential('min' + sub_title, 'temp(Fahrenheit)', self.temp_min, 'frequency', self.fire_freq)
        analysis_exponential('mean' + sub_title, 'temp(Fahrenheit)', self.temp_mean, 'frequency', self.fire_freq)

    def temp_freq_quadratic(self) -> None:
        """Use quadratic model to analise the relation between the temperature and wildfire frequency of the state."""
        sub_title = ' temp-wildfire frequency of ' + self.name + ' (quadratic model)'
        analysis_quadratic('max' + sub_title, 'temp(Fahrenheit)', self.temp_max, 'frequency', self.fire_freq)
        analysis_quadratic('min' + sub_title, 'temp(Fahrenheit)', self.temp_min, 'frequency', self.fire_freq)
        analysis_quadratic('mean' + sub_title, 'temp(Fahrenheit)', self.temp_mean, 'frequency', self.fire_freq)

    def temp_size_exponential(self) -> None:
        """Use exponential model to analise the relation between the temperature and wildfire size of the state."""
        sub_title = ' temp-wildfire size of ' + self.name + ' (exponential model)'
        analysis_exponential('max' + sub_title, 'temp(Fahrenheit)', self.temp_max, 'size(acre)', self.fire_size)
        analysis_exponential('min' + sub_title, 'temp(Fahrenheit)', self.temp_min, 'size(acre)', self.fire_size)
        analysis_exponential('mean' + sub_title, 'temp(Fahrenheit)', self.temp_mean, 'size(acre)', self.fire_size)

    def temp_size_quadratic(self) -> None:
        """Use quadratic model to analise the relation between the temperature and wildfire size of the state."""
        sub_title = ' temp-wildfire size of ' + self.name + ' (quadratic model)'
        analysis_quadratic('max' + sub_title, 'temp(Fahrenheit)', self.temp_max, 'size(acre)', self.fire_size)
        analysis_quadratic('min' + sub_title, 'temp(Fahrenheit)', self.temp_min, 'size(acre)', self.fire_size)
        analysis_quadratic('mean' + sub_title, 'temp(Fahrenheit)', self.temp_mean, 'size(acre)', self.fire_size)

    def prcp_freq_inverse(self) -> None:
        """Use inverse model to analise the relation between the precipitation and wildfire frequency of the state"""
        title = 'prcp-wildfire frequency of ' + self.name + ' (inverse model)'
        analysis_inverse(title, 'prcp(mm)', self.prcp, 'frequency', self.fire_freq)

    def prcp_freq_logarithm(self) -> None:
        """Use logarithm model to analise the relation between the precipitation and wildfire frequency of the state"""
        title = 'prcp-wildfire frequency of ' + self.name + ' (logarithm model)'
        analysis_logarithm(title, 'prcp(mm)', self.prcp, 'frequency', self.fire_freq)

    def prcp_size_inverse(self) -> None:
        """Use inverse model to analise the relation between the precipitation and wildfire size of the state"""
        title = 'prcp-wildfire size of ' + self.name + ' (inverse model)'
        analysis_inverse(title, 'prcp(mm)', self.prcp, 'size(acre)', self.fire_size)

    def prcp_size_logarithm(self) -> None:
        """Use logarithm model to analise the relation between the precipitation and wildfire size of the state"""
        title = 'prcp-wildfire size of ' + self.name + ' (logarithm model)'
        analysis_logarithm(title, 'prcp(mm)', self.prcp, 'size(acre)', self.fire_size)

    def analise_all(self) -> None:
        """Analise all data of the state."""
        self.temp_time()
        self.prcp_time()
        self.temp_freq_exponential()
        self.temp_freq_quadratic()
        self.temp_size_exponential()
        self.temp_size_quadratic()
        self.prcp_freq_inverse()
        self.prcp_freq_logarithm()
        self.prcp_size_inverse()
        self.prcp_size_logarithm()


def analysis_exponential(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> None:
    """Modeling between x and y using exponential model."""
    a, b, c, rmse = models.fit_exponential(x, y)
    print(title, ': RMSE=', rmse)
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.exponential(x_i, a, b, c) for x_i in new_x]
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)


def analysis_quadratic(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> None:
    """Modeling between x and y using quadratic model."""
    a, b, c, rmse = models.fit_quadratic(x, y)
    print(title, ': RMSE=', rmse)
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.quadratic(x_i, a, b, c) for x_i in new_x]
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)


def analysis_inverse(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> None:
    """Modeling between x and y using inverse model."""
    a, b, rmse = models.fit_inverse(x, y)
    print(title, ': RMSE=', rmse)
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.inverse(x_i, a, b) for x_i in new_x]
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)


def analysis_logarithm(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> None:
    """Modeling between x and y using inverse model."""
    a, b, rmse = models.fit_logarithm(x, y)
    print(title, ': RMSE=', rmse)
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.logarithm(x_i, a, b) for x_i in new_x]
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)


def analysis_periodic(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> None:
    """Modeling between x and y using periodic model."""
    a, b, c, d, e, rmse = models.fit_periodic(x, y, [20, 0.015, 120, 0, 60])
    print(title, ': RMSE=', rmse, 'a=', a, 'b=', b, 'c=', c, 'd=', d, 'e=', e)
    new_x = list(np.linspace(min(x), max(x), 5000))
    prediction_y = [models.periodic(x_i, a, b, c, d, e) for x_i in new_x]
    figure.double_figure_line(title, x_label, x, new_x, y_label, y, prediction_y)


def generate_date_list(begin: int, end: int) -> List[datetime.date]:
    """Return a list of months from the year of begin to the year of end."""
    date_list_so_far = []
    for year in range(begin, end + 1):
        for month in range(1, 13):
            date = datetime.date(year, month, 1)
            date_list_so_far.append(date)
    return [datetime.date(year, month, 1) for month in range(1, 13) for year in range(begin, end + 1)]


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


if __name__ == '__main__':
    analysis_ca = StateDataAnalysis('CA', 1994, 2004, 'ca_climate.csv', 'wildfire_data2.csv')
    # analysis_ca.analise_all()
    analysis_ca.temp_time()
    analysis_ca.prcp_time()
    # analysis_tx = StateDataAnalysis('TX', 1994, 2013, 'tx_climate.csv', 'wildfire_data2.csv')
    # analysis_tx.analise_all()
