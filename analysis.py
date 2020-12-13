"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains the analysis of models.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
from data_prep import Temperature, Precipitation, Wildfire
import figure
import models
from typing import List


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
    prediction_y = [models.exponential(x_i, a, b, c) for x_i in x]
    figure.double_figure_dot(title, x_label, x, y_label, y, prediction_y)


def analysis_quadratic(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> None:
    """Modeling between x and y using quadratic model."""
    a, b, c, rmse = models.fit_quadratic(x, y)
    print(title, ': RMSE=', rmse)
    prediction_y = [models.quadratic(x_i, a, b, c) for x_i in x]
    figure.double_figure_dot(title, x_label, x, y_label, y, prediction_y)


def analysis_inverse(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> None:
    """Modeling between x and y using inverse model."""
    a, b, rmse = models.fit_inverse(x, y)
    print(title, ': RMSE=', rmse)
    prediction_y = [models.inverse(x_i, a, b) for x_i in x]
    figure.double_figure_dot(title, x_label, x, y_label, y, prediction_y)


def analysis_logarithm(title: str, x_label: str, x: List[float], y_label: str, y: List[float]) -> None:
    """Modeling between x and y using inverse model."""
    a, b, rmse = models.fit_logarithm(x, y)
    print(title, ': RMSE=', rmse)
    prediction_y = [models.logarithm(x_i, a, b) for x_i in x]
    figure.double_figure_dot(title, x_label, x, y_label, y, prediction_y)


if __name__ == '__main__':
    analysis_ca = StateDataAnalysis('CA', 1994, 2013, 'ca_climate.csv', 'wildfire_data2.csv')
    analysis_ca.analise_all()
    # analysis_tx = StateDataAnalysis('TX', 1994, 2013, 'tx_climate.csv', 'wildfire_data2.csv')
    # analysis_tx.analise_all()
