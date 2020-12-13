"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains the analysis of models.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
import data_prep
import figure
import models
from typing import List


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


# Generate the data of temperature and wildfire of CA
ca_temp = data_prep.Temperature('CA', 'ca_climate.csv', 'DATE', 'TAVG', 'TMAX', 'TMIN')
ca_wildfire = data_prep.Wildfire('CA', 'wildfire_data2.csv', 'STATE', 'FIRE_YEAR', 'DISCOVERY_DOY', 'FIRE_SIZE')
ca_prcp = data_prep.Precipitation('CA', 'ca_climate.csv', 'DATE', 'PRCP')

# Set time from the year of begin to the year of end
begin = 1994
end = 2013
time = figure.get_time_list(begin, end)  # get a list

# temperature data
temp_max = ca_temp.max_months(begin, end)
temp_min = ca_temp.min_months(begin, end)
temp_mean = ca_temp.mean_months(begin, end)

# precipitation data
prcp = ca_prcp.total_months(begin, end)

# wildfire data
fire_freq = ca_wildfire.frequency_months(begin, end)
fire_size = ca_wildfire.mean_size_months(begin, end)

# temperature - wildfire frequency
analysis_exponential('max temp-wildfire frequency-CA (exponentialr model)',
                     'temp(Fahrenheit)', temp_max, 'frequency', fire_freq)
analysis_exponential('min temp-wildfire frequency-CA (exponential model)',
                     'temp(Fahrenheit)', temp_min, 'frequency', fire_freq)
analysis_exponential('mean temp-wildfire frequency-CA (exponential model)',
                     'temp(Fahrenheit)', temp_mean, 'frequency', fire_freq)
analysis_quadratic('max temp-wildfire frequency-CA (quadratic model)',
                   'temp(Fahrenheit)', temp_max, 'frequency', fire_freq)
analysis_quadratic('min temp-wildfire frequency-CA (quadratic model)',
                   'temp(Fahrenheit)', temp_min, 'frequency', fire_freq)
analysis_quadratic('mean temp-wildfire frequency-CA (quadratic model)',
                   'temp(Fahrenheit)', temp_mean, 'frequency', fire_freq)

# temperature - wildfire size
# analysis_exponential('max temp-wildfire size-CA (exponential model)',
#                      'temp(Fahrenheit)', temp_max, 'frequency', fire_size)
# analysis_exponential('min temp-wildfire size-CA (exponential model)',
#                      'temp(Fahrenheit)', temp_min, 'frequency', fire_size)
# analysis_exponential('mean temp-wildfire size-CA exponential model)',
#                      'temp(Fahrenheit)', temp_mean, 'frequency', fire_size)
analysis_quadratic('max temp-wildfire size-CA (quadratic model)',
                   'temp(Fahrenheit)', temp_max, 'frequency', fire_size)
analysis_quadratic('min temp-wildfire size-CA (quadratic model)',
                   'temp(Fahrenheit)', temp_min, 'frequency', fire_size)
analysis_quadratic('mean temp-wildfire size-CA (quadratic model)',
                   'temp(Fahrenheit)', temp_mean, 'frequency', fire_size)

# precipitation - wildfire frequency
analysis_inverse('prcp-wildfire frequency-CA (inverse model)', 'prcp(mm)', prcp, 'frequency', fire_freq)
analysis_logarithm('prcp-wildfire frequency-CA (logarithm model)', 'prcp(mm)', prcp, 'frequency', fire_freq)

# precipitation - wildfire size
analysis_inverse('prcp-wildfire size-CA (inverse model)', 'prcp(mm)', prcp, 'size(acre)', fire_size)
analysis_logarithm('prcp-wildfire size-CA (logarithm model)', 'prcp(mm)', prcp, 'size(acre)', fire_size)
