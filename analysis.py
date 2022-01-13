"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains the analysis of models.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
from typing import List, Tuple
import datetime
import numpy as np
from data_prep import Temperature, Precipitation, Wildfire
import figure
import models


class StateDataAnalysis:
    """The analysis of a state's data.

    Instance Attributes:
      - name: the name of the state, represented by two capital letters
      - begin: the year when the analysis begins
      - end: the year when the analysis ends
      - temp: the temperature data of the state,
              which contains max, min and mean temperature in turn,
              represented by month
      - prcp: the precipitation data of the state, represented by month
      - fire_freq: the wildfire frequency data of the state, represented by month
      - fire_size:the wildfire size data of the state, represented by month
    """
    name: str
    begin: int
    end: int
    temp: Tuple[List[float], List[float], List[float]]
    prcp: List[float]
    fire_freq: List[int]
    fire_size: List[float]

    def __init__(self, name: str, time_range: Tuple[int, int],
                 climate_file: str, wildfire_file: str) -> None:
        """Initialize a new state data for analysis."""
        self.name = name
        self.begin, self.end = time_range
        # initialize a Temperature class using climate_file and several column names needed
        temp = Temperature(name, climate_file, 'DATE', 'TAVG', 'TMAX', 'TMIN')
        # initialize a Precipitation class using climate_file and several column names needed
        prcp = Precipitation(name, climate_file, 'DATE', 'PRCP')
        # initialize a wildfire class using climate_file and several column names needed
        wildfire = Wildfire(name, wildfire_file, 'FIRE_YEAR', 'DISCOVERY_DOY', 'FIRE_SIZE')
        # call functions in Temperature class to obtain three temperature data lists
        temp_max = temp.max_temperature(self.begin, self.end)
        temp_min = temp.min_temperature(self.begin, self.end)
        temp_mean = temp.mean_temperature(self.begin, self.end)
        self.temp = (temp_max, temp_min, temp_mean)
        # initialize self.prcp by calling a function in Precipitation class
        self.prcp = prcp.total_precipitation(self.begin, self.end)
        # initialize fire data by calling functions in Wildfire class
        self.fire_freq = wildfire.number_of_fires_in_months(self.begin, self.end)
        self.fire_size = wildfire.mean_size_of_fires_in_months(self.begin, self.end)

    def temp_time(self, max_guess: tuple, min_guess: tuple, mean_guess: tuple, test_range: tuple)\
            -> List[tuple]:
        """Use periodic model to analise the relation between the temperature
         and wildfire frequency of the state."""
        time_lag_list = generate_time_lag_list(self.begin, self.end)
        sub_title = ' temp - time of ' + self.name + ' (periodic model)'
        # modelling by call analysis functions
        max_result = analysis_periodic(('max' + sub_title, 'days', 'temperature(Fahrenheit)'),
                                       time_lag_list, self.temp[0],
                                       max_guess, test_range)
        min_result = analysis_periodic(('min' + sub_title, 'days', 'temperature(Fahrenheit)'),
                                       time_lag_list, self.temp[1],
                                       min_guess, test_range)
        mean_result = analysis_periodic(('mean' + sub_title, 'days', 'temperature(Fahrenheit)'),
                                        time_lag_list, self.temp[2],
                                        mean_guess, test_range)
        return [max_result, min_result, mean_result]

    def prcp_time(self, guess: tuple, test_range: tuple) -> tuple:
        """Use periodic model to analise the relation between the precipitation
         and wildfire frequency of the state."""
        time_lag_list = generate_time_lag_list(self.begin, self.end)
        title = 'prcp - time of ' + self.name + ' (periodic model)'
        # modelling by call analysis functions
        result = analysis_periodic((title, 'days', 'precipitation(mm)'),
                                   time_lag_list, self.prcp, guess, test_range)
        return result

    def temp_freq_exponential(self) -> List[tuple]:
        """Use exponential model to analise the relation between the temperature
         and wildfire frequency of the state."""
        sub_title = ' temp-wildfire frequency of ' + self.name + ' (exponential model)'
        # modelling by call analysis functions
        max_result = analysis_exponential('max' + sub_title,
                                          'temp(Fahrenheit)', self.temp[0],
                                          'frequency', self.fire_freq)
        min_result = analysis_exponential('min' + sub_title,
                                          'temp(Fahrenheit)', self.temp[1],
                                          'frequency', self.fire_freq)
        mean_result = analysis_exponential('mean' + sub_title,
                                           'temp(Fahrenheit)', self.temp[2],
                                           'frequency', self.fire_freq)
        return [max_result, min_result, mean_result]

    def temp_freq_quadratic(self) -> List[tuple]:
        """Use quadratic model to analise the relation between the temperature
         and wildfire frequency of the state."""
        sub_title = ' temp-wildfire frequency of ' + self.name + ' (quadratic model)'
        # modelling by call analysis functions
        max_result = analysis_quadratic('max' + sub_title,
                                        'temp(Fahrenheit)', self.temp[0],
                                        'frequency', self.fire_freq)
        min_result = analysis_quadratic('min' + sub_title,
                                        'temp(Fahrenheit)', self.temp[1],
                                        'frequency', self.fire_freq)
        mean_result = analysis_quadratic('mean' + sub_title,
                                         'temp(Fahrenheit)', self.temp[2],
                                         'frequency', self.fire_freq)
        return [max_result, min_result, mean_result]

    def temp_size_exponential(self) -> List[tuple]:
        """Use exponential model to analise the relation between the temperature
         and wildfire size of the state."""
        sub_title = ' temp-wildfire size of ' + self.name + ' (exponential model)'
        # modelling by call analysis functions
        max_result = analysis_exponential('max' + sub_title,
                                          'temp(Fahrenheit)', self.temp[0],
                                          'size(acre)', self.fire_size)
        min_result = analysis_exponential('min' + sub_title,
                                          'temp(Fahrenheit)', self.temp[1],
                                          'size(acre)', self.fire_size)
        mean_result = analysis_exponential('mean' + sub_title,
                                           'temp(Fahrenheit)', self.temp[2],
                                           'size(acre)', self.fire_size)
        return [max_result, min_result, mean_result]

    def temp_size_quadratic(self) -> List[tuple]:
        """Use quadratic model to analise the relation between the temperature
         and wildfire size of the state."""
        sub_title = ' temp-wildfire size of ' + self.name + ' (quadratic model)'
        # modelling by call analysis functions
        max_result = analysis_quadratic('max' + sub_title,
                                        'temp(Fahrenheit)', self.temp[0],
                                        'size(acre)', self.fire_size)
        min_result = analysis_quadratic('min' + sub_title,
                                        'temp(Fahrenheit)', self.temp[1],
                                        'size(acre)', self.fire_size)
        mean_result = analysis_quadratic('mean' + sub_title,
                                         'temp(Fahrenheit)', self.temp[2],
                                         'size(acre)', self.fire_size)
        return [max_result, min_result, mean_result]

    def prcp_freq_inverse(self) -> tuple:
        """Use inverse model to analise the relation between the precipitation
        and wildfire frequency of the state"""
        title = 'prcp-wildfire frequency of ' + self.name + ' (inverse model)'
        # modelling by call analysis functions
        result = analysis_inverse(title, 'prcp(mm)', self.prcp, 'frequency', self.fire_freq)
        return result

    def prcp_freq_logarithm(self) -> tuple:
        """Use logarithm model to analise the relation between the precipitation
        and wildfire frequency of the state"""
        title = 'prcp-wildfire frequency of ' + self.name + ' (logarithm model)'
        # modelling by call analysis functions
        result = analysis_logarithm(title, 'prcp(mm)', self.prcp, 'frequency', self.fire_freq)
        return result

    def prcp_size_inverse(self) -> tuple:
        """Use inverse model to analise the relation between the precipitation
         and wildfire size of the state"""
        title = 'prcp-wildfire size of ' + self.name + ' (inverse model)'
        # modelling by call analysis functions
        result = analysis_inverse(title, 'prcp(mm)', self.prcp, 'size(acre)', self.fire_size)
        return result

    def prcp_size_logarithm(self) -> tuple:
        """Use logarithm model to analise the relation between the precipitation
        and wildfire size of the state"""
        title = 'prcp-wildfire size of ' + self.name + ' (logarithm model)'
        # modelling by call analysis functions
        result = analysis_logarithm(title, 'prcp(mm)', self.prcp, 'size(acre)', self.fire_size)
        return result

    def analise_wildfire(self) -> None:
        """Analise all wildfire data with climate data of the state."""
        self.temp_freq_exponential()
        self.temp_freq_quadratic()
        self.temp_size_exponential()
        self.temp_size_quadratic()
        self.prcp_freq_inverse()
        self.prcp_freq_logarithm()
        self.prcp_size_inverse()
        self.prcp_size_logarithm()

    def predict_temp(self, time_range: Tuple[int, int], max_guess: tuple,
                     min_guess: tuple, mean_guess: tuple) -> tuple:
        """Prediction the temperature of the state from the year of begin to the year of the end."""
        g1 = max_guess
        g2 = min_guess
        g3 = mean_guess
        date = generate_date_list(time_range[0], time_range[1])
        time_lag = calculate_time_lag_list(self.begin, time_range[0], time_range[1])
        # obtain the temperature data with the given parameters
        temp_predict1 = [models.periodic(x, g1[0], g1[1], g1[2], g1[3], g1[4], g1[5], g1[6], g1[7])
                         for x in time_lag]
        temp_predict2 = [models.periodic(x, g2[0], g2[1], g2[2], g2[3], g2[4], g2[5], g2[6], g2[7])
                         for x in time_lag]
        temp_predict3 = [models.periodic(x, g3[0], g3[1], g3[2], g3[3], g3[4], g3[5], g3[6], g3[7])
                         for x in time_lag]
        # show the prediction in figures
        figure.figure_line('Prediction: temperature (max mean min)',
                           'date', date, 'temp(Fahrenheit)', temp_predict1)
        figure.figure_line('Prediction: temperature (max mean min)',
                           'date', date, 'temp(Fahrenheit)', temp_predict2)
        figure.figure_line('Prediction: temperature (max mean min)',
                           'date', date, 'temp(Fahrenheit)', temp_predict3)
        return (temp_predict1, temp_predict2, temp_predict3)

    def predict_prcp(self, begin: int, end: int, guess: tuple) -> list:
        """Prediction the precipitation of the state from the year of begin
         to the year of the end."""
        g = guess
        date = generate_date_list(begin, end)
        time_lag = calculate_time_lag_list(self.begin, begin, end)
        # obtain the precipitation data with the given parameters
        prcp_predict = [models.periodic(x, g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7])
                        for x in time_lag]
        # show the prediction in figures
        figure.figure_line('Prediction: precipitation', 'date', date,
                           'precipitation(mm)', prcp_predict)
        return prcp_predict

    def predict_freq_temp_exp(self, time_range: Tuple[int, int], max_guess: tuple,
                              min_guess: tuple, mean_guess: tuple) -> None:
        """Use exponential module to predict wildfire frequency of the state till
         from the year of begin to the year of end,based on the prediction of temperature."""
        begin, end = time_range
        # obtain the temperature data with the given parameters
        temp_predict = self.predict_temp((begin, end), max_guess, min_guess, mean_guess)
        f_max, f_min, f_mean = self.temp_freq_exponential()
        date = generate_date_list(begin, end)
        # obtain the wildfire frequency data with the given parameters
        freq_predict1 = [models.exponential(x, f_max[0], f_max[1], f_max[2])
                         for x in temp_predict[0]]
        freq_predict2 = [models.exponential(x, f_min[0], f_min[1], f_min[2])
                         for x in temp_predict[1]]
        freq_predict3 = [models.exponential(x, f_mean[0], f_mean[1], f_mean[2])
                         for x in temp_predict[2]]
        # show the prediction in figures
        figure.figure_line('Prediction: wildfire frequency based on the max temperature',
                           'date', date, 'frequency', freq_predict1)
        figure.figure_line('Prediction: wildfire frequency based on the min temperature',
                           'date', date, 'frequency', freq_predict2)
        figure.figure_line('Prediction: wildfire frequency based on the mean temperature',
                           'date', date, 'frequency', freq_predict3)

    def predict_freq_temp_qua(self, time_range: Tuple[int, int], max_guess: tuple,
                              min_guess: tuple, mean_guess: tuple) -> None:
        """Use quadratic module to predict wildfire frequency of the state till
        from the year of begin to the year of end,based on the prediction of temperature."""
        begin, end = time_range
        temp_predict = self.predict_temp((begin, end), max_guess, min_guess, mean_guess)
        f_max, f_min, f_mean = self.temp_freq_quadratic()
        date = generate_date_list(begin, end)
        # obtain the wildfire frequency data with the given parameters
        freq_predict1 = [models.quadratic(x, f_max[0], f_max[1], f_max[2])
                         for x in temp_predict[0]]
        freq_predict2 = [models.quadratic(x, f_min[0], f_min[1], f_min[2])
                         for x in temp_predict[1]]
        freq_predict3 = [models.quadratic(x, f_mean[0], f_mean[1], f_mean[2])
                         for x in temp_predict[2]]
        # show the prediction in figures
        figure.figure_line('Prediction: wildfire frequency based on the max temperature',
                           'date', date, 'frequency', freq_predict1)
        figure.figure_line('Prediction: wildfire frequency based on the min temperature',
                           'date', date, 'frequency', freq_predict2)
        figure.figure_line('Prediction: wildfire frequency based on the mean temperature',
                           'date', date, 'frequency', freq_predict3)

    def predict_freq_prcp_inv(self, begin: int, end: int, guess: tuple) \
            -> None:
        """Use inverse module to predict wildfire frequency of the state till
        from the year of begin to the year of end,based on the prediction of precipitation."""
        fa, fb = self.prcp_freq_inverse()
        date = generate_date_list(begin, end)
        # obtain the precipitation data with the given parameters
        prcp_predict = self.predict_prcp(begin, end, guess)
        # obtain the wildfire frequency data with the given parameters
        freq_predict = [models.inverse(x, fa, fb) for x in prcp_predict]
        # show the prediction in figures
        figure.figure_line('Prediction: wildfire frequency based on the precipitation',
                           'date', date, 'frequency', freq_predict)

    def predict_freq_prcp_log(self, begin: int, end: int, guess: tuple) \
            -> None:
        """Use logarithm module to predict wildfire frequency of the state till
        from the year of begin to the year of end,based on the prediction of precipitation."""
        fa, fb = self.prcp_freq_logarithm()
        date = generate_date_list(begin, end)
        # obtain the precipitation data with the given parameters
        prcp_predict = self.predict_prcp(begin, end, guess)
        # obtain the wildfire frequency data with the given parameters
        freq_predict = [models.logarithm(x, fa, fb) for x in prcp_predict]
        # show the prediction in figures
        figure.figure_line('Prediction: wildfire frequency based on the precipitation',
                           'date', date, 'frequency', freq_predict)


def analysis_exponential(title: str, x_label: str, x: List[float], y_label: str,
                         y: List[float]) -> tuple:
    """Modeling between x and y using exponential model."""
    a, b, c, rmse = models.fit_exponential(x, y)
    # print the the value of RMSE and the function
    print(title + ': RMSE=%f, y = %f * (%f^x) + %f' % (rmse, a, b, c))
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.exponential(x_i, a, b, c) for x_i in new_x]
    # show the prediction figure to compare with the original data
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b, c)


def analysis_quadratic(title: str, x_label: str, x: List[float], y_label: str,
                       y: List[float]) -> tuple:
    """Modeling between x and y using quadratic model."""
    a, b, c, rmse = models.fit_quadratic(x, y)
    # print the the value of RMSE and the function
    print(title + ': RMSE=%f, y = %f * x^2 + %f * x + %f' % (rmse, a, b, c))
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.quadratic(x_i, a, b, c) for x_i in new_x]
    # show the prediction figure to compare with the original data
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b, c)


def analysis_inverse(title: str, x_label: str, x: List[float], y_label: str,
                     y: List[float]) -> tuple:
    """Modeling between x and y using inverse model."""
    a, b, rmse = models.fit_inverse(x, y)
    # print the the value of RMSE and the function
    print(title + ': RMSE=%f, y = %f / x + %f' % (rmse, a, b))
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.inverse(x_i, a, b) for x_i in new_x]
    # show the prediction figure to compare with the original data
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b)


def analysis_logarithm(title: str, x_label: str, x: List[float], y_label: str,
                       y: List[float]) -> tuple:
    """Modeling between x and y using inverse model."""
    a, b, rmse = models.fit_logarithm(x, y)
    # print the the value of RMSE and the function
    print(title + ': RMSE=%f, y = %f * log(x) + %f' % (rmse, a, b))
    new_x = list(np.linspace(min(x), max(x), 1000))
    prediction_y = [models.logarithm(x_i, a, b) for x_i in new_x]
    # show the prediction figure to compare with the original data
    figure.double_figure_dot_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (a, b)


def analysis_periodic(figure_text: Tuple[str, str, str], x: List[float], y: List[float],
                      initial_guess: Tuple[float, float, float, float, float, float, float, float],
                      test_range: Tuple[float, float]) -> tuple:
    """Modeling between x and y using periodic model.
    figure_text contains title, x_label, y_label to draw a figure.
    """
    title, x_label, y_label = figure_text
    para = models.fit_periodic(x, y, initial_guess, test_range)
    # print the the value of RMSE and the function
    print(title + ': RMSE=%f, y = %f * (cos(%f * (x - %f))) + '
                  '(%f) * (cos(%f * (x - %f))) + %f * x + %f'
          % (para[8], para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7]))
    new_x = list(np.linspace(min(x), max(x), 5000))
    prediction_y = [models.periodic(x_i, para[0], para[1], para[2], para[3],
                                    para[4], para[5], para[6], para[7])
                    for x_i in new_x]
    # show the prediction figure to compare with the original data
    figure.double_figure_line(title, x_label, x, new_x, y_label, y, prediction_y)
    return (para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7])


def generate_date_list(begin: int, end: int) -> List[datetime.date]:
    """Return a list of months from the year of begin to the year of end."""
    date_list_so_far = []
    for year in range(begin, end + 1):
        for month in range(1, 13):
            date = datetime.date(year, month, 1)
            date_list_so_far.append(date)
    return date_list_so_far


def generate_time_lag_list(begin: int, end: int) -> List[int]:
    """Return a list of time lag between the first day of every month and the first day
    of the year of begin, till the year of end."""
    begin_date = datetime.date(begin, 1, 1)
    time_lag_list_so_far = []
    for year in range(begin, end + 1):
        for month in range(1, 13):
            time_lag = (datetime.date(year, month, 1) - begin_date).days
            time_lag_list_so_far.append(time_lag)
    return time_lag_list_so_far


def calculate_time_lag_list(base: int, begin: int, end: int) -> List[int]:
    """Return a list of time lag between the first day of every month and the first day
    of the year of base, from the year of begin to the year of end."""
    base_date = datetime.date(base, 1, 1)
    time_lag_list_so_far = []
    for year in range(begin, end + 1):
        for month in range(1, 13):
            time_lag = (datetime.date(year, month, 1) - base_date).days
            time_lag_list_so_far.append(time_lag)
    return time_lag_list_so_far


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['data_prep', 'figure', 'models', 'numpy', 'typing', 'datetime'],
        'allowed-io': ['analysis_periodic',
                       'analysis_logarithm',
                       'analysis_inverse',
                       'analysis_quadratic',
                       'analysis_exponential'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
