"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains the classes of Temperature, Precipitation and Wildfire.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
import pandas as pd
import datetime
from typing import Dict, List


class Temperature:
    """The temperature data of a state.

    Use the first day of a month in datetime.date to represent this month.

    Instance Attributes:
      - name: the name of the state
      - mean: a mapping from a month to a list of this month's average temperature from different stations
      - max: a mapping from a month to a list of this month's highest temperature from different stations
      - min: a mapping from a month to a list of this month's lowest temperature from different stations
    """
    name: str
    mean: Dict[datetime.date, List[float]]
    max: Dict[datetime.date, List[float]]
    min: Dict[datetime.date, List[float]]

    def __init__(self, name: str, file: str, date_title: str, mean_title: str, max_title: str, min_title: str) -> None:
        """Initialize a new temperature data for a state."""
        self.name = name
        self.mean = {}
        self.max = {}
        self.min = {}
        data = pd.read_csv(file, low_memory=False)
        length = len(data)
        for i in range(length):
            date = datetime.datetime.strptime(data[date_title][i], '%Y-%m').date()
            if float(data[mean_title][i]) > 0:
                if date not in self.mean:
                    self.mean[date] = [float(data[mean_title][i])]
                    self.max[date] = [float(data[max_title][i])]
                    self.min[date] = [float(data[min_title][i])]
                else:
                    self.mean[date].append(float(data[mean_title][i]))
                    self.max[date].append(float(data[max_title][i]))
                    self.min[date].append(float(data[min_title][i]))

    def mean_months(self, begin: int, end: int) -> List[float]:
        """Return a list of every month's average temperature from the year of begin to the year of end."""
        mean_every_month = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                month_data = self.mean[datetime.date(year, month, 1)]
                mean_every_month.append(sum(month_data) / len(month_data))

        return mean_every_month

    def max_months(self, begin: int, end: int) -> List[float]:
        """Return a list of every month's highest temperature from the year of begin to the year of end."""
        max_every_month = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                max_every_month.append(max(self.max[datetime.date(year, month, 1)]))

        return max_every_month

    def min_months(self, begin: int, end: int) -> List[float]:
        """Return a list of every month's lowest temperature from the year of begin to the year of end."""
        min_every_month = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                min_every_month.append(max(self.min[datetime.date(year, month, 1)]))

        return min_every_month


class Precipitation:
    """The precipitation data of a state.

    Use the first day of a month in datetime.date to represent this month.

    Instance Attributes:
      - name: the name of the state
      - total: a mapping from a month to a list of this month's total precipitation from different stations
    """
    name: str
    total: Dict[datetime.date, List[float]]

    def __init__(self, name: str, file: str, date_title: str, prcp_title: str) -> None:
        """Initialize a new precipitation data for a state."""
        self.name = name
        self.total = {}
        data = pd.read_csv(file, low_memory=False)
        length = len(data)
        for i in range(length):
            date = datetime.datetime.strptime(data[date_title][i], '%Y-%m').date()
            if float(data[prcp_title][i]) >= 0:
                if date not in self.total:
                    self.total[date] = [float(data[prcp_title][i])]
                else:
                    self.total[date].append(float(data[prcp_title][i]))

    def total_months(self, begin: int, end: int) -> List[float]:
        """Return a list of every month's total precipitation from the year of begin to the year of end, which is the
        average of all stations.
        """
        total_every_month = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                month_data = self.total[datetime.date(year, month, 1)]
                month_mean = max(sum(month_data) / len(month_data), 0.1)
                total_every_month.append(month_mean)

        return total_every_month


class Wildfire:
    """The wildfire data of a state.

    Use the first day of a year in datetime.date to represent this year.

    Instance Attribute:
      - name: the name of the state
      - occurs: a mapping from a year to a list of every wildfire's size happening in this year
    """
    name: str
    occurs: Dict[datetime.date, List[float]]

    def __init__(self, name: str, file: str, name_title: str, year_title: str, day_title: str, size_title: str) -> None:
        """Initialize a new wildfire data for a state."""
        self.name = name
        self.occurs = {}
        data = pd.read_csv(file, low_memory=False)
        length = len(data)
        for i in range(length):
            if data[name_title][i] == name:
                year = int(data[year_title][i])
                month = int(int(data[day_title][i]) / 31) + 1
                date = datetime.date(year, month, 1)
                if date not in self.occurs:
                    self.occurs[date] = [float(data[size_title][i])]
                else:
                    self.occurs[date].append(float(data[size_title][i]))

    def frequency_months(self, begin: int, end: int) -> List[int]:
        """Return a list of every month's frequency of wildfire from the year of begin to the year of end.
        The frequency is represented by the number of wildfire.
        """
        frequency_so_far = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                date = datetime.date(year, month, 1)
                frequency_so_far.append(len(self.occurs[date]))
        return frequency_so_far

    def mean_size_months(self, begin: int, end: int) -> List[float]:
        """Return a list of every month's average size of wildfire from the year of begin to the year of end."""
        mean_size_so_far = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                date = datetime.date(year, month, 1)
                mean_size_so_far.append(sum(self.occurs[date]) / len(self.occurs[date]))
        return mean_size_so_far
