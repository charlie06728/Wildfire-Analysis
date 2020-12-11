"""The class of climate and wildfire. Climate is divided into temperature and precipitation."""
import pandas as pd
import datetime
from typing import Dict, List
from dataclasses import dataclass


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
        data = pd.read_csv(file)
        length = len(data)
        for i in range(length):
            date = datetime.datetime.strptime(data[date_title][i], '%Y-%m').date()
            if date not in self.mean:
                self.mean[date] = [float(data[mean_title][i])]
                self.max[date] = [float(data[max_title][i])]
                self.min[date] = [float(data[min_title][i])]
            else:
                self.mean[date].append(float(data[mean_title][i]))
                self.max[date].append(float(data[max_title][i]))
                self.min[date].append(float(data[min_title][i]))

    def mean_years(self, begin: int, end: int) -> List[float]:
        """Return a list of every year's average temperature from the year of begin to the year of end."""
        mean_every_year = []
        for year in range(begin, end + 1):
            mean_every_month = []
            for month in range(1, 13):
                month_data = self.mean[datetime.date(year, month, 1)]
                mean_every_month += sum(month_data) / len(month_data)
            mean_every_year.append(sum(mean_every_month) / 12)

        return mean_every_year

    def max_years(self, begin: int, end: int) -> List[float]:
        """Return a list of every year's highest temperature from the year of begin to the year of end."""
        max_every_year = []
        for year in range(begin, end + 1):
            data_of_year = []
            for month in range(1, 13):
                data_of_year += self.max[datetime.date(year, month, 1)]
            max_every_year.append(max(data_of_year))

        return max_every_year

    def min_years(self, begin: int, end: int) -> List[float]:
        """Return a list of every year's lowest temperature from the year of begin to the year of end."""
        min_every_year = []
        for year in range(begin, end + 1):
            data_of_year = []
            for month in range(1, 13):
                data_of_year += self.min[datetime.date(year, month, 1)]
            min_every_year.append(min(data_of_year))

        return min_every_year


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
        data = pd.read_csv(file)
        length = len(data)
        for i in range(length):
            date = datetime.datetime.strptime(data[date_title][i], '%Y-%m').date()
            if date not in self.total:
                self.total[date] = [float(data[prcp_title][i])]
            else:
                self.total[date].append(float(data[date][i]))

    def total_year(self, begin: int, end: int) -> List[float]:
        """Return a list of every year's total precipitation from the year of begin to the year of end."""
        total_every_year = []
        for year in range(begin, end + 1):
            mean_every_month = []
            for month in range(1, 13):
                month_data = self.total[datetime.date(year, month, 1)]
                mean_every_month += sum(month_data) / len(month_data)
            total_every_year.append(sum(mean_every_month))

        return total_every_year


class Wildfire:
    """The wildfire data of a state.

    Use the first day of a year in datetime.date to represent this year.

    Instance Attribute:
      - name: the name of the state
      - occurs: a mapping from a year to a list of every wildfire's size happening in this year
    """
    name: str
    occurs: Dict[datetime.date, List[float]]

    def __init__(self, name: str, file: str, name_title: str, date_title: str, size_title: str) -> None:
        """Initialize a new wildfire data for a state."""
        self.name = name
        self.occurs = {}
        data = pd.read_csv(file)
        length = len(data)
        for i in range(length):
            if data[name_title][i] == name:
                date = datetime.date(int(data[date_title][i]), 1, 1)
                if date not in self.occurs:
                    self.occurs[date] = [float(data[size_title][i])]
                else:
                    self.occurs[date].append(float(data[size_title][i]))

    def frequency_years(self, begin: int, end: int) -> List[int]:
        """Return a list of every year's frequency of wildfire from the year of begin to the year of end.
        The frequency is represented by the number of wildfire.
        """
        frequency_so_far = []
        for year in range(begin, end + 1):
            date = datetime.date(year, 1, 1)
            frequency_so_far.append(len(self.occurs[date]))
        return frequency_so_far

    def total_size_years(self, begin: int, end: int) -> List[float]:
        """Return a list of every year's total size of wildfire from the year of begin to the year of end."""
        total_size_so_far = []
        for year in range(begin, end + 1):
            date = datetime.date(year, 1, 1)
            total_size_so_far.append(sum(self.occurs[date]))
        return total_size_so_far


@dataclass
class State:
    """A state that is researched.

    Instance Attributes:
      - name: the name of the state
      - temp_data: the temperature data of the state
      - prcp_data: the precipitation data of the state
      - wildfire_data: the wildfire data of the state
    """
    name: str
    temp: Temperature
    prcp: Precipitation
    wildfire: Wildfire
