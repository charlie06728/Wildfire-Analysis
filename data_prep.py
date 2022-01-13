"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains functions and classes that are used for preparing and storing
data from the raw csv files.

The three classes used for storing data are Temperature, Precipitation and Wildfire, each
storing their respective data for a particular state (e.g. California).

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
from typing import Dict, List
import datetime
import pandas as pd
import sqlite3 as sql
import os
import csv
from sqlite3 import Error


def sqlite_extract() -> None:
    """Process the original SQlite data and output a csv file called wildfire_data.csv"""
    try:
        # Connect to database
        conn = sql.connect('FPA_FOD_20170508.sqlite')

        # Export data into CSV file
        print("Exporting data into CSV............")
        cursor = conn.cursor()
        cursor.execute("select * from Fires")
        with open("wildfire_data.csv", "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow([i[0] for i in cursor.description])
            csv_writer.writerows(cursor)

        dirpath = os.getcwd() + "/wildfire_data.csv"
        print("Data exported Successfully into {}".format(dirpath))
        conn.close()

    except Error as e:
        print(e)


def filter_data() -> None:
    """Create instances to filter the data in wildfire_data.csv and ca_climate.csv
    by dropping columns that are not needed, then create two new files: new_wildfire_data.csv
    and new_ca_climate.csv to store the filtered data."""
    print("Filtering data.....")
    fire_data = pd.read_csv('wildfire_data.csv', low_memory=False)
    fire_data = fire_data[['STATE', 'FIRE_YEAR', 'DISCOVERY_DOY', 'FIRE_SIZE']]
    new_fire_data = fire_data.drop(fire_data[fire_data['STATE'] != 'CA'].index)
    new_fire_data.to_csv('new_wildfire_data.csv')
    climate_data = pd.read_csv('ca_climate.csv', low_memory=False)
    climate_data = climate_data[['DATE', 'TAVG', 'TMAX', 'TMIN', 'PRCP']]
    climate_data.to_csv('new_ca_climate.csv')


class Temperature:
    """A class that stores monthly temperature data collected from various weather stations
    of a state.

    Instance Attributes:
        - name: the name of the state, represented by two capital letters
        - mean: a mapping from a month to a list of this month's average temperature
        from different stations
        - max: a mapping from a month to a list of this month's highest temperature
        from different stations
        - min: a mapping from a month to a list of this month's lowest temperature
        from different stations

    Note: a month is represented using datetime.date() object, and it uses the first day
    of the month. E.g. 1989-12 => datetime.date(1989, 12, 1).
    """
    name: str
    mean: Dict[datetime.date, List[float]]
    max: Dict[datetime.date, List[float]]
    min: Dict[datetime.date, List[float]]

    def __init__(self, name: str, file: str, date_column: str, mean_column: str,
                 max_column: str, min_column: str) -> None:
        """Initialize a new temperature data for a state.

        Parameters:
            - name: the name of the state, represented by two capital letters.
            - file: location of the csv file.
            - date_column: name of the column containing the month of when the
            temperature data is collected.
            - mean_column: name of the column containing mean temperature data (of a month).
            - max_column: name of the column containing maximum temperature data  (of a month).
            - min_column: name of the column containing minimum temperature data (of a month).
        """
        self.name = name
        self.mean = {}
        self.max = {}
        self.min = {}

        # Read data from csv file.
        data = pd.read_csv(file, low_memory=False)

        length = len(data)

        for i in range(length):

            # Convert month string (in the form: yyyy-mm) from csv file to datetime.date() object.
            date = datetime.datetime.strptime(data[date_column][i], '%Y-%m').date()

            # Checking if the mean, max, min temperature entries for the row are valid (non-empty).
            # The temperature entry is only added if it is valid.
            mean_isvalid, max_isvalid, min_isvalid = \
                data[mean_column][i] > -500, \
                data[max_column][i] > -500, \
                data[min_column][i] > -500

            # Add data to dictionaries.
            if date not in self.mean:
                if mean_isvalid:
                    self.mean[date] = [float(data[mean_column][i])]
                if max_isvalid:
                    self.max[date] = [float(data[max_column][i])]
                if min_isvalid:
                    self.min[date] = [float(data[min_column][i])]
            else:
                if mean_isvalid:
                    self.mean[date].append(float(data[mean_column][i]))
                if max_isvalid:
                    self.max[date].append(float(data[max_column][i]))
                if min_isvalid:
                    self.min[date].append(float(data[min_column][i]))

    def mean_temperature(self, begin: int, end: int) -> List[float]:
        """Return a list of the mean temperature of every month's mean temperature entries
        (collected from the various weather stations) from the begin year to the end year."""
        mean_every_month = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                month_data = self.mean[datetime.date(year, month, 1)]
                mean_every_month.append(sum(month_data) / len(month_data))

        return mean_every_month

    def max_temperature(self, begin: int, end: int) -> List[float]:
        """Return a list of the maximum temperature of every month's max temperature entries
        (collected from the various weather stations) from the begin year to the end year."""
        max_every_month = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                max_every_month.append(max(self.max[datetime.date(year, month, 1)]))

        return max_every_month

    def min_temperature(self, begin: int, end: int) -> List[float]:
        """Return a list of the minimum temperature of every month's min temperature entries
        (collected from the various weather stations) from the begin year to the end year."""
        min_every_month = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                min_every_month.append(min(self.min[datetime.date(year, month, 1)]))

        return min_every_month


class Precipitation:
    """A class that stores monthly precipitation data collected from various weather
    stations of a state.

    Instance Attributes:
        - name: the name of the state, represented by two capital letters.
        - total: a mapping from a month to a list of this month's total precipitation
        from different stations.

    Note: a month is represented using datetime.date() object, and it uses the first day
    of the month. E.g. 1989-12 => datetime.date(1989, 12, 1).
    """
    name: str
    total: Dict[datetime.date, List[float]]

    def __init__(self, name: str, file: str, date_column: str, prcp_column: str) -> None:
        """Initialize a new precipitation data for a state.

        Parameters:
            - name: the name of the state, represented by two capital letters.
            - file: location of the csv file.
            - date_column: name of the column containing the month of when the temperature data
            is collected.
            - prcp_column: name of the column containing the total precipitation (of a month).
        """
        self.name = name
        self.total = {}

        # Read data from csv file.
        data = pd.read_csv(file, low_memory=False)

        length = len(data)

        for i in range(length):

            # Convert month string (in the form: yyyy-mm) from csv file to datetime.date() object.
            date = datetime.datetime.strptime(data[date_column][i], '%Y-%m').date()

            # Checking if the precipitation entry for the row is non-empty.
            # The precipitation entry is only added if it is non-empty.
            if data[prcp_column][i] >= 0:
                if date not in self.total:
                    self.total[date] = [float(data[prcp_column][i])]
                else:
                    self.total[date].append(float(data[prcp_column][i]))

    def total_precipitation(self, begin: int, end: int) -> List[float]:
        """Return a list of the mean precipitation of every month's total precipitation entries
        (collected from the various weather stations) from the begin year to the end year."""
        total_every_month = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                month_data = self.total[datetime.date(year, month, 1)]
                # Due to the configuration of the modelling part of the project, month_mean cannot
                # be 0, so a value close to 0 is chosen instead.
                month_mean = max(sum(month_data) / len(month_data), 0.1)
                total_every_month.append(month_mean)

        return total_every_month


class Wildfire:
    """A class that stores wildfire data of a state.

    Instance Attribute:
        - name: the name of the state, represented by two capital letters
        - occurs: a mapping from a month to a list of every wildfire's size happened in the month
    """
    name: str
    occurs: Dict[datetime.date, List[float]]

    def __init__(self, name: str, file: str, year_column: str, day_column: str,
                 size_column: str) -> None:
        """Initialize a new wildfire data for a state.

        Parameters:
            - name: the name of the state, represented by two capital letters.
            - file: location of the csv file.
            - year_column: name of the column containing the year of when the wildfire occurred.
            - day_column: name of the column containing the day of year of when the wildfire
            occurred.
            - size_column: name of the column containing the burnt area of the wildfire.
        """
        self.name = name
        self.occurs = {}

        # Read data from csv file.
        data = pd.read_csv(file, low_memory=False)

        length = len(data)

        for i in range(length):
            year = int(data[year_column][i])
            # Estimate the month in year when the wildfire occurred
            month = int(int(data[day_column][i]) / 31) + 1
            date = datetime.date(year, month, 1)
            if date not in self.occurs:
                self.occurs[date] = [float(data[size_column][i])]
            else:
                self.occurs[date].append(float(data[size_column][i]))

    def number_of_fires_in_months(self, begin: int, end: int) -> List[int]:
        """Return a list of every month's number of wildfires from the begin year to the end
        year."""
        fires_so_far = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                date = datetime.date(year, month, 1)
                fires_so_far.append(len(self.occurs[date]))
        return fires_so_far

    def mean_size_of_fires_in_months(self, begin: int, end: int) -> List[float]:
        """Return a list of every month's mean size of wildfire from the begin year to the end
        year."""
        mean_size_so_far = []
        for year in range(begin, end + 1):
            for month in range(1, 13):
                date = datetime.date(year, month, 1)
                mean_size_so_far.append(sum(self.occurs[date]) / len(self.occurs[date]))
        return mean_size_so_far


if __name__ == '__main__':
    sqlite_extract()
    # filter_data()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['pandas', 'datetime', 'typing'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R0913', 'R1705', 'C0200']
    })
