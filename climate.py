""" Containing classes of the climate data of a state"""
import pandas as pd
from typing import Tuple, List, Optional
import datetime
from priority_queue import ClimateQueue, FireQueue, FireEvent


class Climate:
    """A abstract class representing a climate data

    Instance Attributes:
     - data: A priority queue that store the data in form of (datetime.date, data)
     """
    data: ClimateQueue

    def __init__(self) -> None:
        """Initialize the class with a empty priority queue."""
        self.data = ClimateQueue()

    def read_data(self, filename: str, date_col_name: str, temp_col_name: str) -> None:
        """Read a csv file and extract the date and temperature data from the file."""
        reader = pd.read_csv(filename, low_memory=False)
        length = len(reader)
        for i in range(length):
            date = datetime.datetime.strptime(reader[date_col_name][i], '%Y-%m').date()
            temp = float(reader[temp_col_name][i])
            self.data.enqueue(date, temp)

        self.data.eliminate_duplicate()

    def access_data(self, date: datetime.date) -> float:
        """Return a temperature data of a specific month or None if the date
        is out of the range."""
        return self.data.access_data(date)

    def get_list(self) -> List[float]:
        """return a list that contain the temperature data without dates. List[float]"""
        return self.data.get_list()

    def get_queue(self) -> List[Tuple[datetime.date, float]]:
        """Return the total priority queue."""
        return self.data.get_queue()


class WildFire:
    """"A class that store the wild fire data of a given state.

    Instance Attributes:
     - data: A priority queue that store the wild fire data in form of (datetime.date, times)
     """
    data: FireQueue

    def __init__(self) -> None:
        """Initialize the class with a empty priority queue."""
        self.data = FireQueue()

    def read_data(self, filename: str, fod_id: str, size: str, year: str,
                  spot_day: str, spot_time: str, end_day: str, end_time: str) -> None:
        """Read a csv file and extract the date and temperature data from the file."""
        reader = pd.read_csv(filename, low_memory=False)
        length = len(reader)

        # for i in range(length):
        i = 0
        while i < length:
            try:
                current_date = self._get_date(int(reader[spot_day][i]), int(reader[year][i]))
                current_id = int(reader[fod_id][i])
                current_size = float(reader[size][i])
                current_duration = self._get_minutes(reader[spot_day][i], reader[spot_time][i],
                                                     reader[end_day][i], reader[end_time][i])
                self.data.enqueue(current_date,
                                  (FireEvent(current_id, current_date, current_size, current_duration)))
            except ValueError:
                pass
            i += 1

        self.data.eliminate_duplicate()
        self.data.complete_timeline()

    def get_queue(self) -> List[Tuple[datetime.date, FireEvent]]:
        """Get queue"""
        return self.data.get_queue()

    def _get_date(self, date: int, year: int) -> datetime.date:
        """Transfer the date from 'the number of date in the year' to the month in number
        We assume that there are 30 days in one month."""
        month = date // 30 + 1
        result = datetime.date(year, month, 1)
        return result

    def _get_minutes(self, spot_day: str, spot_time: str, end_day: str, end_time: str) -> int:
        """Return the minutes of the timedelta from it's original form hhmm where hh=hour, mm=minutes"""
        minus = 0
        day_diff = int(end_day) - int(spot_day)
        minus += day_diff * 1440

        min_diff = int(int(end_time) % 100) - int(int(spot_time) % 100)
        hour_diff = int(int(end_time) // 100) - int(int(spot_time) // 100)

        minus += min_diff + hour_diff * 60
        return minus
