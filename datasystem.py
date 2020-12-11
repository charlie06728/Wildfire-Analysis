"""Containing a class that contain data system class"""
from climate import Climate, WildFire
from typing import Optional, List, Tuple
import datetime


class FireEvent:
    """A class that represents a single wild fire event

    Instance Attributes:
     - fod_id: Global unique identifier(FOD_ID) of the fire event
     - date: the date that this fire event was spotted and reported.
     - size: Estimate of acres within the final perimeter of the fire.
     - duration: Time of day that the fire was declared contained
            or otherwise controlled (hhmm where hh=hour, mm=minutes).
     - year: Calendar year in which the fire was discovered or confirmed to exist.
     - times: the times of event, default 1 and will be changed when merging data, after
     calling datasystem(), this value represents the times of wild fire that happened in
     this month.
     """
    fod_id: int
    date: datetime.date
    size: float
    duration: int
    times: int

    def __init__(self, fod_id: int, date: datetime.date, size: float, duration: int) -> None:
        """Initialize a a FireEvent class """
        self.fod_id = fod_id
        self.date = date
        self.size = size
        self.duration = duration
        self.times = 1


class DataSystem:
    """A class that keep track of the climate data.

    Instance attributes:
     - temperature: the temperature data.
     - precipitation: the precipitation data
     - wildfire: the data of wildfire
     """
    ca_temperature: List[Tuple[datetime.date, float]]
    ca_precipitation: List[Tuple[datetime.date, float]]
    tex_temperature: List[Tuple[datetime.date, float]]
    tex_precipitation: List[Tuple[datetime.date, float]]
    wildfire: List[Tuple[datetime.date, FireEvent]]

    def __init__(self) -> None:
        """Initialize  a new data system

        The system starts with no entities.
        """
        self.ca_temperature = []
        self.tex_temperature = []
        self.ca_precipitation = []
        self.tex_precipitation = []
        self.wildfire = []

        self._load_data()

    def _load_data(self) -> None:
        """A private function, don't need to read."""
        ca_temp = Climate()
        ca_temp.read_data('ca_climate.csv', 'DATE', 'TAVG')
        self.temperature = ca_temp.get_queue()

        ca_prcp = Climate()
        ca_prcp.read_data('ca_climate.csv', 'DATE', 'PRCP')
        self.ca_precipitation = ca_prcp.get_queue()

        tex_temp = Climate()
        tex_temp.read_data('Texas_climate.csv', 'DATE', 'TAVG')
        self.tex_temperature = tex_temp.get_queue()

        tex_temp = Climate()
        tex_temp.read_data('Texas_climate.csv', 'DATE', 'PRCP')
        self.tex_temperature = tex_temp.get_queue()

        fire_data = WildFire()
        fire_data.read_data('wildfire_data.csv', 'FOD_ID', 'FIRE_SIZE', 'FIRE_YEAR', 'DISCOVERY_DOY',
                            'DISCOVERY_TIME', 'CONT_DOY', 'CONT_TIME')
        self.wildfire = fire_data.get_queue()







