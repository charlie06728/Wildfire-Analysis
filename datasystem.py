"""Containing a class that contain data system class"""
from climate import Temperature, Precipitation, WildFire
from typing import Optional

class DataSystem:
    """A class that keep track of the climate data.

    Instance attributes:
     - temperature: the temperature data.
     - precipitation: the precipitation data
     - wildfire: the data of wildfire
     """
    temperature: Optional[Temperature] = None
    precipitation: Optional[Precipitation] = None
    wildfire: Optional[WildFire] = None

    def __init__(self):
        """Initialize  a new data system

        The system starts with no entities.
        """
        self.temperature = None
