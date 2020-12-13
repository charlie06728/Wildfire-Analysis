"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains functions to draw figures.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
import datetime
from matplotlib import pyplot as plt


def get_time_list(begin: int, end: int) -> list:
    """Return a list of months from the year of begin to the year of end"""
    time_list = []
    for year in range(begin, end + 1):
        for month in range(1, 13):
            time_list.append(datetime.date(year, month, 1))

    return time_list


def figure_dot(title: str, x_label: str, x: list, y_label: str, y: list) -> None:
    """Draw a dot figure."""
    plt.plot(x, y, '.')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def double_figure_dot(title: str, x_label: str, x: list, y_label: str, y: list, z: list) -> None:
    """Draw a dot figure with x-y and x-z"""
    plt.figure(title)
    plt.plot(x, y, '.')
    plt.plot(x, z, '.')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
