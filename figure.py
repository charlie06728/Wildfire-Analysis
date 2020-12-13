"""CSC110 Final Project: Impact of climate change on wildfires in California and Texas

Module Description
==================
This Python file contains functions to draw figures.

This file is Copyright (c) 2020 Yuzhi Tang, Zeyang Ni, Junru Lin, and Jasmine Zhuang.
"""
from matplotlib import pyplot as plt


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


def figure_line(title: str, x_label: str, x: list, y_label: str, y: list) -> None:
    """Draw a dot figure."""
    plt.figure(title)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
