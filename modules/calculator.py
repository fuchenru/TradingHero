import numpy as np
import scipy as sp
from datetime import datetime

# test
import streamlit as st

def linear_regression_line(dates, y_list):
    if not any(dates):
        return 0, 0
    if isinstance(dates[0], datetime):
        dates = [ts.timestamp() for ts in dates]
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    if not isinstance(y_list, np.ndarray):
        y_list = np.array(y_list)

    mean_x = np.mean(dates)
    mean_y = np.mean(y_list)

    # Calculate the slope (m) and y-intercept (c) of the regression line
    m = np.sum((dates - mean_x) * (y_list - mean_y)) / np.sum((dates - mean_x) ** 2)
    c = mean_y - m * mean_x

    return m, c


def linear_regression_points(dates, y_list):
    if not any(dates):
        return 0, 0
    if isinstance(dates[0], datetime):
        dates = [ts.timestamp() for ts in dates]
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    if not isinstance(y_list, np.ndarray):
        y_list = np.array(y_list)

    m, c = linear_regression_line(dates, y_list)

    return m * dates + c


def linear_regression(dates, y_list):
    if not any(dates):
        return 0, 0
    if isinstance(dates[0], datetime):
        dates = [ts.timestamp() for ts in dates]
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    if not isinstance(y_list, np.ndarray):
        y_list = np.array(y_list)

    m, c = linear_regression_line(dates, y_list)

    y_low = m * dates[0] + c
    y_high = m * dates[-1] + c

    return y_low, y_high


def normalize(data_list, high=1.0, low=-1.0):
    data = np.array(data_list)
    min_val = np.min(data)
    max_val = np.max(data)
    delta = max_val - min_val
    new_delta = high - low

    data = ((data - min_val) * new_delta / delta) + low
    return data

def intercepts(data1, data2):
    intercept_indices = []
    prev_data1_is_above = True
    for index, (data1_v, data2_v) in enumerate(zip(data1, data2)):
        data1_is_above = data1_v - data2_v >= 0.0
        if index is not 0 and (not any(intercept_indices) or index is not intercept_indices[-1] + 1) and prev_data1_is_above is not data1_is_above:
            intercept_indices.append(index)
        prev_data1_is_above = data1_is_above

    return intercept_indices