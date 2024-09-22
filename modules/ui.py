import streamlit as st

import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from plotly.subplots import make_subplots

import numpy as np
from scipy import signal

import calculator

# chart datapoint icons
raw_symbols = SymbolValidator().values
up_arrow = raw_symbols[5]
down_arrow = raw_symbols[6]


def create_candlestick(fig, dates, dataset, title, y_label):
    candlestick = go.Candlestick(name=y_label, 
                                 x=dates,
                                 open=dataset['Open'],
                                 high=dataset['High'],
                                 low=dataset['Low'],
                                 close=dataset['Close'])
    fig.add_trace(candlestick)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )


def create_indicators(fig, datasets):
    for indicator in datasets:
        indicator_data = datasets[indicator]

        marker_color="lightskyblue"
        marker_symbol = 0
        if 'IsBullish' in indicator_data:
            if indicator_data['IsBullish']:
                marker_color = 'green'
                marker_symbol = 5
            else:
                marker_color = 'red'
                marker_symbol = 6

        indicator_plot = go.Scatter(name=indicator,
                                    mode="markers",
                                    x=indicator_data['Date'],
                                    y=indicator_data['Values'],
                                    marker_symbol=marker_symbol,
                                    marker_line_color="midnightblue",
                                    marker_color=marker_color,
                                    marker_line_width=2,
                                    marker_size=15,
                                    hovertemplate="%{indicator}: %{y}%{x}<br>number: %{marker.symbol}<extra></extra>")
        fig.add_trace(indicator_plot)


def create_lines(fig, dates, datasets, title, y_label):
    for key in datasets:
        line = go.Scatter(name=key, x=dates, y=datasets[key])
        fig.add_trace(line)


def create_markers(fig, dates, dataset, title, y_label, marker_symbol=3, marker_color="blue", marker_size=15):
    line = go.Scatter(name=title, x=dates, y=dataset,
                      mode="markers",
                      marker_symbol=marker_symbol,
                      marker_line_color="midnightblue",
                      marker_color=marker_color,
                      marker_line_width=2,
                      marker_size=marker_size)
    fig.add_trace(line)


def create_line(fig, dates, dataset, title="title", y_label="values", marker_symbol=4, marker_size=15, color='rgba(0,100,80,0.2)'):
    line = go.Scatter(name=title, x=dates, y=dataset, marker_line_color="yellow", fillcolor=color)
    fig.add_trace(line)


def create_fill_area(fig, dates, y_low, y_high, title, color='rgba(0,100,80,0.2)'):
    # line_low = go.Scatter(name=title, x=dates, y=y_low, fillcolor=color, showlegend=False)
    # fig.add_trace(line_low)
    # line_high = go.Scatter(name=title, x=dates, y=y_low, fillcolor=color, showlegend=False)
    # fig.add_trace(line_high)
    fill_area = go.Scatter(
        name=title,
        x=dates + dates[::-1],
        y=y_high + y_low[::-1],
        fill='toself',
        fillcolor=color,
        line=dict(color=color)
    )
    fig.add_trace(fill_area)


def create_spectrogram(dates, data_list, sampling_frequency=1, num_points_fft=128, overlap_percent=50.0, title="title", log_scale=True):
    data_list = calculator.normalize(data_list, 1, -1)

    # Spectrogram
    w = signal.blackman(num_points_fft)
    freqs, bins, pxx = signal.spectrogram(data_list, sampling_frequency, window=w, nfft=num_points_fft, noverlap=int(num_points_fft*overlap_percent/100.0))

    dates_subset = [dates[int(bin)] for bin in bins]

    if log_scale:
        z = 10 * np.log10(pxx)
    else:
        z = pxx

    trace = [go.Heatmap(
        x=dates_subset,
        y=freqs,
        z=z,
        colorscale='Jet',
    )]
    layout = go.Layout(
        title=title,
        yaxis=dict(title='Frequency'),  # x-axis label
        xaxis=dict(title='Time'),  # y-axis label
    )
    fig = go.Figure(data=trace, layout=layout)
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

    return fig


def create_heatmap(dates, data_list, bin_count=10, time_steps=20, title='title'):
    time_width = int(len(dates)/time_steps)
    dates_subset = [dates[index*time_width] for index in range(time_steps)]
    min_val = np.min(data_list)
    max_val = np.max(data_list)
    delta = (max_val-min_val)
    min_val -= 0.2*delta
    max_val += 0.2*delta
    delta = (max_val-min_val)

    bin_width = delta/(bin_count + 1)
    bins = np.arange(min_val, max_val, bin_width)

    values = np.empty(shape=(time_steps, bin_count))
    for time_index in range(time_steps):
        data_subset = data_list[time_index*time_width:time_index*time_width+time_width]
        counts, res_bins = np.histogram(data_subset, bins=bins)
        values[time_index:] = counts

    trace = [go.Heatmap(
        x=dates_subset,
        y=bins,
        z=values.transpose(),
        colorscale='Jet',
    )]
    layout = go.Layout(
        title=title,
        yaxis=dict(title='Values'),  # x-axis label
        xaxis=dict(title='Time'),  # y-axis label
    )
    fig = go.Figure(data=trace, layout=layout)
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

    return fig


def add_mouse_indicator(fig, selected_points, min, max):
    if any(selected_points):
        fig.add_shape(
                go.layout.Shape(
                    type='line',
                    x0=selected_points[0]['x'],
                    x1=selected_points[0]['x'],
                    y0=min,
                    y1=max,
                    line=dict(color='red', width=2),
                )
            )
