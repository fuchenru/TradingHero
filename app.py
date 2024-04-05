import streamlit

import TradingHero

import calculator
from scipy.io import wavfile
import ui
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from streamlit_plotly_events import plotly_events
import plotly.express as px
import streamlit as st


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    TradingHero.run()
