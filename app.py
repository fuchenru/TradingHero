import streamlit

import page_symbol_details

# test
import calculator
from scipy.io import wavfile
import ui
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from streamlit_plotly_events import plotly_events
import plotly.express as px
import streamlit as st

# test end



if __name__ == '__main__':
    st.set_page_config(layout="wide")
    page_symbol_details.run()
