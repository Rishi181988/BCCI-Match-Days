import streamlit as st
import pandas as pd

st.title('🎈 BCCI Match Data')

st.write('This app compile all the BCCI domestic matches in one place')

with st.expander('Data'):
  st.write('**Raw Data**')

# Data Preparations
with st.sidebar:
  st.header('Input Features')
