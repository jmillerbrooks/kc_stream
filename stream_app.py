import streamlit as st
import pandas as pd

st.title('Streamlit Demo with KC Housing Data')


map_data = pd.read_pickle('./data/forest_pred_map_df.pkl')

st.map(map_data)