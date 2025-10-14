import streamlit as st
import pandas as pd

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def upload_section():
    st.title("Step 1: Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    df = load_data(uploaded_file)
    if df is not None:
        st.success("Data successfully uploaded!")
        st.write("Preview:")
        st.dataframe(df.head())
    return df
