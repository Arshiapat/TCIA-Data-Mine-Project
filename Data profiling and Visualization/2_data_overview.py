import streamlit as st

def data_overview(df):
    st.title("Step 2: Data Overview")
    if df is not None:
        st.subheader("Basic Information")
        st.write(df.info(verbose=True, buf=None))
        st.write("Shape:", df.shape)
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        st.subheader("Summary Statistics")
        st.write(df.describe())
    else:
        st.warning("Please upload a dataset first.")
