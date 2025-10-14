import streamlit as st

def summary(df):
    st.title("Step 5: Summary and Insights")
    if df is not None:
        st.write("Here’s a brief summary of your dataset:")
        st.markdown(f"""
        - Rows: **{df.shape[0]}**
        - Columns: **{df.shape[1]}**
        - Missing values: **{df.isnull().sum().sum()}**
        - Numeric features: **{len(df.select_dtypes(include='number').columns)}**
        """)
        st.info("Tip: Explore outliers, missing data patterns, and feature relationships for deeper insights.")
    else:
        st.warning("Please upload a dataset first.")
