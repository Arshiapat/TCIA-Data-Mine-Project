import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    st.title("Step 3: Data Visualization")
    if df is not None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        selected_col = st.selectbox("Select a column to visualize:", numeric_cols)
        
        st.subheader("Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please upload a dataset first.")
