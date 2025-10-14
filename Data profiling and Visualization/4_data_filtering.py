import streamlit as st

def filter_data(df):
    st.title("Step 4: Data Filtering")
    if df is not None:
        columns = df.columns.tolist()
        selected_col = st.selectbox("Filter by column:", columns)
        unique_vals = df[selected_col].unique()
        selected_val = st.selectbox("Select value:", unique_vals)
        filtered_df = df[df[selected_col] == selected_val]
        st.write(f"Filtered Data ({len(filtered_df)} rows):")
        st.dataframe(filtered_df)
        return filtered_df
    else:
        st.warning("Please upload a dataset first.")
