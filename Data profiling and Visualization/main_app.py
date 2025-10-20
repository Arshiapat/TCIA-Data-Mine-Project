import streamlit as st
from 1_upload_data import upload_section
from 2_data_overview import data_overview
from 3_data_visualization import visualize_data
from 4_data_filtering import filter_data
from 5_summary import summary

st.sidebar.title("Navigation")
pages = ["Upload Data", "Overview", "Visualization", "Filtering", "Summary"]
choice = st.sidebar.radio("Go to", pages)

df = None

if choice == "Upload Data":
    df = upload_section()
elif choice == "Overview":
    df = st.session_state.get('df')
    data_overview(df)
elif choice == "Visualization":
    df = st.session_state.get('df')
    visualize_data(df)
elif choice == "Filtering":
    df = st.session_state.get('df')
    filter_data(df)
elif choice == "Summary":
    df = st.session_state.get('df')
    summary(df)
