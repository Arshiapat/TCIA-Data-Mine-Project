import os
import streamlit as st
import pandas as pd
import sqlite3
import json
import time
import hmac
import hashlib
from datetime import datetime, timezone

APP_TITLE = "TCIA Proposal Submissions"
DB_PATH = os.path.join("data", "submissions.db")

st.set_page_config(page_title=APP_TITLE, page_icon="📤", layout="wide")

# Database query helper
def query_submissions(proposal_type: str | None = None, date: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        q = "SELECT * FROM submissions"
        filters = []
        params = []
        if proposal_type:
            filters.append("proposal_type = ?")
            params.append(proposal_type)
        if date:
            filters.append("created_at LIKE ?")
            params.append(f"{date}%")
        if filters:
            q += " WHERE " + " AND ".join(filters)
        q += " ORDER BY created_at DESC"
        df = pd.read_sql_query(q, conn, params=params)
        return df
    finally:
        conn.close()

# Token utilities
def sign_token(data: str, ttl_seconds: int = 1800) -> str:
    exp = int(time.time()) + ttl_seconds
    payload = f"{data}.{exp}"
    sig = hmac.new(b"change-me", payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"

def verify_token(token: str) -> bool:
    try:
        data, exp, sig = token.rsplit(".", 2)
        payload = f"{data}.{exp}"
        sig_expected = hmac.new(b"change-me", payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, sig_expected):
            return False
        return int(exp) >= int(time.time())
    except Exception:
        return False

def header():
    st.title(APP_TITLE)
    st.caption("Submit proposals for TCIA publication. Public form; admin view requires PIN.")

def sidebar_nav():
    return st.sidebar.radio("Navigation", ["Submit", "Admin"], index=1)

def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    pin = st.text_input("Enter Admin PIN", type="password")
    if st.button("Get Token"):
        if pin == os.getenv("ADMIN_PIN", "123456"):
            tok = sign_token("admin")
            st.session_state.admin_token = tok
            st.success("Token issued. You have 30 minutes.")
        else:
            st.error("Invalid PIN")

    token = st.text_input("Or paste an existing token", value=st.session_state.get("admin_token", ""))

    if not (token and verify_token(token)):
        st.info("Enter a valid PIN or token to proceed.")
        return

    st.markdown("---")
    ptype = st.selectbox("Proposal Type", ["", "new_collection", "analysis_results"], index=0)
    date = st.text_input("Filter by date (YYYY-MM-DD) or leave blank for all", value="")

    if st.button("Load Submissions", use_container_width=True):
        df = query_submissions(ptype if ptype else None, date if date else None)
        if df is None or df.empty:
            st.warning("No submissions found for the selected filters.")
            return
        st.dataframe(df, use_container_width=True)

        # Two download buttons side-by-side
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"tcia_submissions_{ptype or 'all'}_{date or 'all'}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            with open(DB_PATH, "rb") as f:
                db_bytes = f.read()
            st.download_button(
                label="Download Full Database (.db)",
                data=db_bytes,
                file_name="submissions.db",
                mime="application/octet-stream",
                use_container_width=True,
            )

nav = sidebar_nav()
if nav == "Admin":
    admin_page()
