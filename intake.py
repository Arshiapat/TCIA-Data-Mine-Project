"""
TCIA Unified Proposal – Streamlit App (SQLite-backed)

Restored full submission form with validation. Submissions are stored in
`data/submissions.db` (SQLite) with one column per form field. Admin view
includes filters and side-by-side CSV + .db download buttons.

This version intentionally omits email/Slack/PDF receipt logic — it keeps
only the form, validation, and database storage as requested.

Run:
- pip install -r requirements.txt (streamlit, pandas)
- streamlit run intake.py
"""

import os
import re
import io
import hmac
import time
import json
import uuid
import hashlib
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# ----------------------------
# Config & Utility
# ----------------------------
APP_TITLE = "TCIA Proposal Submissions"
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "submissions.db")
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "change-me")
ADMIN_PIN = os.getenv("ADMIN_PIN", "123456")

SUBMIT_COOLDOWN = 20

st.set_page_config(page_title=APP_TITLE, page_icon="📤", layout="wide")

# Session state init
if "last_submit_ts" not in st.session_state:
    st.session_state.last_submit_ts = 0
if "admin_authed_until" not in st.session_state:
    st.session_state.admin_authed_until = 0
if "admin_token" not in st.session_state:
    st.session_state.admin_token = None

# Ensure data dir exists
os.makedirs(DB_DIR, exist_ok=True)

# ----------------------------
# Controlled vocab
# ----------------------------
IMAGE_TYPES = [
    "MR","CT","PET","PET-CT","PET-MR","Mammograms","Ultrasound","Xray",
    "Radiation Therapy (RTSTRUCT, RTDOSE, RTPLAN)",
    "Whole Slide Image","CODEX","Single-cell Image","Photomicrograph",
    "Microarray","Multiphoton","Immunofluorescence"
]
SUPPORTING_DATA = [
    "Clinical (e.g. demographics, medical history/comorbidities, treatment details, outcomes)",
    "Image Analyses (e.g. segmentations, radiologist/pathologist reports, image features)",
    "Image Registrations",
    "Genomics (e.g. gene expression, copy number variation, methylation)",
    "Proteomics (e.g. mass spectrometry, protein arrays)",
    "Software / Source Code",
    "No additional data (only images)"
]
ANALYSIS_DERIVED_TYPES = [
    "Segmentation","Classification","Quantitative Feature","Image (e.g. converted, processed or registered images)"
]
PUBLISH_REASONS = [
    "To meet a funding agency's data sharing requirements",
    "To meet a journal's data sharing requirements",
    "To facilitate a collaborative project with investigators outside my institution",
    "To facilitate a challenge competition"
]

# ----------------------------
# Validators
# ----------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[+\d][\d\s().-]{6,}$")
ORCID_RE = re.compile(r"^(\d{4}-){3}\d{3}[\dX]$")


def valid_email(x: str) -> bool:
    return bool(x and EMAIL_RE.match(x.strip()))


def valid_phone(x: str) -> bool:
    return bool(x and PHONE_RE.match(x.strip()))


def extract_orcids(text: str) -> list:
    # naive capture of ORCIDs in a free text list
    if not text:
        return []
    return ORCID_RE.findall(text)

# ----------------------------
# Security helpers (Admin)
# ----------------------------

def sign_token(data: str, ttl_seconds: int = 1800) -> str:
    exp = int(time.time()) + ttl_seconds
    payload = f"{data}.{exp}"
    sig = hmac.new(TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str) -> bool:
    try:
        data, exp, sig = token.rsplit(".", 2)
        payload = f"{data}.{exp}"
        sig_expected = hmac.new(TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, sig_expected):
            return False
        return int(exp) >= int(time.time())
    except Exception:
        return False


def client_ip() -> str:
    return st.session_state.get("client_ip", "unknown")

# ----------------------------
# Database: init & helpers
# ----------------------------
# Full column list mirroring the original form fields
COLUMNS = {
    "submission_id": "TEXT PRIMARY KEY",
    "created_at": "TEXT",
    "proposal_type": "TEXT",
    "email": "TEXT",
    "scientific_poc": "TEXT",
    "technical_poc": "TEXT",
    "legal_admin": "TEXT",
    "time_constraints": "TEXT",
    "dataset_title": "TEXT",
    "dataset_nickname": "TEXT",
    "authors": "TEXT",
    "dataset_description": "TEXT",
    "client_ip": "TEXT",
    "user_agent": "TEXT",

    # data collection
    "published_elsewhere": "TEXT",
    "disease_site": "TEXT",
    "histologic_dx": "TEXT",
    "image_types": "TEXT",
    "image_types_other": "TEXT",
    "supporting_data": "TEXT",
    "supporting_data_other": "TEXT",
    "file_formats": "TEXT",
    "n_subjects": "INTEGER",
    "n_studies": "TEXT",
    "disk_space": "TEXT",
    "preprocessing": "TEXT",
    "faces": "TEXT",
    "usage_policy": "TEXT",
    "usage_policy_other": "TEXT",
    "dataset_publications": "TEXT",
    "additional_publications": "TEXT",
    "acknowledgments": "TEXT",
    "why_publish": "TEXT",

    # analysis-only
    "tcia_collections": "TEXT",
    "derived_types": "TEXT",
    "derived_other": "TEXT",
    "n_patients": "TEXT",
    "have_records": "TEXT",
    "primary_citation": "TEXT",
    "extra_pubs": "TEXT",
    "reasons": "TEXT",
    "reasons_other": "TEXT",
}


def init_db(path: str = DB_PATH):
    cols_sql = ",\n    ".join([f"{k} {v}" for k, v in COLUMNS.items()])
    create_sql = f"CREATE TABLE IF NOT EXISTS submissions (\n    {cols_sql}\n);"
    conn = sqlite3.connect(path, timeout=20)
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
    finally:
        conn.close()



def insert_submission(record: dict, path: str = DB_PATH):
    # normalize and map values
    norm = {}
    for k in COLUMNS.keys():
        v = record.get(k)
        if isinstance(v, list):
            norm[k] = json.dumps(v)
        elif v is None:
            norm[k] = None
        else:
            if k == "n_subjects":
                try:
                    norm[k] = int(v) if v != "" and v is not None else None
                except Exception:
                    norm[k] = None
            else:
                norm[k] = str(v) if v is not None else None

    cols = ", ".join([k for k in COLUMNS.keys()])
    placeholders = ", ".join(["?" for _ in COLUMNS.keys()])
    values = [norm[k] for k in COLUMNS.keys()]

    conn = sqlite3.connect(path, timeout=20)
    try:
        cur = conn.cursor()
        cur.execute(f"INSERT INTO submissions ({cols}) VALUES ({placeholders})", values)
        conn.commit()
    finally:
        conn.close()


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

# Initialize DB
init_db()

# ----------------------------
# UI Components
# ----------------------------

def header():
    st.title(APP_TITLE)
    st.caption("Submit proposals for TCIA publication. Public form; admin view requires PIN.")


def sidebar_nav():
    return st.sidebar.radio("Navigation", ["Submit", "Admin"], index=0)


def basic_info_block():
    st.subheader("Basic Information")
    email = st.text_input("Email *")
    sci_poc = st.text_area("Provide a scientific point of contact *", placeholder="Name, email, phone")
    tech_poc = st.text_area("Provide a technical point of contact *", placeholder="Name, email, phone")
    legal_admin = st.text_area("Provide a legal/contracts administrator *", placeholder="Name, email")
    time_constraints = st.text_area("Are there any time constraints associated with sharing your data set? *")
    return email, sci_poc, tech_poc, legal_admin, time_constraints


def dataset_pub_block(include_nickname_required: bool):
    st.subheader("Dataset Publication")
    title = st.text_input("Suggest a descriptive title for your dataset *")
    nickname_label = "Suggest a shorter nickname for your dataset" + (" *" if include_nickname_required else "")
    nickname = st.text_input(nickname_label, help="< 30 chars; letters, numbers, dashes")
    authors = st.text_area("List the authors of this data set *", help="List as (FAMILY, GIVEN); include OrcIDs")
    desc_label = "Provide a Dataset Description *"
    description = st.text_area(desc_label, height=200)
    return title, nickname, authors, description


def data_collection_block():
    st.subheader("Data Collection Details")
    published_elsewhere = st.text_area("Has this data ever been published elsewhere? *")
    disease_site = st.text_input("Primary disease site/location *", help="Use terms from provided document (Column A)")
    histologic_dx = st.text_input("Histologic diagnosis *", help="Use terms from provided document (Column B)")

    image_types = st.multiselect("Which image types are included in the data set? *", IMAGE_TYPES)
    image_types_other = st.text_input("Other image types (if any)")

    supporting_data = st.multiselect("Which kinds of supporting data are included in the data set? *", SUPPORTING_DATA)
    supporting_data_other = st.text_input("Other supporting data (if any)")

    file_formats = st.text_area("Specify the file format utilized for each type of data *")
    n_subjects = st.number_input("How many subjects are in your data set? *", min_value=0, step=1)
    n_studies = st.text_input("How many total radiology studies or pathology slides? *")
    disk_space = st.text_input("Approximate disk space required *")
    preprocessing = st.text_area("Steps taken to modify your data prior to submitting *")

    faces = st.radio("Does your data contain any images of patient faces? *", ["Yes", "No"])

    usage_policy = st.radio("Do you need to request any exceptions to TCIA's Open Access & Usage Policy? *", ["No exceptions requested", "Other"])
    usage_policy_other = st.text_input("If Other, please describe", disabled=(usage_policy=="No exceptions requested"))

    dataset_publications = st.text_area("Publications specifically about dataset contents/how to use it (citations)")
    additional_publications = st.text_area("Additional publications where findings derived from these data were discussed")
    acknowledgments = st.text_area("Additional acknowledgments or funding statements *")
    why_publish = st.text_area("Why would you like to publish this dataset on TCIA? *")

    return {
        "published_elsewhere": published_elsewhere,
        "disease_site": disease_site,
        "histologic_dx": histologic_dx,
        "image_types": image_types,
        "image_types_other": image_types_other,
        "supporting_data": supporting_data,
        "supporting_data_other": supporting_data_other,
        "file_formats": file_formats,
        "n_subjects": int(n_subjects),
        "n_studies": n_studies,
        "disk_space": disk_space,
        "preprocessing": preprocessing,
        "faces": faces,
        "usage_policy": usage_policy,
        "usage_policy_other": usage_policy_other,
        "dataset_publications": dataset_publications,
        "additional_publications": additional_publications,
        "acknowledgments": acknowledgments,
        "why_publish": why_publish,
    }


def analysis_only_block():
    st.subheader("Analysis-Specific Details")
    tcia_collections = st.text_input("Which TCIA collection(s) did you analyze? *")
    derived_types = st.multiselect("What types of derived data are included? *", ANALYSIS_DERIVED_TYPES)
    derived_other = st.text_input("Other derived data type (if any)")
    n_patients = st.text_input("How many patients? Include DICOM series or slide totals if known. *")
    disk_space = st.text_input("Approximate disk space required *")
    have_records = st.radio("Do you know exactly which TCIA images were analyzed? *", ["Yes", "No"])
    file_formats = st.text_area("Specify file format utilized for each type of data *")
    primary_citation = st.text_area("If there is a publication to cite when utilizing this data please provide the citation")
    extra_pubs = st.text_area("If there are any additional publications about this data set please list them")
    acknowledgments = st.text_area("Additional acknowledgements or funding statements *")

    reasons = st.multiselect("Why would you like to publish this dataset on TCIA? *", PUBLISH_REASONS)
    reasons_other = st.text_input("Other reason (if any)")

    return {
        "tcia_collections": tcia_collections,
        "derived_types": derived_types,
        "derived_other": derived_other,
        "n_patients": n_patients,
        "disk_space": disk_space,
        "have_records": have_records,
        "file_formats": file_formats,
        "primary_citation": primary_citation,
        "extra_pubs": extra_pubs,
        "acknowledgments": acknowledgments,
        "reasons": reasons,
        "reasons_other": reasons_other,
    }

# ----------------------------
# Submit Page
# ----------------------------

def submit_page():
    header()

    # Honeypot value we read back via query params (best-effort)
    st.markdown(
        "<div style='position:absolute;left:-10000px;' aria-hidden='true'>\n        <input name='website' value='' />\n        </div>",
        unsafe_allow_html=True,
    )
    honeypot = st.query_params.get("website", "") if hasattr(st, "query_params") else ""

    with st.form("proposal_form", clear_on_submit=False):
        proposal_type = st.selectbox("Proposal Type", ["New Collection", "Analysis Results"], index=0)

        email, sci_poc, tech_poc, legal_admin, time_constraints = basic_info_block()
        title, nickname, authors, description = dataset_pub_block(include_nickname_required=(proposal_type=="New Collection"))

        if proposal_type == "New Collection":
            details = data_collection_block()
        else:
            details = analysis_only_block()

        # Alerts config per submission (removed alert wiring, UI left optional in case desired later)
        st.markdown("---")
        st.subheader("Notification Options (optional)")
        alert_recipients = st.text_input("Recipient email(s) for alerts (comma-separated)")
        send_submitter_receipt = st.checkbox("Email a receipt to the submitter", value=True)

        submitted = st.form_submit_button("Submit Proposal", use_container_width=True)

    if submitted:
        # cooldown
        now = time.time()
        if now - st.session_state.last_submit_ts < SUBMIT_COOLDOWN:
            st.warning("You're submitting too fast. Please wait a few seconds and try again.")
            return

        st.session_state.last_submit_ts = now

        # validations
        errors = []
        if honeypot:
            errors.append("Spam detected.")
        if not valid_email(email):
            errors.append("Valid Email is required.")
        if not sci_poc.strip():
            errors.append("Scientific point of contact is required.")
        if not tech_poc.strip():
            errors.append("Technical point of contact is required.")
        if not legal_admin.strip():
            errors.append("Legal/contracts administrator is required.")
        if not time_constraints.strip():
            errors.append("Time constraints field is required.")
        if not title.strip():
            errors.append("Dataset title is required.")
        if proposal_type == "New Collection" and not nickname.strip():
            errors.append("Dataset nickname is required for New Collection.")
        if not authors.strip():
            errors.append("Authors are required.")
        if not description.strip():
            errors.append("Dataset description is required.")

        # Additional requireds by type
        if proposal_type == "New Collection":
            must = [
                (details["published_elsewhere"], "Published elsewhere details are required."),
                (details["disease_site"], "Primary disease site is required."),
                (details["histologic_dx"], "Histologic diagnosis is required."),
                (details["image_types"], "At least one image type is required."),
                (details["supporting_data"], "At least one supporting data type is required."),
                (details["file_formats"], "File formats are required."),
                (details["n_subjects"] >= 0, None),
                (details["n_studies"], "Total studies/slides is required."),
                (details["disk_space"], "Approximate disk space is required."),
                (details["preprocessing"], "Preprocessing steps are required."),
                (details["faces"], "Faces selection is required."),
                (details["usage_policy"], "Usage policy selection is required."),
                (details["acknowledgments"], "Acknowledgments are required."),
                (details["why_publish"], "Reason to publish is required."),
            ]
            for val, msg in must:
                if (isinstance(val, list) and len(val)==0) or (isinstance(val, str) and not val.strip()):
                    if msg:
                        errors.append(msg)
        else:
            must = [
                (details["tcia_collections"], "TCIA collections analyzed are required."),
                (details["derived_types"], "At least one derived data type is required."),
                (details["n_patients"], "Patient/series totals are required."),
                (details["disk_space"], "Approximate disk space is required."),
                (details["have_records"], "Record knowledge selection is required."),
                (details["file_formats"], "File formats are required."),
                (details["acknowledgments"], "Acknowledgments are required."),
                (details["reasons"], "At least one reason is required."),
            ]
            for val, msg in must:
                if (isinstance(val, list) and len(val)==0) or (isinstance(val, str) and not val.strip()):
                    if msg:
                        errors.append(msg)

        if errors:
            st.error("\n".join([f"• {e}" for e in errors]))
            return

        submission_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        record = {
            "submission_id": submission_id,
            "created_at": created_at,
            "proposal_type": proposal_type.replace(" ", "_").lower(),
            "email": email.strip(),
            "scientific_poc": sci_poc.strip(),
            "technical_poc": tech_poc.strip(),
            "legal_admin": legal_admin.strip(),
            "time_constraints": time_constraints.strip(),
            "dataset_title": title.strip(),
            "dataset_nickname": nickname.strip(),
            "authors": authors.strip(),
            "dataset_description": description.strip(),
            "client_ip": client_ip(),
            "user_agent": st.session_state.get("user_agent", "unknown"),
        }
        # merge details
        for k, v in details.items():
            record[k] = v

        # persist to sqlite
        insert_submission(record)

        st.success("Submission received! Your Submission ID is: " + submission_id)
        st.download_button("Download Receipt (JSON)", json.dumps(record, indent=2), file_name=f"tcia_receipt_{submission_id}.json")


# ----------------------------
# Admin Page
# ----------------------------

def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    col1, col2 = st.columns(2)
    with col1:
        pin = st.text_input("Enter Admin PIN", type="password")
    with col2:
        if st.button("Get Token", use_container_width=True):
            if pin == ADMIN_PIN:
                tok = sign_token("admin", ttl_seconds=1800)
                st.session_state.admin_token = tok
                st.session_state.admin_authed_until = time.time() + 1800
                st.success("Token issued. You have 30 minutes.")
            else:
                st.error("Invalid PIN")

    token = st.text_input("Or paste an existing token", value=st.session_state.admin_token or "")

    if not (token and verify_token(token)):
        st.info("Enter a valid PIN or token to proceed.")
        return

    st.markdown("---")
    ptype = st.selectbox("Proposal Type", ["", "new_collection", "analysis_results"], index=0)
    date = st.text_input("Date partition (YYYY-MM-DD) or leave blank for all", value="")

    if st.button("Load Submissions", use_container_width=True):
        records_df = query_submissions(ptype if ptype else None, date if date else None)
        if records_df is None or records_df.empty:
            st.warning("No submissions found for the selected filters.")
        else:
            st.dataframe(records_df, use_container_width=True)
            colA, colB = st.columns(2)
            with colA:
                csv = records_df.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, file_name=f"tcia_{ptype or 'all'}_{date or 'all'}.csv")
            with colB:
                with open(DB_PATH, "rb") as f:
                    db_bytes = f.read()
                st.download_button(
                    "Download Full Database (.db)",
                    db_bytes,
                    file_name="submissions.db",
                    mime="application/octet-stream",
                )


# ----------------------------
# Main
# ----------------------------
nav = sidebar_nav()
if nav == "Submit":
    submit_page()
else:
    admin_page()
