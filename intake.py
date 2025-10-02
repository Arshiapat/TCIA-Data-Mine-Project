"""
TCIA Unified Proposal – Streamlit App

Features
- Public submission (no login) with required-field validation
- Single app for two proposal types: New Collection Proposal, Analysis Results Proposal
- Controlled vocab for multi-select fields from user's spec (with "Other" text boxes)
- Append-only Parquet dataset partitioned by proposal_type & dt=YYYY-MM-DD
- UUID + created_at, client IP (best-effort) and user agent snapshot
- Basic abuse protection: honeypot, per-session cooldown, simple request throttle
- Admin reviewer view protected by PIN + time-limited signed token (HMAC)
- Optional alerts: SMTP email + Slack webhook (env-driven)
- Optional PDF receipt (ReportLab) for submitter (env-driven)
- Streamlit Community Cloud friendly (local file storage)

Env (.streamlit/secrets.toml or OS env)
- DATA_DIR = "./data/parquet"
- ADMIN_PIN = "123456"
- ADMIN_IP_ALLOWLIST = ""  # comma-separated (optional), e.g. "1.2.3.4,5.6.7.8"
- TOKEN_SECRET = "change-me"
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS (optional)
- EMAIL_FROM_ADDR = "johntuck4234@gmail.com"  # testing default
- EMAIL_FROM_NAME = "TCIA Submissions"
- SLACK_WEBHOOK_URL (optional)
- BRAND_LOGO_URL (optional, used in emails/PDF)

Run locally
- pip install -r requirements.txt
- streamlit run app.py

Deploy: Streamlit Community Cloud
- Add these env vars in the app settings → Secrets.
- Ensure the app has write access to ./data/ (default in Community Cloud ephemeral FS).
"""

import os
import re
import io
import hmac
import time
import json
import uuid
import smtplib
import hashlib
import textwrap
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import pandas as pd
import streamlit as st

# ----------------------------
# Config & Utility
# ----------------------------
APP_TITLE = "TCIA Proposal Submissions"
DATA_DIR = os.getenv("DATA_DIR", "C:\\Users\\harir\\OneDrive\\Documents\\College\\Year3\\Fall2025\\TDM")
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "change-me")
ADMIN_PIN = os.getenv("ADMIN_PIN", "123456")
ADMIN_IP_ALLOWLIST = [ip.strip() for ip in os.getenv("ADMIN_IP_ALLOWLIST", "").split(',') if ip.strip()]

EMAIL_FROM_ADDR = os.getenv("EMAIL_FROM_ADDR", "johntuck4234@gmail.com")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "TCIA Submissions")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
BRAND_LOGO_URL = os.getenv("BRAND_LOGO_URL")

# Cooldowns / throttling (seconds)
SUBMIT_COOLDOWN = 20

st.set_page_config(page_title=APP_TITLE, page_icon="📤", layout="wide")

# Session state init
if "last_submit_ts" not in st.session_state:
    st.session_state.last_submit_ts = 0
if "admin_authed_until" not in st.session_state:
    st.session_state.admin_authed_until = 0
if "admin_token" not in st.session_state:
    st.session_state.admin_token = None

# Ensure data dir
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Controlled vocab (from user paste)
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
    return bool(EMAIL_RE.match(x.strip())) if x else False


def valid_phone(x: str) -> bool:
    return bool(PHONE_RE.match(x.strip())) if x else False


def extract_orcids(text: str) -> list:
    # naive capture of ORCIDs in a free text list
    return ORCID_RE.findall(text)

# ----------------------------
# Parquet engine helper (PyArrow preferred, fallback to Fastparquet)
# ----------------------------

def parquet_engine() -> str:
    """Return pandas parquet engine ('pyarrow' or 'fastparquet'); error if neither installed."""
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return "fastparquet"
        except Exception as e:
            raise RuntimeError("Install either 'pyarrow' or 'fastparquet' to use Parquet.") from e

# ----------------------------
# Storage: append-only parquet dataset
# ----------------------------

def write_record(proposal_type: str, record: dict):
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    part_dir = os.path.join(DATA_DIR, f"proposal_type={proposal_type}", f"dt={dt}")
    os.makedirs(part_dir, exist_ok=True)
    file_path = os.path.join(part_dir, "data.parquet")
    df_new = pd.DataFrame([record])
    if os.path.exists(file_path):
        df_existing = pd.read_parquet(file_path, engine=parquet_engine())
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all.to_parquet(file_path, engine=parquet_engine(), index=False)
    else:
        df_new.to_parquet(file_path, engine=parquet_engine(), index=False)


# ----------------------------
# Alerts (optional)
# ----------------------------

def send_slack(message: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        import requests
        requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=6)
    except Exception:
        pass


def send_email(to_addrs: list[str], subject: str, html_body: str, attachments: list[tuple[str, bytes]] | None = None):
    if not SMTP_HOST or not EMAIL_FROM_ADDR:
        return
    msg = MIMEMultipart()
    msg["From"] = f"{EMAIL_FROM_NAME} <{EMAIL_FROM_ADDR}>"
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    for name, data in (attachments or []):
        part = MIMEApplication(data, Name=name)
        part["Content-Disposition"] = f'attachment; filename="{name}"'
        msg.attach(part)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        if SMTP_USER and SMTP_PASS:
            s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(EMAIL_FROM_ADDR, to_addrs, msg.as_string())


# ----------------------------
# PDF receipt (optional)
# ----------------------------

def render_pdf_receipt(record: dict) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        width, height = LETTER
        y = height - 1*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y, "TCIA Proposal Submission Receipt")
        y -= 0.3*inch
        c.setFont("Helvetica", 10)
        c.drawString(1*inch, y, f"Submission ID: {record['submission_id']}")
        y -= 0.2*inch
        c.drawString(1*inch, y, f"Type: {record['proposal_type']}")
        y -= 0.2*inch
        c.drawString(1*inch, y, f"Created At (UTC): {record['created_at']}")
        y -= 0.3*inch
        wrap = textwrap.wrap(f"Title: {record.get('dataset_title','')}", width=90)
        for line in wrap:
            c.drawString(1*inch, y, line); y -= 0.18*inch
        y -= 0.1*inch
        wrap = textwrap.wrap(f"Email: {record.get('email','')}", width=90)
        for line in wrap:
            c.drawString(1*inch, y, line); y -= 0.18*inch
        c.showPage(); c.save()
        return buf.getvalue()
    except Exception:
        return None


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
    # Streamlit Cloud may not expose IP; allow forwarding headers if proxied
    return st.session_state.get("client_ip", "unknown")


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

    st.markdown(
        "<div style='position:absolute;left:-10000px;' aria-hidden='true'>\n        <input name='website' value='' />\n        </div>",
        unsafe_allow_html=True,
    )
    # Honeypot value we read back via query params (best-effort)
    honeypot = st.query_params.get("website", "") if hasattr(st, "query_params") else ""

    with st.form("proposal_form", clear_on_submit=False):
        proposal_type = st.selectbox("Proposal Type", ["New Collection", "Analysis Results"], index=0)

        email, sci_poc, tech_poc, legal_admin, time_constraints = basic_info_block()
        title, nickname, authors, description = dataset_pub_block(include_nickname_required=(proposal_type=="New Collection"))

        if proposal_type == "New Collection":
            details = data_collection_block()
        else:
            details = analysis_only_block()

        # Alerts config per submission (optional)
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
            # normalize lists to json strings for Parquet compatibility
            if isinstance(v, list):
                record[k] = json.dumps(v)
            else:
                record[k] = v

        # persist
        write_record(record["proposal_type"], record)

        # PDF receipt
        pdf_bytes = render_pdf_receipt(record) or b""

        # Alerts
        recipients = [x.strip() for x in (alert_recipients or "").split(',') if x.strip() and valid_email(x.strip())]
        if recipients:
            send_email(
                recipients,
                subject=f"[TCIA] New {proposal_type} submission",
                html_body=f"<p>New submission received.</p><pre>{json.dumps(record, indent=2)}</pre>",
                attachments=[("receipt.pdf", pdf_bytes)] if pdf_bytes else None,
            )
        send_slack(f"New {proposal_type} submission: {submission_id}")

        # Submitter receipt
        if send_submitter_receipt and valid_email(record["email"]):
            send_email(
                [record["email"]],
                subject=f"TCIA Submission Receipt – {submission_id}",
                html_body=f"<p>Thank you for your submission.</p><pre>{json.dumps({k:v for k,v in record.items() if k not in ['client_ip','user_agent']}, indent=2)}</pre>",
                attachments=[("receipt.pdf", pdf_bytes)] if pdf_bytes else None,
            )

        st.success("Submission received! Your Submission ID is: " + submission_id)
        st.download_button("Download Receipt (JSON)", json.dumps(record, indent=2), file_name=f"tcia_receipt_{submission_id}.json")


# ----------------------------
# Admin Page
# ----------------------------


# John's admin page without e
#def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    # IP allowlist check (best-effort; often not available on Streamlit Cloud)
    if ADMIN_IP_ALLOWLIST:
        # Placeholder hint; real client IP capture would need a backend
        ip = client_ip()
        if ip not in ADMIN_IP_ALLOWLIST:
            st.error("Your IP is not allowlisted.")
            return

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

    # Controls
    st.markdown("---")
    ptype = st.selectbox("Proposal Type", ["new_collection", "analysis_results"], index=0)
    date = st.text_input("Date partition (YYYY-MM-DD) or leave blank for all", value="")

    if st.button("Load Submissions", use_container_width=True):
        records_df = load_records(ptype, date)
        if records_df is None or records_df.empty:
            st.warning("No submissions found for the selected filters.")
        else:
            st.dataframe(records_df, use_container_width=True)
            csv = records_df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, file_name=f"tcia_{ptype}_{date or 'all'}.csv")
            # Parquet download of the concatenated view
            try:
                buf = io.BytesIO()
                records_df.to_parquet(buf, engine=parquet_engine(), index=False)
                st.download_button("Download Parquet", buf.getvalue(), file_name=f"tcia_{ptype}_{date or 'all'}.parquet")
            except Exception as e:
                st.info("Parquet download not available: " + str(e))


#def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    # Auth check
    if ADMIN_IP_ALLOWLIST:
        ip = client_ip()
        if ip not in ADMIN_IP_ALLOWLIST:
            st.error("Your IP is not allowlisted.")
            return

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

    # Controls
    st.markdown("---")
    ptype = st.selectbox("Proposal Type", ["new_collection", "analysis_results"], index=0)
    date = st.text_input("Date partition (YYYY-MM-DD) or leave blank for all", value="")

    if st.button("Load Submissions", use_container_width=True):
        records_df = load_records(ptype, date)
        if records_df is None or records_df.empty:
            st.warning("No submissions found for the selected filters.")
        else:
            st.dataframe(records_df, use_container_width=True)

            # Select submission to edit
            submission_ids = records_df["submission_id"].tolist()
            selected_id = st.selectbox("Select a Submission to Edit", [""] + submission_ids)

            if selected_id:
                record = records_df[records_df["submission_id"] == selected_id].iloc[0].to_dict()

                st.write("### Edit Submission")
                with st.form("edit_form", clear_on_submit=False):
                    new_title = st.text_input("Dataset Title", value=record.get("dataset_title", ""))
                    new_desc = st.text_area("Dataset Description", value=record.get("dataset_description", ""))
                    new_authors = st.text_area("Authors", value=record.get("authors", ""))

                    updated = st.form_submit_button("Update Submission", use_container_width=True)

                if updated:
                    # Update record in dataframe
                    idx = records_df.index[records_df["submission_id"] == selected_id][0]
                    records_df.at[idx, "dataset_title"] = new_title.strip()
                    records_df.at[idx, "dataset_description"] = new_desc.strip()
                    records_df.at[idx, "authors"] = new_authors.strip()

                    # Save back to Parquet (overwrite partition)
                    part_dir = os.path.join(DATA_DIR, f"proposal_type={ptype}")
                    if date:
                        part_dir = os.path.join(part_dir, f"dt={date}")
                    os.makedirs(part_dir, exist_ok=True)
                    file_path = os.path.join(part_dir, "data.parquet")
                    records_df.to_parquet(file_path, engine=parquet_engine(), index=False)

                    st.success(f"Submission {selected_id} updated successfully!")



# def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    # --- Auth ---
    if ADMIN_IP_ALLOWLIST:
        ip = client_ip()
        if ip not in ADMIN_IP_ALLOWLIST:
            st.error("Your IP is not allowlisted.")
            return

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

    # --- Filters ---
    st.markdown("---")
    ptype = st.selectbox("Proposal Type", ["new_collection", "analysis_results"], index=0)
    date = st.text_input("Date partition (YYYY-MM-DD) or leave blank for all", value="")

    # --- Load submissions into session_state ---
    if st.button("Load Submissions", use_container_width=True):
        records_df = load_records(ptype, date)
        if records_df is None or records_df.empty:
            st.warning("No submissions found for the selected filters.")
        else:
            st.session_state.records_df = records_df
            st.session_state.ptype = ptype
            st.session_state.date = date
            st.success("Submissions loaded.")

    # --- Show submissions if available ---
    #if "records_df" in st.session_state and not st.session_state.records_df.empty:
        records_df = st.session_state.records_df
        st.dataframe(records_df, use_container_width=True)

        submission_ids = records_df["submission_id"].tolist()
        selected_id = st.selectbox("Select a Submission to Edit", [""] + submission_ids)

        if selected_id:
            record = records_df[records_df["submission_id"] == selected_id].iloc[0].to_dict()

            st.write("### Edit Submission")
            with st.form("edit_form", clear_on_submit=False):
                new_title = st.text_input("Dataset Title", value=record.get("dataset_title", ""))
                new_desc = st.text_area("Dataset Description", value=record.get("dataset_description", ""))
                new_authors = st.text_area("Authors", value=record.get("authors", ""))

                updated = st.form_submit_button("Update Submission", use_container_width=True)

            if updated:
                ...
    
    if "records_df" in st.session_state and not st.session_state.records_df.empty:
        records_df = st.session_state.records_df
        st.dataframe(records_df, use_container_width=True)

    submission_ids = records_df["submission_id"].tolist()
    selected_id = st.selectbox("Select a Submission", [""] + submission_ids)

    if selected_id:
        record = records_df[records_df["submission_id"] == selected_id].iloc[0].to_dict()

        st.write("### Manage Submission")

        # ---- Edit Form (dynamic) ----
        updated_values = {}
        with st.form("edit_form", clear_on_submit=False):
            for k, v in record.items():
                if k in ["submission_id", "created_at", "client_ip", "user_agent"]:
                    st.text_input(k, value=str(v), disabled=True)
                    updated_values[k] = v
                    continue

                if isinstance(v, (int, float)):
                    updated_values[k] = st.number_input(k, value=float(v))
                elif isinstance(v, str) and len(v) > 150:
                    updated_values[k] = st.text_area(k, value=v)
                else:
                    updated_values[k] = st.text_input(k, value=str(v))

            updated = st.form_submit_button("Update Submission", use_container_width=True)

        if updated:
            records_df = records_df[records_df["submission_id"] != selected_id].copy()
            records_df = pd.concat([records_df, pd.DataFrame([updated_values])], ignore_index=True)

            part_dir = os.path.join(DATA_DIR, f"proposal_type={st.session_state.ptype}")
            if st.session_state.date:
                part_dir = os.path.join(part_dir, f"dt={st.session_state.date}")
            os.makedirs(part_dir, exist_ok=True)
            file_path = os.path.join(part_dir, "data.parquet")
            records_df.to_parquet(file_path, engine=parquet_engine(), index=False)

            st.session_state.records_df = records_df
            st.success(f"Submission {selected_id} updated successfully! Old version removed.")

        # ---- Delete Button ----
        if st.button(f"Delete Submission {selected_id}", type="primary"):
            records_df = records_df[records_df["submission_id"] != selected_id].copy()

            part_dir = os.path.join(DATA_DIR, f"proposal_type={st.session_state.ptype}")
            if st.session_state.date:
                part_dir = os.path.join(part_dir, f"dt={st.session_state.date}")
            os.makedirs(part_dir, exist_ok=True)
            file_path = os.path.join(part_dir, "data.parquet")
            records_df.to_parquet(file_path, engine=parquet_engine(), index=False)

            st.session_state.records_df = records_df
            st.success(f"Submission {selected_id} deleted successfully!")
        if "records_df" in st.session_state:
            records_df = st.session_state.records_df

        if records_df is None or records_df.empty:
            st.warning("No submissions available.")
            return  # stop here so it doesn't try to build the selectbox

        st.dataframe(records_df, use_container_width=True)

        submission_ids = records_df["submission_id"].tolist()
        selected_id = st.selectbox("Select a Submission", [""] + submission_ids)

# def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    # --- Auth ---
    if ADMIN_IP_ALLOWLIST:
        ip = client_ip()
        if ip not in ADMIN_IP_ALLOWLIST:
            st.error("Your IP is not allowlisted.")
            return

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

    # --- Filters ---
    st.markdown("---")
    ptype = st.selectbox("Proposal Type", ["new_collection", "analysis_results"], index=0)
    date = st.text_input("Date partition (YYYY-MM-DD) or leave blank for all", value="")

    # --- Load submissions into session_state ---
    if st.button("Load Submissions", use_container_width=True):
        records_df = load_records(ptype, date)
        if records_df is None or records_df.empty:
            st.warning("No submissions found for the selected filters.")
            # Explicitly clear session state on empty load to prevent stale data display
            st.session_state.pop("records_df", None)
            return # Exit after warning
        else:
            st.session_state.records_df = records_df
            st.session_state.ptype = ptype
            st.session_state.date = date
            st.success("Submissions loaded.")

    # --- Retrieve records_df from session state or initialize it ---
    # This ensures records_df is bound if it was previously loaded.
    records_df = st.session_state.get("records_df")

    # --- Show submissions if available ---
    if records_df is None or records_df.empty:
        st.warning("No submissions available. Use 'Load Submissions' to view data.")
        return  # Stop execution if no data is loaded

    # Data loaded and available, proceed with display and management
    st.dataframe(records_df, use_container_width=True)

    submission_ids = records_df["submission_id"].tolist()
    selected_id = st.selectbox("Select a Submission", [""] + submission_ids)

    if selected_id:
        record = records_df[records_df["submission_id"] == selected_id].iloc[0].to_dict()

        st.write("### Manage Submission")

        # ---- Edit Form (dynamic) ----
        updated_values = {}
        with st.form("edit_form", clear_on_submit=False):
            for k, v in record.items():
                if k in ["submission_id", "created_at", "client_ip", "user_agent"]:
                    st.text_input(k, value=str(v), disabled=True)
                    updated_values[k] = v
                    continue

                # The logic for setting updated_values[k] for editable fields remains the same
                if isinstance(v, (int, float)):
                    updated_values[k] = st.number_input(k, value=float(v))
                elif isinstance(v, str) and len(v) > 150:
                    updated_values[k] = st.text_area(k, value=v)
                else:
                    updated_values[k] = st.text_input(k, value=str(v))

            updated = st.form_submit_button("Update Submission", use_container_width=True)

        if updated:
            # Update logic remains the same (delete old row, append new row)
            records_df = records_df[records_df["submission_id"] != selected_id].copy()
            records_df = pd.concat([records_df, pd.DataFrame([updated_values])], ignore_index=True)

            # Persist and update session state
            part_dir = os.path.join(DATA_DIR, f"proposal_type={st.session_state.ptype}")
            if st.session_state.date:
                part_dir = os.path.join(part_dir, f"dt={st.session_state.date}")
            os.makedirs(part_dir, exist_ok=True)
            file_path = os.path.join(part_dir, "data.parquet")
            records_df.to_parquet(file_path, engine=parquet_engine(), index=False)

            st.session_state.records_df = records_df
            st.success(f"Submission {selected_id} updated successfully! Old version removed.")
            st.rerun() # Use st.rerun() to refresh the display immediately

        # ---- Delete Button ----
        # This is where the error you specifically mentioned (deleting all records)
        # would occur if not handled by the check above.
        if st.button(f"Delete Submission {selected_id}", type="primary"):
            records_df = records_df[records_df["submission_id"] != selected_id].copy()

            # Persist and update session state
            part_dir = os.path.join(DATA_DIR, f"proposal_type={st.session_state.ptype}")
            if st.session_state.date:
                part_dir = os.path.join(part_dir, f"dt={st.session_state.date}")
            os.makedirs(part_dir, exist_ok=True)
            file_path = os.path.join(part_dir, "data.parquet")
            
            # Write back the new (potentially empty) DataFrame
            if records_df.empty:
                # If the DataFrame is empty, you might want to delete the file/partition
                # For simplicity here, we'll write an empty Parquet file, which is valid.
                records_df.to_parquet(file_path, engine=parquet_engine(), index=False)
                # If the last record was deleted, we explicitly clear the session state.
                st.session_state.pop("records_df", None)
            else:
                records_df.to_parquet(file_path, engine=parquet_engine(), index=False)
                st.session_state.records_df = records_df

            st.success(f"Submission {selected_id} deleted successfully!")
            st.rerun() # Use st.rerun() to refresh the display immediately

# def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    # --- Auth ---
    if ADMIN_IP_ALLOWLIST:
        ip = client_ip()
        if ip not in ADMIN_IP_ALLOWLIST:
            st.error("Your IP is not allowlisted.")
            return

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

    # --- Filters ---
    st.markdown("---")
    # Use session state defaults if they exist, otherwise use the default values
    default_ptype = st.session_state.get("ptype", "new_collection")
    default_date = st.session_state.get("date", "")
    
    ptype = st.selectbox("Proposal Type", ["new_collection", "analysis_results"], index=["new_collection", "analysis_results"].index(default_ptype))
    date = st.text_input("Date partition (YYYY-MM-DD) or leave blank for all", value=default_date)

    # --- Load submissions into session_state ---
    if st.button("Load Submissions", use_container_width=True):
        records_df = load_records(ptype, date)
        if records_df is None or records_df.empty:
            st.warning("No submissions found for the selected filters.")
            st.session_state.pop("records_df", None) # Clear stale data
        else:
            st.session_state.records_df = records_df
            st.session_state.ptype = ptype
            st.session_state.date = date
            st.success("Submissions loaded.")
            
        st.rerun() # Rerun to refresh the UI elements below

    # --- Retrieve records_df from session state or initialize it ---
    records_df = st.session_state.get("records_df")

    if records_df is None or records_df.empty:
        st.warning("No submissions available. Use 'Load Submissions' to view data.")
        # Do not proceed with display or management if no data is loaded
        return

    # Data loaded and available, proceed with display and management
    st.dataframe(records_df, use_container_width=True)

    submission_ids = records_df["submission_id"].tolist()
    selected_id = st.selectbox("Select a Submission", [""] + submission_ids)

    if selected_id:
        record = records_df[records_df["submission_id"] == selected_id].iloc[0].to_dict()

        st.write("### Manage Submission")

        # ---- Edit Form (dynamic) ----
        updated_values = {}
        with st.form("edit_form", clear_on_submit=False):
            # Create editable fields for all keys in the selected record
            for k, v in record.items():
                # Fields that shouldn't be edited via this general form
                if k in ["submission_id", "created_at", "client_ip", "user_agent", "proposal_type"]:
                    st.text_input(k, value=str(v), disabled=True)
                    updated_values[k] = v
                    continue

                # Dynamic field creation based on value type/length
                if isinstance(v, (int, float)):
                    updated_values[k] = st.number_input(k, value=float(v), key=f"edit_{k}")
                elif isinstance(v, str) and len(v) > 150:
                    updated_values[k] = st.text_area(k, value=v, key=f"edit_{k}")
                else:
                    updated_values[k] = st.text_input(k, value=str(v), key=f"edit_{k}")

            updated = st.form_submit_button("Update Submission", use_container_width=True)

        if updated:
            # 1. DELETE OLD RECORD: Filter out the old record
            records_df = records_df[records_df["submission_id"] != selected_id].copy()
            
            # 2. ADD NEW/UPDATED RECORD: Append the updated values as a new row
            records_df = pd.concat([records_df, pd.DataFrame([updated_values])], ignore_index=True)

            # 3. PERSIST (OVERWRITE PARTITION): Save back to Parquet
            part_dir = os.path.join(DATA_DIR, f"proposal_type={st.session_state.ptype}")
            if st.session_state.date:
                part_dir = os.path.join(part_dir, f"dt={st.session_state.date}")
            os.makedirs(part_dir, exist_ok=True)
            file_path = os.path.join(part_dir, "data.parquet")
            
            # Use the correct engine and index=False
            records_df.to_parquet(file_path, engine=parquet_engine(), index=False)

            # 4. UPDATE SESSION STATE: Store the new DataFrame
            st.session_state.records_df = records_df
            st.success(f"Submission {selected_id} updated successfully! Old version removed.")
            st.rerun() # Rerun to refresh the display

        # ---- Delete Button ----
        if st.button(f"Delete Submission {selected_id}", type="primary"):
            # 1. DELETE RECORD: Filter out the selected record
            records_df = records_df[records_df["submission_id"] != selected_id].copy()

            # 2. PERSIST (OVERWRITE PARTITION)
            part_dir = os.path.join(DATA_DIR, f"proposal_type={st.session_state.ptype}")
            if st.session_state.date:
                part_dir = os.path.join(part_dir, f"dt={st.session_state.date}")
            os.makedirs(part_dir, exist_ok=True)
            file_path = os.path.join(part_dir, "data.parquet")

            # Save the new (potentially empty) DataFrame
            if records_df.empty:
                # If the last record was deleted, we still write an empty file or remove it.
                # Writing an empty Parquet file is safer for data integrity/loading.
                records_df.to_parquet(file_path, engine=parquet_engine(), index=False)
                st.session_state.pop("records_df", None) # Clear from session state
            else:
                records_df.to_parquet(file_path, engine=parquet_engine(), index=False)
                st.session_state.records_df = records_df # Update session state

            st.success(f"Submission {selected_id} deleted successfully!")
            st.rerun() # Rerun to refresh the display

import streamlit as st
import pandas as pd
import os
import time
# Assuming the following functions and constants are defined elsewhere in intake.py:
# header, sidebar_nav, client_ip, sign_token, verify_token, parquet_engine, load_records
# ADMIN_IP_ALLOWLIST, ADMIN_PIN, DATA_DIR

def admin_page():
    header()
    st.subheader("Admin / Reviewer Portal")

    # --- Auth ---
    if ADMIN_IP_ALLOWLIST:
        ip = client_ip()
        if ip not in ADMIN_IP_ALLOWLIST:
            st.error("Your IP is not allowlisted.")
            return

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

    # --- Filters ---
    st.markdown("---")
    # Use session state defaults if they exist, otherwise use the default values
    default_ptype = st.session_state.get("ptype", "new_collection")
    default_date = st.session_state.get("date", "")
    
    ptype = st.selectbox("Proposal Type", ["new_collection", "analysis_results"], 
                         index=["new_collection", "analysis_results"].index(default_ptype))
    
    date = st.text_input("Date partition (YYYY-MM-DD) or leave blank for all", value=default_date)

    # --- Load submissions into session_state ---
    if st.button("Load Submissions", use_container_width=True):
        records_df = load_records(ptype, date)
        if records_df is None or records_df.empty:
            st.warning("No submissions found for the selected filters.")
            # Clear all relevant session state variables on empty load
            st.session_state.pop("records_df", None)
            st.session_state.pop("ptype", None)
            st.session_state.pop("date", None)
        else:
            st.session_state.records_df = records_df
            st.session_state.ptype = ptype
            st.session_state.date = date
            st.success("Submissions loaded.")
            
        st.rerun() # Rerun to refresh the UI elements below

    # --- Retrieve records_df from session state (Fixes UnboundLocalError) ---
    records_df = st.session_state.get("records_df")
    current_load_date = st.session_state.get("date")
    current_load_ptype = st.session_state.get("ptype")

    if records_df is None or records_df.empty:
        st.warning("No submissions available. Use 'Load Submissions' to view data.")
        return

    # --- Display Data ---
    st.dataframe(records_df, use_container_width=True)

    submission_ids = records_df["submission_id"].tolist()
    selected_id = st.selectbox("Select a Submission", [""] + submission_ids)

    
    # --- Modification Logic (Update/Delete) ---
    
    # This check is crucial: modifications are only safe when a single partition is loaded.
    modification_allowed = bool(current_load_date)
    if not modification_allowed:
        st.error("🚨 **Error:** You cannot edit or delete records when the 'Date partition' is blank (All Dates). Please select a specific date (YYYY-MM-DD) and reload to manage submissions for that day.")

    if selected_id:
        record = records_df[records_df["submission_id"] == selected_id].iloc[0].to_dict()

        st.write("### Manage Submission")
        
        # ---- Edit Form (dynamic) ----
        updated_values = {}
        with st.form("edit_form", clear_on_submit=False):
            for k, v in record.items():
                if k in ["submission_id", "created_at", "client_ip", "user_agent", "proposal_type"]:
                    st.text_input(k, value=str(v), disabled=True)
                    updated_values[k] = v
                    continue

                unique_key = f"edit_{selected_id}_{k}"
                
                if isinstance(v, (int, float)):
                    updated_values[k] = st.number_input(k, value=float(v) if v is not None else 0.0, key=unique_key)
                elif isinstance(v, str) and len(v) > 150:
                    updated_values[k] = st.text_area(k, value=v, key=unique_key)
                else:
                    updated_values[k] = st.text_input(k, value=str(v), key=unique_key)

            updated = st.form_submit_button("Update Submission", use_container_width=True, disabled=not modification_allowed)

        if updated and modification_allowed:
            # 1. DELETE OLD RECORD: Filter out the old record (Fixes update issue)
            records_df = records_df[records_df["submission_id"] != selected_id].copy()
            
            # 2. ADD NEW/UPDATED RECORD: Append the updated values as a new row
            records_df = pd.concat([records_df, pd.DataFrame([updated_values])], ignore_index=True)

            # 3. PERSIST (OVERWRITE PARTITION)
            part_dir = os.path.join(DATA_DIR, f"proposal_type={current_load_ptype}", f"dt={current_load_date}")
            os.makedirs(part_dir, exist_ok=True)
            file_path = os.path.join(part_dir, "data.parquet")
            
            records_df.to_parquet(file_path, engine=parquet_engine(), index=False)

            # 4. UPDATE SESSION STATE
            st.session_state.records_df = records_df
            st.success(f"Submission {selected_id} updated successfully! Old version removed.")
            st.rerun()

        # ---- Delete Button (Individual Record) ----
        if st.button(f"Delete Submission {selected_id}", type="primary", disabled=not modification_allowed):
            if not modification_allowed:
                return

            # 1. DELETE RECORD: Filter out the selected record
            records_df = records_df[records_df["submission_id"] != selected_id].copy()

            # 2. PERSIST (OVERWRITE PARTITION)
            part_dir = os.path.join(DATA_DIR, f"proposal_type={current_load_ptype}", f"dt={current_load_date}")
            os.makedirs(part_dir, exist_ok=True)
            file_path = os.path.join(part_dir, "data.parquet")

            if records_df.empty:
                # Overwrite with 0 rows
                records_df.to_parquet(file_path, engine=parquet_engine(), index=False)
                st.session_state.pop("records_df", None)
            else:
                records_df.to_parquet(file_path, engine=parquet_engine(), index=False)
                st.session_state.records_df = records_df

            st.success(f"Submission {selected_id} deleted successfully!")
            st.rerun()

    
    # --- Bulk Delete Partition Button (NEW FEATURE) ---
    st.markdown("---")
    
    if st.button(f"⚠️ Delete ALL {len(records_df)} Submissions for {current_load_date or 'No Date Filter'}?", use_container_width=True, disabled=not modification_allowed):
        if not modification_allowed:
            # Should be disabled, but good to have a final check
            st.error("Cannot delete all if 'Date partition' is blank.")
            return

        # 1. Determine the file path to the partition
        part_dir = os.path.join(DATA_DIR, f"proposal_type={current_load_ptype}", f"dt={current_load_date}")
        file_path = os.path.join(part_dir, "data.parquet")

        # 2. Delete the physical file
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # 3. Clear session state to reflect the change
            st.session_state.pop("records_df", None)
            st.success(f"Successfully deleted ALL submissions and removed the partition file for **{current_load_date}**.")
            
            # 4. Rerun to refresh the page and show no data
            st.rerun()
        else:
            st.warning(f"No partition file found at {file_path}. Nothing was deleted.")
            
    st.markdown("---")

def load_records(proposal_type: str, date: str | None):
    base = os.path.join(DATA_DIR, f"proposal_type={proposal_type}")
    if not os.path.isdir(base):
        return None

    paths = []
    if date:
        p = os.path.join(base, f"dt={date}", "data.parquet")
        if os.path.exists(p):
            paths.append(p)
    else:
        # all partitions
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".parquet"):
                    paths.append(os.path.join(root, f))

    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p, engine=parquet_engine()))
        except Exception:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


# ----------------------------
# Main
# ----------------------------
nav = sidebar_nav()
if nav == "Submit":
    submit_page()
else:
    admin_page()


# ----------------------------
# requirements.txt (place this in a separate file)
# streamlit==1.37.1
# pandas==2.2.2
# pyarrow==16.1.0  # preferred parquet engine (or)
# fastparquet==2024.5.0  # fallback parquet engine
# reportlab==4.2.2
# requests==2.32.3  # optional, for Slack
